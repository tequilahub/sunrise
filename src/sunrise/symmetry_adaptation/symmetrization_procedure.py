from abc import ABC, abstractmethod
from dataclasses import dataclass
from re import sub
import tequila
from tequila import Molecule
import pandas as pd
from itertools import groupby, combinations
import numpy
from .irrep_provider import IrrepProviderBase
from .point_group import PointGroup
from .point_group_representation import PointGroupRepresentation
from .fock_space_state import FockSpaceState
from .qcircuit_representation_builder import QCircuitRepresentationBuilder

@dataclass
class SymmetrizationProcedure(ABC):

	mol: Molecule

	@abstractmethod
	def symmetrize(self, states: pd.DataFrame) -> pd.DataFrame:
		"""Given a state in Fock space, returns a new state that is a symmetry-adapted linear combination of the original state and its images under the symmetry operations of the point group."""
		pass


@dataclass
class SpinSymmetrizationProcedure(SymmetrizationProcedure):
	"""A symmetrization procedure that symmetrizes states with respect to spin."""

	@staticmethod
	def _get_connected_components(matrix, tol=1e-10):
		"""Finds the connected components of a block-diagonal matrix."""
		n = matrix.shape[0]
		visited = [False] * n
		components = []
		for i in range(n):
			if not visited[i]:
				component = []
				stack = [i]
				visited[i] = True
				while stack:
					node = stack.pop()
					component.append(node)
					for neighbor in range(n):
						if not visited[neighbor] and abs(matrix[node, neighbor]) > tol:
							visited[neighbor] = True
							stack.append(neighbor)
				components.append(component)
		return components

	def symmetrize(self, states: list[FockSpaceState] | pd.DataFrame) -> list[FockSpaceState] | pd.DataFrame:
		is_dataframe = isinstance(states, pd.DataFrame)
		if is_dataframe:
			state_list: list[FockSpaceState] = list(states.itertuples(index=False, name="FockSpaceState")) # produces namedtuples with attribute access
		else:
			state_list = list(states)

		# assume the IrrepProviderBase is the same for all states; it is not used in the S2
		# diagonalization below, only propagated to the output states.
		irrep_provider = state_list[0].irrep_provider

		# Create an empty dataframe with the same columns as df
		diagonalised_states: list[FockSpaceState] = []

		# Group by and m_s/⟨S_z⟩
		# an additional grouping by mo_occ could be added however mo_occ does not always exist
		state_list_sorted = sorted(state_list, key=lambda s: s.m_s)
		grouped = groupby(state_list_sorted, key=lambda s: s.m_s)

		# Iterate over each group
		s2_op = self.mol.make_s2_op()
		for ms_val, group in grouped:
			sub_states: list[FockSpaceState] = list(group)
			mini_matrix = numpy.zeros((len(sub_states), len(sub_states)), dtype=complex)

			s2_applied = [s2_op(state.wavefunction) for state in sub_states]  # O(n) applications

			for i, state1 in enumerate(sub_states):
				for j in range(i, len(sub_states)):  # only upper triangle
					val = state1.wavefunction.inner(s2_applied[j])
					mini_matrix[i, j] = val
					mini_matrix[j, i] = numpy.conj(val)  # Hermitian

			mini_matrix = numpy.round(mini_matrix, decimals=10)

			# Diagonalize each connected component separately to prevent
			# degenerate states from different spatial irreps from mixing
			# It looks at where the non-zero matrix elements are, and since
			# the off-diagonal blocks are zero, it naturally separates states from different irreps
			components = self._get_connected_components(mini_matrix, tol=1e-10)

			for component in components:
				sub_matrix = mini_matrix[numpy.ix_(component, component)]
				eigvals, eigvecs = numpy.linalg.eigh(sub_matrix)

				for k in range(len(eigvals)):
					eigvec = eigvecs[:, k]
					s2_value = float(numpy.round(eigvals[k], decimals=10))

					new_wfn = tequila.QubitWaveFunction(n_qubits=self.mol.n_orbitals*2)
					for coef, idx in zip(eigvec, component):
						new_wfn += coef * sub_states[idx].wavefunction

					diagonalised_states.append(FockSpaceState(
						mol=self.mol,
						wavefunction=new_wfn,
						provider=irrep_provider,
						S2=s2_value
					))
		if is_dataframe:
			return FockSpaceState.dataframe(diagonalised_states)
		return diagonalised_states



@dataclass
class SymmetryAdaptedLinearCombintationSymmetrization(SymmetrizationProcedure):
	"""A symmetrization procedure that symmetrizes states by constructing symmetry-adapted linear combinations."""
	
	pg: PointGroup
	pgr: PointGroupRepresentation | None = None

	def __post_init__(self):
		if self.pgr is None:
			self.pgr = QCircuitRepresentationBuilder(self.mol, self.pg).build_qcircuit_representation()
		
		# Pre-compute spatial permutations for fast bitstring application
		ao_repr = QCircuitRepresentationBuilder(self.mol, self.pg).build_ao_permutation_representation()
		self._spatial_perms = {
			label: list(numpy.argmax(ao_repr.operations[label], axis=1))
			for label in self.pg.character_table.operation_symbols
		}

	@staticmethod
	def _salc_key(wfn: tequila.QubitWaveFunction, decimals: int = 8) -> tuple:
		"""Canonical, hashable representation of a wavefunction for O(1) dedup lookups,
		invariant to overall global phase (states differing only by an overall
		phase factor represent the same physical state and must hash identically)."""

		items = sorted(wfn.items())  # deterministic order by bitstring
		if not items:
			return ()

		# Find the phase of the first nonzero coefficient and divide it out,
		# so phase-equivalent states collapse to the same canonical key.
		first_coef = items[0][1]
		if abs(first_coef) < 1e-15:
			return ()
		phase = first_coef / abs(first_coef)

		return tuple(
			(bitstring, complex(round((coef / phase).real, decimals), round((coef / phase).imag, decimals)))
			for bitstring, coef in items
		)

	@staticmethod
	def _apply_spatial_permutation_bitstring(
		wfn: tequila.QubitWaveFunction, 
		perm: list[int], 
		n_orbitals: int
	) -> tequila.QubitWaveFunction:
		"""
		Apply a spatial orbital permutation directly to bitstrings.
		
		Parameters
		----------
		wfn : QubitWaveFunction
			Input wavefunction.
		perm : list[int]
			Spatial orbital permutation. perm[i] = j means orbital i maps to orbital j.
		n_orbitals : int
			Number of spatial orbitals.
		"""
		n_qubits = 2 * n_orbitals
		result_terms = {}
		
		for bitstring, coef in wfn.items():
			# Get the bitstring as a string of 0s and 1s
			bs_int = int(bitstring)
			bs_str = format(bs_int, f'0{n_qubits}b')
			
			# Apply spatial permutation and track occupied spin-orbitals
			new_bits = [0] * n_qubits
			occupied_original = []
			occupied_new = []
			
			for i in range(n_orbitals):
				alpha = int(bs_str[2*i])
				beta = int(bs_str[2*i+1])
				j = perm[i]
				
				if alpha:
					new_bits[2*j] = 1
					occupied_original.append(2*i)
					occupied_new.append(2*j)
				if beta:
					new_bits[2*j+1] = 1
					occupied_original.append(2*i+1)
					occupied_new.append(2*j+1)
			
			# Compute fermionic sign: parity of permutation needed to sort occupied_new
			# Count inversions in occupied_new
			inversions = 0
			for a in range(len(occupied_new)):
				for b in range(a+1, len(occupied_new)):
					if occupied_new[a] > occupied_new[b]:
						inversions += 1
			
			sign = (-1) ** inversions
			
			# Build new bitstring integer
			new_bs_int = 0
			for k, bit in enumerate(new_bits):
				if bit:
					new_bs_int |= (1 << (n_qubits - 1 - k))
			
			# Accumulate
			new_bs_str = format(new_bs_int, f'0{n_qubits}b')
			result_terms[new_bs_str] = result_terms.get(new_bs_str, 0) + sign * coef
		
		# Build result wavefunction
		result = tequila.QubitWaveFunction(n_qubits=n_qubits)
		for bs_str, c in result_terms.items():
			if abs(c) > 1e-15:
				result += c * tequila.QubitWaveFunction.from_string(f"|{bs_str}>")
		
		return result

	def symmetrize(self, states: list[FockSpaceState] | pd.DataFrame) -> list[FockSpaceState] | pd.DataFrame:
		is_dataframe = isinstance(states, pd.DataFrame)
		if is_dataframe:
			state_list: list[FockSpaceState] = list(states.itertuples(index=False, name="FockSpaceState")) # produces namedtuples with attribute access
		else:
			state_list = list(states)

		# assume the IrrepProviderBase is the same for all states; the SALC construction below
		# uses self.pg's character table, not irrep_provider, which is only propagated to
		# the output states.
		irrep_provider = state_list[0].irrep_provider

		# assumes the representation as quantum circuit
		SALC_list: list[tequila.QubitWaveFunction] = []
		
		# Get number of orbitals once outside the loop
		n_orbitals = self.mol.n_orbitals

		for state in state_list:
			# assumes a consistent ordering of the operations
			resulting_state_list: list[tequila.QubitWaveFunction] = []

			# apply all symmetry operations using fast bitstring permutations
			# (Iterating over operation_symbols also guarantees strict alignment with the character table)
			for op_label in self.pg.character_table.operation_symbols:
				perm = self._spatial_perms[op_label]
				resulting_state_list.append(self._apply_spatial_permutation_bitstring(state.wavefunction, perm, n_orbitals))

			for j, irrep in enumerate(self.pg.character_table.dict.keys()):
				state_SALC = tequila.QubitWaveFunction(n_qubits=self.mol.n_orbitals * 2)
				for k, character in enumerate(self.pg.character_table.dict[irrep]):
					if character != 0:
						state_SALC += character * resulting_state_list[k]
				
				norm_sq = state_SALC.inner(state_SALC).real
				if norm_sq > 1e-10:
					SALC_list.append(state_SALC.normalize())

		SALC_list_unique: list[tequila.QubitWaveFunction] = []
		seen_keys: set[tuple] = set()
		for state in SALC_list:
			key = self._salc_key(state)
			if key not in seen_keys:
				seen_keys.add(key)
				SALC_list_unique.append(state)

		#assert len(SALC_list_unique) == len(state_list), "Assertion failed: the number of unique resulting SALC states is not equal to the number of original states."

		result_objects = [FockSpaceState(self.mol, SALC, irrep_provider) for SALC in SALC_list_unique]
		if is_dataframe:
			return FockSpaceState.dataframe(result_objects)
		return result_objects

