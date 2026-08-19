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
import math

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
		fragment_orbitals = getattr(state_list[0], '_fragment_orbitals', None)

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
						S2=s2_value,
						fragment_orbitals=fragment_orbitals
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
			state_list: list[FockSpaceState] = list(states.itertuples(index=False, name="FockSpaceState"))
		else:
			state_list = list(states)

		irrep_provider = state_list[0].irrep_provider
		fragment_orbitals = getattr(state_list[0], '_fragment_orbitals', None)

		SALC_list: list[tequila.QubitWaveFunction] = []
		n_orbitals = self.mol.n_orbitals

		# --- NEW: select which operations to apply -----------------------------
		# For an fragment-space fragment, only operations that map the fragment orbital
		# set onto itself are symmetries of the fragment. Any other operation would
		# move electrons out of the fragment space. With no fragment space, use all ops.
		if fragment_orbitals is not None:
			fragment_set = set(fragment_orbitals)
			selected_ops = [
				(k, op_label)
				for k, op_label in enumerate(self.pg.character_table.operation_symbols)
				if all(self._spatial_perms[op_label][i] in fragment_set for i in fragment_orbitals)
			]
		else:
			selected_ops = list(enumerate(self.pg.character_table.operation_symbols))
		# ------------------------------------------------------------------------

		for state in state_list:
			resulting_state_list: list[tequila.QubitWaveFunction] = []
			op_indices: list[int] = []  # original index into the character table

			# --- CHANGED: apply only the selected operations -------------------
			for k, op_label in selected_ops:
				perm = self._spatial_perms[op_label]
				resulting_state_list.append(self._apply_spatial_permutation_bitstring(state.wavefunction, perm, n_orbitals))
				op_indices.append(k)
			# ---------------------------------------------------------------------

			for j, irrep in enumerate(self.pg.character_table.dict.keys()):
				state_SALC = tequila.QubitWaveFunction(n_qubits=self.mol.n_orbitals * 2)
				# --- CHANGED: look up characters by the ORIGINAL op index --------
				for idx, k in enumerate(op_indices):
					character = self.pg.character_table.dict[irrep][k]
					if character != 0:
						state_SALC += character * resulting_state_list[idx]
				# ------------------------------------------------------------------

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

		result_objects = [FockSpaceState(self.mol, SALC, irrep_provider, fragment_orbitals=fragment_orbitals) for SALC in SALC_list_unique]
		if is_dataframe:
			return FockSpaceState.dataframe(result_objects)
		return result_objects

@dataclass
class SpinCGSymmetrizationProcedure(SymmetrizationProcedure):
	"""Builds spin-adapted CSFs by genealogical Clebsch-Gordan coupling,
	reproducing the O^{S,M}_N spin eigenfunctions of Marti-Dafcik et al.
	(Appendix A). Unlike S2 diagonalization, the degenerate S subspace is
	returned in the specific genealogical coupling basis, one CSF per path."""

	fragment_orbitals: list[int] | None = None

	@staticmethod
	def _cg(j12: int, m12: int, j3: int, m3: int, J2: int, M2: int) -> float:
		"""Clebsch-Gordan <j1 m1 j2 m2 | J M>, all arguments doubled (integers). Racah formula."""
		if m12 + m3 != M2:
			return 0.0
		if abs(j12 - j3) > J2 or j12 + j3 < J2 or abs(M2) > J2:
			return 0.0
		f = math.factorial
		d1, d2, d3 = (j12+j3-J2)//2, (j12-j3+J2)//2, (-j12+j3+J2)//2
		if d1 < 0 or d2 < 0 or d3 < 0:
			return 0.0
		pref_sq = (J2 + 1) * f(d1)*f(d2)*f(d3) / f((j12+j3+J2)//2 + 1)
		for x in (j12+m12, j12-m12, j3+m3, j3-m3, J2+M2, J2-M2):
			pref_sq *= f(x//2)
		pref = math.sqrt(pref_sq)

		z_min = max(0, -((J2-j3+m12)//2), -((J2-j12-m3)//2))
		z_max = min((j12+j3-J2)//2, (j12-m12)//2, (j3+m3)//2)
		s = 0.0
		for z in range(z_min, z_max + 1):
			den = (f(z) * f((j12+j3-J2)//2 - z) * f((j12-m12)//2 - z)
			       * f((j3+m3)//2 - z) * f((J2-j3+m12)//2 + z) * f((J2-j12-m3)//2 + z))
			s += ((-1)**z) / den
		return pref * s

	@classmethod
	def _coupling_paths(cls, n: int, S2: int) -> list[tuple[int, ...]]:
		"""All genealogical intermediate-spin paths (doubled values) to total S."""
		paths: list[tuple[int, ...]] = []
		def rec(step: int, j: int, path: tuple[int, ...]):
			if step == n:
				if j == S2:
					paths.append(path)
				return
			for j_next in range(abs(j - 1), j + 2, 2):
				rec(step + 1, j_next, path + (j_next,))
		rec(1, 1, ())
		return paths

	@staticmethod
	def _occupation_patterns(n_orbitals: int, n_electrons: int) -> list[list[int]]:
		"""All unique occupation patterns for n_electrons in n_orbitals (max 2 per orbital)."""
		patterns: list[list[int]] = []
		def recurse(idx: int, remaining: int, current: list[int]):
			if idx == n_orbitals:
				if remaining == 0:
					patterns.append(list(current))
				return
			for occ in range(min(remaining, 2) + 1):
				current.append(occ)
				recurse(idx + 1, remaining - occ, current)
				current.pop()
		recurse(0, n_electrons, [])
		return patterns

	def _build_spin_function(self, orbitals: list[int], S2: int, M2: int, path: tuple[int, ...], closed_shell_orbitals: list[int] | None = None) -> tequila.QubitWaveFunction:
		"""CSF with open-shell electrons in `orbitals` coupled along `path`.
		closed_shell_orbitals are doubly-occupied spectators contributing |11⟩."""
		n = len(orbitals)
		n_qubits = 2 * self.mol.n_orbitals
		terms: dict[int, float] = {}

		closed_shell_mask = 0
		if closed_shell_orbitals is not None:
			for orb in closed_shell_orbitals:
				closed_shell_mask |= (1 << (2*orb)) | (1 << (2*orb + 1))

		def recurse(idx: int, j_so_far: int, m_so_far: int, det: int, coef: float):
			if coef == 0:
				return
			if idx == n:
				if m_so_far == M2:
					full_det = det | closed_shell_mask
					terms[full_det] = terms.get(full_det, 0.0) + coef
				return
			orb = orbitals[idx]
			for s2 in (0, 1):
				m_e = 2*s2 - 1
				new_m = m_so_far + m_e
				bit_pos = 2*orb + (1 - s2)
				new_det = det | (1 << bit_pos)
				if idx == 0:
					recurse(1, 1, new_m, new_det, coef)
				else:
					j_next = path[idx - 1]
					c = self._cg(j_so_far, m_so_far, 1, m_e, j_next, new_m)
					if c != 0:
						recurse(idx + 1, j_next, new_m, new_det, coef * c)

		recurse(0, 0, 0, 0, 1.0)

		wfn = tequila.QubitWaveFunction(n_qubits=n_qubits)
		for det, coef in terms.items():
			wfn += coef * tequila.QubitWaveFunction.from_string(f"|{format(det, f'0{n_qubits}b')}>")
		return wfn.normalize()

	def symmetrize(self, states: list[FockSpaceState] | pd.DataFrame) -> list[FockSpaceState] | pd.DataFrame:
		"""Generates all spin-adapted CSFs across every occupation sector of the fragment."""
		is_dataframe = isinstance(states, pd.DataFrame)
		if is_dataframe:
			state_list: list[FockSpaceState] = list(states.itertuples(index=False, name="FockSpaceState"))
		else:
			state_list = list(states)

		fragment_orbitals = getattr(state_list[0], '_fragment_orbitals', None)
		if fragment_orbitals is None:
			fragment_orbitals = self.fragment_orbitals
		if fragment_orbitals is None:
			fragment_orbitals = list(range(self.mol.n_orbitals))

		orbitals = list(fragment_orbitals)
		n_orb = len(orbitals)
		irrep_provider = state_list[0].irrep_provider

		# Determine electron count from the first input state
		n_electrons = int(round(sum(state_list[0].mo_occ[i] for i in orbitals)))

		patterns = self._occupation_patterns(n_orb, n_electrons)

		all_states: list[FockSpaceState] = []
		for mo_occ in patterns:
			open_shell   = [orbitals[i] for i, occ in enumerate(mo_occ) if occ == 1]
			closed_shell = [orbitals[i] for i, occ in enumerate(mo_occ) if occ == 2]
			n_open = len(open_shell)

			if n_open == 0:
				# Pure closed-shell: single determinant, S=0, m_s=0
				det = 0
				for orb in closed_shell:
					det |= (1 << (2*orb)) | (1 << (2*orb + 1))
				wfn = tequila.QubitWaveFunction.from_string(f"|{format(det, f'0{2*self.mol.n_orbitals}b')}>")
				state = FockSpaceState(mol=self.mol, wavefunction=wfn, provider=irrep_provider,
				                       S2=0.0, m_s=0.0, fragment_orbitals=orbitals)
				state.coupling_path = ()
				all_states.append(state)
			else:
				for M2 in range(-n_open, n_open + 1, 2):
					m_s = M2 / 2.0
					for S2 in range(abs(M2), n_open + 1, 2):
						S = S2 / 2.0
						for path in self._coupling_paths(n_open, S2):
							wfn = self._build_spin_function(open_shell, S2, M2, path, closed_shell)
							state = FockSpaceState(mol=self.mol, wavefunction=wfn, provider=irrep_provider,
							                       S2=float(S * (S + 1)), m_s=m_s, fragment_orbitals=orbitals)
							state.coupling_path = tuple(p / 2 for p in path)
							all_states.append(state)

		if is_dataframe:
			return FockSpaceState.dataframe(all_states)
		return all_states

	def build_spin_adapted_states(self, S: int, m_s: int) -> list[FockSpaceState]:
		"""Returns one CSF per independent genealogical coupling path to (S, m_s)."""
		S2, M2 = 2 * S, 2 * m_s
		if S < 0:
			raise ValueError("S must be non-negative.")
		if abs(M2) > S2:
			raise ValueError(f"|m_s| = {abs(m_s)} cannot exceed S = {S}.")
		orbitals = list(self.fragment_orbitals) if self.fragment_orbitals is not None else list(range(self.mol.n_orbitals))
		n = len(orbitals)
		if n % 2 != S2 % 2:
			raise ValueError(f"Cannot couple {n} electrons to S = {S}.")

		paths = self._coupling_paths(n, S2)
		# print(f"[DEBUG] n_electrons={n}, S={S}, m_s={m_s}, S(S+1)={S*(S+1)}")
		# print(f"[DEBUG] fragment_orbitals={orbitals}")
		# print(f"[DEBUG] coupling paths found: {len(paths)}")
		# for p in paths:
		# 	print(f"[DEBUG]   path (doubled)={p}  ->  (halved)={tuple(x/2 for x in p)}")

		states: list[FockSpaceState] = []
		for path in paths:
			wfn = self._build_spin_function(orbitals, S2, M2, path)

			# Verify S² by direct application of the operator
			s2_op = self.mol.make_s2_op()
			s2_check = float(numpy.round(wfn.inner(s2_op(wfn)).real, 10))

			state = FockSpaceState(mol=self.mol, wavefunction=wfn, provider=None,
			                       S2=float(S * (S + 1)), fragment_orbitals=orbitals)
			state.coupling_path = tuple(p / 2 for p in path)
			states.append(state)

			# print(f"\n[DEBUG] path={state.coupling_path}")
			# print(f"[DEBUG]   S²(expected)={S*(S+1)}, S²(operator check)={s2_check}")
			# print(f"[DEBUG]   m_s check={float(numpy.round(wfn.inner(self.mol.make_sz_op()(wfn)).real, 10))}")
			# print(f"[DEBUG]   norm²={float(numpy.round(wfn.inner(wfn).real, 10))}")
			# print(f"[DEBUG]   terms:")
			for bs, coef in sorted(wfn.items()):
				bs_str = format(int(bs), f'0{2*self.mol.n_orbitals}b')
				frag_bs = "".join(bs_str[2*i:2*i+2] for i in orbitals)
				# print(f"[DEBUG]     {coef:+.6f} |{bs_str}>  (fragment: |{frag_bs}>)")

		return states

	def all_spin_adapted_states(self) -> list[FockSpaceState]:
		"""Generates the complete basis of spin-adapted CSFs for the fragment, spanning all valid S and m_s."""
		orbitals = list(self.fragment_orbitals) if self.fragment_orbitals is not None else list(range(self.mol.n_orbitals))
		n = len(orbitals)
		
		all_states = []
		
		# Total m_s ranges from -n/2 to n/2. In doubled units, M2 ranges from -n to n.
		# The step is 2 because M2 and N must have the same parity.
		for M2 in range(-n, n + 1, 2):
			m_s = M2 / 2.0
			# For a given M2, 2*S (S2) ranges from |M2| to n in steps of 2
			for S2 in range(abs(M2), n + 1, 2):
				S = S2 / 2.0
				for path in self._coupling_paths(n, S2):
					wfn = self._build_spin_function(orbitals, S2, M2, path)
					state = FockSpaceState(mol=self.mol, wavefunction=wfn, provider=None,
										S2=float(S * (S + 1)), m_s=m_s, fragment_orbitals=orbitals)
					state.coupling_path = tuple(p / 2 for p in path)
					all_states.append(state)
					
		return all_states