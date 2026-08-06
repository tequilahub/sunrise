from abc import ABC, abstractmethod
from dataclasses import dataclass
from re import sub
import tequila
from tequila import Molecule
import pandas as pd
from itertools import groupby
import numpy
from .irrep_provider import IrrepProvider
from .point_group import PointGroup
from .point_group_representation import PointGroupRepresentation
from .fock_space_state import FockSpaceState
from .qcircuit_representation_builder import QCircuitRepresentationBuilder

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

	def symmetrize(self, states: list[FockSpaceState] | pd.DataFrame) -> list[FockSpaceState] | pd.DataFrame:
		is_dataframe = isinstance(states, pd.DataFrame)
		if is_dataframe:
			state_list: list[FockSpaceState] = list(states.itertuples(index=False, name="FockSpaceState")) # produces namedtuples with attribute access
		else:
			state_list = list(states)

		# assume the IrrepProvider is the same for all states
		irrep_provider = state_list[0].irrep_provider
		if irrep_provider is None or not isinstance(irrep_provider, IrrepProvider):
			raise ValueError(f"IrrepProvider is an illegal argument: {irrep_provider}.")

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

			# for i, state1 in enumerate(sub_states):
			# 	for j, state2 in enumerate(sub_states):
			# 		val = state1.wavefunction.inner(self.mol.make_s2_op()(state2.wavefunction))
			# 		mini_matrix[i, j] = val

			mini_matrix = numpy.round(mini_matrix, decimals=10)

			# Diagonalize each connected component separately to prevent
			# degenerate states from different spatial irreps from mixing
			components = _get_connected_components(mini_matrix, tol=1e-10)

			for component in components:
				sub_matrix = mini_matrix[numpy.ix_(component, component)]
				eigvals, eigvecs = numpy.linalg.eigh(sub_matrix)

				for k in range(len(eigvals)):
					eigvec = eigvecs[:, k]

					new_wfn = tequila.QubitWaveFunction(n_qubits=self.mol.n_orbitals*2)
					for coef, idx in zip(eigvec, component):
						new_wfn += coef * sub_states[idx].wavefunction

					diagonalised_states.append(FockSpaceState(
						mol=self.mol,
						wavefunction=new_wfn,
						provider=irrep_provider
					))
		if is_dataframe:
			return FockSpaceState.dataframe(diagonalised_states)
		return diagonalised_states

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

@dataclass
class SymmetryAdaptedLinearCombintationSymmetrization(SymmetrizationProcedure):
	"""A symmetrization procedure that symmetrizes states by constructing symmetry-adapted linear combinations."""
	
	pg: PointGroup
	pgr: PointGroupRepresentation | None = None

	def __post_init__(self):
		if self.pgr is None:
			self.pgr = QCircuitRepresentationBuilder(self.mol, self.pg).build_qcircuit_representation()

	def symmetrize(self, states: list[FockSpaceState] | pd.DataFrame) -> list[FockSpaceState] | pd.DataFrame:
		is_dataframe = isinstance(states, pd.DataFrame)
		if is_dataframe:
			state_list: list[FockSpaceState] = list(states.itertuples(index=False, name="FockSpaceState")) # produces namedtuples with attribute access
		else:
			state_list = list(states)

		# assume the IrrepProvider is the same for all states
		irrep_provider = state_list[0].irrep_provider
		if irrep_provider is None or not isinstance(irrep_provider, IrrepProvider):
			raise ValueError(f"IrrepProvider is an illegal argument: {irrep_provider}.")

		# assumes the representation as quantum circuit
		SALC_list: list[tequila.QubitWaveFunction] = []

		for state in state_list:
			# assumes a consistent ordering of the operations
			resulting_state_list: list[tequila.QubitWaveFunction] = []

			# apply all symmetry operations to the state
			for op_label, op in self.pgr.operations.items():
				resulting_state_list.append(self.pgr.apply(op, state.wavefunction))

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
			key = _salc_key(state)
			if key not in seen_keys:
				seen_keys.add(key)
				SALC_list_unique.append(state)

		#assert len(SALC_list_unique) == len(state_list), "Assertion failed: the number of unique resulting SALC states is not equal to the number of original states."

		result_objects = [FockSpaceState(self.mol, SALC, irrep_provider) for SALC in SALC_list_unique]
		if is_dataframe:
			return FockSpaceState.dataframe(result_objects)
		return result_objects

				