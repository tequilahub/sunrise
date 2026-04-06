from abc import ABC, abstractmethod
from dataclasses import dataclass
import tequila
import pandas as pd
from tequila import Molecule
import numpy
from .irrep_provider import IrrepProvider
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

	def symmetrize(self, states: pd.DataFrame) -> pd.DataFrame:
		df: pd.DataFrame = states.copy(deep=True)

		# assume the IrrepProvider is the same for all states
		irrep_provider = df['irrep_provider'].iloc[0]
		if irrep_provider is None or not isinstance(irrep_provider, IrrepProvider):
			raise ValueError(f"IrrepProvider is an illegal argument: {irrep_provider}.")

		# Create an empty dataframe with the same columns as df
		diagonalised_states: list[FockSpaceState] = []

		# Convert mo_occ lists to tuples for grouping
		# This is sort of bodged but the most simple solution
		df['mo_occ_tuple'] = df['mo_occ'].apply(tuple)
		
		# Group by the tuple version and m_s/⟨S_z⟩
		grouped = df.groupby(['mo_occ_tuple', 'm_s'])

		# Iterate over each group
		for (mo_occ_val, ms_sz_val), sub_df in grouped:
			mini_matrix = numpy.zeros((len(sub_df), len(sub_df)), dtype=complex)

			for i, wfn1 in enumerate(sub_df['wavefunction']):
				for j, wfn2 in enumerate(sub_df['wavefunction']):
					val = wfn1.inner(self.mol.make_s2_op()(wfn2))
					mini_matrix[i, j] = val


			eigvals, eigvecs = numpy.linalg.eigh(mini_matrix)

			for k in range(len(eigvals)):
				#eigval = eigvals[k]
				eigvec = eigvecs[:, k]

				new_wfn = tequila.QubitWaveFunction(n_qubits=self.mol.n_orbitals*2)
				for coef, wfn in zip(eigvec, sub_df['wavefunction']):
					new_wfn += coef * wfn

				diagonalised_states.append(FockSpaceState(
					mol=self.mol,
					wavefunction=new_wfn,
					provider=irrep_provider
				))
				
		return FockSpaceState.dataframe(diagonalised_states)


@dataclass
class SymmetryAdaptedLinearCombintationSymmetrization(SymmetrizationProcedure):
	"""A symmetrization procedure that symmetrizes states by constructing symmetry-adapted linear combinations."""
	
	pg: PointGroup
	pgr: PointGroupRepresentation | None = None

	def __post_init__(self):
		if self.pgr is None:
			self.pgr = QCircuitRepresentationBuilder(self.mol, self.pg).build_qcircuit_representation()

	def symmetrize(self, states: pd.DataFrame) -> pd.DataFrame:
		df: pd.DataFrame = states.copy(deep=True)

		# assume the IrrepProvider is the same for all states
		irrep_provider = df['irrep_provider'].iloc[0]
		if irrep_provider is None or not isinstance(irrep_provider, IrrepProvider):
			raise ValueError(f"IrrepProvider is an illegal argument: {irrep_provider}.")

		# assumes the representation as quantum circuit
		SALC_list: list[tequila.QubitWaveFunction] = []

		for i, state in df.iterrows():
			# assumes a consistent ordering of the operations
			resulting_state_list: list[tequila.QubitWaveFunction] = []

			# apply all symmetry operations to the state
			for op_label, op in self.pgr.operations.items():
				resulting_state_list.append(self.pgr.apply(op, state["wavefunction"]))

			for j, irrep in enumerate(self.pg.character_table.dict.keys()):
				state_SALC = tequila.QubitWaveFunction(n_qubits=self.mol.n_orbitals*2)
				
				state_SALC_summands = []
				for k, character in enumerate(self.pg.character_table.dict[irrep]):
					state_SALC_summands.append(character * resulting_state_list[k])

				state_SALC = tequila.QubitWaveFunction(n_qubits=self.mol.n_orbitals * 2)
				for wfn in state_SALC_summands:
					state_SALC += wfn
				
				# when an empty QubitWavefunction is normalized, the result is not nescessarily an empty QubitWavefunction
				if not numpy.allclose(state_SALC.to_array(), tequila.QubitWaveFunction(n_qubits=self.mol.n_orbitals*2).to_array()):
					SALC_list.append(state_SALC.normalize())

		# only keep linearly independent states
		SALC_list_unique: list[tequila.QubitWaveFunction] = []
		for state in SALC_list:
			if not any(state.isclose(existing_state) for existing_state in SALC_list_unique):
				SALC_list_unique.append(state)

		assert len(SALC_list_unique) == df.shape[0], "Assertion failed: the number of unique resulting SALC states is not equal to the number of original states."

		return FockSpaceState.dataframe([ FockSpaceState(self.mol, SALC, irrep_provider) for SALC in SALC_list_unique ])

				