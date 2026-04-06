from dataclasses import dataclass
import tequila
from tequila import Molecule
import numpy
from pandas import DataFrame
from .irrep_provider import IrrepProvider


@dataclass
class FockSpaceState:
	"""A class representing a state in Fock space, i.e. a state of occupation numbers for each orbital."""

	mol: Molecule
	wavefunction: tequila.QubitWaveFunction
	irrep_provider: IrrepProvider


	def __init__(self, mol: Molecule, wavefunction: str | tequila.QubitWaveFunction, provider: IrrepProvider):
		self.mol = mol

		if isinstance(wavefunction, str):
			self.wavefunction = tequila.QubitWaveFunction.from_string(f"|{wavefunction}>")
		else:
			self.wavefunction = wavefunction
		self.irrep_provider = provider


	@property
	def is_superposition(self) -> bool:
		"""Returns True if the state is a superposition of multiple occupation number states, False if it is a single occupation number state."""
		return numpy.count_nonzero(numpy.abs(self.wavefunction.to_array()) > 1e-9) != 1
	

	@property
	def bitstring(self) -> str:
		return str(self.wavefunction)[9:-2] if not self.is_superposition else None


	@property
	def mo_occ(self) -> list[int]:
		"""Returns the occupation numbers of the molecular orbitals corresponding to the state."""

		# TODO too restrictive
		if self.is_superposition:
			return None
		
		n_orbitals = len(self.bitstring) // 2
		mo_occ = [0.0] * n_orbitals

		for i in range(n_orbitals):
			# Each MO has two qubits: qubit 2*i (alpha) and qubit 2*i+1 (beta)
			mo_occ[i] = int(self.bitstring[2*i]) + int(self.bitstring[2*i + 1])

		return mo_occ

	
	@property
	def m_s(self) -> numpy.float64:
		"""Returns the total spin projection (m_s or ⟨S_z⟩) of the state."""

		# import openfermion as of
		
		return self.wavefunction.inner(self.mol.make_sz_op()(self.wavefunction)).real
		#return  self.wavefunction.to_array().T @ of.linalg.get_sparse_operator(self.mol.make_sz_op().to_openfermion()) @ self.wavefunction.to_array()
	
	@property
	def S2(self) -> numpy.float64:
		"""Returns the expectation value of the total spin squared operator (⟨S^2⟩) for the state."""
		#import openfermion as of
		return self.wavefunction.inner(self.mol.make_s2_op()(self.wavefunction)).real
		#return self.wavefunction.to_array().T @ self.mol.make_s2_op().to_matrix() @ self.wavefunction.to_array()
		#return  self.wavefunction.to_array().T @ of.linalg.get_sparse_operator(self.mol.make_s2_op().to_openfermion()) @ self.wavefunction.to_array()

	@property
	def spin_multiplicity(self) -> int:
		"""Returns the spin multiplicity of the state, defined by 2S + 1 as string."""
		
		rounded_eigval = round(self.S2 * 4) / 4
		spin_type = {0: "singlet", 0.75: "doublet", 2: "triplet", 3.75: "quartet", 6: "quintet", 8.75: "sextet", 12: "septet", 15.75: "octet", 20: "nonet"}.get(rounded_eigval, "mixed")
		return spin_type

	@classmethod
	def dataframe(cls, states: list['FockSpaceState']) -> DataFrame:
		"""Returns a pandas DataFrame representation of a list of FockSpaceStates."""

		return DataFrame(
			data=[ [state.bitstring, state.wavefunction, state.mo_occ, state.m_s, state.S2, state.spin_multiplicity, state.irrep_provider.get_irrep(state), state.irrep_provider] for state in states ],
			columns=["bitstring", "wavefunction", "mo_occ", "m_s", "S2", "spin multiplicity", "irrep", "irrep_provider"]
		)
	
	@classmethod
	def all_states(cls, mol: Molecule, provider: IrrepProvider) -> list['FockSpaceState']:
		"""Generates all possible Fock space states for a given molecule."""
		import pyscf

		n_qubits = mol.pyscf_molecule.nelectron * 2
		states = []
		for i in range(2**n_qubits):
			bitstring = format(i, f'0{n_qubits}b')
			states.append(cls(mol, bitstring, provider))
		return states
	
	@classmethod
	def non_ionic_states(cls, mol: Molecule, provider: IrrepProvider) -> list['FockSpaceState']:
		"""Generates all non-ionic Fock space states for a given molecule, i.e. states where the sum of the electron occupations is equal to the ``mol.n_electrons``."""
		import pyscf
		
		n_qubits = mol.pyscf_molecule.nelectron * 2
		states = []
		for i in range(2**n_qubits):
			bitstring = format(i, f'0{n_qubits}b')
			state = cls(mol, bitstring, provider)
			if sum(state.mo_occ) == mol.pyscf_molecule.nelectron:
				states.append(state)
		return states