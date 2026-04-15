from dataclasses import dataclass
from collections.abc import Generator
from itertools import product
import gc
from typing import Any
import tequila
from tequila import Molecule
import numpy
from pandas import DataFrame
from sunrise.symmetry_adaptation.irrep_provider import IrrepProvider


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
		return len([x for x in self.wavefunction.items()]) != 1
	

	@property
	def bitstring(self) -> str:
		return str(self.wavefunction)[9:-2] if not self.is_superposition else None


	@property
	def mo_occ(self) -> list[int]:
		"""Returns the occupation numbers of the molecular orbitals corresponding to the state."""

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
		return numpy.round(self.wavefunction.inner(self.mol.make_sz_op()(self.wavefunction)).real, decimals=10)
		
	
	@property
	def S2(self) -> numpy.float64:
		"""Returns the expectation value of the total spin squared operator (⟨S^2⟩) for the state."""
		return numpy.round(self.wavefunction.inner(self.mol.make_s2_op()(self.wavefunction)).real, decimals=10)


	@property
	def spin_multiplicity(self) -> int:
		"""Returns the spin multiplicity of the state, defined by 2S + 1 as string."""
		
		rounded_eigval = round(self.S2 * 4) / 4
		spin_type = {0: "singlet", 0.75: "doublet", 2: "triplet", 3.75: "quartet", 6: "quintet", 8.75: "sextet", 12: "septet", 15.75: "octet", 20: "nonet"}.get(rounded_eigval, "mixed")
		return spin_type


	@property
	def irrep(self) -> str:
		"""Returns the irreducible representation of the state as a string."""
		return self.irrep_provider.get_irrep(self)


	@classmethod
	def dataframe(cls, states: list['FockSpaceState']) -> DataFrame:
		"""Returns a pandas DataFrame representation of a list of FockSpaceStates."""

		return DataFrame(
			data=[ [state.bitstring, state.wavefunction, state.mo_occ, state.m_s, state.S2, state.spin_multiplicity, state.irrep, state.irrep_provider] for state in states ],
			columns=["bitstring", "wavefunction", "mo_occ", "m_s", "S2", "spin multiplicity", "irrep", "irrep_provider"]
		)
	
	@classmethod
	def all_states(cls, mol: Molecule, provider: IrrepProvider) -> list['FockSpaceState']:
		"""Generates all possible Fock space states for a given molecule."""
		return cls.by_filter(mol, provider)
	
	@classmethod
	def non_ionic_states(cls, mol: Molecule, provider: IrrepProvider) -> list['FockSpaceState']:
		"""Generates all non-ionic Fock space states for a given molecule."""
		return cls.by_filter(mol, provider, non_ionic=True)
	
	@classmethod
	def by_filter(cls, mol: Molecule, provider: IrrepProvider, mo_occ: list[int] | None = None, m_s: int | None = None, S2: int | None = None, spin_multiplicity: str | None = None, irrep: str | None = None, non_ionic: bool = False, max_count: int | None = None, bitstring_generator: Generator[str, None, None] | None = None) -> list['FockSpaceState']:
		"""Efficiently generates Fock space states for a given molecule taking various filters into account."""
		import pyscf

		def mo_occ_bitstring_generator(values: list[int]) -> Generator[str, Any, None]:
			"""
			maxs a list of 0s, 1s, and 2s and yields encoded strings.
			- 0  -> "00"
			- 2  -> "11"
			- 1  -> either "10" or "01", yielding all combinations of these for each 1 in the input list.
			"""
			TWO_ONE_COMBOS = ["01", "10"]
			ones_count = values.count(1)
			one_options = [TWO_ONE_COMBOS] * ones_count

			for combo in product(*one_options):
				result = []
				one_iter = iter(combo)
				for v in values:
					if v == 0:
						result.append("00")
					elif v == 2:
						result.append("11")
					else:  # v == 1
						result.append(next(one_iter))
				yield "".join(result)

		n_qubits: int = 2 * mol.n_orbitals

		# If bitstring_generator is provided it maxs precedence over mo_occ,
		# otherwise the search is optimized for mo_occ. If neither is provided,
		# all states are considered.
		generator = None
		if bitstring_generator is None:
			if mo_occ is None:
				generator = range(2**n_qubits)
			else:
				generator = mo_occ_bitstring_generator(mo_occ)
		else:
			generator: Generator[str, None, None] = bitstring_generator
		
		states: list['FockSpaceState'] = []
		for i in generator:
			# only return the max of max_count states
			if max_count is not None:
				if len(states) >= max_count:
					return states
				
			# if generator delivers a range of integers, convert to bitstring, otherwise assume it's already a bitstring
			bitstring = i if isinstance(i, str) else format(i, f'0{n_qubits}b')
			state = cls(mol, bitstring, provider)
			if non_ionic:
				if sum(state.mo_occ) != mol.n_electrons:
					del state
					gc.collect()
					continue
			if mo_occ is not None:
				if state.mo_occ != mo_occ:
					del state
					gc.collect()
					continue
			if m_s is not None:
				if state.m_s != m_s:
					del state
					gc.collect()
					continue
			if S2 is not None:
				if state.S2 != S2:
					del state
					gc.collect()
					continue
			if spin_multiplicity is not None:
				if state.spin_multiplicity != spin_multiplicity:
					del state
					gc.collect()
					continue
			if irrep is not None:
				if state.irrep != irrep:
					del state
					gc.collect()
					continue
			states.append(state)
		return states