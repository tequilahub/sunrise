from dataclasses import dataclass
from collections.abc import Generator
from itertools import product
import gc
from typing import Any
import tequila
from tequila import Molecule
import numpy
from pandas import DataFrame
from sunrise.symmetry_adaptation.irrep_provider import IrrepProviderBase
from functools import cached_property
from itertools import combinations

def _fixed_popcount_bitstrings(n_qubits: int, n_ones: int) -> Generator[str, None, None]:
    """Yields exactly the bitstrings of length n_qubits with n_ones bits set —
    i.e. the non-ionic Fock space, with no wasted candidates."""
    for ones_positions in combinations(range(n_qubits), n_ones):
        chars = ['0'] * n_qubits
        for p in ones_positions:
            chars[p] = '1'
        yield ''.join(chars)

def _fragment_bitstrings(mol: Molecule, fragment_orbitals: list[int], fragment_electrons: int | str | None = None) -> Generator[str, None, None]:
    """Yields full-space bitstrings for a fragment, padding infragment orbitals with '00' (vacuum)."""
    n_spatial = mol.n_orbitals
    n_fragment_qubits = 2 * len(fragment_orbitals)
    
    if fragment_electrons is None:
        fragment_electrons = len(fragment_orbitals)  # one electron per orbital
        electron_counts = [fragment_electrons]
    elif fragment_electrons == "all":
        # Special keyword to generate the full Fock space of the fragment (all possible electron numbers)
        electron_counts = range(n_fragment_qubits + 1)
    else:
        electron_counts = [fragment_electrons]
        
    if fragment_electrons != "all" and (fragment_electrons < 0 or fragment_electrons > n_fragment_qubits):
        raise ValueError(f"Invalid fragment space: {fragment_electrons} electrons cannot fit into {len(fragment_orbitals)} orbitals.")

    for n_e in electron_counts:
        for fragment_bs in _fixed_popcount_bitstrings(n_fragment_qubits, n_e):
            full = ['0'] * (2 * n_spatial)
            for idx, sp in enumerate(fragment_orbitals):
                full[2*sp]   = fragment_bs[2*idx]
                full[2*sp+1] = fragment_bs[2*idx+1]
            yield ''.join(full)

@dataclass
class FockSpaceState:
	"""A class representing a state in Fock space, i.e. a state of occupation numbers for each orbital."""

	mol: Molecule
	wavefunction: tequila.QubitWaveFunction
	irrep_provider: IrrepProviderBase
	_S2: float | None = None
	_irrep: str | None = None


	def __init__(self, mol: Molecule, wavefunction: str | tequila.QubitWaveFunction, provider: IrrepProviderBase, S2=None, irrep=None, fragment_orbitals: list[int] | None = None, m_s=None):
		self.mol = mol
		if isinstance(wavefunction, str):
			self.wavefunction = tequila.QubitWaveFunction.from_string(f"|{wavefunction}>")
		else:
			self.wavefunction = wavefunction
		self.irrep_provider = provider
		self._S2 = S2
		self._irrep = irrep
		self._fragment_orbitals = fragment_orbitals
		self._m_s = m_s


	@property
	def is_superposition(self) -> bool:
		"""Returns True if the state is a superposition of multiple occupation number states, False if it is a single occupation number state."""
		return len([x for x in self.wavefunction.items()]) != 1
	
	@cached_property
	def fragment_wavefunction(self) -> str:
		"""Readable state restricted to the fragment orbitals.
		Single determinant -> |bitstring>; superposition -> sum of terms."""
		if not self.is_superposition:
			return f"|{self.fragment_bitstring}>"
		n_qubits = 2 * self.mol.n_orbitals
		parts = []
		for bs, coef in self.wavefunction.items():
			bs_str = format(int(bs), f'0{n_qubits}b')
			if self._fragment_orbitals is not None:
				fragment_bs = "".join(bs_str[2*i : 2*i+2] for i in self._fragment_orbitals)
			else:
				fragment_bs = bs_str
			c = complex(coef)
			coef_str = f"{c.real:+.4f}" if abs(c.imag) < 1e-10 else f"({c.real:+.4f}{c.imag:+.4f}j)"
			parts.append(f"{coef_str}|{fragment_bs}>")
		return " ".join(parts)

	@cached_property
	def bitstring(self) -> str:
		return str(self.wavefunction)[9:-2] if not self.is_superposition else None

	@cached_property
	def fragment_bitstring(self) -> str | None:
		"""Returns the bitstring sliced to only the fragment fragment orbitals."""
		if self.bitstring is None: return None
		if self._fragment_orbitals is None: return self.bitstring
		return "".join(self.bitstring[2*i : 2*i+2] for i in self._fragment_orbitals)

	@cached_property
	def mo_occ(self) -> list[float]:
		"""Returns the expectation value of the occupation numbers of the molecular orbitals corresponding to the state."""

		n_orbitals = self.mol.n_orbitals
		mo_occ = [0.0] * n_orbitals

		if not self.is_superposition:
			for i in range(n_orbitals):
				# Each MO has two qubits: qubit 2*i (alpha) and qubit 2*i+1 (beta)
				mo_occ[i] = float(int(self.bitstring[2*i]) + int(self.bitstring[2*i + 1]))
			return mo_occ

		# For superpositions, compute the weighted average (expectation value)
		n_qubits = 2 * n_orbitals
		for bs, coef in self.wavefunction.items():
			# Format the bitstring to ensure it has the correct leading zeros
			bs_str = format(int(bs), f'0{n_qubits}b')
			weight = abs(coef)**2
			
			for i in range(n_orbitals):
				mo_occ[i] += weight * (int(bs_str[2*i]) + int(bs_str[2*i + 1]))

		return [round(x, 4) for x in mo_occ]

	@cached_property
	def fragment_mo_occ(self) -> list[float]:
		"""Returns the occupation sliced to only the fragment fragment orbitals."""
		if self._fragment_orbitals is None: return self.mo_occ
		return [self.mo_occ[i] for i in self._fragment_orbitals]


	@cached_property
	def m_s(self) -> numpy.float64:
		"""Returns the total spin projection (m_s or ⟨S_z⟩) of the state."""
		if self._m_s is not None:
			return numpy.round(self._m_s, decimals=10)
		return numpy.round(self.wavefunction.inner(self.mol.make_sz_op()(self.wavefunction)).real, decimals=10)
		
	
	@cached_property
	def S2(self):
		if self._S2 is not None:
			return self._S2
		return numpy.round(self.wavefunction.inner(
			self.mol.make_s2_op()(self.wavefunction)).real, decimals=10)


	@cached_property
	def spin_multiplicity(self) -> int:
		"""Returns the spin multiplicity of the state, defined by 2S + 1 as string."""
		
		rounded_eigval = round(self.S2 * 4) / 4
		spin_type = {0: "singlet", 0.75: "doublet", 2: "triplet", 3.75: "quartet", 6: "quintet", 8.75: "sextet", 12: "septet", 15.75: "octet", 20: "nonet"}.get(rounded_eigval, "mixed")
		return spin_type


	@cached_property
	def irrep(self):
		if self._irrep is not None:
			return self._irrep
		if not isinstance(self.irrep_provider, IrrepProviderBase):
			raise ValueError(
				f"Cannot determine the irrep of this FockSpaceState: irrep_provider is "
				f"{self.irrep_provider!r}, not an IrrepProviderBase. Construct the state with a real "
				f"IrrepProviderBase (e.g. LocalizedIrrepProvider or PySCFCanonicalIrrepProvider) if irrep "
				f"information is needed."
			)
		return self.irrep_provider.get_irrep(self)


	@classmethod
	def dataframe(cls, states: list['FockSpaceState']) -> DataFrame:
		"""Returns a pandas DataFrame representation of a list of FockSpaceStates."""

		return DataFrame(
			data=[ [state.bitstring, state.wavefunction, state.mo_occ, state.m_s, state.S2, state.spin_multiplicity, state.irrep, state.irrep_provider] for state in states ],
			columns=["bitstring", "wavefunction", "mo_occ", "m_s", "S2", "spin multiplicity", "irrep", "irrep_provider"]
		)
	
	@classmethod
	def all_states(cls, mol: Molecule, provider: IrrepProviderBase) -> list['FockSpaceState']:
		"""Generates all possible Fock space states for a given molecule."""
		return cls.by_filter(mol, provider)
	
	@classmethod
	def non_ionic_states(cls, mol: Molecule, provider: IrrepProviderBase, fragment_orbitals: list[int] | None = None, fragment_electrons: int | None = None) -> list['FockSpaceState']:
		"""Generates all non-ionic Fock space states for a given molecule."""
		if fragment_orbitals is None:
			n_qubits = 2 * mol.n_orbitals
			gen = _fixed_popcount_bitstrings(n_qubits, mol.n_electrons)
			return cls.by_filter(mol, provider, bitstring_generator=gen)
		return cls.by_filter(mol, provider, fragment_orbitals=fragment_orbitals, fragment_electrons=fragment_electrons)
	
	@classmethod
	def by_filter(cls, mol: Molecule, provider: IrrepProviderBase, mo_occ: list[int] | None = None, m_s: int | None = None, S2: int | None = None, spin_multiplicity: str | None = None, irrep: str | None = None, non_ionic: bool = False, max_count: int | None = None, bitstring_generator: Generator[str, None, None] | None = None, fragment_orbitals: list[int] | None = None, fragment_electrons: int | None = None) -> list['FockSpaceState']:
		"""Efficiently generates Fock space states for a given molecule taking various filters into account."""
		# import pyscf

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
		if bitstring_generator is not None:
			generator = bitstring_generator
		elif fragment_orbitals is not None:
			generator = _fragment_bitstrings(mol, fragment_orbitals, fragment_electrons)
		elif mo_occ is None:
			generator = range(2**n_qubits)
		else:
			if len(mo_occ) != mol.n_orbitals:
				raise ValueError(f"Length of mo_occ list must match the number of orbitals in the molecule ({mol.n_orbitals}), but got {len(mo_occ)}.")
			generator = mo_occ_bitstring_generator(mo_occ)
		
		states: list['FockSpaceState'] = []
		for i in generator:
			# only return the max of max_count states
			if max_count is not None:
				if len(states) >= max_count:
					return states
				
			# if generator delivers a range of integers, convert to bitstring, otherwise assume it's already a bitstring
			bitstring = i if isinstance(i, str) else format(i, f'0{n_qubits}b')
			state = cls(mol, bitstring, provider, fragment_orbitals=fragment_orbitals)
			if non_ionic:
				if sum(state.mo_occ) != mol.n_electrons:
					continue
			if mo_occ is not None:
				if state.mo_occ != mo_occ:
					continue
			if m_s is not None:
				if state.m_s != m_s:
					continue
			if S2 is not None:
				if state.S2 != S2:
					continue
			if spin_multiplicity is not None:
				if state.spin_multiplicity != spin_multiplicity:
					continue
			if irrep is not None:
				if state.irrep != irrep:
					continue
			states.append(state)
		return states