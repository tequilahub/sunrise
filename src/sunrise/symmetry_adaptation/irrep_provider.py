from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy
if TYPE_CHECKING:
	# only for type hints, importing it at runtime would be circular
	from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase


@dataclass
class IrrepProviderBase(ABC):
	mol: QuantumChemistryBase
	
	@abstractmethod
	def get_irrep(self, state) -> str:
		pass


class LocalizedIrrepProvider(IrrepProviderBase):
	"""
	An irrep provider that determines the irreducible representation of a state by directly
	comparing the effect of the symmetry operations on the state to the character table.
	This assumes a representation that is based on localized occupancies.
	"""
	
	def __init__(self, mol: QuantumChemistryBase):
		super().__init__(mol)

		# Fetch the MO-specific permutations and phases, keyed by the operations of the point group
		self._pg = mol.point_group
		self._spatial_perms, self._spatial_phases = mol.compute_mo_perms_and_phases()

	def get_irrep(self, state) -> str | None:
		from .symmetrization_procedure import SymmetryAdaptedLinearCombintationSymmetrization

		n_orbitals = self.mol.n_orbitals
		pg = self._pg
		
		fragment_orbitals = getattr(state, 'fragment_orbitals', None)
		if fragment_orbitals is not None:
			fragment_set = set(fragment_orbitals)
			selected_ops = [
				label for label in pg.character_table.operation_symbols
				if all(self._spatial_perms[label][i] in fragment_set for i in fragment_orbitals)
			]
		else:
			selected_ops = pg.character_table.operation_symbols

		chars = []
		for label in selected_ops:
			perm = self._spatial_perms[label]
			phases = self._spatial_phases[label]  
			
			transformed = SymmetryAdaptedLinearCombintationSymmetrization._apply_spatial_permutation_bitstring(
				state.wavefunction, perm, n_orbitals, phases
			)
			expectation = state.wavefunction.inner(transformed).real
			chars.append(numpy.round(expectation))
		
		character_vector = numpy.array(chars)
		for irrep_name, irrep_chars in pg.character_table.dict.items():
			expected = [irrep_chars[pg.character_table.operation_symbols.index(label)] for label in selected_ops]
			if all(abs(c - e) < 0.5 for c, e in zip(character_vector, expected)):
				return irrep_name
				
		return None
	


class PySCFCanonicalIrrepProvider(IrrepProviderBase):
	"""
	An irrep provider that uses PySCF to determine the irreducible representation of a state in the canonical molecular orbital basis.
	PySCF only labels orbitals in Abelian groups, so the states are labelled in mol.abelian_point_group,
	e.g. in D2h for a molecule with D4h symmetry.
	"""

	def __init__(self, mol: QuantumChemistryBase):
		super().__init__(mol)
		self._pg = mol.abelian_point_group
		self._mol_irreps = mol.compute_orbital_irreps()

	
	@property
	def mol_irreps(self) -> list[str]:
		return self._mol_irreps
	

	def get_irrep(self, state) -> str | None:
		if state.mo_occ is None:
			return None

		# Canonical irrep assignment only works for states with integer occupations
		# (single determinants or superpositions within the same occupation sector).
		# Fractional occupations indicate a superposition across sectors, which
		# requires the LocalizedIrrepProvider's character-vector approach instead.
		for i, occ in enumerate(state.mo_occ):
			if abs(occ - round(occ)) > 1e-4:
				raise ValueError(
					f"PySCFCanonicalIrrepProvider cannot determine the irrep of a state "
					f"with fractional occupation {occ:.4f} on orbital {i}. "
					f"This typically means the state is a superposition across occupation "
					f"sectors. Use IrrepProvider(mol, 'loc') instead."
				)

		pg = self._pg
		character_vector = numpy.ones(pg.order)
		for occ, irrep in zip(state.mo_occ, self.mol_irreps):
			chars = numpy.asarray(pg.character_table.dict[irrep], dtype=numpy.complex128)
			character_vector = character_vector * numpy.power(chars, int(round(occ)))
		irrep = pg.character_table.vec_to_str(character_vector)
		return irrep if irrep is not None else None


class MEAOIrrepProvider(LocalizedIrrepProvider):
	"""
	An irrep provider for MEAOs.
	Inherits from LocalizedIrrepProvider but uses MO-basis permutations and phases.
	"""
	def __init__(self, mol: QuantumChemistryBase):
		# Skip parent __init__'s AO permutation logic
		IrrepProviderBase.__init__(self, mol)

		self._pg = mol.point_group
		self._spatial_perms, self._spatial_phases = mol.compute_mo_perms_and_phases()


SUPPORTED_IRREP_PROVIDERS = ["loc", "canon", "meao"]
INSTALLED_IRREP_PROVIDERS = {"loc": LocalizedIrrepProvider, "canon": PySCFCanonicalIrrepProvider, "meao": MEAOIrrepProvider}

def show_available_modules():
    print("Available Irrep Providers")
    for k in INSTALLED_IRREP_PROVIDERS.keys():
        print(k)

def show_supported_modules():
    print(SUPPORTED_IRREP_PROVIDERS)

# TODO: change "provider" with a better keyword
def IrrepProvider(mol: QuantumChemistryBase, provider:str="canon", *args, **kwargs) -> IrrepProviderBase:
    r'''
    ADD SOMETHING
    '''
    #any kwargs and circuit form should be managed inside each class
    return INSTALLED_IRREP_PROVIDERS[provider.lower()](mol=mol, *args,**kwargs) 