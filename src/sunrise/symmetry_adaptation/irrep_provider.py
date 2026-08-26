from abc import ABC, abstractmethod
from dataclasses import dataclass
import tequila
from tequila import Molecule
import numpy
from .point_group import PointGroup
from .point_group_representation import PointGroupRepresentation
from .qcircuit_representation_builder import QCircuitRepresentationBuilder


@dataclass
class IrrepProviderBase(ABC):
	mol: Molecule
	pg: PointGroup
	
	@abstractmethod
	def get_irrep(self, state) -> str:
		pass


class LocalizedIrrepProvider(IrrepProviderBase):
	"""
	An irrep provider that determines the irreducible representation of a state by directly
	comparing the effect of the symmetry operations on the state to the character table.
	This assumes a representation that is based on localized occupancies.
	"""
	
	def __init__(self, mol: Molecule, pg: PointGroup):
		super().__init__(mol, pg)
		self._representation: PointGroupRepresentation[tequila.QCircuit, tequila.QubitWaveFunction] = QCircuitRepresentationBuilder(mol, pg).build_qcircuit_representation()

		# Fetch the MO-specific permutations and phases
		builder = QCircuitRepresentationBuilder(mol, pg)
		self._spatial_perms, self._spatial_phases = builder._get_mo_perms_and_phases()

	@property
	def representation(self) -> PointGroupRepresentation[tequila.QCircuit, tequila.QubitWaveFunction]:
		return self._representation

	def get_irrep(self, state) -> str | None:
		from .symmetrization_procedure import SymmetryAdaptedLinearCombintationSymmetrization

		n_orbitals = self.mol.n_orbitals
		
		fragment_orbitals = getattr(state, 'fragment_orbitals', None)
		if fragment_orbitals is not None:
			fragment_set = set(fragment_orbitals)
			selected_ops = [
				label for label in self.pg.character_table.operation_symbols
				if all(self._spatial_perms[label][i] in fragment_set for i in fragment_orbitals)
			]
		else:
			selected_ops = self.pg.character_table.operation_symbols

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
		for irrep_name, irrep_chars in self.pg.character_table.dict.items():
			expected = [irrep_chars[self.pg.character_table.operation_symbols.index(label)] for label in selected_ops]
			if all(abs(c - e) < 0.5 for c, e in zip(character_vector, expected)):
				return irrep_name
				
		return None
	


class PySCFCanonicalIrrepProvider(IrrepProviderBase):
	"""
	An irrep provider that uses PySCF to determine the irreducible representation of a state in the canonical molecular orbital basis.
	"""

	def __init__(self, mol: Molecule, pg: PointGroup):
		import pyscf

		super().__init__(mol, pg)


		# TODO find a better solution for this
		# Rebuild the molecule with symmetry enabled BEFORE creating RHF
		pyscf_mol = pyscf.gto.M(
			atom=mol.pyscf_molecule.atom,
			basis=mol.pyscf_molecule.basis,
			symmetry=True,
			symmetry_subgroup=self.pg.schoenflies_label,
			charge=mol.pyscf_molecule.charge,
			spin=mol.pyscf_molecule.spin,
			unit=mol.pyscf_molecule.unit
		)
		
		# Now create RHF with the properly configured molecule
		mf = pyscf.scf.RHF(pyscf_mol)
		mf.verbose = 0
		mf.kernel()
		
		self._mol_irreps = pyscf.symm.label_orb_symm(
			pyscf_mol, 
			pyscf_mol.irrep_name, 
			pyscf_mol.symm_orb, 
			mf.mo_coeff
		).tolist()[-len(mol.orbitals):]

	
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
					f"sectors. Use IrrepProvider(mol, pg, 'loc') instead."
				)

		character_vector = numpy.ones(self.pg.order)
		for occ, irrep in zip(state.mo_occ, self.mol_irreps):
			chars = numpy.asarray(self.pg.character_table.dict[irrep], dtype=numpy.complex128)
			character_vector = character_vector * numpy.power(chars, int(round(occ)))
		irrep = self.pg.character_table.vec_to_str(character_vector)
		return irrep if irrep is not None else None


class MEAOIrrepProvider(LocalizedIrrepProvider):
	"""
	An irrep provider for MEAOs.
	Inherits from LocalizedIrrepProvider but uses MO-basis permutations and phases.
	"""
	def __init__(self, mol: Molecule, pg: PointGroup):
		# Skip parent __init__'s AO permutation logic
		IrrepProviderBase.__init__(self, mol, pg)
		self._representation = QCircuitRepresentationBuilder(mol, pg).build_qcircuit_representation()

		builder = QCircuitRepresentationBuilder(mol, pg)
		self._spatial_perms, self._spatial_phases = builder._get_mo_perms_and_phases()


SUPPORTED_IRREP_PROVIDERS = ["loc", "canon", "meao"]
INSTALLED_IRREP_PROVIDERS = {"loc": LocalizedIrrepProvider, "canon": PySCFCanonicalIrrepProvider, "meao": MEAOIrrepProvider}

def show_available_modules():
    print("Available Irrep Providers")
    for k in INSTALLED_IRREP_PROVIDERS.keys():
        print(k)

def show_supported_modules():
    print(SUPPORTED_IRREP_PROVIDERS)

# TODO: change "provider" with a better keyword
def IrrepProvider(mol: Molecule, pg: PointGroup, provider:str="canon", *args, **kwargs) -> IrrepProviderBase:
    r'''
    ADD SOMETHING
    '''
    #any kwargs and circuit form should be managed inside each class
    return INSTALLED_IRREP_PROVIDERS[provider.lower()](mol=mol, pg=pg, *args,**kwargs) 