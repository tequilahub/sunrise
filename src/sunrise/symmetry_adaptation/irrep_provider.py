from abc import ABC, abstractmethod
from dataclasses import dataclass
import tequila
from tequila import Molecule
import numpy
from .point_group import PointGroup
from .point_group_representation import PointGroupRepresentation
from .fock_space_state import FockSpaceState
from .qcircuit_representation_builder import QCircuitRepresentationBuilder


@dataclass
class IrrepProvider(ABC):
	mol: Molecule
	pg: PointGroup
	
	@abstractmethod
	def get_irrep(self, state: FockSpaceState) -> str:
		pass


class LocalizedIrrepProvider(IrrepProvider):
	"""
	An irrep provider that determines the irreducible representation of a state by directly
	comparing the effect of the symmetry operations on the state to the character table.
	This assumes a representation that is based on localized occupancies.
	"""
	
	def __init__(self, mol: Molecule, pg: PointGroup):
		super().__init__(mol, pg)
		self._representation: PointGroupRepresentation[tequila.QCircuit, tequila.QubitWaveFunction] = QCircuitRepresentationBuilder(mol, pg).build_qcircuit_representation()


	@property
	def representation(self) -> PointGroupRepresentation[tequila.QCircuit, tequila.QubitWaveFunction]:
		return self._representation

	def get_irrep(self, state: 'FockSpaceState') -> str | None:
		character_vector = self.representation.character_vector(state.wavefunction)
		irrep = self.pg.character_table.vec_to_str(character_vector)
		return irrep if irrep is not None else None
	


class PySCFCanonicalIrrepProvider(IrrepProvider):
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
	

	def get_irrep(self, state: 'FockSpaceState') -> str | None:
		if state.mo_occ is None:
			return None

		character_vector = numpy.ones(self.pg.order)
		for occ, irrep in zip(state.mo_occ, self.mol_irreps):
			chars = numpy.asarray(self.pg.character_table.dict[irrep], dtype=numpy.complex128)
			character_vector = character_vector * numpy.power(chars, int(occ))
		
		irrep = self.pg.character_table.vec_to_str(character_vector)
		return irrep if irrep is not None else None