from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy

from .point_group_representation import PointGroupRepresentation

if TYPE_CHECKING:
    # only for type hints, importing it at runtime would be circular
    from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase


@dataclass
class IrrepAssignerBase(ABC):
    mol: QuantumChemistryBase

    @abstractmethod
    def get_irrep(self, state) -> str:
        pass


class ProjectionIrrepAssigner(IrrepAssignerBase):
    """
    An irrep assigner that determines the irreducible representation of a state by directly
    comparing the effect of the symmetry operations on the state to the character table.
    The operations act on the orbitals of the molecule (see mol.mo_representation), which can be any orbitals
    the operations map onto themselves as a set, e.g. canonical, native or hybrid orbitals. The state is
    labelled with the irrep whose projector keeps it whole, otherwise (a mix of irreps) the irrep is None.
    """

    def __init__(self, mol: QuantumChemistryBase):
        super().__init__(mol)
        self.pg = mol.point_group
        # the operations as orbital transformations acting on the bitstrings, see mol.mo_representation
        self._representation = mol.mo_representation()

    @property
    def representation(self) -> PointGroupRepresentation:
        return self._representation

    def get_irrep(self, state) -> str | None:
        expectation_values = self.representation.character_vector(state.wavefunction)
        table = self.pg.character_table
        for irrep in table.irreps:
            characters = numpy.asarray(table.dict[irrep], dtype=numpy.complex128)
            # <psi|P_irrep|psi> with the projector P_irrep = dim / order * sum_O chi_irrep(O)^* O
            weight = (
                characters[0]
                * numpy.vdot(characters, expectation_values)
                / self.pg.order
            )
            if numpy.isclose(weight.real, 1.0, atol=1e-4):
                return irrep
        return None


class PySCFIrrepAssigner(IrrepAssignerBase):
    """
    An irrep assigner that uses PySCF to determine the irreducible representation of a state in the canonical molecular orbital basis.
    PySCF only labels orbitals in Abelian groups, so the states are labelled in mol.abelian_point_group,
    e.g. in D2h for a molecule with D4h symmetry.
    """

    def __init__(self, mol: QuantumChemistryBase):
        super().__init__(mol)
        self.pg = mol.abelian_point_group
        self._mol_irreps = mol.compute_orbital_irreps()

    @property
    def mol_irreps(self) -> list[str]:
        return self._mol_irreps

    def get_irrep(self, state) -> str | None:
        # Canonical irrep assignment only works for states with integer occupations
        # (single determinants or superpositions within the same occupation sector).
        # Fractional occupations indicate a superposition across sectors, which
        # requires the ProjectionIrrepAssigner's character-vector approach instead.
        for i, occ in enumerate(state.mo_occ):
            if abs(occ - round(occ)) > 1e-4:
                raise ValueError(
                    f"PySCFIrrepAssigner cannot determine the irrep of a state "
                    f"with fractional occupation {occ:.4f} on orbital {i}. "
                    f"This typically means the state is a superposition across occupation "
                    f"sectors. Use IrrepAssigner(mol, 'projection') instead."
                )

        character_vector = numpy.ones(self.pg.order)
        for occ, irrep in zip(state.mo_occ, self.mol_irreps):
            chars = numpy.asarray(
                self.pg.character_table.dict[irrep], dtype=numpy.complex128
            )
            character_vector = character_vector * numpy.power(chars, round(occ))

        irrep = self.pg.character_table.vec_to_str(character_vector)
        return irrep if irrep is not None else None


SUPPORTED_IRREP_ASSIGNERS = ["projection", "pyscf"]
INSTALLED_IRREP_ASSIGNERS = {
    "projection": ProjectionIrrepAssigner,
    "pyscf": PySCFIrrepAssigner,
}


def show_available_modules():
    print("Available Irrep Assigners")
    for k in INSTALLED_IRREP_ASSIGNERS:
        print(k)


def show_supported_modules():
    print(SUPPORTED_IRREP_ASSIGNERS)


def IrrepAssigner(
    mol: QuantumChemistryBase, method: str = "pyscf", *args, **kwargs
) -> IrrepAssignerBase:
    r"""
    Returns the irrep assigner that assigns irreducible representations to the FockSpaceStates of a molecule.

    Parameters
    ----------
    mol
        The molecule, its point group is mol.point_group.
    method
        "pyscf" (PySCFIrrepAssigner): from the irreps PySCF assigns to the canonical orbitals, in
        mol.abelian_point_group, for states with integer occupations.
        "projection" (ProjectionIrrepAssigner): by applying the symmetry operations to the state, in mol.point_group, for
        any orbitals the operations map onto themselves (e.g. native or hybrid orbitals).
    """
    # any kwargs should be managed inside each class
    return INSTALLED_IRREP_ASSIGNERS[method.lower()](*args, mol=mol, **kwargs)
