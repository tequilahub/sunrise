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
class IrrepProviderBase(ABC):
    mol: QuantumChemistryBase

    @abstractmethod
    def get_irrep(self, state) -> str:
        pass


class LocalizedIrrepProvider(IrrepProviderBase):
    """
    An irrep provider that determines the irreducible representation of a state by directly
    comparing the effect of the symmetry operations on the state to the character table.
    The operations act on the orbitals of the molecule (see mol.mo_representation), which can be any orbitals
    the operations map onto themselves as a set, e.g. canonical, native or hybrid orbitals. The characters
    <psi|O|psi> are rounded to +/-1, so only states of one-dimensional irreps are labelled.
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
        character_vector = self.representation.character_vector(state.wavefunction)
        irrep = self.pg.character_table.vec_to_str(character_vector)
        return irrep if irrep is not None else None


class PySCFCanonicalIrrepProvider(IrrepProviderBase):
    """
    An irrep provider that uses PySCF to determine the irreducible representation of a state in the canonical molecular orbital basis.
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

        character_vector = numpy.ones(self.pg.order)
        for occ, irrep in zip(state.mo_occ, self.mol_irreps):
            chars = numpy.asarray(
                self.pg.character_table.dict[irrep], dtype=numpy.complex128
            )
            character_vector = character_vector * numpy.power(chars, round(occ))

        irrep = self.pg.character_table.vec_to_str(character_vector)
        return irrep if irrep is not None else None


SUPPORTED_IRREP_PROVIDERS = ["loc", "canon"]
INSTALLED_IRREP_PROVIDERS = {
    "loc": LocalizedIrrepProvider,
    "canon": PySCFCanonicalIrrepProvider,
}


def show_available_modules():
    print("Available Irrep Providers")
    for k in INSTALLED_IRREP_PROVIDERS:
        print(k)


def show_supported_modules():
    print(SUPPORTED_IRREP_PROVIDERS)


# TODO: change "provider" with a better keyword
def IrrepProvider(
    mol: QuantumChemistryBase, provider: str = "canon", *args, **kwargs
) -> IrrepProviderBase:
    r"""
    ADD SOMETHING
    """
    # any kwargs and circuit form should be managed inside each class
    return INSTALLED_IRREP_PROVIDERS[provider.lower()](*args, mol=mol, **kwargs)
