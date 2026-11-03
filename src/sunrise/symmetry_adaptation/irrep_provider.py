"""Irrep providers: objects that assign a point-group irreducible representation to a Fock-space state."""

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
    """Abstract base class of the irrep providers.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule whose point group defines the irreps.
    """

    mol: QuantumChemistryBase

    @abstractmethod
    def get_irrep(self, state) -> str:
        """Return the irrep label of a state.

        Parameters
        ----------
        state : FockSpaceState
            State to classify.

        Returns
        -------
        str or None
            Irrep name, None if it cannot be determined.
        """


class LocalizedIrrepProvider(IrrepProviderBase):
    r"""Irrep provider that applies the symmetry operations to the state and compares with the character table.

    The operations act on the orbitals of the molecule (see ``mol.mo_representation``), which can be
    any orbitals the operations map onto themselves as a set, e.g. canonical, native or hybrid
    orbitals. Only states of one-dimensional irreps are labelled: :math:`\langle\psi|O|\psi\rangle = \pm 1`
    for every operation :math:`O`, otherwise the irrep is None.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule whose ``point_group`` defines the irreps.
    """

    def __init__(self, mol: QuantumChemistryBase):
        super().__init__(mol)
        self.pg = mol.point_group
        # the operations as orbital transformations acting on the bitstrings, see mol.mo_representation
        self._representation = mol.mo_representation()

    @property
    def representation(self) -> PointGroupRepresentation:
        """PointGroupRepresentation : the symmetry operations as orbital transformations, see ``mol.mo_representation``."""
        return self._representation

    def get_irrep(self, state) -> str | None:
        """Return the irrep of ``state`` from its character vector.

        Parameters
        ----------
        state : FockSpaceState
            State to classify, possibly a superposition.

        Returns
        -------
        str or None
            Irrep name, None if the state is not an eigenstate of every operation (characters other
            than +1 or -1) or matches no irrep.
        """
        character_vector = self.representation.character_vector(state.wavefunction)
        # a state that is not an eigenstate of every operation belongs to no one-dimensional irrep
        if not numpy.allclose(numpy.abs(character_vector), 1.0, atol=1e-4):
            return None
        irrep = self.pg.character_table.vec_to_str(numpy.round(character_vector))
        return irrep if irrep is not None else None


class PySCFCanonicalIrrepProvider(IrrepProviderBase):
    """Irrep provider based on the canonical molecular orbitals labelled by PySCF.

    PySCF only labels orbitals in Abelian groups, so the states are labelled in
    ``mol.abelian_point_group``, e.g. in D2h for a molecule with D4h symmetry. The irrep of a
    state is the product of the irreps of the occupied orbitals.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule built with the PySCF backend.
    """

    def __init__(self, mol: QuantumChemistryBase):
        super().__init__(mol)
        self.pg = mol.abelian_point_group
        self._mol_irreps = mol.compute_orbital_irreps()

    @property
    def mol_irreps(self) -> list[str]:
        """list of str : irrep label of each molecular orbital."""
        return self._mol_irreps

    def get_irrep(self, state) -> str | None:
        # Canonical irrep assignment only works for states with integer occupations
        # (single determinants or superpositions within the same occupation sector).
        # Fractional occupations indicate a superposition across sectors, which
        # requires the LocalizedIrrepProvider's character-vector approach instead.
        """Return the irrep of ``state`` as the product of the irreps of its occupied orbitals.

        Parameters
        ----------
        state : FockSpaceState
            State with integer orbital occupations.

        Returns
        -------
        str or None
            Irrep name, None if the character product matches no irrep.

        Raises
        ------
        ValueError
            If an orbital occupation is fractional, e.g. for a superposition of different occupation
            sectors. Use the ``"loc"`` provider in that case.
        """
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
    """Print the names of the irrep providers installed on this system."""
    print("Available Irrep Providers")
    for k in INSTALLED_IRREP_PROVIDERS:
        print(k)


def show_supported_modules():
    """Print the names of the supported irrep providers."""
    print(SUPPORTED_IRREP_PROVIDERS)


# TODO: change "provider" with a better keyword
def IrrepProvider(
    mol: QuantumChemistryBase, provider: str = "canon", *args, **kwargs
) -> IrrepProviderBase:
    """Create the irrep provider that assigns irreducible representations to the FockSpaceStates of a molecule.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule whose point group defines the irreps.
    provider : str {"canon", "loc"}
        ``"canon"`` (``PySCFCanonicalIrrepProvider``) uses the irreps PySCF assigns to the canonical
        orbitals, in ``mol.abelian_point_group``, for states with integer occupations. ``"loc"``
        (``LocalizedIrrepProvider``) applies the symmetry operations to the state, in
        ``mol.point_group``, for any orbitals the operations map onto themselves (e.g. native or
        hybrid orbitals), and only for states of one-dimensional irreps.
    *args, **kwargs
        Passed to the provider class.

    Returns
    -------
    IrrepProviderBase

    Examples
    --------
    >>> provider = IrrepProvider(mol, "canon")
    >>> states = FockSpaceState.non_ionic_states(mol, provider)
    """
    # any kwargs should be managed inside each class
    return INSTALLED_IRREP_PROVIDERS[provider.lower()](*args, mol=mol, **kwargs)
