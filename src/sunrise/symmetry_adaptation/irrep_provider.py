"""Irrep providers: objects that assign a point-group irreducible representation to a Fock-space state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy

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
    """Irrep provider based on the characters of the symmetry operations.

    The operations of the point group are applied to the state as permutations (with phases) of
    the localized orbitals, and the resulting character vector is compared to the character
    table. If the state defines ``fragment_orbitals``, only operations that map the fragment
    onto itself are used.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule with a ``point_group``.
    """

    def __init__(self, mol: QuantumChemistryBase):
        super().__init__(mol)

        # Fetch the MO-specific permutations and phases, keyed by the operations of the point group
        self._pg = mol.point_group
        self._spatial_perms, self._spatial_phases = mol.compute_mo_perms_and_phases()

    def get_irrep(self, state) -> str | None:
        """Return the irrep of ``state`` from its character vector, None if it matches no irrep.

        Parameters
        ----------
        state : FockSpaceState
            State to classify, possibly a superposition.
        """
        from .symmetrization_procedure import (
            SymmetryAdaptedLinearCombinationSymmetrization,
        )

        n_orbitals = self.mol.n_orbitals
        pg = self._pg

        fragment_orbitals = getattr(state, "fragment_orbitals", None)
        if fragment_orbitals is not None:
            fragment_set = set(fragment_orbitals)
            selected_ops = [
                label
                for label in pg.character_table.operation_symbols
                if all(
                    self._spatial_perms[label][i] in fragment_set
                    for i in fragment_orbitals
                )
            ]
        else:
            selected_ops = pg.character_table.operation_symbols

        chars = []
        for label in selected_ops:
            perm = self._spatial_perms[label]
            phases = self._spatial_phases[label]

            transformed = SymmetryAdaptedLinearCombinationSymmetrization._apply_spatial_permutation_bitstring(
                state.wavefunction, perm, n_orbitals, phases
            )
            expectation = state.wavefunction.inner(transformed).real
            chars.append(numpy.round(expectation))

        character_vector = numpy.array(chars)
        for irrep_name, irrep_chars in pg.character_table.dict.items():
            expected = [
                irrep_chars[pg.character_table.operation_symbols.index(label)]
                for label in selected_ops
            ]
            if all(abs(c - e) < 0.5 for c, e in zip(character_vector, expected)):
                return irrep_name

        return None


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
        self._pg = mol.abelian_point_group
        self._mol_irreps = mol.compute_orbital_irreps()

    @property
    def mol_irreps(self) -> list[str]:
        """list of str : Irrep label of each molecular orbital."""
        return self._mol_irreps

    def get_irrep(self, state) -> str | None:
        """Return the irrep of ``state`` as the product of the irreps of its occupied orbitals.

        Parameters
        ----------
        state : FockSpaceState
            State with integer orbital occupations.

        Returns
        -------
        str or None
            Irrep name. None if the state has no ``mo_occ`` or the character product matches no irrep.

        Raises
        ------
        ValueError
            If an orbital occupation is fractional, e.g. for a superposition of different occupation
            sectors. Use the ``"loc"`` provider in that case.
        """
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
            chars = numpy.asarray(
                pg.character_table.dict[irrep], dtype=numpy.complex128
            )
            character_vector = character_vector * numpy.power(chars, round(occ))
        irrep = pg.character_table.vec_to_str(character_vector)
        return irrep if irrep is not None else None


class MEAOIrrepProvider(LocalizedIrrepProvider):
    """Irrep provider for MEAO orbitals.

    Same as ``LocalizedIrrepProvider`` but the permutations and phases are those of the
    molecular-orbital basis of ``mol``.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule with a ``point_group``.
    """

    def __init__(self, mol: QuantumChemistryBase):
        # Skip parent __init__'s AO permutation logic
        IrrepProviderBase.__init__(self, mol)

        self._pg = mol.point_group
        self._spatial_perms, self._spatial_phases = mol.compute_mo_perms_and_phases()


SUPPORTED_IRREP_PROVIDERS = ["loc", "canon", "meao"]
INSTALLED_IRREP_PROVIDERS = {
    "loc": LocalizedIrrepProvider,
    "canon": PySCFCanonicalIrrepProvider,
    "meao": MEAOIrrepProvider,
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
    """Create an irrep provider by name.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule whose point group defines the irreps.
    provider : str {"canon", "loc", "meao"}
        ``"canon"`` uses the PySCF-labelled canonical orbitals (Abelian groups only), ``"loc"`` the
        characters of the symmetry operations on localized orbitals, ``"meao"`` the same for MEAOs.
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
    # any kwargs and circuit form should be managed inside each class
    return INSTALLED_IRREP_PROVIDERS[provider.lower()](*args, mol=mol, **kwargs)
