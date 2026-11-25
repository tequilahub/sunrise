"""Fermionic molecule with integrals and classical methods from Psi4."""

from sunrise.molecules.qubit_base.psi4_interface import Psi4Mixin

from .fer_base import FermionicBase


class QuantumChemistryPsi4(Psi4Mixin, FermionicBase):
    """Fermionic molecule with integrals and classical methods from Psi4, see ``Psi4Mixin`` for the
    backend options.
    """
