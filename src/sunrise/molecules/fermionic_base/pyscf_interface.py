"""Fermionic molecule with integrals and classical methods from PySCF."""

from sunrise.molecules.qubit_base.pyscf_interface import PySCFMixin

from .fer_base import FermionicBase


class QuantumChemistryPySCF(PySCFMixin, FermionicBase):
    """Fermionic molecule with integrals and classical methods from PySCF, see ``PySCFMixin`` for
    the backend options.
    """
