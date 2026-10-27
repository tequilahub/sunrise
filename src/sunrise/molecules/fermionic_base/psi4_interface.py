from sunrise.molecules.qubit_base.psi4_interface import Psi4Mixin

from .fer_base import FermionicBase


class QuantumChemistryPsi4(Psi4Mixin, FermionicBase):
    """
    Fermionic molecule with the integrals and the classical methods of psi4
    see sunrise.molecules.qubit_base.psi4_interface for all options
    """
