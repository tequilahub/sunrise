from sunrise.molecules.qubit_base.pyscf_interface import PySCFMixin

from .fer_base import FermionicBase


class QuantumChemistryPySCF(PySCFMixin, FermionicBase):
    """
    Fermionic molecule with the integrals and the classical methods of PySCF
    see sunrise.molecules.qubit_base.pyscf_interface for all options
    """
