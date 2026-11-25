"""Energy objective of a linear combination of non-orthogonal circuits (``<H> / <S>``)."""

from __future__ import annotations

from numpy import sign
from tequila import BraKet as tqBraket
from tequila import Objective, QCircuit, TequilaException
from tequila import Overlap as tqOverlap

from sunrise import FCircuit
from sunrise.expval import Braket, Overlap
from sunrise.molecules.qubit_base import QuantumChemistryBase


def BigExpVal(
    circuits: list[QCircuit | FCircuit],
    coefficcents: list[float],
    mol: QuantumChemistryBase,
    **kwargs,
) -> Objective:
    """Return the energy objective of the state ``sum_i c_i |U_i>`` built from several circuits.

    The result is ``E / S`` with ``E = sum_ij c_i c_j <U_i|H|U_j>`` and ``S = sum_ij c_i c_j <U_i|U_j>``.
    The overlap ``S`` is clipped to at least 1e-6 in absolute value.

    Parameters
    ----------
    circuits : list of QCircuit or list of FCircuit
        Circuits preparing the basis states; all of the same type.
    coefficcents : list of float or tq.Variable
        Coefficients ``c_i`` (the argument name is spelled ``coefficcents``).
    mol : QuantumChemistryBase
        Molecule defining the Hamiltonian.
    **kwargs
        ``H`` : Hamiltonian for qubit circuits. Default is ``mol.make_hamiltonian()``.

    Returns
    -------
    tq.Objective

    Raises
    ------
    TequilaException
        If fermionic and qubit circuits are mixed.
    """
    n = len(circuits)
    SS = 0.0
    EE = 0.0
    ferm = all(isinstance(circuits[i], FCircuit) for i in range(n))
    qub = all(isinstance(circuits[i], QCircuit) for i in range(n))
    if not ferm and not qub:
        raise TequilaException(
            "Mixture of Fermionic and Qubit Circuits provided, can't handle it."
        )
    for i in range(n):
        for j in range(n):  # oder n ?
            if ferm:
                EE += (
                    (1 * coefficcents[i])
                    * (1 * coefficcents[j])
                    * Braket(ket=circuits[j], bra=circuits[i], mol=mol, operator="H")
                )
                SS += (
                    (1 * coefficcents[i])
                    * (1 * coefficcents[j])
                    * Overlap(ket=circuits[j], bra=circuits[i], mol=mol)
                )
            else:
                if "H" in kwargs:
                    H = kwargs["H"]
                    kwargs.pop("H")
                else:
                    H = mol.make_hamiltonian()
                EE += (
                    (1 * coefficcents[i])
                    * (1 * coefficcents[j])
                    * tqBraket(ket=circuits[i], bra=circuits[j], operator=H)
                )
                SS += (
                    (1 * coefficcents[i])
                    * (1 * coefficcents[j])
                    * tqOverlap(ket=circuits[i], bra=circuits[j])
                )
    f = lambda x: sign(x) * max(abs(x), 1.0e-6)
    SS = SS.apply(f)
    return EE / SS
