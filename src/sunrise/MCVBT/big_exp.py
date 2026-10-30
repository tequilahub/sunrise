from __future__ import annotations

from tequila import BraKet as tqBraket
from tequila import Objective, QCircuit, TequilaException
from tequila import Overlap as tqOverlap
from tequila.autograd_imports import numpy as anp

from sunrise import FCircuit
from sunrise.expval import Braket, Overlap
from sunrise.molecules.qubit_base import QuantumChemistryBase

_NORM_TOL = 1.0e-10


def BigExpVal(
    circuits: list[QCircuit | FCircuit],
    coefficcents: list[float],
    mol: QuantumChemistryBase,
    **kwargs,
) -> Objective:
    n = len(circuits)
    SS = 0.0
    EE = 0.0
    ferm = all(isinstance(circuits[i], FCircuit) for i in range(n))
    qub = all(isinstance(circuits[i], QCircuit) for i in range(n))
    if not ferm:
        if "H" in kwargs:
            H = kwargs["H"]
            kwargs.pop("H")
        else:
            H = mol.make_hamiltonian()
    if not ferm and not qub:
        raise TequilaException(
            "Mixture of Fermionic and Qubit Circuits provided, can't handle it."
        )
    for i in range(n):
        for j in range(i, n):
            weight = (1 if i == j else 2) * coefficcents[i] * coefficcents[j]
            if ferm:
                EE += weight * Braket(
                    ket=circuits[j], bra=circuits[i], mol=mol, operator="H"
                )
                SS += weight * Overlap(ket=circuits[j], bra=circuits[i], mol=mol)
            elif i == j:
                EE += weight * tqBraket(ket=circuits[i], bra=circuits[i], operator=H)
                SS += weight * tqOverlap(ket=circuits[i], bra=circuits[i])
            else:
                weight = 0.5 * weight
                EE += weight * (
                    tqBraket(ket=circuits[j], bra=circuits[i], operator=H)
                    + tqBraket(ket=circuits[i], bra=circuits[j], operator=H)
                )
                SS += weight * (
                    tqOverlap(ket=circuits[j], bra=circuits[i])
                    + tqOverlap(ket=circuits[i], bra=circuits[j])
                )
    # SS = <Phi|Phi> >= 0 with Phi = sum_i c_i |i>. If the components collapse (SS ~ 0)
    # the energy is set to inf, so the optimizer rejects that step instead of using E/S
    # with a vanishing norm. anp.where keeps the transformations differentiable.
    norm = SS.apply(lambda x: anp.where(x < _NORM_TOL, 1.0, x))
    collapsed = SS.apply(lambda x: anp.where(x < _NORM_TOL, anp.inf, 0.0))
    return EE / norm + collapsed
