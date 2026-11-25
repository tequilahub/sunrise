"""Fermionic swap circuits (``FermionicSWAP``) for rearranging orbitals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import tequila

if TYPE_CHECKING:
    # only for type hints, importing it at runtime would be circular
    from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase


# This is really inefficient but will work for now
@dataclass
class FermionicSWAP:
    """Build fermionic swap circuits for a molecule with qubits ordered as [up0, down0, up1, down1, ...].

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule the circuits are built for.

    Examples
    --------
    >>> fswap = FermionicSWAP(mol)
    >>> U = fswap.f_swap(0, 3)
    """

    mol: QuantumChemistryBase

    def f(self, i: int, j: int) -> tequila.QCircuit:
        """Return the exact fermionic swap of two adjacent qubit modes, the building block of ``f_swap``.

        The gate is a SWAP followed by a CZ, built from 2 CNOTs (and 2 Hadamards)::

            |00> -> |00>, |01> -> |10>, |10> -> |01>, |11> -> -|11>

        Parameters
        ----------
        i : int
            First qubit index.
        j : int
            Second qubit index.

        Returns
        -------
        tq.QCircuit

        Notes
        -----
        Written by Francisco Javier Del Arco Santos.

        Examples
        --------
        >>> U = FermionicSWAP(mol).f(0, 1)
        """
        return (
            tequila.gates.H(target=i)
            + tequila.gates.CNOT(target=j, control=i)
            + tequila.gates.CNOT(target=i, control=j)
            + tequila.gates.H(target=j)
        )

    def f_swap(self, i: int, j: int) -> tequila.QCircuit:
        """Swap orbitals ``i`` and ``j`` (both spin components) with fermionic antisymmetry.

        Parameters
        ----------
        i : int
            Index of the first orbital.
        j : int
            Index of the second orbital.

        Returns
        -------
        tq.QCircuit
            A circuit that performs the fermionic swap.

        Notes
        -----
        Assumes the qubits are ordered as [up0, down0, up1, down1, ...], so that the spin-up and
        spin-down qubits of each orbital are adjacent. Written by Francisco Javier Del Arco Santos.

        Examples
        --------
        ``f_swap(0, 1)`` performs the following transformations::

            +|0011> -> +|1100>
            +|0101> -> -|0101>

        >>> U = FermionicSWAP(mol).f_swap(0, 3)
        """
        U = tequila.QCircuit()
        for k in range(i, j):
            U += (
                self.f(2 * k + 1, 2 * (k + 1))
                + self.f(2 * k, 2 * k + 1)
                + self.f(2 * (k + 1), 2 * (k + 1) + 1)
                + self.f(2 * k + 1, 2 * (k + 1))
            )
        for k in range(i, j - 1)[::-1]:
            U += (
                self.f(2 * k + 1, 2 * (k + 1))
                + self.f(2 * k, 2 * k + 1)
                + self.f(2 * (k + 1), 2 * (k + 1) + 1)
                + self.f(2 * k + 1, 2 * (k + 1))
            )
        return U
