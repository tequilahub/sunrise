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
    mol: QuantumChemistryBase

    def f(self, i: int, j: int) -> tequila.QCircuit:
        """
        This is a helper function for fermionic_swap.

        The fermionic swap (SWAP followed by CZ) of the two adjacent qubit
        modes (i, j), built from 2 CNOTs:
                |00> -> |00>, |01> -> |10>, |10> -> |01>, |11> -> -|11>

        Parameters
        ----------
        i : int
                First qubit index.
        j : int
                Second qubit index.

        Author
        ------
        Francisco Javier Del Arco Santos
        """
        return (
            tequila.gates.H(target=i)
            + tequila.gates.CNOT(target=j, control=i)
            + tequila.gates.CNOT(target=i, control=j)
            + tequila.gates.H(target=j)
        )

    def f_swap(self, i: int, j: int) -> tequila.QCircuit:
        """
        A function that swaps two orbitals (i, j) while accounting for fermionic antisymmetry.

        Note
        ----
        This implementation assumes that the orbitals are ordered as
        [up0, down0, up1, down1, ...] so that the spin-up and spin-down qubits
        for each orbital are adjacent.

        Parameters
        ----------
        i : int
                Index of the first orbital.
        j : int
                Index of the second orbital.

        Returns
        -------
        tequila.QCircuit
                A circuit that will perform the specified fermionic swap.

        Examples
        --------
        For example fermionic_swap(0,1) would perform the following transformations:
                +|0011> -> +|1100>
                +|0101> -> -|0101>

        Author
        ------
        Francisco Javier Del Arco Santos
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
