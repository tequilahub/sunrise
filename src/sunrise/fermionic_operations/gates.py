"""Factory functions that create single-gate ``FCircuit`` objects (excitations, orbital rotations, phases, Givens decompositions)."""

from __future__ import annotations

import numbers
import typing

from numpy import ndarray
from tequila.objective.objective import FixedVariable, Variable

from .circuit import FCircuit
from .fgateimpl import *
from .givens_rotations import get_givens_circuit


def FermionicExcitation(
    indices: list | tuple | None,
    variables: typing.Hashable | numbers.Real | Variable | FixedVariable | None = None,
    reordered: bool = False,
) -> FCircuit:
    """Create a generic fermionic excitation gate.

    Parameters
    ----------
    indices : list of tuple of int
        Pairs ``(from, to)`` of spin orbitals, e.g. ``[(0, 2), (1, 3)]`` is a double excitation.
    variables : str or float or tq.Variable, optional
        Angle of the gate: a variable name, a number or a tequila variable/objective.
    reordered : bool, optional
        True if the indices use up-then-down ordering, False (default) if they use
        interleaved up, down, up, down, ... ordering.

    Returns
    -------
    FCircuit

    Examples
    --------
    >>> U = sun.gates.FermionicExcitation([(0, 2), (1, 3)], "a")
    """
    return FCircuit.wrap_gate(FermionicExcitationImpl(indices, variables, reordered))


def UR(
    i: int,
    j: int,
    variables: typing.Hashable | numbers.Real | Variable | FixedVariable | None = None,
) -> FCircuit:
    r"""Create an orbital rotation gate between spatial orbitals ``i`` and ``j``.

    It is the pair of single excitations ``(2i, 2j)`` and ``(2i+1, 2j+1)`` (spin up and down)
    with the same angle:

    .. math::
        U_R(\theta) = e^{-i \frac{\theta}{2} (G_{2i \to 2j} + G_{2i+1 \to 2j+1})}, \qquad
        G_{p \to q} = i (a^\dagger_q a_p - a^\dagger_p a_q)

    Parameters
    ----------
    i, j : int
        Spatial orbital indices.
    variables : str or float or tq.Variable, optional
        Angle of the gate.

    Returns
    -------
    FCircuit

    Examples
    --------
    >>> U = sun.gates.UR(0, 1, variables="a") + sun.gates.UR(2, 3, variables="a")
    """
    return FCircuit.wrap_gate(URImpl(i, j, variables))


def UC(
    i: int,
    j: int,
    variables: typing.Hashable | numbers.Real | Variable | FixedVariable | None = None,
) -> FCircuit:
    r"""Create an orbital correlator gate between spatial orbitals ``i`` and ``j``.

    It is the paired double excitation ``[(2i, 2j), (2i+1, 2j+1)]``, moving an electron
    pair from orbital ``i`` to ``j``:

    .. math::
        U_C(\theta) = e^{-i \frac{\theta}{2} G}, \qquad
        G = i (a^\dagger_{2j} a_{2i} a^\dagger_{2j+1} a_{2i+1} - \mathrm{h.c.})

    Parameters
    ----------
    i, j : int
        Spatial orbital indices.
    variables : str or float or tq.Variable, optional
        Angle of the gate.

    Returns
    -------
    FCircuit

    Examples
    --------
    >>> U = sun.gates.UC(0, 1, "a")
    """
    return FCircuit.wrap_gate(UCImpl(i, j, variables))


def UX(
    indices: list | tuple | None,
    variables: typing.Hashable | numbers.Real | Variable | FixedVariable | None = None,
    reordered: bool = False,
) -> FCircuit:
    """Create a generic fermionic excitation gate; alias of ``FermionicExcitation`` with the same parameters."""
    return FCircuit.wrap_gate(FermionicExcitationImpl(indices, variables, reordered))


def Phase(
    target: int,
    variables: typing.Hashable | numbers.Real | Variable | FixedVariable | None = None,
    reordered: bool = False,
) -> FCircuit:
    """Create a phase gate on a spin orbital.

    ``Phase(p, theta)`` applies ``exp(-i * theta / 2 * n_p)``, i.e. multiplies the states where
    spin orbital ``p`` is occupied by ``exp(-i * theta / 2)``. Note the factor -1/2 with respect to
    ``tq.gates.Phase``. Some simulators only accept real wavefunctions, so take care with the angles.

    Parameters
    ----------
    target : int
        Spin orbital index.
    variables : str or float or tq.Variable, optional
        Phase angle.
    reordered : bool, optional
        True if the index uses up-then-down ordering. Default is False.

    Returns
    -------
    FCircuit

    Examples
    --------
    >>> U = sun.gates.Phase(0, variables="b")
    """
    return FCircuit.wrap_gate(PhaseImpl(target, variables, reordered))


def Givens(
    unitary: ndarray,
    tol: float = 1e-12,
    ordering: list | tuple | str | None = "OPTIMIZED_ORDERING",
) -> FCircuit:
    """Create a Givens rotation circuit that implements a real orthogonal matrix.

    The matrix (e.g. an orbital basis change) is decomposed into Givens rotations, applied as
    ``UR`` gates between pairs of orbitals, and ``Phase`` gates for the remaining signs.

    Parameters
    ----------
    unitary : numpy.ndarray
        Real unitary matrix (orbital basis change) to implement.
    tol : float, optional
        Angles below this value are dropped. Default is 1e-12.
    ordering : list of tuple of int or str or None, optional
        Order of the (row, column) eliminations, or ``"Optimized"`` for a linear-depth
        ordering (see ``get_givens_decomposition``).

    Returns
    -------
    FCircuit
    """
    return get_givens_circuit(unitary, tol, ordering)
