"""Run the SPA (separable pair approximation) optimization with edges chosen by a graph heuristic."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
from numpy import eye
from numpy.typing import NDArray

from ..molecules.qubit_base import QuantumChemistryBase
from ..spafastprototype import run_spa, spa_solve_graph
from .shared import generate_min_global_distance_edges


class CustomData(TypedDict):
    """Named piece of extra data attached to an optimization result."""

    name: str
    data: list | str | float | int | np.number


class OptimizationResult(TypedDict):
    """Result of an orbital/circuit optimization: energy, orbitals, optimized variables, circuit, molecule and optional custom data."""

    energy: float
    orbital_coefficients: NDArray | None
    orbital_transformation: NDArray | None
    variables: dict | None
    circuit: Any | None
    molecule: Any | None
    custom_data: Sequence[CustomData] | None


def generate_initial_guess_from_edges(
    vertices: NDArray[np.float64], edges: list[tuple[int, int]]
) -> NDArray:
    """Build the initial guess for the orbital transformation, mixing the two orbitals of each edge.

    Parameters
    ----------
    vertices : numpy.ndarray
        Atom coordinates; only their number (even) is used.
    edges : list of tuple of int
        Pairs of orbitals ``(i, j)``.

    Returns
    -------
    numpy.ndarray
        Square matrix of size ``len(vertices)``: identity, with the 2x2 block ``[[1, 1], [-1, 1]]`` on each edge (not normalized).

    Raises
    ------
    ValueError
        If the number of vertices is odd.
    """
    count = len(vertices)
    if count % 2 != 0:
        raise ValueError("count needs to be a multipe of two")
    initial_guess = eye(count)
    for edge in edges:
        initial_guess[edge[0]][edge[0]] = 1.0
        initial_guess[edge[0]][edge[1]] = 1.0
        initial_guess[edge[1]][edge[0]] = -1.0
        initial_guess[edge[1]][edge[1]] = 1.0
    return initial_guess


def run_spa_optimization(
    molecule: QuantumChemistryBase,
    *args,
    coordinates,
    key_heuristic=None,
    num_atoms=None,
    edges=None,
    **kwargs,
):
    """Optimize a molecule with SPA, choosing the pairs of orbitals (edges) from the geometry.

    Parameters
    ----------
    molecule : QuantumChemistryBase
        Molecule to optimize; its native orbitals are used.
    coordinates : numpy.ndarray
        Coordinates of the atoms (one orbital pair per two atoms).
    key_heuristic : callable, optional
        Heuristic returning the edges, either ``f(coordinates)`` or ``f(num_atoms, coordinates)``.
        Default is ``generate_min_global_distance_edges``.
    num_atoms : int, optional
        Number of atoms. Default is ``len(coordinates)``.
    edges : list of tuple of int, optional
        Edges to use directly, skipping the heuristic.

    Returns
    -------
    OptimizationResult
        Dictionary with ``energy``, ``orbital_coefficients``, ``orbital_transformation``,
        ``variables``, ``circuit`` (the SPA ansatz), ``molecule`` and ``custom_data`` (None).

    See Also
    --------
    sunrise.optimize_orbitals : Orbital optimization for a given circuit.
    minimum_weight_perfect_performance : Exact pairing, a common choice of ``key_heuristic``.

    Examples
    --------
    >>> result = run_spa_optimization(mol, coordinates=coordinates, key_heuristic=minimum_weight_perfect_performance)
    >>> result["energy"]
    """
    if num_atoms == None:
        num_atoms = len(coordinates)
    if edges == None:
        if key_heuristic == None:
            edges = generate_min_global_distance_edges(coordinates)

        else:
            sig = inspect.signature(key_heuristic)
            if len(sig.parameters) > 1:
                edges = key_heuristic(num_atoms, coordinates)
            else:
                edges = key_heuristic(coordinates)

    initial_guess = generate_initial_guess_from_edges(
        edges=edges, vertices=coordinates
    ).T

    opt = run_spa(
        mol=molecule.use_native_orbitals(),
        edges=edges,
        initial_guess=initial_guess,
        silent=True,
    )

    mol = opt.molecule

    graph = [tuple(edge) for edge in edges]
    variables, energy = spa_solve_graph(mol, graph)
    U_x = mol.make_ansatz(name="SPA", edges=edges)

    return OptimizationResult(
        energy=energy,
        orbital_coefficients=opt.molecule.integral_manager.orbital_coefficients,
        orbital_transformation=opt.mo_coeff,
        variables=variables,
        circuit=U_x,
        molecule=mol,
        custom_data=None,
    )
