import inspect
from typing import Any, Sequence, TypedDict

import numpy as np
from numpy import eye
from numpy.typing import NDArray

from ..molecules.qubit_base import QuantumChemistryBase
from .shared import generate_min_global_distance_edges
from ..spafastprototype import spa_solve_graph, run_spa


class CustomData(TypedDict):
    name: str
    data: "list | str | float | int | np.number"


class OptimizationResult(TypedDict):
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
