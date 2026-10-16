import random

import pytest
import numpy as np
import sunrise as sn
import tequila as tq
from sunrise.graph_selection import (
    generate_min_global_distance_edges,
    random_matching,
    generate_all_possible_edges,
    total_distance,
    brute_force,
    nearest_insertion,
    two_opt,
    simulated_annealing,
    genetic_algorithm,
    minimum_weight_perfect_performance,
    blossom_scaled,
    ant_colony_opt,
    genetic_algorithm_best_n_solutions,
    simulated_annealing_best_n_solutions,
    two_opt_best_n_solutions,
    random_blossom,
    random_blossom_scaled,
    run_spa_optimization,
)


def is_valid_matching(edges, num_atoms):
    if len(edges) != num_atoms // 2:
        return False
    atoms = set()
    for edge in edges:

        a, b = int(edge[0]), int(edge[1])
        if a == b or len(edge) != 2:
            return False

        if a in atoms or b in atoms:
            return False
        atoms.add(a)
        atoms.add(b)
    return True


def test_random_matching():
    num_atoms = 6
    edges = random_matching(num_atoms)
    assert is_valid_matching(edges, num_atoms)


def test_generate_all_possible_edges():
    edges = generate_all_possible_edges([0, 1, 2, 3])
    assert len(edges) > 0
    for edge in edges:
        assert is_valid_matching(edge, 4)


def test_generate_all_possible_edges_odd_atoms():

    edges = generate_all_possible_edges([0, 1, 2])
    assert edges == []


def test_total_distance():
    coordinates = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    edges = [(0, 1), (2, 3)]
    assert total_distance(edges, coordinates) == pytest.approx(2.0)


@pytest.fixture
def fixed_coordinates():
    rand = np.random.default_rng(seed=42)
    return rand.random((6, 3)) * 10


@pytest.mark.parametrize(
    "heuristic",
    [
        brute_force,
        two_opt,
        simulated_annealing,
        genetic_algorithm,
        minimum_weight_perfect_performance,
        blossom_scaled,
        ant_colony_opt,
    ],
)
def test_heuristic_return_valid_edges(heuristic, fixed_coordinates):
    num_atoms = 6
    edges = heuristic(num_atoms, fixed_coordinates)
    assert is_valid_matching(edges, num_atoms)


def test_nearest_insertion(fixed_coordinates):
    num_atoms = 6
    edges = nearest_insertion(fixed_coordinates)
    assert is_valid_matching(edges, num_atoms)


def test_generate_min_global_distance_edges(fixed_coordinates):
    num_atoms = 6
    edges = generate_min_global_distance_edges(fixed_coordinates)
    assert is_valid_matching(edges, num_atoms)


@pytest.mark.parametrize(
    "heuristic",
    [
        genetic_algorithm_best_n_solutions,
        simulated_annealing_best_n_solutions,
        two_opt_best_n_solutions,
        random_blossom,
        random_blossom_scaled,
    ],
)
def test_best_n_heuristics_return_valid_edges(heuristic, fixed_coordinates):
    num_atoms = 6
    kwargs = {"restart": 20} if heuristic is two_opt_best_n_solutions else {}
    # simulated annealing returns no solution if no move is accepted
    random.seed(0)
    solutions = heuristic(num_atoms, fixed_coordinates, best_n_solutions=3, **kwargs)
    assert 0 < len(solutions) <= 3
    for edges in solutions:
        assert is_valid_matching(edges, num_atoms)


def test_optimal_on_separated_pairs():
    # two well separated H2 units, the atom order interleaves them
    coordinates = np.array([[0.0, 0, 0], [10.0, 0, 0], [0.5, 0, 0], [10.5, 0, 0]])
    for edges in [generate_min_global_distance_edges(coordinates), minimum_weight_perfect_performance(4, coordinates)]:
        assert sorted(tuple(sorted(int(i) for i in edge)) for edge in edges) == [(0, 2), (1, 3)]


def test_nearest_neighbour_atom_indices(fixed_coordinates):
    # the edges have to refer to the original atom order for every start point
    edges = generate_min_global_distance_edges(fixed_coordinates)
    optimal = brute_force(6, fixed_coordinates)
    assert total_distance(edges, fixed_coordinates) >= total_distance(optimal, fixed_coordinates) - 1e-10
    assert total_distance(minimum_weight_perfect_performance(6, fixed_coordinates), fixed_coordinates) == pytest.approx(
        total_distance(optimal, fixed_coordinates)
    )


def test_run_spa_optimization():
    # no spatial symmetry, so that the default FCI guess finds the ground state
    coordinates = np.array([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [1.1, 1.5, 0.2], [-0.2, 1.3, 0.4]])
    geometry = "".join(f"h {x} {y} {z}\n" for x, y, z in coordinates)
    mol = sn.Molecule(geometry=geometry, basis_set="sto-3g")
    result = run_spa_optimization(mol, coordinates=coordinates, key_heuristic=minimum_weight_perfect_performance)

    fci = mol.compute_energy("fci")
    hf = mol.compute_energy("hf")
    assert fci - 1e-8 <= result["energy"] < hf

    # the readout angles are close to the optimal SPA angles in the optimized orbitals
    vqe = tq.minimize(tq.ExpectationValue(H=result["molecule"].make_hamiltonian(), U=result["circuit"]), silent=True)
    assert result["energy"] == pytest.approx(vqe.energy, abs=1e-4)
