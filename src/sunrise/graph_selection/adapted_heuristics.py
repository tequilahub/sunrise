"""Variants of the pairing heuristics of ``shared`` that return several good pairings (best n solutions) instead of one."""

import math
import random

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform

from .shared import matrix_scaling


def total_distance(edges: list[tuple[int, int]], coordinates: np.ndarray):
    """Return the summed distance of the ``(i, j)`` pairs in ``edges`` for the given coordinates.

    Parameters
    ----------
    edges : list of tuple of int
        Pairs ``(i, j)`` of atom indices.
    coordinates : numpy.ndarray
        Atomic coordinates, one row per atom.

    Returns
    -------
    float

    Examples
    --------
    >>> d = total_distance(edges, coordinates)
    """
    return sum(np.linalg.norm(coordinates[i] - coordinates[j]) for i, j in edges)


def random_matching(num_atoms: int):
    """Return a random perfect matching of ``num_atoms`` atoms as a list of ``(i, j)`` pairs.

    Examples
    --------
    >>> edges = random_matching(num_atoms)
    """
    atoms = list(range(num_atoms))
    random.shuffle(atoms)
    random_edges = [(atoms[i], atoms[i + 1]) for i in range(0, num_atoms, 2)]
    return random_edges


# genetic algorithm for best n solutions


def crossover(
    parent_a: list[tuple[int, int]], parent_b: list[tuple[int, int]], num_atoms: int
):
    """Combine two parent pairings into a child pairing.

    Parameters
    ----------
    parent_a, parent_b : list of tuple of int
        Parent pairings.
    num_atoms : int
        Number of atoms.

    Returns
    -------
    list of tuple of int

    Notes
    -----
    **Crossover strategy (three-phase inheritance):**

    1. Common edges: edges that appear in BOTH parents are always inherited.
    2. Compatible parent edges: from the remaining atoms, inherit the edges of ``parent_a``,
       then of ``parent_b``, whose two atoms are both still unpaired.
    3. Random completion: any remaining unpaired atoms are paired randomly.
    """

    child_edges = []
    used_atoms = set()

    for edge in parent_a:
        if edge in parent_b:
            child_edges.append(edge)
            used_atoms.update(edge)

    for parent in [parent_a, parent_b]:
        for a, b in parent:
            if a not in used_atoms and b not in used_atoms:
                child_edges.append((a, b))
                used_atoms.update((a, b))

    remaining_atoms = [a for a in range(num_atoms) if a not in used_atoms]
    random.shuffle(remaining_atoms)

    child_edges += [
        (remaining_atoms[i], remaining_atoms[i + 1])
        for i in range(0, len(remaining_atoms) - 1, 2)
    ]

    return child_edges


def mutation(edges: list):
    """Return a mutated copy of the pairing, obtained by recombining the atoms of two randomly chosen edges."""

    new_edges = edges[:]
    i, j = random.sample(range(len(new_edges)), 2)
    (a, b), (c, d) = new_edges[i], new_edges[j]

    if random.random() < 0.5:
        edge1, edge2 = (a, c), (b, d)
    else:
        edge1, edge2 = (d, a), (b, c)

    if edge1[0] == edge1[1] or edge2[0] == edge2[1] or edge1 == edge2:
        return edges

    new_edges[i], new_edges[j] = edge1, edge2

    return new_edges


def genetic_algorithm_best_n_solutions(
    num_atoms: int,
    coordinates: np.ndarray,
    pop_size=50,
    max_iter=200,
    mutation_rate=0.9,
    elite_size=2,
    best_n_solutions=10,
):
    """Run a genetic algorithm and return the best pairings of the final population.

    Parameters
    ----------
    num_atoms : int
        Number of atoms (vertices); must be even.
    coordinates : numpy.ndarray
        Atom coordinates, shape ``(num_atoms, 3)``.
    pop_size : int, optional
        Population size. Default is 50.
    max_iter : int, optional
        Number of generations. Default is 200.
    mutation_rate : float, optional
        Probability that a child is mutated. Default is 0.9.
    elite_size : int, optional
        Number of best pairings copied unchanged to the next generation. Default is 2.
    best_n_solutions : int, optional
        Number of solutions to return. Default is 10.

    Returns
    -------
    list of list of tuple of int
        Up to ``best_n_solutions`` pairings sorted by increasing total distance.
    """

    population = [random_matching(num_atoms=num_atoms) for i in range(pop_size)]

    for iter in range(max_iter):
        fitness = [total_distance(edge, coordinates=coordinates) for edge in population]
        ranked = sorted(zip(fitness, population), key=lambda x: x[0])

        new_population = [edge for i, edge in ranked[:elite_size]]

        while len(new_population) < pop_size:
            parent_a = random.choice(ranked[: pop_size // 2])[1]
            parent_b = random.choice(ranked[: pop_size // 2])[1]
            child = crossover(parent_a, parent_b, num_atoms)

            if random.random() < mutation_rate:
                child = mutation(child)

            new_population.append(child)

        population = new_population

    ranked = sorted(
        [(total_distance(edge, coordinates=coordinates), edge) for edge in population],
        key=lambda x: x[0],
    )
    # print(ranked[:best_n_solutions])
    return [edge for _, edge in ranked[:best_n_solutions]]


def random_neighbour(edges: list[tuple[int, int]]):
    """Return a neighbouring pairing obtained by swapping the partners of two randomly chosen edges."""

    new_edges = edges[:]
    i, j = random.sample(range(len(new_edges)), 2)
    (a, b), (c, d) = new_edges[i], new_edges[j]

    if random.random() < 0.5:
        edge1, edge2 = (a, c), (b, d)
    else:
        edge1, edge2 = (d, a), (b, c)

    if edge1[0] == edge1[1] or edge2[0] == edge2[1] or edge1 == edge2:
        return edges

    new_edges[i], new_edges[j] = edge1, edge2

    return new_edges


# simulated annealing for best n solutions


def simulated_annealing_best_n_solutions(
    num_atoms: int,
    coordinates: np.ndarray,
    start=1.0,
    end=1e-3,
    alpha=0.95,
    max_iter=1000,
    best_n_solutions=10,
):
    """Run simulated annealing and return the best distinct pairings visited.

    Parameters
    ----------
    num_atoms : int
        Number of atoms (vertices); must be even.
    coordinates : numpy.ndarray
        Atom coordinates, shape ``(num_atoms, 3)``.
    start : float, optional
        Initial temperature. Default is 1.0.
    end : float, optional
        Final temperature. Default is 0.001.
    alpha : float, optional
        Cooling factor applied each step. Default is 0.95.
    max_iter : int, optional
        Maximum number of steps. Default is 1000.
    best_n_solutions : int, optional
        Number of solutions to return. Default is 10.

    Returns
    -------
    list of list of tuple of int
        Up to ``best_n_solutions`` pairings sorted by increasing total distance.
    """

    starting_edges = random_matching(num_atoms)

    current_edges = starting_edges[:]
    current_distance = total_distance(current_edges, coordinates=coordinates)

    solutions = {}

    T = start
    i = 0
    no_improv = 0
    while start > end and i < max_iter:
        i += 1
        new_edges = random_neighbour(current_edges)
        new_distance = total_distance(new_edges, coordinates=coordinates)

        difference = new_distance - current_distance

        if difference < 0 or random.random() < math.exp(-difference / T):
            current_edges = new_edges
            current_distance = new_distance

            key = tuple(sorted(tuple(sorted(edge)) for edge in current_edges))
            solutions[key] = (current_distance, current_edges[:])

            if difference < 0:
                no_improv = 0
            else:
                no_improv += 1

            if no_improv >= 50:
                break

        T *= alpha

    best = sorted(solutions.values(), key=lambda x: x[0])
    return [edge for _, edge in best[:best_n_solutions]]


def two_opt_best_n_solutions(
    num_atoms: int,
    coordinates: np.ndarray,
    best_n_solutions=10,
    max_iter=1000,
    restart=1000,
):
    """Run 2-opt local search from many random starts and return the best distinct pairings found.

    Parameters
    ----------
    num_atoms : int
        Number of atoms (vertices); must be even.
    coordinates : numpy.ndarray
        Atom coordinates, shape ``(num_atoms, 3)``.
    best_n_solutions : int, optional
        Number of solutions to return. Default is 10.
    max_iter : int, optional
        Maximum number of improvement sweeps per restart. Default is 1000.
    restart : int, optional
        Number of random restarts. Default is 1000.

    Returns
    -------
    list of list of tuple of int
        Up to ``best_n_solutions`` pairings sorted by increasing total distance.
    """
    solutions = {}
    for _ in range(restart):
        edges = random_matching(num_atoms)
        current_distance = total_distance(edges, coordinates=coordinates)

        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(len(edges)):
                for j in range(i + 1, len(edges)):
                    a, b = edges[i]
                    c, d = edges[j]

                    candiate_edges = [
                        edges[:i]
                        + edges[i + 1 : j]
                        + edges[j + 1 :]
                        + [(a, c), (b, d)],
                        edges[:i]
                        + edges[i + 1 : j]
                        + edges[j + 1 :]
                        + [(a, d), (b, c)],
                    ]

                    for edge in candiate_edges:
                        new_distance = total_distance(edge, coordinates=coordinates)
                        if new_distance < current_distance:
                            edges = edge
                            current_distance = new_distance
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        key = tuple(sorted(tuple(sorted(edge)) for edge in edges))
        solutions[key] = (current_distance, edges[:])

    rank = sorted(solutions.values(), key=lambda x: x[0])
    # print(len(rank))
    return [edge for _, edge in rank[:best_n_solutions]]


# blossom for best n solutions
def random_blossom(num_atoms: int, coordinates: np.ndarray, best_n_solutions=10):
    """Return several pairings from the Blossom algorithm, each time excluding one random edge of the previous matching.

    Parameters
    ----------
    num_atoms : int
        Number of atoms (vertices); must be even.
    coordinates : numpy.ndarray
        Atom coordinates, shape ``(num_atoms, 3)``.
    best_n_solutions : int, optional
        Number of solutions to return. Default is 10.

    Returns
    -------
    list of list of tuple of int
        ``best_n_solutions`` matchings (the first one is the optimal matching).
    """
    sol = []

    excluded_edges = set()

    distance_matrix = squareform(pdist(coordinates))

    for _ in range(best_n_solutions):
        graph = nx.Graph()
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                edge = (i, j)

                if edge in excluded_edges:
                    continue

                graph.add_edge(i, j, weight=distance_matrix[i][j])

        matching = nx.algorithms.matching.min_weight_matching(graph, weight="weight")
        matching = list(matching)
        sol.append(matching)

        excluded_edges = random.choice(matching)
        excluded_edges = tuple(sorted(excluded_edges))
        excluded_edges = {excluded_edges}
    return sol


def random_blossom_scaled(num_atoms: int, coordinates: np.ndarray, best_n_solutions=10):
    """Same as ``random_blossom`` but with the scaled distance matrix (see ``matrix_scaling``).

    Parameters
    ----------
    num_atoms : int
        Number of atoms (vertices); must be even.
    coordinates : numpy.ndarray
        Atom coordinates, shape ``(num_atoms, 3)``.
    best_n_solutions : int, optional
        Number of solutions to return. Default is 10.

    Returns
    -------
    list of list of tuple of int
        ``best_n_solutions`` matchings (the first one is optimal for the scaled distances).
    """
    sol = []

    excluded_edges = set()

    distance_matrix = squareform(pdist(coordinates))
    scaled_matrix = matrix_scaling(distance_matrix)
    for _ in range(best_n_solutions):
        graph = nx.Graph()
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                edge = (i, j)

                if edge in excluded_edges:
                    continue

                graph.add_edge(i, j, weight=scaled_matrix[i][j])

        matching = nx.algorithms.matching.min_weight_matching(graph, weight="weight")
        matching = list(matching)
        sol.append(matching)

        excluded_edges = random.choice(matching)
        excluded_edges = tuple(sorted(excluded_edges))
        excluded_edges = {excluded_edges}
    return sol


heuristics = {
    "genetic_algorithm": genetic_algorithm_best_n_solutions,
    "simulated_annealing": simulated_annealing_best_n_solutions,
    "2-opt": two_opt_best_n_solutions,
    "blossom": random_blossom,
    "scaled blossom": random_blossom_scaled,
}
