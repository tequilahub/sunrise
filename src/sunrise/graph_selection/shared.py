"""Heuristics that pair up atoms (perfect matchings, i.e. the edges used by SPA circuits), mostly minimizing the total edge length."""

import math
import random

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform


def generate_min_local_distance_edges(vertices: np.ndarray):
    """Pair atoms greedily by repeatedly taking the shortest edge between unpaired atoms.

    The result depends on the order of the vertices.

    Parameters
    ----------
    vertices : numpy.ndarray
        Atom coordinates, shape ``(n, 3)``.

    Returns
    -------
    list of tuple of int
        Edges ``(i, j)`` with ``i < j``, one per pair.

    Examples
    --------
    >>> edges = generate_min_local_distance_edges(coordinates)
    """
    all_edges = []
    done = set()
    our_vertices = vertices
    while len(done) < len(vertices):
        # find shortest edgee
        shortest_edge = None
        shortest_edge_indices = (0, 0)
        for a, vertex_a in enumerate(our_vertices):
            if tuple(vertex_a.tolist()) in done:
                continue
            for b, vertex_b in enumerate(our_vertices):
                if (
                    np.array_equal(vertex_a, vertex_b)
                    or tuple(vertex_b.tolist()) in done
                ):
                    continue
                if shortest_edge is None or shortest_edge > np.linalg.norm(
                    vertex_a - vertex_b
                ):
                    shortest_edge = np.linalg.norm(vertex_a - vertex_b)
                    shortest_edge_indices = (a, b)
        all_edges.append(tuple(sorted(shortest_edge_indices)))
        done.add(tuple(our_vertices[shortest_edge_indices[0]].tolist()))
        done.add(tuple(our_vertices[shortest_edge_indices[1]].tolist()))
    return all_edges


def generate_min_global_distance_edges(vertices: np.ndarray, nth_best=0):
    """Pair atoms by running the greedy nearest-neighbour search from every start vertex and keeping the shortest result.

    Default heuristic of ``run_spa_optimization``.

    Parameters
    ----------
    vertices : numpy.ndarray
        Atom coordinates, shape ``(n, 3)``.
    nth_best : int, optional
        Return the n-th best result instead of the best one (0). Default is 0.

    Returns
    -------
    list of tuple of int
        Edges ``(i, j)`` with ``i < j``, one per pair.

    Examples
    --------
    >>> edges = generate_min_global_distance_edges(coordinates)
    """
    all_edges = []
    all_edge_lengths = []
    for i in range(len(vertices)):
        edges, length_sum = generate_local_optimal_edges_from_vertices(
            vertices, start=i
        )
        all_edges.append(edges)
        all_edge_lengths.append(length_sum)

    min_edge_lengths = min(all_edge_lengths)
    max_edge_lengths = max(all_edge_lengths)
    if max_edge_lengths > min_edge_lengths:
        pass
        # print("FOUND MORE OPTIMAL EDGES", min_edge_lengths, max_edge_lengths)
    sorted_edges = np.argsort(all_edge_lengths)
    if nth_best > len(all_edges):
        raise ValueError(f"nth_best {nth_best} does not exist")
    return all_edges[sorted_edges[nth_best]]


def generate_local_optimal_edges_from_vertices(vertices: np.ndarray, start=0):
    """Pair atoms greedily with the nearest unpaired atom, starting the search from vertex ``start``.

    Parameters
    ----------
    vertices : numpy.ndarray
        Atom coordinates, shape ``(n, 3)``.
    start : int, optional
        Index of the vertex where the greedy search begins. Default is 0.

    Returns
    -------
    edges : list of tuple of int
        Edges ``(i, j)`` with ``i < j``.
    edge_length_sum : float
        Total length of the edges.
    """
    first_vertice = vertices[start]
    our_vertices = np.delete(vertices, start, axis=0)
    our_vertices = np.insert(our_vertices, 0, first_vertice, axis=0)
    # our_vertices[k] is vertices[order[k]]
    order = [start] + [i for i in range(len(vertices)) if i != start]
    done = set()
    edges = []
    edge_length_sum = 0
    for a, vertex_a in enumerate(our_vertices):
        if tuple(vertex_a.tolist()) in done:
            continue
        nearest_vertex = None
        nearest_vertex_index = None
        for b, vertex_b in enumerate(our_vertices):
            if np.array_equal(vertex_a, vertex_b) or tuple(vertex_b.tolist()) in done:
                continue
            if nearest_vertex is None or np.linalg.norm(
                vertex_a - nearest_vertex
            ) > np.linalg.norm(vertex_a - vertex_b):
                nearest_vertex_index = b
                nearest_vertex = vertex_b

        edges.append(tuple(sorted([order[a], order[nearest_vertex_index]])))
        edge_length_sum += np.linalg.norm(vertex_a - nearest_vertex)
        done.update([tuple(vertex_a.tolist()), tuple(nearest_vertex.tolist())])
    return (edges, edge_length_sum)


# additional heuristics


# helper functions
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


def matrix_scaling(distance_matrix: np.ndarray):
    """Return the distance matrix scaled element-wise as ``tanh(pi * d / 2)``.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Pairwise distances.
    """
    scaled_matrix = np.tanh(distance_matrix * np.pi / 2)
    return scaled_matrix


# brute force


def generate_all_possible_edges(atoms: list[int]):
    """Generate all perfect matchings of the given atoms.

    Parameters
    ----------
    atoms : list of int
        Atom indices; the length must be even.

    Returns
    -------
    list of list of tuple of int
        All pairings; each pairing is a list of edges ``(i, j)``. Empty if the number of atoms is odd.

    Examples
    --------
    >>> list(generate_all_possible_edges([0, 1, 2, 3]))  # three perfect matchings
    """
    edges = []
    partial_edges = [([], atoms)]

    if len(atoms) % 2 != 0:
        return edges

    while len(partial_edges) > 0:
        next_partial_edge = []

        for current_edge_pair, remaining in partial_edges:
            if len(remaining) == 0:
                edges.append(current_edge_pair)
                continue

            atom_1 = remaining[0]

            for i in range(1, len(remaining)):
                atom_2 = remaining[i]
                new_edge_pair = (atom_1, atom_2)
                new_remaining = []

                for j in range(1, len(remaining)):
                    if j != i:
                        new_remaining.append(remaining[j])

                next_partial_edge.append(
                    (current_edge_pair + [new_edge_pair], new_remaining)
                )

        partial_edges = next_partial_edge

    return edges


def brute_force(num_atoms: int, coordinates: np.ndarray):
    """Find the pairing with minimum total distance by exhaustive search.

    Parameters
    ----------
    num_atoms : int
        Number of atoms (vertices); must be even.
    coordinates : numpy.ndarray
        Atom coordinates, shape ``(num_atoms, 3)``.

    Returns
    -------
    list of tuple of int
        Best pairing.

    See Also
    --------
    minimum_weight_perfect_performance : Exact, with the Blossom algorithm.
    blossom_scaled : Exact Blossom on the scaled distance matrix.
    simulated_annealing : Heuristic, simulated annealing.
    two_opt : Heuristic, 2-opt local search.
    genetic_algorithm : Heuristic, genetic algorithm.

    Examples
    --------
    >>> edges = brute_force(len(coordinates), coordinates)  # coordinates: (n_atoms, 3) array
    """

    distance_matrix = squareform(pdist(coordinates))
    atoms = list(range(num_atoms))

    edges = generate_all_possible_edges(atoms)

    iterator = 0

    best_edge_config = None
    smallest_total_distance = None

    for edge in edges:
        total_distance = 0.0

        for a, b in edge:
            total_distance += distance_matrix[a][b]
            iterator += 1

        if smallest_total_distance is None or total_distance < smallest_total_distance:
            smallest_total_distance = total_distance
            best_edge_config = edge

    return best_edge_config


# nearest insertion


def nearest_insertion(coordinates: np.ndarray):
    """Build a pairing from a tour constructed with the nearest-insertion heuristic.

    The tour is cut into consecutive pairs.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Atom coordinates, shape ``(n, 3)``.

    Returns
    -------
    list of tuple of int
        Edges ``(i, j)``.

    Examples
    --------
    >>> edges = nearest_insertion(coordinates)
    """

    distance_matrix = squareform(pdist(coordinates))
    rows = distance_matrix.shape[0]
    unused = set(range(rows))
    explore = [0]
    unused.remove(0)

    nearest = min(unused, key=lambda j: distance_matrix[0, j])
    explore.append(nearest)
    unused.remove(nearest)

    while unused:
        best = None
        for atom in unused:
            for i in range(len(explore)):
                distance = distance_matrix[explore[i], atom]
                if best is None or distance < best[0]:
                    best = (distance, explore[i], atom)

        _, pre, new = best
        index = explore.index(pre)
        explore.insert(index + 1, new)
        unused.remove(new)
    edges = [(explore[i], explore[i + 1]) for i in range(0, len(explore) - 1, 2)]
    return edges


# 2-opt
def two_opt(num_atoms: int, coordinates: np.ndarray, max_iter=200):
    """Improve a random pairing with a 2-opt local search (swapping partners of two edges).

    Parameters
    ----------
    num_atoms : int
        Number of atoms (vertices); must be even.
    coordinates : numpy.ndarray
        Atom coordinates, shape ``(num_atoms, 3)``.
    max_iter : int, optional
        Maximum number of improvement sweeps. Default is 200.

    Returns
    -------
    list of tuple of int
        Improved pairing.

    See Also
    --------
    brute_force : Exact, by exhaustive search (small systems only).
    minimum_weight_perfect_performance : Exact, with the Blossom algorithm.
    blossom_scaled : Exact Blossom on the scaled distance matrix.
    simulated_annealing : Heuristic, simulated annealing.
    genetic_algorithm : Heuristic, genetic algorithm.

    Examples
    --------
    >>> edges = two_opt(len(coordinates), coordinates)
    """

    edges = random_matching(num_atoms=num_atoms)
    distance = total_distance(edges=edges, coordinates=coordinates)

    improved = True
    iter = 0

    while improved == True and iter < max_iter:
        improved = False
        iter += 1

        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                (a, b) = edges[i]
                (c, d) = edges[j]

                new_edge_opt_one = (
                    edges[:i] + edges[i + 1 : j] + edges[j + 1 :] + [(a, c), (b, d)]
                )
                new_edge_opt_two = (
                    edges[:i] + edges[i + 1 : j] + edges[j + 1 :] + [(a, d), (b, c)]
                )

                distance_opt_one = total_distance(new_edge_opt_one, coordinates)
                distance_opt_two = total_distance(new_edge_opt_two, coordinates)

                if distance_opt_one < distance:
                    edges = new_edge_opt_one
                    distance = distance_opt_one
                    improved = True
                    break

                elif distance_opt_two < distance:
                    edges = new_edge_opt_two
                    distance = distance_opt_two
                    improved = True
                    break
            if improved == True:
                break

    return edges


# simulated annealing


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


def simulated_annealing(
    num_atoms: int,
    coordinates: np.ndarray,
    start=1.0,
    end=1e-3,
    alpha=0.95,
    max_iter=1000,
):
    """Find a short pairing by simulated annealing.

    Starts from a random pairing and moves to neighbouring pairings, accepting worse ones
    with a probability that decreases as the temperature drops.

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

    Returns
    -------
    list of tuple of int
        Best pairing found.

    See Also
    --------
    brute_force : Exact, by exhaustive search (small systems only).
    minimum_weight_perfect_performance : Exact, with the Blossom algorithm.
    blossom_scaled : Exact Blossom on the scaled distance matrix.
    two_opt : Heuristic, 2-opt local search.
    genetic_algorithm : Heuristic, genetic algorithm.

    Examples
    --------
    >>> edges = simulated_annealing(len(coordinates), coordinates)
    """

    starting_edges = random_matching(num_atoms)

    current_edges = starting_edges[:]
    current_distance = total_distance(current_edges, coordinates=coordinates)

    best_edges, best_distance = current_edges[:], current_distance

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
            if current_distance < best_distance:
                best_edges = current_edges
                best_distance = current_distance
                no_improv = 0
            else:
                no_improv += 1

            if no_improv >= 20:
                break

        T *= alpha

    return best_edges


# genetic algorithm


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


def genetic_algorithm(
    num_atoms: int,
    coordinates: np.ndarray,
    pop_size=50,
    max_iter=200,
    mutation_rate=0.2,
    elite_size=2,
):
    """Find a short pairing with a genetic algorithm.

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
        Probability that a child is mutated. Default is 0.2.
    elite_size : int, optional
        Number of best pairings copied unchanged to the next generation. Default is 2.

    Returns
    -------
    list of tuple of int
        Best pairing found.

    See Also
    --------
    brute_force : Exact, by exhaustive search (small systems only).
    minimum_weight_perfect_performance : Exact, with the Blossom algorithm.
    blossom_scaled : Exact Blossom on the scaled distance matrix.
    simulated_annealing : Heuristic, simulated annealing.
    two_opt : Heuristic, 2-opt local search.

    Examples
    --------
    >>> edges = genetic_algorithm(len(coordinates), coordinates)
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

    _, best_edges = min(
        [(total_distance(edge, coordinates=coordinates), edge) for edge in population],
        key=lambda x: x[0],
    )

    return best_edges


# blossom


def minimum_weight_perfect_performance(num_atoms: int, coordinates: np.ndarray):
    """Find the pairing with minimum total distance with the Blossom algorithm (exact).

    Parameters
    ----------
    num_atoms : int
        Number of atoms (vertices); must be even.
    coordinates : numpy.ndarray
        Atom coordinates, shape ``(num_atoms, 3)``.

    Returns
    -------
    list of tuple of int
        Optimal pairing.

    See Also
    --------
    brute_force : Exact, by exhaustive search (small systems only).
    blossom_scaled : Exact Blossom on the scaled distance matrix.
    simulated_annealing : Heuristic, simulated annealing.
    two_opt : Heuristic, 2-opt local search.
    genetic_algorithm : Heuristic, genetic algorithm.

    Examples
    --------
    >>> edges = minimum_weight_perfect_performance(len(coordinates), coordinates)
    """

    distance_matrix = squareform(pdist(coordinates))
    graph = nx.Graph()

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            graph.add_edge(i, j, weight=distance_matrix[i][j])

    edges = nx.algorithms.matching.min_weight_matching(graph, weight="weight")
    return list(edges)


def blossom_scaled(num_atoms: int, coordinates: np.ndarray):
    """Find the minimum-weight pairing with the Blossom algorithm on the scaled distance matrix (see
    ``matrix_scaling``).

    Parameters
    ----------
    num_atoms : int
        Number of atoms (vertices); must be even.
    coordinates : numpy.ndarray
        Atom coordinates, shape ``(num_atoms, 3)``.

    Returns
    -------
    list of tuple of int
        Optimal pairing for the scaled distances.

    See Also
    --------
    brute_force : Exact, by exhaustive search (small systems only).
    minimum_weight_perfect_performance : Exact, with the Blossom algorithm.
    simulated_annealing : Heuristic, simulated annealing.
    two_opt : Heuristic, 2-opt local search.
    genetic_algorithm : Heuristic, genetic algorithm.

    Examples
    --------
    >>> edges = blossom_scaled(len(coordinates), coordinates)
    """
    distance_matrix = squareform(pdist(coordinates))
    scaled_matrix = matrix_scaling(distance_matrix)
    graph = nx.Graph()
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            graph.add_edge(i, j, weight=scaled_matrix[i][j])

    edges = nx.algorithms.matching.min_weight_matching(graph, weight="weight")
    return list(edges)
