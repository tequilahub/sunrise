from dataclasses import dataclass
from os import PathLike
import numpy as np
import pandas as pd
import random 
import math 
import networkx as nx
from numpy.typing import NDArray
from tequila.quantumchemistry import QuantumChemistryBase
from scipy.spatial.distance import pdist, squareform



@dataclass
class Result:
    job_id: int
    method: str
    optimized_energy: float
    optimized_variables: list[float]
    edges: list[tuple[int, int]]


@dataclass
class DataFile:
    file_path: PathLike
    df: pd.DataFrame
    coordinates: NDArray[np.float64]
    method_results: dict[str, Result]


def read_result(job) -> Result:
    job_id = job["job_id"]
    method = job["method"]
    optimized_energy = job["optimized_energy"]
    if "optimized_variables" in job:
        optimized_variables = job["optimized_variables"]
        optimized_variable_counts = job["optimized_variable_count"]
        optimized_variable_cols = [
            [f"optimized_variable_{i}" for i in range(optimized_variable_count)]
            for optimized_variable_count in optimized_variable_counts
        ]

        for i, cols in enumerate(optimized_variable_cols):
            sample_variables = [job[col][i] for col in cols]
            optimized_variables.append(sample_variables)
    else:
        optimized_variables = []
    if "edges" in job:
        edges = job["edges"]
    else:
        edges = []
    return Result(
        job_id=job_id,
        method=method,
        optimized_energy=optimized_energy,
        optimized_variables=optimized_variables,
        edges=edges,
    )


def read_data_file(file_path: PathLike):
    # load the csv file into a pandas dataframe
    df = pd.read_csv(file_path)

    method_results = {}
    job_groups = df.groupby("method")
    for group_key, group in job_groups:
        method_results[group_key] = read_result(group)

    atom_counts = df["atom_count"]
    coordinate_cols = [
        [(f"x_{i}", f"y_{i}", f"z_{i}") for i in range(atom_count)]
        for atom_count in atom_counts
    ]

    coordinates = []
    for i, cols in enumerate(coordinate_cols):
        sample_coordinates = [
            (df[col[0]][i], df[col[1]][i], df[col[2]][i]) for col in cols
        ]
        coordinates.append(sample_coordinates)
    coordinates = np.array(coordinates)

    if "edge_0_start" in df:
        edges = []
        edge_counts = df["edge_count"]

        edge_cols = [
            [(f"edge_{i}_start", f"edge_{i}_end") for i in range(edge_count)]
            for edge_count in edge_counts
        ]

        for i, cols in enumerate(edge_cols):
            sample_edges = [(df[col[0]][i], df[col[1]][i]) for col in cols]
            edges.append(sample_edges)
    else:
        edges = []

    return DataFile(
        file_path=file_path,
        df=df,
        coordinates=coordinates,
        method_results=method_results,
    )


def generate_min_local_distance_edges(vertices: np.ndarray):
    """A locally optimal solution to the edge generation problem where it always considers the next possible shortest edge.

    Warning: This always depends on the order of the vertices, so the result may vary.

    Args:
        vertices: All vertices in the graph

    Returns
        A list of edges that make sure each vertex is connected to another vertex with the shortest possible edge.
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
    """
    input: vertices: np.ndarray
    A globally optimal solution to the edge generation problem.
    It will consider all possible start points for the optimization and return the one with the total minimum edge length.
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
    """this is a locally optimal solution to the edge generation problem, when inputting the start parameter, the greedy search will start from that index"""
    first_vertice = vertices[start]
    our_vertices = np.delete(vertices, start, axis=0)
    our_vertices = np.insert(our_vertices, 0, first_vertice, axis=0)
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
            if nearest_vertex is None:
                nearest_vertex_index = b
                nearest_vertex = vertex_b
            elif np.linalg.norm(vertex_a - nearest_vertex) > np.linalg.norm(
                vertex_a - vertex_b
            ):
                nearest_vertex_index = b
                nearest_vertex = vertex_b

        edges.append(tuple(sorted([a, nearest_vertex_index])))
        edge_length_sum += np.linalg.norm(vertex_a - nearest_vertex)
        done.update([tuple(vertex_a.tolist()), tuple(nearest_vertex.tolist())])
    return (edges, edge_length_sum)



# additional heuristics

#helper functions
def total_distance(edges: list[tuple[int, int]], coordinates: np.ndarray):
    return sum(np.linalg.norm(coordinates[i] - coordinates[j]) for i,j in edges)

def random_matching(num_atoms):
    atoms = list(range(num_atoms))
    random.shuffle(atoms)
    random_edges = [(atoms[i], atoms[i + 1]) for i in range(0, num_atoms, 2)]
    return random_edges


# brute force

def generate_all_possible_edges(atoms: list[int]):
    """ 
    Generate all perfect matchings of atoms.

    Parameters
    ----------
    atoms : list[int]
        Atom indices (must have even length).

    Returns
    -------
    list[list[tuple[int, int]]]
        All possible pairings of atoms. Each pairing is a list of edges (i, j).
        Returns an empty list if the number of atoms is odd.
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
                
                next_partial_edge.append((current_edge_pair + [new_edge_pair], new_remaining))

        partial_edges = next_partial_edge

    return edges

def brute_force(num_atoms: int, coordinates: np.ndarray):
    """
    Find optimal pairing via exhaustive search (minimum total distance).

    Parameters
    ----------
    num_atoms : int
    coordinates : np.ndarray

    Returns
    -------
    list[tuple[int, int]]
        Best edge configuration.
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
    """
    Build pairing using nearest insertion heuristic.

    Parameters
    ----------
    coordinates : np.ndarray

    Returns
    -------
    list[tuple[int, int]]
        Edge pairs from constructed tour.
    """

    distance_matrix = squareform(pdist(coordinates))  
    rows = distance_matrix.shape[0]
    unused = set(range(rows))
    explore = [0]
    unused.remove(0)
    
    nearest = min(unused, key = lambda j: distance_matrix[0, j])
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
def two_opt(num_atoms: int, coordinates: np.ndarray, max_iter = 200):

    """
    Improve pairing using 2-opt local search.

    Parameters
    ----------
    num_atoms : int
    coordinates : np.ndarray
    max_iter : int

    Returns
    -------
    list[tuple[int, int]]
        Improved edge configuration.
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

                new_edge_opt_one = edges[:i] + edges[i+1:j] + edges[j+1:] + [(a,c), (b,d)]
                new_edge_opt_two = edges[:i] + edges[i+1:j] + edges[j+1:] + [(a,d), (b,c)]

                distance_opt_one = total_distance(new_edge_opt_one, coordinates)
                distance_opt_two = total_distance(new_edge_opt_two, coordinates)

                if(distance_opt_one < distance):
                    edges = new_edge_opt_one
                    distance = distance_opt_one
                    improved = True
                    break

                elif(distance_opt_two < distance):
                    edges = new_edge_opt_two
                    distance = distance_opt_two
                    improved = True
                    break
            if(improved == True):
                break
    
    return edges

# simulated annealing

def random_neighbour(edges: list[tuple[int, int]]):

    """
    Generate neighboring solution via edge swap.

    Returns
    -------
    list[tuple[int, int]]
    """

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

def simulated_annealing(num_atoms: int,
                        coordinates: np.ndarray,
                        start = 1.0,
                        end = 1e-3,
                        alpha=0.95, 
                        max_iter = 1000):

    """
    Optimize pairing using simulated annealing.

    Parameters
    ----------
    num_atoms : int
    coordinates : np.ndarray
    start : float
    end : float
    alpha : float
    max_iter : int

    Returns
    -------
    list[tuple[int, int]]
        Best found configuration.
    """

    starting_edges = random_matching(num_atoms)
        
    current_edges = starting_edges[:]
    current_distance = total_distance(current_edges, coordinates=coordinates)

    best_edges, best_distance = current_edges[:], current_distance

    T = start
    i = 0
    no_improv = 0
    while(start > end and i < max_iter):

        i += 1
        new_edges = random_neighbour(current_edges)
        new_distance = total_distance(new_edges, coordinates=coordinates)

        difference = new_distance - current_distance

        if(difference < 0 or random.random() < math.exp(-difference / T)):
            current_edges = new_edges
            current_distance = new_distance
            if(current_distance < best_distance):
                best_edges = current_edges
                best_distance = current_distance
                no_improv = 0
            else:
                no_improv += 1

            if(no_improv >= 20):
                break    

        T *= alpha

    return best_edges

# genetic algorithm

def crossover(parent_a: list[tuple[int, int]], 
              parent_b: list[tuple[int, int]], 
              num_atoms: int):
    
    """
    Combine two parent solutions into a child.

    Returns
    -------
    list[tuple[int, int]]

    Notes
    -----
    **Crossover Strategy (Three-Phase Inheritance):**
    
    1. Common edges: Edges that appear in BOTH parents are always inherited
    
    2. Compatible parent edges: From remaining atoms, inherit edges where 
    
    3. Random completion: Any remaining unpaired atoms are randomly paired
    """

    child_edges = []
    used_atoms = set()

    for edge in parent_a:
        if edge in parent_b:
            child_edges.append(edge)
            used_atoms.update(edge)

    for parent in [parent_a, parent_b]:
        for(a, b) in parent:
            if a not in used_atoms and b not in used_atoms:
                child_edges.append((a, b))
                used_atoms.update((a, b))


    remaining_atoms = [a for a in range(num_atoms) if a not in used_atoms]
    random.shuffle(remaining_atoms)

    child_edges += [(remaining_atoms[i], remaining_atoms[i+1]) for i in range(0, len(remaining_atoms) - 1, 2)]

    return child_edges
    
def mutation(edges: list):

    """
    Mutate solution by swapping edges.

    Returns
    -------
    list[tuple[int, int]]
    """

    new_edges = edges[:]
    i, j = random.sample(range(len(new_edges)), 2)
    (a, b), (c, d) = new_edges[i], new_edges[j]

    if(random.random() < 0.5):
        edge1, edge2 = (a, c), (b, d)
    else:
        edge1, edge2 = (d, a), (b, c)

    if(edge1[0] == edge1[1] or edge2[0] == edge2[1] or edge1 == edge2):
        return edges  

    new_edges[i], new_edges[j] = edge1, edge2

    return new_edges

def genetic_algorithm(num_atoms: int, 
                      coordinates: np.ndarray, 
                      pop_size = 50, max_iter = 200, 
                      mutation_rate = 0.2, 
                      elite_size = 2):

    """
    Solve using genetic algorithm.

    Parameters
    ----------
    num_atoms : int
    coordinates : np.ndarray
    pop_size : int, optional
    max_iter : int, optional
    mutation_rate : float, optional
    elite_size : int, optional

    Returns
    -------
    list[tuple[int, int]]
        Best solution found.
    """

    population = [random_matching(num_atoms=num_atoms) for i in range(pop_size)]

    for iter in range(max_iter):
        fitness = [total_distance(edge, coordinates=coordinates) for edge in population]
        ranked = sorted(zip(fitness, population), key=lambda x: x[0])

        new_population = [edge for i, edge in ranked[:elite_size]]
        
        while(len(new_population) < pop_size):
            parent_a = random.choice(ranked[:pop_size//2])[1]
            parent_b = random.choice(ranked[:pop_size//2])[1]
            child = crossover(parent_a, parent_b, num_atoms)
            
            if(random.random() < mutation_rate):
                child = mutation(child)
                
            new_population.append(child)

        population = new_population
    
    _, best_edges = min([(total_distance(edge, coordinates=coordinates), edge) for edge in population], key=lambda x: x[0])

    return best_edges

# blossom

def minimum_weight_perfect_performance(num_atoms: int, coordinates: np.ndarray):
    
    """
    Solve optimal matching using Blossom algorithm.

    Parameters
    ----------
    num_atoms : int
    coordinates : np.ndarray

    Returns
    -------
    list[tuple[int, int]]
        Optimal matching.
    """

    distance_matrix = squareform(pdist(coordinates))  
    graph = nx.Graph()

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            graph.add_edge(i ,j, weight = distance_matrix[i][j])

    edges = nx.algorithms.matching.min_weight_matching(graph, weight="weight")
    return list(edges)
