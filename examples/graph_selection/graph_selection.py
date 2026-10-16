"""
Select the SPA graph of a hydrogen cluster from the atomic positions and compute the SPA energy
with the angles read out from the hardcore-boson Hamiltonian (no VQE for the final energy).
"""

import numpy as np

import sunrise as sn
from sunrise.graph_selection import (
    generate_min_global_distance_edges,
    nearest_insertion,
    brute_force,
    two_opt,
    simulated_annealing,
    genetic_algorithm,
    minimum_weight_perfect_performance,
    blossom_scaled,
    ant_colony_opt,
    run_spa_optimization,
    total_distance,
)

coordinates = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.8, 0.1, 0.0],
        [1.1, 1.5, 0.2],
        [-0.2, 1.3, 0.4],
        [2.5, 0.3, 0.1],
        [3.1, 0.9, 0.6],
    ]
)
geometry = "".join(f"h {x} {y} {z}\n" for x, y, z in coordinates)
mol = sn.Molecule(geometry=geometry, basis_set="sto-3g")
num_atoms = len(coordinates)

graphs = {
    "generate_min_global_distance_edges": generate_min_global_distance_edges(
        coordinates
    ),
    "nearest_insertion": nearest_insertion(coordinates),
}
for heuristic in [
    brute_force,
    two_opt,
    simulated_annealing,
    genetic_algorithm,
    minimum_weight_perfect_performance,
    blossom_scaled,
    ant_colony_opt,
]:
    graphs[heuristic.__name__] = heuristic(num_atoms, coordinates)

for name, edges in graphs.items():
    print(
        f"{name:36s} {edges}  total distance = {total_distance(edges, coordinates):.4f}"
    )

result = run_spa_optimization(
    mol, coordinates=coordinates, key_heuristic=minimum_weight_perfect_performance
)
print("SPA energy ", result["energy"])
print("FCI energy ", mol.compute_energy("fci"))
