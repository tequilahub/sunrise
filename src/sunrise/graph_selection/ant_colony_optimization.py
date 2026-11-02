"""Ant colony optimization heuristic for pairing atoms."""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def ant_colony_opt(
    num_atoms: int,
    coordinates: np.ndarray,
    num_ants=10,
    num_iter=10,
    alpha=1,
    beta=2,
    evaporation=0.5,
    Q=100,
):
    """Find a short pairing of atoms with ant colony optimization.

    Each ant builds a perfect matching choosing partners with probability proportional to
    ``pheromone**alpha * heuristic**beta`` (heuristic = ``1 / (1 + distance)``). Pheromone
    evaporates every iteration and is reinforced on the edges of short matchings.

    Parameters
    ----------
    num_atoms : int
        Number of atoms (vertices); must be even.
    coordinates : numpy.ndarray
        Atom coordinates, shape ``(num_atoms, 3)``.
    num_ants : int, optional
        Ants (candidate matchings) per iteration. Default is 10.
    num_iter : int, optional
        Number of iterations. Default is 10.
    alpha : float, optional
        Weight of the pheromone. Default is 1.
    beta : float, optional
        Weight of the distance heuristic. Default is 2.
    evaporation : float, optional
        Fraction of pheromone lost each iteration. Default is 0.5.
    Q : float, optional
        Pheromone deposit scale. Default is 100.

    Returns
    -------
    list of tuple of int
        Best matching found, as edges ``(i, j)``.

    Examples
    --------
    >>> edges = ant_colony_opt(len(coordinates), coordinates)
    """
    distance_matrix = squareform(pdist(coordinates))
    heuristic = np.ones((num_atoms, num_atoms))
    pheromone = np.ones((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = distance_matrix[i][j]

            heuristic[i][j] = 1 / (1 + distance)
            heuristic[j][i] = heuristic[i][j]

    heuristic += 1e-12

    best_matching = None
    best_cost = float("inf")

    for _ in range(num_iter):
        ant_matchings = []
        for _ in range(num_ants):
            unmatched = set(range(num_atoms))
            matching = []

            while unmatched:
                i = min(unmatched)
                candidates = list(unmatched - {i})

                probs = []

                for j in candidates:
                    val = pow(pheromone[i][j], alpha) * pow(heuristic[i][j], beta)
                    probs.append(val)

                probs = np.array(probs)
                probs /= probs.sum()

                j = np.random.choice(candidates, p=probs)

                matching.append((i, j))

                unmatched.remove(i)
                unmatched.remove(j)

            ant_matchings.append(matching)

        for matching in ant_matchings:
            cost = sum(distance_matrix[i][j] for i, j in matching)

            if cost < best_cost:
                best_cost = cost
                best_matching = matching

        pheromone *= 1 - evaporation

        for matching in ant_matchings:
            cost = sum(distance_matrix[i][j] for i, j in matching)

            deposit = Q / (cost + 1e-12)

            for i, j in matching:
                pheromone[i][j] += deposit
                pheromone[j][i] += deposit

    return best_matching
