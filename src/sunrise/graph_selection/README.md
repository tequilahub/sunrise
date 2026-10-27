# graph_selection – choosing SPA edges

Pair ansätze like SPA ([Kottmann & Aspuru-Guzik, PRA 2022](https://doi.org/10.1103/PhysRevA.105.032449)) pair up orbitals (or atoms) along a **graph**: each edge is one electron pair. Which graph is best is a combinatorial problem — essentially a minimum-weight perfect matching / routing problem on the atoms.

This module offers several strategies, from exact to heuristic:

| kind | functions |
|---|---|
| exact | `brute_force`, `blossom_scaled` ([blossom algorithm](https://en.wikipedia.org/wiki/Blossom_algorithm)) |
| distance based | `generate_min_global_distance_edges`, `generate_min_local_distance_edges` |
| heuristics | `simulated_annealing`, `genetic_algorithm`, `two_opt`, `ant_colony_opt`, `nearest_insertion` |

and `run_spa_optimization` to score a candidate graph by its SPA energy.

```python
import numpy as np
from sunrise.graph_selection import generate_min_global_distance_edges
coords = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2.5], [0, 0, 3.5]])
edges = generate_min_global_distance_edges(coords)
```

**See also:** [example](../../../examples/graph_selection/graph_selection.py) · [tests](../../../tests/test_graph_selection.py)
