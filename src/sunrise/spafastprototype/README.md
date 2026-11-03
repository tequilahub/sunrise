# spafastprototype – fast SPA through circuit decomposition

An SPA circuit factorizes over clusters of orbitals, so its expectation values (usually of the hard-core boson Hamiltonian) can be split into smaller ones.
Each RDM element is expanded in Pauli strings, $\langle O\rangle_U = \sum_k c_k \langle P_k\rangle_U$, and each $\langle P_k\rangle_U$ is the product of the expectation values on the clusters, $\langle P_{k,0}\rangle_{U_0} \langle P_{k,1}\rangle_{U_1} \cdots$.
Only these local expectation values are computed, instead of simulating the full system.
It is qubit based and does not mix with fermionic expectation values for now.

- `run_spa` – optimize an SPA circuit together with the orbitals, using the decomposition for the RDMs. `grouping=` sets the clusters: None follows the circuit connectivity, a number groups blocks of about that size, a list sets them by hand.
- `decompose`, `make_decomposed_clusters`, `make_decomposed_circuits` – split a circuit and its expectation values into independent clusters
- `fast_rdm`, `fast_qtensor` – RDMs and tensors of expectation values from the decomposed circuits
- `spa_solve_graph`, `spa_angles_for_graph`, `spa_rdms` – classical SPA solution of a graph, without simulating any circuit

```python
import sunrise as sun
from numpy import eye

geometry = "".join(f"H 0 0 {i}\n" for i in range(8))
mol = sun.Molecule(
    geometry=geometry, basis_set="sto-3g", units="a"
).use_native_orbitals()
edges = [(2 * i, 2 * i + 1) for i in range(mol.n_orbitals // 2)]  # H-H H-H H-H H-H

initial_guess = eye(mol.n_orbitals)  # bonding/antibonding pair on each edge
for i, j in edges:
    initial_guess[i, j] = 1
    initial_guess[j, i] = -1

opt = sun.SPAFP.run_spa(mol, edges, initial_guess=initial_guess.T, silent=True)
print(opt.energy)  # -4.2620 (Hartree)
```

For larger molecules, combine it with the CLPO orbitals and edges (see [`CLPO`](../CLPO)).

**See also:** [example](../../../examples/spafastprototype/spafastprotoype.py) · [VB embedding example](../../../examples/VB_embedding/embedding.py) · [tests](../../../tests/test_fastspa.py)
