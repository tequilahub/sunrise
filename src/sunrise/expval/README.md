# expval – expectation values and optimization

Everything needed to evaluate $\langle\psi|H|\psi\rangle$ (and overlaps $\langle\phi|\psi\rangle$) for fermionic circuits, and to optimize them.

- `Braket` / `Expval` – build an objective from a molecule/Hamiltonian and one or two circuits. Simulation is delegated to a **fermionic backend**:

  | backend | notes |
  |---|---|
  | `spex` | [spex](https://github.com/tequilahub/spex) sparse simulator, installed with sunrise (default) |
  | `tcc` | [TenCirChem](https://github.com/tencent-quantum-lab/TenCirChem) |
  | `fqe` | [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE) |
  | `tequila` | maps to qubits and uses any [tequila](https://github.com/tequilahub/tequila) simulator |

- `minimize` – optimize the circuit parameters.
- `optimize_orbitals` – orbital optimization around a fixed circuit, a cheap way to boost compact ansätze such as SPA ([Mizukami et al., PRR 2020](https://doi.org/10.1103/PhysRevResearch.2.033421)).

```python
import sunrise as sn

mol = sn.Molecule(
    geometry="H 0 0 0\nH 0 0 1\nH 0 0 2\nH 0 0 3", basis_set="sto-3g", nature="f"
)
U = sn.FCircuit.from_edges(n_orb=mol.n_orbitals, edges=[(0, 1), (2, 3)])
opt = sn.optimize_orbitals(circuit=U, molecule=mol, backend="tcc")
print(opt.energy)
```

**See also:** [tests](../../../tests/test_fermionic_expval.py) · [mixed qubit/fermionic tests](../../../tests/test_mixed_expval.py)
