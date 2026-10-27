# molecules – the `Molecule` object

Entry point of sunrise. `sn.Molecule(...)` returns a molecule holding the integrals and orbitals; its **nature** decides how circuits are represented:

| `nature` | base class | circuits are |
|---|---|---|
| `"qubit"` (default) | `qubit_base` | tequila qubit circuits (JW, BK, … encodings) |
| `"fermionic"` / `"f"` | `fermionic_base` | `FCircuit` fermionic gates |
| `"hybrid"` / `"h"` | `hybrid_base` | mixed: some orbitals as hard-core bosons, others as fermions (`select=`), see [del Arco Santos et al., QST 2025](https://doi.org/10.1088/2058-9565/adbdee) |

Integrals come from [PySCF](https://pyscf.org/), [Psi4](https://psicode.org/) or [MADNESS](https://github.com/m-a-d-n-e-s-s/madness) (PNO-based basis-set-free orbitals, [Kottmann et al., JPCL 2021](https://doi.org/10.1021/acs.jpclett.0c03410)).

Useful methods: `make_ansatz("SPA", edges=...)`, `make_hamiltonian()`, `compute_energy("fci")`, `use_native_orbitals()`, `use_CLPO_orbitals_and_edges()`, `save()`.

```python
import sunrise as sn

mol = sn.Molecule(
    geometry="H 0 0 0\nLi 0 0 1.5", basis_set="sto-3g", nature="h", select="BBFBF"
)
U = mol.make_ansatz("SPA", edges=[(0, 1)])
mol.save("lih.h5")
mol2 = sn.load_molecule("lih.h5")
```

Converters: `MoleculeFromPyscf`, `MoleculeToPyscf`, `MoleculeFromOpenFermion`, `MoleculeFromSunrise`.

**See also:** [hybrid example](../../../examples/hybrid_molecule/README_example.py) · [tests](../../../tests/test_chemistry.py), [hybrid tests](../../../tests/test_hybrid_molecule.py)
