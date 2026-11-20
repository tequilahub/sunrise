# MEAO – maximally entangled atomic orbitals

Builds molecules in **maximally entangled atomic orbitals** (MEAO, [Ding, Matito & Schilling, arXiv 2025](https://arxiv.org/abs/2501.15699)), atom-centered orbitals defined through orbital entanglement. They are optimized in the space of the intrinsic atomic orbitals (IAOs) of the Hartree-Fock occupied orbitals, with the original [MEAO](https://github.com/schilling-group/MEAO) code through PySCF.

MEAO is not installed with sunrise:
```bash
pip install "MEAO @ git+https://github.com/schilling-group/MEAO"
```

| Function | Returns |
|---|---|
| `generate_MEAO_molecule(mol, minao="minao")` | molecule in MEAO basis; for basis sets larger than `minao` the MEAOs are completed by their orthonormal complement |

The same is available as molecule method for all natures: `mol.use_MEAO_orbitals()`. With a frozen core (the default) the core MEAOs are replaced by the canonical core orbitals; use `frozen_core=False` to get exactly the orbitals of the original code.

```python
import sunrise as sun

mol = sun.Molecule(geometry="Be 0 0 0\nH 0 0 1.3\nH 0 0 -1.3", basis_set="sto-3g")
mol = mol.use_MEAO_orbitals(minao="sto-3g")  # PySCF's "minao" has no p shell on Be
sun.plot_MO(mol, filename="BeH2_MEAO")  # look at the orbitals
```

**See also:** [CLPO and HAO](../CLPO) · [tests](../../../tests/test_chemistry.py)
