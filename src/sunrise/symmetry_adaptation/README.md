# symmetry_adaptation – spin and point-group adapted states

Builds many-electron states that respect the molecule's symmetries: total spin ($S^2$) and spatial point group. Starting from Fock-space determinants, it forms linear combinations that transform as a single irreducible representation (*irrep*) — useful as reference states and to restrict ansätze to the right symmetry sector.

- `FockSpaceState` – a determinant/linear combination; `FockSpaceState.dataframe(...)` prints a table
- `IrrepProvider` – assigns irreps to orbitals (canonical or localized)
- `PointGroup`, `PointGroupRepresentation` – character tables
- `SpinSymmetrizationProcedure`, `SpinCGSymmetrizationProcedure` (Clebsch–Gordan), `SymmetryAdaptedLinearCombintationSymmetrization` ([SALC](https://en.wikipedia.org/wiki/Symmetry-adapted_linear_combination))

```python
import sunrise as sun
from sunrise.symmetry_adaptation import (
    FockSpaceState,
    IrrepProvider,
    SpinSymmetrizationProcedure,
)

mol = sun.chemistry.Molecule(
    geometry="H 0 0 0\nH 0 0 0.748", basis_set="sto-3g", backend="pyscf"
)
states = FockSpaceState.non_ionic_states(mol, IrrepProvider(mol, "canon"))
print(FockSpaceState.dataframe(SpinSymmetrizationProcedure(mol).symmetrize(states)))
```

**See also:** [canonical H₂](../../../examples/symmetry_adaptation/canonical_H2.py) · [localized H₂](../../../examples/symmetry_adaptation/localized_H2.py) · [point groups](../../../examples/symmetry_adaptation/point_group_representations.py) · [tests](../../../tests/test_symmetry_adaptation.py)
