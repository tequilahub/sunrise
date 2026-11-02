# ADAPT

Adaptive circuit construction ([ADAPT-VQE](https://doi.org/10.1038/s41467-019-10988-2)): instead of fixing the ansatz up front, the circuit is grown one operator at a time. At each step the operator from a *pool* with the largest energy gradient is appended and the parameters are re-optimized.

**Main objects**
- `MolecularPool`, `PseudoSingletMolecularPool` – operator pools built from molecular excitations (e.g. `indices="UpCCSD"`).
- `Adapt` – the solver; works with qubit and fermionic molecules and any expectation-value backend.
- `ObjectiveFactorySequentialExcitedState` – variant for excited states ([Kottmann et al., Chem. Sci. 2021](https://doi.org/10.1039/D0SC06627C)).

```python
import sunrise as sun
from sunrise.ADAPT import Adapt, MolecularPool

mol = sun.Molecule(geometry="H 0 0 0\nLi 0 0 1", basis_set="sto-3g", nature="fermionic")
pool = MolecularPool(molecule=mol, indices="UpCCSD")
result = Adapt(operator_pool=pool, molecule=mol, backend="spex")()
print(result.energy, result.U)
```

**See also:** [example](../../../examples/ADAPT/adapt.py) · [tests](../../../tests/test_adapt.py)
