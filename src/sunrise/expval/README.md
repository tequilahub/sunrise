# expval – expectation values and optimization

Everything needed to evaluate $\langle\psi|H|\psi\rangle$ (and overlaps $\langle\phi|\psi\rangle$) for fermionic circuits, and to optimize them.

- `Braket` / `ExpectationValue` – build an objective from a molecule/Hamiltonian and one or two circuits. Simulation is delegated to a **fermionic backend**:

  | backend | notes |
  |---|---|
  | `spex` | [spex](https://github.com/tequilahub/spex) sparse simulator, installed with sunrise (default) |
  | `tcc` | [TenCirChem](https://github.com/tencent-quantum-lab/TenCirChem) |
  | `fqe` | [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE) |
  | `ffsim` | [ffsim](https://github.com/qiskit-community/ffsim) dense CI vectors, gates and operators must conserve N and S_z |
  | `tequila` | maps to qubits and uses any [tequila](https://github.com/tequilahub/tequila) simulator |

  spex is installed with sunrise; `pip install fqe` and `pip install ffsim` add FQE and ffsim. For TenCirChem-NG
  use our fork, `pip install git+https://github.com/davibinco/TenCirChem-NG.git`: the official installation command
  currently deploys the previous TenCirChem version, and the fork fixes [the parameter convention](https://github.com/tensorcircuit/TenCirChem-NG/pull/4)
  for consistency with the other backends, supports excitations of more than 2 electrons and the phase gates needed
  for the [sunrise/tequila gradients](https://doi.org/10.1039/D0SC06627C).

- `minimize` – optimize the circuit parameters.
- `optimize_orbitals` – orbital optimization around a fixed circuit, a cheap way to boost compact ansätze such as SPA ([Mizukami et al., PRR 2020](https://doi.org/10.1103/PhysRevResearch.2.033421)).

```python
import sunrise as sun
from openfermion import FermionOperator

geometry = "H 0 0 0\nH 0 0 1.5\nH 0 0 3.0\nH 0 0 4.5"
mol = sun.Molecule(geometry=geometry, basis_set="sto-3g", nature="f")
U = mol.make_ansatz("SPA", edges=[(0, 2), (1, 3)])  # starts from Hartree-Fock
HF = mol.prepare_reference()

E = sun.Braket(ket=U, mol=mol)  # <U|H|U>: the Hamiltonian is the default operator
result = sun.minimize(
    E, silent=True, backend="fqe"
)  # or "spex", "tcc", "ffsim", "tequila"
wfn = sun.simulate(U, variables=result.variables, backend="spex")  # final wavefunction

S = sun.Braket(ket=U, bra=HF, operator="I", mol=mol)  # overlap <HF|U>
op = FermionOperator(
    "0^ 1^ 1 0", 0.5
)  # interleaved spin orbitals: 0 = orbital 0 up, 1 = orbital 0 down
T = sun.Braket(ket=U, bra=HF, operator=op, mol=mol)  # transition element <HF|op|U>
print(sun.simulate(S, variables=result.variables))  # 0.978

opt = sun.optimize_orbitals(molecule=mol, circuit=U, silent=True)
print(result.energy, opt.energy)  # -1.8637 -1.8659 (Hartree)
```

**See also:** [tests](../../../tests/test_fermionic_expval.py) · [mixed qubit/fermionic tests](../../../tests/test_mixed_expval.py)
