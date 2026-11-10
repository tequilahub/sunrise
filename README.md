<p align="left">
  <img src="https://raw.githubusercontent.com/tequilahub/sunrise/master/sunrise_logo.png" width="175"/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE) [![PyPI version](https://badge.fury.io/py/project-sunrise.svg)](https://pypi.org/project/project-sunrise/) ![CI](https://github.com/tequilahub/sunrise/actions/workflows/ci_basic.yml/badge.svg)

Sunrise is a package for all-around fermionic state preparation and simulation on quantum computers, built as a [tequila](https://github.com/tequilahub/tequila) chemistry extension.
Its circuits are made of fermionic gates (orbital rotations, pair excitations, generic excitations): they hold no qubits, only the physical information, which makes them efficient to store and manipulate.
They can be simulated with fermionic backends or compiled to qubit circuits for any tequila simulator.

- **Molecules** of three natures: qubit (tequila circuits), fermionic and hybrid (some orbitals as hard-core bosons), with integrals from PySCF, Psi4 or MADNESS.
- **Fermionic circuits** and ansätze such as SPA, UpCCGSD and UCCSD.
- **Expectation values** and minimization with fermionic backends (spex, FQE, TenCirChem, ffsim) or tequila.
- **Orbitals**: orbital optimization, localized bond orbitals (CLPO, HAO) and automatic SPA edges.
- Orbital correlation measures, symmetry adaptation, optimized measurements and circuit drawing.

# Installation
Sunrise runs on macOS and Linux (PySCF is not supported on Windows). We recommend a conda environment:
```bash
conda create -n tq-sun python=3.12
conda activate tq-sun
```

Install the latest release from PyPI:
```bash
pip install project-sunrise
```

or the latest version from GitHub (`@devel` for the most recent updates):
```bash
pip install git+https://github.com/tequilahub/sunrise.git
```

For development, clone the repository and install it in editable mode:
```bash
git clone https://github.com/tequilahub/sunrise.git
cd sunrise
pip install -e .
```

## Fermionic backends
[spex](https://github.com/tequilahub/spex) is installed with sunrise and used by default. The other backends are optional:

| Backend | Install |
|---|---|
| [FQE](https://github.com/quantumlib/OpenFermion-FQE) | `pip install fqe` |
| [ffsim](https://github.com/qiskit-community/ffsim) | `pip install ffsim` |
| [TenCirChem-NG](https://github.com/tensorcircuit/tensorcircuit-ng) | `pip install git+https://github.com/davibinco/TenCirChem-NG.git` (our fork, see [expval](src/sunrise/expval)) |

`sun.show_available_modules()` lists the installed ones.

# Quick start
```python
import sunrise as sun

geometry = "H 0 0 0\nH 0 0 1.5\nH 0 0 3.0\nH 0 0 4.5"
mol = sun.Molecule(geometry=geometry, basis_set="sto-3g", nature="fermionic")
mol = mol.use_native_orbitals()  # atom-like orbitals, so that the edges are H-H bonds

# Separable Pair Ansatz: one electron pair on each edge
U = mol.make_ansatz("SPA", edges=[(0, 1), (2, 3)])

E = sun.ExpectationValue(U=U, mol=mol)  # <U|H|U>
result = sun.minimize(E, silent=True)
print(result.energy)  # -0.897 with the fixed native orbitals

# optimize the orbitals together with the circuit
opt = sun.optimize_orbitals(molecule=mol, circuit=U, silent=True)
print(opt.energy)  # -1.980 (FCI: -1.996)
```

With `nature="qubit"` the same ansatz is a tequila circuit, to be used with any tequila simulator.

# Modules
Each module in [`src/sunrise`](src/sunrise) has a short README with an introduction, a code snippet and references.

| Module | What it does |
|---|---|
| [molecules](src/sunrise/molecules) | `Molecule` object: qubit, fermionic and hybrid representations |
| [fermionic_operations](src/sunrise/fermionic_operations) | fermionic gates and circuits (`FCircuit`) |
| [expval](src/sunrise/expval) | expectation values, fermionic backends, orbital optimization |
| [ADAPT](src/sunrise/ADAPT) | adaptive circuit construction (ADAPT-VQE) |
| [CLPO](src/sunrise/CLPO) | Chemist's Localized Property-optimized Orbitals (CLPO) and hybrid atomic orbitals (HAO) |
| [graph_selection](src/sunrise/graph_selection) | choosing SPA edges |
| [spafastprototype](src/sunrise/spafastprototype) | splits (HCB) expectation values into smaller pieces to speed up SPA |
| [MCVBT](src/sunrise/MCVBT) | multi-configurational valence bond theory |
| [measurement](src/sunrise/measurement) | optimized measurements with hard-core bosons |
| [orbital_correlation](src/sunrise/orbital_correlation) | orbital entanglement and correlation measures |
| [symmetry_adaptation](src/sunrise/symmetry_adaptation) | spin and point-group adapted states |
| [graphical](src/sunrise/graphical) | circuit drawing |
| [plot_mo](src/sunrise/plot_mo.py) | orbital `.cube` files (PySCF cubegen), to visualize with any chemistry viewer such as [VESTA](https://jp-minerals.org/vesta/en/) or [Avogadro](https://www.openchemistry.org/projects/avogadro2/), see the [example](examples/plot_MO/plot_MO_options.py) |
| [utils](src/sunrise/utils) | internal helpers |

# Examples
The [examples](examples/) folder has scripts for most modules, for instance the [hybrid molecule](examples/hybrid_molecule/README_example.py), [CLPO orbitals](examples/CLPO_orbitals), [ADAPT](examples/ADAPT), the [circuit visualizer](examples/circuit_visualizer) and the [valence bond embedding](examples/VB_embedding/embedding.py).

# License
This project is licensed under the [MIT License](LICENSE). JANPA, used for the CLPO orbitals, comes with [janpa-py](https://github.com/davibinco/janpa-py), which lists its own citation requirements.
