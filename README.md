<p align="left">
  <img src="https://raw.githubusercontent.com/tequilahub/sunrise/master/sunrise_logo.png" width="175"/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE) [![PyPI version](https://badge.fury.io/py/project-sunrise.svg)](https://pypi.org/project/project-sunrise/) ![CI](https://github.com/tequilahub/sunrise/actions/workflows/ci_basic.yml/badge.svg)

Sunrise is a package for all-around fermionic state preparation and simulation on quantum computers.
It holds no qubits, but only physical information, making it efficient for storage and manipulation.

[Tequila](https://github.com/tequilahub) chemistry extension package.


# Installation
Sunrise is compatible with macOS and Linux operating systems. PySCF is not supported on Windows.
Please, install the package along with all required dependencies:

In order to create a conda environment use:
```bash
conda create -n tq-sun python=3.12
conda activate tq-sun
```

1. Install the master branch from PyPi as:
```bash
pip install project-sunrise
```

2. Clone the repository and install in developer mode:
```bash
git clone https://github.com/tequilahub/sunrise.git
cd project-sunrise
pip install -e .
```

3. Install `sunrise` directly with pip over:
```bash
pip install git+https://github.com/tequilahub/sunrise.git
```
4. Or install from devel branch (most recent updates):
```bash
pip install git+https://github.com/tequilahub/sunrise.git@devel
```

# Examples
All the examples can be found in the folder [examples](examples/). \
In the following we show the most relevant ones.

## Fermionic Backends
To see the supported fermionic backends use:
```python
import sunrise as sun  # First time might take some seconds
sun.show_supported_modules()
```
### [FQE](https://github.com/quantumlib/OpenFermion-FQE)
Install with:
```bash
pip install fqe
```
### [TenCirChem - Next Generation](https://github.com/tensorcircuit/tensorcircuit-ng) (TCC-NG)
The package's specified installation command is erroneous and deploys the prior version (plain TCC). However, the implementation on sunrise has been built over the existing ng package, waiting to be fixed. In the meantime, we recommend to install it as:
```bash
pip install git+https://github.com/davibinco/TenCirChem-NG.git
```
Note that the link above is not the original repo but our own fork, fixing [this bug](https://github.com/tensorcircuit/TenCirChem-NG/pull/4) until implemented on the code.

## Fermionic Circuit
The FCircuit has been developed in order to preserve abstract fermionic layers. This circuit consists of two main parts, the `initial_state` and the circuit itself. The `initial_state` relies on a tequila `QubitWaveFunction`, whose states are checked to be particle conserving. It can be set through any way of creating a `tq.QubitWaveFunction`, either with a `FCircuit` or `QCircuit` with already mapped variables, or the options available in `QubitWaveFunction.convert_from()`. **It is important to take into account that both FQE and TCC work with `initial_states` in upthendown format, i.e., Reordered Jordan-Wigner, therefore we are keeping it here as mandatory**. In order to create the circuit itself, some gates have been already defined at sunrise.gates.

A representative example could be:
```python
from sunrise import gates
U = sun.FCircuit()
U.initial_state = '1*|11001100>'
U += gates.UR(0,1,"a") + gates.UR(2,3,"b") 
U += gates.UC(1,2,'c') + gates.UC(0,3,'d')
U += gates.FermionicExcitation([(0,4),(1,5)],'e')
```

They can also be created with the Fermionic Molecule:
```python
geom = "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8"
mol = sun.Molecule(geometry=geom, basis_set='sto-3g', nature='f')
U = mol.make_ansatz('UpCCSD')
```
Finally, they can be employed in a similar way to tequila `QCircuit` or converted to them:
```python
U = ...
# The expectation value object is explained below
E = sun.Braket(U=U, backend='tcc', mol=mol)
res = sun.minimize(E, silent=True)
print(sun.simulate(U, variables=res.variables))
print('Energy ', res.energy)
print('Variables ', res.variables)
opt = sun.optimize_orbitals(molecule=mol, circuit=U, backend='fqe', silent=True)
```

or compiled to qubit and being employed as in regular tequila:
```python
mol = sun.Molecule(geometry=geom, basis_set='sto-3g', nature='tequila')
U = U.to_qcircuit(mol)
```


## Fermionic Expectation Value
Fermionic Expectation Value, interface with FQE, TCC and TQ. Created in a similar form to tequila BraKet. 

```python
# If not specified, the default operator is the molecular Hamiltonian
E = sun.Braket(molecule=mol, U=U, backend='tcc')
ov = sun.Braket(bra=U1, ket=U2, operator='I', backend='fqe')
# It can also employ custom operators
from openfermion import FermionOperator
op = FermionOperator('0^ 1^ 1 0', 0.5)
r = sun.Braket(bra=U1, ket=U2, op)
res = sun.simulate / sun.minimize / ...
```

## [Hybrid Molecule](https://doi.org/10.1088/2058-9565/adbdee)
Create a molecule with hybrid encoding.

Example:
```python
import sunrise as sun
import tequila as tq

molecule  = sun.Molecule(geometry="H 0. 0. 0. \n Li 0. 0. 1.5", basis_set="sto-3g", select="BBFBF", nature='h')
print(molecule.select)
```

Which can be also initialized as:
```python
import sunrise as sun
from sunrise.molecules import HyMolecule
import tequila as tq
molecule  = HyMolecule(geometry="H 0. 0. 0. \n Li 0. 0. 1.5", basis_set="sto-3g", select={2:"F",4:"F"})
print(molecule.select)
```

### Construct a quantum circuit
The SPA circuit (and all the tequila built-in circuits) are already adapted to the encoding.

```python
Uspa = molecule.make_ansatz("SPA",edges=[(0,1)])
```
However, one can also build its own circuits:
```python
U = tq.QCircuit() # see more on https://github.com/tequilahub/tequila-tutorials/
U += molecule.prepare_reference() # Prepare the reference HF state if no other is provided
U += molecule.UC(0, 2, angle=(0,2,"a")) # Paired 2e excitation from MO 0 to MO 2
U += molecule.UR(2, 4, angle=(2,4,"UR")) # Two One-electron excitation: MO 2_up->4_up + 2_down->4_down TAKE CARE OF THE ENCODING
U += molecule.make_excitation_gate(indices=[(0,4),(1,8)], angle=tq.Variable('a')) # Generic excitation
```

### Minimize the Energy of the Circuit Expectation Value

```python
# The molecular Hamiltonian for a given encoding is automatically built
# For custom Hamiltonians please check tutorial above for the Fermionic Expectation Value
H = molecule.make_hamiltonian()
exp = tq.ExpectationValue(H=H, U=U) # Create the Expectation Value Object
# Minimize the energy. You can provide initial variables
mini = tq.minimize(objective=exp, silent=False, initial_values={}) 
print('Minimized Angles:\n', mini.angles)
print('Minimized Energy: ', mini.energy)
```

### Optimize your Orbitals
Molecular orbitals can be optimized taking advantage of this Hybrid Encoding.

```python
result = sun.optimize_orbitals(molecule=molecule, circuit=Uspa, initial_guess='random') 
# Since it is a random guess, it may take some time
omol = result.molecule
print("Opt SPA Energy = ", result.energy)
print("Select: ", omol.select)
```

Please, note that the present example can be found in the test file.

## Orbital Hybridization
Automated framework for the generation of ansatz-specific optimized molecular orbitals. Given the Molecule Geometry 
it identifies atomic hybridization states in order to construct an orbital coefficient matrix and generate a edge 
list of electron pairings and bond assignments. Currently only implemented with Sunrise Molcules (nature = 'hybrid' and 'fermionic')
```python
import sunrise as sun
import tequila as tq

geometry = """O 0.000000 0.000000 0.000000\n H 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000"""

mol = sun.Molecule(geometry=geometry, basis_set='sto-3g',nature='h').use_native_orbitals()
initial_guess = mol.get_spa_guess()
edges = mol.get_spa_edges()
U = mol.make_ansatz(name="HCB-SPA", edges=edges)
opt = sun.optimize_orbitals(molecule=mol, circuit=U, initial_guess=initial_guess.T)
```
You can check the resulting orbitals with the plot_MO function. See below

**Currently implementation works only for sto-3g and s,p orbitals**

If further orbitals hybridization / basis set is desired, similar orbitals can be achieved as indicated on [sunrise.orbitals.CLPO](https://janpa.sourceforge.net/).

## Plotting Molecular Orbitals
Interface with the [PYSCF](https://pyscf.org/)  [cubegen](https://pyscf.org/pyscf_api_docs/pyscf.tools.html#module-pyscf.tools.cubegen) tool. It generates the orbital '.cube' files. They may be visualized with many chemical visualizing programs such as [VESTA](https://jp-minerals.org/vesta/en/) or [Avogadro](https://www.openchemistry.org/projects/avogadro2/).

```python
import sunrise as sun
import tequila as tq

geometry = """O 0.000000 0.000000 0.000000\n H 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000"""

mol = sun.Molecule(geometry=geometry, basis_set='sto-3g',nature='h')
sun.plot_MO(molecule=mol,filename="water")
```
The cubefile generation may take some time. Here we provided a tq.Molecule but it also accepts any molecule class with mol.parameters and mol.integral_manager

##  Circuit Visualizer
Improved circuit visualizer which creates the circuit qpic file with improved circuit structures in common chemistry building blocks as the electronic excitation gates. It creates gates in molecular orbitals picture, halving the number of qubits displayed.

### Automatized way
```python
import tequila as tq
import sunrise as sun

mol = tq.Molecule(geometry="H 0. 0. 0. \n H 0. 0. 1.",basis_set="sto-3g")
U = tq.QCircuit()
U += tq.gates.Y(2) # Generic gate
U += mol.make_excitation_gate(indices=[(0,2),(1,3)],angle="a") # Double excitation
U += mol.make_excitation_gate(indices=[(4,6)],angle="b") # Single excitation
U += tq.gates.QubitExcitation(target=[5,7],angle="c") # Qubit excitation
U += tq.gates.Trotterized(generator=mol.make_excitation_generator(indices=[(0,2)]),angle="d") # Trotterized rotation
U += mol.make_excitation_gate(indices=[(4,6)],angle="b")
U += mol.make_excitation_gate(indices=[(5,7)],angle="b") # Paired single excitation
U += mol.UR(0,1,1) # Orbital rotator
U += mol.UC(1,2,2) # Pair correlator

visual_circuit = sun.graphical.GCircuit.from_circuit(U, n_qubits_is_double=False) # Translate tq.QCircuit in renderable Circuit

visual_circuit.export_qpic("from_circuit_example") # Create qpic file
visual_circuit.export_to("from_circuit_example.pdf") # Create pdf file
visual_circuit.export_to("from_circuit_example.png") # Create png file
```
Similarly, the same protocol can be followed for FCircuits.

# Acknowledgments & Citations

This software incorporates components from the following third-party projects. If you use this software to obtain results for publication, you are required to cite these works as follows:

## 1. JANPA package of programs
This product includes components from the **JANPA package of programs** ([http://janpa.sourceforge.net/](http://janpa.sourceforge.net/)) developed by Tymofii Nikolaienko.

**Required Citations:**
* T. Yu. Nikolaienko, L. A. Bulavin; *Localized orbitals for optimal decomposition of molecular properties*, Int. J. Quantum Chem. (2019), Vol.119, page e25798, DOI: [10.1002/qua.25798](https://doi.org/10.1002/qua.25798)
* T.Y.Nikolaienko, L.A.Bulavin, D.M.Hovorun; *JANPA: an open source cross-platform implementation of the Natural Population Analysis on the Java platform*, Comput.Theor.Chem.(2014), V.1050, P.15-22, DOI: [10.1016/j.comptc.2014.10.002](https://doi.org/10.1016/j.comptc.2014.10.002)

## 2. Molden2AIM
This product includes the **Molden2AIM** utility ([https://github.com/zorkzou/Molden2AIM](https://github.com/zorkzou/Molden2AIM)) developed by Zhi-Qiang Zou.

**Suggested Citation:**
* Zhi-Qiang Zou, *Molden2AIM: A program for converting MOLDEN files to WFN/WFX files*, [https://github.com/zorkzou/Molden2AIM](https://github.com/zorkzou/Molden2AIM).
     
Licensing > This project is licensed under the MIT License. It includes third-party components (JANPA and Molden2AIM) with specific attribution and citation requirements. See LICENSE-THIRD-PARTY for details.