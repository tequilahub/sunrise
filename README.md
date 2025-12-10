Sunrise is a package for all-around fermionic state preparation and simulation on quantum computers.
It holds no qubits, but only physical information, making it efficient for storage and manipulation.

[Tequila](https://github.com/tequilahub) chemistry extension package.


# Installation
Sunrise is compatible with macOS and Linux operating systems. PySCF is not supported on Windows.
Please, install the package along with all required dependencies:

```bash
conda create -n myenv python=3.10
conda activate myenv

cd project-sunrise
pip install -e .
```

You can install `sunrise` directly with pip over:
```bash
pip install git+https://github.com/tequilahub/sunrise.git
```
Install from devel branch (most recent updates):
```bash
pip install git+https://github.com/tequilahub/sunrise.git@devel
```

## Fermionic Backends
To see the supported fermionic backends:
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
The package's specified installation command is erroneous and deploys the prior version (plain TCC). However, the implementation on sunrise has been built over the existing ng package, waiting to be fixed. In the mean time, we recomend to install it as:
```bash
pip install git+https://github.com/davibinco/TenCirChem-NG.git
```
Note that the link above is not the original repo but our own fork, fixing [this bug](https://github.com/tensorcircuit/TenCirChem-NG/pull/4) until implemented on the code.

# Fermionic Circuit
The FCircuit has been developed in order to preserve abstract fermionic layers. This circuit consists of two main parts, the `initial_state` and the circuit itself. The `initial_state` relies on a tequila `QubitWaveFunction`, whose states are checked to be particle conserving. It can be set through any way of creating a `tq.QubitWaveFunction`, either with a `FCircuit` or `QCircuit` with already mapped variables, or the options available in `QubitWaveFunction.convert_from()`. **It is important to take into account that both fqe and tcc work with initial_states in upthendown, therefore we are keeping here that as mandatory**. In order to create the circuit itself, some gates have been already defined at sunrise.gates. 

A representative example could be:
```python
from sunrise import gates
U = sun.FCircuit()
U.initial_state = '1*|11001100>'
U += gates.UR(0,1,"a") + gates.UR(2,3,"b") 
U += gates.UC(1,2,'c') + gates.UC(0,3,'d')
U += gates.FermionicExcitation([(0,4),(1,5)],'e')
print(U)
```

They can also be created with the Fermionic Molecule:
```python
geom = "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8"
mol = sun.Molecule(geometry=geom,basis_set='sto-3g',nature='f')
U = mol.make_ansatz('UpCCSD')
print(U)
```
Finally, they can be employed in a similar way to tequila `QCircuit` or converted to them:
```python
U = ...
#The expectation value object is explained bellow
E = sun.Braket(U=U,backend='tcc',mol=mol)
res = sun.minimize(E,silent=True)
print(sun.simulate(U,variables=res.variables))
print('Energy ',res.energy)
print('Variables ',res.variables)
opt = sun.optimize_orbitals(molecule=mol,circuit=U,backend='fqe',silent=True)
```

or compiled to qubit and being employed as in regular tequila:
```python
mol = sun.Molecule(geometry=geom,basis_set='sto-3g',nature='tequila')
print(U)
U = U.to_qcircuit(mol)
print(U)
```


# Fermionic Expectation Value
Fermionic Expectation Value, interface with FQE, TCC and TQ. Created in a similar form to tequila BraKet. 

```python
#If not especified, the default operator is the molecular Hamiltonian
E = sun.Braket(molecule=mol,U=U,backend='tcc')
ov = sun.Braket(bra=U1,ket=U2,operator='I',backend='fqe')
#It can alse be employed qustom operators
from openfermion import FermionOperator
op = FermionOperator('0^ 1^ 1 0',0.5)
r = sun.Braket(bra=U1,ket=U2,op)
res = sun.simulate/minimize/...
```


# [Hybrid Molecule](https://doi.org/10.1088/2058-9565/adbdee)
Create your Hybrid Molecule.

Example:
```python
import sunrise as sun
import tequila as tq

molecule  = sun.Molecule(geometry="H 0. 0. 0. \n Li 0. 0. 1.5",basis_set="sto-3g",select="BBFBF",nature='h')
print(molecule.select)
```

Which can be also initialized as:
```python
import sunrise as sun
from sunrise.molecules import HyMolecule
import tequila as tq
molecule  = HyMolecule(geometry="H 0. 0. 0. \n Li 0. 0. 1.5",basis_set="sto-3g",select={2:"F",4:"F"})
print(molecule.select)
```

### Construct your circuit
The SPA circuit (and all the automatically built circuits) are already adapted to your encoding.

```python
Uspa = molecule.make_ansatz("SPA",edges=[(0,1)])
```
However, one can also  build its own circuits:
```python
U = tq.QCircuit() # see more on https://github.com/tequilahub/tequila-tutorials/
# Prepare the reference HF state if any other provided
U += molecule.prepare_reference() 
# Paired 2e excitation from MO 0 to MO 2
U += molecule.UC(0,2,angle=(0,2,"a")) 
# Two One-electron excitation: MO 2_up->4_up + 2_down->4_down TAKE CARE ENCODING
U += molecule.UR(2,4,angle=(2,4,"UR")) 
# Generic excitation
U += molecule.make_excitation_gate(indices=[(0,4),(1,8)],angle=tq.Variable('a')) 
```

### Minimize the Energy of the Circuit Expectation Value

```python
# The molecular Hamiltonian for a given Encoding is automatically built. 
# For custom Hamiltonians please check tutorial above
H = molecule.make_hamiltonian() 
exp = tq.ExpectationValue(H=H,U=U) #Create the Expectation Value Object
# Minimize the energy. You can provide initial variables
mini = tq.minimize(objective=exp,silent=False,initial_values={}) 
print('Minimized Angles:\n',mini.angles)
print('Minimized Energy: ', mini.energy)
```

### Optimize your Orbitals
Molecular orbitals can be optimized taking advantage of this Hybrid Encoding.

```python
result = sun.optimize_orbitals(molecule=molecule,circuit=Uspa,initial_guess='random') 
#Since random guess, may take some time
omol = result.molecule
print("Opt SPA Energy = ",result.energy)
print("Select: ",omol.select)
```

Please, note that the present example can be found in the test file.


# [HCB measurement optimization](https://arxiv.org/abs/2504.03019)
Optimize the measurement procedure of a molecule energy by using the HCB encoding and multiple rotations.

Example using quantum circuit (Scenario II):
```python
import tequila as tq
import sunrise as sun
import numpy as np

# Create the molecule
mol = tq.Molecule(geometry="h 0.0 0.0 0.0\nh 0.0 0.0 1.5\nh 0.0 0.0 3.0\nh 0.0 0.0 4.5", basis_set="sto-3g").use_native_orbitals()
fci = mol.compute_energy("fci")
H = mol.make_hamiltonian()

# Create circuit
U0 = mol.make_ansatz(name="SPA", edges=[(0,1),(2,3)])
UR1 = mol.UR(0,1,angle=np.pi/2) + mol.UR(2,3,angle=np.pi/2) + mol.UR(0,3,angle=-0.2334) + mol.UR(1,2,angle=-0.2334)

UR2 = mol.UR(1,2,angle=np.pi/2) + mol.UR(0,3,angle=np.pi/2)
UR2+= mol.UR(0,1,angle="x") + mol.UR(0,2,angle="y") + mol.UR(1,3,angle="xx") + mol.UR(2,3,angle="yy") + mol.UR(1,2,angle="z") + mol.UR(0,3,angle="zz")
UC2 = mol.UC(1,2,angle="b") + mol.UC(0,3,angle="c")
U = U0 + UR1.dagger() + UR2 + UC2 + UR2.dagger()

variables = {((0, 1), 'D', None): -0.644359150621798, ((2, 3), 'D', None): -0.644359150621816, "x": 0.4322931478168998, "y": 4.980327764918099e-14, "xx": -3.07202211218271e-14, "yy": 0.7167447375727501, "z": -3.982666230146327e-14, "zz": 1.2737831353027637e-13, "c": -0.011081251246998072, "b": 0.49719805420976604}
E = tq.ExpectationValue(H=H, U=U)
full_energy = tq.simulate(E, variables=variables)
print(f"Energy error: {(full_energy-fci)*1000:.2f} mE_h\n")

# Create rotators
graphs = [
    [(0,1),(2,3)],
    [(0,3),(1,2)],
    [(0,2),(1,3)]
]
rotators = []
for graph in graphs:
    UR = tq.QCircuit()
    for edge in graph:
        UR += mol.UR(edge[0], edge[1], angle=np.pi/2)
    rotators.append(UR)

# Apply the measurement protocol
result = sun.measurement.rotate_and_hcb(molecule=mol, circuit=U, variables=variables, rotators=rotators, target=full_energy, silent=False)
print(result) # the list of HCB molecules to measure and the residual element discarded

# Compute the energy
energy = 0
for i,hcb_mol in enumerate(result[0]):
    expval = tq.ExpectationValue(U=U+rotators[i], H=hcb_mol.make_hamiltonian())
    energy += tq.simulate(expval, variables=variables)

print(f"Energy of the accumulated HCB contributions: {energy}")
print(f"Error: {energy-full_energy}")
```

# Hybridization
Automated framework for the generation of ansatz-specific optimized molecular orbitals. Given the Molecule Geometry 
it identifies atomic hybridization states in order to construct an orbital coefficient matrix and generate a edge 
list of electron pairings and bond assignments. Currently only implemented with Sunrise Molcules (nature = 'hybrid' and 'fermionic')
```python
import sunrise as sun
import tequila as tq

geometry = """    O 0.000000 0.000000 0.000000\n H 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000"""

mol = sun.Molecule(geometry=geometry, basis_set='sto-3g',nature='h').use_native_orbitals()
initial_guess = mol.get_spa_guess()
edges = mol.get_spa_edges()
U = mol.make_ansatz(name="HCB-SPA", edges=edges)
opt = sun.optimize_orbitals(molecule=mol, circuit=U, initial_guess=initial_guess.T)
```
You can check the resulting orbitals with the plot_MO function. See bellow

**Currently implementation works only for sto-3g and s,p orbitals**

If further orbitals hybridization / basis set is desired, similar orbitals can be achieved as indicated on [sunrise.orbitals.CLPO](https://janpa.sourceforge.net/).

# plot_MO
Interface with the [PYSCF](https://pyscf.org/)  [cubegen](https://pyscf.org/pyscf_api_docs/pyscf.tools.html#module-pyscf.tools.cubegen) tool. It generates the orbital '.cube' files. They may be visualized with many chemical visualizing programs such as [VESTA](https://jp-minerals.org/vesta/en/) or [Avogadro](https://www.openchemistry.org/projects/avogadro2/).

```python
mol = opt.molecule
sun.plot_MO(molecule=mol,filename="water")
```
The cubefile generation may take some time. Here we provided a tq.Molecule but it also accepts any molecule class with mol.parameters and mol.integral_manager

#  Circuit Visualizer
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

visual_circuit = sun.graphical.GCircuit.from_circuit(U, n_qubits_is_double=True) # Translate tq.QCircuit in renderable Circuit

visual_circuit.export_qpic("from_circuit_example") # Create qpic file
visual_circuit.export_to("from_circuit_example.pdf") # Create pdf file
visual_circuit.export_to("from_circuit_example.png") # Create png file
```
Similarly, the same protocol can be followed for FCircuits.

### Manual way
```python
import tequila as tq
import sunrise as sun
from sunrise import graphical as g

circuit = g.GCircuit([
    # Initial state gate, qubits are halved 
    g.GenericGate(U=tq.gates.X([0,1,2,3]), name="initialstate", n_qubits_is_double=True),

    # Singe excitation, i,j correspond to Spin Orbital index --> ((i,j))
    g.SingleExcitation(i=1,j=7,angle=1),

    # Double excitation, i,j,k,l correspond to Spin Orbital index --> ((i,j),(k,l))
    g.DoubleExcitation(i=0,j=4,k=1,l=7,angle=2),

    # Generic gate in the middle of the circuit, qubits are not halved
    g.GenericGate(U=tq.gates.Y([0, 3]), name="simple", n_qubits_is_double=False),

    # Orbital rotator (double single-excitation), i,j correspond to Molecular Orbital index --> ((2*i,2*j),(2*i+1,2*j+1))
    g.OrbitalRotatorGate(i=0,j=1,angle=3),

    # Pair correlator (paired double excitation), i,j correspond to Molecular Orbital index --> ((2*i,2*j),(2*i+1,2*j+1))
    g.PairCorrelatorGate(i=1,j=3,angle=4)
])

circuit.export_to("test.png") # Create png file
circuit.export_to("test.pdf") # Create pdf file
```

Which can be latter exported to tequila circuit:
```python
U = circuit.construct_circuit()
U += tq.gates.X(1)
print(U)
```
     
