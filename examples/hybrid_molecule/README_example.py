import tequila as tq
import sunrise as sun
### INITIALIZE YOUR MOLECULE

molecule  = sun.Molecule(geometry="H 0. 0. 0. \n Li 0. 0. 1.5",basis_set="sto-3g",select="BBFBF",nature='h')
print(molecule.select)
molecule  = sun.Molecule(geometry="H 0. 0. 0. \n Li 0. 0. 1.5",basis_set="sto-3g",select={2:"F",4:"F"},nature='h')
print(molecule.select)
molecule  = sun.Molecule(geometry="H 0. 0. 0. \n Li 0. 0. 1.5",basis_set="sto-3g",select=[2,4],nature='h')
print(molecule.select)
### CREATE YOUR CIRCUITS
Uspa = molecule.make_ansatz("SPA",edges=[(0,1)])
U = tq.QCircuit() # see more on https://github.com/tequilahub/tequila-tutorials/blob/main/BasicUsage.ipynb
U += molecule.prepare_reference() # Prepares the reference HF state if any other provided
U += molecule.UC(0,2,angle=(0,2,"a")) #Paired 2e excitation from MO 0 to MO 2
U += molecule.UR(2,4,angle=(2,4,"UR")) # Two One-electron excitation: MO 2_up->4_up + 2_down->4_down TAKE CARE ENCODING
U += molecule.make_excitation_gate(indices=[(0,4),(1,9)],angle=tq.Variable('a')) #Generic excitation


### ENERGY CIRCUIT MINIMIZATION
H = molecule.make_hamiltonian() # The molecular Hamiltonian for a given Encoding is automatically built. For custom Hamiltonians please check tutorial above
exp = tq.ExpectationValue(H=H,U=U) #Create the Expectation Value Object
mini = tq.minimize(objective=exp,silent=True,initial_values={}) #Then you minimize the energy. You can provide initial variables
print('Minimized Angles:\n',mini.angles)
print('Minimized Energy: ', mini.energy)


### ORBITAL OPTIMIZATION
result = molecule.optimize_orbitals(molecule=molecule,circuit=Uspa,initial_guess='random',silent=True) #Since random guess, may take some time
omol = result.molecule
print("Opt SPA Energy = ",result.energy)
print("Select: ",omol.select)