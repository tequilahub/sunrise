import sunrise as sun
import tequila as tq

geometry = "O 0.000000 0.000000 0.000000\n H 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000"

mol = sun.Molecule(geometry=geometry, basis_set='sto-3g',nature='h').use_native_orbitals()
initial_guess = mol.get_spa_guess()
edges = mol.get_spa_edges()
print("Edges:", edges)
U = mol.make_ansatz(name="HCB-SPA", edges=edges)
opt = sun.optimize_orbitals(molecule=mol, circuit=U, initial_guess=initial_guess.T)
sun.plot_MO(molecule=opt.molecule,filename="water")