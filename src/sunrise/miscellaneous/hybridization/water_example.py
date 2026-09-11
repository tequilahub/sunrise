import sunrise as sun
import tequila as tq
from sunrise.miscellaneous.hybridization import Graph
geometry = "O 0.000000 0.000000 0.000000\n H 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000"

mol = sun.Molecule(geometry=geometry, basis_set='sto-3g',nature='h').use_native_orbitals()
graph = Graph.parse_xyz(mol.get_xyz())
print(graph)
edges, initial_guess = graph.get_spa_edges(collapse=True,strip_orbitals=True), graph.get_orbital_coefficient_matrix(strip_orbitals=True)
print("Edges:", edges)
U = mol.make_ansatz(name="HCB-SPA", edges=edges)
opt = sun.optimize_orbitals(molecule=mol, circuit=U, initial_guess=initial_guess.T)
sun.plot_MO(molecule=opt.molecule,filename="water")