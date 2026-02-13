import sunrise as sn
from numpy import eye
#Step 1: Molecule intialization
n_atoms = 6
geo = "".join([f'H 0. 0. {i}'.format(i) for i in range(n_atoms)])
mol = sn.Molecule(geometry=geo,basis_set='sto-3g',backend='pyscf').use_native_orbitals()
edges = [(2*i,2*i+1) for i in range(mol.n_orbitals//2)]

#Step 2: The SPA fast prototype already contain a orbital optimizer which takes advantage of this decompostion


initial_guess = eye(mol.n_orbitals) # See explanation DOI: 10.22331/q-2023-08-03-1073
for edge in edges:
    initial_guess[edge[0],edge[1]] = 1
    initial_guess[edge[1],edge[0]] = -1

# For further molecules, we recomend the combination with sn.CLPO.generate_CLPO_molecule_edges
# see Sunrise/examples/CLPO_orbitals for an example

grouping = None # Grouping keyword maybe None,number, or list of numbers
#If none, the circuit conectivity is employed. 
#If Number, the circuit conectivity is grouped in blocks of your number +-1
#If list of numbers, you decide it manually

opt = sn.SPAFP.run_spa(mol,edges,initial_guess=initial_guess.T,grouping=grouping)
sn.plot_MO(molecule=opt.mo_coeff,filename='SPA')
