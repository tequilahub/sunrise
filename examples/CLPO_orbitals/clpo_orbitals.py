import sunrise as sn

basis = 'sto-3g'
geo = 'N 0. 0. 0. \n N 0. 0. 1.098'
nmol = sn.Molecule(geometry=geo,basis_set=basis,backend='pyscf',units='a')
nmol,edges = sn.CLPO.generate_CLPO_molecule_edges(nmol)
sn.plot_MO(molecule=nmol,filename='N2_CLPO')
print('N2 Edges ',edges)

U = nmol.make_ansatz('HCB-SPA',edges=edges)


#With CLPO orbitals, the SPA initial guess is just the identity matrix
opt = sn.optimize_orbitals(nmol,circuit=U,silent=True,use_hcb=True)
sn.plot_MO(opt.molecule,filename='N2_SPA')


# One can also define the desired graph orbitals
# For instance, consider H4:
geo = 'H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3.'
mol = sn.Molecule(geometry=geo,basis_set=basis,backend='pyscf',units='a')

# At this distance, the automatic graph would be H-H H-H == [(0,1),(2,3)]
mol1,edges = sn.CLPO.generate_CLPO_molecule_edges(mol)
print('Automatic Edges: ',edges)
sn.plot_MO(mol1,'1st_graph')

# However, one may also want:
mol2,edges = sn.CLPO.generate_CLPO_molecule_edges(mol,edges=[(0,3),(1,2)])
print('Custom Edges: ',edges) # Note that the edges are the same but the underlaying orbitals have changed
sn.plot_MO(mol2,'2n_graph') 

# For more complex molecules we recommend first to generate the HAO molecule to see what each node actually means.
geometry = '''
    C 0.00000 0.00000 0.00000
    C 1.48460 0.00000 0.00000
    C -0.76837 0.00000 -1.12008
    C 2.25297 0.00000 -1.12008
    H -0.47297 -0.00038 0.97971
    H 1.95470 -0.00556 0.97953
    H -1.85042 -0.00018 -1.03002
    H -0.35993 -0.00000 -2.12298
    H 3.33365 -0.01193 -1.02751
    H 1.85521 0.00845 -2.12564'''

mol = sn.Molecule(geometry=geometry, basis_set='sto-3g', backend='pyscf', nature='t')
mol3 =  sn.CLPO.generate_HAO_molecule(mol)
sn.plot_MO(mol3,filename='HAO',orbital=[2,6,10,14])
# For this example, we want to build the C-C=C-C CLPO (that's why only plotting these orbitals)
# One could define the complete molecular graph, but no need, it will be autocomplete with the reference wvf
# built from the HF or MP2 state (see sn.CLPO.generate_molden for more details)
mol1,edges = sn.CLPO.generate_CLPO_molecule_edges(mol, edges=[(2,6),(10,14)], rm_files = True, use_active = True)
sn.plot_MO(mol1, filename = 'CLPO', orbital = [4,5,14,15])
print('Edges ',edges)

# Please note that if the keyword rm_files is set to False on generate_CLPO_molecule_edges or generate_HAO_molecule, the .molden files will be stored
# It also be generated a file called 'graph' which contain a bit more insight on the edges/orbitals.
# Also the m2a.ini is kept, which is the input file for Molden2AIM, see their github for more info.