import sunrise as sn

basis = 'sto-3g'
geo = 'N 0. 0. 0. \n N 0. 0. 1.098'
nmol = sn.Molecule(geometry=geo,basis_set=basis,backend='pyscf',units='a')
nmol,edges = sn.CLPO.generate_CLPO_molecule_edges(nmol)
sn.plot_MO(molecule=nmol,filename='CLPO')

U = nmol.make_ansatz('HCB-SPA',edges=edges)


#With CLPO orbitals, the SPA initial guess is just the identity matrix
opt = sn.optimize_orbitals(nmol,circuit=U,silent=True,use_hcb=True)
sn.plot_MO(opt.molecule,filename='SPA')
