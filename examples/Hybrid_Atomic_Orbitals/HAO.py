import sunrise as sn

basis = 'sto-3g'
geo = 'N 0. 0. 0. \n N 0. 0. 1.098'
nmol = sn.Molecule(geometry=geo,basis_set=basis,backend='pyscf',units='a')
nmol = sn.CLPO.generate_HAO_molecule(nmol)
# sn.plot_MO(molecule=nmol,filename='HAO')
