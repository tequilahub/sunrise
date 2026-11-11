import sunrise as sun

basis = "sto-3g"
geo = "N 0. 0. 0. \n N 0. 0. 1.098"
nmol = sun.Molecule(geometry=geo, basis_set=basis, backend="pyscf", units="a")
nmol = sun.CLPO.generate_HAO_molecule(nmol)
# sun.plot_MO(molecule=nmol,filename='HAO')
