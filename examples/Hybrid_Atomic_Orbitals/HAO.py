import sunrise as sn

geo = '''
    C	0.0000000	0.0000000	0.6669790
    C	0.0000000	0.0000000	-0.6669790
    H	0.0000000	0.9220090	1.2297650 
    H	0.0000000	-0.9220090	1.2297650
    H	0.0000000	-0.9220090	-1.2297650
    H	0.0000000	0.9220090	-1.2297650'''
basis = 'sto-3g'
geo = 'N 0. 0. 0. \n N 0. 0. 1.098'
nmol = sn.Molecule(geometry=geo, basis_set=basis,nature='f').use_HAO_orbitals()
sn.plot_MO(molecule=nmol,filename='hao_direct')
nmol = sn.Molecule(geometry=geo,basis_set=basis,backend='pyscf',units='a',nature='h')
orb = nmol.get_HAO_orbitals_coeff()
nmol.integral_manager.orbital_coefficients=orb.T
sn.plot_MO(molecule=nmol,filename='hao_indirect')
