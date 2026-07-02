import sunrise as sn

# Small Interface with pyscf cubegen module

geo = '''
    H	0.0000000	0.9220090	1.2297650 
    H	0.0000000	-0.9220090	1.2297650
    H	0.0000000	-0.9220090	-1.2297650
    H	0.0000000	0.9220090	-1.2297650
    C	0.0000000	0.0000000	0.6669790
    C	0.0000000	0.0000000	-0.6669790'''
mol = sn.Molecule(geometry=geo,basis_set='sto-3g',backend='pyscf',frozen_core=True)
# NOTE:  Just to show some orbital transformation
mol = sn.CLPO.generate_CLPO_molecule(mol)

# NOTE: By default, in all our orbital transformations, the non valence orbitals are kept frozen unless specified. 
# They are kept on the HF orbitals and always unchanged. Thats why the use_active keyword (default True) is introduced.
# The desired orbitals to be plotted can be specified through the orbital keyword. The indices are expected to be consisten
# with the use_active value, therefore here [0,1,2] is from the active orbitals, see mol.integral_manager.orbital[x].idx/idx_total
sn.plot_MO(molecule=mol,use_active=True,orbital=[0,1,2],filename='hello_world')


geo = 'H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3.'
mol = sn.Molecule(geometry=geo,basis_set='sto-3g',backend='pyscf')


# It can also be plotted the electron density and molecular electrostatic potential
sn.plot_MO(mol,print_orbital=False,density=True,mep=True,filename='HF_rdm1')



# By default, the HF rdm1 is computed, but a custom one can be passed through the rdm1 keyword
# Both of the total orbital size or only the active space are accepted
# In case only active space size provided, the exclude_core 

mol,edges = sn.CLPO.generate_CLPO_molecule_edges(mol)
U = mol.make_ansatz('HCB-SPA',edges)
H = mol.make_hardcore_boson_hamiltonian()
E1 = sn.SPAFP.decompose(H=H,U=U,grouping=12)
res1 = sn.minimize(E1,silent=True,gradient="2-point",method_options={"finite_diff_rel_step":1.e-5})
rdm1,_ = sn.SPAFP.fast_rdm(mol=mol,U=U,clusters = sn.SPAFP.make_decomposed_clusters(U, 12),variables=res1.angles)
sn.plot_MO(molecule=mol,print_orbital=False,density=True,mep=True,rdm1=rdm1,filename='custom_rdm1')