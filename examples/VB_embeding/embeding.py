import tequila as tq
import sunrise as sn
import sys
import pickle
import numpy as np
from sunrise.spafastprototype import decompose
from copy import deepcopy
## Example of protocol presented on http://arxiv.org/abs/2606.26882, using many tools already presented on this package examples.

sys.setrecursionlimit(100000)
geometry = '''
C	0.1153180	0.7340720	0.5637030
C	-0.1153180	-0.7340720	0.5637030
C	-0.1153180	1.5483470	-0.5001190
C	0.1153180	-1.5483470	-0.5001190
H	0.4791790	1.1749410	1.5033220
H	-0.4791790	-1.1749410	1.5033220
H	0.0929010	2.6229830	-0.4467250
H	-0.5293470	1.1571020	-1.4380990
H	-0.0929010	-2.6229830	-0.4467250
H	0.5293470	-1.1571020	-1.4380990
'''
file_name = 'butadiene_spa.data'

def step1():
    '''
    Generating and optimizing the Molecule
    Orbital optimization may take a few minutes
    '''
    mol = sn.Molecule(geometry=geometry, basis_set='sto-3g',nature='h')
    print('HF ',mol.compute_energy('HF')) # -153.0089465151146
    print('CISD ',mol.compute_energy('CISD')) # -153.28133918647
    print('CCSD(T) ',mol.compute_energy('CCSD(T)')) # -153.3266212343813
    mol,edges = sn.CLPO.generate_CLPO_molecule_edges(mol)
    print('Edges ',edges) # [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (20, 21)]
    opt = sn.SPAFP.run_spa(mol=mol, edges=edges,initial_guess=False,grouping=12)
    print('SPA Energy ',opt.energy) # -153.20959140553126
    with open(file_name, 'wb') as file:
        pickle.dump(opt.molecule, file)

def plot():
    objects = []
    with (open(file_name, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    mol:sn.molecules.HyMolecule = objects[0]
    sn.plot_MO(mol)

def step2():
    '''
    Analize the orbitals and clasify them by bond nature
    0,1   C1-C3 sigma/*
    2,3   C1-C3 pi/*
    4,5   C1-C2 sigma/*
    6,7   C1-H1 sigma/*
    8,9   C2-C4 sigma/*
    10,11 C2-C4 pi/*
    12,13 C2-H2 sigma/*
    14,15 C3-H3 sigma/*
    16,17 C3-H4 sigma/*
    18,19 C4-H5 sigma/*
    20,21 C4-H6 sigma/*

    Grouping options:
        pi_list = [2,3,10,11]
        cc_list = [0,1,4,5,8,9]
        ch_list = [6,7,12,13,14,15,16,17,18,19,20,21]
    '''

def step3():
    objects = []
    with (open(file_name, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    mol:sn.molecules.hybrid_base.HybridBase = objects[0]
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (20, 21)]
    pi_list = [2,3,10,11]
    cc_list = [0,1,4,5,8,9]
    ch_list = [6,7,12,13,14,15,16,17,18,19,20,21]

    H = mol.make_hamiltonian()
    U_spa = mol.make_ansatz('SPA',edges=edges)
    # Just for using the variables as initial_values for the SPA+, it should be the same as the opt.energy
    E_spa = decompose(H=H,U=U_spa,grouping=6)
    respa = tq.minimize(E_spa,silent=True, gradient="2-point",method_options={"finite_diff_rel_step":1.e-4})
    print("SPA ",respa.energy) # -153.2095915533836


    mol.update_select(pi_list)
    qpi_list = list(set([mol.transformation.up(i) for i in pi_list]+[mol.transformation.down(i) for i in pi_list])) #I want it ordered
    qcc_list = [mol.transformation.up(i) for i in cc_list]
    qch_list = [mol.transformation.up(i) for i in ch_list]
    print("C-C pi/* qubits ",qpi_list)
    print("C-C sigma/* qubits ",qcc_list)
    print("C-H sigma/* qubits ",qch_list)

    H = mol.make_hamiltonian()
    U_spa = mol.make_ansatz('SPA',edges=edges) # Don't forget to rebuild your circuits/remap your circuits when changing the F/B 
    

    UR = mol.UR(2,3,(tq.Variable('R_C1C3')+0.5)*np.pi) + mol.UR(10,11,(tq.Variable('R_C2C4')+0.5)*np.pi)
    UR1 = mol.UR(3,11,(tq.Variable('R_C1C2')+0.5)*np.pi) + mol.UR(2,10,(tq.Variable('R_C3C4')+0.5)*np.pi)
    UC1 = mol.UC(3,11,tq.Variable('C_C1C2')) + mol.UC(2,10,tq.Variable('C_C3C4'))

    # Note: This step is just to make sure that the UR gates have been properly placed
    if True:
        core = len(mol.integral_manager.orbital_coefficients)-mol.n_orbitals
        r0mol = deepcopy(mol)
        mUR = deepcopy(UR)
        mUR = mUR.map_variables({d:0 for d in mUR.extract_variables()})
        rot = sn.measurement.gates_to_orb_rot(mUR,len(mol.integral_manager.orbital_coefficients),core=core)
        r0mol = r0mol.transform_orbitals(rot.T,ignore_active_space=True)
        sn.plot_MO(r0mol,filename='back_to_native',orbital=pi_list)

        r0mol = deepcopy(mol)
        mUR = deepcopy(UR+UR1)
        mUR = mUR.map_variables({d:0 for d in mUR.extract_variables()})
        rot = sn.measurement.gates_to_orb_rot(mUR,len(mol.integral_manager.orbital_coefficients),core=core)
        r0mol = r0mol.transform_orbitals(rot.T,ignore_active_space=True)
        sn.plot_MO(r0mol,filename='1st_graph',orbital=pi_list)
    
    U_spaplus = U_spa + UR + UR1 + UC1 + UR1.dagger() + UR.dagger()
    E_spaplus = decompose(H=H,U=U_spaplus,grouping=[qpi_list,qcc_list,qch_list])
    respaplus = tq.minimize(E_spaplus,silent=True, gradient="2-point",method_options={"finite_diff_rel_step":1.e-4},initial_values=respa.angles)
    print("SPA+ ",respaplus.energy) # -153.21210112546612

    # NOTE: Aftter the minimal VB circuit, it could be extended in many different ways. Here I just try to extend the C-C sigma description by 
    # an UpCCD layer, but many other options exist.

    Ucc = tq.QCircuit()
    for i in cc_list:
        for j in cc_list:
            if j > i: 
                Ucc += mol.UC(i,j,tq.Variable(f'CC-{i}-{j}'))

    Eext = decompose(H=H,U=U_spaplus + Ucc ,grouping=[qpi_list,qcc_list,qch_list])
    resext = tq.minimize(Eext,silent=True, gradient="2-point",method_options={"finite_diff_rel_step":1.e-4},initial_values=respa.angles)
    print('SPA+ + UCC ',resext.energy)

def orbital_refinement():
    '''
    On this section we follow the procedure from:
        - https://github.com/FabianLangkabel/FrayedEnds/blob/main/examples/minbas.py
        - https://github.com/FabianLangkabel/FrayedEnds/blob/main/examples/spa_from_ao.py
    but combined with our work.
    '''
    try:
        import frayedends as fe
        from frayedends.atomicbasisprojector import AtomicBasisProjector
    except ImportError:
        pass
    world = fe.MadWorld(thresh=1e-6,ndims=3)
    objects = []
    with (open(file_name, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    mol:sn.molecules.hybrid_base.HybridBase = objects[0]
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (20, 21)]
    C = mol.integral_manager.orbital_coefficients.copy()
    bp = AtomicBasisProjector(world, geometry, units="a", aobasis="sto-3g")
    basis = bp.get_orbitals()
    
    integrals=fe.Integrals(world)
    orbitals = integrals.transform(basis, C) #transforms orbitals according to: orbtials[i] = sum[j] basis[j]*C[j,i]
    orbitals = integrals.orthonormalize(orbitals=orbitals)
    core = [orbitals[i.idx_total] for i in mol.integral_manager.orbitals if i.idx is None]
    active = [orbitals[i.idx_total] for i in mol.integral_manager.orbitals if i.idx is not None]
    c,_,_ = mol.get_integrals()
    nuc_repulsion = mol.integral_manager.constant_term
    Vnuc = bp.get_nuclear_potential()
    iteration = 0
    converged = False
    energy = 0
    thres = 1.e-6
    grouping = 12
    max_iter = 20
    print("="*10, " Starting Orbital Refination ","="*10)
    while not converged or iteration < max_iter:
        G = integrals.compute_two_body_integrals(active,ordering='chem')
        FC_int = integrals.compute_frozen_core_interaction(core, active)
        T = integrals.compute_kinetic_integrals(active)
        V = integrals.compute_potential_integrals(active, Vnuc)

        mol = sn.Molecule(geometry=geometry, one_body_integrals=T + V + FC_int, two_body_integrals=G, nuclear_repulsion=c, n_electrons=mol.n_electrons, units='a',nature='h',ordering='chem',frozen_core=False)
        U = mol.make_ansatz(name="SPA", edges=edges)
        H = mol.make_hamiltonian()
        E = sn.SPAFP.decompose(H=H,U=U,grouping=grouping)
        result = tq.minimize(E, silent=True, gradient="2-point",method_options={"finite_diff_rel_step":1.e-4})
        clusters = sn.SPAFP.make_decomposed_clusters(U,grouping)
        rdm1, rdm2 = sn.SPAFP.fast_rdm(U=U,mol=mol,variables=result.variables,clusters=clusters)
        print("iteration {} energy {}".format(iteration, result.energy))

        opti = fe.OrbitalRefinement(world, Vnuc, nuc_repulsion)
        core, active = opti.get_orbitals(
        orbitals=[core, active], rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
            )
        c = opti.get_c()

        for o in range(len(core)):
            core[o].save_to_file(f"cor_SPA_orb{o}.data")
        for o in range(len(active)):
            active[o].save_to_file(f"act_SPA_orb{o}.data")
        with open('ref_'+file_name, 'wb') as file:
            pickle.dump(mol, file)
        if abs(energy-result.energy) < thres:
            converged = True
            print("="*10,f' Converged After {iteration} Iterations ',"="*10,)
            energy = result.energy
            print('Final SPA Energy: ',energy)
        else: 
            energy = result.energy
            iteration += 1
        

def plot_refined():
    try:
        import frayedends as fe
    except ImportError:
        pass
    world = fe.MadWorld(thresh=1e-6,ndims=3)
    core = []
    for i in range(4):
        core.append(fe.SavedFct3D(f"cor_SPA_orb{i}.data"))
    for i in range(4):
        world.cube_plot(f"cor_HF_orb{i}",core[i],geometry,zoom=5)
    active = []
    for i in range(22):
        active.append(fe.SavedFct3D(f"act_SPA_orb{i}.data"))
    for i in range(22):
        world.cube_plot(f"act_SPA_orb{i}",active[i],geometry,zoom=5)

def spa_refined():
    objects = []
    with (open('ref_' + file_name, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    mol:sn.molecules.hybrid_base.HybridBase = objects[0]
    pi_list = [2,3,10,11]
    cc_list = [0,1,4,5,8,9]
    ch_list = [6,7,12,13,14,15,16,17,18,19,20,21]
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (20, 21)]

    H = mol.make_hamiltonian()
    U_spa = mol.make_ansatz('SPA',edges=edges)
    E_spa = decompose(H=H,U=U_spa,grouping=6)
    respa = tq.minimize(E_spa,silent=True, gradient="2-point",method_options={"finite_diff_rel_step":1.e-4})
    print("SPA ",respa.energy)


    mol.update_select([2,3,10,11])
    H = mol.make_hamiltonian()
    U_spa = mol.make_ansatz('SPA',edges=edges) 
    qpi_list = list(set([mol.transformation.up(i) for i in pi_list]+[mol.transformation.down(i) for i in pi_list])) #I want it ordered
    qcc_list = [mol.transformation.up(i) for i in cc_list]
    qch_list = [mol.transformation.up(i) for i in ch_list]

    UR = mol.UR(2,3,(tq.Variable('R_C1C3')+0.5)*np.pi) + mol.UR(10,11,(tq.Variable('R_C2C4')+0.5)*np.pi)
    UR1 = mol.UR(3,11,(tq.Variable('R_C1C2')+0.5)*np.pi) + mol.UR(2,10,(tq.Variable('R_C3C4')+0.5)*np.pi)
    UC1 = mol.UC(3,11,tq.Variable('C_C1C2')) + mol.UC(2,10,tq.Variable('C_C3C4'))
    U_spaplus = U_spa + UR + UR1 + UC1 + UR1.dagger() + UR.dagger()
    E_spaplus = decompose(H=H,U=U_spaplus,grouping=[qpi_list,qcc_list,qch_list])
    respaplus = tq.minimize(E_spaplus,silent=True, gradient="2-point",method_options={"finite_diff_rel_step":1.e-4},initial_values=respa.angles)
    print("SPA+ ",respaplus.energy) 

step1()
plot()
step3()
orbital_refinement()
spa_refined()