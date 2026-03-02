import tequila as tq
import sunrise as sun
import pytest
import numpy as np
import openfermion as of
import scipy

# @pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="psi4/pyscf not found")
@pytest.mark.parametrize("backend",tq.chemistry.INSTALLED_QCHEMISTRY_BACKENDS)
def test_hcb_measurement_linearH4_scenario2(backend):
    if backend == 'base':
        pytest.skip("Base")
    # Create the molecule
    mol = tq.Molecule(geometry="h 0.0 0.0 0.0\nh 0.0 0.0 1.5\nh 0.0 0.0 3.0\nh 0.0 0.0 4.5", basis_set="sto-3g", backend=backend).use_native_orbitals()
    H = mol.make_hamiltonian()

    # Create circuit
    U0 = mol.make_ansatz(name="SPA", edges=[(0,1),(2,3)])
    UR1 = mol.UR(0,1,angle=np.pi/2) + mol.UR(2,3,angle=np.pi/2) + mol.UR(0,3,angle=-0.2334) + mol.UR(1,2,angle=-0.2334)
    energy1 = -1.979853695572642
    variables1 = {((0, 1), 'D', None): -0.6621692220931846, ((2, 3), 'D', None): -0.6621692220931825}

    UR2 = mol.UR(1,2,angle=np.pi/2) + mol.UR(0,3,angle=np.pi/2)
    UR2+= mol.UR(0,1,angle="x") + mol.UR(0,2,angle="y") + mol.UR(1,3,angle="xx") + mol.UR(2,3,angle="yy") + mol.UR(1,2,angle="z") + mol.UR(0,3,angle="zz")
    UC2 = mol.UC(1,2,angle="b") + mol.UC(0,3,angle="c")
    U = U0 + UR1.dagger() + UR2 + UC2 + UR2.dagger()
    variables = {((0, 1), 'D', None): -0.644359150621798, ((2, 3), 'D', None): -0.644359150621816, "x": 0.4322931478168998, "y": 4.980327764918099e-14,
                 "xx": -3.07202211218271e-14, "yy": 0.7167447375727501, "z": -3.982666230146327e-14, "zz": 1.2737831353027637e-13, "c": -0.011081251246998072,
                 "b": 0.49719805420976604}
    E = tq.ExpectationValue(H=H, U=U)
    energy = tq.simulate(E, variables=variables)

    # Create rotators
    graphs = [
        [(0,1),(2,3)],
        [(0,3),(1,2)],
        [(0,2),(1,3)]
    ]
    rotators = []
    for graph in graphs:
        UR = tq.QCircuit()
        for edge in graph:
            UR += mol.UR(edge[0], edge[1], angle=np.pi/2)
        rotators.append(UR)

    # Test rotation of H and U simultaneously
    HX = sun.fold_rotators(mol, rotators[0]).make_hamiltonian()
    EX = tq.ExpectationValue(H=HX, U=U+rotators[0])
    assert np.isclose(energy, tq.simulate(EX, variables))

    # Test measurement of H rotated in SPA with SPA circuit
    E_test = tq.ExpectationValue(U=U0+rotators[0].dagger(), H=mol.make_hamiltonian())
    tmol = sun.fold_rotators(mol, rotators[0])
    hcb_mol, res_mol = sun.get_hcb_part(tmol)
    EX = tq.ExpectationValue(U=U0+rotators[0].dagger() + rotators[0], H=hcb_mol.make_hamiltonian())
    assert np.isclose(tq.simulate(EX, variables1), tq.simulate(E_test, variables1))

    # Apply the measurement protocol
    result = sun.rotate_and_hcb(molecule=mol, circuit=U, variables=variables, rotators=rotators, target=energy, silent=True)

    test_energy = 0
    for i,molecule in enumerate(result[0]):
        E = tq.ExpectationValue(H=molecule.make_hamiltonian(), U=U+rotators[i])
        test_energy += tq.simulate(E, variables=variables)
    assert np.isclose(test_energy, energy, 10**-3)

@pytest.mark.parametrize("backend",tq.chemistry.INSTALLED_QCHEMISTRY_BACKENDS)
def test_hcb_measurement_linearH4_scenario1(backend):
    if backend == 'base':
        pytest.skip("Base")
    # Create the molecule
    mol = tq.Molecule(geometry="h 0.0 0.0 0.0\nh 0.0 0.0 1.5\nh 0.0 0.0 3.0\nh 0.0 0.0 4.5", basis_set="sto-3g", backend=backend).use_native_orbitals()
    fci = mol.compute_energy("fci")
    H = mol.make_hamiltonian()

    # Create true wave function
    Hof = H.to_openfermion()
    Hsparse = of.linalg.get_sparse_operator(Hof)
    v,vv = scipy.sparse.linalg.eigsh(Hsparse, sigma=fci)
    wfn = tq.QubitWaveFunction.from_array(vv[:,0])
    energy = wfn.inner(H * wfn).real

    # Create rotators
    graphs = [
        [(0,1),(2,3)],
        [(0,3),(1,2)],
        [(0,2),(1,3)]
    ]
    rotators = []
    for graph in graphs:
        UR = tq.QCircuit()
        for edge in graph:
            UR += mol.UR(edge[0], edge[1], angle=np.pi/2)
        rotators.append(UR)

    # Test rotation of H and U simultaneously
    HX = sun.fold_rotators(mol, rotators[0]).make_hamiltonian()
    EX = tq.ExpectationValue(H=HX, U=rotators[0])
    assert np.isclose(energy, tq.simulate(EX, initial_state=wfn))

    # Apply the measurement protocol
    result = sun.rotate_and_hcb(molecule=mol, rotators=rotators, target=fci, initial_state=wfn, silent=True)

    test_energy = 0
    for i,molecule in enumerate(result[0]):
        E = tq.ExpectationValue(H=molecule.make_hamiltonian(), U=rotators[i])
        test_energy += tq.simulate(E, initial_state=wfn)
    assert np.isclose(test_energy, energy, 10**-3)