import numpy as np
import tequila as tq
import pytest
import sunrise as sn
from sunrise.expval import INSTALLED_FERMIONIC_BACKENDS,Braket
from numpy import isclose
import random
from datetime import datetime

HAS_TCC = 'tcc' in INSTALLED_FERMIONIC_BACKENDS

@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_spa(geom,backend):
    mol = sn.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
    edges = sn.Molecule(geometry=geom,basis_set='sto-3g',nature='hybrid').get_spa_edges()
    U = mol.make_ansatz("SPA",edges=edges,optimize=False)
    circuit = sn.FCircuit.from_edges(edges=edges,n_orb=mol.n_orbitals)
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
    sunval = Braket(molecule=mol, ket=circuit)
    tqE = tq.minimize(expval,silent=True)
    e = sn.minimize(sunval, silent=True, backend=backend)
    sunE = e.energy
    tqwfn = tq.simulate(U,tqE.angles)
    sunwfn = sn.simulate(U,e.variables, backend=backend)
    assert isclose(tqE.energy,sunE)
    assert isclose(abs(tqwfn.inner(sunwfn)),1,1.e-3)


@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_upccsd(geom,backend):
    mol = sn.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner')
    U = mol.make_ansatz("UpCCSD")
    fmol = sn.Molecule(geometry=geom, basis_set='sto-3g', nature='fermionic')
    circuit = fmol.make_ansatz("UpCCSD")
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(), U=U)
    sunval = Braket(molecule=mol, ket=circuit)
    tqE = tq.minimize(expval, silent=True)
    snE = sn.minimize(sunval, silent=True, backend=backend)
    assert isclose(tqE.energy,snE.energy)

@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_transition(backend):
    geom = 'H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3.'
    mol = sn.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
    H = mol.make_hamiltonian()
    U1 = mol.make_ansatz("SPA", edges=[(0,1),(2,3)])
    U2 = mol.make_ansatz("SPA", edges=[(0,2),(1,3)])
    res1 = tq.minimize(tq.ExpectationValue(H=H,U=U1), silent=True)
    res2 = tq.minimize(tq.ExpectationValue(H=H,U=U2), silent=True)
    bra = sn.FCircuit.from_edges([(0,1),(2,3)], n_orb=mol.n_orbitals)
    ket = sn.FCircuit.from_edges([(0,2),(1,3)], n_orb=mol.n_orbitals)
    ov = tq.BraKet(bra=U1, ket=U2, H=H)
    res1.angles.update(res2.angles)
    tq_ov = tq.simulate(ov, variables=res1.angles)
    sn_ov = sn.simulate(Braket(molecule=mol, bra=bra, ket=ket), backend=backend, variables=res1.angles)
    assert isclose(tq_ov, sn_ov, atol=1.e-3)

@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_mapped_variables(geom,backend):
    random.seed(datetime.now().timestamp())
    mol = sn.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
    edges = sn.Molecule(geometry=geom,basis_set='sto-3g',nature='hybrid').get_spa_edges()
    U = mol.make_ansatz("SPA",edges=edges,optimize=False)
    mapa = {d:random.random()*np.pi for d in U.extract_variables()}
    U = U.map_variables(mapa)
    circuit = sn.FCircuit.from_edges(edges=edges, n_orb=mol.n_orbitals)
    circuit = circuit.map_variables(mapa)
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(), U=U)
    sunval = Braket(molecule=mol, ket=circuit)
    tqE = tq.simulate(expval,{})
    sunE = sn.simulate(sunval, {}, backend=backend)
    assert isclose(tqE,sunE)


@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize("use_hcb",[True,False])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_optimize_orbitals(geom,backend,use_hcb):
    if backend == "tequila":
        pytest.skip("Tequila backend requires a Qubit-based molecule")
    if backend == "fqe":
        pytest.skip("Check https://github.com/quantumlib/OpenFermion-FQE/issues/142")
    snmol = sn.Molecule(geometry=geom,basis_set='sto-3g',nature='f')
    snmol, edges = snmol.use_CLPO_orbitals_and_edges()
    tqmol = sn.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner')
    # same CLPO starting orbitals on the tequila side, so both optimizations start from the same point
    tqmol = sn.CLPO.generate_CLPO_molecule(tqmol)
    snU = snmol.make_ansatz('SPA',edges=edges)
    tqU = tqmol.make_ansatz('HCB-SPA',edges=edges)
    snopt = sn.optimize_orbitals(molecule=snmol,circuit=snU,backend=backend,silent=True,use_hcb=use_hcb)
    tqopt = sn.optimize_orbitals(molecule=tqmol,circuit=tqU,use_hcb=True,silent=True)
    assert isclose(snopt.energy,tqopt.energy)

#TODO: recursion limit problem on tequila, return when fixed
@pytest.mark.parametrize("geom", ["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_overlap_minimization(geom,backend):
    if backend == "tequila":
        pytest.skip("Skipping tequila")
    mol = tq.Molecule(geometry=geom, basis_set='sto-3g',transformation='reordered-jordan-wigner', units='a').use_native_orbitals()
    H = mol.make_hamiltonian()
    U1 = mol.make_ansatz("SPA", edges=[(0, 1), (2, 3)])
    U2 = mol.make_ansatz("SPA", edges=[(0, 2), (1, 3)])
    bra = sn.FCircuit.from_edges([(0, 1), (2, 3)], n_orb=mol.n_orbitals)
    ket = sn.FCircuit.from_edges([(0, 2), (1, 3)], n_orb=mol.n_orbitals)
    tqS = tq.minimize(tq.BraKet(bra=U1, ket=U2, H=H), silent=True)
    snS = sn.minimize(sn.Braket(molecule=mol, ket=ket, bra=bra), backend=backend,silent=True)
    assert isclose(tqS.energy, snS.energy, atol=1.e-3)

@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_gradient(geom,backend):
    tqmol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner',units='a').use_native_orbitals()
    snmol = sn.Molecule(geometry=geom,basis_set='sto-3g',nature='f').use_native_orbitals()
    random.seed(datetime.now().timestamp())
    tqU = tqmol.make_ansatz('UpCCSD',hcb_optimization=False)
    snU = snmol.make_ansatz('UpCCSD')
    tqval = tq.ExpectationValue(H=tqmol.make_hamiltonian(),U=tqU)
    snval = sn.Braket(molecule=snmol,ket=snU)
    n = random.sample(range(0, len(tqval.extract_variables())), 5)
    variables = [tqval.extract_variables()[i] for i in n]
    values = {d:random.random()*np.pi for d in tqval.extract_variables()}
    tqg = [tq.simulate(tq.grad(tqval,v),variables=values) for v in variables]
    sng = [sn.simulate(sn.grad(snval,v),variables=values, backend=backend) for v in variables]
    assert np.allclose(tqg,sng,atol=1.e-5)

@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_circuit_simulate(geom,backend):
    tqmol = sn.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner',units='a')
    snmol = sn.Molecule(geometry=geom,basis_set='sto-3g',nature='f')
    random.seed(datetime.now().timestamp())
    tqU = tqmol.make_ansatz('UpCCSD',hcb_optimization=False)
    snU = snmol.make_ansatz('UpCCSD')
    variables = {d:random.random()*np.pi for d in tqU.extract_variables()}
    
    tq_wfn = tq.simulate(tqU,variables=variables)
    sn_wfn = sn.simulate(snU,variables=variables,n_orb=snmol.n_orbitals,backend=backend)
    
    assert isclose(tq_wfn.inner(sn_wfn),1)

@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',["statevector", "civector","civector-large","pyscf","tensornetwork"])
@pytest.mark.skipif(condition=not HAS_TCC, reason="test specific for tcc")
def test_tcc_backends_grandients(geom,backend):
    import tencirchem
    if not hasattr(tencirchem.utils.misc, "get_null_projector_active_mask"):
        pytest.skip("Main tcc installation detected, this test is only for our custom one.")
    mol = tq.Molecule(geometry=geom, basis_set="sto-3g", transformation="reordered-jordan-wigner").use_native_orbitals()
    snmol = sn.Molecule(geometry=geom, basis_set="sto-3g",nature='f').use_native_orbitals()
    H = mol.make_hamiltonian()
    tqO = tq.ExpectationValue(H=H,U=mol.make_ansatz('UpCCSD',hcb_optimization=False))
    snO = sn.ExpectationValue(U=snmol.make_ansatz('UpCCSD'),mol=mol)
    variables = tqO.extract_variables()
    point = {d:random.random()*np.pi for d in variables}
    tqgrad = [np.real(tq.simulate(tq.grad(tqO, v), variables=point)) for v in variables]
    sngrad = [np.real(sn.simulate(sn.grad(snO, v), variables=point, backend='tcc', backend_kwargs={"engine":backend})) for v in variables]
    assert np.allclose(tqgrad, sngrad, atol=1.e-6)
