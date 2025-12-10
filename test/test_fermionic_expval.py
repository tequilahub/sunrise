import numpy as np
import tequila as tq
import pytest
import sunrise as sn
from sunrise.expval import INSTALLED_FERMIONIC_BACKENDS,Braket
from numpy import isclose
import random
from datetime import datetime



HAS_TCC = "tcc" in INSTALLED_FERMIONIC_BACKENDS
HAS_FQE = "fqe" in INSTALLED_FERMIONIC_BACKENDS

@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_spa(geom,backend):
    mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
    edges = sn.Molecule(geometry=geom,basis_set='sto-3g',nature='hybrid').get_spa_edges()
    U = mol.make_ansatz("SPA",edges=edges,optimize=False)
    circuit = sn.FCircuit.from_edges(edges=edges,n_orb=mol.n_orbitals)
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
    sunval = Braket(molecule=mol,ket=circuit,backend=backend)
    tqE = tq.minimize(expval,silent=True)
    e = sn.minimize(sunval, silent=True)
    sunE = e.energy
    tqwfn = tq.simulate(U,tqE.angles)
    sunwfn = sn.simulate(U,e.variables)
    assert isclose(tqE.energy,sunE)
    assert isclose(abs(tqwfn.inner(sunwfn)),1,1.e-3)


@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_upccsd(geom,backend):
    mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner')
    U = mol.make_ansatz("UpCCSD")
    fmol = sn.Molecule(geometry=geom,basis_set='sto-3g',nature='fermionic')
    circuit = fmol.make_ansatz("UpCCSD")
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
    sunval = Braket(molecule=mol,circuit=circuit,backend=backend)
    tqE = tq.minimize(expval,silent=True)
    snE = sn.minimize(sunval, silent=True)
    assert isclose(tqE.energy,snE.energy)

@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_transition(backend):
    geom = 'H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3.'
    mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
    H = mol.make_hamiltonian()
    U1 = mol.make_ansatz("SPA",edges=[(0,1),(2,3)])
    U2 = mol.make_ansatz("SPA",edges=[(0,2),(1,3)])
    res1 = tq.minimize(tq.ExpectationValue(H=H,U=U1),silent=True)
    res2 = tq.minimize(tq.ExpectationValue(H=H,U=U2),silent=True)
    bra = sn.FCircuit.from_edges([(0,1),(2,3)],n_orb=mol.n_orbitals)
    ket = sn.FCircuit.from_edges([(0,2),(1,3)],n_orb=mol.n_orbitals)
    rov,iov = tq.BraKet(bra=U1,ket=U2,H=H)
    res1.angles.update(res2.angles)
    tq_ov = tq.simulate(rov,variables=res1.angles) + tq.simulate(iov,variables=res1.angles)
    sn_ov = sn.simulate(Braket(molecule=mol,bra=bra,ket=ket,backend=backend),variables=res1.angles)
    assert isclose(tq_ov,sn_ov,atol=1.e-3)

@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_maped_variables(geom,backend):
    random.seed(datetime.now().timestamp())
    mol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
    edges = sn.Molecule(geometry=geom,basis_set='sto-3g',nature='hybrid').get_spa_edges()
    U = mol.make_ansatz("SPA",edges=edges,optimize=backend!='tequila')
    mapa = {d:random.random()*np.pi for d in U.extract_variables()}
    U = U.map_variables(mapa)
    circuit = sn.FCircuit.from_edges(edges=edges,n_orb=mol.n_orbitals)
    circuit = circuit.map_variables(mapa)
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(),U=U)
    sunval = Braket(molecule=mol,ket=circuit,backend=backend)
    tqE = tq.simulate(expval,{})
    sunE = sn.simulate(sunval,{})
    assert isclose(tqE,sunE)


@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_optimize_orbitals(geom,backend):
    if backend == "tequila":
        pytest.skip("Tequila backend requires a Qubit-based molecule")
    snmol = sn.Molecule(geometry=geom,basis_set='sto-3g',nature='f').use_native_orbitals()
    edges = snmol.get_spa_edges()
    initial_guess = snmol.get_spa_guess().T
    tqmol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
    snU = snmol.make_ansatz('SPA',edges=edges)
    tqU = tqmol.make_ansatz('HCB-SPA',edges=edges)
    snopt = sn.optimize_orbitals(molecule=snmol,circuit=snU,backend=backend,silent=True,initial_guess=initial_guess)
    tqopt = tq.chemistry.optimize_orbitals(molecule=tqmol,circuit=tqU,use_hcb=True,silent=True,initial_guess=initial_guess)
    assert isclose(snopt.energy,tqopt.energy)

#TODO: recursion limit problem on tequila, return when fixed
# @pytest.mark.parametrize("geom", ["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8"])
# @pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
# def test_overlap_minimzation(geom,backend):
#     if backend == "tequila":
#         pytest.skip("Skipping tequila")
#     mol = tq.Molecule(geometry=geom, basis_set='sto-3g',transformation='reordered-jordan-wigner').use_native_orbitals()
#     H = mol.make_hamiltonian()
#     U1 = mol.make_ansatz("SPA", edges=[(0, 1), (2, 3)])
#     U2 = mol.make_ansatz("SPA", edges=[(0, 2), (1, 3)])
#     bra = sn.FCircuit.from_edges([(0, 1), (2, 3)], n_orb=mol.n_orbitals)
#     ket = sn.FCircuit.from_edges([(0, 2), (1, 3)], n_orb=mol.n_orbitals)

#     tqS = tq.minimize(tq.BraKet(bra=U1, ket=U2, H=H)[0], silent=True)
#     snS = sn.minimize(sn.Braket(molecule=mol, ket=ket, bra=bra,backend=backend),silent=True)
#     assert isclose(tqS.energy, snS.energy, atol=1.e-3)

@pytest.mark.parametrize("geom",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize('backend',INSTALLED_FERMIONIC_BACKENDS)
def test_gradient(geom,backend):
    tqmol = tq.Molecule(geometry=geom,basis_set='sto-3g',transformation='reordered-jordan-wigner',units='a').use_native_orbitals()
    snmol = sn.Molecule(geometry=geom,basis_set='sto-3g',nature='f').use_native_orbitals()
    random.seed(datetime.now().timestamp())
    tqU = tqmol.make_ansatz('UpCCSD',hcb_optimization=backend=='fqe')
    snU = snmol.make_ansatz('UpCCSD')
    tqval = tq.ExpectationValue(H=tqmol.make_hamiltonian(),U=tqU)
    snval = sn.Braket(molecule=snmol,ket=snU,backend=backend)
    n = random.sample(range(0, len(tqval.extract_variables())), 5)
    variables = [tqval.extract_variables()[i] for i in n]
    values = {d:random.random()*np.pi for d in tqval.extract_variables()}
    tqg = [tq.simulate(tq.grad(tqval,v),variables=values) for v in variables]
    sng = [sn.simulate(sn.grad(snval,v),variables=values) for v in variables]
    assert np.allclose(tqg,sng)