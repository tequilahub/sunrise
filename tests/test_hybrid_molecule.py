import tequila as tq
import pytest
import numpy
import sunrise as sn
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize("select",["FBFBFFBFBFBFB","BBFFBBFFBBFF"])
@pytest.mark.parametrize("two_qubit",[True,False])
def test_hamiltonian(system,select,two_qubit):
    mol= sn.Molecule(geometry=system,basis_set="sto-6g",select=select,backend='pyscf',two_qubit=two_qubit,nature='hybrid')
    tqmol=tq.Molecule(basis_set="sto-6g",geometry=system,backend='pyscf')
    edges = mol.get_spa_edges()
    U1=tqmol.make_ansatz("SPA",edges=edges)
    H1 = tqmol.make_hamiltonian()
    U = mol.make_ansatz("SPA",edges=edges)
    H = mol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    E1 = tq.ExpectationValue(H=H1, U=U1)
    result = tq.minimize(E,silent=True)
    result1 = tq.minimize(E1,silent=True)
    assert numpy.isclose(result1.energy,result.energy,10**-5)

@pytest.mark.parametrize("two_qubit",[True,False])
@pytest.mark.parametrize("transformation",["Jordan-Wigner","reordered-Jordan-Wigner"])
def test_opt_SPA(two_qubit,transformation):
    mol= sn.Molecule(geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8",basis_set="sto-6g",select="BBFFBBFFBBFF",backend='pyscf',two_qubit=two_qubit,transformation=transformation,nature='hybrid') #could be any select
    tqmol=tq.Molecule(basis_set="sto-6g",geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8",backend='pyscf',transformation=transformation)
    edges = mol.get_spa_edges()
    initial_guess = mol.get_spa_guess().T
    tqopt = tq.quantumchemistry.optimize_orbitals(molecule=tqmol,circuit=tqmol.make_ansatz("HCB-SPA",edges=edges),silent=True,initial_guess=initial_guess,use_hcb=True)
    opt = sn.optimize_orbitals(molecule=mol,circuit=mol.make_ansatz("SPA",edges=edges),silent=True,initial_guess=initial_guess)
    assert numpy.isclose(tqopt.energy,opt.energy,10**-5)
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize("select",["FFFFFFFFFFF","FBFBFBFBFB"])
@pytest.mark.parametrize("excit",[ [(0,4)] , [(2,4),(3,5)], [(2,6),(3,7),(1,5)] , [(2,4),(3,5),(1,7),(0,6)] ])
def test_exc_gate(system,select,excit):
    mol_mix = sn.Molecule(geometry=system, basis_set="sto-6g", select=select,condense=False,backend='pyscf',nature='hybrid')
    mol_jw = tq.Molecule(basis_set="sto-6g", geometry=system,backend='pyscf')

    U_jw = mol_jw.prepare_reference()
    U_mix = mol_mix.prepare_reference()

    U_jw += mol_jw.make_excitation_gate(indices=excit, angle="a")
    U_mix += mol_mix.make_excitation_gate(indices=excit, angle="a")

    H_jw = mol_jw.make_hamiltonian()
    H_mix = mol_mix.make_hamiltonian()

    E_jw = tq.ExpectationValue(H=H_jw, U=U_jw)
    E_mix = tq.ExpectationValue(H=H_mix, U=U_mix)

    result_jw = tq.minimize(E_jw)
    result_mix = tq.minimize(E_mix)

    U_mix += mol_mix.transformation.hcb_to_me(bos=True)
    U_mix.n_qubits=U_jw.n_qubits
    wfn_jw = tq.simulate(U_jw,variables=result_jw.variables)
    wfn_mix = tq.simulate(U_mix,variables=result_mix.variables)
    F = abs(wfn_jw.inner(wfn_mix))

    assert numpy.isclose(F, 1.0, 10 ** -4)
    assert numpy.isclose(result_mix.energy, result_jw.energy, 10 ** -4)
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize("hcb_optimization",[True, False])
@pytest.mark.parametrize("order",[1, 2])
def test_UpCCGD_BOS(system,hcb_optimization,order):
    mol = sn.Molecule(geometry=system,basis_set="sto-6g",select="",condense=False,backend="pyscf",nature='hybrid')
    tqmol = tq.Molecule(geometry=system,basis_set="sto-6g",backend="pyscf")

    H = mol.make_hamiltonian()
    tqH = tqmol.make_hamiltonian()

    U = mol.make_ansatz("UPCCGD",include_reference=True,hcb_optimization=hcb_optimization,order=order)
    tqU = tqmol.make_ansatz("UPCCGD",include_reference=True,hcb_optimization=hcb_optimization,order=order)

    E = tq.ExpectationValue(H=H, U=U)
    tqE = tq.ExpectationValue(H=tqH, U=tqU)
    res = tq.minimize(E, silent=True)
    tqres = tq.minimize(tqE, silent=True)
    wfn =tq.simulate(U + mol.hcb_to_me(bos=True), variables=res.variables)
    tqwfn = tq.simulate(tqU , variables=tqres.variables)
    assert numpy.isclose(res.energy,tqres.energy,atol=1.e-4)
    assert numpy.isclose(abs(tqwfn.inner(wfn)),1.0,atol=1.e-4)
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize("hcb_optimization",[True, False])
def test_UpCCGSD_FER(system,hcb_optimization):
    mol = sn.Molecule(geometry=system, basis_set="sto-6g", select="FFFFFFFFFFFFFFFFFFFFF", condense=False,backend='pyscf',nature='hybrid')
    tqmol = tq.Molecule(geometry=system, basis_set="sto-6g",backend='pyscf')

    H = mol.make_hamiltonian()
    tqH = tqmol.make_hamiltonian()

    U = mol.make_ansatz("UPCCGSD", include_reference=True, hcb_optimization=hcb_optimization)
    tqU = tqmol.make_ansatz("UPCCSGD", include_reference=True, hcb_optimization=hcb_optimization)

    E = tq.ExpectationValue(H=H, U=U)
    tqE = tq.ExpectationValue(H=tqH, U=tqU)
    res = tq.minimize(E, silent=True)
    tqres = tq.minimize(tqE, silent=True)
    wfn = tq.simulate(U , variables=res.variables)
    tqwfn = tq.simulate(tqU, variables=tqres.variables)
    assert numpy.isclose(res.energy, tqres.energy, atol=1.e-4)
    assert numpy.isclose(abs(tqwfn.inner(wfn)), 1.0, atol=1.e-4)
#def test_UCCSD_FER():
#    '''
#    Take care, expensive test
#    '''
#    mol = sn.Molecule(geometry="H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2", basis_set="sto-6g", select="FFFFFFFFFFFFFFFFFFFFF", condense=False,backend='pyscf',nature='hybrid')
#    tqmol = tq.Molecule(geometry="H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2", basis_set="sto-6g",backend='pyscf')
#
#    H = mol.make_hamiltonian()
#    tqH = tqmol.make_hamiltonian()
#
#    U = mol.make_ansatz("UCCSD",add_singles=False)
#    tqU = tqmol.make_ansatz("UCCSD",add_singles=False)
#
#    E = tq.ExpectationValue(H=H, U=U)
#    tqE = tq.ExpectationValue(H=tqH, U=tqU)
#    res = tq.minimize(E, silent=True)
#    tqres = tq.minimize(tqE, silent=True)
#    wfn = tq.simulate(U, variables=res.variables)
#    tqwfn = tq.simulate(tqU, variables=tqres.variables)
#    assert numpy.isclose(res.energy, tqres.energy, atol=1.e-4)
#    assert numpy.isclose(abs(tqwfn.inner(wfn)), 1.0, atol=1.e-4)
