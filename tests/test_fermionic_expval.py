import random
from datetime import datetime
from unittest import mock

import numpy as np
import pytest
import tequila as tq
from numpy import isclose

import sunrise as sn
import sunrise.expval.minimize as sn_minimize
import sunrise.expval.simulate_circuit as sn_simulate_circuit
from sunrise.expval import (
    INSTALLED_FERMIONIC_BACKENDS,
    SUPPORTED_FERMIONIC_BACKENDS,
    Braket,
)

HAS_TCC = "tcc" in INSTALLED_FERMIONIC_BACKENDS


@pytest.mark.parametrize(
    "geom",
    [
        "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8",
        "H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2",
    ],
)
@pytest.mark.parametrize("backend", INSTALLED_FERMIONIC_BACKENDS)
def test_spa(geom, backend):
    mol = sn.Molecule(
        geometry=geom, basis_set="sto-3g", transformation="reordered-jordan-wigner", backend="pyscf"
    ).use_native_orbitals()
    edges = sn.Molecule(
        geometry=geom, basis_set="sto-3g", nature="hybrid", backend="pyscf"
    ).get_spa_edges()
    U = mol.make_ansatz("SPA", edges=edges, optimize=False)
    circuit = sn.FCircuit.from_edges(edges=edges, n_orb=mol.n_orbitals)
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(), U=U)
    sunval = Braket(molecule=mol, ket=circuit)
    tqE = tq.minimize(expval, silent=True)
    e = sn.minimize(sunval, silent=True, backend=backend)
    sunE = e.energy
    tqwfn = tq.simulate(U, tqE.angles)
    sunwfn = sn.simulate(U, e.variables, backend=backend)
    assert isclose(tqE.energy, sunE)
    assert isclose(abs(tqwfn.inner(sunwfn)), 1, 1.0e-3)


@pytest.mark.parametrize(
    "geom",
    [
        "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8",
        "H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2",
    ],
)
@pytest.mark.parametrize("backend", INSTALLED_FERMIONIC_BACKENDS)
def test_upccsd(geom, backend):
    mol = sn.Molecule(
        geometry=geom, basis_set="sto-3g", transformation="reordered-jordan-wigner"
    )
    U = mol.make_ansatz("UpCCSD")
    fmol = sn.Molecule(geometry=geom, basis_set="sto-3g", nature="fermionic")
    circuit = fmol.make_ansatz("UpCCSD")
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(), U=U)
    sunval = Braket(molecule=mol, ket=circuit)
    tqE = tq.minimize(expval, silent=True)
    snE = sn.minimize(sunval, silent=True, backend=backend)
    assert isclose(tqE.energy, snE.energy)


@pytest.mark.parametrize("backend", INSTALLED_FERMIONIC_BACKENDS)
def test_transition(backend):
    geom = "H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3."
    mol = sn.Molecule(
        geometry=geom, basis_set="sto-3g", transformation="reordered-jordan-wigner"
    ).use_native_orbitals()
    H = mol.make_hamiltonian()
    U1 = mol.make_ansatz("SPA", edges=[(0, 1), (2, 3)])
    U2 = mol.make_ansatz("SPA", edges=[(0, 2), (1, 3)])
    res1 = tq.minimize(tq.ExpectationValue(H=H, U=U1), silent=True)
    res2 = tq.minimize(tq.ExpectationValue(H=H, U=U2), silent=True)
    bra = sn.FCircuit.from_edges([(0, 1), (2, 3)], n_orb=mol.n_orbitals)
    ket = sn.FCircuit.from_edges([(0, 2), (1, 3)], n_orb=mol.n_orbitals)
    ov = tq.BraKet(bra=U1, ket=U2, H=H)
    res1.angles.update(res2.angles)
    tq_ov = tq.simulate(ov, variables=res1.angles)
    sn_ov = sn.simulate(
        Braket(molecule=mol, bra=bra, ket=ket), backend=backend, variables=res1.angles
    )
    assert isclose(tq_ov, sn_ov, atol=1.0e-3)


@pytest.mark.parametrize(
    "geom",
    [
        "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8",
        "H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2",
    ],
)
@pytest.mark.parametrize("backend", INSTALLED_FERMIONIC_BACKENDS)
def test_mapped_variables(geom, backend):
    random.seed(datetime.now().timestamp())
    mol = sn.Molecule(
        geometry=geom, basis_set="sto-3g", transformation="reordered-jordan-wigner", backend="pyscf"
    ).use_native_orbitals()
    edges = sn.Molecule(
        geometry=geom, basis_set="sto-3g", nature="hybrid", backend="pyscf"
    ).get_spa_edges()
    U = mol.make_ansatz("SPA", edges=edges, optimize=False)
    mapa = {d: random.random() * np.pi for d in U.extract_variables()}
    U = U.map_variables(mapa)
    circuit = sn.FCircuit.from_edges(edges=edges, n_orb=mol.n_orbitals)
    circuit = circuit.map_variables(mapa)
    expval = tq.ExpectationValue(H=mol.make_hamiltonian(), U=U)
    sunval = Braket(molecule=mol, ket=circuit)
    tqE = tq.simulate(expval, {})
    sunE = sn.simulate(sunval, {}, backend=backend)
    assert isclose(tqE, sunE)


@pytest.mark.parametrize(
    "geom",
    [
        "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8",
        "H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2",
    ],
)
@pytest.mark.parametrize("backend", INSTALLED_FERMIONIC_BACKENDS)
def test_optimize_orbitals(geom, backend):
    if backend == "fqe":
        pytest.skip("Check https://github.com/quantumlib/OpenFermion-FQE/issues/142")
    snmol = sn.Molecule(geometry=geom, basis_set="sto-3g", nature="f", backend='pyscf')
    snmol, edges = snmol.use_CLPO_orbitals_and_edges()
    tqmol = sn.Molecule(
        geometry=geom, basis_set="sto-3g", transformation="reordered-jordan-wigner", backend="pyscf"
    )
    tqmol = sn.CLPO.generate_CLPO_molecule(tqmol)
    snU = snmol.make_ansatz("SPA", edges=edges)
    tqU = tqmol.make_ansatz("HCB-SPA", edges=edges)
    snopt = sn.optimize_orbitals(
        molecule=snmol, circuit=snU, backend=backend, silent=True
    )
    tqopt = sn.optimize_orbitals(molecule=tqmol, circuit=tqU, use_hcb=True, silent=True)
    assert isclose(snopt.energy, tqopt.energy)


@pytest.mark.parametrize(
    "geom", ["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8"]
)
@pytest.mark.parametrize("backend", INSTALLED_FERMIONIC_BACKENDS)
def test_overlap_minimization(geom, backend):
    if backend == "tequila":
        pytest.skip("Skipping tequila")
    mol = sn.Molecule(
        geometry=geom,
        basis_set="sto-3g",
        transformation="reordered-jordan-wigner",
        units="a",
    ).use_native_orbitals()
    H = mol.make_hamiltonian()
    U1 = mol.make_ansatz("SPA", edges=[(0, 1), (2, 3)])
    U2 = mol.make_ansatz("SPA", edges=[(0, 2), (1, 3)])
    bra = sn.FCircuit.from_edges([(0, 1), (2, 3)], n_orb=mol.n_orbitals)
    ket = sn.FCircuit.from_edges([(0, 2), (1, 3)], n_orb=mol.n_orbitals)
    tqS = tq.minimize(tq.BraKet(bra=U1, ket=U2, H=H), silent=True)
    snS = sn.minimize(
        sn.Braket(molecule=mol, ket=ket, bra=bra), backend=backend, silent=True
    )
    assert isclose(tqS.energy, snS.energy, atol=1.0e-3)


@pytest.mark.parametrize(
    "geom",
    [
        "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8",
        "H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2",
    ],
)
@pytest.mark.parametrize("backend", INSTALLED_FERMIONIC_BACKENDS)
def test_gradient(geom, backend):
    tqmol = sn.Molecule(
        geometry=geom,
        basis_set="sto-3g",
        transformation="reordered-jordan-wigner",
        units="a",
    ).use_native_orbitals()
    snmol = sn.Molecule(
        geometry=geom, basis_set="sto-3g", nature="f"
    ).use_native_orbitals()
    random.seed(datetime.now().timestamp())
    tqU = tqmol.make_ansatz("UpCCSD", hcb_optimization=False)
    snU = snmol.make_ansatz("UpCCSD")
    tqval = tq.ExpectationValue(H=tqmol.make_hamiltonian(), U=tqU)
    snval = sn.Braket(molecule=snmol, ket=snU)
    n = random.sample(range(len(tqval.extract_variables())), 5)
    variables = [tqval.extract_variables()[i] for i in n]
    values = {d: random.random() * np.pi for d in tqval.extract_variables()}
    tqg = [tq.simulate(tq.grad(tqval, v), variables=values) for v in variables]
    sng = [
        sn.simulate(sn.grad(snval, v), variables=values, backend=backend)
        for v in variables
    ]
    assert np.allclose(tqg, sng, atol=1.0e-5)


@pytest.mark.parametrize(
    "geom",
    [
        "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8",
        "H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2",
    ],
)
@pytest.mark.parametrize("backend", INSTALLED_FERMIONIC_BACKENDS)
def test_circuit_simulate(geom, backend):
    tqmol = sn.Molecule(
        geometry=geom,
        basis_set="sto-3g",
        transformation="reordered-jordan-wigner",
        units="a",
    )
    snmol = sn.Molecule(geometry=geom, basis_set="sto-3g", nature="f")
    random.seed(datetime.now().timestamp())
    tqU = tqmol.make_ansatz("UpCCSD", hcb_optimization=False)
    snU = snmol.make_ansatz("UpCCSD")
    variables = {d: random.random() * np.pi for d in tqU.extract_variables()}

    tq_wfn = tq.simulate(tqU, variables=variables)
    sn_wfn = sn.simulate(
        snU, variables=variables, n_orb=snmol.n_orbitals, backend=backend
    )

    assert isclose(tq_wfn.inner(sn_wfn), 1)


@pytest.mark.parametrize(
    "geom",
    [
        "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8",
        "H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2",
    ],
)
@pytest.mark.parametrize(
    "backend", ["statevector", "civector", "civector-large", "pyscf", "tensornetwork"]
)
@pytest.mark.skipif(condition=not HAS_TCC, reason="test specific for tcc")
def test_tcc_backends_grandients(geom, backend):
    import tencirchem

    if not hasattr(tencirchem.utils.misc, "get_null_projector_active_mask"):
        pytest.skip(
            "Main tcc installation detected, this test is only for our custom one."
        )
    mol = sn.Molecule(
        geometry=geom, basis_set="sto-3g", transformation="reordered-jordan-wigner"
    ).use_native_orbitals()
    snmol = sn.Molecule(
        geometry=geom, basis_set="sto-3g", nature="f"
    ).use_native_orbitals()
    H = mol.make_hamiltonian()
    tqO = tq.ExpectationValue(H=H, U=mol.make_ansatz("UpCCSD", hcb_optimization=False))
    snO = sn.ExpectationValue(U=snmol.make_ansatz("UpCCSD"), mol=mol)
    variables = tqO.extract_variables()
    point = {d: random.random() * np.pi for d in variables}
    tqgrad = [np.real(tq.simulate(tq.grad(tqO, v), variables=point)) for v in variables]
    sngrad = [
        np.real(
            sn.simulate(
                sn.grad(snO, v),
                variables=point,
                backend="tcc",
                backend_kwargs={"engine": backend},
            )
        )
        for v in variables
    ]
    assert np.allclose(tqgrad, sngrad, atol=1.0e-6)


@pytest.mark.skipif(not HAS_TCC, reason="tcc not installed")
@pytest.mark.parametrize(
    "kwargs", [{"fbackend": "tcc"}, {"fermionic_backend": "tcc"}, {"backend": "tcc"}]
)
def test_minimize_respects_fermionic_backend(kwargs):
    # all fermionic backends give the same energy, so check the backend handed to compile
    mol = sn.Molecule(
        geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8",
        basis_set="sto-3g",
        nature="f",
        backend="pyscf",
    )
    mol, edges = mol.use_CLPO_orbitals_and_edges()
    U = mol.make_ansatz("SPA", edges=edges)
    with mock.patch.object(
        sn_minimize, "compile", wraps=sn_minimize.compile
    ) as compile_spy:
        sn.minimize(
            Braket(molecule=mol, ket=U, operator="H"), maxiter=2, silent=True, **kwargs
        )
    assert {
        c.kwargs.get("fbackend") or c.kwargs.get("backend")
        for c in compile_spy.call_args_list
    } == {"tcc"}


@pytest.mark.skipif(not HAS_TCC, reason="tcc not installed")
@pytest.mark.parametrize("kwargs", [{"fermionic_backend": "tcc"}, {"backend": "tcc"}])
def test_optimize_orbitals_respects_fermionic_backend(kwargs):
    mol = sn.Molecule(
        geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8",
        basis_set="sto-3g",
        nature="f",
        backend="pyscf",
    )
    mol, edges = mol.use_CLPO_orbitals_and_edges()
    U = mol.make_ansatz("SPA", edges=edges)
    with mock.patch.object(
        sn_minimize, "compile", wraps=sn_minimize.compile
    ) as compile_spy:
        sn.optimize_orbitals(
            molecule=mol,
            circuit=U,
            silent=True,
            pyscf_arguments={"max_cycle_macro": 1},
            **kwargs,
        )
    # compile also takes the fermionic backend through `backend` (used for the RDMs)
    assert {
        c.kwargs.get("fbackend") or c.kwargs.get("backend")
        for c in compile_spy.call_args_list
    } == {"tcc"}


def test_pick_backend_default():
    # without a backend the first installed one in SUPPORTED_FERMIONIC_BACKENDS is chosen, spex first
    expected = next(
        b for b in SUPPORTED_FERMIONIC_BACKENDS if b in INSTALLED_FERMIONIC_BACKENDS
    )
    assert sn.pick_backend() == expected
    assert SUPPORTED_FERMIONIC_BACKENDS[0] == "spex"


@pytest.mark.parametrize("backend", INSTALLED_FERMIONIC_BACKENDS)
def test_pick_backend_verifies(backend):
    assert sn.pick_backend(backend) == backend
    assert sn.pick_backend(backend.upper()) == backend
    if backend != "tequila":
        # only the tequila backend can sample or simulate noise
        with pytest.raises(tq.TequilaException):
            sn.pick_backend(backend, samples=100)


def test_pick_backend_samples_noise_random():
    assert sn.pick_backend(samples=100) == "tequila"
    assert sn.pick_backend(noise="device") == "tequila"
    assert sn.pick_backend("tequila", samples=100) == "tequila"
    assert sn.pick_backend("random") in INSTALLED_FERMIONIC_BACKENDS
    assert sn.pick_backend("random", samples=100) == "tequila"
    with pytest.raises(tq.TequilaException):
        sn.pick_backend("qulacs")


def test_compile_uses_picked_backend():
    mol = sn.Molecule(
        geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.6",
        basis_set="sto-3g",
        nature="f",
        backend="pyscf",
    )
    compiled = sn_minimize.compile(
        Braket(molecule=mol, ket=mol.make_ansatz("UpCCSD"), operator="H")
    )
    assert isinstance(compiled.args[0], INSTALLED_FERMIONIC_BACKENDS[sn.pick_backend()])


def test_simulate_fcircuit_uses_picked_backend():
    picked = sn.pick_backend()
    if picked not in sn_simulate_circuit.INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR:
        pytest.skip(f"{picked} has no fermionic circuit simulator")
    mol = sn.Molecule(
        geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.6",
        basis_set="sto-3g",
        nature="f",
        backend="pyscf",
    )
    U = mol.make_ansatz("UpCCSD")
    variables = {v: 0.1 for v in U.extract_variables()}
    simulator = mock.Mock(
        wraps=sn_simulate_circuit.INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR[picked]
    )
    with mock.patch.dict(
        sn_simulate_circuit.INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR, {picked: simulator}
    ):
        sn.simulate(U, variables)
    simulator.assert_called_once()


@pytest.mark.parametrize(
    "kwargs, fermionic, qubit",
    [
        ({"backend": "spex"}, "spex", "spex"),
        ({"fbackend": "tequila", "backend": "spex"}, "tequila", "spex"),
        ({"fbackend": "tcc", "backend": "spex"}, "tcc", "spex"),
        ({"backend": "tcc"}, "tcc", None),
    ],
)
def test_compile_keeps_spex_for_tequila(kwargs, fermionic, qubit):
    # spex is also a tequila backend: backend='spex' uses it for both, and an explicit fbackend is not overridden
    if fermionic not in INSTALLED_FERMIONIC_BACKENDS:
        pytest.skip(f"{fermionic} not installed")
    mol = sn.Molecule(
        geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.6",
        basis_set="sto-3g",
        nature="f",
        backend="pyscf",
    )
    E = Braket(molecule=mol, ket=mol.make_ansatz("UpCCSD"), operator="H")
    with (
        mock.patch.object(
            sn_minimize, "compile_fexpval", wraps=sn_minimize.compile_fexpval
        ) as fspy,
        mock.patch.object(
            sn_minimize, "tq_compile", wraps=sn_minimize.tq_compile
        ) as qspy,
    ):
        sn_minimize.compile(E, **kwargs)
    assert {c.kwargs["backend"] for c in fspy.call_args_list} == {fermionic}
    assert {c.kwargs["backend"] for c in qspy.call_args_list} == {qubit}
