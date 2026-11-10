import random
from datetime import datetime
from unittest import mock

import numpy as np
import pytest
import tequila as tq
from numpy import isclose
from openfermion import FermionOperator

import sunrise as sn
import sunrise.expval.minimize as sn_minimize
import sunrise.expval.simulate_circuit as sn_simulate_circuit
from sunrise.expval import (
    INSTALLED_FERMIONIC_BACKENDS,
    SUPPORTED_FERMIONIC_BACKENDS,
    Braket,
)

HAS_TCC = "tcc" in INSTALLED_FERMIONIC_BACKENDS
HAS_SPEX = "spex" in INSTALLED_FERMIONIC_BACKENDS
HAS_FFSIM = "ffsim" in INSTALLED_FERMIONIC_BACKENDS


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
        geometry=geom,
        basis_set="sto-3g",
        transformation="reordered-jordan-wigner",
        backend="pyscf",
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
# TODO: don't mark tequila slow when tequilahub/tequila#479 is in tequila master
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(b, marks=pytest.mark.slow if b == "tequila" else ())
        for b in INSTALLED_FERMIONIC_BACKENDS
    ],
)
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
        geometry=geom,
        basis_set="sto-3g",
        transformation="reordered-jordan-wigner",
        backend="pyscf",
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
# TODO: don't mark tequila slow when tequilahub/tequila#479 is in tequila master
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(b, marks=pytest.mark.slow if b == "tequila" else ())
        for b in INSTALLED_FERMIONIC_BACKENDS
    ],
)
def test_optimize_orbitals(geom, backend):
    snmol = sn.Molecule(geometry=geom, basis_set="sto-3g", nature="f", backend="pyscf")
    snmol, edges = snmol.use_CLPO_orbitals_and_edges()
    tqmol = sn.Molecule(
        geometry=geom,
        basis_set="sto-3g",
        transformation="reordered-jordan-wigner",
        backend="pyscf",
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


@pytest.mark.skipif(not HAS_SPEX, reason="spex not installed")
def test_givens_circuit_signed_permutation():
    # swapping orbitals 0 and 3 needs rotations by pi/2 and a phase of pi on orbital 3
    geom = "H 0.0 0.0 0.0\nH 0.0 0.0 1.5\nH 0.0 0.0 3.0\nH 0.0 0.0 4.5"
    tqmol = sn.Molecule(
        geometry=geom, basis_set="sto-3g", transformation="reordered-jordan-wigner"
    )
    snmol = sn.Molecule(geometry=geom, basis_set="sto-3g", nature="f")
    V = np.eye(4)
    V[[0, 3]] = V[[3, 0]]
    tqU = tqmol.make_ansatz("UpCCSD", hcb_optimization=False)
    snU = snmol.make_ansatz("UpCCSD")
    variables = {d: 0.1 + 0.3 * k for k, d in enumerate(tqU.extract_variables())}

    tq_wfn = tq.simulate(tqU + tqmol.get_givens_circuit(V), variables=variables)
    sn_wfn = sn.simulate(
        snU + snmol.get_givens_circuit(V),
        variables=variables,
        n_orb=snmol.n_orbitals,
        backend="spex",
    )
    qc_wfn = tq.simulate(
        tqU + snmol.get_givens_circuit(V).to_qcircuit(tqmol), variables=variables
    )

    # tequila drops the constant part of the number operator in n_rotation, a global phase
    assert isclose(abs(tq_wfn.inner(sn_wfn)), 1)
    assert isclose(abs(tq_wfn.inner(qc_wfn)), 1)


@pytest.mark.skipif(not HAS_SPEX, reason="spex not installed")
def test_n_rotation_pi():
    mol = sn.Molecule(
        geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.5\nH 0.0 0.0 3.0\nH 0.0 0.0 4.5",
        basis_set="sto-3g",
        nature="f",
    )
    # orbital 0 is doubly and orbital 1 singly occupied, n_rotation(i, pi) gives (-1)^n_i
    state = "|11101000>"
    for i, sign in [(0, 1), (1, -1)]:
        ket = sn.FCircuit(initial_state=state) + mol.n_rotation(i, np.pi)
        ov = sn.simulate(
            Braket(
                molecule=mol,
                bra=sn.FCircuit(initial_state=state),
                ket=ket,
                operator="I",
            ),
            backend="spex",
        )
        assert isclose(ov, sign)


@pytest.mark.skipif(not (HAS_SPEX and HAS_FFSIM), reason="spex or ffsim not installed")
def test_ffsim_matches_spex():
    geom = "H 0.0 0.0 0.0\nH 0.0 0.0 0.9\nH 0.0 0.0 1.8\nH 0.0 0.0 2.7"
    mol = sn.Molecule(geometry=geom, basis_set="sto-3g", nature="f")
    qmol = sn.Molecule(
        geometry=geom, basis_set="sto-3g", transformation="reordered-jordan-wigner"
    )
    ket = mol.make_ansatz("UpCCGSD") + sn.gates.Phase(1, "a") + sn.gates.Phase(6, "b")
    bra = mol.make_ansatz("UpCCD")
    rng = np.random.default_rng(7)
    variables = {
        v: rng.uniform(-1, 1) for v in ket.extract_variables() + bra.extract_variables()
    }
    fop = FermionOperator("0^ 2", 0.3) + FermionOperator("2^ 0", 0.3)
    for operator in ["H", "I", qmol.make_hamiltonian(), fop, 0.5]:
        E = Braket(ket=ket, bra=bra, operator=operator, mol=mol)
        spex = sn.simulate(E, variables, backend="spex")
        ffsim = sn.simulate(E, variables, backend="ffsim")
        assert isclose(spex, ffsim, atol=1.0e-10)
    spex = sn.simulate(ket, variables, backend="spex", n_orb=mol.n_orbitals)
    ffsim = sn.simulate(ket, variables, backend="ffsim", n_orb=mol.n_orbitals)
    assert isclose(spex.inner(ffsim), 1)


@pytest.mark.parametrize("backend", INSTALLED_FERMIONIC_BACKENDS)
def test_hcb_operator(backend):
    # on a seniority-zero state the hardcore-boson Hamiltonian gives the same energy as H
    if backend == "tequila":
        pytest.xfail(
            "tequila's HCB is the one-qubit-per-pair Hamiltonian, it only acts on the alpha qubits"
        )
    geom = "H 0.0 0.0 0.0\nH 0.0 0.0 0.9\nH 0.0 0.0 1.8\nH 0.0 0.0 2.7"
    mol = sn.Molecule(geometry=geom, basis_set="sto-3g", nature="f")
    ket = mol.make_ansatz("UpCCD")
    rng = np.random.default_rng(3)
    variables = {v: rng.uniform(-1, 1) for v in ket.extract_variables()}
    E = {
        op: sn.simulate(
            Braket(ket=ket, operator=op, mol=mol), variables, backend=backend
        )
        for op in ["H", "HCB"]
    }
    assert isclose(E["H"], E["HCB"], atol=1.0e-10)


@pytest.mark.skipif(not HAS_FFSIM, reason="ffsim not installed")
def test_ffsim_spin_flip_raises():
    # ffsim works in one (n_alpha, n_beta) sector: a+_beta a_alpha can not be simulated
    mol = sn.Molecule(
        geometry="H 0.0 0.0 0.0\nH 0.0 0.0 0.9", basis_set="sto-3g", nature="f"
    )
    U = sn.gates.FermionicExcitation(indices=[(0, 3)], variables="a")
    with pytest.raises(tq.TequilaException):
        sn.simulate(Braket(ket=U, mol=mol), {"a": 0.1}, backend="ffsim")


@pytest.mark.skipif(not (HAS_SPEX and HAS_FFSIM), reason="spex or ffsim not installed")
def test_ffsim_rdms():
    # the rdms are brakets of FermionOperators, used by optimize_orbitals
    mol = sn.Molecule(
        geometry="H 0.0 0.0 0.0\nH 0.0 0.0 0.9\nH 0.0 0.0 1.8\nH 0.0 0.0 2.7",
        basis_set="sto-3g",
        nature="f",
    )
    U = mol.make_ansatz("UpCCGSD")
    rng = np.random.default_rng(7)
    variables = {v: rng.uniform(-1, 1) for v in U.extract_variables()}
    for ordering in ["dirac", "mulliken"]:
        spex = mol.compute_rdms(U, variables, backend="spex", ordering=ordering)
        ffsim = mol.compute_rdms(U, variables, backend="ffsim", ordering=ordering)
        assert np.allclose(spex[0], ffsim[0], atol=1.0e-10)
        assert np.allclose(spex[1], ffsim[1], atol=1.0e-10)


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


@pytest.mark.parametrize(
    "backend, module",
    [
        ("spex", "spex_tequila"),
        ("tcc", "tencirchem"),
        ("fqe", "fqe"),
        ("ffsim", "ffsim"),
    ],
)
def test_installed_fermionic_backends(backend, module):
    # the backends are registered in a try/except ImportError, so if the library is installed
    # but the backend fails to import, its tests would just disappear
    pytest.importorskip(module)
    assert backend in INSTALLED_FERMIONIC_BACKENDS


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


def test_map_qubits_keeps_initial_state():
    U = sn.FCircuit(initial_state="0.6|1100> - 0.8|0011>")
    mapped = U.map_qubits({0: 2, 1: 3, 2: 0, 3: 1})
    expected = tq.QubitWaveFunction.from_string("0.6|0011> - 0.8|1100>")
    assert mapped.initial_state.isclose(expected)


def test_to_matrix():
    U = sn.FCircuit.from_edges(edges=[(0, 1)], n_orb=2) + sn.gates.UR(0, 1, "r")
    variables = {v: 0.3 for v in U.extract_variables()}
    M = U.to_matrix(variables=variables)
    assert np.allclose(M.conj().T @ M, np.eye(16))
    # the column of the initial state |1010> is the simulated wavefunction
    wfn = sn.simulate(U, variables=variables, backend="tequila", n_orb=2)
    assert np.allclose(M[:, 0b1010], wfn.to_array())
    # an alpha-only excitation does not act on the last qubits, which are still included
    G = sn.FCircuit() + sn.gates.FermionicExcitation(indices=[(0, 2)], variables="a")
    M = G.to_matrix(variables={"a": 0.7})
    assert M.shape == (16, 16)
    U = sn.FCircuit(gates=G.gates, initial_state="1.0|1001>")
    wfn = sn.simulate(U, {"a": 0.7}, backend="tequila", n_orb=2)
    assert np.allclose(M[:, 0b1001], wfn.to_array())


def test_to_qcircuit_jordan_wigner():
    # to_qcircuit has to switch a plain Jordan-Wigner molecule to up-then-down ordering
    mol = sn.Molecule(
        geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.6", basis_set="sto-3g", nature="q"
    )
    U = sn.FCircuit.from_edges(edges=[(0, 1)], n_orb=2) + sn.gates.UR(0, 1, "r")
    variables = {v: 0.3 for v in U.extract_variables()}
    wfn = tq.simulate(U.to_qcircuit(mol), variables=variables)
    expected = sn.simulate(U, variables=variables, backend="tequila", n_orb=2)
    assert np.allclose(wfn.to_array(), expected.to_array())


def test_controlled_fcircuit():
    geom = "H 0.0 0.0 0.0\nH 0.0 0.0 1.6"
    fmol = sn.Molecule(geometry=geom, basis_set="sto-3g", nature="f")
    qmol = sn.Molecule(geometry=geom, basis_set="sto-3g", nature="q")
    variables = {"c": 0.4, "r": 0.3, "p": 0.2}
    U = fmol.UC(0, 1, angle="c") + fmol.UR(0, 1, angle="r") + sn.gates.Phase(1, "p")
    expected = U.to_qcircuit(qmol).add_controls(4).to_matrix(variables)
    # add_controls returns a controlled copy
    assert np.allclose(
        U.add_controls(4).to_qcircuit(qmol).to_matrix(variables), expected
    )
    assert not U.gates[0].is_controlled()
    # same circuit with control= on every gate
    C = fmol.UC(0, 1, angle="c", control=4) + fmol.UR(0, 1, angle="r", control=4)
    C += sn.gates.Phase(1, "p", control=4)
    assert np.allclose(C.to_qcircuit(qmol).to_matrix(variables), expected)


def test_controlled_fcircuit_fermionic_backends():
    # only the tequila backend goes through the qubit circuit, the others can't simulate controls
    mol = sn.Molecule(
        geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.6", basis_set="sto-3g", nature="f"
    )
    U = sn.FCircuit(
        gates=mol.UR(0, 1, angle="r", control=4).gates, initial_state="1.0|1010>"
    )
    E = Braket(ket=U, mol=mol, operator="H")
    for backend in INSTALLED_FERMIONIC_BACKENDS:
        if backend == "tequila":
            # the control qubit is |0>, so the rotation does nothing
            assert np.isclose(
                sn.simulate(E, variables={"r": 0.3}, backend=backend),
                sn.simulate(E, variables={"r": 0.0}, backend=backend),
            )
            continue
        with pytest.raises(tq.TequilaException):
            sn.simulate(E, variables={"r": 0.3}, backend=backend)
        with pytest.raises(tq.TequilaException):
            sn.simulate(U, variables={"r": 0.3}, backend=backend, n_orb=2)


@pytest.mark.skipif(
    "fqe" not in INSTALLED_FERMIONIC_BACKENDS, reason="fqe not installed"
)
def test_fqe_initial_state_sign():
    # the relative sign of the initial state amplitudes is kept
    G = sn.FCircuit.from_edges(edges=[(0, 1)], n_orb=2) + sn.gates.UR(0, 1, "r")
    variables = {v: 0.3 for v in G.extract_variables()}
    init = "0.6|1010> - 0.8|0101>"
    U = sn.FCircuit(gates=G.gates, initial_state=init)
    wfn = sn.simulate(U, variables=variables, backend="fqe", n_orb=2)
    expected = (
        G.to_matrix(variables) @ tq.QubitWaveFunction.from_string(init).to_array()
    )
    assert np.allclose(wfn.to_array(), expected)


def test_to_qcircuit_initial_state():
    # empty first qubits and negative amplitudes are prepared too
    mol = sn.Molecule(
        geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.6", basis_set="sto-3g", nature="q"
    )
    for init, occupied in [("1.0|0110>", "0110"), ("-1.0|1010>", "1010")]:
        U = sn.FCircuit(initial_state=init)
        wfn = tq.simulate(U.to_qcircuit(mol))
        assert wfn.isclose(tq.QubitWaveFunction.from_string(f"1.0|{occupied}>"))


def test_braket_operator_list():
    # a list of operators gives a QTensor, evaluated like the single brakets
    mol = sn.Molecule(
        geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.6", basis_set="sto-3g", nature="f"
    )
    U = mol.make_ansatz("SPA", edges=[(0, 1)])
    variables = {v: 0.3 for v in U.extract_variables()}
    ops = [
        "H",
        FermionOperator("0^ 0"),
        FermionOperator("2^ 2"),
        FermionOperator("1^ 1"),
    ]
    E = Braket(ket=U, operator=ops, mol=mol)
    assert isinstance(E, tq.QTensor) and E.shape == (4,)
    assert Braket(ket=U, operator=ops, mol=mol, shape=(2, 2)).shape == (2, 2)
    for backend in INSTALLED_FERMIONIC_BACKENDS:
        expected = [
            sn.simulate(Braket(ket=U, operator=op, mol=mol), variables, backend=backend)
            for op in ops
        ]
        assert np.allclose(sn.simulate(E, variables, backend=backend), expected)
