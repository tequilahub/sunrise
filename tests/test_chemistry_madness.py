import os
import subprocess
import sys

import numpy
import pytest
import tequila as tq

import sunrise as sun
from sunrise.molecules.qubit_base import INSTALLED_QCHEMISTRY_BACKENDS

has_pyscf = "pyscf" in INSTALLED_QCHEMISTRY_BACKENDS
has_frayedends = "frayedends" in INSTALLED_QCHEMISTRY_BACKENDS

# madness runs only in the frayedends worker process, check once that it starts (e.g. MRA_DATA_DIR pointing to a missing folder breaks it)
madness_error = None
if has_frayedends:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import frayedends as fe; world = fe.MadWorld(ndims=3); del world",
        ],
        capture_output=True,
        text=True,
        check=False,  # the error is reported by test_madness_runs
    )
    if proc.returncode != 0:
        madness_error = proc.stderr[-2000:]
has_madness = has_frayedends and madness_error is None


def test_madness_runs():
    # frayedends installed but madness does not start: fail here instead of skipping all madness tests silently
    if has_frayedends and madness_error is not None:
        raise tq.TequilaException(
            f"frayedends is installed but madness does not start:\n{madness_error}"
        )


def test_madness_he_data():
    geomstring = "He 0.0 0.0 0.0"
    molecule = sun.Molecule(
        name="data/madness/he", geometry=geomstring, units="angstrom", n_pno="read"
    )
    H = molecule.make_hamiltonian()
    UHF = molecule.prepare_reference()
    EHF = tq.simulate(tq.ExpectationValue(H=H, U=UHF))
    assert numpy.isclose(-2.861522e00, EHF, atol=1.0e-5)
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    print(result.energy)
    assert numpy.isclose(-2.87761809, result.energy, atol=1.0e-5)


@pytest.mark.skipif(not has_madness, reason="madness was not found")
def test_madness_frozen_core():
    molecule = sun.chemistry.Molecule(
        geometry="Li 0.0 0.0 0.0\nLi 0.0 0.0 1.6", units="angstrom"
    )
    assert molecule.n_orbitals == 2
    assert molecule.parameters.get_number_of_core_electrons() == 4


@pytest.mark.skipif(not has_madness, reason="madness was not found")
def test_madness_full_he():
    # relies on madness being compiled and MAD_ROOT_DIR exported
    # or pno_integrals in the path
    geomstring = "He 0.0 0.0 0.0"
    molecule = sun.Molecule(geometry=geomstring, n_pno=1, units="angstrom")
    H = molecule.make_hamiltonian()
    UHF = molecule.prepare_reference()
    EHF = tq.simulate(tq.ExpectationValue(H=H, U=UHF))
    assert numpy.isclose(-2.8616792, EHF, atol=1.0e-5)  # numerical HF limit -2.8616800
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    assert numpy.isclose(-2.8777604, result.energy, atol=1.0e-5)


@pytest.mark.skipif(not has_madness, reason="madness was not found")
def test_madness_data_io():
    sun.Molecule(geometry="he 0.0 0.0 0.0", units="angstrom")
    sun.Molecule(geometry="he 0.0 0.0 0.0", n_pno="read", units="angstrom")

    sun.Molecule(geometry="he 0.0 0.0 0.0", datadir="1/2/3", units="angstrom")
    sun.Molecule(
        geometry="he 0.0 0.0 0.0", datadir="1/2/3", n_pno="read", units="angstrom"
    )

    sun.Molecule(
        geometry="he 0.0 0.0 0.0", datadir="1/2/3", name="asd", units="angstrom"
    )
    sun.Molecule(
        geometry="he 0.0 0.0 0.0",
        datadir="1/2/3",
        name="asd",
        n_pno="read",
        units="angstrom",
    )


@pytest.mark.skipif(not has_madness, reason="madness was not found")
def test_madness_full_li_plus():
    # relies on madness being compiled and MAD_ROOT_DIR exported
    # or pno_integrals in the path
    geomstring = "Li 0.0 0.0 0.0"
    molecule = sun.Molecule(
        name="li+",
        geometry=geomstring,
        units="angstrom",
        n_pno=1,
        charge=1,
        frozen_core=False,
    )  # need to deactivate frozen_core, otherwise there is no active orbital
    H = molecule.make_hamiltonian()
    UHF = molecule.prepare_reference()
    EHF = tq.simulate(tq.ExpectationValue(H=H, U=UHF))
    assert numpy.isclose(-7.2364144, EHF, atol=1.0e-5)  # numerical HF limit -7.2364152
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    assert numpy.isclose(-7.2513397, result.energy, atol=1.0e-5)


@pytest.mark.skipif(not has_madness, reason="madness was not found")
def test_madness_full_be():
    # relies on madness being compiled and MAD_ROOT_DIR exported
    # or pno_integrals in the path
    geomstring = "Be 0.0 0.0 0.0"
    molecule = sun.chemistry.Molecule(
        name="be", geometry=geomstring, units="angstrom", n_pno=3, frozen_core=True
    )
    H = molecule.make_hamiltonian()
    UHF = molecule.prepare_reference()
    EHF = tq.simulate(tq.ExpectationValue(H=H, U=UHF))
    assert numpy.isclose(
        -14.5730220, EHF, atol=1.0e-5
    )  # numerical HF limit -14.5730232
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    assert numpy.isclose(-14.614662051580321, result.energy, atol=1.0e-3)


@pytest.mark.parametrize(
    "trafo",
    [
        "BravyiKitaev",
        "JordanWigner",
        "BravyiKitaevTree",
        "ReorderedJordanWigner",
        "ReorderedBravyiKitaev",
    ],
)
@pytest.mark.skipif(
    not has_madness and not os.path.isfile("balanced_be_gtensor.npy"),
    reason="madness was not found",
)
def test_madness_upccgsd(trafo):
    n_pno = 2
    if os.path.isfile("balanced_be_gtensor.npy"): # Idea will fix the tests so far but waiting to improved  behaviour on frayed
        n_pno = "read"
    mol = sun.chemistry.Molecule(
        name="balanced_be",
        frozen_core=False,
        geometry="Be 0.0 0.0 0.0",
        units="angstrom",
        n_pno=n_pno,
        pno={"diagonal": True, "maxrank": 1},
        transformation=trafo,
    )

    H = mol.make_hardcore_boson_hamiltonian()
    oigawert = numpy.linalg.eigvalsh(H.to_matrix())[0]
    U = mol.make_upccgsd_ansatz(name="HCB-UpCCGD", direct_compiling=True)
    E = tq.ExpectationValue(H=H, U=U)
    assert len(E.extract_variables()) == 6
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, oigawert, atol=1.0e-3)

    U = mol.make_upccgsd_ansatz(name="HCB-PNO-UpCCD")
    E = tq.ExpectationValue(H=H, U=U)
    assert len(E.extract_variables()) == 2
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, oigawert, atol=1.0e-3)

    H = mol.make_hamiltonian()
    numpy.linalg.eigvalsh(H.to_matrix())[0]
    U = mol.make_upccgsd_ansatz(name="SPA-D")
    E = tq.ExpectationValue(H=H, U=U)
    assert len(E.extract_variables()) == 2
    variables = result.variables
    energy = tq.simulate(E, variables)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, oigawert, atol=1.0e-3)
    assert numpy.isclose(energy, oigawert, atol=1.0e-3)
    e3 = mol.compute_energy(method="SPA")
    assert numpy.isclose(result.energy, e3)
    e3 = mol.compute_energy(method="HCB-SPA")
    assert numpy.isclose(result.energy, e3)
    e3 = mol.compute_energy(method="SPA-UpCCD")
    assert numpy.isclose(result.energy, e3)

    U = mol.make_upccgsd_ansatz(name="SPA-UpCCSD")
    E = tq.ExpectationValue(H=H, U=U)
    assert len(E.extract_variables()) == 4
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, -14.60266198, atol=1.0e-3)

    U = mol.make_upccgsd_ansatz(
        name="SPA-UpCCGSD"
    )  # in this case no difference to SPA-UpCCSD
    E = tq.ExpectationValue(H=H, U=U)
    assert len(E.extract_variables()) == 4
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, -14.60266198, atol=1.0e-3)

    U = mol.make_upccgsd_ansatz(name="UpCCSD")
    E = tq.ExpectationValue(H=H, U=U)
    print(U)
    assert len(E.extract_variables()) == 8
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, oigawert, atol=1.0e-3)

    U = mol.make_upccgsd_ansatz(name="UpCCGSD")
    E = tq.ExpectationValue(H=H, U=U)
    assert len(E.extract_variables()) == 12
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, oigawert, atol=1.0e-3)

    U = mol.make_upccgsd_ansatz(name="UpCCGSD", direct_compiling=False)
    E = tq.ExpectationValue(H=H, U=U)
    assert len(E.extract_variables()) == 12
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, oigawert, atol=1.0e-3)


@pytest.fixture(scope="module")
def madness_be_dir(tmp_path_factory):
    # one madness calculation, the molecules of the natures read its files
    if not has_madness:
        pytest.skip("madness was not found")
    path = tmp_path_factory.mktemp("madness_be")
    cwd = os.getcwd()
    os.chdir(path)
    try:
        sun.Molecule(**MADNESS_ANSATZ_CASES["be"][0], n_pno=2)
    finally:
        os.chdir(cwd)
    return path


# molecule keyword arguments, number of orbitals and number of variables of the ansatz names
MADNESS_ANSATZ_CASES = {
    "he": (
        {"name": "data/madness/he", "geometry": "He 0.0 0.0 0.0", "units": "angstrom"},
        2,
        {"SPA-D": 1, "SPA-UpCCSD": 2, "UpCCGD": 1, "UpCCGSD": 2},
    ),
    "be": (
        {
            "name": "be",
            "geometry": "Be 0.0 0.0 0.0",
            "units": "angstrom",
            "frozen_core": False,
            "pno": {"diagonal": True, "maxrank": 1},
        },
        4,
        {"SPA-D": 2, "SPA-UpCCSD": 4, "UpCCGD": 6, "UpCCGSD": 12},
    ),
}


def _ansatz_energy(mol, U):
    if isinstance(U, tq.QCircuit):
        E = tq.ExpectationValue(H=mol.make_hamiltonian(), U=U)
        return tq.minimize(E, silent=True).energy
    # fermionic circuit
    return sun.minimize(sun.Braket(molecule=mol, ket=U), silent=True).energy


@pytest.mark.parametrize("case", ["he", "be"])
@pytest.mark.parametrize(
    "nature,select",
    [("qubit", None), ("fermionic", None), ("hybrid", "F"), ("hybrid", "B")],
)
def test_madness_upccgsd_natures(case, nature, select, request, monkeypatch):
    # the PNO restricted ansatzes give the same circuits and energies in all natures
    if case == "be":
        monkeypatch.chdir(request.getfixturevalue("madness_be_dir"))
    kwargs, n_orbitals, n_variables = MADNESS_ANSATZ_CASES[case]
    kwargs = {**kwargs, "n_pno": "read"}
    reference = sun.Molecule(**kwargs)
    if select is not None:
        kwargs["select"] = select * n_orbitals
    mol = sun.Molecule(**kwargs, nature=nature)

    # the bosonic orbitals only hold electron pairs, no singles
    # seniority zero ansatzes have the same energy with the hardcore boson Hamiltonian
    names = [x for x in n_variables if select != "B" or "S" not in x.split("-")[-1]]
    energies = {}
    for name in names:
        U = mol.make_upccgsd_ansatz(name=name)
        assert len(U.extract_variables()) == n_variables[name], name
        energies[name] = _ansatz_energy(mol, U)
        reference_energy = _ansatz_energy(
            reference, reference.make_upccgsd_ansatz(name=name)
        )
        assert numpy.isclose(energies[name], reference_energy, atol=1.0e-5), name
    if case == "he" and "UpCCGSD" in energies:
        assert numpy.isclose(energies["UpCCGSD"], -2.87761809, atol=1.0e-5)

    # the excitations within the PNO pairs are the SPA
    U = mol.make_pno_upccgsd_ansatz(name="UpCCD")
    assert len(U.extract_variables()) == n_variables["SPA-D"]
    assert numpy.isclose(_ansatz_energy(mol, U), energies["SPA-D"], atol=1.0e-5)
    if select != "B":
        U = mol.make_pno_upccgsd_ansatz(name="UpCCSD")
        assert len(U.extract_variables()) == n_variables["SPA-UpCCSD"]
        assert numpy.isclose(
            _ansatz_energy(mol, U), energies["SPA-UpCCSD"], atol=1.0e-5
        )


@pytest.mark.skipif(not has_madness, reason="madness was not found")
def test_madness_save_load(tmp_path, monkeypatch):
    # the PNO pair structure is saved with the molecule, it is loaded without madness files
    monkeypatch.chdir(tmp_path)
    geometry = "Be 0.0 0.0 0.0"
    molecules = {
        "qubit": sun.Molecule(
            geometry=geometry,
            n_pno=3,
            transformation="ReorderedJordanWigner",
            point_group="c2v",
        )
    }
    # the other natures read the files of the madness calculation
    molecules["fermionic"] = sun.Molecule(
        geometry=geometry, n_pno="read", point_group="c2v", nature="fermionic"
    )
    molecules["hybrid"] = sun.Molecule(
        geometry=geometry,
        n_pno="read",
        point_group="c2v",
        select="FBBF",
        nature="hybrid",
    )
    for nature, mol in molecules.items():
        mol.save(str(tmp_path / f"be_{nature}.h5"))
    (tmp_path / "other").mkdir()
    monkeypatch.chdir(tmp_path / "other")

    for nature, mol in molecules.items():
        loaded = sun.load_molecule(str(tmp_path / f"be_{nature}.h5"))
        assert type(loaded) is type(mol)
        assert [(o.pair, o.occ, o.idx) for o in loaded.integral_manager.orbitals] == [
            (o.pair, o.occ, o.idx) for o in mol.integral_manager.orbitals
        ]
        assert loaded.point_group.schoenflies_label == "C2v"
        assert loaded.make_hamiltonian() == mol.make_hamiltonian()
        # the madness methods use the pair structure
        if nature == "qubit":
            assert loaded.make_spa_ansatz() == mol.make_spa_ansatz()
            assert loaded.make_upccgsd_ansatz("UpCCGSD") == mol.make_upccgsd_ansatz(
                "UpCCGSD"
            )
            assert loaded.local_qubit_map() == mol.local_qubit_map()
        else:
            assert loaded.get_spa_edges() == mol.get_spa_edges()
        if nature == "hybrid":
            assert loaded.select == mol.select


@pytest.mark.skipif(not has_frayedends, reason="frayedends not installed")
@pytest.mark.skipif(not has_pyscf, reason="PySCF not installed")
def test_frayed_save_load(tmp_path, monkeypatch):
    # the MRA orbitals (SavedFct bytes) and their world parameters are saved with the molecule
    # built from integrals, so no madness worker is needed
    geometry = "Li 0.0 0.0 0.0\nH 0.0 0.0 1.6"
    c, h, g = sun.Molecule(
        geometry=geometry, basis_set="sto-3g", backend="pyscf", frozen_core=False
    ).get_integrals()
    mra_orbitals = [bytes([i, 0, 255]) * (i + 1) for i in range(h.shape[0])]
    mra_parameters = {"L": 30.0, "k": 7, "thresh": 1.0e-5}
    mol = sun.Molecule(
        geometry=geometry,
        basis_set="sto-3g",
        backend="frayedends",
        one_body_integrals=h,
        two_body_integrals=g,
        nuclear_repulsion=c,
        active_orbitals=[1, 2, 3, 4, 5],
        transformation="ReorderedJordanWigner",
        mra_orbitals=mra_orbitals,
        mra_parameters=mra_parameters,
    )
    mol.save(str(tmp_path / "lih.h5"))
    (tmp_path / "other").mkdir()
    monkeypatch.chdir(tmp_path / "other")
    loaded = sun.load_molecule(str(tmp_path / "lih.h5"))

    assert type(loaded) is type(mol)
    assert loaded.mra_orbitals == mra_orbitals
    assert loaded.mra_parameters == mra_parameters
    assert [o.idx_total for o in loaded.integral_manager.active_orbitals] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert loaded.make_hamiltonian() == mol.make_hamiltonian()


@pytest.mark.skipif(not has_frayedends, reason="frayedends not installed")
@pytest.mark.skipif(not has_pyscf, reason="PySCF not installed")
def test_frayed_pno_read_and_save(tmp_path, monkeypatch):
    # PNO route without madness: the worker result is faked (HF orbitals 0, 1 and two PNOs of the pair (1, 1))
    import json
    import tempfile

    from sunrise.molecules.qubit_base import _frayed_worker

    geometry = "Li 0.0 0.0 0.0\nH 0.0 0.0 1.6"
    c, h, g = sun.Molecule(
        geometry=geometry, basis_set="sto-3g", backend="pyscf", frozen_core=False
    ).get_integrals(ordering="mulliken")
    requests = []

    def fake_run_frayed(request):
        requests.append(request)
        workdir = tempfile.mkdtemp(dir=tmp_path)
        for i in range(4):
            with open(os.path.join(workdir, f"orbital_{i}"), "wb") as f:
                f.write(bytes([i]) * 5)
        with open(os.path.join(workdir, "mra_parameters.json"), "w") as f:
            json.dump({"L": 50.0, "k": 7}, f)
        return {
            "h": h[:4, :4],
            "g": g.elems[:4, :4, :4, :4],
            "nuclear_repulsion": numpy.float64(c),
            "occ": numpy.array([2.0, 2.0, 0.1, 0.05]),
            "pair": numpy.array([(0, 0), (1, 1), (1, 1), (1, 1)]),
            "workdir": workdir,
        }

    monkeypatch.setattr(_frayed_worker, "run_frayed", fake_run_frayed)
    monkeypatch.chdir(tmp_path)
    mol = sun.Molecule(
        geometry=geometry, backend="frayedends", name="lih", n_pno=2, pno={"maxrank": 2}
    )

    assert requests[0]["n_orbitals"] == 4
    assert requests[0]["madness"] == {"dft": {"charge": 0}, "pno": {"maxrank": 2}}
    # the Li 1s (no PNOs) is not active
    assert [o.idx_total for o in mol.integral_manager.active_orbitals] == [1, 2, 3]
    assert list(mol.integral_manager.active_space.reference_orbitals) == [0, 1]
    o1 = mol.integral_manager.orbitals[1]
    assert [x.idx_total for x in mol.get_pair_orbitals(o1, o1, exclude=o1)] == [2, 3]

    read = sun.Molecule(
        geometry=geometry, backend="frayedends", name="lih", n_pno="read"
    )
    assert len(requests) == 1
    assert read.make_hamiltonian() == mol.make_hamiltonian()
    assert read.mra_orbitals == mol.mra_orbitals
    assert [o.pair for o in read.integral_manager.orbitals] == [
        o.pair for o in mol.integral_manager.orbitals
    ]

    mol.save_mra_orbitals("act_SPA_orb", "orbs")
    assert sorted(os.listdir("orbs")) == [f"act_SPA_orb{i}.data" for i in range(4)]


@pytest.mark.skipif(not has_pyscf, reason="PySCF not installed")
@pytest.mark.parametrize("solver", ["fci", "dmrg"])
def test_frayed_state_average_rdms(solver):
    # the state average rdms (dirac) give the mean energy of the roots
    from sunrise.molecules.qubit_base._frayed_worker import _SA_SOLVERS

    if solver == "dmrg":
        pytest.importorskip("pyblock2")
    mol = sun.Molecule(
        geometry="H 0.0 0.0 -1.5\nH 0.0 0.0 -0.5\nH 0.0 0.0 0.5\nH 0.0 0.0 1.5",
        basis_set="6-31g",
        backend="pyscf",
    )
    c, h, g = mol.get_integrals(ordering="chem")
    energies, rdm1, rdm2 = _SA_SOLVERS[solver](c, h, g.elems, mol.n_electrons, 3)
    e_rdm = (
        c
        + numpy.einsum("pq,pq", h, rdm1)
        + 0.5 * numpy.einsum("pqrs,prqs", rdm2, g.elems)
    )
    assert len(energies) == 3
    assert numpy.isclose(numpy.trace(rdm1), mol.n_electrons)
    assert numpy.isclose(e_rdm, numpy.mean(energies), atol=1.0e-8)


@pytest.mark.skipif(not has_pyscf, reason="PySCF not installed")
@pytest.mark.skipif(not has_madness, reason="madness was not found")
def test_madness_pyscf_bridge():
    mol = sun.chemistry.Molecule(
        geometry="Be 0.0 0.0 0.0",
        units="angstrom",
        n_pno=2,
        pno={"diagonal": True, "maxrank": 2},
    )
    H = mol.make_hamiltonian()
    e1 = numpy.linalg.eigvalsh(H.to_matrix())[0]
    e2 = mol.compute_energy("fci")
    e3 = mol.compute_energy("ccsd")
    e4 = mol.compute_energy("ccsd(t)")
    assert numpy.isclose(e1, e2)
    # CCSD(T) and CCSD are not exact but close in this case
    # primarily testing a functioning interface and keywords here
    assert numpy.isclose(e1, e4, atol=1.0e-3)
    assert numpy.isclose(e3, e4, atol=1.0e-3)


@pytest.mark.skipif(condition=not has_pyscf, reason="pyscf not found")
@pytest.mark.skipif(not has_madness, reason="madness was not found")
@pytest.mark.parametrize("restrict_to_hcb", [False, True])
def test_orbital_optimization(restrict_to_hcb):
    name = "data/beh2_{R}"
    R = 4.5
    n_pno = None
    if os.path.isfile(name.format(R=R) + "_gtensor.npy"):
        n_pno = "read"
    success = False
    for _ in range(10):
        mol = sun.Molecule(n_pno=n_pno, name=name.format(R=R))
        # the next attempts only change the random start, not the madness data
        n_pno = "read"
        U = mol.make_upccgsd_ansatz("SPA")
        if restrict_to_hcb:
            U = mol.make_upccgsd_ansatz("HCB-SPA")
        opt_mol = sun.optimize_orbitals(
            molecule=mol,
            circuit=U,
            silent=False,
            initial_guess="random_loc=0.0_scale=1.0",
            vqe_solver_arguments={"restrict_to_hcb": restrict_to_hcb},
        ).molecule
        if restrict_to_hcb:
            H = opt_mol.make_hardcore_boson_hamiltonian()
        else:
            H = opt_mol.make_hamiltonian()
        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=False)
        success = numpy.isclose(result.energy, -15.562016590541667, atol=1.0e-2)
        if success:
            break
    assert success


@pytest.mark.skipif(not has_madness, reason="madness was not found")
@pytest.mark.skipif(condition=not has_pyscf, reason="pyscf not found")
def test_orbital_optimization_adapt():
    name = "data/beh2_{R}"
    R = 4.5
    n_pno = None
    if os.path.isfile(name.format(R=R) + "_gtensor.npy"):
        n_pno = "read"
    success = False
    for _ in range(10):
        mol = sun.Molecule(n_pno=n_pno, name=name.format(R=R))
        # the next attempts only change the random start, not the madness data
        n_pno = "read"
        U = mol.make_upccgsd_ansatz("SPA")
        operator_pool = sun.ADAPT.adapt.MolecularPool(molecule=mol, indices="UCCD")

        class AdaptWrapper:
            class ReturnWrapper:
                def __init__(self, circuit, variables, energy):
                    self.circuit = circuit
                    self.variables = variables
                    self.energy = energy

            def __init__(self, operator_pool, spa):
                self.operator_pool = operator_pool
                self.spa = spa

            def __call__(self, H, circuit, molecule, *args, **kwargs):
                solver = sun.adapt.Adapt(
                    H=H, Upre=self.spa, operator_pool=self.operator_pool, maxiter=1
                )
                result = solver(Upre=self.spa, operator_pool=self.operator_pool)
                final_circuit = self.spa + result.U
                return self.ReturnWrapper(
                    circuit=final_circuit,
                    variables=result.variables,
                    energy=result.energy,
                )

        result = sun.optimize_orbitals(
            molecule=mol,
            circuit=U,
            silent=False,
            initial_guess="random_loc=0.0_scale=1.0",
            vqe_solver=AdaptWrapper(operator_pool, U),
        )
        success = numpy.isclose(result.energy, -15.562016590541667, atol=1.0e-2)
        if success:
            break
    assert success
