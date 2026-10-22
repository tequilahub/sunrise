import os

import numpy
import pytest
import tequila as tq

import sunrise as sun
from sunrise.molecules.qubit_base import INSTALLED_QCHEMISTRY_BACKENDS

has_pyscf = "pyscf" in INSTALLED_QCHEMISTRY_BACKENDS

executable = sun.chemistry.madness_interface.QuantumChemistryMadness.find_executable()
root = os.environ.get("MAD_ROOT_DIR")
if executable is None:
    executable = (
        sun.chemistry.madness_interface.QuantumChemistryMadness.find_executable(root)
    )
print("root = ", root)
print("executable = ", executable)


def test_executable():
    if root is not None and executable is None:
        raise tq.TequilaException(
            "Found no pno_integrals executable but found MAD_ROOT_DIR={root}\nSeems like you wanted that tested"
        )


@pytest.mark.skipif(
    not (os.path.isfile("he_gtensor.npy") and os.path.isfile("he_htensor.npy")),
    reason="data not there",
)
def test_madness_he_data():
    # relies that he_xtensor.npy are present (x=g,h)
    geomstring = "He 0.0 0.0 0.0"
    molecule = sun.Molecule(name="he", geometry=geomstring, units="angstrom")
    H = molecule.make_hamiltonian()
    UHF = molecule.prepare_reference()
    EHF = tq.simulate(tq.ExpectationValue(H=H, U=UHF))
    assert numpy.isclose(-2.861522e00, EHF, atol=1.0e-5)
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    print(result.energy)
    assert numpy.isclose(-2.87761809, result.energy, atol=1.0e-5)


@pytest.mark.skipif(executable is None, reason="madness was not found")
def test_madness_frozen_core():
    molecule = sun.chemistry.Molecule(
        geometry="Li 0.0 0.0 0.0\nLi 0.0 0.0 1.6", units="angstrom"
    )
    assert molecule.n_orbitals == 2
    assert molecule.parameters.get_number_of_core_electrons() == 4


@pytest.mark.skipif(executable is None, reason="madness was not found")
def test_madness_full_he():
    # relies on madness being compiled and MAD_ROOT_DIR exported
    # or pno_integrals in the path
    geomstring = "He 0.0 0.0 0.0"
    molecule = sun.Molecule(geometry=geomstring, n_pno=1, units="angstrom")
    H = molecule.make_hamiltonian()
    UHF = molecule.prepare_reference()
    EHF = tq.simulate(tq.ExpectationValue(H=H, U=UHF))
    assert numpy.isclose(-2.861522e00, EHF, atol=1.0e-5)
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    assert numpy.isclose(-2.87761809, result.energy, atol=1.0e-5)


@pytest.mark.skipif(executable is None, reason="madness was not found")
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


@pytest.mark.skipif(executable is None, reason="madness was not found")
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
    assert numpy.isclose(-7.236247e00, EHF, atol=1.0e-5)
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    assert numpy.isclose(-7.251177798, result.energy, atol=1.0e-5)


@pytest.mark.skipif(executable is None, reason="madness was not found")
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
    assert numpy.isclose(-14.57269300, EHF, atol=1.0e-5)
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
    executable is None and not os.path.isfile("balanced_be_gtensor.npy"),
    reason="madness was not found",
)
def test_madness_upccgsd(trafo):
    n_pno = 2
    if os.path.isfile("balanced_be_gtensor.npy"):
        n_pno = None
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


@pytest.mark.skipif(executable is None, reason="madness was not found")
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


@pytest.mark.skipif(not has_pyscf, reason="PySCF not installed")
@pytest.mark.skipif(
    executable is None and not os.path.isfile("balanced_be_gtensor.npy"),
    reason="madness not installed and no files found",
)
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
@pytest.mark.skipif(
    executable is None and not os.path.isfile("balanced_be_gtensor.npy"),
    reason="madness not installed and no files found",
)
@pytest.mark.parametrize("restrict_to_hcb", [False, True])
def test_orbital_optimization(restrict_to_hcb):
    name = "data/beh2_{R}"
    R = 4.5
    n_pno = None
    success = False
    for _ in range(10):
        mol = sun.Molecule(n_pno=n_pno, name=name.format(R=R))
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


# # test takes a while
# @pytest.mark.skipif(condition=not has_pyscf, reason="pyscf not found")
# def test_orbital_optimization_adapt():
#     name = "data/beh2_{R}"
#     R = 4.5
#     n_pno = None
#     success = False
#     for _ in range(1):
#         mol = sun.Molecule(n_pno=n_pno, name=name.format(R=R))
#         U = mol.make_upccgsd_ansatz("SPA")
#         operator_pool = sun.ADAPT.adapt.MolecularPool(molecule=mol, indices="UCCD")

#         class AdaptWrapper:
#             class ReturnWrapper:
#                 def __init__(self, circuit, variables, energy):
#                     self.circuit = circuit
#                     self.variables = variables
#                     self.energy = energy

#             def __init__(self, operator_pool, spa):
#                 self.operator_pool = operator_pool
#                 self.spa = spa

#             def __call__(self, H, circuit, molecule, *args, **kwargs):
#                 solver = sun.adapt.Adapt(H=H, Upre=self.spa, operator_pool=self.operator_pool, maxiter=1)
#                 result = solver(Upre=self.spa, operator_pool=self.operator_pool)
#                 final_circuit = self.spa + result.U
#                 return self.ReturnWrapper(circuit=final_circuit, variables=result.variables, energy=result.energy)

#         opt_mol = sun.optimize_orbitals(
#             molecule=mol,
#             circuit=U,
#             silent=False,
#             initial_guess="random_loc=0.0_scale=1.0",
#             vqe_solver=AdaptWrapper(operator_pool, U),
#         ).molecule
#         H = opt_mol.make_hamiltonian()
#         E = tq.ExpectationValue(H=H, U=U)
#         result = tq.minimize(E, silent=False)
#         success = numpy.isclose(result.energy, -15.562016590541667, atol=1.0e-2)
#         if success:
#             break
#     assert success
