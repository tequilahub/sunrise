import os

import numpy
import pytest

import sunrise as sun

HAS_PSI4 = "psi4" in sun.chemistry.INSTALLED_QCHEMISTRY_BACKENDS


@pytest.mark.skipif(condition=not HAS_PSI4, reason="you don't have psi4")
def test_correction_psi4_active():
    geomstring = "He 0.0 0.0 0.0"
    # This active space does not need to make sense
    mol = sun.Molecule(
        geometry=geomstring,
        units="angstrom",
        basis_set="cc-pvqz",
        active_orbitals={"AG": [0, 1], "B1U": [0, 1]},
    )
    rdminfo = {"rdm__psi4_method": "detci"}
    dE = mol.perturbative_f12_correction(cabs_type="active", **rdminfo)

    assert numpy.isclose(dE, -0.023784170991254745, atol=1.0e-7)


@pytest.mark.skipif(condition=not HAS_PSI4, reason="you don't have psi4")
def test_correction_psi4_cabsplus():
    # Choice of CABS not sensible here
    geomstring = "He 0.0 0.0 0.0"
    mol = sun.Molecule(
        geometry=geomstring,
        units="angstrom",
        basis_set="cc-pvdz",
        active_orbitals=[i for i in range(2)],
        point_group="c1",
    )
    rdminfo = {"rdm__psi4_method": "detci"}
    cabs_options = {"cabs_name": "cc-pvqz"}
    dE = mol.perturbative_f12_correction(
        cabs_type="cabs+", cabs_options=cabs_options, **rdminfo
    )

    assert numpy.isclose(dE, -0.02285941739816253, atol=1.0e-7)


data_path = "data/f12/"
# the skipif is evaluated at collection, before conftest.py changes into the tests directory
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)


@pytest.mark.skipif(
    not (
        os.path.isfile(data_dir + "he-f12_gtensor.npy")
        and os.path.isfile(data_dir + "he-f12_htensor.npy")
        and os.path.isfile(data_dir + "he-f12_f12tensor.npy")
    ),
    reason="data not there",
)
@pytest.mark.parametrize("trafo", sun.chemistry.encodings.known_encodings())
def test_correction_madness(trafo):
    geomstring = "He 0.0 0.0 0.0"
    mol = sun.Molecule(
        name="data/f12/he-f12",
        geometry=geomstring,
        units="angstrom",
        basis_set="madness",
        n_pno="read",
        active_orbitals=[0, 1],
        madness_root_dir=None,
        transformation=trafo,
    )
    if not mol.supports_ucc():
        pytest.skip(f"{trafo} has no UCC support")
    U = mol.make_upccgsd_ansatz(name="UpCCD")
    angles = {(((0, 1),), "D", (None, 0)): -0.1278860185100716}
    rdminfo = {"U": U, "variables": angles}
    dE = mol.perturbative_f12_correction(
        f12_filename=str(data_path) + "he-f12_f12tensor.npy", **rdminfo
    )

    assert numpy.isclose(dE, -0.020945873534783065, atol=1.0e-7)
