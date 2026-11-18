import numpy as np
import pytest

import sunrise as sun
from sunrise.expval import INSTALLED_FERMIONIC_BACKENDS
from sunrise.orbital_correlation.density_matrix_utils import (
    compute_one_orb_rdm,
    compute_one_spin_rdm,
    compute_two_orb_rdm,
)

HAS_PYSCF = "pyscf" in sun.chemistry.INSTALLED_QCHEMISTRY_BACKENDS
GEOMETRY = "H 0 0 0\nH 0 0 1.5\nH 0 0 3.0\nH 0 0 4.5"


@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            b,
            marks=pytest.mark.xfail(
                reason="tcc gets spin-flip terms like 0^ 5^ 4 1 wrong"
            )
            if b == "tcc"
            else (),
        )
        for b in INSTALLED_FERMIONIC_BACKENDS
        if b != "tequila"
    ],
)
def test_rdms_fcircuit(backend):
    # same UpCCSD state as qubit circuit (reference) and as FCircuit on a fermionic and a hybrid molecule
    qmol = sun.Molecule(geometry=GEOMETRY, basis_set="sto-3g", backend="pyscf")
    fmol = sun.Molecule(
        geometry=GEOMETRY,
        basis_set="sto-3g",
        backend="pyscf",
        nature="fermionic",
        fermionic_backend=backend,
    )
    hmol = sun.Molecule(
        geometry=GEOMETRY, basis_set="sto-3g", backend="pyscf", nature="hybrid"
    )
    Uq = qmol.make_ansatz("UpCCSD")
    Uf = fmol.make_ansatz("UpCCSD")
    variables = {k: 0.1 * (i + 1) for i, k in enumerate(Uq.extract_variables())}

    rho1 = compute_one_orb_rdm(qmol, Uq, variables, one_orb=1)
    rho2 = compute_two_orb_rdm(qmol, Uq, variables, p_orb=0, q_orb=2, PSSR=True)
    rhos = compute_one_spin_rdm(qmol, Uq, variables, one_orb=1, spin="down")
    assert np.allclose(compute_one_orb_rdm(fmol, Uf, variables, one_orb=1), rho1)
    assert np.allclose(
        compute_two_orb_rdm(fmol, Uf, variables, p_orb=0, q_orb=2, PSSR=True), rho2
    )
    assert np.allclose(
        compute_one_spin_rdm(fmol, Uf, variables, one_orb=1, spin="down"), rhos
    )
    # hybrid molecules with FCircuits measure the fermionic operators too, also on boson orbitals
    assert np.allclose(
        compute_two_orb_rdm(hmol, Uf, variables, p_orb=0, q_orb=2, PSSR=True), rho2
    )


@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_rdms_reordered_jordan_wigner():
    mol = sun.Molecule(geometry=GEOMETRY, basis_set="sto-3g", backend="pyscf")
    rmol = sun.Molecule(
        geometry=GEOMETRY,
        basis_set="sto-3g",
        backend="pyscf",
        transformation="reordered-jordan-wigner",
    )
    U = mol.make_ansatz("UpCCSD")
    Ur = rmol.make_ansatz("UpCCSD")
    variables = {k: 0.1 * (i + 1) for i, k in enumerate(U.extract_variables())}
    assert np.allclose(
        compute_two_orb_rdm(rmol, Ur, variables, p_orb=0, q_orb=2),
        compute_two_orb_rdm(mol, U, variables, p_orb=0, q_orb=2),
    )
