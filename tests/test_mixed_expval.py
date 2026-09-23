import numpy as np
import tequila as tq
from tequila.utils import to_float
import pytest
import sunrise as sn
from sunrise.expval import INSTALLED_FERMIONIC_BACKENDS
from numpy import isclose


@pytest.mark.parametrize("same_variables", [True, False], ids=["same_variables", "different_variables"])
@pytest.mark.parametrize("braket", [False, True], ids=["expval", "braket"])
@pytest.mark.parametrize("backend", INSTALLED_FERMIONIC_BACKENDS)
def test_mixed_objective(backend, braket, same_variables):
    # <snU|snH|snU> + <tqU|tqH|tqU> simulated with sunrise, compared with the same objective built only with tequila
    geom = "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8"
    snmol = sn.Molecule(geometry=geom, basis_set='sto-3g', nature='f').use_native_orbitals()
    # qubit version of snmol, for the tequila reference of the sunrise term
    refmol = tq.Molecule(geometry=geom, basis_set='sto-3g', transformation='reordered-jordan-wigner').use_native_orbitals()
    # different geometry for the tequila term, so that mixing up the two Hamiltonians would be noticed
    tqgeom = "H 0.0 0.0 0.0\nH 0.0 0.0 1.5\nH 0.0 0.0 3.0\nH 0.0 0.0 4.5"
    tqmol = tq.Molecule(geometry=tqgeom, basis_set='sto-3g', transformation='reordered-jordan-wigner').use_native_orbitals()

    edges = {"bra": [(0, 1), (2, 3)], "ket": [(0, 2), (1, 3)]}
    snU = {k: sn.FCircuit.from_edges(edges=e, n_orb=snmol.n_orbitals) for k, e in edges.items()}
    refU = {k: refmol.make_ansatz("SPA", edges=e, optimize=False) for k, e in edges.items()}
    tqU = {k: tqmol.make_ansatz("SPA", edges=e, optimize=False) for k, e in edges.items()}
    if not same_variables:
        tqU = {k: U.map_variables({v: tq.Variable(("tq", v)) for v in U.extract_variables()}) for k, U in tqU.items()}

    if braket:
        sn_term = sn.Braket(bra=snU["bra"], ket=snU["ket"], mol=snmol)
        ref_term = tq.BraKet(bra=refU["bra"], ket=refU["ket"], operator=refmol.make_hamiltonian())
        tq_term = tq.BraKet(bra=tqU["bra"], ket=tqU["ket"], operator=tqmol.make_hamiltonian())
    else:
        sn_term = sn.ExpectationValue(U=snU["ket"], mol=snmol)
        ref_term = tq.ExpectationValue(H=refmol.make_hamiltonian(), U=refU["ket"])
        tq_term = tq.ExpectationValue(H=tqmol.make_hamiltonian(), U=tqU["ket"])

    objective = sn_term + tq_term
    assert len(objective.extract_variables()) == len(sn_term.extract_variables()) * (1 if same_variables else 2)

    rng = np.random.default_rng(42)
    variables = {v: rng.uniform(-np.pi, np.pi) for v in objective.extract_variables()}
    reference = tq.simulate(ref_term + tq_term, variables=variables)
    result = sn.simulate(objective, variables=variables, backend=backend)
    # tq.BraKet is complex valued: isclose compares the whole complex number, so an imaginary part would fail
    assert isclose(result, reference)


@pytest.mark.parametrize("same_variables", [True, False], ids=["same_variables", "different_variables"])
@pytest.mark.parametrize("backend", INSTALLED_FERMIONIC_BACKENDS)
def test_mixed_objective_minimize(backend, same_variables):
    # expectation values of test_mixed_objective, minimized with sunrise and compared with tq.minimize from the same
    # starting point. no braket: tq.minimize of a tq.BraKet fails on tequila devel (complex objective) and takes
    # more than 10 minutes per case on tequila PR #472 (gradients through the ancilla decomposition)
    geom = "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8"
    snmol = sn.Molecule(geometry=geom, basis_set='sto-3g', nature='f').use_native_orbitals()
    refmol = tq.Molecule(geometry=geom, basis_set='sto-3g', transformation='reordered-jordan-wigner').use_native_orbitals()
    tqgeom = "H 0.0 0.0 0.0\nH 0.0 0.0 1.5\nH 0.0 0.0 3.0\nH 0.0 0.0 4.5"
    tqmol = tq.Molecule(geometry=tqgeom, basis_set='sto-3g', transformation='reordered-jordan-wigner').use_native_orbitals()

    edges = [(0, 2), (1, 3)]
    snU = sn.FCircuit.from_edges(edges=edges, n_orb=snmol.n_orbitals)
    refU = refmol.make_ansatz("SPA", edges=edges, optimize=False)
    tqU = tqmol.make_ansatz("SPA", edges=edges, optimize=False)
    if not same_variables:
        tqU = tqU.map_variables({v: tq.Variable(("tq", v)) for v in tqU.extract_variables()})

    sn_term = sn.ExpectationValue(U=snU, mol=snmol)
    ref_term = tq.ExpectationValue(H=refmol.make_hamiltonian(), U=refU)
    tq_term = tq.ExpectationValue(H=tqmol.make_hamiltonian(), U=tqU)

    # no to_float cast: tequila optimizers cannot differentiate through it on tequila devel
    objective = sn_term + tq_term
    rng = np.random.default_rng(42)
    initial_values = {v: rng.uniform(-np.pi, np.pi) for v in objective.extract_variables()}
    reference = tq.minimize(ref_term + tq_term, initial_values=initial_values, silent=True)
    result = sn.minimize(objective, initial_values=initial_values, backend=backend, silent=True)
    assert isclose(result.energy, reference.energy)


@pytest.mark.parametrize("backend", INSTALLED_FERMIONIC_BACKENDS)
def test_mixed_objective_complex(backend):
    # an Rz on the ket gives the tequila braket an imaginary part, so the mixed objective is complex
    geom = "H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8"
    snmol = sn.Molecule(geometry=geom, basis_set='sto-3g', nature='f').use_native_orbitals()
    tqmol = tq.Molecule(geometry=geom, basis_set='sto-3g', transformation='reordered-jordan-wigner').use_native_orbitals()
    H = tqmol.make_hamiltonian()

    edges = {"bra": [(0, 1), (2, 3)], "ket": [(0, 2), (1, 3)]}
    snU = {k: sn.FCircuit.from_edges(edges=e, n_orb=snmol.n_orbitals) for k, e in edges.items()}
    tqU = {k: tqmol.make_ansatz("SPA", edges=e, optimize=False) for k, e in edges.items()}

    sn_term = sn.Braket(bra=snU["bra"], ket=snU["ket"], mol=snmol)
    ref_term = tq.BraKet(bra=tqU["bra"], ket=tqU["ket"], operator=H)
    tq_term = tq.BraKet(bra=tqU["bra"], ket=tqU["ket"] + tq.gates.Rz(angle="phi", target=0), operator=H)

    rng = np.random.default_rng(42)
    variables = {v: rng.uniform(-np.pi, np.pi) for v in (sn_term + tq_term).extract_variables()}
    reference = tq.simulate(ref_term + tq_term, variables=variables)
    result = sn.simulate(sn_term + tq_term, variables=variables, backend=backend)
    assert not isclose(reference.imag, 0.0)
    assert isclose(result, reference)
    # the cast of test_mixed_objective refuses a result with an imaginary part
    with pytest.raises(TypeError):
        sn.simulate((sn_term + tq_term).wrap(to_float), variables=variables, backend=backend)
