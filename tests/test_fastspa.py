import numpy as np
import pytest
import tequila as tq
from numpy import isclose

import sunrise as sun


@pytest.mark.parametrize(
    "geom",
    [
        "H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2",
        "O 0.00000 0.00000 0.11779\nH 0.00000 0.75545 -0.47116\nH 0.00000 -0.75545 -0.47116",
    ],
)
@pytest.mark.parametrize("nature", ["qubit", "hybrid"])
def test_fast_spa_opt(geom, nature):
    tqmol = sun.Molecule(geometry=geom, basis_set="sto-3g", units="a", backend="pyscf")
    snmol = sun.Molecule(
        geometry=geom, basis_set="sto-3g", units="a", nature=nature, backend="pyscf"
    )

    tqmol, tqedges = sun.CLPO.generate_CLPO_molecule_edges(mol=tqmol)
    snmol, snedges = sun.CLPO.generate_CLPO_molecule_edges(mol=snmol)

    tqU = tqmol.make_ansatz("SPA", edges=tqedges) + tqmol.UC(1, 2, "a")
    snU = snmol.make_ansatz("SPA", edges=snedges) + snmol.UC(1, 2, "a")

    tqres = tq.minimize(
        tq.ExpectationValue(H=tqmol.make_hamiltonian(), U=tqU), silent=True
    )
    snres = sun.minimize(
        sun.SPAFP.decompose(H=snmol.make_hamiltonian(), U=snU, grouping=4),
        silent=True,
        gradient="2-point",
        method_options={"finite_diff_rel_step": 1.0e-4},
    )

    assert isclose(tqres.energy, snres.energy)


@pytest.mark.parametrize(
    "geom",
    [
        "H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2",
        "O 0.00000 0.00000 0.11779\nH 0.00000 0.75545 -0.47116\nH 0.00000 -0.75545 -0.47116",
    ],
)
@pytest.mark.parametrize("nature", ["qubit", "hybrid"])
def test_fast_spa_opt_orb(geom, nature):
    tqmol = sun.Molecule(geometry=geom, basis_set="sto-3g", units="a", backend="pyscf")
    snmol = sun.Molecule(
        geometry=geom, basis_set="sto-3g", units="a", nature=nature, backend="pyscf"
    )

    tqmol, tqedges = sun.CLPO.generate_CLPO_molecule_edges(mol=tqmol)
    snmol, snedges = sun.CLPO.generate_CLPO_molecule_edges(mol=snmol)

    tqopt = sun.optimize_orbitals(
        molecule=tqmol,
        circuit=tqmol.make_ansatz("HCB-SPA", edges=tqedges),
        silent=True,
        use_hcb=True,
    )
    snopt = sun.SPAFP.run_spa(
        mol=snmol, edges=snedges, silent=True, initial_guess=False, grouping=4
    )

    assert isclose(tqopt.energy, snopt.energy)


def test_clpo_atom_edges_keyword():
    # the atom connectivity is atom_edges; the former name edges is still accepted
    mol = sun.Molecule(
        geometry="H 0. 0. 0.\nH 0. 0. 1.6\nH 0. 0. 3.2\nH 0. 0. 4.8",
        basis_set="sto-3g",
        units="a",
        backend="pyscf",
    )
    atom_edges = [(1, 2), (0, 3)]
    new, new_edges = sun.CLPO.generate_CLPO_molecule_edges(mol, atom_edges=atom_edges)
    old, old_edges = sun.CLPO.generate_CLPO_molecule_edges(mol, edges=atom_edges)
    assert new_edges == old_edges
    # the given connectivity is used: the orbitals differ from the automatic ones
    default, _ = sun.CLPO.generate_CLPO_molecule_edges(mol)
    assert not np.allclose(
        default.integral_manager.orbital_coefficients,
        new.integral_manager.orbital_coefficients,
    )
    assert (
        new.integral_manager.orbital_coefficients
        == old.integral_manager.orbital_coefficients
    ).all()
    assert (
        mol.use_CLPO_orbitals(edges=atom_edges).integral_manager.orbital_coefficients
        == new.integral_manager.orbital_coefficients
    ).all()
    with pytest.raises(TypeError):
        sun.CLPO.generate_CLPO_molecule_edges(
            mol, atom_edges=atom_edges, edges=atom_edges
        )
