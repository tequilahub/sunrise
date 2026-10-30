from collections import Counter

import numpy as np
import openfermion
import pytest
import tequila as tq

import sunrise as sun
from sunrise.symmetry_adaptation import (
    FockSpaceState,
    IrrepProvider,
    PointGroup,
    PointGroupRepresentation,
    SpinCGSymmetrizationProcedure,
    SpinSymmetrizationProcedure,
    SymmetryAdaptedLinearCombintationSymmetrization,
)
from sunrise.utils.transformation_helpers import FermionicSWAP

HAS_PYSCF = "pyscf" in sun.chemistry.INSTALLED_QCHEMISTRY_BACKENDS

molecule_pointgroup_list: list[tuple[str, str]] = [
    ("D2h", "H 0. 0. 0. \n H 0. 0. 0.74804"),
    # ("C2v", "O 0.000000 0.000000 0.117790 \n H 0.000000 0.755453 -0.471161 \n H 0.000000 -0.755453 -0.471161"), # longer testing time
    ("C2h", "H 1. -1. 0. \n H 1. 1. 0. \n H -1. 1. 0. \n H -1. -1. 0."),
]


# Statically tests the values for D2h as imported from pyscf
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_point_group_correctness() -> None:
    D2h_expected_character_table = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, -1, -1, 1, -1, 1, 1, -1],
            [1, -1, 1, -1, -1, 1, -1, 1],
            [1, 1, -1, -1, -1, -1, 1, 1],
        ]
    )
    character_table = PointGroup.from_pyscf("D2h").character_table.matrix
    assert np.array_equal(character_table, D2h_expected_character_table)


# Tests the point group detected from the geometry and setting a subgroup instead
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.parametrize(
    "geometry, expected",
    [
        ("H 0. 0. 0. \n H 0. 0. 0.74804", "D2h"),
        ("H 1. -1. 0. \n H 1. 1. 0. \n H -1. 1. 0. \n H -1. -1. 0.", "D4h"),
        (
            "O 0.000000 0.000000 0.117790 \n H 0.000000 0.755453 -0.471161 \n H 0.000000 -0.755453 -0.471161",
            "C2v",
        ),
    ],
)
def test_point_group_detection(geometry, expected) -> None:
    mol = sun.Molecule(geometry=geometry, basis_set="sto-3g", backend="pyscf")
    assert mol.point_group.schoenflies_label == expected

    mol.point_group = "C2h" if expected != "C2v" else "Cs"
    assert mol.point_group.schoenflies_label == ("C2h" if expected != "C2v" else "Cs")


# Tests the point group given with the point_group keyword, the backend computes in its Abelian subgroup
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.parametrize(
    "point_group, expected, expected_abelian",
    [("D4h", "D4h", "D2h"), ("c2h", "C2h", "C2h")],
)
def test_point_group_keyword(point_group, expected, expected_abelian) -> None:
    mol = sun.Molecule(
        geometry="H 1. -1. 0. \n H 1. 1. 0. \n H -1. 1. 0. \n H -1. -1. 0.",
        basis_set="sto-3g",
        backend="pyscf",
        point_group=point_group,
    )
    assert mol.point_group.schoenflies_label == expected
    assert mol.abelian_point_group.schoenflies_label == expected_abelian
    assert mol.pyscf_molecule.groupname == expected_abelian


# Tests the non-Abelian character tables: the operations form a group, the characters
# are class functions and the irreps are orthogonal
def test_non_abelian_character_tables() -> None:
    for name in ["C3v", "C4v", "D2d", "D3h", "D3d", "D4h", "D6h", "D8h", "Td", "Oh"]:
        pg = getattr(PointGroup, name)
        operations = [
            np.asarray(op, dtype=float)
            for op in PointGroupRepresentation.get_geometric_representation(
                pg
            ).operations.values()
        ]
        characters = pg.character_table.matrix
        for g in operations:
            for j, h in enumerate(operations):
                k = next(
                    k for k, op in enumerate(operations) if np.allclose(op, g @ h @ g.T)
                )
                assert np.allclose(characters[:, k], characters[:, j]), name
        assert np.allclose(
            characters @ characters.T, pg.order * np.eye(len(characters))
        )


# Tests the operations of the non-Abelian groups on the native orbitals: the orbital transformations decompose
# into the expected irreps (for CH4 the C 2s, 2p and the four H 1s orbitals, for NH3 the p orbitals of N mix under C3)
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.parametrize(
    "geometry, expected, expected_irreps",
    [
        (
            "C 0. 0. 0. \n H 0.6276 0.6276 0.6276 \n H -0.6276 -0.6276 0.6276 \n H -0.6276 0.6276 -0.6276 \n H 0.6276 -0.6276 -0.6276",
            "Td",
            {"A1": 2, "T2": 2},
        ),
        (
            "\n".join(
                f"H {np.cos(k * np.pi / 4):.10f} {np.sin(k * np.pi / 4):.10f} 0."
                for k in range(8)
            ),
            "D8h",
            {"A1g": 1, "B1g": 1, "E1u": 1, "E2g": 1, "E3u": 1},
        ),
        (
            "\n".join(
                f"H {np.cos(2 * k * np.pi / 3):.10f} {np.sin(2 * k * np.pi / 3):.10f} {z}"
                for z in (0.6, -0.6)
                for k in range(3)
            ),
            "D3h",
            {"A1'": 1, "E'": 1, "A2''": 1, "E''": 1},
        ),
        (
            "H 1. 0. 0. \n H -1. 0. 0. \n H 0. 1. 0. \n H 0. -1. 0. \n H 0. 0. 1. \n H 0. 0. -1.",
            "Oh",
            {"A1g": 1, "Eg": 1, "T1u": 1},
        ),
        (
            "N 0. 0. 0.12 \n H 0.94 0. -0.27 \n H -0.47 0.8140638796 -0.27 \n H -0.47 -0.8140638796 -0.27",
            "C3v",
            {"A1": 3, "E": 2},
        ),
    ],
)
def test_orbital_representation_non_abelian(
    geometry, expected, expected_irreps
) -> None:
    mol = sun.Molecule(
        geometry=geometry, basis_set="sto-3g", backend="pyscf"
    ).use_native_orbitals()
    assert mol.point_group.schoenflies_label == expected

    operations = mol.mo_representation().operations
    table = mol.point_group.character_table
    # the character of an operation is the trace of its orbital transformation
    characters = np.array([np.trace(operations[op]) for op in table.operation_symbols])
    multiplicities = np.round(table.matrix @ characters / mol.point_group.order, 10)
    assert dict(zip(table.irreps, multiplicities)) == {
        irrep: expected_irreps.get(irrep, 0) for irrep in table.irreps
    }


# Tests that the operations are rotated onto the symmetry axes of a molecule that is not in the standard orientation:
# the canonical determinants of water get the same irreps in both orientations
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_rotated_molecule() -> None:
    geometry = "O 0. 0. 0.1271 \n H 0.758 0. -0.5085 \n H -0.758 0. -0.5085"
    c, s = np.cos(0.7), np.sin(0.7)
    rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ np.array(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    )
    rotated = "\n".join(
        f"{atom[0]} "
        + " ".join(f"{x:.10f}" for x in rotation @ np.array(atom[1:], float))
        for atom in (line.split() for line in geometry.split("\n"))
    )

    irreps = []
    for g in [geometry, rotated]:
        mol = sun.Molecule(geometry=g, basis_set="sto-3g", backend="pyscf")
        assert mol.point_group.schoenflies_label == "C2v"
        provider = IrrepProvider(mol, "loc")
        irreps.append(
            [state.irrep for state in FockSpaceState.non_ionic_states(mol, provider)]
        )
    assert None not in irreps[0]
    assert irreps[0] == irreps[1]


# Tests that the canonical provider falls back to the largest Abelian subgroup for non-Abelian groups
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_canonical_abelian_fallback() -> None:
    mol = sun.Molecule(
        geometry="H 1. -1. 0. \n H 1. 1. 0. \n H -1. 1. 0. \n H -1. -1. 0.",
        basis_set="sto-3g",
        backend="pyscf",
    )
    assert mol.point_group.schoenflies_label == "D4h"
    assert mol.abelian_point_group.schoenflies_label == "D2h"

    states = FockSpaceState.non_ionic_states(mol, IrrepProvider(mol, "canon"))
    for state in states:
        assert state.irrep in mol.abelian_point_group.character_table.irreps


# Tests the symmetry of charged molecules, atoms and geometries in bohr
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.parametrize(
    "geometry, kwargs, expected_orbital_irreps",
    [
        ("H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2.", {"charge": 1}, ["Ag", "B1u", "Ag"]),
        ("He 0. 0. 0.", {}, ["Ag"]),
        ("H 0. 0. 0. \n H 0. 0. 1.4", {"units": "bohr"}, ["Ag", "B1u"]),
    ],
)
def test_point_group_special_molecules(
    geometry, kwargs, expected_orbital_irreps
) -> None:
    mol = sun.Molecule(geometry=geometry, basis_set="sto-3g", backend="pyscf", **kwargs)
    assert mol.point_group.schoenflies_label == "D2h"
    assert mol.compute_orbital_irreps() == expected_orbital_irreps
    assert mol.mo_representation().operations["i"].shape == (
        mol.n_orbitals,
        mol.n_orbitals,
    )


# Tests that the localized irreps work for every encoding and nature of the molecule
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.parametrize(
    "nature, kwargs",
    [
        ("qubit", {"transformation": "BravyiKitaev"}),
        ("hybrid", {"select": "FB"}),
        ("fermionic", {}),
    ],
)
def test_localized_irreps_nature(nature, kwargs) -> None:
    mol = sun.Molecule(
        geometry="H 0. 0. 0. \n H 0. 0. 0.74804",
        basis_set="sto-3g",
        backend="pyscf",
        nature=nature,
        **kwargs,
    )
    for state in FockSpaceState.non_ionic_states(mol, IrrepProvider(mol, "loc")):
        assert state.irrep is not None


# Statically tests the inversion of H4 (square) with D2h on the native orbitals: it swaps opposite atoms
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_orbital_representation_correctness() -> None:
    mol = sun.Molecule(
        geometry="H 1. -1. 0. \n H 1. 1. 0. \n H -1. 1. 0. \n H -1. -1. 0.",
        basis_set="sto-3g",
        backend="pyscf",
    ).use_native_orbitals()
    mol.point_group = "D2h"
    D = mol.mo_representation().operations["i"]
    assert np.allclose(D, np.eye(4)[[2, 3, 0, 1]], atol=1e-10)

    # the orbitals 1 and 2 of the canonical orbitals are a degenerate pair, which C4 mixes:
    # an active space with only one of them is not mapped onto itself
    mol = sun.Molecule(
        geometry="H 1. -1. 0. \n H 1. 1. 0. \n H -1. 1. 0. \n H -1. -1. 0.",
        basis_set="sto-3g",
        backend="pyscf",
        active_orbitals=[0, 1],
    )
    with pytest.raises(ValueError, match="out of the active space"):
        mol.mo_representation()


# Tests that orbital i is on the qubits 2*i (up) and 2*i+1 (down) of the states, as in the rest of sunrise
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_orbital_order() -> None:
    mol = sun.Molecule(
        geometry="H 0. 0. 0. \n H 0. 0. 0.74804", basis_set="sto-3g", backend="pyscf"
    )
    # one electron in the bonding (Ag) or in the antibonding (B1u) orbital
    for bitstring, mo_occ, irrep in [
        ("1000", [1.0, 0.0], "Ag"),
        ("0010", [0.0, 1.0], "B1u"),
    ]:
        for provider in ["canon", "loc"]:
            state = FockSpaceState(mol, bitstring, IrrepProvider(mol, provider))
            assert state.mo_occ == mo_occ
            assert state.irrep == irrep

    mol4 = sun.Molecule(
        geometry="H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3.",
        basis_set="sto-3g",
        backend="pyscf",
    )
    state = SpinCGSymmetrizationProcedure(mol4).build_physical_state(
        open_shell_orbitals=[0, 1],
        S=0,
        m_s=0,
        coupling="singlet",
        closed_shell_orbitals=[2],
    )
    assert state.mo_occ == [1.0, 1.0, 2.0, 0.0]


# Test spin symmetrization for canonical orbitals
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.parametrize("molecule_pointgroup", molecule_pointgroup_list)
def test_spin_symmetrization(molecule_pointgroup) -> None:
    mol = sun.Molecule(
        geometry=molecule_pointgroup[1], basis_set="sto-3g", backend="pyscf"
    )
    mol.point_group = molecule_pointgroup[0]

    provider_canonical = IrrepProvider(mol, "canon")
    states_list = FockSpaceState.non_ionic_states(mol, provider_canonical)

    symm_list = SpinSymmetrizationProcedure(mol).symmetrize(states_list)
    for state in symm_list:
        assert state.spin_multiplicity is not None
        assert state.spin_multiplicity != "mixed"


# Statically tests the values for H2 with canonical MOs
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_spin_symmetrization_correctness() -> None:
    mol = sun.Molecule(
        geometry="H 0. 0. 0. \n H 0. 0. 0.74804", basis_set="sto-3g", backend="pyscf"
    )
    mol.point_group = "D2h"
    provider_canonical = IrrepProvider(mol, "canon")
    states_list = FockSpaceState.non_ionic_states(mol, provider_canonical)
    symm_list = SpinSymmetrizationProcedure(mol).symmetrize(states_list)

    expected_wavefunctions = [
        tq.QubitWaveFunction.from_string(s)
        for i, s in enumerate(
            [
                "+1.0000 |0101>",
                "+1.0000 |0011>",
                "-0.7071 |0110> +0.7071 |1001>",
                "+1.0000 |1100>",
                "+0.7071 |0110> +0.7071 |1001>",
                "+1.0000 |1010>",
            ],
            start=1,
        )
    ]

    assert len(symm_list) == len(expected_wavefunctions)
    for state in symm_list:
        assert any(
            state.wavefunction.isclose(expected) for expected in expected_wavefunctions
        )


# Tests SALC symmetrization for localized orbitals
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.parametrize("molecule_pointgroup", molecule_pointgroup_list)
def test_SALC_symmetrization(molecule_pointgroup) -> None:
    mol = sun.Molecule(
        geometry=molecule_pointgroup[1], basis_set="sto-3g", backend="pyscf"
    )
    mol.point_group = molecule_pointgroup[0]

    provider_localized = IrrepProvider(mol, "loc")
    states_list = FockSpaceState.non_ionic_states(mol, provider_localized)

    symm_list = SymmetryAdaptedLinearCombintationSymmetrization(mol).symmetrize(
        states_list
    )
    for state in symm_list:
        assert state.irrep is not None


# Statically tests the values for H2 with localized MOs
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_SALC_symmetrization_correctness() -> None:
    mol = sun.Molecule(
        geometry="H 0. 0. 0. \n H 0. 0. 0.74804", basis_set="sto-3g", backend="pyscf"
    ).use_native_orbitals()
    mol.point_group = "D2h"

    provider_localized = IrrepProvider(mol, "loc")
    states_list = FockSpaceState.non_ionic_states(mol, provider_localized)

    symm_list = SymmetryAdaptedLinearCombintationSymmetrization(mol).symmetrize(
        states_list
    )

    expected_wavefunctions = [
        tq.QubitWaveFunction.from_string(s)
        for i, s in enumerate(
            [
                "+0.7071 |1100> +0.7071 |0011>",
                "-0.7071 |1100> +0.7071 |0011>",
                "+1.0000 |0101>",
                "+0.7071 |0110> -0.7071 |1001>",
                "+0.7071 |0110> +0.7071 |1001>",
                "+1.0000 |1010>",
            ],
            start=1,
        )
    ]

    assert len(symm_list) == len(expected_wavefunctions)
    for state in symm_list:
        assert any(
            state.wavefunction.isclose(expected) for expected in expected_wavefunctions
        )


# Tests SALC symmetrization when the operations mix orbitals: C3 mixes the degenerate canonical orbitals of H3+,
# and every SALC is kept by the projector of its irrep, (d / |G|) sum_g chi(g) g
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_SALC_symmetrization_mixed_orbitals() -> None:
    mol = sun.Molecule(
        geometry="H 1. 0. 0. \n H -0.5 0.8660254038 0. \n H -0.5 -0.8660254038 0.",
        basis_set="sto-3g",
        backend="pyscf",
        charge=1,
    )
    assert mol.point_group.schoenflies_label == "D3h"

    states_list = FockSpaceState.non_ionic_states(mol, IrrepProvider(mol, "canon"))
    procedure = SymmetryAdaptedLinearCombintationSymmetrization(mol)
    symm_list = procedure.symmetrize(states_list)
    assert len(symm_list) == len(states_list)
    assert Counter(state.irrep for state in symm_list) == {"A1'": 2, "A2'": 3, "E'": 10}

    table = mol.point_group.character_table
    for state in symm_list:
        characters = table.dict[state.irrep]
        projected = tq.QubitWaveFunction(n_qubits=2 * mol.n_orbitals)
        for D, character in zip(procedure.pgr.operations.values(), characters):
            projected += (
                characters[0] * character / mol.point_group.order
            ) * procedure.pgr.apply(D, state.wavefunction)
        assert np.isclose(state.wavefunction.inner(projected), 1.0)


# Tests CG symmetrization: complete basis, pure spin states, and paper correctness
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.parametrize("molecule_pointgroup", molecule_pointgroup_list)
def test_spin_CG_symmetrization(molecule_pointgroup) -> None:
    mol = sun.Molecule(
        geometry=molecule_pointgroup[1], basis_set="sto-3g", backend="pyscf"
    )
    mol.point_group = molecule_pointgroup[0]
    provider_canonical = IrrepProvider(mol, "canon")

    det_states = FockSpaceState.non_ionic_states(mol, provider_canonical)
    cg = SpinCGSymmetrizationProcedure(mol)
    csf_states = cg.symmetrize(det_states)

    # CSF basis spans the same space as the determinant basis
    assert len(csf_states) == len(det_states)

    # Every CSF is a pure spin eigenfunction
    for state in csf_states:
        assert state.spin_multiplicity is not None
        assert state.spin_multiplicity != "mixed"


# Tests build_physical_state against the paper's Appendix A spin eigenfunctions
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_spin_CG_build_physical_state_correctness() -> None:
    # H2: O_{2,1}^{0,0} singlet (Eq A1)
    mol2 = sun.Molecule(
        geometry="H 0. 0. 0. \n H 0. 0. 1.", basis_set="sto-3g", backend="pyscf"
    )
    cg2 = SpinCGSymmetrizationProcedure(mol2)
    s = cg2.build_physical_state(
        open_shell_orbitals=[0, 1], S=0, m_s=0, coupling="singlet"
    )
    assert abs(s.S2) < 1e-6 and s.spin_multiplicity == "singlet"

    # H4: O_{4,2}^{0,0} two singlet pairs (Eq A4)
    mol4 = sun.Molecule(
        geometry="H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3.",
        basis_set="sto-3g",
        backend="pyscf",
    )
    cg4 = SpinCGSymmetrizationProcedure(mol4)
    pairs = cg4.build_physical_state(
        open_shell_orbitals=[0, 1, 2, 3], S=0, m_s=0, coupling="singlet_pairs"
    )
    assert abs(pairs.S2) < 1e-6 and pairs.spin_multiplicity == "singlet"
    assert all(abs(o - 1.0) < 1e-6 for o in pairs.mo_occ)

    # H6: O_{6,1}^{0,0} two quartets -> singlet via Hund's rule (Eq A6)
    mol6 = sun.Molecule(
        geometry="H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3. \n H 0. 0. 4. \n H 0. 0. 5.",
        basis_set="sto-3g",
        backend="pyscf",
    )
    cg6 = SpinCGSymmetrizationProcedure(mol6)
    n2 = cg6.build_physical_state(
        open_shell_orbitals=[0, 1, 2, 3, 4, 5],
        S=0,
        m_s=0,
        coupling={"type": "high_spin", "groups": [[0, 1, 2], [3, 4, 5]]},
    )
    assert abs(n2.S2) < 1e-6 and n2.spin_multiplicity == "singlet"
    assert (
        len(list(n2.wavefunction.items())) == 20
    )  # O_{6,1}^{0,0} has exactly 20 terms


# Tests the fermionic swap against the exact openfermion operator 1 - n_p - n_q + a_p^ a_q + a_q^ a_p (Jordan-Wigner):
# f(p, q) on two adjacent modes, f_swap(0, 3) as the orbital swap of H4 and the inversion commuting with the Hamiltonian
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_fermionic_swap_correctness() -> None:
    mol = sun.Molecule(
        geometry="H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3.",
        basis_set="sto-3g",
        backend="pyscf",
    ).use_native_orbitals()
    fswap = FermionicSWAP(mol)
    n_qubits = 2 * mol.n_orbitals

    exact = {}
    for p, q, n in [(0, 1, 2), (0, 6, n_qubits), (1, 7, n_qubits)]:
        F = (
            openfermion.FermionOperator("")
            - openfermion.FermionOperator(f"{p}^ {p}")
            - openfermion.FermionOperator(f"{q}^ {q}")
            + openfermion.FermionOperator(f"{p}^ {q}")
            + openfermion.FermionOperator(f"{q}^ {p}")
        )
        exact[(p, q)] = openfermion.get_sparse_operator(F, n_qubits=n).toarray()

    # the columns of the circuit matrices are the simulated basis states
    U = fswap.f(0, 1)
    f_matrix = np.array(
        [tq.simulate(U, initial_state=s).to_array() for s in range(4)]
    ).T
    assert np.allclose(f_matrix, exact[(0, 1)])

    U = fswap.f_swap(0, 3) + tq.gates.I(target=list(range(n_qubits)))
    f_swap_matrix = np.array(
        [tq.simulate(U, initial_state=s).to_array() for s in range(2**n_qubits)]
    ).T
    assert np.allclose(f_swap_matrix, exact[(0, 6)] @ exact[(1, 7)])

    U = (
        fswap.f_swap(0, 3)
        + fswap.f_swap(1, 2)
        + tq.gates.I(target=list(range(n_qubits)))
    )
    R = np.array(
        [tq.simulate(U, initial_state=s).to_array() for s in range(2**n_qubits)]
    ).T
    H = mol.make_hamiltonian().to_matrix()
    assert np.allclose(H @ R, R @ H)
