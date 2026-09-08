import tequila as tq
import sunrise as sun
import numpy as np
import pytest

from sunrise.symmetry_adaptation import PointGroup, QCircuitRepresentationBuilder, FockSpaceState, IrrepProvider, SymmetryAdaptedLinearCombintationSymmetrization, SpinSymmetrizationProcedure, SpinCGSymmetrizationProcedure


HAS_PYSCF = "pyscf" in tq.chemistry.INSTALLED_QCHEMISTRY_BACKENDS

molecule_pointgroup_list: list[tuple[str, str]] = [
	("D2h", "H 0. 0. 0. \n H 0. 0. 0.74804"),
	#("C2v", "O 0.000000 0.000000 0.117790 \n H 0.000000 0.755453 -0.471161 \n H 0.000000 -0.755453 -0.471161"), # longer testing time
	("C2h", "H 1. -1. 0. \n H 1. 1. 0. \n H -1. 1. 0. \n H -1. -1. 0.")
]


# Statically tests the values for D2h as imported from pyscf
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_point_group_correctness() -> None:
	D2h_expected_character_table = np.array([[ 1,  1,  1,  1,  1,  1,  1,  1],
															[ 1, -1, -1,  1,  1, -1, -1,  1],
															[ 1, -1,  1, -1,  1, -1,  1, -1],
															[ 1,  1, -1, -1,  1,  1, -1, -1],
															[ 1,  1,  1,  1, -1, -1, -1, -1],
															[ 1, -1, -1,  1, -1,  1,  1, -1],
															[ 1, -1,  1, -1, -1,  1, -1,  1],
															[ 1,  1, -1, -1, -1, -1,  1,  1]])
	character_table = PointGroup.from_pyscf("D2h").character_table.matrix
	assert np.array_equal(character_table, D2h_expected_character_table)


@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.parametrize("molecule_pointgroup", molecule_pointgroup_list)
def test_representation_builder(molecule_pointgroup) -> None:
	mol = tq.Molecule(geometry=molecule_pointgroup[1],basis_set='sto-3g', backend="pyscf")
	pg = PointGroup.from_pyscf(molecule_pointgroup[0])
	provider_canonical = IrrepProvider(mol, pg, "canon")
	states = FockSpaceState.non_ionic_states(mol, provider_canonical)

	rep_ao = QCircuitRepresentationBuilder(mol, pg).build_ao_permutation_representation()
	rep_qcircuit = QCircuitRepresentationBuilder(mol, pg).build_qcircuit_representation()
	
	# assert the correct size for the ao matrices and the correct number of qubits for the qcircuit representation
	assert rep_ao.operations["E"].shape[0] == mol.n_orbitals
	assert rep_qcircuit.operations["E"].n_qubits == 2 * mol.n_orbitals

	# test inversibility if the operation i exists in the point group
	if "i" in pg.character_table.operation_symbols:
		assert np.array_equal(rep_ao.operations["i"], np.linalg.inv(rep_ao.operations["i"]))

	# test the application of the identity on a state in the Fock space
	rep_qcircuit.apply(rep_qcircuit.operations["E"], states[0].wavefunction)

	# test inversibility if the operation i exists in the point group
	if "i" in pg.character_table.operation_symbols:
		res = rep_qcircuit.apply(rep_qcircuit.operations["i"], states[0].wavefunction)
		res = rep_qcircuit.apply(rep_qcircuit.operations["i"], res)
		assert rep_qcircuit.is_close_function(states[0].wavefunction, res)


# Statically tests the values for the inversion of H4 (square) with D2h
# as imported from pyscf in its AO permutation representation
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_representation_builder_correctness() -> None:
	mol = tq.Molecule(geometry="H 1. -1. 0. \n H 1. 1. 0. \n H -1. 1. 0. \n H -1. -1. 0.",basis_set='sto-3g', backend="pyscf")
	pg = PointGroup.from_pyscf("D2h")
	rep_ao = QCircuitRepresentationBuilder(mol, pg).build_ao_permutation_representation()

	i_expected = np.array([[0, 0, 1, 0],
							[0, 0, 0, 1],
							[1, 0, 0, 0],
							[0, 1, 0, 0]])
	assert np.array_equal(rep_ao.operations["i"], i_expected)


# Test spin symmetrization for canonical orbitals
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.parametrize("molecule_pointgroup", molecule_pointgroup_list)
def test_spin_symmetrization(molecule_pointgroup) -> None:
	mol = tq.Molecule(geometry=molecule_pointgroup[1],basis_set='sto-3g', backend="pyscf")
	pg = sun.symmetry_adaptation.PointGroup.from_pyscf(molecule_pointgroup[0])

	provider_canonical = IrrepProvider(mol, pg, "canon")
	states_list = FockSpaceState.non_ionic_states(mol, provider_canonical)
	
	symm_list = SpinSymmetrizationProcedure(mol).symmetrize(states_list)
	for state in symm_list:
		assert state.spin_multiplicity is not None
		assert state.spin_multiplicity != "mixed"


# Statically tests the values for H2 with canonical MOs
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_spin_symmetrization_correctness() -> None:
	mol = tq.Molecule(geometry="H 0. 0. 0. \n H 0. 0. 0.74804",basis_set='sto-3g', backend="pyscf")
	pg = sun.symmetry_adaptation.PointGroup.from_pyscf("D2h")
	provider_canonical = IrrepProvider(mol, pg, "canon")
	states_list = FockSpaceState.non_ionic_states(mol, provider_canonical)
	symm_list = SpinSymmetrizationProcedure(mol).symmetrize(states_list)
	rep_qcircuit = QCircuitRepresentationBuilder(mol, pg).build_qcircuit_representation()

	expected_wavefunctions = [tq.QubitWaveFunction.from_string(s) for i, s in enumerate(
	[
		"+1.0000 |0101>",
		"+1.0000 |0011>",
		"-0.7071 |0110> +0.7071 |1001>",
		"+1.0000 |1100>",
		"+0.7071 |0110> +0.7071 |1001>",
		"+1.0000 |1010>",
	], start=1,) ]

	assert len(symm_list) == len(expected_wavefunctions)
	for state in symm_list:
		assert any([bool(rep_qcircuit.is_close_function(state.wavefunction, expected)) for expected in expected_wavefunctions])
	
	
# Tests SALC symmetrization for localized orbitals
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.parametrize("molecule_pointgroup", molecule_pointgroup_list)
def test_SALC_symmetrization(molecule_pointgroup) -> None:
	mol = tq.Molecule(geometry=molecule_pointgroup[1],basis_set='sto-3g', backend="pyscf")
	pg = sun.symmetry_adaptation.PointGroup.from_pyscf(molecule_pointgroup[0])

	provider_localized = IrrepProvider(mol, pg, "loc")
	states_list = FockSpaceState.non_ionic_states(mol, provider_localized)
	
	symm_list = SymmetryAdaptedLinearCombintationSymmetrization(mol, pg).symmetrize(states_list)
	for state in symm_list:
		assert state.irrep is not None


# Statically tests the values for H2 with localized MOs
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_SALC_symmetrization_correctness() -> None:
	mol = tq.Molecule(geometry="H 0. 0. 0. \n H 0. 0. 0.74804",basis_set='sto-3g', backend="pyscf").use_native_orbitals()
	pg = sun.symmetry_adaptation.PointGroup.from_pyscf("D2h")

	provider_localized = IrrepProvider(mol, pg, "loc")
	states_list = FockSpaceState.non_ionic_states(mol, provider_localized)
	
	symm_list = SymmetryAdaptedLinearCombintationSymmetrization(mol, pg).symmetrize(states_list)
	rep_qcircuit = QCircuitRepresentationBuilder(mol, pg).build_qcircuit_representation()

	expected_wavefunctions = [tq.QubitWaveFunction.from_string(s) for i, s in enumerate(
	[
		"+0.7071 |1100> +0.7071 |0011>",
		"-0.7071 |1100> +0.7071 |0011>",
		"+1.0000 |0101>",
		"+0.7071 |0110> -0.7071 |1001>",
		"+0.7071 |0110> +0.7071 |1001>",
		"+1.0000 |1010>",
	], start=1,) ]

	assert len(symm_list) == len(expected_wavefunctions)
	for state in symm_list:
		assert any([bool(rep_qcircuit.is_close_function(state.wavefunction, expected)) for expected in expected_wavefunctions])

# Tests fragment state generation (subset of orbitals treated as an open-shell fragment)
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_fragment_states() -> None:
	mol = tq.Molecule(geometry="H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3.", basis_set='sto-3g', backend="pyscf")
	pg = sun.symmetry_adaptation.PointGroup.from_pyscf("D2h")
	provider = IrrepProvider(mol, pg, "loc")

	# Full non-ionic space for reference: C(8,4) = 70 determinants
	full_states = FockSpaceState.non_ionic_states(mol, provider)
	assert len(full_states) == 70

	# Fragment on orbitals [0, 1]: 2 electrons in 2 orbitals -> C(4,2) = 6 states
	fragment_orbitals = [0, 1]
	frag_states = FockSpaceState.non_ionic_states(mol, provider, fragment_orbitals=fragment_orbitals)
	assert len(frag_states) == 6

	for state in frag_states:
		# each fragment state holds exactly 2 electrons in the fragment
		assert abs(sum(state.fragment_mo_occ) - 2) < 1e-6
		# orbitals outside the fragment are vacuum (occupation 0)
		full_occ = state.mo_occ
		for i in range(mol.n_orbitals):
			if i not in fragment_orbitals:
				assert abs(full_occ[i]) < 1e-6

# Tests CG symmetrization: complete basis, pure spin states, and paper correctness
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.parametrize("molecule_pointgroup", molecule_pointgroup_list)
def test_spin_CG_symmetrization(molecule_pointgroup) -> None:
	mol = tq.Molecule(geometry=molecule_pointgroup[1], basis_set='sto-3g', backend="pyscf")
	pg = sun.symmetry_adaptation.PointGroup.from_pyscf(molecule_pointgroup[0])
	provider_canonical = IrrepProvider(mol, pg, "canon")

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
	mol2 = tq.Molecule(geometry="H 0. 0. 0. \n H 0. 0. 1.", basis_set='sto-3g', backend="pyscf")
	cg2 = SpinCGSymmetrizationProcedure(mol2)
	s = cg2.build_physical_state(open_shell_orbitals=[0, 1], S=0, m_s=0, coupling="singlet")
	assert abs(s.S2) < 1e-6 and s.spin_multiplicity == "singlet"

	# H4: O_{4,2}^{0,0} two singlet pairs (Eq A4)
	mol4 = tq.Molecule(geometry="H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3.",
					   basis_set='sto-3g', backend="pyscf")
	cg4 = SpinCGSymmetrizationProcedure(mol4)
	pairs = cg4.build_physical_state(open_shell_orbitals=[0, 1, 2, 3], S=0, m_s=0, coupling="singlet_pairs")
	assert abs(pairs.S2) < 1e-6 and pairs.spin_multiplicity == "singlet"
	assert all(abs(o - 1.0) < 1e-6 for o in pairs.mo_occ)

	# H6: O_{6,1}^{0,0} two quartets -> singlet via Hund's rule (Eq A6)
	mol6 = tq.Molecule(geometry="H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3. \n H 0. 0. 4. \n H 0. 0. 5.",
					   basis_set='sto-3g', backend="pyscf")
	cg6 = SpinCGSymmetrizationProcedure(mol6)
	n2 = cg6.build_physical_state(
		open_shell_orbitals=[0, 1, 2, 3, 4, 5], S=0, m_s=0,
		coupling={"type": "high_spin", "groups": [[0, 1, 2], [3, 4, 5]]}
	)
	assert abs(n2.S2) < 1e-6 and n2.spin_multiplicity == "singlet"
	assert len(list(n2.wavefunction.items())) == 20  # O_{6,1}^{0,0} has exactly 20 terms