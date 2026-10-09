"""ADAPT tests that depend on chemistry.

Migrated from tequila (tests/test_adapt.py) together with the chemistry module.
The purely qubit-based ADAPT tests stay in tequila.
"""

import numpy
import sunrise as sun
import pytest
import openfermion

def make_test_molecule(nature):
    one = numpy.array([[-1.94102524, -0.31651552], [-0.31651552, -0.0887454]])
    two = numpy.array(
        [
            [
                [[1.02689005, 0.31648659], [0.31648659, 0.22767214]],
                [[0.31648659, 0.22767214], [0.85813498, 0.25556095]],
            ],
            [
                [[0.31648659, 0.85813498], [0.22767214, 0.25556095]],
                [[0.22767214, 0.25556095], [0.25556095, 0.76637672]],
            ],
        ]
    )

    return sun.Molecule(
        geometry="He 0.0 0.0 0.0", backend="base", one_body_integrals=one, two_body_integrals=two, nature=nature
    )

@pytest.mark.parametrize("nature", ["qubit", "hybrid", "fermionic"])
def test_molecular_example(nature):
    mol = make_test_molecule(nature)
    if nature == "hybrid":
        mol.update_select("FF")
    Upost = mol.make_excitation_gate(angle="a", indices=[(0, 2)])
    Upost += mol.make_excitation_gate(angle="a", indices=[(1, 3)])
    operator_pool = sun.ADAPT.MolecularPool(molecule=mol, indices="UpCCSD")
    H = mol.make_hamiltonian()
    if nature == "fermionic":
        H = "H"
    solver = sun.ADAPT.Adapt(
        H=H, Upre=mol.prepare_reference(), Upost=Upost, operator_pool=operator_pool,
    )
    result = solver(operator_pool=operator_pool, label=0)
    energy = mol.compute_energy('fci')
    assert numpy.isclose(result.energy, energy, atol=1.0e-4)

@pytest.mark.parametrize("nature", ["qubit", "hybrid", "fermionic"])
def test_molecular_excited_example(nature):
    mol = make_test_molecule(nature)
    if nature == "hybrid":
        mol.update_select("FF")
    H = mol.make_hamiltonian()
    if nature == "fermionic":
        sparse_mat = openfermion.get_sparse_operator(H, n_qubits=2*mol.n_orbitals).toarray()
        eigenvalues, eigenvectors = numpy.linalg.eigh(sparse_mat)
        n_qubits = 2*mol.n_orbitals
        H = "H"
    else:
        eigenvalues, eigenvectors = numpy.linalg.eigh(H.to_matrix())
        n_qubits = H.n_qubits
    reference_basis_state = 2 ** (n_qubits - 1) + 2 ** (n_qubits - 2)
    energies = []
    for i in range(len(eigenvalues)):
        if not numpy.isclose(eigenvectors[:, i][reference_basis_state], 0.0, atol=1.0e-4):
            energies.append(eigenvalues[i])

    operator_pool = sun.ADAPT.MolecularPool(molecule=mol, indices="UpCCSD") 

    circuits = []
    variables = {}
    for state in range(3):
        Upre = mol.prepare_reference()
        objective_factory = sun.ADAPT.ObjectiveFactorySequentialExcitedState(
            Upre=Upre, H=H, circuits=circuits, factors=[100.0] * len(circuits), molecule=mol,
        )
        solver = sun.ADAPT.Adapt(
            objective_factory=objective_factory, Upre=Upre, operator_pool=operator_pool,
        )
        result = solver(operator_pool=operator_pool, label=state, static_variables=variables)
        circuits.append(result.U)
        variables = {**variables, **result.variables}
        assert numpy.isclose(result.energy, energies[state], atol=1.0e-4)
