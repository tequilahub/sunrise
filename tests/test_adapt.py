"""ADAPT tests that depend on chemistry.

Migrated from tequila (tests/test_adapt.py) together with the chemistry module.
The purely qubit-based ADAPT tests stay in tequila.
"""

import numpy
import tequila as tq
import sunrise as sun


def make_test_molecule():
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

    return sun.chemistry.Molecule(
        geometry="He 0.0 0.0 0.0", backend="base", one_body_integrals=one, two_body_integrals=two
    )


def test_molecular_example():
    mol = make_test_molecule()
    Upost = mol.make_excitation_gate(angle="a", indices=[(0, 2)])
    Upost += mol.make_excitation_gate(angle="a", indices=[(1, 3)])
    operator_pool = tq.adapt.MolecularPool(molecule=mol, indices="UpCCSD")
    solver = tq.adapt.Adapt(
        H=mol.make_hamiltonian(), Upre=mol.prepare_reference(), Upost=Upost, operator_pool=operator_pool
    )
    result = solver(operator_pool=operator_pool, label=0)
    energy = numpy.linalg.eigvalsh(mol.make_hamiltonian().to_matrix())[0]
    assert numpy.isclose(result.energy, energy, atol=1.0e-4)
