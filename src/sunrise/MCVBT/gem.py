"""Generalized eigenvalue problem in the span of several (non-orthogonal) circuits."""

import numpy as np
import scipy
from tequila import INSTALLED_BACKENDS
from tequila import BraKet as tqBraket
from tequila import Overlap as tqOverlap
from tequila.utils import to_float

from sunrise import simulate
from sunrise.expval import SUPPORTED_FERMIONIC_BACKENDS, Braket, Overlap
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase


def gem_fast(circuits, solver, variables, mol: QuantumChemistryBase, silent=True):
    """Solve the generalized eigenvalue problem ``H c = E S c`` in the basis of the circuit states.

    The Hamiltonian and overlap matrices are evaluated with the chosen backend.

    Parameters
    ----------
    circuits : list of QCircuit or list of FCircuit
        Circuits preparing the basis states.
    solver : str
        Backend: a fermionic backend or an installed tequila backend.
    variables : dict
        Values of the circuit variables.
    mol : QuantumChemistryBase
        Molecule defining the Hamiltonian.
    silent : bool, optional
        If False print the Hamiltonian and overlap matrices. Default is True.

    Returns
    -------
    v : numpy.ndarray
        Eigenvalues in ascending order (``v[0]`` is the ground-state energy).
    vv : numpy.ndarray
        Eigenvectors as columns (coefficients of the circuits).

    Raises
    ------
    ValueError
        If ``solver`` is unknown.
    """
    transition_matrix = np.eye(len(circuits))
    overlap_matrix = np.eye(len(circuits))

    for i in range(len(circuits)):
        for j in range(i, len(circuits)):
            if solver.lower() in SUPPORTED_FERMIONIC_BACKENDS:
                transition_element = Braket(
                    ket=circuits[j], bra=circuits[i], mol=mol, operator="H"
                )
                overlap_element = Overlap(ket=circuits[j], bra=circuits[i], mol=mol)
                transition_element = simulate(
                    transition_element, variables=variables, backend=solver
                )
                overlap_element = simulate(
                    overlap_element, variables=variables, backend=solver
                )
            elif solver.lower() in INSTALLED_BACKENDS:
                H = mol.make_hamiltonian()
                transition_element = simulate(
                    tqBraket(circuits[i], circuits[j], H), variables
                )
                overlap_element = simulate(
                    tqOverlap(circuits[i], circuits[j]), variables
                )
            else:
                raise ValueError(f"Unknown solver {solver}")
            transition_matrix[i, j] = to_float(transition_element)
            transition_matrix[j, i] = transition_matrix[i, j]

            overlap_matrix[i, j] = to_float(overlap_element)
            overlap_matrix[j, i] = overlap_matrix[i, j]

    if silent is False:
        print("======================")
        print("Transition matrix")
        print(transition_matrix)
        print("----------------------")
        print("Overlap matrix")
        print(overlap_matrix)
        print("======================")

    v, vv = scipy.linalg.eigh(a=transition_matrix, b=overlap_matrix)

    return v, vv
