import numpy as np
import scipy
from sunrise.expval import Braket, Overlap
from tequila.quantumchemistry import QuantumChemistryBase
from tequila import BraKet as tqBraket
from tequila import Overlap as tqOverlap
from tequila.utils import to_float
from sunrise import simulate
from sunrise.expval import SUPPORTED_FERMIONIC_BACKENDS
from tequila import INSTALLED_BACKENDS

def gem_fast(circuits, solver, variables, mol: QuantumChemistryBase, silent=True):
    """

    """
    #todo add description and rename function evtl.
    transition_matrix = np.eye(len(circuits))
    overlap_matrix    = np.eye(len(circuits))

    for i in range(len(circuits)):
        for j in range(i,len(circuits)):
            if solver.lower() in SUPPORTED_FERMIONIC_BACKENDS:
                transition_element = Braket(ket=circuits[j], bra=circuits[i], mol=mol, operator="H")
                overlap_element = Overlap(ket=circuits[j], bra=circuits[i], mol=mol)
                transition_element = simulate(transition_element, variables=variables, backend=solver)
                overlap_element = simulate(overlap_element, variables=variables, backend=solver)
            elif solver.lower() in INSTALLED_BACKENDS:
                H = mol.make_hamiltonian()
                transition_element  = simulate(tqBraket(circuits[i], circuits[j], H),variables)
                overlap_element     = simulate(tqOverlap(circuits[i], circuits[j]),variables)
            else:
                raise ValueError("Unknown solver {}".format(solver))
            transition_matrix[i,j] = to_float(transition_element)
            transition_matrix[j,i] = transition_matrix[i,j]

            overlap_matrix[i,j] = to_float(overlap_element)
            overlap_matrix[j,i] = overlap_matrix[i,j]

    if silent is False:
        print("======================")
        print("Transition matrix")
        print(transition_matrix)
        print("----------------------")
        print("Overlap matrix")
        print(overlap_matrix)
        print("======================")

    v,vv = scipy.linalg.eigh(a=transition_matrix,b=overlap_matrix)


    return v,vv