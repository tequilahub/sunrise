import numpy as np
import scipy


try:
    from sunrise.expval.fqe_expval import FQEBraKet
except ImportError:
    pass
from sunrise.expval import Braket
from tequila.quantumchemistry import QuantumChemistryBase
from sunrise.MCVBT.QulacsBraKet import BraKetQulacs


def geminal_equation(circuits, solver, variables, mol: QuantumChemistryBase, silent=True):
    """

    """
    #todo add description and rename function evtl.
    transition_matrix = np.eye(len(circuits))
    overlap_matrix    = np.eye(len(circuits))

    for i in range(len(circuits)):
        for j in range(i,len(circuits)):

            if solver == "TCC":
                transition_element = Braket(ket=circuits[j], bra=circuits[i], molecule=mol, backend='tcc')
                overlap_element = Braket(ket=circuits[j], bra=circuits[i], backend='tcc', molecule=mol, operator='I')

            elif solver == "FQE":
                transition_element = FQEBraKet(ket_fcircuit=circuits[i], bra_fcircuit=circuits[j], molecule=mol)
                overlap_element    = FQEBraKet(ket_fcircuit=circuits[i], bra_fcircuit=circuits[j],
                                               n_ele=mol.n_electrons, n_orbitals=mol.n_orbitals)
            elif solver == "Qulacs":
                H = mol.make_hamiltonian()
                transition_element  = BraKetQulacs(circuits[i], circuits[j], H)
                overlap_element     = BraKetQulacs(circuits[i], circuits[j], H=None)

            else:
                raise ValueError("Unknown solver {}".format(solver))

            transition_matrix[i,j] = transition_element(variables)
            transition_matrix[j,i] = transition_matrix[i,j]

            overlap_matrix[i,j] = overlap_element(variables)
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