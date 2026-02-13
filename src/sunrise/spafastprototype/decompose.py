import tequila as tq
import networkx as nx
import time
import sys
from copy import deepcopy
from spafastprototype.utils import timing
from typing import Union
from collections import defaultdict

sys.setrecursionlimit(10000)

def make_decomposed_clusters(circuit:tq.QCircuit, grouping:Union[int,list,None]=None)->list:
    """
    Args
    ----
    grouping: int, list, or None
        Determines how qubits are clustered:
        - int: Target cluster size. Try to merge smaller clusters to reach this size +-1.
        - list: User-defined clusters, returned as-is.
        - None: Clusters are determined automatically based on the circuit's connectivity.
    circuit: QCircuit
        
    Returns
    -------
    List of qubit sets, eg. [{0,1}, {2,3}, {4,5}]
    """
    if isinstance(grouping, list):
        return grouping
    
    connections = circuit.to_networkx()
    clusters = list(nx.connected_components(connections))
    
    if isinstance(grouping,int):
        assert grouping >= 0
        temp = []
        lo = [len(c) for c in clusters]
        for i in range(len(clusters)):
            if lo[i] < grouping and lo[i]:
                t = deepcopy(set(clusters[i]))
                j =i+1
                while (len(t)<= grouping+1) and j<len(clusters):
                    if lo[j] and (len(t) + lo[j]) <= grouping+1:
                        t = t.union(deepcopy(clusters[j]))
                        clusters[j] = {}
                        lo[i] += lo[j]
                        lo[j] = 0
                    j += 1
                clusters[i] = {}
                temp.append(t)
            else: temp.append(clusters[i])
        clusters = [c for c in temp if len(c)]

    return clusters

def make_decomposed_circuits(U:tq.QCircuit, grouping:Union[int,list,None]=None, clusters:Union[list,None]=None)->list:
    """
    Given a global circuit return a list of local sub-circuits based on qubit clusters.
    Note: If a gate acts on qubits spanning two or more clusters, then it is NOT added to any sub-circuit!
    """
    if clusters is None : clusters = make_decomposed_clusters(U, grouping)
    circuits = []
    for c in clusters:
        UX = [gate for gate in U.gates if all([q in c for q in gate.qubits])]
        UX = tq.QCircuit(gates=UX)
        circuits.append(UX)

    return circuits

@timing
def decompose(H:tq.QubitHamiltonian, U:tq.QCircuit,nolist=True,grouping:Union[int,list,None]=None):
    """
    In the case where U is a tensor product: U = U_0 * U_1 * U_2 ... 
    Since H = c_0 + \sum_k (c_k P_k)
    where each global Pauli string is a tensor product of local Pauli strings:
    P_k = P_k,0 * P_k,1 * P_k,2 ...
    
    We can decompose the global expectation value:
    <H>_U --> c_0 + \sum_k (c_k <P_k,0>_U0 * <P_k,1>_U1 * <P_k,2>_U2 ...) 

    Args
    ----
    nolist: bool, optional
        If True, treats H as a single Hamiltonian and returns a 1D QTensor. 
        If False, H is expected to be a list of Hamiltonians. Default is True.
    grouping: int, list, or None, optional
        Determines how qubits are clustered:
        - int: Target cluster size. Try to merge smaller clusters to reach this size +-1.
        - list: User-defined clusters (e.g., [{0,1}, {2,3}]).
        - None: Clusters are determined automatically based on the circuit's connectivity.

    Returns
    -------
    QTensor containing the decomposed ExpectationValue.
    """

    all_expvals = {}
    
    # stupid thing used to try something
    if nolist:
        HH = [H]
    else:
        HH = H
    
    # if nolist, this is just a stupid wrapper and we have a one-dimensional object
    X = tq.QTensor(shape=[len(HH)])

    clusters = make_decomposed_clusters(U, grouping)
    circuits = make_decomposed_circuits(U, clusters=clusters)
    qubits_in_clusters = set()
    for c in clusters:
        for q in c:
            qubits_in_clusters.add(q)
    for k,H in enumerate(HH):
        paulis = H.paulistrings
        total_expval = 0.0
        if not len(paulis):
            continue

        for ps in paulis:
            # constant part of H
            if len(ps.qubits) == 0:
                total_expval += float(ps.coeff.real)
                continue

            # check if any ps.qubits are not inside clusters
            # skip if any unused qubit is not 'Z' -->  <0|X|0> = <0|Y|0> = 0
            unused_qubits = set()
            for q in ps.qubits:
                if q not in qubits_in_clusters:
                    unused_qubits.add(q)
            if any(ps[q] != 'Z' for q in unused_qubits):
                continue

            Pk_expval = float(ps.coeff.real)

            # split ps across circuit clusters and multiply local expectation values 
            #    <P_k>_U --> c_k <P_k,0>_U0 * <P_k,1>_U1 * <P_k,2>_U2 ...
            for i, u in enumerate(circuits):
                Pki = {q:ps[q] for q in ps.qubits if q in u.qubits}
                
                if len(Pki) == 0:
                    continue

                # create a specific key for this local ps on this sub-circuit
                # and use it to avoid double evaluations
                Pki = tq.PauliString(Pki)
                key = (str(Pki), i) 
                
                if key in all_expvals:
                    Pki_expval = all_expvals[key]
                else:
                    Hi = tq.QubitHamiltonian.from_paulistrings(Pki)
                    Pki_expval = tq.ExpectationValue(H=Hi, U=u)
                    all_expvals[key] = Pki_expval
                
                Pk_expval *= Pki_expval

            # <H>_U --> c_0 + \sum_k <P_k>_U
            total_expval += Pk_expval
        X[k] = total_expval
    return X

