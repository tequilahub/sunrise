import tequila as tq
from tequila import QCircuit,QubitWaveFunction,QubitHamiltonian
from tequila.objective.objective import Variables
from tequila.quantumchemistry.qc_base import QuantumChemistryBase as tqMolecule
import numpy as np
from scipy.linalg import logm, eigh
import itertools
import math
from typing import List,Tuple,Union
from sunrise.orbital_correlation.density_matrix_utils import *

# Quantum entropy S(rho)
def quantum_entropy(rho:np.ndarray)->float:
    """
    Compute the quantum entropy S(rho).

    Parameters:
        rho (ndarray): Density matrix rho (Hermitian, positive semidefinite, trace = 1).

    Returns:
        float: The quantum entropy S(rho).
    """
    # Ensure rho is a numpy array
    rho = np.array(rho, dtype=np.complex128)

    # Validate the input density matrix
    if not np.allclose(rho, rho.conj().T):
        raise ValueError("rho must be Hermitian.")
    if not np.isclose(np.trace(rho), 1):
        raise ValueError("Trace of rho must be 1.")
    if np.any(np.linalg.eigvalsh(rho).round() < 0):
        raise ValueError("rho must be positive semidefinite.")

    # Compute directly
    # log_rho = logm(rho)
    # entropy = -np.trace(rho @ log_rho).real

    # Compute through eigenvalues
    rho_evals, rho_evecs = eigh(rho)
    rho_evals = np.clip(rho_evals, 1e-12, None)
    log_rho = rho_evecs @ np.diag(np.log(rho_evals)) @ rho_evecs.conj().T
    entropy = -np.trace(rho @ log_rho).real

    return entropy

# Quantum relative entropy S(rho||sigma)
def quantum_relative_entropy(rho:np.ndarray, sigma:np.ndarray)->float:
    """
    Compute the quantum relative entropy S(rho || sigma).

    Parameters:
        rho (ndarray): Density matrix rho (Hermitian, positive semidefinite, trace = 1).
        sigma (ndarray): Density matrix sigma (Hermitian, positive semidefinite, trace = 1).

    Returns:
        float: The quantum relative entropy S(rho || sigma).
    """
    # Ensure rho and sigma are numpy arrays
    rho = np.array(rho, dtype=np.complex128)
    sigma = np.array(sigma, dtype=np.complex128)

    # Validate the input density matrices
    if not np.allclose(rho, rho.conj().T):
        raise ValueError("rho must be Hermitian.")
    if not np.allclose(sigma, sigma.conj().T):
        raise ValueError("sigma must be Hermitian.")
    if not np.isclose(np.trace(rho), 1):
        raise ValueError("Trace of rho must be 1.")
    if not np.isclose(np.trace(sigma), 1):
        raise ValueError("Trace of sigma must be 1.")
    if np.any(np.linalg.eigvalsh(rho).round() < 0):
        raise ValueError("rho must be positive semidefinite.")
    if np.any(np.linalg.eigvalsh(sigma).round() < 0):
        raise ValueError("sigma must be positive semidefinite.")

    # Add some noise to make it positive semidefinite
    # epsilon = 1e-10
    # rho = (rho + epsilon * np.eye(sigma.shape[0])).real
    # sigma = (sigma + epsilon * np.eye(sigma.shape[0])).real

    # Compute directly
    # log_rho = logm(rho)
    # log_sigma = logm(sigma)
    # relative_entropy = np.trace(rho @ (log_rho - log_sigma)).real

    # Compute through eigenvalues
    rho_evals, rho_evecs = eigh(rho)
    sigma_evals, sigma_evecs = eigh(sigma)
    rho_evals = np.clip(rho_evals, 1e-12, None)
    sigma_evals = np.clip(sigma_evals, 1e-12, None)
    log_rho = rho_evecs @ np.diag(np.log(rho_evals)) @ rho_evecs.conj().T
    log_sigma = sigma_evecs @ np.diag(np.log(sigma_evals)) @ sigma_evecs.conj().T
    relative_entropy = np.trace(rho @ (log_rho - log_sigma)).real

    return relative_entropy

def mutual_info_2ordm(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, orb_b:int=1, PSSR:bool=False, NSSR:bool=False)->float:
    rho_a = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_a)
    S_a = quantum_entropy(rho_a)
    rho_b = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_b)
    S_b = quantum_entropy(rho_b)
    rho_ab = compute_two_orb_rdm(mol, circuit, variables, initial_state, p_orb=orb_a, q_orb=orb_b, PSSR=PSSR, NSSR=NSSR)
    S_ab = quantum_entropy(rho_ab)

    # return 0.5 * (S_a + S_b - S_ab) # there might be a 0.5 depending to convention
    return S_a + S_b - S_ab

def mutual_info_1ordm(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, orb_b:int=1, PSSR:bool=False, NSSR:bool=False): # TODO: orb_b is not necessary because I'm using only orb_a
    rho_a = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_a)
    rho_b = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_b)
    if PSSR:
        rho_a_evals, rho_a_evecs = eigh(rho_a)
        I = (rho_a_evals[0]+rho_a_evals[3])*np.log(rho_a_evals[0]+rho_a_evals[3]) + \
            (rho_a_evals[1]+rho_a_evals[2])*np.log(rho_a_evals[1]+rho_a_evals[2]) - \
            2*(rho_a_evals[0]*np.log(rho_a_evals[0])+rho_a_evals[1]*np.log(rho_a_evals[1])+\
               rho_a_evals[2]*np.log(rho_a_evals[2])+rho_a_evals[3]*np.log(rho_a_evals[3]))
    elif NSSR:
        rho_a_evals, rho_a_evecs = eigh(rho_a)
        I = rho_a_evals[0]*np.log(rho_a_evals[0]) + \
            (rho_a_evals[1]+rho_a_evals[2])*np.log(rho_a_evals[1]+rho_a_evals[2]) + \
            rho_a_evals[3]*np.log(rho_a_evals[3]) - \
            2*(rho_a_evals[0]*np.log(rho_a_evals[0])+rho_a_evals[1]*np.log(rho_a_evals[1])+\
               rho_a_evals[2]*np.log(rho_a_evals[2])+rho_a_evals[3]*np.log(rho_a_evals[3]))
    else:
        # S_a = quantum_entropy(rho_a)
        # S_b = quantum_entropy(rho_b)
        # rho_ab = compute_two_orb_rdm(mol, circuit, p_orb=orb_a, q_orb=orb_b, PSSR=PSSR, NSSR=NSSR)
        # S_ab = quantum_entropy(rho_ab)
        # I = S_a + S_b - S_ab
        I = 2*pure_state_entanglement(mol, circuit, variables, initial_state, orb_a=orb_a, orb_b=orb_b)

    return I

def total_mutual_info(mol, circuit=None, variables=None, initial_state=0, orbs=[0,1], PSSR=False, NSSR=False):
    """
    Compute the mutual information between specified orbitals and the rest of the system.
    """
    one_entropies = []
    for orb in orbs:
        one_rdm = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb)
        one_entropies.append(quantum_entropy(one_rdm))
    one_entropy = sum(one_entropies)

    state = tq.simulate(circuit, variables=variables, initial_state=initial_state)
    rho = tq.paulis.Projector(state).to_matrix().real
    system_entropy = quantum_entropy(rho)

    return one_entropy - system_entropy

def pure_state_entanglement(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, orb_b:int=1, PSSR:bool=False, NSSR:bool=False)->float:
    rho_a = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_a)
    rho_b = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_b)
    if PSSR==True:
        rho_a_evals, rho_a_evecs = eigh(rho_a)
        # Eq.(29) https://doi.org/10.1021/acs.jctc.0c00559
        plnp = [(rho_a_evals[i]+rho_a_evals[-1-i])*np.log(rho_a_evals[i]+rho_a_evals[-1-i]) if not np.isclose(rho_a_evals[i]+rho_a_evals[-1-i], 0, atol=1.e-6) else 0.0 for i in range(len(rho_a_evals)//2)]
        xlnx =[i * np.log(i) if not np.isclose(i, 0, atol=1.e-6) else 0.0 for i in rho_a_evals]
        E = (plnp[0]+plnp[1] ) - (xlnx[0]+xlnx[1]+xlnx[2]+xlnx[3])
    elif NSSR==True:
        rho_a_evals, rho_a_evecs = eigh(rho_a)
        # Eq.(29) https://doi.org/10.1021/acs.jctc.0c00559
        xlnx =[i * np.log(i) if not np.isclose(i, 0, atol=1.e-6) else 0.0 for i in rho_a_evals]
        E = (rho_a_evals[1]+rho_a_evals[2])*np.log(rho_a_evals[1]+rho_a_evals[2]) if not np.isclose(rho_a_evals[1]+rho_a_evals[2], 0, atol=1.e-6) else 0.0
        E -= (xlnx[1] + xlnx[2])
    else:
        S_a = quantum_entropy(rho_a)
        S_b = quantum_entropy(rho_b)
        # assert np.isclose(S_a,S_b)
        E = S_a

    return E






# def func(x, d, rho):
#     y = []
#     x = x.reshape(2*d, 4)
#     for i in range(0, len(x), 2):
#         y.append((1/d, x[i], x[i+1]))

#     sigma = create_separable_mixed_state(y)
#     sigma = change
#     return quantum_relative_entropy(rho=rho, sigma=sigma)




def get_classical_diagonal_guess(rho):
    """
    Generates the Classical Diagonal state parameters from a 16x16 Density Matrix.
    """
    # 1. Define the 4 basis vectors for a single orbital
    # |0>, |up>, |down>, |updown>
    basis_vectors = np.eye(4) 
    
    # Format 1: List of tuples
    states = []
    
    # Format 2: Flat list (x0)
    x0 = []

    # 2. Iterate through the diagonal of the 16x16 matrix
    # The total dimension is 16, corresponding to indices 0..15
    for k in range(16):
        prob = np.real(rho[k, k])
        
        # We only care about non-zero probabilities 
        # (Optimization Tip: Ignore states with prob ~ 0 to speed up)
        if prob > 1e-6:
            
            # Decode the flat index k into two orbital indices (i, j)
            # i is for Orbital A (Left), j is for Orbital B (Right)
            idx_A = k // 4  # Integer division
            idx_B = k % 4   # Modulo
            
            # Get the "One-Hot" vectors
            vec_A = basis_vectors[idx_A] # e.g., [1, 0, 0, 0]
            vec_B = basis_vectors[idx_B] # e.g., [0, 1, 0, 0]
            
            # --- POPULATE FORMAT 1 ---
            # (p, vector_A, vector_B)
            states.append((prob, vec_A, vec_B))
            
            # --- POPULATE FORMAT 2 (x0) ---
            # Strategy: To fit your exact format [a,b,c,d, e,f,g,h...],
            # we absorb the sqrt(probability) into the vectors.
            # This way, norm(vec_A) * norm(vec_B) = probability.
            
            weighted_vec_A = vec_A * np.sqrt(prob)
            weighted_vec_B = vec_B * np.sqrt(prob)
            
            # Extend the flat list
            x0.extend(weighted_vec_A)
            x0.extend(weighted_vec_B)

    return states, np.array(x0)