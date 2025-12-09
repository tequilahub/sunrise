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
from scipy.optimize import minimize as scp_min

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

def mutual_info_simple(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, orb_b:int=1, PSSR:bool=False, NSSR:bool=False)->float:
    rho_a = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_a)
    S_a = quantum_entropy(rho_a)
    rho_b = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_b)
    S_b = quantum_entropy(rho_b)
    rho_ab = compute_two_orb_rdm(mol, circuit, variables, initial_state, p_orb=orb_a, q_orb=orb_b, PSSR=PSSR, NSSR=NSSR)
    S_ab = quantum_entropy(rho_ab)

    return S_a + S_b - S_ab # there might be a 0.5 depending to convention

def one_orb_mutual_info(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, PSSR:bool=False, NSSR:bool=False):
    rho_a = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_a)
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
        I = 2*one_orb_entanglement(mol, circuit, variables, initial_state, orb_a=orb_a)

    return I

def total_mutual_info(mol, circuit=None, variables=None, initial_state=0, orbs=None, PSSR=False, NSSR=False):
    """
    Compute the mutual information between specified orbitals and the rest of the system.
    """

    if not orbs:
        orbs = list(range(mol.n_orbitals))

    one_entropies = []
    for orb in orbs:
        one_rdm = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb)
        one_entropies.append(quantum_entropy(one_rdm))
    one_entropy = sum(one_entropies)

    state = tq.simulate(circuit, variables=variables, initial_state=initial_state)
    rho = tq.paulis.Projector(state).to_matrix().real
    system_entropy = quantum_entropy(rho)

    return one_entropy - system_entropy

def one_orb_entanglement(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, PSSR:bool=False, NSSR:bool=False)->float:
    rho_a = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_a)
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
        # assert np.isclose(S_a,S_b)
        E = S_a

    return E



def func(x, d, rho):
    """
    Computes S(rho || sigma) where sigma is constructed from x.
    Probabilities are derived from the norms of the vectors in x.
    """
    x_reshaped = x.reshape(2*d, 4)
    components = []
    
    # Extract vectors
    vecs_A = x_reshaped[0::2]
    vecs_B = x_reshaped[1::2]
    
    # Calculate norms
    nA = np.linalg.norm(vecs_A, axis=1)
    nB = np.linalg.norm(vecs_B, axis=1)
    
    # Calculate weights: p_k = |v_A|^2 * |v_B|^2
    # Add epsilon to avoid division by zero
    raw_weights = (nA**2) * (nB**2)
    total_weight = np.sum(raw_weights)
    
    if total_weight < 1e-12:
        # Fallback for zero vectors
        probs = np.ones(d) / d
    else:
        probs = raw_weights / total_weight

    for k in range(d):
        # Normalize vectors for the state creation
        # (create_separable_mixed_state handles normalization, but we need norms for probs)
        vA = vecs_A[k]
        vB = vecs_B[k]
        components.append((probs[k], vA, vB))

    # Construct Sigma
    sigma = create_separable_mixed_state(components)
    sigma = change_basis(sigma, 'to_molecular')
    
    return quantum_relative_entropy(rho=rho, sigma=sigma)

def relative_entropy_gradient_matrix(rho, sigma, epsilon=1e-12):
    """
    Computes the matrix gradient of -Tr(rho log sigma) w.r.t sigma
    using the Daleckii-Krein formula (divided differences).
    """
    # 1. Diagonalize sigma
    s_evals, s_evecs = eigh(sigma)
    
    # Clip eigenvalues to avoid log(0) errors
    s_evals = np.clip(s_evals, epsilon, None)
    
    # 2. Rotate rho to sigma basis: rho_rot = U^dag @ rho @ U
    rho_rot = s_evecs.conj().T @ rho @ s_evecs
    
    # 3. Prepare differences
    lambda_j = s_evals[:, None] 
    lambda_k = s_evals[None, :]
    
    lin_diff = lambda_j - lambda_k
    mask = np.abs(lin_diff) < epsilon
    
    D = np.zeros_like(rho)
    
    # --- CASE 1: Non-Degenerate ---
    log_lambda = np.log(s_evals)
    log_diff = log_lambda[:, None] - log_lambda[None, :]
    D[~mask] = log_diff[~mask] / lin_diff[~mask]
    
    # --- CASE 2: Degenerate (Limit Case) ---
    rows, cols = np.nonzero(mask)
    D[rows, cols] = 1.0 / s_evals[rows]
    
    # 4. Element-wise multiply with rotated rho
    grad_rot = -rho_rot * D
    
    # 5. Rotate back to original basis
    grad_sigma = s_evecs @ grad_rot @ s_evecs.conj().T
    
    return grad_sigma.real

def gradient_func(x, d, rho):
    x_reshaped = x.reshape(2*d, -1)
    
    # --- 1. Forward Pass ---
    vecs_A = x_reshaped[0::2]
    vecs_B = x_reshaped[1::2]
    
    nA = np.linalg.norm(vecs_A, axis=1)
    nB = np.linalg.norm(vecs_B, axis=1)
    
    psi_A = vecs_A / (nA[:, None] + 1e-12)
    psi_B = vecs_B / (nB[:, None] + 1e-12)
    
    raw_weights = nA**2 * nB**2
    total_weight = np.sum(raw_weights)
    probs = raw_weights / total_weight
    
    components = []
    for k in range(d):
        components.append((probs[k], psi_A[k], psi_B[k]))
        
    # Create sigma in computational basis
    sigma_sep = create_separable_mixed_state(components)
    
    # Transform to molecular basis to compute gradient against rho
    sigma_mol = change_basis(sigma_sep, 'to_molecular')
    
    # --- 2. Matrix Gradient in Molecular Basis ---
    grad_sigma_mol = relative_entropy_gradient_matrix(rho, sigma_mol)
    
    # --- 3. Transform Gradient back to Computational Basis ---
    # This replaces the inverse permutation logic
    grad_sigma_comp = change_basis(grad_sigma_mol, 'to_computational')
        
    # --- 4. Chain Rule for Norm-Weighted Params ---
    grad_x = np.zeros_like(x_reshaped)
    dimA, dimB = psi_A.shape[1], psi_B.shape[1]
    
    # Use the computational basis gradient for backprop
    G_tensor = grad_sigma_comp.reshape(dimA, dimB, dimA, dimB)
    
    traces = np.zeros(d)
    grad_projectors_A = []
    grad_projectors_B = []
    
    for k in range(d):
        # Contract for Projector Gradients
        g_PA = probs[k] * np.einsum('ikjl,k,l->ij', G_tensor, psi_B[k], psi_B[k].conj()).real
        g_PB = probs[k] * np.einsum('ikjl,i,j->kl', G_tensor, psi_A[k], psi_A[k].conj()).real
        
        grad_projectors_A.append(g_PA)
        grad_projectors_B.append(g_PB)
        
        # Contract for Weight Gradients
        term = np.einsum('ij,ij->', g_PA, np.outer(psi_A[k], psi_A[k].conj())).real
        traces[k] = term / probs[k] if probs[k] > 1e-10 else 0

    avg_trace = np.dot(probs, traces)
    dL_dw = (traces - avg_trace) / total_weight
    
    for k in range(d):
        # Projector gradient part
        def unnorm_grad(g_psi, psi, n):
            return (g_psi - np.dot(psi.conj(), g_psi).real * psi) / (n + 1e-12)

        g_psiA = 2 * grad_projectors_A[k] @ psi_A[k]
        g_psiB = 2 * grad_projectors_B[k] @ psi_B[k]
        
        term1_A = unnorm_grad(g_psiA, psi_A[k], nA[k])
        term1_B = unnorm_grad(g_psiB, psi_B[k], nB[k])
        
        # Weight gradient part
        d_weight_dA = dL_dw[k] * 2 * nA[k] * (nB[k]**2)
        d_weight_dB = dL_dw[k] * 2 * nB[k] * (nA[k]**2)
        
        term2_A = d_weight_dA * psi_A[k]
        term2_B = d_weight_dB * psi_B[k]
        
        grad_x[2*k] = term1_A + term2_A
        grad_x[2*k+1] = term1_B + term2_B

    return grad_x.flatten()

def get_classical_diagonal_guess(rho):
    """
    Generates the Classical Diagonal state parameters from a 16x16 Density Matrix.
    """
    # 1. Define the 4 basis vectors for a single orbital
    basis_vectors = np.eye(4) 
    
    states = []
    x0 = []

    # FIX: The input rho is in the Molecular Basis.
    # We must rotate it to Computational Basis to interpret index k as (idx_A * 4 + idx_B).
    rho_computational = change_basis(rho, direction='to_computational')

    # 2. Iterate through the diagonal of the COMPUTATIONAL basis matrix
    for k in range(16):
        prob = np.real(rho_computational[k, k])
        
        if prob > 1e-6:
            # Now k correctly corresponds to the computational basis index
            idx_A = k // 4 
            idx_B = k % 4 
            
            vec_A = basis_vectors[idx_A] 
            vec_B = basis_vectors[idx_B] 
            
            states.append((prob, vec_A, vec_B))
            
            weighted_vec_A = vec_A * np.sqrt(prob)
            weighted_vec_B = vec_B * np.sqrt(prob)
            
            x0.extend(weighted_vec_A)
            x0.extend(weighted_vec_B)

    return states, np.array(x0)

def two_orbs_entanglement(
    mol: tq.Molecule, 
    circuit: tq.QCircuit = None, 
    variables = None, 
    initial_state = None, 
    orb_a: int = 0, 
    orb_b: int = 1, 
    PSSR: bool = False, 
    NSSR: bool = False,
    min_components: int = 8,
    silent: bool = True
) -> float:
    """
    Computes the Relative Entropy of Entanglement E_R(rho) for a pair of molecular orbitals.
    
    Automates the process of:
    1. Computing the 2-orbital RDM (rho_ab).
    2. Generating a classical guess for the separable state.
    3. Padding the guess to ensure enough degrees of freedom (d >= min_components).
    4. Minimizing S(rho || sigma) using analytic gradients.
    
    Args:
        mol: Tequila Molecule object.
        circuit: The ansatz circuit (optional).
        variables: Optimized variables for the circuit (optional).
        initial_state: Initial state for RDM computation (optional).
        orb_a, orb_b: Indices of the two orbitals to test.
        PSSR, NSSR: Flags passed to compute_two_orb_rdm.
        min_components: Minimum number of separable components (d) to enforce. 
                        Prevents getting stuck in pure-state local minima.
        silent: If False, prints optimization progress.

    Returns:
        float: The minimized relative entropy (entanglement).
    """
    
    # 1. Compute the Reduced Density Matrix (rho_ab)
    # Note: Ensure your compute_two_orb_rdm handles the 'variables' and 'circuit' mapping correctly
    # If variables are provided, usually we map them to the circuit before passing, 
    # but here we pass them through as requested.
    if not silent: 
        print(f"Computing RDM for orbitals ({orb_a}, {orb_b})...")
        
    # Map variables if provided and circuit exists, otherwise pass as is
    # (Adjust this line based on how compute_two_orb_rdm expects the circuit)
    if circuit is not None and variables is not None:
        circuit = circuit.map_variables(variables)

    rho = compute_two_orb_rdm(mol, circuit, variables, initial_state, p_orb=orb_a, q_orb=orb_b, PSSR=PSSR, NSSR=NSSR)

    # 2. Generate Initial Guess (Classical Diagonal)
    states0, x0 = get_classical_diagonal_guess(rho)
    current_d = len(states0)
    
    # 3. Robustness Padding
    # If the guess is too simple (e.g., just the HF state), we need to add "empty" components
    # to allow the optimizer to find a mixed state solution.
    d = current_d
    if current_d < min_components:
        if not silent:
            print(f"  Padding components from {current_d} to {min_components} for robustness.")
        
        missing_d = min_components - current_d
        # Each component has 2 vectors of size 4 -> 8 parameters per component
        # Initialize with small random noise to break symmetry
        padding = np.random.randn(missing_d * 8) * 0.1 
        x0 = np.concatenate([x0, padding])
        d = min_components

    # 4. Symmetry Breaking
    # Add slight noise to the whole vector to avoid saddle points
    x0 = x0 + np.random.randn(*x0.shape) * 0.05

    # 5. Run Optimization
    args = (d, rho)
    
    if not silent:
        print("  Minimizing relative entropy...")

    result = scp_min(
        fun=func, 
        x0=x0, 
        args=args, 
        method="L-BFGS-B",  # Efficient gradient-based method
        jac=gradient_func,  # Analytic Gradient
        options={
            "maxiter": 2000, 
            "ftol": 1e-9, 
            "disp": False
        }
    )

    if not result.success and not silent:
        print(f"  Warning: Optimization finished with status: {result.message}")

    # Return the entanglement value
    return result.fun

def two_orbs_quantum_correlation(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, orb_b:int=1, PSSR:bool=False, NSSR:bool=False)->float:
    
    rho_ab = compute_two_orb_rdm(mol, circuit, variables, initial_state, p_orb=orb_a, q_orb=orb_b, PSSR=PSSR, NSSR=NSSR)
    chi = np.diag(np.diag(rho_ab))
        
    return quantum_relative_entropy(rho_ab, chi)

def two_orbs_classical_correlation(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, orb_b:int=1, PSSR:bool=False, NSSR:bool=False)->float:
    
    rho_ab = compute_two_orb_rdm(mol, circuit, variables, initial_state, p_orb=orb_a, q_orb=orb_b, PSSR=PSSR, NSSR=NSSR)
    chi = np.diag(np.diag(rho_ab))

    o1 = compute_one_orb_rdm(mol, circuit, one_orb=orb_a)
    o2 = compute_one_orb_rdm(mol, circuit, one_orb=orb_b)
    pi = np.kron(o1,o2)
    pi = change_basis(pi, 'to_molecular')
        
    return quantum_relative_entropy(chi, pi)

def two_orbs_mutual_info(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, orb_b:int=1, PSSR:bool=False, NSSR:bool=False)->float:
    
    rho_ab = compute_two_orb_rdm(mol, circuit, variables, initial_state, p_orb=orb_a, q_orb=orb_b, PSSR=PSSR, NSSR=NSSR)
    o1 = compute_one_orb_rdm(mol, circuit, one_orb=orb_a)
    o2 = compute_one_orb_rdm(mol, circuit, one_orb=orb_b)
    pi = np.kron(o1,o2)
    pi = change_basis(pi, 'to_molecular')
        
    return quantum_relative_entropy(rho_ab, pi)
