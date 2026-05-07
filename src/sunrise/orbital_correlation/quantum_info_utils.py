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
    """
    Compute the pairwise mutual information I(A:B) = S(rho_A) + S(rho_B) - S(rho_AB)
    between two orbitals A and B.

    Parameters:
        mol (tqMolecule): Tequila molecule object defining the system.
        circuit (QCircuit, optional): Ansatz quantum circuit.
        variables (Variables, optional): Optimized parameters for the circuit.
        initial_state (QubitWaveFunction, optional): Initial state for the simulation.
        orb_a (int): Index of the first orbital.
        orb_b (int): Index of the second orbital.
        PSSR (bool): If True, apply the Parity Superselection Rule when
                     computing the two-orbital RDM. Defaults to False.
        NSSR (bool): If True, apply the Particle Number Superselection Rule when
                     computing the two-orbital RDM. Defaults to False.

    Returns:
        float: The mutual information I(A:B) between the two orbitals.
    """
    rho_a = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_a)
    S_a = quantum_entropy(rho_a)
    rho_b = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_b)
    S_b = quantum_entropy(rho_b)
    rho_ab = compute_two_orb_rdm(mol, circuit, variables, initial_state, p_orb=orb_a, q_orb=orb_b, PSSR=PSSR, NSSR=NSSR)
    S_ab = quantum_entropy(rho_ab)

    return S_a + S_b - S_ab # there might be a 0.5 depending to convention

def one_orb_mutual_info(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, PSSR:bool=False, NSSR:bool=False):
    """
    Compute the single-orbital mutual information I_i for orbital i.

    Depending on the symmetry restriction flags, different formulas are applied:
    - PSSR (Particle-number Symmetry Sector Restriction): Uses eigenvalue-based expression
      corresponding to Eq.(29) of https://doi.org/10.1021/acs.jctc.0c00559 with particle-hole
      pairing of eigenvalues.
    - NSSR (Number Symmetry Sector Restriction): Uses a variant where only number-conserving
      blocks are grouped.
    - Default (no restriction): Returns twice the single-orbital entanglement entropy via
      `one_orb_entanglement`.

    Parameters:
        mol (tqMolecule): Tequila molecule object defining the system.
        circuit (QCircuit, optional): Ansatz quantum circuit. Defaults to None.
        variables (Variables, optional): Optimized parameters for the circuit. Defaults to None.
        initial_state (QubitWaveFunction, optional): Initial state for the simulation. Defaults to None.
        orb_a (int): Index of the orbital. Defaults to 0.
        PSSR (bool): Apply Parity Superselection Rule. Defaults to False.
        NSSR (bool): Apply Particle Number Superselection Rule. Defaults to False.

    Returns:
        float: The single-orbital mutual information I_i.
    """
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
    Compute the total mutual information between specified orbitals and the rest of the system.

    The total mutual information is defined as the sum of single-orbital entropies minus the
    entropy of the full system state: I_total = sum_i S(rho_i) - S(rho).

    Parameters:
        mol: Tequila molecule object defining the system.
        circuit (QCircuit, optional): Ansatz quantum circuit.
        variables (Variables, optional): Optimized parameters for the circuit.
        initial_state (int or QubitWaveFunction, optional): Initial state.
        orbs (list of int, optional): List of orbital indices to include. If None, all
                                      molecular orbitals are used.

    Returns:
        float: The total mutual information summed over the specified orbitals.
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
    """
    Compute the single-orbital entanglement entropy E_i for orbital i.

    Depending on the symmetry restriction flags, the entanglement entropy is computed using
    a symmetry-adapted formula based on Eq.(29) of https://doi.org/10.1021/acs.jctc.0c00559:
    - PSSR: Pairs eigenvalues symmetrically (particle-hole pairing) before computing entropy.
    - NSSR: Groups number-conserving off-diagonal blocks when computing the entropy.
    - Default: Returns the standard von Neumann entropy S(rho_a).

    Parameters:
        mol (tqMolecule): Tequila molecule object defining the system.
        circuit (QCircuit, optional): Ansatz quantum circuit.
        variables (Variables, optional): Optimized parameters for the circuit.
        initial_state (QubitWaveFunction, optional): Initial state for the simulation.
        orb_a (int): Index of the orbital to compute entanglement for.
        PSSR (bool): Apply Parity Superselection Rule.
        NSSR (bool): Apply Particle Number Superselection Rule.

    Returns:
        float: The single-orbital entanglement entropy E_i.
    """
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
        E = S_a

    return E


def func(x, d, rho):
    """
    Objective function for minimizing the relative entropy of entanglement.

    Constructs a separable state sigma from the parameter vector x and computes the
    quantum relative entropy S(rho || sigma). The separable state is a convex combination
    of d product states, where the mixing probabilities are derived from the squared norms
    of the component vectors encoded in x.

    Each component k contributes a term p_k * |psi_A_k><psi_A_k| ⊗ |psi_B_k><psi_B_k|
    to sigma, where p_k = |v_A_k|^2 * |v_B_k|^2 (normalized over all components).

    Parameters:
        x (ndarray): Flat parameter vector of shape (2*d*4,), encoding d pairs of
                     4-dimensional vectors (one for subsystem A, one for subsystem B).
        d (int): Number of separable components in the decomposition.
        rho (ndarray): Target density matrix (16x16) in the molecular orbital basis.

    Returns:
        float: The quantum relative entropy S(rho || sigma).
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
        vA = vecs_A[k]
        vB = vecs_B[k]
        components.append((probs[k], vA, vB))

    # Construct Sigma
    sigma = create_separable_mixed_state(components)
    sigma = change_basis(sigma, 'to_molecular')
    
    return quantum_relative_entropy(rho=rho, sigma=sigma)

def relative_entropy_gradient_matrix(rho, sigma, epsilon=1e-12):
    """
    Compute the matrix gradient of -Tr(rho log sigma) with respect to sigma.

    Uses the Daleckii-Krein formula (divided differences) to handle both degenerate
    and non-degenerate cases of sigma's spectrum. The result is a matrix G such that
    the directional derivative of -Tr(rho log sigma) along dSigma is Tr(G @ dSigma).

    Parameters:
        rho (ndarray): Density matrix rho (Hermitian, positive semidefinite).
        sigma (ndarray): Density matrix sigma (Hermitian, positive semidefinite).
                         Must not have zero eigenvalues (clipped internally with epsilon).
        epsilon (float): Small value for clipping eigenvalues and detecting degeneracy.
                         Defaults to 1e-12.

    Returns:
        ndarray: Real-valued gradient matrix of shape (n, n), where n is the
                 dimension of the density matrices.
    """
    # 1. Diagonalize sigma
    s_evals, s_evecs = eigh(sigma)
    
    # Clip eigenvalues to avoid log(0) errors
    s_evals = np.clip(s_evals, epsilon, None)
    
    # 2. Rotate rho to sigma basis: rho_rot = U^dag @ rho @ U
    rho_rot = s_evecs.conj().T @ rho @ s_evecs
    
    # 3. Prepare differences using Daleckii-Krein divided differences
    lambda_j = s_evals[:, None] 
    lambda_k = s_evals[None, :]
    
    lin_diff = lambda_j - lambda_k
    mask = np.abs(lin_diff) < epsilon
    
    D = np.zeros_like(rho)
    
    # --- CASE 1: Non-Degenerate (standard divided difference) ---
    log_lambda = np.log(s_evals)
    log_diff = log_lambda[:, None] - log_lambda[None, :]
    D[~mask] = log_diff[~mask] / lin_diff[~mask]
    
    # --- CASE 2: Degenerate (Limit: f'(lambda) = 1/lambda for f=log) ---
    rows, cols = np.nonzero(mask)
    D[rows, cols] = 1.0 / s_evals[rows]
    
    # 4. Element-wise multiply with rotated rho
    grad_rot = -rho_rot * D
    
    # 5. Rotate back to original basis
    grad_sigma = s_evecs @ grad_rot @ s_evecs.conj().T
    
    return grad_sigma.real

def gradient_func(x, d, rho):
    """
    Compute the analytic gradient of the objective function `func` with respect to x.

    Performs a full forward pass to reconstruct sigma from x, then backpropagates
    through the relative entropy computation using the matrix gradient of -Tr(rho log sigma)
    (via `relative_entropy_gradient_matrix`) and the chain rule through the norm-weighted
    parameterization of the separable state.

    The gradient accounts for two contributions per component k:
    - The projector gradient: how the direction of each vector (psi_A_k, psi_B_k) affects sigma.
    - The weight gradient: how the magnitude of each vector affects the mixing probability p_k.

    Parameters:
        x (ndarray): Flat parameter vector of shape (2*d*4,), same format as `func`.
        d (int): Number of separable components in the decomposition.
        rho (ndarray): Target density matrix (16x16) in the molecular orbital basis.

    Returns:
        ndarray: Gradient vector of shape (2*d*4,), matching the shape of x.
    """
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
        # Projector gradient part (unnormalized gradient for a unit vector psi in direction v)
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
    Generate an initial guess for the separable state optimization from a 16x16 density matrix.

    Constructs a classical diagonal separable state by reading the diagonal of rho in the
    computational basis. Each non-negligible diagonal entry rho[k,k] is interpreted as a
    probability weight for the corresponding product basis state |i_A> ⊗ |i_B>, where
    k = idx_A * 4 + idx_B.

    The parameter vector x0 encodes each component as a pair of vectors (vec_A, vec_B)
    scaled by sqrt(prob), so that the norm-squared product recovers the diagonal probability.

    Parameters:
        rho (ndarray): Density matrix (16x16) in the molecular orbital basis. Internally
                       converted to the computational basis before reading the diagonal.

    Returns:
        tuple:
            states (list of tuple): List of (prob, vec_A, vec_B) tuples for each component
                                    with diagonal weight > 1e-6.
            x0 (ndarray): Flat parameter vector encoding the initial guess, suitable as
                          input to `func` and `gradient_func`.
    """
    # 1. Define the 4 basis vectors for a single orbital
    basis_vectors = np.eye(4) 
    
    states = []
    x0 = []

    # The input rho is in the Molecular Basis.
    # We must rotate it to Computational Basis to interpret index k as (idx_A * 4 + idx_B).
    rho_computational = change_basis(rho, direction='to_computational')

    # 2. Iterate through the diagonal of the COMPUTATIONAL basis matrix
    for k in range(16):
        prob = np.real(rho_computational[k, k])
        
        if prob > 1e-6:
            # k correctly corresponds to the computational basis index
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
    if not silent: 
        print(f"Computing RDM for orbitals ({orb_a}, {orb_b})...")
        
    if circuit is not None and variables is not None:
        circuit = circuit.map_variables(variables)

    rho = compute_two_orb_rdm(mol, circuit, variables, initial_state, p_orb=orb_a, q_orb=orb_b, PSSR=PSSR, NSSR=NSSR)

    # 2. Generate Initial Guess (Classical Diagonal)
    states0, x0 = get_classical_diagonal_guess(rho)
    current_d = len(states0)
    
    # 3. Robustness Padding
    # If the guess is too simple (e.g., just the HF state), add "empty" components
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
    """
    Compute the quantum correlation between two molecular orbitals.

    Quantum correlation is measured as the relative entropy between the two-orbital RDM
    rho_ab and its closest classical (diagonal) state chi = diag(rho_ab), i.e.
    S(rho_ab || chi). This quantifies the non-classical coherence present in rho_ab.

    Parameters:
        mol (tqMolecule): Tequila molecule object defining the system.
        circuit (QCircuit, optional): Ansatz quantum circuit.
        variables (Variables, optional): Optimized parameters for the circuit.
        initial_state (QubitWaveFunction, optional): Initial state for the simulation.
        orb_a (int): Index of the first orbital.
        orb_b (int): Index of the second orbital.
        PSSR (bool): Apply Parity Number Superselection Rule.
        NSSR (bool): Apply Particle Number Superselection Rule.

    Returns:
        float: The quantum correlation S(rho_ab || diag(rho_ab)).
    """
    rho_ab = compute_two_orb_rdm(mol, circuit, variables, initial_state, p_orb=orb_a, q_orb=orb_b, PSSR=PSSR, NSSR=NSSR)
    chi = np.diag(np.diag(rho_ab))
        
    return quantum_relative_entropy(rho_ab, chi)

def two_orbs_classical_correlation(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, orb_b:int=1, PSSR:bool=False, NSSR:bool=False)->float:
    """
    Compute the classical correlation between two molecular orbitals.

    Classical correlation is measured as the relative entropy between the diagonal state
    chi = diag(rho_ab) and the product of marginals pi = rho_a ⊗ rho_b, i.e.
    S(chi || pi). This captures the classical statistical dependence between the orbitals
    after discarding quantum coherences.

    Parameters:
        mol (tqMolecule): Tequila molecule object defining the system.
        circuit (QCircuit, optional): Ansatz quantum circuit.
        variables (Variables, optional): Optimized parameters for the circuit.
        initial_state (QubitWaveFunction, optional): Initial state for the simulation.
        orb_a (int): Index of the first orbital.
        orb_b (int): Index of the second orbital.
        PSSR (bool): Apply Parity Number Superselection Rule.
        NSSR (bool): Apply Particle Number Superselection Rule.

    Returns:
        float: The classical correlation S(diag(rho_ab) || rho_a ⊗ rho_b).
    """
    rho_ab = compute_two_orb_rdm(mol, circuit, variables, initial_state, p_orb=orb_a, q_orb=orb_b, PSSR=PSSR, NSSR=NSSR)
    chi = np.diag(np.diag(rho_ab))

    o1 = compute_one_orb_rdm(mol, circuit, variables, initial_state, one_orb=orb_a)
    o2 = compute_one_orb_rdm(mol, circuit, variables, initial_state, one_orb=orb_b)
    pi = np.kron(o1,o2)
    pi = change_basis(pi, 'to_molecular')
        
    return quantum_relative_entropy(chi, pi)

def two_orbs_mutual_info(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, orb_a:int=0, orb_b:int=1, PSSR:bool=False, NSSR:bool=False)->float:
    """
    Compute the quantum mutual information between two molecular orbitals via relative entropy.

    The mutual information is defined as S(rho_ab || rho_a ⊗ rho_b), i.e. the relative
    entropy between the joint two-orbital RDM and the product of the individual marginals.
    This is equivalent to I(A:B) = S(rho_A) + S(rho_B) - S(rho_AB) but computed directly
    from the relative entropy formulation.

    Note: The product state pi = rho_a ⊗ rho_b is constructed in the computational basis
    via a Kronecker product and then transformed to the molecular basis before computing
    the relative entropy.

    Parameters:
        mol (tqMolecule): Tequila molecule object defining the system.
        circuit (QCircuit, optional): Ansatz quantum circuit.
        variables (Variables, optional): Optimized parameters for the circuit.
        initial_state (QubitWaveFunction, optional): Initial state for the simulation.
        orb_a (int): Index of the first orbital.
        orb_b (int): Index of the second orbital.
        PSSR (bool): Apply Parity Number Superselection Rule.
        NSSR (bool): Apply Particle Number Superselection Rule.

    Returns:
        float: The mutual information S(rho_ab || rho_a ⊗ rho_b).
    """
    rho_ab = compute_two_orb_rdm(mol, circuit, variables, initial_state, p_orb=orb_a, q_orb=orb_b, PSSR=PSSR, NSSR=NSSR)
    o1 = compute_one_orb_rdm(mol, circuit, variables, initial_state, one_orb=orb_a)
    o2 = compute_one_orb_rdm(mol, circuit, variables, initial_state, one_orb=orb_b)
    pi = np.kron(o1,o2)
    pi = change_basis(pi, 'to_molecular')
        
    return quantum_relative_entropy(rho_ab, pi)