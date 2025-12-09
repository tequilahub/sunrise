import tequila as tq
from tequila import QCircuit,QubitWaveFunction,QubitHamiltonian
from tequila.objective.objective import Variables
from tequila.quantumchemistry.qc_base import QuantumChemistryBase as tqMolecule
import numpy as np
from scipy.linalg import logm, eigh
import itertools
import math
from typing import List,Tuple,Union

class Input_State:
    """
    Wrapper class to hold the input state to measure the operators
    The input state can be a quantum circuit or a wavefunction
    """
    circuit = None
    variables = None
    wavefunction = None

    def __init__(self, circuit=None, variables=None, wavefunction=None):
        if circuit is None and wavefunction is None:
            raise ValueError("Either a circuit or a wavefunction must be provided")
        if circuit is not None and wavefunction is not None and wavefunction != 0:
            raise ValueError("Only one of circuit or wavefunction must be provided")
        
        self.circuit = circuit
        self.variables = variables
        self.wavefunction = wavefunction

    def get_circuit(self):
        """
        Get the circuit from the wavefunction or the circuit
        """
        if self.wavefunction is not None and self.circuit is None:
            self.circuit = tq.gates.Rz(angle=0, target=range(self.wavefunction.n_qubits)) # dummy empty circuit
        return self.circuit
        
    def get_wavefunction(self, variables=None):
        """
        Get the wavefunction from the circuit or the wavefunction
        """
        if self.circuit is not None and self.wavefunction is None:
            if variables is None:
                variables = self.variables
            self.wavefunction = tq.compile(self.circuit)(variables)
        return self.wavefunction

def compute_one_orb_rdm(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, one_orb:int=0)->np.ndarray:

    # Initialize the input state in a wrapper class
    state = Input_State(circuit=circuit, variables=variables, wavefunction=initial_state)
    if initial_state is not None and circuit is None:
        circuit = state.get_circuit()
        initial_state = state.get_wavefunction()
    elif circuit is None and initial_state is None:
        raise ValueError("Either a circuit or a wavefunction must be provided")
    up = mol.transformation.up(one_orb)
    down = mol.transformation.down(one_orb)
    ops = {
        "vacuum": tq.paulis.I(),
        "a_pu_dag": mol.make_creation_op(up),
        "a_pu": mol.make_annihilation_op(up),
        "n_pu": mol.make_number_op(up),
        "a_pd_dag": mol.make_creation_op(down),
        "a_pd": mol.make_annihilation_op(down),
        "n_pd": mol.make_number_op(down)
        }
    
    row = [
        ["vacuum"], # |00>
        ["a_pd_dag"], # |0down>
        ["a_pu_dag"], # |up0>
        ["a_pu_dag","a_pd_dag"] # |updown>
    ]

    # Create column as the daggered row
    column = []
    for tmp_ops in row:
        col_ops = []
        for op in tmp_ops:
            if op.endswith("_dag"):
                col_ops.append(op[:-4]) # removes the last 4 characters "_dag"
            else: 
                col_ops.append(op)
        # reverse as we are daggering
        col_ops.reverse()
        column.append(col_ops)

    rho = np.zeros((4,4))
    for i,state_i in enumerate(row):
        for j,state_j in enumerate(column):
            if i==j: # Select only specific terms as in https://iopscience.iop.org/article/10.1088/2058-9565/aca4ee/meta Eq.(27) 
                     # but the order is spin-down first and then spin-up
                
                # for convenience we multiply individual operators before
                op_i = tq.paulis.I()
                for v in state_i:
                    op_i *= ops[v]
                op_j = tq.paulis.I()
                for v in state_j:
                    op_j *= ops[v]

                P = op_i * (1-ops["n_pu"])*(1-ops["n_pd"]) * op_j

                if P.is_hermitian():
                    rho[i][j] = tq.simulate(tq.ExpectationValue(circuit,P), variables=variables, initial_state=initial_state)
                else:
                    P_herm, P_non_herm = P.split()
                    rho[i][j] = tq.simulate(tq.ExpectationValue(circuit,P_herm), variables=variables, initial_state=initial_state) + \
                        tq.simulate(tq.ExpectationValue(circuit,-1j * P_non_herm), variables=variables, initial_state=initial_state)

    return rho

def compute_two_orb_rdm(mol:tqMolecule, circuit:QCircuit=None, variables:Variables=None, initial_state:QubitWaveFunction=None, p_orb:int=0, q_orb:int=1, PSSR:bool=False, NSSR:bool=False)->np.ndarray:

    # Initialize the input state in a wrapper class
    state = Input_State(circuit=circuit, variables=variables, wavefunction=initial_state)
    if initial_state is not None and circuit is None:
        circuit = state.get_circuit()
        initial_state = state.get_wavefunction()
    elif circuit is None and initial_state is None:
        raise ValueError("Either a circuit or a wavefunction must be provided")
    
    pup = mol.transformation.up(p_orb)
    pdown = mol.transformation.down(p_orb)
    qup = mol.transformation.up(q_orb)
    qdown = mol.transformation.down(q_orb)

    ops = {
        "vacuum": tq.paulis.I(),
        "a_pu_dag": mol.make_creation_op(pup),
        "a_pu": mol.make_annihilation_op(pup),
        "n_pu": mol.make_number_op(pup),
        "a_pd_dag": mol.make_creation_op(pdown),
        "a_pd": mol.make_annihilation_op(pdown),
        "n_pd": mol.make_number_op(pdown),
        "a_qu_dag": mol.make_creation_op(qup),
        "a_qu": mol.make_annihilation_op(qup),
        "n_qu": mol.make_number_op(qup),
        "a_qd_dag": mol.make_creation_op(qdown),
        "a_qd": mol.make_annihilation_op(qdown),
        "n_qd": mol.make_number_op(qdown)
    }

    # For row and column order refer to https://pubs.acs.org/doi/10.1021/acs.jctc.0c00559 Figure 4
    row = [
        ["vacuum"], # |0000>
        ["a_qd_dag"], ["a_pd_dag"], # |000down>, |0down00>
        ["a_qu_dag"], ["a_pu_dag"], # |00up0>, |up000>
        ["a_pd_dag","a_qd_dag"], # |0down0down>
        ["a_qu_dag","a_qd_dag"], ["a_pd_dag","a_qu_dag"], ["a_pu_dag","a_qd_dag"], ["a_pu_dag","a_pd_dag"], # |00updown>, |0downup0>, |up00down>, |updown00>
        ["a_pu_dag","a_qu_dag"], # |up0up0>
        ["a_pd_dag","a_qu_dag","a_qd_dag"], ["a_pu_dag","a_pd_dag","a_qd_dag"], # |0downupdown>, |updown0down>
        ["a_pu_dag","a_qu_dag","a_qd_dag"], ["a_pu_dag","a_pd_dag","a_qu_dag"], # |up0updown>, |updownup0>
        ["a_pu_dag","a_pd_dag","a_qu_dag","a_qd_dag"] # |updownupdown>
    ]
    # Create column as the daggered row
    column = []
    for tmp_ops in row:
        col_ops = []
        for op in tmp_ops:
            if op.endswith("_dag"):
                col_ops.append(op[:-4]) # removes the last 4 characters "_dag"
            else: 
                col_ops.append(op)
        # reverse as we are daggering
        col_ops.reverse()
        column.append(col_ops)

    rho = np.zeros((16,16)) # 16x16 matrix
    for i,state_i in enumerate(row):
        for j,state_j in enumerate(column):
            if (i == j or 
                (i in {1, 2} and j in {1, 2}) or
                (i in {3, 4} and j in {3, 4}) or
                (i in {6, 7, 8, 9} and j in {6, 7, 8, 9}) or
                (i in {11, 12} and j in {11, 12}) or
                (i in {13, 14} and j in {13, 14})): # Select only specific terms as in https://pubs.acs.org/doi/10.1021/acs.jctc.0c00559 Figure 4

                # In order to evaluate SSR we make a local number counter for p and q on state i and state j
                N_p_i = N_q_i = N_p_j = N_q_j = 0
                for v,w in zip(state_i,state_j):
                    if v.startswith("a_p"):
                        N_p_i += 1
                    elif v.startswith("a_q"):
                        N_q_i += 1
                    if w.startswith("a_p"):
                        N_p_j += 1
                    elif w.startswith("a_q"):
                        N_q_j += 1
                # P-SSR (odd or even number of particles on local orbitals)
                if PSSR and (N_p_i%2!=N_p_j%2 or N_q_i%2!=N_q_j%2):
                    continue
                # N-SSR (same number of particles on local orbitals)
                if NSSR and (N_p_i!=N_p_j or N_q_i!=N_q_j):
                    continue

                # for convenience we multiply individual operators before
                op_i = tq.paulis.I()
                for v in state_i:
                    op_i *= ops[v]
                op_j = tq.paulis.I()
                for v in state_j:
                    op_j *= ops[v]

                P = op_i * (1-ops["n_pu"])*(1-ops["n_pd"])*(1-ops["n_qu"])*(1-ops["n_qd"]) * op_j
                if P.is_hermitian():
                    rho[i][j] = tq.simulate(tq.ExpectationValue(circuit,P), variables=variables, initial_state=initial_state)
                else:
                    P_herm, P_non_herm = P.split()
                    rho[i][j] = tq.simulate(tq.ExpectationValue(circuit,P_herm), variables=variables, initial_state=initial_state) + \
                        tq.simulate(tq.ExpectationValue(circuit,-1j * P_non_herm), variables=variables, initial_state=initial_state)

    return rho

def outer(psi):
    """Helper: Computes |psi><psi|."""
    return np.outer(psi, psi.conj())

def normalize_vector(psi):
    """Helper: Normalizes a vector if needed."""
    norm = np.linalg.norm(psi)
    if norm > 1e-9 and not np.isclose(norm, 1.0):
        return psi / norm
    return psi

def create_product_state(psi_A, psi_B):
    """
    Returns the density matrix for a pure product state:
    rho = |psi_A><psi_A| (tensor) |psi_B><psi_B|
    """
    # Normalize inputs
    psi_A = normalize_vector(psi_A)
    psi_B = normalize_vector(psi_B)
    
    rho_A = outer(psi_A)
    rho_B = outer(psi_B)
    return np.kron(rho_A, rho_B)

def create_separable_mixed_state(components, reorder_indices=None):
    """
    Returns a Separable Mixed State (convex sum of products).
    
    Args:
        components: List of tuples (probability, psi_A, psi_B)
        reorder_indices: (Optional) List of indices to permute the final matrix.
        
    Formula:
        rho = Sum( p_i * (|psiA_i><psiA_i| tensor |psiB_i><psiB_i|) )
    """
    if not components:
        raise ValueError("Components list is empty.")

    # 1. Normalize Probabilities
    probs = [c[0] for c in components]
    total_prob = sum(probs)
    
    if total_prob <= 0:
        raise ValueError("Sum of probabilities must be positive.")
        
    if not np.isclose(total_prob, 1.0):
        # Create a new list with normalized probabilities to avoid modifying input in place
        components = [(p / total_prob, psi_A, psi_B) for p, psi_A, psi_B in components]

    # Infer dimension from first element to initialize matrix
    _, pA_0, pB_0 = components[0]
    dim_A = len(pA_0)
    dim_B = len(pB_0)
    total_dim = dim_A * dim_B
    
    rho_total = np.zeros((total_dim, total_dim))
    
    for p, psi_A, psi_B in components:
        # 2. Normalize State Vectors
        psi_A = normalize_vector(psi_A)
        psi_B = normalize_vector(psi_B)
        
        # Create local density matrices
        rho_A = outer(psi_A)
        rho_B = outer(psi_B)
        
        # Tensor them together
        product_rho = np.kron(rho_A, rho_B)
        
        # Add weighted term
        rho_total += p * product_rho
    
    # Apply permutation if requested
    if reorder_indices is not None:
        if len(reorder_indices) != total_dim:
            raise ValueError(f"Permutation order length ({len(reorder_indices)}) "
                             f"must match matrix dimension ({total_dim}).")
        # Apply np.ix_ to reorder rows and columns simultaneously
        rho_total = rho_total[np.ix_(reorder_indices, reorder_indices)]
        
    return rho_total

def create_general_mixed_state(components, reorder_indices=None):
    """
    Returns a General Mixed State (sum of global pure states).
    This allows for mixing Entangled states.
    
    Args:
        components: List of tuples (probability, psi_AB)
        reorder_indices: (Optional) List of indices to permute the final matrix.
                         Useful for changing basis (e.g., to particle number basis).
        Note: psi_AB is a vector in the FULL Hilbert space.
        
    Formula:
        rho = Sum( p_i * |psiAB_i><psiAB_i| )
    """
    if not components:
        raise ValueError("Components list is empty.")

    # 1. Normalize Probabilities
    probs = [c[0] for c in components]
    total_prob = sum(probs)
    
    if total_prob <= 0:
        raise ValueError("Sum of probabilities must be positive.")
        
    if not np.isclose(total_prob, 1.0):
        components = [(p / total_prob, psi_AB) for p, psi_AB in components]

    # Infer dimension
    _, psi_0 = components[0]
    dim_total = len(psi_0)
    
    rho_total = np.zeros((dim_total, dim_total))
    
    for p, psi_AB in components:
        # 2. Normalize State Vector
        psi_AB = normalize_vector(psi_AB)
        
        # Create global density matrix term
        rho_term = outer(psi_AB)
        
        rho_total += p * rho_term
    
    # Apply permutation if requested
    if reorder_indices is not None:
        if len(reorder_indices) != dim_total:
            raise ValueError(f"Permutation order length ({len(reorder_indices)}) "
                             f"must match matrix dimension ({dim_total}).")
        # Apply np.ix_ to reorder rows and columns simultaneously
        rho_total = rho_total[np.ix_(reorder_indices, reorder_indices)]
        
    return rho_total

def change_basis(rho, direction='to_molecular'):
    """
    Switches matrix between Computational Kronecker Basis and Molecular Particle-Number Basis.
    
    Args:
        rho: The density matrix (16x16)
        direction: 'to_molecular' (Computational -> Molecular)
                   'to_computational'  (Molecular -> Computational)
    """
    MOLECULAR_PERM_ORDER = [0, 1, 4, 2, 8, 5, 3, 6, 9, 12, 10, 7, 13, 11, 14, 15]
    if direction == 'to_molecular':
        idx = MOLECULAR_PERM_ORDER
    elif direction == 'to_computational':
        idx = np.argsort(MOLECULAR_PERM_ORDER).tolist()
    else:
        raise ValueError("Direction must be 'to_molecular' or 'to_computational'")
        
    # Apply permutation to both rows and columns
    return rho[np.ix_(idx, idx)]

def get_reduced_states(rho, d_A, d_B):
    """Calculates reduced density matrices rho_A and rho_B."""
    # Reshape to (dim_A, dim_B, dim_A, dim_B)
    rho_tensor = rho.reshape(d_A, d_B, d_A, d_B)
    rho_A = np.trace(rho_tensor, axis1=1, axis2=3)
    rho_B = np.trace(rho_tensor, axis1=0, axis2=2)
    return rho_A, rho_B

def is_entangled_ppt(rho, d_A, d_B, tol=1e-9):
    """Peres-Horodecki Criterion (Partial Transpose)."""
    # Transpose B indices (1 and 3)
    rho_tensor = rho.reshape(d_A, d_B, d_A, d_B)
    rho_pt = rho_tensor.transpose(0, 3, 2, 1).reshape(d_A*d_B, d_A*d_B)
    min_eig = np.linalg.eigvalsh(rho_pt).min()
    return min_eig < -tol

def classify_7_categories(rho, tol=1e-7):
    rho = np.array(rho)
    dim = rho.shape[0]
    # Assume equal partition for molecular orbitals (e.g., 4x4 -> 2x2, 16x16 -> 4x4)
    d = int(np.sqrt(dim)) 
    
    # 1. Purity Check
    purity = np.trace(rho @ rho).real
    is_pure = np.abs(purity - 1.0) < tol
    
    rho_A, rho_B = get_reduced_states(rho, d, d)
    
    # --- BRANCH: PURE STATES ---
    if is_pure:
        # Check if subsystem is pure (Trace(rho_A^2) == 1)
        purity_A = np.trace(rho_A @ rho_A).real
        if np.abs(purity_A - 1.0) < tol:
            return "Pure Product State (Uncorrelated)"
        else:
            return "Pure Entangled State"

    # --- BRANCH: MIXED STATES ---
    else:
        # Check Entanglement
        if is_entangled_ppt(rho, d, d, tol):
            return "Mixed Entangled State"
        
        # If Separable:
        # Check for Product State (Uncorrelated)
        rho_product = np.kron(rho_A, rho_B)
        if np.linalg.norm(rho - rho_product) < tol:
            return "Mixed Product State (Uncorrelated Noise)"
            
        # Check Commutativity for Classical/Quantum correlations
        # [rho, rho_A x I]
        term_A = np.kron(rho_A, np.eye(d))
        comm_A = np.linalg.norm(rho @ term_A - term_A @ rho) < tol
        
        # [rho, I x rho_B]
        term_B = np.kron(np.eye(d), rho_B)
        comm_B = np.linalg.norm(rho @ term_B - term_B @ rho) < tol
        
        if comm_A and comm_B:
            return "Classically Correlated Separable State (CC)"
        elif comm_A or comm_B:
            return "Classical-Quantum State (CQ)"
        else:
            return "Discordant State (Separable but Quantum)"

def classify_hierarchy(rho, tol=1e-7):
    """
    Classifies according to the nested sets in Figure 1:
    D0 (Black) subset D_cl (Red) subset D_sep (Blue) subset All
    """
    rho = np.array(rho)
    d = int(np.sqrt(rho.shape[0]))
    
    rho_A, rho_B = get_reduced_states(rho, d, d)

    # 1. Check Entanglement (Outside D_sep)
    # The Gray Zone
    if is_entangled_ppt(rho, d, d, tol):
        return "Entangled (Outside D_sep)"

    # 2. Check Uncorrelated (D_0) - The Core
    # The Black Line
    rho_product = np.kron(rho_A, rho_B)
    if np.linalg.norm(rho - rho_product) < tol:
        return "D_0 (Uncorrelated / Product)"

    # 3. Check Classically Correlated (D_cl) - The Red Zone
    # Definition: Diagonal in the product basis of its marginals.
    # Mathematical test: Commutes with local reduced states.
    term_A = np.kron(rho_A, np.eye(d))
    term_B = np.kron(np.eye(d), rho_B)
    
    comm_A = np.linalg.norm(rho @ term_A - term_A @ rho) < tol
    comm_B = np.linalg.norm(rho @ term_B - term_B @ rho) < tol
    
    if comm_A and comm_B:
        return "D_cl (Classically Correlated - Red Zone)"

    # 4. Separable but Quantum (D_sep \ D_cl) - The Blue Zone
    return "D_sep (Separable with Quantum Discord - Blue Zone)"