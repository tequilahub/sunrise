"""Entropies and correlation measures between molecular orbitals (mutual information, entanglement, relative-entropy measures)."""

import numpy as np
import tequila as tq
from scipy.linalg import eigh
from scipy.optimize import minimize as scp_min
from tequila import QCircuit, QubitWaveFunction
from tequila.objective.objective import Variables

from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase
from sunrise.orbital_correlation.density_matrix_utils import *


# Quantum entropy S(rho)
def quantum_entropy(rho: np.ndarray) -> float:
    """Compute the von Neumann entropy S(rho) = -Tr(rho log rho).

    Parameters
    ----------
    rho : numpy.ndarray
        Density matrix (Hermitian, positive semidefinite, trace 1).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If rho is not Hermitian, has trace different from 1, or is not positive semidefinite.
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
def quantum_relative_entropy(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute the quantum relative entropy S(rho || sigma) = Tr(rho (log rho - log sigma)).

    Parameters
    ----------
    rho, sigma : numpy.ndarray
        Density matrices (Hermitian, positive semidefinite, trace 1).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If rho or sigma is not a valid density matrix.

    Examples
    --------
    >>> from sunrise.orbital_correlation.quantum_info_utils import quantum_relative_entropy
    >>> S = quantum_relative_entropy(rho, sigma)
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


def mutual_info_simple(
    mol: QuantumChemistryBase,
    circuit: QCircuit = None,
    variables: Variables = None,
    initial_state: QubitWaveFunction = None,
    orb_a: int = 0,
    orb_b: int = 1,
    PSSR: bool = False,
    NSSR: bool = False,
) -> float:
    """Compute the mutual information I(A:B) = S(rho_A) + S(rho_B) - S(rho_AB) between two orbitals.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule defining the orbitals and the fermion-to-qubit encoding.
    circuit : tequila.QCircuit, optional
        Circuit preparing the state. Provide it or ``initial_state``.
    variables : tequila.objective.objective.Variables, optional
        Values of the circuit variables.
    initial_state : tequila.QubitWaveFunction, optional
        Wavefunction to measure when no circuit is given (or to start the circuit from).
    orb_a, orb_b : int
        Indices of the two spatial orbitals.
    PSSR : bool
        If True, apply the parity superselection rule when computing the two-orbital RDM.
    NSSR : bool
        If True, apply the particle-number superselection rule when computing the two-orbital RDM.

    Returns
    -------
    float

    Examples
    --------
    >>> mutual_info_simple(mol, U, orb_a=0, orb_b=1)
    """
    rho_a = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_a)
    S_a = quantum_entropy(rho_a)
    rho_b = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_b)
    S_b = quantum_entropy(rho_b)
    rho_ab = compute_two_orb_rdm(
        mol,
        circuit,
        variables,
        initial_state,
        p_orb=orb_a,
        q_orb=orb_b,
        PSSR=PSSR,
        NSSR=NSSR,
    )
    S_ab = quantum_entropy(rho_ab)

    return S_a + S_b - S_ab  # there might be a 0.5 depending to convention


def one_orb_mutual_info(
    mol: QuantumChemistryBase,
    circuit: QCircuit = None,
    variables: Variables = None,
    initial_state: QubitWaveFunction = None,
    orb_a: int = 0,
    PSSR: bool = False,
    NSSR: bool = False,
):
    """Compute the single-orbital mutual information I_i of an orbital.

    The default is twice the single-orbital entanglement (``one_orb_entanglement``). With PSSR or
    NSSR the eigenvalues of the one-orbital RDM are grouped as allowed by the superselection rule
    (see Eq. 29 of https://doi.org/10.1021/acs.jctc.0c00559). PSSR takes precedence over NSSR.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule defining the orbitals and the fermion-to-qubit encoding.
    circuit : tequila.QCircuit, optional
        Circuit preparing the state. Provide it or ``initial_state``.
    variables : tequila.objective.objective.Variables, optional
        Values of the circuit variables.
    initial_state : tequila.QubitWaveFunction, optional
        Wavefunction to measure when no circuit is given (or to start the circuit from).
    orb_a : int
        Index of the orbital.
    PSSR : bool
        If True, apply the parity superselection rule.
    NSSR : bool
        If True, apply the particle-number superselection rule.

    Returns
    -------
    float

    Examples
    --------
    >>> from sunrise.orbital_correlation.quantum_info_utils import one_orb_mutual_info
    >>> I = one_orb_mutual_info(mol, U, orb_a=0)
    """
    rho_a = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_a)
    if PSSR:
        rho_a_evals, _rho_a_evecs = eigh(rho_a)
        I = (
            (rho_a_evals[0] + rho_a_evals[3]) * np.log(rho_a_evals[0] + rho_a_evals[3])
            + (rho_a_evals[1] + rho_a_evals[2])
            * np.log(rho_a_evals[1] + rho_a_evals[2])
            - 2
            * (
                rho_a_evals[0] * np.log(rho_a_evals[0])
                + rho_a_evals[1] * np.log(rho_a_evals[1])
                + rho_a_evals[2] * np.log(rho_a_evals[2])
                + rho_a_evals[3] * np.log(rho_a_evals[3])
            )
        )
    elif NSSR:
        rho_a_evals, _rho_a_evecs = eigh(rho_a)
        I = (
            rho_a_evals[0] * np.log(rho_a_evals[0])
            + (rho_a_evals[1] + rho_a_evals[2])
            * np.log(rho_a_evals[1] + rho_a_evals[2])
            + rho_a_evals[3] * np.log(rho_a_evals[3])
            - 2
            * (
                rho_a_evals[0] * np.log(rho_a_evals[0])
                + rho_a_evals[1] * np.log(rho_a_evals[1])
                + rho_a_evals[2] * np.log(rho_a_evals[2])
                + rho_a_evals[3] * np.log(rho_a_evals[3])
            )
        )
    else:
        I = 2 * one_orb_entanglement(
            mol, circuit, variables, initial_state, orb_a=orb_a
        )

    return I


def total_mutual_info(
    mol,
    circuit=None,
    variables=None,
    initial_state=0,
    orbs=None,
    PSSR=False,
    NSSR=False,
):
    """Compute the total mutual information of orbitals, sum_i S(rho_i) - S(rho).

    The entropy S(rho) is that of the full state of the circuit, obtained by building the whole
    density matrix, so it is only feasible for small systems.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule defining the orbitals.
    circuit : tequila.QCircuit
        Circuit preparing the state.
    variables : tequila.objective.objective.Variables, optional
        Values of the circuit variables.
    initial_state : int or tequila.QubitWaveFunction
        Initial state of the circuit. Default is 0.
    orbs : list of int, optional
        Orbitals to include. Default is all orbitals.
    PSSR, NSSR : bool
        Unused.

    Returns
    -------
    float
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


def one_orb_entanglement(
    mol: QuantumChemistryBase,
    circuit: QCircuit = None,
    variables: Variables = None,
    initial_state: QubitWaveFunction = None,
    orb_a: int = 0,
    PSSR: bool = False,
    NSSR: bool = False,
) -> float:
    """Compute the single-orbital entanglement entropy E_i of an orbital.

    The default is the von Neumann entropy of the one-orbital RDM. With PSSR or NSSR the
    eigenvalues are grouped as allowed by the superselection rule (see Eq. 29 of
    https://doi.org/10.1021/acs.jctc.0c00559). PSSR takes precedence over NSSR.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule defining the orbitals and the fermion-to-qubit encoding.
    circuit : tequila.QCircuit, optional
        Circuit preparing the state. Provide it or ``initial_state``.
    variables : tequila.objective.objective.Variables, optional
        Values of the circuit variables.
    initial_state : tequila.QubitWaveFunction, optional
        Wavefunction to measure when no circuit is given (or to start the circuit from).
    orb_a : int
        Index of the orbital.
    PSSR : bool
        If True, apply the parity superselection rule.
    NSSR : bool
        If True, apply the particle-number superselection rule.

    Returns
    -------
    float

    Examples
    --------
    >>> one_orb_entanglement(mol, U, orb_a=0)
    """
    rho_a = compute_one_orb_rdm(mol, circuit, variables, initial_state, orb_a)
    if PSSR:
        rho_a_evals, _rho_a_evecs = eigh(rho_a)
        # Eq.(29) https://doi.org/10.1021/acs.jctc.0c00559
        plnp = [
            (rho_a_evals[i] + rho_a_evals[-1 - i])
            * np.log(rho_a_evals[i] + rho_a_evals[-1 - i])
            if not np.isclose(rho_a_evals[i] + rho_a_evals[-1 - i], 0, atol=1.0e-6)
            else 0.0
            for i in range(len(rho_a_evals) // 2)
        ]
        xlnx = [
            i * np.log(i) if not np.isclose(i, 0, atol=1.0e-6) else 0.0
            for i in rho_a_evals
        ]
        E = (plnp[0] + plnp[1]) - (xlnx[0] + xlnx[1] + xlnx[2] + xlnx[3])
    elif NSSR:
        rho_a_evals, _rho_a_evecs = eigh(rho_a)
        # Eq.(29) https://doi.org/10.1021/acs.jctc.0c00559
        xlnx = [
            i * np.log(i) if not np.isclose(i, 0, atol=1.0e-6) else 0.0
            for i in rho_a_evals
        ]
        E = (
            (rho_a_evals[1] + rho_a_evals[2]) * np.log(rho_a_evals[1] + rho_a_evals[2])
            if not np.isclose(rho_a_evals[1] + rho_a_evals[2], 0, atol=1.0e-6)
            else 0.0
        )
        E -= xlnx[1] + xlnx[2]
    else:
        S_a = quantum_entropy(rho_a)
        E = S_a

    return E


def func(x, d, rho):
    """Objective of the relative entropy of entanglement: S(rho || sigma) for a parametrized separable sigma.

    sigma is a convex combination of d product states built from ``x``::

        sigma = sum_k p_k |psi_A_k><psi_A_k| (x) |psi_B_k><psi_B_k|,   p_k ~ |v_A_k|^2 |v_B_k|^2

    where ``psi`` are the normalized component vectors ``v`` and the weights ``p_k`` are normalized
    over all components. It is transformed to the molecular basis before the
    comparison with rho.

    Parameters
    ----------
    x : numpy.ndarray
        Flat vector of shape (2*d*4,), d pairs of 4-dimensional vectors (subsystems A and B).
    d : int
        Number of product-state components.
    rho : numpy.ndarray
        Target density matrix (16x16) in the molecular basis.

    Returns
    -------
    float
    """
    x_reshaped = x.reshape(2 * d, 4)
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
    sigma = change_basis(sigma, "to_molecular")

    return quantum_relative_entropy(rho=rho, sigma=sigma)


def relative_entropy_gradient_matrix(rho, sigma, epsilon=1e-12):
    """Compute the gradient matrix G of -Tr(rho log sigma) with respect to sigma.

    The derivative along dSigma is Tr(G dSigma). The Daleckii-Krein formula handles degenerate
    and non-degenerate spectra of sigma.

    Parameters
    ----------
    rho, sigma : numpy.ndarray
        Density matrices. Eigenvalues of sigma are clipped at ``epsilon``.
    epsilon : float
        Clipping value and tolerance to detect degenerate eigenvalues.

    Returns
    -------
    numpy.ndarray
        Real matrix with the shape of rho.
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
    """Compute the analytic gradient of ``func`` with respect to ``x``.

    A forward pass rebuilds sigma from ``x``; the gradient is then propagated back through the
    relative entropy with the matrix gradient of -Tr(rho log sigma) (see
    ``relative_entropy_gradient_matrix``) and the chain rule through the norm-weighted
    parameterization of the separable state. Each component k contributes twice: through the
    direction of its vectors (psi_A_k, psi_B_k), which sets its projector, and through their
    norm, which sets its weight p_k.

    Parameters
    ----------
    x : numpy.ndarray
        Flat vector of shape (2*d*4,), as in ``func``.
    d : int
        Number of product-state components.
    rho : numpy.ndarray
        Target density matrix (16x16) in the molecular basis.

    Returns
    -------
    numpy.ndarray
        Gradient of the same shape as ``x``.
    """
    x_reshaped = x.reshape(2 * d, -1)

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
    sigma_mol = change_basis(sigma_sep, "to_molecular")

    # --- 2. Matrix Gradient in Molecular Basis ---
    grad_sigma_mol = relative_entropy_gradient_matrix(rho, sigma_mol)

    # --- 3. Transform Gradient back to Computational Basis ---
    grad_sigma_comp = change_basis(grad_sigma_mol, "to_computational")

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
        g_PA = (
            probs[k]
            * np.einsum("ikjl,k,l->ij", G_tensor, psi_B[k], psi_B[k].conj()).real
        )
        g_PB = (
            probs[k]
            * np.einsum("ikjl,i,j->kl", G_tensor, psi_A[k], psi_A[k].conj()).real
        )

        grad_projectors_A.append(g_PA)
        grad_projectors_B.append(g_PB)

        # Contract for Weight Gradients
        term = np.einsum("ij,ij->", g_PA, np.outer(psi_A[k], psi_A[k].conj())).real
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
        d_weight_dA = dL_dw[k] * 2 * nA[k] * (nB[k] ** 2)
        d_weight_dB = dL_dw[k] * 2 * nB[k] * (nA[k] ** 2)

        term2_A = d_weight_dA * psi_A[k]
        term2_B = d_weight_dB * psi_B[k]

        grad_x[2 * k] = term1_A + term2_A
        grad_x[2 * k + 1] = term1_B + term2_B

    return grad_x.flatten()


def get_classical_diagonal_guess(rho):
    r"""Build an initial guess for the separable-state optimization from the diagonal of rho.

    Each diagonal element above 1e-6 of rho in the tensor-product basis gives one product basis
    state :math:`|i_A\rangle \otimes |i_B\rangle` (index ``4 * i_A + i_B``) with that probability.

    Parameters
    ----------
    rho : numpy.ndarray
        Density matrix (16x16) in the molecular basis.

    Returns
    -------
    states : list of tuple
        ``(prob, vec_A, vec_B)`` of each component.
    x0 : numpy.ndarray
        Flat parameter vector for ``func`` and ``gradient_func``, with vectors scaled by sqrt(prob).
    """
    # 1. Define the 4 basis vectors for a single orbital
    basis_vectors = np.eye(4)

    states = []
    x0 = []

    # The input rho is in the Molecular Basis.
    # We must rotate it to Computational Basis to interpret index k as (idx_A * 4 + idx_B).
    rho_computational = change_basis(rho, direction="to_computational")

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
    mol: QuantumChemistryBase,
    circuit: tq.QCircuit = None,
    variables=None,
    initial_state=None,
    orb_a: int = 0,
    orb_b: int = 1,
    PSSR: bool = False,
    NSSR: bool = False,
    min_components: int = 8,
    silent: bool = True,
) -> float:
    """Compute the relative entropy of entanglement E_R(rho_ab) of two orbitals.

    The two-orbital RDM is computed, then S(rho || sigma) is minimized over separable states
    sigma with L-BFGS-B and analytic gradients, starting from a classical diagonal guess padded
    with random components and noise. The result is therefore slightly random between calls.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule defining the orbitals and the fermion-to-qubit encoding.
    circuit : tequila.QCircuit, optional
        Circuit preparing the state. Provide it or ``initial_state``.
    variables : tequila.objective.objective.Variables, optional
        Values of the circuit variables.
    initial_state : tequila.QubitWaveFunction, optional
        Wavefunction to measure when no circuit is given (or to start the circuit from).
    orb_a, orb_b : int
        Indices of the two spatial orbitals.
    PSSR : bool
        If True, apply the parity superselection rule when computing the two-orbital RDM.
    NSSR : bool
        If True, apply the particle-number superselection rule when computing the two-orbital RDM.
    min_components : int
        Minimum number of product-state components, to avoid local minima of pure states.
    silent : bool
        If False, print the progress of the optimization.

    Returns
    -------
    float
        Minimized relative entropy.

    Examples
    --------
    >>> from sunrise.orbital_correlation.quantum_info_utils import two_orbs_entanglement
    >>> E = two_orbs_entanglement(mol, U, orb_a=0, orb_b=1)
    """

    # 1. Compute the Reduced Density Matrix (rho_ab)
    if not silent:
        print(f"Computing RDM for orbitals ({orb_a}, {orb_b})...")

    if circuit is not None and variables is not None:
        circuit = circuit.map_variables(variables)

    rho = compute_two_orb_rdm(
        mol,
        circuit,
        variables,
        initial_state,
        p_orb=orb_a,
        q_orb=orb_b,
        PSSR=PSSR,
        NSSR=NSSR,
    )

    # 2. Generate Initial Guess (Classical Diagonal)
    states0, x0 = get_classical_diagonal_guess(rho)
    current_d = len(states0)

    # 3. Robustness Padding
    # If the guess is too simple (e.g., just the HF state), add "empty" components
    # to allow the optimizer to find a mixed state solution.
    d = current_d
    if current_d < min_components:
        if not silent:
            print(
                f"  Padding components from {current_d} to {min_components} for robustness."
            )

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
        options={"maxiter": 2000, "ftol": 1e-9, "disp": False},
    )

    if not result.success and not silent:
        print(f"  Warning: Optimization finished with status: {result.message}")

    # Return the entanglement value
    return result.fun


def two_orbs_quantum_correlation(
    mol: QuantumChemistryBase,
    circuit: QCircuit = None,
    variables: Variables = None,
    initial_state: QubitWaveFunction = None,
    orb_a: int = 0,
    orb_b: int = 1,
    PSSR: bool = False,
    NSSR: bool = False,
) -> float:
    """Compute the quantum correlation S(rho_ab || chi) of two orbitals, with chi = diag(rho_ab).

    chi is the closest classical (diagonal) state, so this measures the non-classical coherence
    of the two-orbital RDM.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule defining the orbitals and the fermion-to-qubit encoding.
    circuit : tequila.QCircuit, optional
        Circuit preparing the state. Provide it or ``initial_state``.
    variables : tequila.objective.objective.Variables, optional
        Values of the circuit variables.
    initial_state : tequila.QubitWaveFunction, optional
        Wavefunction to measure when no circuit is given (or to start the circuit from).
    orb_a, orb_b : int
        Indices of the two spatial orbitals.
    PSSR : bool
        If True, apply the parity superselection rule when computing the two-orbital RDM.
    NSSR : bool
        If True, apply the particle-number superselection rule when computing the two-orbital RDM.

    Returns
    -------
    float

    Examples
    --------
    >>> from sunrise.orbital_correlation.quantum_info_utils import two_orbs_quantum_correlation
    >>> Q = two_orbs_quantum_correlation(mol, U, orb_a=0, orb_b=1)
    """
    rho_ab = compute_two_orb_rdm(
        mol,
        circuit,
        variables,
        initial_state,
        p_orb=orb_a,
        q_orb=orb_b,
        PSSR=PSSR,
        NSSR=NSSR,
    )
    chi = np.diag(np.diag(rho_ab))

    return quantum_relative_entropy(rho_ab, chi)


def two_orbs_classical_correlation(
    mol: QuantumChemistryBase,
    circuit: QCircuit = None,
    variables: Variables = None,
    initial_state: QubitWaveFunction = None,
    orb_a: int = 0,
    orb_b: int = 1,
    PSSR: bool = False,
    NSSR: bool = False,
) -> float:
    """Compute the classical correlation S(chi || rho_a (x) rho_b) of two orbitals, with chi = diag(rho_ab).

    It measures the statistical dependence of the orbitals after dropping the coherences.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule defining the orbitals and the fermion-to-qubit encoding.
    circuit : tequila.QCircuit, optional
        Circuit preparing the state. Provide it or ``initial_state``.
    variables : tequila.objective.objective.Variables, optional
        Values of the circuit variables.
    initial_state : tequila.QubitWaveFunction, optional
        Wavefunction to measure when no circuit is given (or to start the circuit from).
    orb_a, orb_b : int
        Indices of the two spatial orbitals.
    PSSR : bool
        If True, apply the parity superselection rule when computing the two-orbital RDM.
    NSSR : bool
        If True, apply the particle-number superselection rule when computing the two-orbital RDM.

    Returns
    -------
    float

    Examples
    --------
    >>> from sunrise.orbital_correlation.quantum_info_utils import two_orbs_classical_correlation
    >>> C = two_orbs_classical_correlation(mol, U, orb_a=0, orb_b=1)
    """
    rho_ab = compute_two_orb_rdm(
        mol,
        circuit,
        variables,
        initial_state,
        p_orb=orb_a,
        q_orb=orb_b,
        PSSR=PSSR,
        NSSR=NSSR,
    )
    chi = np.diag(np.diag(rho_ab))

    o1 = compute_one_orb_rdm(mol, circuit, variables, initial_state, one_orb=orb_a)
    o2 = compute_one_orb_rdm(mol, circuit, variables, initial_state, one_orb=orb_b)
    pi = np.kron(o1, o2)
    pi = change_basis(pi, "to_molecular")

    return quantum_relative_entropy(chi, pi)


def two_orbs_mutual_info(
    mol: QuantumChemistryBase,
    circuit: QCircuit = None,
    variables: Variables = None,
    initial_state: QubitWaveFunction = None,
    orb_a: int = 0,
    orb_b: int = 1,
    PSSR: bool = False,
    NSSR: bool = False,
) -> float:
    """Compute the mutual information S(rho_ab || rho_a (x) rho_b) of two orbitals.

    Equivalent to ``mutual_info_simple``, i.e. I(A:B) = S(rho_a) + S(rho_b) - S(rho_ab), but computed
    as a relative entropy. The product state rho_a (x) rho_b is built in the computational basis
    (Kronecker product) and then transformed to the molecular basis.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule defining the orbitals and the fermion-to-qubit encoding.
    circuit : tequila.QCircuit, optional
        Circuit preparing the state. Provide it or ``initial_state``.
    variables : tequila.objective.objective.Variables, optional
        Values of the circuit variables.
    initial_state : tequila.QubitWaveFunction, optional
        Wavefunction to measure when no circuit is given (or to start the circuit from).
    orb_a, orb_b : int
        Indices of the two spatial orbitals.
    PSSR : bool
        If True, apply the parity superselection rule when computing the two-orbital RDM.
    NSSR : bool
        If True, apply the particle-number superselection rule when computing the two-orbital RDM.

    Returns
    -------
    float

    Examples
    --------
    >>> from sunrise.orbital_correlation.quantum_info_utils import two_orbs_mutual_info
    >>> I = two_orbs_mutual_info(mol, U, orb_a=0, orb_b=1)
    """
    rho_ab = compute_two_orb_rdm(
        mol,
        circuit,
        variables,
        initial_state,
        p_orb=orb_a,
        q_orb=orb_b,
        PSSR=PSSR,
        NSSR=NSSR,
    )
    o1 = compute_one_orb_rdm(mol, circuit, variables, initial_state, one_orb=orb_a)
    o2 = compute_one_orb_rdm(mol, circuit, variables, initial_state, one_orb=orb_b)
    pi = np.kron(o1, o2)
    pi = change_basis(pi, "to_molecular")

    return quantum_relative_entropy(rho_ab, pi)
