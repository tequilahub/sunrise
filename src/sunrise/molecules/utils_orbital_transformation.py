import numpy
from typing import Tuple
from copy import deepcopy
from tequila.quantumchemistry.qc_base import QuantumChemistryBase
from sunrise.molecules.hybrid_base import HybridBase
from sunrise.molecules.fermionic_base import FermionicBase

def orthogonalize(c:numpy.ndarray, s:numpy.ndarray) -> numpy.ndarray:
    """
    Symmetrically orthogonalize orbital coefficients.
    c: (basis_functions, orbitals)
    s: (basis_functions, basis_functions)
    """
    # 1. Compute the overlap of the current MOs: S' = C^T * S * C
    # This replaces your entire nested loop and inner() function.
    sprima = c.T @ s @ c
    # 2. Diagonalize S'
    lam_s, l_s = numpy.linalg.eigh(sprima)

    # Optional but recommended: Clip tiny negative eigenvalues due to numerical noise
    lam_s = numpy.maximum(lam_s, 1e-14)

    # 3. Construct (S')^{-1/2}
    # This is much faster/cleaner than inverting a full matrix
    lam_sqrt_inv = numpy.diag(1.0 / numpy.sqrt(lam_s))
    symm_orthog = l_s @ lam_sqrt_inv @ l_s.T

    # 4. Transform coefficients: C_new = C * (S')^{-1/2}
    jcoef = c @ symm_orthog

    return jcoef

def orthogonalize_active_space(c:numpy.ndarray, s:numpy.ndarray, frozen_idx:list[int], active_idx:list[int]) -> numpy.ndarray:
    """
    Symmetrically orthogonalize orbital coefficients defined by colums with indices 'active_idx' while keeping untouched those defined by 'frozen_idx'.
    c: (basis_functions, orbitals)
    s: (basis_functions, basis_functions)
    frozen_idx: list of orbitals to left untoched
    active_idx: list of orbitals to orthogonalize
    """
    c_f = c[:, frozen_idx]
    c_a = c[:, active_idx]

    sf = c_f.T @ s @ c_f
    cross = c_f.T @ s @ c_a

    proj = c_f @ numpy.linalg.solve(sf, cross)
    c_a_proj = c_a - proj

    sa = c_a_proj.T @ s @ c_a_proj
    e, U = numpy.linalg.eigh(sa)
    e = numpy.maximum(e, 1e-12)
    X = U @ numpy.diag(1.0 / numpy.sqrt(e)) @ U.T

    c_new = c.copy()
    c_new[:, active_idx] = c_a_proj @ X
    return c_new

def get_active(c_orig:numpy.ndarray, d_orig:numpy.ndarray, s:numpy.ndarray, active_idx_c:list[int]) -> list[int]:
    """
    Safely identifies active orbitals in d_orig by projecting them into the entire active subspace of c_orig.
    c_orig: original orbital matrix which will be frozen (typicall HF)
    d_orig: original orbital matrix to look for active w.r.t. c_orig (i.e. native orbital matrix or CLPO matrix before active space considerations)
    s: overlap_integrals
    active_idx_c: subspace from c_orig to  look for the active indices for d_orig
    """
    # 1. Extract the entire reference active space block from c_orig
    c_active = c_orig[:, active_idx_c]

    # 2. Compute the full overlap matrix between reference active and all d_orig orbitals
    # Shape will be (n_active_ref, n_total_orbitals_d)
    overlap_matrix = c_active.T @ s @ d_orig

    # 3. Sum of squares along the reference axis gives the total "active character"
    # Shape will be (n_total_orbitals_d,)
    active_weights = numpy.sum(overlap_matrix**2, axis=0)

    # 4. Sort all orbital indices of d_orig by weight in descending order
    sorted_d_indices = numpy.argsort(active_weights)[::-1]

    # 5. Select the top N orbitals that match the active space best
    chosen_active_idx = sorted_d_indices[:len(active_idx_c)]

    return sorted(chosen_active_idx)

def get_core(c_orig:numpy.ndarray, d_orig:numpy.ndarray, s:numpy.ndarray, active_idx_d:list[int]):
    """
    Given the active space indices of d_orig, finds which occupied orbitals in c_orig should be frozen (core orbitals).
    
    Parameters:
    -----------
    c_orig: original orbital matrix which will be frozen (typicall HF)
    d_orig: original orbital matrix to look for active w.r.t. c_orig (i.e. native orbital matrix or CLPO matrix before active space considerations)
    s: overlap_integrals
    active_idx_d: The indices of the active space orbitals in d_orig.
    """
    n_occ_c = d_orig.shape[1] - len(active_idx_d)

    # 1. Extract the active subspace block from d_orig
    d_active = d_orig[:, active_idx_d]

    # 2. Compute the overlap between all c_orig orbitals and the d_orig active subspace
    # Shape will be (n_total_orbitals_c, n_active_d)
    overlap_matrix = c_orig.T @ s @ d_active

    # 3. Sum of squares along the d_active axis gives the "active character" of each c_orig orbital
    active_weights = numpy.sum(overlap_matrix**2, axis=1)

    # 4. Sort the orbitals by their active weight in ASCENDING order
    # The orbitals with the LOWEST active weight are your core (frozen) orbitals!
    sorted_fr_indices = numpy.argsort(active_weights)

    chosen_fr_idx = sorted_fr_indices[:n_occ_c]

    return sorted(chosen_fr_idx)

def transform(modified:QuantumChemistryBase, original:QuantumChemistryBase, orbital_type:str = None) -> Tuple[QuantumChemistryBase, dict]:
    '''
    Procedure similar to what is done in use_native_orbitals but for arbitrary basis keeped insied modified. Keeps frozen orbitals canHF
    orthogonalized with the active modified ones
    Returns modified molecule with the core orbitals of the original one
    And a dictionary with the form {active_orbital_index_before:active_orbital_index_after}
    The frozen orbitals will always be the N first on the orbital matrix
    '''
    core = [i.idx_total for i in original.integral_manager.orbitals if i.idx is None]
    assert len(original.integral_manager.orbitals) == len(modified.integral_manager.orbitals)
    c_orig = original.integral_manager.orbital_coefficients.copy()
    d_orig = modified.integral_manager.orbital_coefficients.copy()
    s = original.integral_manager.overlap_integrals.copy()
    n_basis = c_orig.shape[0]
    active = get_active(c_orig, d_orig, s, [i.idx_total for i in original.integral_manager.orbitals if i.idx is not None])
    to_active =  [i for i in range(n_basis) if  i not in core]
    to_active = {active[i] : to_active[i] for i in range(len(active))}
    reference_orbitals = core.copy()
    i =0
    while len(reference_orbitals) < original.parameters.total_n_electrons//2:
        if i not in reference_orbitals:
            reference_orbitals.append(i)
        i += 1
    n_core = len(core)
    n_active = len(active)
    c_combined = numpy.zeros((n_basis, n_core + n_active))
    for i,idx in enumerate(core):
        c_combined[:, i] = c_orig[:, idx]
    for act_idx in active:
        c_combined[:, to_active[act_idx]] = d_orig[:, act_idx]
    jcoef = orthogonalize_active_space(c_combined, s,core, [*to_active.values()])
    ref = [i.idx_total for i in original.integral_manager.reference_orbitals if i not in original.integral_manager.active_reference_orbitals]
    ref.extend([i for i in range(n_basis) if  i not in core][:len(original.integral_manager.active_reference_orbitals)])
    integral_manager = modified.initialize_integral_manager(one_body_integrals=original.integral_manager.one_body_integrals,
                    two_body_integrals=original.integral_manager.two_body_integrals, constant_term=original.integral_manager.constant_term,
                    active_orbitals= [i for i in range(n_basis) if  i not in core], frozen_orbitals=core, orbital_coefficients=jcoef,
                    overlap_integrals=original.integral_manager.overlap_integrals, reference_orbitals=ref, orbital_type=orbital_type)
    parameters = deepcopy(original.parameters)
    if isinstance(modified, FermionicBase):
        return FermionicBase(parameters=parameters, integral_manager=integral_manager, fermionic_backend=modified.fermionic_backend), to_active
    elif isinstance(modified, HybridBase):
        return HybridBase(parameters=parameters, integral_manager=integral_manager, transformation=modified.transformation, select=modified.select, two_qubit=modified.two_qubit, condense=modified.condense), to_active
    return QuantumChemistryBase(parameters=parameters, integral_manager=integral_manager, transformation=modified.transformation), to_active
