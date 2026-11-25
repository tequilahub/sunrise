"""Orthogonalization and active-space matching of orbital coefficient matrices."""

from __future__ import annotations

from copy import deepcopy

import numpy

from sunrise.molecules.fermionic_base import FermionicBase
from sunrise.molecules.hybrid_base import HybridBase
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase


def orthogonalize(c: numpy.ndarray, s: numpy.ndarray) -> numpy.ndarray:
    """Symmetrically (Loewdin) orthogonalize orbital coefficients.

    Parameters
    ----------
    c : numpy.ndarray
        Orbital coefficients, shape (basis functions, orbitals).
    s : numpy.ndarray
        Overlap matrix of the basis functions.

    Returns
    -------
    numpy.ndarray
        Orthonormal orbital coefficients, same shape as ``c``.
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


def orthogonalize_active_space(
    c: numpy.ndarray, s: numpy.ndarray, frozen_idx: list[int], active_idx: list[int]
) -> numpy.ndarray:
    """Orthogonalize the active orbitals against the frozen ones and among themselves.

    Columns in ``frozen_idx`` are kept unchanged. Columns in ``active_idx`` are
    projected out of the frozen space and then symmetrically orthogonalized.
    The frozen orbitals are assumed to be orthonormal.

    Parameters
    ----------
    c : numpy.ndarray
        Orbital coefficients, shape (basis functions, orbitals).
    s : numpy.ndarray
        Overlap matrix of the basis functions.
    frozen_idx : list of int
        Columns of ``c`` to leave untouched.
    active_idx : list of int
        Columns of ``c`` to orthogonalize.

    Returns
    -------
    numpy.ndarray
        Copy of ``c`` with the active columns replaced.
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


def get_active(
    c_orig: numpy.ndarray,
    d_orig: numpy.ndarray,
    s: numpy.ndarray,
    active_idx_c: list[int],
) -> list[int]:
    """Find the orbitals of ``d_orig`` that best span the active space of ``c_orig``.

    Each orbital of ``d_orig`` is weighted by its squared projection onto the
    active orbitals of ``c_orig``; the highest-weight ones are chosen.

    Parameters
    ----------
    c_orig : numpy.ndarray
        Reference orbital coefficients (typically Hartree-Fock).
    d_orig : numpy.ndarray
        Orbital coefficients to select from (e.g. native or CLPO orbitals).
    s : numpy.ndarray
        Overlap matrix of the basis functions.
    active_idx_c : list of int
        Active orbital columns of ``c_orig``.

    Returns
    -------
    list of int
        Sorted column indices of ``d_orig``, as many as ``active_idx_c``.
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
    chosen_active_idx = sorted_d_indices[: len(active_idx_c)]

    return sorted(chosen_active_idx)


def get_core(
    c_orig: numpy.ndarray,
    d_orig: numpy.ndarray,
    s: numpy.ndarray,
    active_idx_d: list[int],
):
    """Find the orbitals of ``c_orig`` to freeze, given the active orbitals of ``d_orig``.

    The orbitals of ``c_orig`` with the smallest overlap with the active
    subspace of ``d_orig`` are chosen as core.

    Parameters
    ----------
    c_orig : numpy.ndarray
        Reference orbital coefficients (typically Hartree-Fock).
    d_orig : numpy.ndarray
        Orbital coefficients containing the active orbitals.
    s : numpy.ndarray
        Overlap matrix of the basis functions.
    active_idx_d : list of int
        Active orbital columns of ``d_orig``.

    Returns
    -------
    list of int
        Sorted column indices of ``c_orig`` to freeze; their number is the
        number of orbitals of ``d_orig`` minus ``len(active_idx_d)``.
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


def transform(
    modified: QuantumChemistryBase,
    original: QuantumChemistryBase,
    orbital_type: str | None = None,
    *args,
    **kwargs,
) -> tuple[QuantumChemistryBase, dict]:
    """Build a molecule that uses the orbitals of ``modified`` as active space and those of ``original`` as core.

    Similar to ``use_native_orbitals``, but for an arbitrary orbital basis stored
    in ``modified``. The frozen core orbitals of ``original`` are kept and the
    active orbitals of ``modified`` are orthogonalized against them. No new
    Hartree-Fock calculation is run, so this works for any backend.

    Parameters
    ----------
    modified : QuantumChemistryBase
        Molecule whose orbital coefficients define the new active orbitals.
    original : QuantumChemistryBase
        Reference (Hartree-Fock) molecule providing integrals, parameters and core orbitals.
    orbital_type : str, optional
        Label shown when printing the molecule.
    **kwargs
        ``core`` and ``active`` (list of int, indices in the full orbital space, not the active
        one) override the automatic choice. Default: the core is the frozen orbitals of
        ``original``. With only ``core``, the active orbitals of ``modified`` are matched to the
        other orbitals of ``original`` (``get_active``); with only ``active`` (indices in
        ``modified``), the core is chosen from ``original`` (``get_core``).

    Returns
    -------
    tuple
        The new molecule (same class as ``modified``; the core orbitals keep their positions in
        the orbital matrix and the active ones fill the others) and a dict
        ``{active orbital index in modified: index in the new molecule}``.
    """
    assert len(original.integral_manager.orbitals) == len(
        modified.integral_manager.orbitals
    )
    c_orig = original.integral_manager.orbital_coefficients.copy()
    d_orig = modified.integral_manager.orbital_coefficients.copy()
    s = original.integral_manager.overlap_integrals.copy()
    n_basis = c_orig.shape[0]
    active = kwargs.pop("active", None)
    core = kwargs.pop("core", None)
    if active is None and core is None:
        core = [
            i.idx_total for i in original.integral_manager.orbitals if i.idx is None
        ]
        active = get_active(
            c_orig,
            d_orig,
            s,
            [
                i.idx_total
                for i in original.integral_manager.orbitals
                if i.idx is not None
            ],
        )
    elif active is None and core is not None:
        # core given: match in modified the complement of core in original
        active = get_active(
            c_orig, d_orig, s, [i for i in range(c_orig.shape[1]) if i not in core]
        )
    elif active is not None and core is None:
        # active given (indices of modified): pick core from original
        core = get_core(c_orig, d_orig, s, active)
    assert len(core) + len(active) == len(original.integral_manager.orbitals)
    core = list(core)
    active = list(active)
    to_active = [i for i in range(n_basis) if i not in core]
    to_active = {active[i]: to_active[i] for i in range(len(active))}
    reference_orbitals = core.copy()
    i = 0
    while len(reference_orbitals) < original.parameters.total_n_electrons // 2:
        if i not in reference_orbitals:
            reference_orbitals.append(i)
        i += 1
    n_core = len(core)
    n_active = len(active)
    c_combined = numpy.zeros((n_basis, n_core + n_active))
    for idx in core:
        c_combined[:, idx] = c_orig[:, idx]
    for act_idx in active:
        c_combined[:, to_active[act_idx]] = d_orig[:, act_idx]
    jcoef = orthogonalize_active_space(c_combined, s, core, [*to_active.values()])
    occ = [i.idx_total for i in original.integral_manager.reference_orbitals]
    ref = [i for i in core if i in occ]
    ref.extend([i for i in range(n_basis) if i not in core][: len(occ) - len(ref)])
    integral_manager = modified.initialize_integral_manager(
        one_body_integrals=original.integral_manager.one_body_integrals,
        two_body_integrals=original.integral_manager.two_body_integrals,
        constant_term=original.integral_manager.constant_term,
        active_orbitals=[i for i in range(n_basis) if i not in core],
        frozen_orbitals=core,
        orbital_coefficients=jcoef,
        overlap_integrals=original.integral_manager.overlap_integrals,
        reference_orbitals=ref,
        orbital_type=orbital_type,
    )
    parameters = deepcopy(original.parameters)
    if isinstance(modified, FermionicBase):
        return FermionicBase(
            parameters=parameters,
            integral_manager=integral_manager,
            fermionic_backend=modified.fermionic_backend,
        ), to_active
    elif isinstance(modified, HybridBase):
        return HybridBase(
            parameters=parameters,
            integral_manager=integral_manager,
            transformation=modified.transformation,
            select=modified.select,
            two_qubit=modified.two_qubit,
            condense=modified.condense,
        ), to_active
    return QuantumChemistryBase(
        parameters=parameters,
        integral_manager=integral_manager,
        transformation=modified.transformation,
    ), to_active
