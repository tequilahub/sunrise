"""Fast, CI-space construction of an operator callable from an arbitrary (up_then_down
spin-orbital ordered) FermionOperator, avoiding the O(2**n_qubits) dense qubit
wavefunction round trip.

Two strategies are provided:

- ``extract_restricted_integrals``: detects whether the operator is expressible as a
  spin-restricted one- and two-body Hamiltonian (same integrals for both spins, which
  covers essentially all physical Hamiltonians/observables built the way TCC itself
  builds its Hamiltonians). When applicable, the operator can be applied directly in
  CI space via ``tencirchem.static.hamiltonian.get_h_fcifunc_from_integral``
  (PySCF's ``direct_nosym`` FCI contraction), at the cost of a CI vector instead of a
  full qubit statevector.
- ``build_sparse_ci_operator``: a fully general fallback for any other 1-body/2-body
  or higher-body FermionOperator (including non-Hermitian excitation-type operators).
  Builds the Jordan-Wigner sparse matrix once and restricts it to the CI-string basis,
  so the O(2**n_qubits) cost is paid once at construction time instead of on every
  evaluation.
"""

from __future__ import annotations

import numpy as np
from openfermion import FermionOperator
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import jordan_wigner, normal_ordered
from scipy.sparse import csr_matrix
from tencirchem.utils.misc import reverse_fop_idx

TOL = 1e-9


def extract_restricted_integrals(
    operator: FermionOperator, n_orb: int
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Express an operator as spin-restricted one- and two-body integrals.

    Parameters
    ----------
    operator : FermionOperator
        Operator in up-then-down spin orbital ordering (indices ``0..n_orb-1`` for
        one spin and ``n_orb..2*n_orb-1`` for the other).
    n_orb : int
        Number of spatial orbitals.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, float) or None
        ``(int1e, int2e, constant)`` as expected by tencirchem's
        ``get_h_fcifunc_from_integral``. None if the operator has terms beyond
        two-body, is complex, does not conserve Sz or is not spin-restricted;
        callers should then use ``build_sparse_ci_operator``.
    """
    n_sorb = 2 * n_orb
    isalpha = lambda i: i >= n_orb
    no = normal_ordered(operator)

    constant = 0.0
    h1e_sorb = np.zeros((n_sorb, n_sorb))
    two_body_terms = []
    for key, val in no.terms.items():
        if abs(np.imag(val)) > TOL:
            return None
        val = float(np.real(val))
        if len(key) == 0:
            constant += val
        elif len(key) == 2:
            (p, ap), (q, aq) = key
            if ap != 1 or aq != 0:
                return None
            h1e_sorb[p, q] += val
        elif len(key) == 4:
            (p, ap), (q, aq), (r, ar), (s, as_) = key
            if (ap, aq, ar, as_) != (1, 1, 0, 0):
                return None
            two_body_terms.append((p, q, r, s, val))
        else:
            return None  # higher than 2-body, cannot be represented via int1e/int2e

    if not np.allclose(h1e_sorb[:n_orb, n_orb:], 0, atol=TOL):
        return None
    if not np.allclose(h1e_sorb[n_orb:, :n_orb], 0, atol=TOL):
        return None
    h1a = h1e_sorb[n_orb:, n_orb:]
    h1b = h1e_sorb[:n_orb, :n_orb]
    if not np.allclose(h1a, h1b, atol=TOL):
        return None  # spin-unrestricted, not representable via a single int1e
    int1e = h1a.copy()

    int2e = np.full((n_orb,) * 4, np.nan)

    def set_or_check(a, b, c, d, v):
        """Store ``v`` in ``int2e`` or check it matches the stored value."""
        a, b, c, d = a % n_orb, b % n_orb, c % n_orb, d % n_orb
        cur = int2e[a, b, c, d]
        if np.isnan(cur):
            int2e[a, b, c, d] = v
            return True
        return abs(cur - v) <= 1e-7

    same_spin_terms = []
    for p, q, r, s, val in two_body_terms:
        sp, sq, sr, ss = isalpha(p), isalpha(q), isalpha(r), isalpha(s)
        if sp == sq:
            # same-spin (aa/bb) block: coefficient is a difference of two int2e
            # entries, so it cannot be read off directly. Validated afterwards
            # against the int2e recovered from the (unambiguous) mixed-spin terms.
            same_spin_terms.append((p, q, r, s, val))
            continue
        if sp == ss and sq == sr:
            if not set_or_check(p, s, q, r, val):
                return None
        elif sp == sr and sq == ss:
            if not set_or_check(p, r, q, s, -val):
                return None
        else:
            return None  # not Sz-conserving

    int2e = np.where(np.isnan(int2e), 0.0, int2e)

    for p, q, r, s, val in same_spin_terms:
        sp, sq, sr, ss = isalpha(p), isalpha(q), isalpha(r), isalpha(s)
        if not (sp == ss and sq == sr):
            return None  # not Sz-conserving
        pm, qm, rm, sm = p % n_orb, q % n_orb, r % n_orb, s % n_orb
        predicted = int2e[pm, sm, qm, rm] - int2e[qm, sm, pm, rm]
        if abs(predicted - val) > 1e-6:
            return None  # not spin-restricted / not consistent with a real int2e

    return int1e, int2e, constant


def build_sparse_ci_operator(
    operator: FermionOperator, n_qubits: int, ci_strings
) -> csr_matrix:
    """Build a sparse matrix of an operator restricted to the CI-string basis.

    The Jordan-Wigner matrix is built once, at cost ``O(2**n_qubits)``, so each
    later application is a cheap sparse matrix-vector product.

    Parameters
    ----------
    operator : FermionOperator
        Operator of any body order, also non-Hermitian.
    n_qubits : int
        Number of qubits (spin orbitals).
    ci_strings : array_like of int
        Bitstrings of the CI basis.

    Returns
    -------
    scipy.sparse.csr_matrix
        Real part of the operator in the CI basis.
    """
    fop_rev = reverse_fop_idx(operator, n_qubits)
    full_sparse = get_sparse_operator(jordan_wigner(fop_rev), n_qubits=n_qubits)
    ci_idx = np.asarray(ci_strings, dtype=np.int64)
    return full_sparse[ci_idx, :][:, ci_idx].real.tocsr()
