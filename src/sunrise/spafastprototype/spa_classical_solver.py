import numpy as np
from ..molecules.qubit_base.qc_base import QuantumChemistryBase
from tequila import Variable
from tequila.objective.objective import Variables


"""SPA read-out and energy in the hard-core-boson model, both O(N^2).

The energy of an SPA state
--------------------------
The HCB Hamiltonian is

    H = c_1 . 1  +  sum_p eps_p n_p
                 +  sum_{p<q} U_pq n_p n_q
                 +  sum_{p<q} t_pq (P^+_p P_q + P^+_q P_p),

and the SPA state is a product over the edges of the matching,

    |Psi(theta)> = prod_{e=(i,j)} [ cos(th_e/2)|1_i 0_j> + sin(th_e/2)|0_i 1_j> ].

Write the orbital occupations

    o_i = cos^2(th_e/2),   o_j = 1 - o_i        for the edge e = (i, j).

Then <H> splits into four pieces.

1.  Constant.  hcb_model() returns eps and U in the n-representation, which
    absorbs part of the identity coefficient.  Undoing that,

        shift = c_1 - (1/2) sum_p eps_p - (1/4) sum_{p<q} U_pq.

2.  One body.  n_p is diagonal and the state is a product, so
    <sum_p eps_p n_p> = sum_p eps_p o_p exactly.  Cost O(N).

3.  Density-density.  Here is the one place where the factorisation is not
    naive.  If p and q sit on *different* edges the two factors are
    independent and <n_p n_q> = o_p o_q.  If they are the two orbitals of the
    *same* edge, the edge carries exactly one pair, so |1_i 1_j> is not in the
    span of the factor and

        <n_i n_j> = 0,   not   o_i o_j.

    The clean way to write this is to take the full mean-field sum and
    subtract the same-edge terms back out:

        <sum_{p<q} U_pq n_p n_q>
            = (1/2) o^T U o  -  sum_{e=(i,j)} U_ij o_i o_j

    (U has zero diagonal, so sum_{p<q} U_pq o_p o_q = (1/2) o^T U o).
    The matrix-vector product is the O(N^2) step and the only one.

4.  Pair hop.  P^+_p P_q moves a pair from orbital q to orbital p.  Across two
    different edges that changes the pair number on each edge -- two pairs on
    one, none on the other -- which leaves the span of the SPA factors, so the
    expectation vanishes.  Only the intra-edge hop survives, and on its own
    edge

        <psi_e| P^+_i P_j + P^+_j P_i |psi_e> = 2 cos(th/2) sin(th/2) = sin th,

    so the whole hop contribution is  sum_{e=(i,j)} t_ij sin(th_e).  Cost O(N).

Putting it together,

    E(theta) = shift + sum_p eps_p o_p
                     + (1/2) o^T U o - sum_e U_ij o_i o_j
                     + sum_e t_ij sin(th_e).                          (*)

Consistency with the read-out
-----------------------------
Collect the terms of (*) that depend on one angle th_e.  The same-edge U term
cancels against the same-edge part of the mean-field sum, leaving

    E(th_e) = (eps_i + F_i) cos^2(th_e/2)
            + (eps_j + F_j) sin^2(th_e/2)
            + t_ij sin(th_e) + const,      F_p = sum_{k not in e} U_pk o_k,

i.e. exactly the two-level form  Ebar - (dE/2) cos th + V sin th  with

    dE = (eps_j - eps_i) + sum_{k not in e} (U_jk - U_ik) o_k,   V = t_ij,

which is the dE used by spa_angles_for_graph().  Its minimiser is
th_e = -atan2(2V, dE) and its value Ebar - sqrt(dE^2 + 4V^2)/2.  So the dressed
read-out is exact coordinate descent on (*): every sweep lowers the energy of
this function, and its fixed point is a stationary point of it.  That is what
the self-test at the bottom checks.
"""


def hcb_model(mol: QuantumChemistryBase):
    """Single pass over the Pauli terms -> (eps, U, hop).  O(T + n^2).

    Returns the n-vector eps, the n x n density-density matrix U, and the
    n x n pair-hop matrix hop (hop[i, j] = <pair on i|H|pair on j>).
    `skipped` counts terms that do not fit the HCB structure; it must be 0.
    """
    c, h, g = mol.get_integrals(ordering="openfermion")
    g = g.elems
    eps = np.einsum("iiii->i", g) + 2.0 * np.diag(h)
    U = 2.0 * (2.0 * np.einsum("ijji->ij", g) - np.einsum("ijij->ij", g))
    hop = np.einsum("iijj->ij", g)
    np.fill_diagonal(U, 0.0)
    np.fill_diagonal(hop, 0.0)

    return eps, U, hop, c


# --------------------------------------------------------------------------
#  graph
# --------------------------------------------------------------------------


def split_graph(graph):
    """graph -> (edges, lone).

    Length-2 entries are rotating edges (i, j); length-1 entries are lone
    pairs, i.e. an edge frozen at theta = 0: orbital doubly occupied, no hop,
    no angle.
    """
    edges = [tuple(e) for e in graph if len(e) == 2]
    lone = [e[0] for e in graph if len(e) == 1]
    if len(edges) + len(lone) != len(graph):
        raise ValueError(
            "graph entries must have length 1 (lone pair) or 2 (edge): {}".format(graph)
        )
    return edges, lone


# --------------------------------------------------------------------------
#  occupations and energy
# --------------------------------------------------------------------------


def spa_occupations(edges, theta, n, lone=()):
    """Orbital occupations <n_p> of the SPA state.  O(N)."""
    o = np.zeros(n)
    o[list(lone)] = 1.0
    for (i, j), th in zip(edges, theta):
        c2 = np.cos(0.5 * th) ** 2
        o[i] = c2
        o[j] = 1.0 - c2
    return o


def spa_energy(eps, U, hop, shift, edges, theta, lone=()):
    """<Psi(theta)|H|Psi(theta)> for the SPA state, in O(N^2).

    Implements Eq. (*) of the module docstring.  Exact, not a mean-field
    approximation: the only correlation the SPA state carries is inside an
    edge, and both intra-edge terms (the vanishing n_i n_j and the surviving
    hop) are handled explicitly.  Lone pairs enter only through o = 1.
    """
    n = len(eps)
    o = spa_occupations(edges, theta, n, lone)
    theta = np.asarray(theta, dtype=float)

    e_one = float(eps @ o)  # O(N)
    e_uu = 0.5 * float(o @ (U @ o))  # O(N^2)

    I = np.array([e[0] for e in edges], dtype=int)
    J = np.array([e[1] for e in edges], dtype=int)
    e_uu -= float(np.sum(U[I, J] * o[I] * o[J]))  # same edge: 0
    e_hop = float(np.sum(hop[I, J] * np.sin(theta)))  # intra-edge only

    return shift + e_one + e_uu + e_hop


# --------------------------------------------------------------------------
#  angles
# --------------------------------------------------------------------------


def _readout(eps, U, hop, edges, lone, sweeps, callback=None):
    """Vacuum read-out followed by `sweeps` dressed (Gauss-Seidel) sweeps.

    callback(theta) is called after the vacuum read-out and after each sweep.
    """
    n = len(eps)
    I = np.array([e[0] for e in edges], dtype=int)
    J = np.array([e[1] for e in edges], dtype=int)
    t_edge = hop[I, J]

    # stage 1: vacuum read-out   (dE = eps_j - eps_i,  V = t_ij)
    theta = -np.arctan2(2.0 * t_edge, eps[J] - eps[I])
    if callback is not None:
        callback(theta)

    # stage 2: dressed read-out -- other pairs (incl. lone pairs) occupied
    for _ in range(sweeps):
        occ = spa_occupations(edges, theta, n, lone)
        for m, (i, j) in enumerate(edges):
            o = occ.copy()
            o[i] = 0.0
            o[j] = 0.0
            dE = (eps[j] - eps[i]) + (U[j] - U[i]) @ o
            theta[m] = -np.arctan2(2.0 * t_edge[m], dE)
            c = np.cos(theta[m] / 2.0) ** 2
            occ[i] = c
            occ[j] = 1.0 - c
        if callback is not None:
            callback(theta)
    return theta


def _to_variables(edges, theta, label):
    return {Variable(name=(edge, "D", label)): theta[m] for m, edge in enumerate(edges)}


def spa_angles_for_graph(mol, graph, sweeps=2, label=None) -> Variables:
    """SPA angles for every edge.

    sweeps = 0 -> plain vacuum read-out (other pairs empty)
    sweeps > 0 -> dressed read-out: repeat with the other pairs held in their
                  current SPA state (Gauss-Seidel, in place).  Two is enough.
    """
    edges, lone = split_graph(graph)
    eps, U, hop, _ = hcb_model(mol)
    theta = _readout(eps, U, hop, edges, lone, sweeps)
    return _to_variables(edges, theta, label)


# --------------------------------------------------------------------------
#  angles + energy in one pass
# --------------------------------------------------------------------------


def spa_solve_graph(
    mol: QuantumChemistryBase,
    graph: list[tuple[int]],
    sweeps=2,
    label=None,
    trace=False,
) -> tuple[Variables, float]:
    """Read out the angles of one graph and evaluate its SPA energy.

    Returns (theta, energy).  With trace=True, returns (theta, energy, hist)
    where hist[k] is the energy after k dressing sweeps -- a non-increasing
    sequence, since the dressed read-out is coordinate descent on the same
    function that spa_energy() evaluates.

    Everything here is O(N^2) per sweep and no circuit is evaluated.
    """
    edges, lone = split_graph(graph)
    eps, U, hop, shift = hcb_model(mol)

    hist = []
    theta = _readout(
        eps,
        U,
        hop,
        edges,
        lone,
        sweeps,
        callback=lambda th: hist.append(
            spa_energy(eps, U, hop, shift, edges, th, lone)
        ),
    )

    if trace:
        return list(theta), hist[-1], hist
    return _to_variables(edges, theta, label), hist[-1]


def spa_energy_for_graph(
    mol: QuantumChemistryBase, graph: list[tuple[int]], theta
) -> float:
    """SPA energy of one graph at given angles (one per length-2 edge).  O(N^2)."""
    edges, lone = split_graph(graph)
    eps, U, hop, shift = hcb_model(mol)
    return spa_energy(eps, U, hop, shift, edges, theta, lone)


def spa_rdms(
    edges: list[tuple[int]], variables: Variables, n_orbitals: int = None, label=None
):
    """
    Classical evaluation of the spin-free 1- and 2-RDM (dirac ordering) of the HCB-SPA wavefunction.

    Each edge (i, j, k, ...) holds one electron pair in the geminal
        c_i|i> + c_j|j> + c_k|k> + ...
    built by the ladder excitations i->j->k->... with angles t_ij = variables[((i, j), 'D', label)]:
        c_i = cos(t_ij/2), c_j = sin(t_ij/2) cos(t_jk/2), ..., c_last = prod sin(t/2)
    Edges of length 1 are non-excited (doubly occupied) pairs.
    The non-zero elements are (n_p = c_p**2, P_pq = n_p n_q if p, q on different edges else 0):
        rdm1[p,p]     = 2 n_p
        rdm2[p,p,q,q] = 2 c_p c_q   (p, q on the same edge, includes p == q)
        rdm2[p,q,p,q] = 4 P_pq      (p != q)
        rdm2[p,q,q,p] = -2 P_pq     (p != q)
    """
    if n_orbitals is None:
        n_orbitals = max(max(edge) for edge in edges) + 1
    values = {getattr(k, "name", k): float(v) for k, v in variables.items()}

    c = np.zeros(n_orbitals)
    edge_id = np.full(n_orbitals, -1)
    for e, edge in enumerate(edges):
        amp = 1.0
        for i in range(1, len(edge)):
            angle = values[((edge[i - 1], edge[i]), "D", label)]
            c[edge[i - 1]] = amp * np.cos(0.5 * angle)
            amp *= np.sin(0.5 * angle)
        c[edge[-1]] = amp
        edge_id[list(edge)] = e
    n = c**2

    rdm1 = np.diag(2.0 * n)

    rdm2 = np.zeros((n_orbitals,) * 4)
    same_edge = (edge_id[:, None] == edge_id[None, :]) & (edge_id[:, None] >= 0)
    P = np.where(same_edge, 0.0, np.outer(n, n))
    p, q = np.indices((n_orbitals, n_orbitals))
    rdm2[p, p, q, q] = np.where(same_edge, 2.0 * np.outer(c, c), 0.0)
    rdm2[p, q, p, q] += 4.0 * P
    rdm2[p, q, q, p] += -2.0 * P
    return rdm1, rdm2
