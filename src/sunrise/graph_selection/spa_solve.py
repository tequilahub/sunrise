import numpy as np
import tequila as tq

def _qubit_map(H):
    qubits = sorted(H.qubits)
    return qubits, {q: k for k, q in enumerate(qubits)}


def hcb_model(H, n, pos):
    """Single pass over the Pauli terms -> (eps, U, hop).  O(T + n^2).

    Returns the n-vector eps, the n x n density-density matrix U, and the
    n x n pair-hop matrix hop (hop[i, j] = <pair on i|H|pair on j>).
    `skipped` counts terms that do not fit the HCB structure; it must be 0.
    """
    cZ = np.zeros(n)
    cZZ = np.zeros((n, n))
    hop = np.zeros((n, n))
    skipped = 0

    for key, coeff in H.items():
        c = complex(coeff).real
        xy = []
        zs = []
        for (q, p) in key:
            p = p.upper()
            k = pos[q]
            if p == "Z":
                zs.append(k)
            else:
                xy.append((k, p))

        if len(xy) == 0:                       # diagonal (identity / Z / ZZ)
            if len(zs) == 0:
                pass                           # constant, drops out of eps & U
            elif len(zs) == 1:
                cZ[zs[0]] += c
            elif len(zs) == 2:
                cZZ[zs[0], zs[1]] += c
                cZZ[zs[1], zs[0]] += c
            else:
                skipped += 1
        elif len(xy) == 2 and len(zs) == 0:    # pair hop: XX or YY
            (a, pa), (b, pb) = xy
            if pa == pb:
                hop[a, b] += c
                hop[b, a] += c
            else:
                skipped += 1                   # XY / YX: absent from a real H
        else:
            skipped += 1

    np.fill_diagonal(cZZ, 0.0)
    U = 4.0 * cZZ
    eps = -2.0 * cZ - 2.0 * cZZ.sum(axis=1)
    return eps, U, hop, skipped


def spa_angles_for_graph(mol, graph, sweeps=2, warn=True):
    """SPA angles for every edge.

    sweeps = 0 -> plain vacuum read-out (other pairs empty)
    sweeps > 0 -> dressed read-out: repeat with the other pairs held in their
                  current SPA state (Gauss-Seidel, in place).  Two is enough.
    """
    H = mol.make_hardcore_boson_hamiltonian()
    edges = [tuple(e) for e in graph]
    qubits, pos = _qubit_map(H)
    n = len(qubits)                                   # = number of orbitals

    eps, U, hop, skipped = hcb_model(H, n, pos)
    if warn and skipped:
        print("warning: {} Pauli term(s) outside the HCB structure "
              "(eps/U/hop model may be incomplete)".format(skipped))

    I = np.array([e[0] for e in edges])
    J = np.array([e[1] for e in edges])
    t_edge = hop[I, J]

    # stage 1: vacuum read-out   (dE = eps_j - eps_i,  V = t_ij)
    theta = -np.arctan2(2.0 * t_edge, eps[J] - eps[I])

    # stage 2: dressed read-out -- other pairs now occupied
    for _ in range(sweeps):
        occ = np.zeros(n)
        c2 = np.cos(theta / 2.0) ** 2
        occ[I] = c2
        occ[J] = 1.0 - c2
        for m, (i, j) in enumerate(edges):
            o = occ.copy()
            o[i] = 0.0
            o[j] = 0.0
            dE = (eps[j] - eps[i]) + (U[j] - U[i]) @ o
            theta[m] = -np.arctan2(2.0 * t_edge[m], dE)
            c = np.cos(theta[m] / 2.0) ** 2
            occ[i] = c
            occ[j] = 1.0 - c

    return list(theta)
