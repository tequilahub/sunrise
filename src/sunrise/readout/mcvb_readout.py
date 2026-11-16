import numpy as np
import tequila as tq

from spa_readout import spa_angles_for_graph, hcb_model, _qubit_map


class BraKetQulacs:
    """
    Hacky Replacement of tq.BraKet
    Speedup of underlying simulation
    Limitations: Only Qulacs can be backend, not differentiable at the moment
    """
    def __init__(self, bra,ket,H):
        # translate tq -> qulacs
        E1 = tq.compile(tq.ExpectationValue(U=bra,H=H), backend="qulacs")
        E2 = tq.compile(tq.ExpectationValue(U=ket,H=H), backend="qulacs")
        # extract qulacs structures
        self.bra = E1.get_expectationvalues()[0]._U
        self.ket = E2.get_expectationvalues()[0]._U
        self.H = E1.get_expectationvalues()[0]._H[0]
        self.n_qubits = ket.n_qubits
        self.is_overlap = H.n_qubits == 0
    def __call__(self, variables, *args, **kwargs):
        # call qulacs structures
        # similar as tequila would, but exploits storing wavefunctions
        self.ket.update_variables(variables)
        self.bra.update_variables(variables)
        state_bra = self.bra.initialize_state(self.n_qubits)
        state_ket = self.ket.initialize_state(self.n_qubits)
        self.bra.circuit.update_quantum_state(state_bra)
        self.ket.circuit.update_quantum_state(state_ket)
        if self.is_overlap:
            vector1 = state_bra.get_vector()
            vector2 = state_ket.get_vector()
            result = vector1.conj().T.dot(vector2)
        else:
            result = self.H.get_transition_amplitude(state_bra, state_ket)

        result=result.real
        return result


def _model(molx, n):
    """(shift, eps, U, hop) for an orbital-rotated molecule.

        H = shift + sum_p eps_p n_p + sum_{p<q} U_pq n_p n_q
                  + sum_{p<q} hop_pq (P+_p P_q + h.c.)

    hcb_model drops the constant because the read-out only needs differences;
    the coordinate descent needs absolute energies (they enter the Rayleigh
    quotient alongside K and L), so it is recovered here.  With
    n_p = (1 - Z_p)/2 the identity coefficient is
    c(I) = shift + sum_p eps_p/2 + sum_{p<q} U_pq/4.
    """
    Hh = molx.make_hardcore_boson_hamiltonian()
    qubits, pos = _qubit_map(Hh)
    eps, U, hop, skipped = hcb_model(Hh, len(qubits), pos)
    if skipped:
        print("warning: {} Pauli term(s) outside the HCB structure".format(skipped))
    cI = sum(complex(c).real for key, c in Hh.items() if len(key) == 0)
    shift = cI - 0.5 * eps.sum() - 0.25 * np.triu(U, 1).sum()
    return shift, eps, U, hop


def _occ(edges, theta, n):
    occ = np.zeros(n)
    for m, (i, j) in enumerate(edges):
        c = np.cos(theta[m] / 2.0) ** 2
        occ[i] = c
        occ[j] = 1.0 - c
    return occ


def _energy(model, edges, theta, n):
    """<Psi|H|Psi> of one SPA product state, O(n^2).

    sum_{p<q} U_pq occ_p occ_q factorises ACROSS edges only: inside an edge the
    pair sits on i or on j, never both, so <n_i n_j> = 0 while
    occ_i occ_j = cos^2 sin^2 != 0.  Same-edge pairs are subtracted back out.
    """
    shift, eps, U, hop = model
    occ = _occ(edges, theta, n)
    E = shift + eps @ occ + 0.5 * (occ @ U @ occ)
    for m, (i, j) in enumerate(edges):
        E -= U[i, j] * occ[i] * occ[j]
        E += hop[i, j] * np.sin(theta[m])
    return float(E)


def _block(model, edges, theta, m, n):
    """2x2 block [[E_A, V], [V, E_B]] with edge m pinned, the rest dressed.

    |A> = |1_i 0_j> (x) spectators, |B> = |0_i 1_j> (x) spectators.  Only the
    hop inside edge m connects them, so V = hop_ij exactly -- moving this pair
    and a spectator at once would need a two-pair term, which H does not have.
    """
    shift, eps, U, hop = model
    i, j = edges[m]
    occ = _occ(edges, theta, n)
    occ[i] = 0.0
    occ[j] = 0.0

    common = shift + eps @ occ + 0.5 * (occ @ U @ occ)
    for k, (a, b) in enumerate(edges):
        if k == m:
            continue
        common -= U[a, b] * occ[a] * occ[b]
        common += hop[a, b] * np.sin(theta[k])

    EA = common + eps[i] + U[i] @ occ
    EB = common + eps[j] + U[j] @ occ
    return np.array([[EA, hop[i, j]], [hop[i, j], EB]])


# ==========================================================================
#  generalised eigenproblem and the one-edge minimiser
# ==========================================================================

def _gem(Hm, Sm, s_thresh=1e-8):
    """Lowest root of Hc = lambda Sc by canonical orthogonalisation.

    A raw eigh(Hm, Sm) on a near-singular overlap returns a spuriously LOW
    energy -- not variational, and it looks like success.
    """
    sv, X = np.linalg.eigh(Sm)
    keep = sv > s_thresh
    Xr = X[:, keep] / np.sqrt(sv[keep])
    w, V = np.linalg.eigh(Xr.T @ Hm @ Xr)
    c = Xr @ V[:, 0]
    return float(w[0]), c / np.linalg.norm(c)


def _solve_edge(A, b, s, cg, K, L, grid, u):
    """Global minimiser over theta of

        E(theta) = (cg^2 u'A u + 2 cg u'b + K) / (cg^2 + 2 cg u's + L),
        u = (cos(theta/2), sin(theta/2))

    on a grid, then one parabolic step for the precision.
    """
    num = cg ** 2 * np.einsum("an,ab,bn->n", u, A, u) + 2.0 * cg * (b @ u) + K
    den = cg ** 2 + 2.0 * cg * (s @ u) + L
    ok = den > 1e-10
    if not ok.any():
        return None
    E = np.where(ok, num / np.where(ok, den, 1.0), np.inf)
    k = int(np.argmin(E))
    if 0 < k < len(grid) - 1:
        y0, y1, y2 = E[k - 1], E[k], E[k + 1]
        curv = y0 - 2.0 * y1 + y2
        if abs(curv) > 1e-14:
            return float(np.clip(grid[k] + 0.5 * (y0 - y2) / curv
                                 * (grid[1] - grid[0]), -np.pi, np.pi))
    return float(grid[k])


# ==========================================================================
#  the solver
# ==========================================================================

def solve_mcvb(mol, graphs, iterations=20, sweeps=2, n_grid=2001,
               tol=1e-10, c_min=1e-3, verbose=True):
    """Returns (variables, energy, coeffs)."""
    n = mol.n_orbitals
    H = mol.make_hamiltonian()

    graphs = [[tuple(sorted(e)) for e in g] for g in graphs]
    n_g = len(graphs)

    circuits = []
    variables = {}
    keys = []
    models = []
    thetas = []

    for i, g in enumerate(graphs):
        fixed = np.eye(n)
        for edge in g:
            fixed[edge[0], edge[1]] = 1.0
            fixed[edge[1], edge[0]] = -1.0

        fixed *= np.sqrt(2) / 2
        fixed = fixed.T

        vars = spa_angles_for_graph(mol.transform_orbitals(fixed), g, sweeps=sweeps)

        U = mol.make_ansatz(name="HCB-SPA", edges=g, label="G{}".format(i))

        var_dict = {}
        for k, param_key in enumerate(U.make_parameter_map()):
            var_dict[param_key] = vars[k]

        opt = tq.quantumchemistry.optimize_orbitals(mol, circuit=U.map_variables(var_dict), use_hcb=True,
                                                    initial_guess=fixed, silent=True)
        vars = spa_angles_for_graph(mol.transform_orbitals(opt.mo_coeff), g, sweeps=sweeps)

        var_dict = {}
        for k, param_key in enumerate(U.make_parameter_map()):
            var_dict[param_key] = vars[k]

        opt = tq.quantumchemistry.optimize_orbitals(mol, circuit=U.map_variables(var_dict), use_hcb=True,
                                                    initial_guess=opt.mo_coeff, silent=True)

        M = np.asarray(opt.mo_coeff, dtype=float)


        circuits.append(mol.make_ansatz(name="SPA", edges=g, label="G{}".format(i))
                        + mol.get_givens_circuit(unitary=M))


        # Note here we read the SPA vars out twice. first with the assumptions that orbitals are rotated by pi/2 and then with the optimized orbitals
        # Such that the approximation is more accurate...

        molM = mol.transform_orbitals(M)
        vars = spa_angles_for_graph(molM, g, sweeps=sweeps)
        var_dict = {key: vars[k] for k, key in enumerate(U.make_parameter_map())}
        variables = {**variables, **var_dict}

        keys.append(list(var_dict))
        thetas.append(np.array(vars, dtype=float))
        models.append(_model(molM, n))

        if verbose:
            print("G{} {}   orbital-opt E = {:+.10f}".format(i, g, opt.energy))

    # ---- cross terms only: the diagonal comes from the O(n^2) model --------
    braH = {(a, b): BraKetQulacs(circuits[a], circuits[b], H)
            for a in range(n_g) for b in range(a + 1, n_g)}
    braS = {(a, b): BraKetQulacs(circuits[a], circuits[b], tq.paulis.I())
            for a in range(n_g) for b in range(a + 1, n_g)}

    def assemble(v):
        Hm = np.diag([_energy(models[g], graphs[g], thetas[g], n)
                      for g in range(n_g)])
        Sm = np.eye(n_g)
        for a in range(n_g):
            for b in range(a + 1, n_g):
                Hm[a, b] = Hm[b, a] = braH[(a, b)](v)
                Sm[a, b] = Sm[b, a] = braS[(a, b)](v)
        return Hm, Sm

    Hm, Sm = assemble(variables)
    energy, coeffs = _gem(Hm, Sm)
    best = (dict(variables), energy, coeffs.copy())
    if verbose:
        print("read-out   E = {:+.10f}   c = {}".format(energy, np.round(coeffs, 4)))

    grid = np.linspace(-np.pi, np.pi, n_grid)
    u = np.stack([np.cos(grid / 2.0), np.sin(grid / 2.0)])

    # ---- coordinate descent: one exact angle at a time --------------------
    for it in range(iterations):
        previous = energy

        for g in range(n_g):
            cg = coeffs[g]
            if abs(cg) < c_min:                  # flat direction
                continue
            others = [h for h in range(n_g) if h != g]
            cc = coeffs[others]
            K = float(cc @ Hm[np.ix_(others, others)] @ cc)
            L = float(cc @ Sm[np.ix_(others, others)] @ cc)

            for m in range(len(graphs[g])):
                A = _block(models[g], graphs[g], thetas[g], m, n)
                b = np.zeros(2)
                s = np.zeros(2)
                for slot, pin in enumerate((0.0, np.pi)):
                    pv = {**variables, keys[g][m]: pin}
                    for h in others:
                        key = (min(g, h), max(g, h))
                        b[slot] += coeffs[h] * braH[key](pv)
                        s[slot] += coeffs[h] * braS[key](pv)

                th = _solve_edge(A, b, s, cg, K, L, grid, u)
                if th is not None:
                    thetas[g][m] = th
                    variables[keys[g][m]] = th

            Hm, Sm = assemble(variables)         # Gauss-Seidel across graphs
            energy, coeffs = _gem(Hm, Sm)

        if energy < best[1]:
            best = (dict(variables), energy, coeffs.copy())
        if verbose:
            print("iter {:3d}   E = {:+.10f}   dE = {:.2e}   c = {}"
                  .format(it, energy, abs(energy - previous), np.round(coeffs, 4)))
        if abs(energy - previous) < tol:
            break

    variables, energy, coeffs = best
    variables = {k: float((v + np.pi) % (2 * np.pi) - np.pi)
                 for k, v in variables.items()}
    return variables, energy, coeffs


