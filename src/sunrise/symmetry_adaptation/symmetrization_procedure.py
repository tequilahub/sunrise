"""Procedures that turn Fock-space states (determinants) into spin- and/or symmetry-adapted combinations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import groupby
from itertools import product as iproduct
from typing import TYPE_CHECKING

import numpy
import pandas as pd
import tequila

from .fock_space_state import FockSpaceState
from .irrep_provider import IrrepProviderBase

if TYPE_CHECKING:
    # only for type hints, importing it at runtime would be circular
    from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase


@dataclass
class SymmetrizationProcedure(ABC):
    """Abstract base class of the symmetrization procedures.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule defining the orbitals and, where needed, the point group.
    """

    mol: QuantumChemistryBase

    @abstractmethod
    def symmetrize(self, states: pd.DataFrame) -> pd.DataFrame:
        """Return symmetry-adapted combinations of the given states.

        Parameters
        ----------
        states : list of FockSpaceState or pandas.DataFrame
            States to process, e.g. from ``FockSpaceState.non_ionic_states`` (a DataFrame as made by
            ``FockSpaceState.dataframe``).

        Returns
        -------
        list of FockSpaceState or pandas.DataFrame
            Same container type as the input.
        """


@dataclass
class SpinSymmetrizationProcedure(SymmetrizationProcedure):
    """Symmetrize states with respect to spin.

    The S^2 operator is diagonalized within each block of equal ``m_s``, so that the new states
    are spin eigenfunctions. Blocks of states that do not couple through S^2 are treated apart.

    Parameters
    ----------
    mol : QuantumChemistryBase

    Examples
    --------
    >>> states = FockSpaceState.non_ionic_states(mol, IrrepProvider(mol, "canon"))
    >>> spin_states = SpinSymmetrizationProcedure(mol).symmetrize(states)
    """

    @staticmethod
    def _get_connected_components(matrix, tol=1e-10):
        """Return the groups of indices connected by nonzero (above ``tol``) matrix elements."""
        n = matrix.shape[0]
        visited = [False] * n
        components = []
        for i in range(n):
            if not visited[i]:
                component = []
                stack = [i]
                visited[i] = True
                while stack:
                    node = stack.pop()
                    component.append(node)
                    for neighbor in range(n):
                        if not visited[neighbor] and abs(matrix[node, neighbor]) > tol:
                            visited[neighbor] = True
                            stack.append(neighbor)
                components.append(component)
        return components

    def symmetrize(
        self, states: list[FockSpaceState] | pd.DataFrame
    ) -> list[FockSpaceState] | pd.DataFrame:
        """Return the eigenstates of S^2 within each ``m_s`` block of the input states.

        Parameters
        ----------
        states : list of FockSpaceState or pandas.DataFrame
            States to process, e.g. from ``FockSpaceState.non_ionic_states`` (a DataFrame as made by
            ``FockSpaceState.dataframe``).

        Returns
        -------
        list of FockSpaceState or pandas.DataFrame
            Spin eigenstates with ``S2`` and ``m_s`` set. The irrep is not set.
        """
        is_dataframe = isinstance(states, pd.DataFrame)
        if is_dataframe:
            state_list: list[FockSpaceState] = list(
                states.itertuples(index=False, name="FockSpaceState")
            )  # produces namedtuples with attribute access
        else:
            state_list = list(states)

        # assume the IrrepProviderBase is the same for all states; it is not used in the S2
        # diagonalization below, only propagated to the output states.
        irrep_provider = state_list[0].irrep_provider
        fragment_orbitals = getattr(state_list[0], "fragment_orbitals", None)

        # Create an empty dataframe with the same columns as df
        diagonalised_states: list[FockSpaceState] = []

        # Group by and m_s/⟨S_z⟩
        # an additional grouping by mo_occ could be added however mo_occ does not always exist
        state_list_sorted = sorted(state_list, key=lambda s: s.m_s)
        grouped = groupby(state_list_sorted, key=lambda s: s.m_s)

        # Iterate over each group
        s2_op = self.mol.make_s2_op()
        for ms_val, group in grouped:
            sub_states: list[FockSpaceState] = list(group)
            mini_matrix = numpy.zeros((len(sub_states), len(sub_states)), dtype=complex)

            s2_applied = [
                s2_op(state.wavefunction) for state in sub_states
            ]  # O(n) applications

            for i, state1 in enumerate(sub_states):
                for j in range(i, len(sub_states)):  # only upper triangle
                    val = state1.wavefunction.inner(s2_applied[j])
                    mini_matrix[i, j] = val
                    mini_matrix[j, i] = numpy.conj(val)  # Hermitian

            mini_matrix = numpy.round(mini_matrix, decimals=6)

            # Diagonalize each connected component separately to prevent
            # degenerate states from different spatial irreps from mixing
            # It looks at where the non-zero matrix elements are, and since
            # the off-diagonal blocks are zero, it naturally separates states from different irreps
            components = self._get_connected_components(mini_matrix, tol=1e-10)

            for component in components:
                sub_matrix = mini_matrix[numpy.ix_(component, component)]
                eigvals, eigvecs = numpy.linalg.eigh(sub_matrix)

                for k in range(len(eigvals)):
                    eigvec = eigvecs[:, k]
                    s2_value = float(eigvals[k])
                    snapped = round(s2_value * 4) / 4
                    if abs(s2_value - snapped) < 1e-4:
                        s2_value = snapped
                    if abs(s2_value) < 1e-10:
                        s2_value = 0.0

                    new_wfn = tequila.QubitWaveFunction(
                        n_qubits=self.mol.n_orbitals * 2
                    )
                    for coef, idx in zip(eigvec, component):
                        new_wfn += coef * sub_states[idx].wavefunction

                    diagonalised_states.append(
                        FockSpaceState(
                            mol=self.mol,
                            wavefunction=new_wfn,
                            provider=irrep_provider,
                            S2=s2_value,
                            m_s=ms_val,
                            fragment_orbitals=fragment_orbitals,
                        )
                    )
        if is_dataframe:
            return FockSpaceState.dataframe(diagonalised_states)
        return diagonalised_states


@dataclass
class SymmetryAdaptedLinearCombinationSymmetrization(SymmetrizationProcedure):
    """Symmetrize states by building symmetry-adapted linear combinations (SALCs).

    Each state is projected onto every irrep of ``mol.point_group`` with the characters of the
    symmetry operations, which act as permutations of the orbitals. The results are grouped by
    (irrep, S^2, m_s), deduplicated and orthonormalized within each group.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule with a ``point_group``.

    Examples
    --------
    >>> states = FockSpaceState.non_ionic_states(mol, IrrepProvider(mol, "loc"))
    >>> salcs = SymmetryAdaptedLinearCombinationSymmetrization(mol).symmetrize(states)
    """

    def __post_init__(self):
        # Fetch the MO-specific permutations and phases, keyed by the operations of the point group
        self._pg = self.mol.point_group
        self._spatial_perms, self._spatial_phases = (
            self.mol.compute_mo_perms_and_phases()
        )
        # for label in self._pg.character_table.operation_symbols:
        # 	print(f"{label}: perm={self._spatial_perms[label]}, phases={self._spatial_phases[label]}")

    @staticmethod
    def _salc_key(wfn: tequila.QubitWaveFunction, decimals: int = 8) -> tuple:
        """Return a hashable key of a wavefunction that is invariant to a global phase, used to detect duplicates."""

        items = sorted(wfn.items())  # deterministic order by bitstring
        if not items:
            return ()

        # Find the phase of the first nonzero coefficient and divide it out,
        # so phase-equivalent states collapse to the same canonical key.
        first_coef = items[0][1]
        if abs(first_coef) < 1e-15:
            return ()
        phase = first_coef / abs(first_coef)

        return tuple(
            (
                bitstring,
                complex(
                    round((coef / phase).real, decimals),
                    round((coef / phase).imag, decimals),
                ),
            )
            for bitstring, coef in items
        )

    @staticmethod
    def _apply_spatial_permutation_bitstring(
        wfn: tequila.QubitWaveFunction,
        perm: list[int],
        n_orbitals: int,
        phases: list[float] | None = None,  # Add phase support
    ) -> tequila.QubitWaveFunction:
        """Apply a permutation of the spatial orbitals to a wavefunction.

        Parameters
        ----------
        wfn : tequila.QubitWaveFunction
        perm : list of int
            Orbital ``i`` is moved to orbital ``perm[i]``.
        n_orbitals : int
        phases : list of float, optional
            Phase of each singly occupied orbital. Default is 1.

        Returns
        -------
        tequila.QubitWaveFunction
            Permuted state, including the fermionic sign.
        """
        if phases is None:
            phases = [1.0] * n_orbitals
        n_qubits = 2 * n_orbitals
        result_terms = {}

        for bitstring, coef in wfn.items():
            bs_int = int(bitstring)
            new_bs = 0
            singly_occ = []
            phase = 1.0

            for i in range(n_orbitals):
                block = (bs_int >> (2 * i)) & 3
                if block:
                    j = perm[i]
                    new_bs |= block << (2 * j)
                    if block in (1, 2):  # Only apply phase to singly occupied orbitals
                        singly_occ.append(j)
                        phase *= phases[i]

            inv = sum(
                1
                for a in range(len(singly_occ))
                for b in range(a + 1, len(singly_occ))
                if singly_occ[a] > singly_occ[b]
            )
            sign = -1 if inv % 2 else 1  # Fermionic sign
            result_terms[new_bs] = result_terms.get(new_bs, 0) + sign * phase * coef

        result = tequila.QubitWaveFunction(n_qubits=n_qubits)
        for bs, c in result_terms.items():
            if abs(c) > 1e-15:
                result += c * tequila.QubitWaveFunction.from_string(
                    f"|{format(bs, f'0{n_qubits}b')}>"
                )
        return result

    def symmetrize(
        self, states: list[FockSpaceState] | pd.DataFrame
    ) -> list[FockSpaceState] | pd.DataFrame:
        """Return the symmetry-adapted combinations of the input states.

        If the first state defines ``fragment_orbitals``, only the symmetry operations that map the
        fragment onto itself are used.

        Parameters
        ----------
        states : list of FockSpaceState or pandas.DataFrame
            States to process, e.g. from ``FockSpaceState.non_ionic_states`` (a DataFrame as made by
            ``FockSpaceState.dataframe``).

        Returns
        -------
        list of FockSpaceState or pandas.DataFrame
            Orthonormal states with ``irrep``, ``S2`` and ``m_s`` set.
        """
        is_dataframe = isinstance(states, pd.DataFrame)
        if is_dataframe:
            state_list: list[FockSpaceState] = list(
                states.itertuples(index=False, name="FockSpaceState")
            )
        else:
            state_list = list(states)

        # assume the IrrepProviderBase is the same for all states; the SALC construction below
        # uses the character table of the point group of self.mol, not irrep_provider, which is only propagated to
        # the output states.
        irrep_provider = state_list[0].irrep_provider
        fragment_orbitals = getattr(state_list[0], "fragment_orbitals", None)

        # assumes the representation as quantum circuit
        n_orbitals = self.mol.n_orbitals
        n_qubits = 2 * n_orbitals
        pg = self._pg

        # select which operations to apply
        # For an fragment-space fragment, only operations that map the fragment orbital
        # set onto itself are symmetries of the fragment. Any other operation would
        # move electrons out of the fragment space. With no fragment space, use all ops.
        if fragment_orbitals is not None:
            fragment_set = set(fragment_orbitals)
            selected_ops = [
                (k, op_label)
                for k, op_label in enumerate(pg.character_table.operation_symbols)
                if all(
                    self._spatial_perms[op_label][i] in fragment_set
                    for i in fragment_orbitals
                )
            ]
        else:
            selected_ops = list(enumerate(pg.character_table.operation_symbols))

        # Group projected SALCs by (irrep, S2, m_s) to preserve spin purity!
        SALC_by_sector: dict[tuple, list[tequila.QubitWaveFunction]] = {}

        for state in state_list:
            # Extract spin quantum numbers from the seed state (O(1) since they are cached)
            s2_val = state.S2
            ms_val = state.m_s

            resulting_state_list: list[tequila.QubitWaveFunction] = []
            op_indices: list[int] = []

            for k, op_label in selected_ops:
                perm = self._spatial_perms[op_label]
                phases = self._spatial_phases[op_label]
                resulting_state_list.append(
                    self._apply_spatial_permutation_bitstring(
                        state.wavefunction, perm, n_orbitals, phases
                    )
                )
                op_indices.append(k)

            for j, irrep in enumerate(pg.character_table.dict.keys()):
                state_SALC = tequila.QubitWaveFunction(n_qubits=n_qubits)
                for idx, k in enumerate(op_indices):
                    character = pg.character_table.dict[irrep][k]
                    if character != 0:
                        state_SALC += character * resulting_state_list[idx]

                norm_sq = state_SALC.inner(state_SALC).real
                if norm_sq > 1e-10:
                    sector = (irrep, s2_val, ms_val)
                    if sector not in SALC_by_sector:
                        SALC_by_sector[sector] = []
                    SALC_by_sector[sector].append(state_SALC.normalize())

        result_objects = []

        for (irrep, s2_val, ms_val), wfn_list in SALC_by_sector.items():
            if not wfn_list:
                continue

            # Deduplicate exact copies within this sector
            unique_wfns = []
            seen_keys = set()
            for wfn in wfn_list:
                key = self._salc_key(wfn)
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_wfns.append(wfn)

            # Orthogonalize within this sector
            orthonormal_wfns = self._orthonormalize_wavefunctions(
                unique_wfns, n_qubits=n_qubits, tol=1e-8
            )

            # Create FockSpaceState objects
            for wfn in orthonormal_wfns:
                result_objects.append(
                    FockSpaceState(
                        mol=self.mol,
                        wavefunction=wfn,
                        provider=irrep_provider,
                        irrep=irrep,
                        S2=s2_val,
                        m_s=ms_val,
                        fragment_orbitals=fragment_orbitals,
                    )
                )

        if is_dataframe:
            return FockSpaceState.dataframe(result_objects)
        return result_objects

    @staticmethod
    def _orthonormalize_wavefunctions(
        wfns: list[tequila.QubitWaveFunction], n_qubits: int, tol: float = 1e-8
    ) -> list[tequila.QubitWaveFunction]:
        """Orthonormalize wavefunctions canonically, dropping linear dependencies (overlap eigenvalues below ``tol``)."""
        if len(wfns) == 0:
            return []

        n = len(wfns)
        S = numpy.zeros((n, n), dtype=complex)

        for i in range(n):
            for j in range(i, n):
                val = wfns[i].inner(wfns[j])
                S[i, j] = val
                S[j, i] = numpy.conj(val)

        S = 0.5 * (S + S.conj().T)

        eigvals, U = numpy.linalg.eigh(S)
        keep = eigvals > tol

        if not numpy.any(keep):
            return []

        U_keep = U[:, keep]
        eig_keep = eigvals[keep]

        orthonormal_wfns = []

        for a in range(U_keep.shape[1]):
            coeffs = U_keep[:, a] / numpy.sqrt(eig_keep[a])

            new_wfn = tequila.QubitWaveFunction(n_qubits=n_qubits)

            for coeff, old_wfn in zip(coeffs, wfns):
                if abs(coeff) > 1e-14:
                    new_wfn += coeff * old_wfn

            orthonormal_wfns.append(new_wfn.normalize())

        return orthonormal_wfns


@dataclass
class SpinCGSymmetrizationProcedure(SymmetrizationProcedure):
    """Generate spin-adapted configuration state functions (CSFs) with Clebsch-Gordan coupling.

    There are two uses: ``symmetrize`` returns a full basis of CSFs for the occupation sectors
    of the given determinants, and ``build_physical_state`` builds one specific spin-coupled state.

    Parameters
    ----------
    mol : QuantumChemistryBase
    irrep_provider : IrrepProviderBase, optional
        Provider attached to the produced states, used when the input states have none.
    fragment_orbitals : list of int, optional
        Spatial orbitals to couple. Default is the fragment of the input state, else all orbitals.

    Examples
    --------
    >>> cg = SpinCGSymmetrizationProcedure(mol)
    >>> state = cg.build_physical_state(open_shell_orbitals=[0, 1], S=0, m_s=0, coupling="singlet")
    """

    irrep_provider: IrrepProviderBase | None = None
    fragment_orbitals: list[int] | None = None

    # =========================================================================
    # 1. INTERNAL: CLEBSCH-GORDAN ENGINE
    # =========================================================================
    @staticmethod
    def _cg(j12: int, m12: int, j3: int, m3: int, J2: int, M2: int) -> float:
        """Return the Clebsch-Gordan coefficient ``<j12 m12 j3 m3 | J M>``, with all arguments doubled (integers)."""
        if m12 + m3 != M2:
            return 0.0
        if abs(m12) > j12 or abs(m3) > j3 or abs(M2) > J2:
            return 0.0
        if abs(j12 - j3) > J2 or j12 + j3 < J2:
            return 0.0

        f = math.factorial
        d1, d2, d3 = (j12 + j3 - J2) // 2, (j12 - j3 + J2) // 2, (-j12 + j3 + J2) // 2
        if d1 < 0 or d2 < 0 or d3 < 0:
            return 0.0

        pref_sq = (J2 + 1) * f(d1) * f(d2) * f(d3) / f((j12 + j3 + J2) // 2 + 1)
        for x in (j12 + m12, j12 - m12, j3 + m3, j3 - m3, J2 + M2, J2 - M2):
            pref_sq *= f(x // 2)
        pref = math.sqrt(pref_sq)

        z_min = max(0, -((J2 - j3 + m12) // 2), -((J2 - j12 - m3) // 2))
        z_max = min((j12 + j3 - J2) // 2, (j12 - m12) // 2, (j3 + m3) // 2)
        s = sum(
            ((-1) ** z)
            / (
                f(z)
                * f(d1 - z)
                * f((j12 - m12) // 2 - z)
                * f((j3 + m3) // 2 - z)
                * f((J2 - j3 + m12) // 2 + z)
                * f((J2 - j12 - m3) // 2 + z)
            )
            for z in range(z_min, z_max + 1)
        )
        return pref * s

    @classmethod
    def _coupling_paths(cls, n: int, S2: int) -> list[tuple[int, ...]]:
        """Return all sequences of intermediate spins (doubled values) that couple ``n`` electrons to total spin ``S2``."""
        paths: list[tuple[int, ...]] = []

        def rec(step: int, j: int, path: tuple[int, ...]):
            if step == n:
                if j == S2:
                    paths.append(path)
                return
            for j_next in range(abs(j - 1), j + 2, 2):
                rec(step + 1, j_next, path + (j_next,))

        rec(1, 1, ())
        return paths

    @staticmethod
    def _occupation_patterns(n_orbitals: int, n_electrons: int) -> list[list[int]]:
        """Return all occupation patterns (0, 1 or 2 per orbital) of ``n_electrons`` in ``n_orbitals``."""
        patterns: list[list[int]] = []

        def recurse(idx: int, remaining: int, current: list[int]):
            if idx == n_orbitals:
                if remaining == 0:
                    patterns.append(list(current))
                return
            for occ in range(min(remaining, 2) + 1):
                current.append(occ)
                recurse(idx + 1, remaining - occ, current)
                current.pop()

        recurse(0, n_electrons, [])
        return patterns

    def _build_spin_function(self, orbitals, S2, M2, path, closed_shell_orbitals=None):
        """Build a normalized CSF wavefunction by sequential coupling along ``path``, with the closed-shell orbitals doubly occupied."""
        n = len(orbitals)
        n_qubits = 2 * self.mol.n_orbitals
        terms: dict[int, float] = {}

        closed_shell_mask = 0
        if closed_shell_orbitals:
            for orb in closed_shell_orbitals:
                closed_shell_mask |= (1 << (2 * orb)) | (1 << (2 * orb + 1))

        def recurse(idx, j_so_far, m_so_far, det, coef):
            if coef == 0:
                return
            if idx == n:
                if m_so_far == M2:
                    terms[det | closed_shell_mask] = (
                        terms.get(det | closed_shell_mask, 0.0) + coef
                    )
                return
            orb = orbitals[idx]
            for s2 in (0, 1):
                m_e = 2 * s2 - 1
                new_det = det | (1 << (2 * orb + (1 - s2)))
                if idx == 0:
                    recurse(1, 1, m_so_far + m_e, new_det, coef)
                else:
                    j_next = path[idx - 1]
                    c = self._cg(j_so_far, m_so_far, 1, m_e, j_next, m_so_far + m_e)
                    if c != 0:
                        recurse(idx + 1, j_next, m_so_far + m_e, new_det, coef * c)

        recurse(0, 0, 0, 0, 1.0)
        wfn = tequila.QubitWaveFunction(n_qubits=n_qubits)
        for det, coef in terms.items():
            wfn += coef * tequila.QubitWaveFunction.from_string(
                f"|{format(det, f'0{n_qubits}b')}>"
            )
        return wfn.normalize()

    def _build_spin_function_from_tree(
        self, orbitals, S2, M2, coupling_tree, closed_shell_orbitals=None
    ):
        """Build a normalized CSF wavefunction from an arbitrary binary coupling tree, with the closed-shell orbitals doubly occupied."""
        n = len(orbitals)
        n_qubits = 2 * self.mol.n_orbitals
        terms = {}

        closed_shell_mask = 0
        if closed_shell_orbitals:
            for orb in closed_shell_orbitals:
                closed_shell_mask |= (1 << (2 * orb)) | (1 << (2 * orb + 1))

        def spin_of(tree):
            return 1 if isinstance(tree, int) else tree[2]

        def electrons_of(tree):
            if isinstance(tree, int):
                return [tree]
            return electrons_of(tree[0]) + electrons_of(tree[1])

        def coeff_of(tree, m2):
            if isinstance(tree, int):
                return 1.0
            left, right, s2_int = tree
            mL = sum(m2[e] for e in electrons_of(left))
            mR = sum(m2[e] for e in electrons_of(right))
            if abs(mL) > spin_of(left) or abs(mR) > spin_of(right):
                return 0.0
            cg = self._cg(spin_of(left), mL, spin_of(right), mR, s2_int, mL + mR)
            return cg * coeff_of(left, m2) * coeff_of(right, m2) if cg != 0 else 0.0

        for choices in iproduct((1, -1), repeat=n):
            if sum(choices) != M2:
                continue
            m2 = {i: choices[i] for i in range(n)}
            c = coeff_of(coupling_tree, m2)
            if abs(c) > 1e-12:
                det = closed_shell_mask
                for i in range(n):
                    det |= 1 << (2 * orbitals[i] + (0 if choices[i] == 1 else 1))
                terms[det] = terms.get(det, 0.0) + c

        wfn = tequila.QubitWaveFunction(n_qubits=n_qubits)
        for det, c in terms.items():
            wfn += c * tequila.QubitWaveFunction.from_string(
                f"|{format(det, f'0{n_qubits}b')}>"
            )
        return wfn.normalize()

    # =========================================================================
    # 2. INTERNAL: TREE BUILDERS (Physics -> Tree)
    # =========================================================================
    @staticmethod
    def _tree_from_groups(atom_groups: list[list[int]], total_S2: int) -> tuple:
        """Return a coupling tree with high spin (Hund's rule) inside each group and the groups coupled to ``total_S2``. At most 2 groups are supported."""
        if len(atom_groups) > 2:
            raise NotImplementedError(
                f"_tree_from_groups currently supports at most 2 groups, got {len(atom_groups)}. "
                f"Coupling 3+ high-spin groups requires specifying intermediate spins explicitly."
            )

        def couple_group(electrons):
            if len(electrons) == 1:
                return electrons[0]
            tree = (electrons[0], electrons[1], 2)
            s2 = 2
            for e in electrons[2:]:
                s2 += 1
                tree = (tree, e, s2)
            return tree

        group_trees = [couple_group(g) for g in atom_groups]
        tree = group_trees[0]
        for gt in group_trees[1:]:
            tree = (tree, gt, None)

        def fill_root(t, s2):
            if isinstance(t, int):
                return t
            l, r, s = t
            return (l, r, s2) if s is None else (l, r, s)

        return fill_root(tree, total_S2)

    @staticmethod
    def _tree_singlet_pairs(n_pairs: int) -> tuple:
        """Return a coupling tree that couples adjacent electrons into singlets and combines them."""
        if n_pairs == 1:
            return (0, 1, 0)
        tree = ((0, 1, 0), (2, 3, 0), 0)
        for k in range(2, n_pairs):
            tree = (tree, (2 * k, 2 * k + 1, 0), 0)
        return tree

    # =========================================================================
    # 3. USER API: TARGETED GENERATION (Returns FockSpaceState)
    # =========================================================================
    def build_physical_state(
        self,
        open_shell_orbitals: list[int],
        S: float,
        m_s: float,
        coupling: str | tuple | dict,
        closed_shell_orbitals: list[int] | None = None,
        symmetry_domain: list[int] | None = None,
    ) -> FockSpaceState:
        """Build one spin-coupled state.

        Parameters
        ----------
        open_shell_orbitals : list of int
            Singly occupied spatial orbitals. Electron ``k`` of the coupling is the one in ``open_shell_orbitals[k]``.
        S : float
            Total spin.
        m_s : float
            Spin projection.
        coupling : str or dict or tuple
            How the open-shell electrons are coupled. ``"singlet"`` or ``"triplet"`` (two electrons),
            ``"singlet_pairs"`` (adjacent electrons paired into singlets),
            ``{"type": "high_spin", "groups": [[0, 1, 2], [3, 4, 5]]}`` (high spin in each group, Hund's
            rule, at most two groups) or an explicit coupling tree such as ``((0, 1, 0), (2, 3, 0), 0)``,
            whose nodes are ``(left, right, doubled spin)``.
        closed_shell_orbitals : list of int, optional
            Doubly occupied spatial orbitals.
        symmetry_domain : list of int, optional
            Orbitals on which the symmetry operations may act. Default is all orbitals of the state
            (open and closed shell).

        Returns
        -------
        FockSpaceState

        Raises
        ------
        ValueError
            For orbital indices out of range, an empty ``open_shell_orbitals``, a coupling that does
            not match the number of electrons, or an unknown coupling.
        """
        # --- Validation ---
        n_orb = self.mol.n_orbitals
        for orb in open_shell_orbitals:
            if orb < 0 or orb >= n_orb:
                raise ValueError(
                    f"Orbital index {orb} out of range. "
                    f"Molecule has {n_orb} orbitals (indices 0–{n_orb - 1})."
                )
        if closed_shell_orbitals:
            for orb in closed_shell_orbitals:
                if orb < 0 or orb >= n_orb:
                    raise ValueError(
                        f"Closed-shell orbital index {orb} out of range. "
                        f"Molecule has {n_orb} orbitals (indices 0–{n_orb - 1})."
                    )

        # --- Validate electron count vs coupling scheme ---
        n_open = len(open_shell_orbitals)
        if n_open == 0:
            raise ValueError("open_shell_orbitals cannot be empty.")
        if isinstance(coupling, str):
            if coupling in ("singlet", "triplet") and n_open != 2:
                raise ValueError(
                    f"coupling='{coupling}' requires exactly 2 open-shell orbitals, got {n_open}."
                )
            if coupling == "singlet_pairs" and n_open % 2 != 0:
                raise ValueError(
                    f"coupling='singlet_pairs' requires an even number of open-shell orbitals, got {n_open}."
                )
        if isinstance(coupling, dict) and coupling.get("type") == "high_spin":
            grouped = [e for g in coupling["groups"] for e in g]
            if sorted(grouped) != list(range(n_open)):
                raise ValueError(
                    f"high_spin groups must cover exactly the indices 0..{n_open - 1}. "
                    f"Got {sorted(grouped)}."
                )
        S2, M2 = int(2 * S), int(2 * m_s)

        # --- named patterns ---
        if isinstance(coupling, str) and coupling == "singlet":
            tree = (0, 1, 0)
        elif isinstance(coupling, str) and coupling == "triplet":
            tree = (0, 1, 2)
        elif isinstance(coupling, str) and coupling == "singlet_pairs":
            tree = self._tree_singlet_pairs(len(open_shell_orbitals) // 2)
        elif isinstance(coupling, dict) and coupling.get("type") == "high_spin":
            tree = self._tree_from_groups(coupling["groups"], S2)
        elif isinstance(coupling, tuple):
            tree = coupling
        else:
            raise ValueError(f"Unknown coupling scheme: {coupling}")

        wfn = self._build_spin_function_from_tree(
            open_shell_orbitals, S2, M2, tree, closed_shell_orbitals
        )

        # Inside build_physical_state, at the very end before returning:
        all_orbitals = sorted(set((closed_shell_orbitals or []) + open_shell_orbitals))

        # Fallback to all_orbitals if symmetry_domain is None to preserve old behavior
        final_fragment = (
            symmetry_domain if symmetry_domain is not None else all_orbitals
        )

        return FockSpaceState(
            mol=self.mol,
            wavefunction=wfn,
            provider=self.irrep_provider,
            S2=float(S * (S + 1)),
            m_s=float(m_s),
            fragment_orbitals=final_fragment,
        )

    # =========================================================================
    # 4. USER API: FULL BASIS GENERATION (Returns list[FockSpaceState])
    # =========================================================================
    def symmetrize(
        self, states: list[FockSpaceState] | pd.DataFrame
    ) -> list[FockSpaceState] | pd.DataFrame:
        """Return a complete basis of CSFs for every occupation sector of the input states.

        The number of electrons and the orbitals are taken from the first state. Each CSF carries its
        coupling path as ``coupling_path``.

        Parameters
        ----------
        states : list of FockSpaceState or pandas.DataFrame
            States to process, e.g. from ``FockSpaceState.non_ionic_states`` (a DataFrame as made by
            ``FockSpaceState.dataframe``).

        Returns
        -------
        list of FockSpaceState or pandas.DataFrame
            CSFs with ``S2`` and ``m_s`` set.
        """
        is_dataframe = isinstance(states, pd.DataFrame)
        state_list = (
            list(states.itertuples(index=False, name="FockSpaceState"))
            if is_dataframe
            else list(states)
        )

        n_orb = len(state_list[0].mo_occ)  # derive from the actual input state
        fragment_orbitals = (
            getattr(state_list[0], "fragment_orbitals", None)
            or self.fragment_orbitals
            or list(range(n_orb))
        )
        irrep_provider = state_list[0].irrep_provider or self.irrep_provider
        n_electrons = round(sum(state_list[0].mo_occ[i] for i in fragment_orbitals))

        all_states: list[FockSpaceState] = []
        for mo_occ in self._occupation_patterns(len(fragment_orbitals), n_electrons):
            open_shell = [
                fragment_orbitals[i] for i, occ in enumerate(mo_occ) if occ == 1
            ]
            closed_shell = [
                fragment_orbitals[i] for i, occ in enumerate(mo_occ) if occ == 2
            ]

            if not open_shell:
                det = sum(
                    (1 << (2 * orb)) | (1 << (2 * orb + 1)) for orb in closed_shell
                )
                wfn = tequila.QubitWaveFunction.from_string(
                    f"|{format(det, f'0{2 * self.mol.n_orbitals}b')}>"
                )
                state = FockSpaceState(
                    self.mol,
                    wfn,
                    irrep_provider,
                    S2=0.0,
                    m_s=0.0,
                    fragment_orbitals=fragment_orbitals,
                )
                state.coupling_path = ()
                all_states.append(state)
            else:
                for M2 in range(-len(open_shell), len(open_shell) + 1, 2):
                    for S2 in range(abs(M2), len(open_shell) + 1, 2):
                        for path in self._coupling_paths(len(open_shell), S2):
                            wfn = self._build_spin_function(
                                open_shell, S2, M2, path, closed_shell
                            )
                            state = FockSpaceState(
                                self.mol,
                                wfn,
                                irrep_provider,
                                S2=float((S2 / 2) * (S2 / 2 + 1)),
                                m_s=M2 / 2,
                                fragment_orbitals=fragment_orbitals,
                            )
                            state.coupling_path = tuple(p / 2 for p in path)
                            all_states.append(state)

        return FockSpaceState.dataframe(all_states) if is_dataframe else all_states
