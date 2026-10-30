from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations, groupby
from itertools import product as iproduct
from typing import TYPE_CHECKING

import numpy
import pandas as pd
import tequila
from scipy.sparse.csgraph import connected_components
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan

from .fock_space_state import FockSpaceState
from .irrep_provider import IrrepProviderBase
from .point_group_representation import PointGroupRepresentation

if TYPE_CHECKING:
    # only for type hints, importing it at runtime would be circular
    from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase


@dataclass
class SymmetrizationProcedure(ABC):
    mol: QuantumChemistryBase

    @abstractmethod
    def symmetrize(self, states: pd.DataFrame) -> pd.DataFrame:
        """Given a state in Fock space, returns a new state that is a symmetry-adapted linear combination of the original state and its images under the symmetry operations of the point group."""


@dataclass
class SpinSymmetrizationProcedure(SymmetrizationProcedure):
    """A symmetrization procedure that symmetrizes states with respect to spin."""

    def symmetrize(
        self, states: list[FockSpaceState] | pd.DataFrame
    ) -> list[FockSpaceState] | pd.DataFrame:
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
            n_components, labels = connected_components(abs(mini_matrix) > 1e-10)
            components = [numpy.flatnonzero(labels == c) for c in range(n_components)]

            for component in components:
                sub_matrix = mini_matrix[numpy.ix_(component, component)]
                eigvals, eigvecs = numpy.linalg.eigh(sub_matrix)

                for k in range(len(eigvals)):
                    eigvec = eigvecs[:, k]

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
                            S2=numpy.round(eigvals[k], decimals=10),
                            m_s=ms_val,
                        )
                    )
        if is_dataframe:
            return FockSpaceState.dataframe(diagonalised_states)
        return diagonalised_states


@dataclass
class SymmetryAdaptedLinearCombintationSymmetrization(SymmetrizationProcedure):
    """A symmetrization procedure that symmetrizes states by constructing symmetry-adapted linear combinations."""

    pgr: PointGroupRepresentation | None = None

    def __post_init__(self):
        self.pg = self.mol.point_group
        if self.pgr is None:
            self.pgr = self.mol.mo_representation()

    def symmetrize(
        self, states: list[FockSpaceState] | pd.DataFrame
    ) -> list[FockSpaceState] | pd.DataFrame:
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

        n_orbitals = self.mol.n_orbitals
        n_qubits = 2 * n_orbitals

        # The operations keep the numbers of up and down electrons, so the states of a block (n_up, n_down) are
        # matrices W[a, b] over its up strings a and down strings b. Written with the up operators before the down
        # ones, the operation D acts as signs * (M_up @ (signs * W) @ M_down.T), where M[T, S] = det(D[T, S]) and
        # signs[a, b] is the fermionic sign of moving the down operators behind the up ones
        blocks = {}

        # the spin operator is built once for all seeds
        s2_op = self.mol.make_s2_op()

        # Group projected SALCs by (irrep, S2, m_s, block) to preserve spin purity!
        SALC_by_sector: dict[tuple, list[numpy.ndarray]] = {}

        for state in state_list:
            terms = []
            for bitstring, coef in state.wavefunction.items():
                bs_str = format(int(bitstring), f"0{n_qubits}b")
                up = tuple(i for i in range(n_orbitals) if bs_str[2 * i] == "1")
                down = tuple(i for i in range(n_orbitals) if bs_str[2 * i + 1] == "1")
                terms.append((up, down, coef))
            block = {(len(up), len(down)) for up, down, _ in terms}
            if len(block) != 1:
                raise ValueError(
                    "The SALC symmetrization needs states with fixed numbers of up and down electrons."
                )
            block = block.pop()

            # Extract spin quantum numbers from the seed state, m_s follows from the block. A single determinant has
            # <S^2> = m_s (m_s + 1) + n_down - (number of doubly occupied orbitals), other seeds need the operator
            ms_val = (block[0] - block[1]) / 2
            if not isinstance(state, FockSpaceState) or state._S2 is not None:
                s2_val = state.S2
            elif len(terms) == 1:
                up, down, _ = terms[0]
                s2_val = ms_val * (ms_val + 1) + len(down) - len(set(up) & set(down))
            else:
                s2_val = numpy.round(
                    state.wavefunction.inner(s2_op(state.wavefunction)).real,
                    decimals=10,
                )

            if block not in blocks:
                strings = [list(combinations(range(n_orbitals), n)) for n in block]
                signs = numpy.array(
                    [
                        [(-1) ** sum(j < i for i in a for j in b) for b in strings[1]]
                        for a in strings[0]
                    ]
                )
                indices = [
                    numpy.array(string, dtype=int).reshape(len(string), n)
                    for string, n in zip(strings, block)
                ]
                matrices = [
                    [
                        numpy.linalg.det(
                            D[index[:, None, :, None], index[None, :, None, :]]
                        )
                        for index in indices
                    ]
                    for D in self.pgr.operations.values()
                ]
                positions = [{string: k for k, string in enumerate(s)} for s in strings]
                blocks[block] = (strings, signs, matrices, positions)
            strings, signs, matrices, positions = blocks[block]

            W = numpy.zeros((len(strings[0]), len(strings[1])), dtype=complex)
            for up, down, coef in terms:
                W[positions[0][up], positions[1][down]] += coef

            # apply all symmetry operations to the state
            resulting_state_list = [
                signs * (M_up @ (signs * W) @ M_down.T) for M_up, M_down in matrices
            ]

            for irrep, characters in self.pg.character_table.dict.items():
                state_SALC = sum(
                    character * W_op
                    for character, W_op in zip(characters, resulting_state_list)
                    if character != 0
                )
                norm = numpy.linalg.norm(state_SALC)
                if norm**2 > 1e-10:
                    sector = (irrep, s2_val, ms_val, block)
                    if sector not in SALC_by_sector:
                        SALC_by_sector[sector] = []
                    SALC_by_sector[sector].append((state_SALC / norm).ravel())

        result_objects = []
        # the states already kept for each (irrep, m_s, block), as rows
        kept_wfns: dict[tuple, numpy.ndarray] = {}

        for (irrep, s2_val, ms_val, block), wfn_list in SALC_by_sector.items():
            strings = blocks[block][0]

            # Deduplicate copies within this sector, also when they differ by a global phase:
            # the key of a state is its rounded coefficients with the phase of the first one divided out
            unique_wfns = {}
            for wfn in wfn_list:
                first = wfn[numpy.argmax(numpy.abs(wfn) > 1e-12)]
                key = (numpy.round(wfn * abs(first) / first, 8) + 0j).tobytes()
                unique_wfns.setdefault(key, wfn)
            V = numpy.array(list(unique_wfns.values()))

            # Remove what the sectors with another S2 already span: when the operations mix orbitals, seeds that
            # are no spin eigenstates (e.g. open-shell determinants) project onto states of several sectors
            kept = kept_wfns.get((irrep, ms_val, block))
            if kept is not None:
                V = V - (V @ kept.conj().T) @ kept

            # Orthonormalize within this sector (canonical / Löwdin-style), removing
            # linear dependencies with overlap eigenvalues below 1e-8
            S = V.conj() @ V.T
            S = 0.5 * (S + S.conj().T)
            eigvals, U = numpy.linalg.eigh(S)
            keep = eigvals > 1e-8
            orthonormal_wfns = (U[:, keep] / numpy.sqrt(eigvals[keep])).T @ V
            orthonormal_wfns /= numpy.linalg.norm(orthonormal_wfns, axis=1)[:, None]
            kept_wfns[(irrep, ms_val, block)] = (
                orthonormal_wfns
                if kept is None
                else numpy.vstack([kept, orthonormal_wfns])
            )

            for vector in orthonormal_wfns:
                new_wfn = tequila.QubitWaveFunction(n_qubits=n_qubits)
                for (a, b), c in zip(
                    numpy.ndindex(len(strings[0]), len(strings[1])), vector
                ):
                    if abs(c) > 1e-14:
                        # qubits 2 * i (up) and 2 * i + 1 (down) of each orbital
                        bits = ["0"] * n_qubits
                        for i in strings[0][a]:
                            bits[2 * i] = "1"
                        for j in strings[1][b]:
                            bits[2 * j + 1] = "1"
                        new_wfn[int("".join(bits), 2)] = c

                result_objects.append(
                    FockSpaceState(
                        mol=self.mol,
                        wavefunction=new_wfn,
                        provider=irrep_provider,
                        irrep=irrep,
                        S2=s2_val,
                        m_s=ms_val,
                    )
                )

        if is_dataframe:
            return FockSpaceState.dataframe(result_objects)
        return result_objects


@dataclass
class SpinCGSymmetrizationProcedure(SymmetrizationProcedure):
    """
    Generates spin-adapted Configuration State Functions (CSFs) using
    Clebsch-Gordan coupling.

    Workflows:
    1. `symmetrize(det_states)`: Takes a list of determinant FockSpaceStates
       and returns a complete basis of CSFs across all occupation sectors.
    2. `build_physical_state(...)`: Directly constructs a specific spin-coupled
       state based on physical intuition and returns a FockSpaceState.
    """

    irrep_provider: IrrepProviderBase | None = None

    def build_physical_state(
        self,
        open_shell_orbitals: list[int],
        S: float,
        m_s: float,
        coupling: str | tuple | dict,
        closed_shell_orbitals: list[int] | None = None,
    ) -> FockSpaceState:
        """
        Builds a specific spin-coupled state and returns a FockSpaceState.

        Parameters:
        - open_shell_orbitals: Orbitals with 1 electron.
        - closed_shell_orbitals: Orbitals with 2 electrons.
        - S, m_s: Target total spin and projection.
        - coupling:
            • "singlet_pairs" (pairs coupled to singlets)
            • {"type": "high_spin", "groups": [[0,1,2], [3,4,5]]} (Hund's rule)
            • A raw tree tuple ((0,1,0), (2,3,0), 0)
        """
        n_orb = self.mol.n_orbitals
        n_open = len(open_shell_orbitals)
        # a negative index would silently wrap around to the last orbitals
        for orb in open_shell_orbitals + (closed_shell_orbitals or []):
            if not 0 <= orb < n_orb:
                raise ValueError(
                    f"Orbital index {orb} out of range, the molecule has {n_orb} orbitals."
                )
        S2, M2 = int(2 * S), int(2 * m_s)

        # --- coupling tree ---
        # A tree is an electron index (spin 1/2) or a tuple (left, right, doubled spin)
        # that couples the electrons of the left and right subtrees to this spin.
        if isinstance(coupling, str) and coupling == "singlet":
            tree = (0, 1, 0)
        elif isinstance(coupling, str) and coupling == "triplet":
            tree = (0, 1, 2)
        elif isinstance(coupling, str) and coupling == "singlet_pairs":
            # adjacent electrons coupled to singlets, then the singlets combined
            tree = (0, 1, 0)
            for k in range(1, n_open // 2):
                tree = (tree, (2 * k, 2 * k + 1, 0), 0)
        elif isinstance(coupling, dict) and coupling.get("type") == "high_spin":
            # Hund's rule: high spin within each group, then the groups coupled to S
            group_trees = []
            for electrons in coupling["groups"]:
                group_tree = electrons[0]
                for k, e in enumerate(electrons[1:], start=2):
                    group_tree = (group_tree, e, k)  # k electrons have spin k / 2
                group_trees.append(group_tree)
            tree = (
                group_trees[0]
                if len(group_trees) == 1
                else (group_trees[0], group_trees[1], S2)
            )
        elif isinstance(coupling, (tuple, int)):
            tree = coupling
        else:
            raise ValueError(f"Unknown coupling scheme: {coupling}")

        # electrons and doubled spin of every subtree, children before their parents
        subtrees = {}
        stack = [tree]
        while stack:
            t = stack[-1]
            if isinstance(t, int):
                subtrees[t] = ([t], 1)
                stack.pop()
            elif t[0] in subtrees and t[1] in subtrees:
                subtrees[t] = (subtrees[t[0]][0] + subtrees[t[1]][0], t[2])
                stack.pop()
            else:
                stack += [child for child in t[:2] if child not in subtrees]

        # the coupling has to contain every open-shell electron exactly once, otherwise
        # the electrons left out would not be coupled to the spin S
        if sorted(subtrees[tree][0]) != list(range(n_open)):
            raise ValueError(
                f"The coupling {coupling} does not couple the {n_open} open-shell electrons "
                f"0..{n_open - 1} exactly once (high_spin couples at most 2 groups)."
            )

        # every determinant with the right m_s gets the product of the Clebsch-Gordan
        # coefficients of all couplings in the tree (all spins and projections doubled)
        n_qubits = 2 * n_orb
        terms = {}
        for choices in iproduct((1, -1), repeat=n_open):
            if sum(choices) != M2:
                continue
            coef = 1.0
            for t, (_, s2) in subtrees.items():
                if isinstance(t, tuple):
                    (left_electrons, left_s2), (right_electrons, right_s2) = (
                        subtrees[t[0]],
                        subtrees[t[1]],
                    )
                    m_left = sum(choices[e] for e in left_electrons)
                    m_right = sum(choices[e] for e in right_electrons)
                    coef *= float(
                        clebsch_gordan(
                            Rational(left_s2, 2),
                            Rational(right_s2, 2),
                            Rational(s2, 2),
                            Rational(m_left, 2),
                            Rational(m_right, 2),
                            Rational(m_left + m_right, 2),
                        )
                    )
            if abs(coef) > 1e-12:
                # qubits 2 * orb (up) and 2 * orb + 1 (down) of each orbital
                det = ["0"] * n_qubits
                for orb in closed_shell_orbitals or []:
                    det[2 * orb] = det[2 * orb + 1] = "1"
                for i, orb in enumerate(open_shell_orbitals):
                    det[2 * orb + (0 if choices[i] == 1 else 1)] = "1"
                det = "".join(det)
                terms[det] = terms.get(det, 0.0) + coef

        wfn = tequila.QubitWaveFunction(n_qubits=n_qubits)
        for det, coef in terms.items():
            wfn += coef * tequila.QubitWaveFunction.from_string(f"|{det}>")

        return FockSpaceState(
            mol=self.mol,
            wavefunction=wfn.normalize(),
            provider=self.irrep_provider,
            S2=float(S * (S + 1)),
            m_s=float(m_s),
        )

    def symmetrize(
        self, states: list[FockSpaceState] | pd.DataFrame
    ) -> list[FockSpaceState] | pd.DataFrame:
        """
        Generates all spin-adapted CSFs across every occupation sector
        of the provided determinant states.
        """
        is_dataframe = isinstance(states, pd.DataFrame)
        state_list = (
            list(states.itertuples(index=False, name="FockSpaceState"))
            if is_dataframe
            else list(states)
        )

        irrep_provider = state_list[0].irrep_provider or self.irrep_provider
        n_electrons = round(sum(state_list[0].mo_occ))

        all_states: list[FockSpaceState] = []
        # all occupation patterns (0, 1 or 2 electrons per orbital) with n_electrons
        for mo_occ in iproduct(range(3), repeat=self.mol.n_orbitals):
            if sum(mo_occ) != n_electrons:
                continue
            open_shell = [i for i, occ in enumerate(mo_occ) if occ == 1]
            closed_shell = [i for i, occ in enumerate(mo_occ) if occ == 2]
            n_open = len(open_shell)

            if not open_shell:
                det = "".join(
                    "11" if orb in closed_shell else "00"
                    for orb in range(self.mol.n_orbitals)
                )
                wfn = tequila.QubitWaveFunction.from_string(f"|{det}>")
                state = FockSpaceState(
                    self.mol,
                    wfn,
                    irrep_provider,
                    S2=0.0,
                    m_s=0.0,
                )
                state.coupling_path = ()
                all_states.append(state)
                continue

            # genealogical coupling paths: the doubled spin after each added electron,
            # which goes up or down by one with every electron (and stays >= 0)
            paths = [(1,)]
            for _ in range(n_open - 1):
                paths = [
                    path + (j,)
                    for path in paths
                    for j in range(abs(path[-1] - 1), path[-1] + 2, 2)
                ]

            for M2 in range(-n_open, n_open + 1, 2):
                for S2 in range(abs(M2), n_open + 1, 2):
                    for path in paths:
                        if path[-1] != S2:
                            continue
                        # a genealogical path couples the electrons one by one,
                        # i.e. it is the tree (((0, 1, j1), 2, j2), 3, j3) ...
                        tree = 0
                        for k, j in enumerate(path[1:], start=1):
                            tree = (tree, k, j)
                        state = self.build_physical_state(
                            open_shell, S2 / 2, M2 / 2, tree, closed_shell
                        )
                        state.irrep_provider = irrep_provider
                        state.coupling_path = tuple(p / 2 for p in path[1:])
                        all_states.append(state)

        return FockSpaceState.dataframe(all_states) if is_dataframe else all_states
