"""
Collections of Fermion-to-Qubit encodings known to tequila
Adapted to the encoding
"""

from __future__ import annotations

from copy import deepcopy

import numpy
import openfermion
from tequila import QubitHamiltonian
from tequila.circuit.circuit import QCircuit
from tequila.circuit.gates import X
from tequila.hamiltonian.paulis import Sm, Sp, Z

from sunrise.molecules.qubit_base import encodings as qubit_encodings
from sunrise.molecules.qubit_base.encodings import EncodingBase as EB


def known_encodings():
    # convenience for testing and I/O
    encodings = {
        "JordanWigner": JordanWigner,
        "BravyiKitaev": BravyiKitaev,
        "BravyiKitaevTree": BravyiKitaevTree,
        "TaperedBinary": TaperedBinary,
        "TaperedBravyiKitaev": TaperedBravyiKitaev,
    }
    # aliases
    encodings = {
        **encodings,
        "ReorderedJordanWigner": lambda **kwargs: JordanWigner(
            up_then_down=True, **kwargs
        ),
        "ReorderedBravyiKitaev": lambda **kwargs: BravyiKitaev(
            up_then_down=True, **kwargs
        ),
        "ReorderedBravyiKitaevTree": lambda **kwargs: BravyiKitaevTree(
            up_then_down=True, **kwargs
        ),
    }
    return {
        k.replace("_", "").replace("-", "").upper(): v for k, v in encodings.items()
    }


class JordanWigner(EB):
    _ucc_support = True

    def __init__(
        self, n_electrons, n_orbitals, select: dict, up_then_down=False, *args, **kwargs
    ):
        if "condense" in kwargs:
            self.condense = kwargs["condense"]
            kwargs.pop("condense")
        else:
            self.condense = True
        if "two_qubit" in kwargs:
            self.two_qubit = kwargs["two_qubit"]
            kwargs.pop("two_qubit")
        else:
            self.two_qubit = False
        if self.two_qubit:
            self.condense = False
        super().__init__(n_electrons, n_orbitals, up_then_down)
        self.update_select(select)

    def __call__(
        self, fermion_operator: openfermion.FermionOperator, *args, **kwargs
    ) -> QubitHamiltonian:
        """
        :param fermion_operator:
            an openfermion FermionOperator
        :return:
            The openfermion QubitOperator of this class ecoding
        """
        fop = self.do_transform(*args, fermion_operator=fermion_operator, **kwargs)
        return self.post_processing(fop)

    def post_processing(self, op, *args, **kwargs):
        return op

    def do_transform(
        self,
        fermion_operator: openfermion.FermionOperator | list | tuple,
        *args,
        **kwargs,
    ) -> QubitHamiltonian:
        """
        Transform the fermi_operator list in its respective gates similar to openfermion function but taking
        into account the mixed condification
        return: qop: the sum of the different operator product multiplied by a prefactor
        """

        def cre(i: int) -> QubitHamiltonian:
            """
            Internal function
            Creates a creation operator on the JW codification acting on the i-th SO
            :param i: number of the SO in full-Fermionic-not-upthendown over the creation operators acts
            :return :creation operator acting on the corresponding qubits
            """
            d = self.pos
            a = Sm(d[i])
            if self.two_qubit or (self.select[i // 2] == "F"):
                for n in self.FER_SO[: self.FER_SO.index(d[i])]:
                    a *= Z(n)
            return a

        def anni(i: int) -> QubitHamiltonian:
            """
            Internal function
            Creates a annihilation operator on the JW codification acting on the i-th SO
            :param i: number of the SO in full-Fermionic-not-upthendown over the creation operators acts
            :return :annihilation operator acting on the corresponding qubits
            """
            d = self.pos
            a = Sp(d[i])
            if self.two_qubit or (self.select[i // 2] == "F"):
                for n in self.FER_SO[: self.FER_SO.index(d[i])]:
                    a *= Z(n)
            return a

        qop = QubitHamiltonian()
        if type(fermion_operator) is openfermion.FermionOperator:
            lista = list(fermion_operator.terms.items())
        else:
            lista = [*fermion_operator]
        for pair in lista:
            index = pair[0]
            temp = pair[1]
            fer = []
            for term in index:
                if (not self.two_qubit) and (
                    self.select[term[0] // 2] == "B" and (term[0] % 2)
                ):
                    pass
                elif type(self) in (BravyiKitaev, BravyiKitaevTree) and (
                    self.two_qubit or self.select[term[0] // 2] == "F"
                ):
                    # encoded below with the qubit_base encoding of the same name, the modes are the positions in FER_SO
                    fer.append((self.FER_SO.index(self.pos[term[0]]), term[1]))
                else:
                    if term[1]:
                        temp *= cre(term[0])
                    else:
                        temp *= anni(term[0])
            if len(fer):
                # the bosonic operators have no strings, they commute with the fermionic ones
                temp *= getattr(qubit_encodings, type(self).__name__)(
                    n_electrons=self.n_electrons, n_orbitals=len(self.FER_SO) // 2
                )(openfermion.FermionOperator(tuple(fer))).map_qubits(
                    dict(enumerate(self.FER_SO))
                )
            qop += temp
        return qop

    def hcb_to_me(self, *args, **kwargs):
        """
        all: CNOT for all orbs, including FER
        bos: CNOT only for the Bos orbitals
        """
        if "all" in kwargs:
            alle = kwargs["all"]
            kwargs.pop("all")
        else:
            alle = False
        if "bos" in kwargs:
            bos = kwargs["bos"]
            kwargs.pop("bos")
        else:
            bos = False

        if bos and self.condense:
            raise ValueError("hcb_to_me called with bos with condensed encoding")
        if bos and alle:
            raise ValueError("hcb_to_me called with all and bos both True")
        if alle and self.condense:
            raise ValueError("hcb_to me asked for all orbitals in condensed encoding")
        U = QCircuit()
        pos = deepcopy(self.pos)
        if not self.two_qubit and bos:
            for i in range(self.n_orbitals):
                if self.select[i] == "B":
                    pos[2 * i + 1] = (
                        i
                        + self.n_orbitals * self.up_then_down
                        + (not self.up_then_down) * (i + 1)
                    )
        for i in range(self.n_orbitals):
            if bos and self.select[i] == "B":
                U += X(target=pos[2 * i + 1], control=pos[2 * i])
            if not bos and (alle or self.two_qubit or self.select[i] == "F"):
                U += X(target=pos[2 * i + 1], control=pos[2 * i])
        if not bos:
            U += self.jw_to_me()
        return U

    def me_to_jw(self) -> QCircuit:
        return self.jw_to_me().dagger()

    def jw_to_me(self) -> QCircuit:
        if type(self) not in (BravyiKitaev, BravyiKitaevTree) or not len(self.FER_SO):
            return QCircuit()
        return (
            getattr(qubit_encodings, type(self).__name__)(
                n_electrons=self.n_electrons, n_orbitals=len(self.FER_SO) // 2
            )
            .jw_to_me()
            .map_qubits(dict(enumerate(self.FER_SO)))
        )

    def map_state(self, state: list, *args, **kwargs):
        state = state + [0] * (2 * self.n_orbitals - len(state))
        if not all(
            state[2 * i] == state[2 * i + 1]
            for i in range(self.n_orbitals)
            if self.select[i] == "B"
        ):
            raise ValueError("Incompatible state and encoding selection")
        if type(self) in (BravyiKitaev, BravyiKitaevTree):
            return EB.map_state(self, state)
        for i in range(len(state) // 2):
            if self.select[i] == "B":
                if self.two_qubit:
                    pass
                else:
                    state[2 * i + 1] = 0
        if self.up_then_down:
            up = [state[2 * i] for i in range(self.n_orbitals)]
            down = [state[2 * i + 1] for i in range(self.n_orbitals)]
            if self.condense:
                for i in self.select:
                    if self.select[i] == "B":
                        down.pop(i)
            return up + down
        elif self.condense:
            for i in [*self.select.keys()][::-1]:
                if self.select[i] == "B":
                    state.pop(2 * i + 1)
            return state
        else:
            return state

    def up(self, i):
        return self.pos[2 * i]

    def down(self, i):
        try:
            return self.pos[2 * i + 1]
        except KeyError:
            return None

    def update_select(
        self, select: str | dict | list | tuple, n_orb: int | None = None
    ):
        """
        Parameters
        ----------
        select: codification of the transformation for each MO.

        Returns
        -------
        Updates the MO cofication data. Returns Instance of the class
        """
        if n_orb is None:
            n_orb = self.n_orbitals

        def verify_selection_str(select: str, n_orb: int) -> dict:
            """
            Internal function
            Checks if the orbital selection string has the appropiated lenght, otherwise corrects it
            :return : corrected selection dict
            """
            sel = {}
            if len(select) >= n_orb:
                select = select[:n_orb]
            else:
                select += (n_orb - len(select)) * "B"
            for i, enc in enumerate(select):
                if enc in ["F", "B"]:
                    sel.update({i: enc})
                else:
                    raise ValueError(
                        f"Warning, encoding character not recognised on position {i}: {enc}.\n Please choose between F (Fermionic) and B (Bosonic)."
                    )
            return sel

        def verify_selection_dict(select: dict, n_orb: int) -> dict:
            """
            Internal function
            Checks if the orbital selection dictionary has the appropiated lenght, otherwise corrects it
            :return : corrected selection dict
            """
            sel = {}
            for i in range(n_orb):
                if i in [*select.keys()]:
                    if select[i] in ["F", "B"]:
                        sel.update({i: select[i]})
                    else:
                        raise ValueError(
                            f"Warning, encoding character not recognised on entry { ({i: sel[i]}) }.\n Please choose between F (Fermionic) and B (Bosonic)."
                        )
                else:
                    sel.update({i: "B"})
            return sel

        def verify_selection_list(select: list | tuple, n_orb: int) -> dict:
            """
            Internal function
            Checks if the orbital selection string has the appropiated lenght, otherwise corrects it
            :return : corrected selection dict
            """
            select = [*select]
            sel = {}
            if len(select) >= n_orb:
                select = select[:n_orb]
            else:
                select = select + (n_orb - len(select)) * ["B"]
            for i in range(len(select)):
                if select[i] in ["F", "B"]:
                    sel.update({i: select[i]})
                else:
                    raise ValueError(
                        f"Warning, encoding character not recognised on position {i}: {select[i]}.\n Please choose between F (Fermionic) and B (Bosonic)."
                    )
            return sel

        def select_to_list(sel):
            """
            Internal function
            Read the select string to make the proper Fer and Bos lists
            :return : list of MOs for the Bos, MOs and SOs for the Fer space
            """
            hcb = 0
            FER_SO = []
            pos = {}
            up = self.up_then_down
            two = self.two_qubit
            for i in sel:
                if sel[i] == "B":
                    pos[2 * i] = i + (i - hcb) * (not up)
                    if two:
                        pos[2 * i + 1] = i + self.n_orbitals * up + (not up) * (i + 1)
                        FER_SO.append(pos[2 * i])
                        FER_SO.append(pos[2 * i + 1])
                    elif self.condense:
                        hcb += 1
                else:
                    pos[2 * i] = i + (i - hcb) * (not up)
                    pos[2 * i + 1] = i - hcb + up * self.n_orbitals + (not up) * (i + 1)
                    FER_SO.append(pos[2 * i])
                    FER_SO.append(pos[2 * i + 1])
            return FER_SO, pos

        if type(select) is dict:
            select = verify_selection_dict(select=select, n_orb=n_orb)
        elif type(select) is str:
            select = verify_selection_str(select=select, n_orb=n_orb)
        elif type(select) is list or type(select) is tuple:
            select = verify_selection_list(select=select, n_orb=n_orb)
        else:
            raise TypeError(
                f"Warning, encoding format not recognised: {type(select)}.\n Please choose either a Str, Dict, List or Tuple."
            )
        self.FER_SO, self.pos = select_to_list(select)
        self.select = select


class BravyiKitaev(JordanWigner):
    """OpenFermion::bravyi_kitaev on the fermionic orbitals"""


class BravyiKitaevTree(JordanWigner):
    """OpenFermion::bravyi_kitaev_tree on the fermionic orbitals"""


class TaperedLinearEncoding(JordanWigner):
    """
    Encodings which are a linear map q = A n + s (mod 2) of the hybrid JordanWigner qubits n.
    The tapered qubits store the fixed electron parities, so they are projected out and stay in |0>.
    Subclasses implement linear_map.
    """

    def linear_map(self) -> tuple[dict, dict, list]:
        """
        :return: rows: {qubit: JordanWigner qubits whose sum it stores}, qubits not present are unchanged
                 shift: {qubit: constant bit added}
                 tapered: list of tapered qubits
        """
        raise NotImplementedError(f"{type(self).__name__}::linear_map")

    def matrix(self) -> tuple[list, numpy.ndarray, numpy.ndarray]:
        """
        :return: the qubits, A and s on that ordering
        """
        rows, shift, _ = self.linear_map()
        qubits = sorted(set(self.pos.values()))
        index = {q: k for k, q in enumerate(qubits)}
        A = numpy.eye(len(qubits), dtype=int)
        for q, row in rows.items():
            A[index[q]] = 0
            for c in row:
                A[index[q], index[c]] ^= 1
        s = numpy.array([shift.get(q, 0) for q in qubits], dtype=int)
        return qubits, A, s

    @staticmethod
    def row_reduction(A: numpy.ndarray) -> list:
        """
        :return: row operations (target, control) row_t += row_c (mod 2) that bring A to the identity
        """
        A = A.copy()
        ops = []
        for col in range(len(A)):
            if not A[col, col]:
                pivot = next(r for r in range(col + 1, len(A)) if A[r, col])
                A[col] ^= A[pivot]
                ops.append((col, pivot))
            for r in range(len(A)):
                if r != col and A[r, col]:
                    A[r] ^= A[col]
                    ops.append((r, col))
        return ops

    def do_transform(
        self,
        fermion_operator: openfermion.FermionOperator | list | tuple,
        *args,
        **kwargs,
    ) -> QubitHamiltonian:
        """
        JordanWigner operator conjugated with jw_to_me and projected onto the tapered qubits in |0>
        """
        qop = super().do_transform(fermion_operator, *args, **kwargs)
        if not len(self.FER_SO):
            return qop
        _, _, tapered = self.linear_map()
        qubits, A, s = self.matrix()
        index = {q: k for k, q in enumerate(qubits)}
        # n = A^-1 (q + s)
        Ainv = numpy.eye(len(qubits), dtype=int)
        for t, c in self.row_reduction(A):
            Ainv[t] ^= Ainv[c]

        def image(q: int, p: str) -> openfermion.QubitOperator:
            # conjugation of the single qubit pauli p on q by jw_to_me
            if p == "Y":
                return 1j * image(q, "X") * image(q, "Z")
            if p == "X":  # flips n_q: flips every qubit storing it
                return openfermion.QubitOperator(
                    tuple((qubits[r], "X") for r in numpy.flatnonzero(A[:, index[q]]))
                )
            row = Ainv[index[q]]  # (-1)^n_q
            return (-1) ** int(row @ s % 2) * openfermion.QubitOperator(
                tuple((qubits[k], "Z") for k in numpy.flatnonzero(row))
            )

        result = openfermion.QubitOperator()
        for term, coeff in qop.to_openfermion().terms.items():
            op = openfermion.QubitOperator((), coeff)
            for q, p in term:
                op *= image(q, p)
            result += op
        # projection onto |0> of the tapered qubits: X and Y vanish, Z is the identity
        projected = openfermion.QubitOperator()
        for term, coeff in result.terms.items():
            if any(q in tapered and p != "Z" for q, p in term):
                continue
            projected += openfermion.QubitOperator(
                tuple((q, p) for q, p in term if q not in tapered), coeff
            )
        projected.compress()
        return QubitHamiltonian.from_openfermion(projected)

    def jw_to_me(self) -> QCircuit:
        U = QCircuit()
        if not len(self.FER_SO):
            return U
        qubits, A, s = self.matrix()
        # A is the product of the row operations reducing it, in reversed order
        for t, c in self.row_reduction(A)[::-1]:
            U += X(target=qubits[t], control=qubits[c])
        for k in numpy.flatnonzero(s):
            U += X(target=qubits[k])
        return U

    def map_state(self, state: list, *args, **kwargs):
        state = list(super().map_state(state, *args, **kwargs))
        if not len(self.FER_SO):
            return state
        qubits, A, s = self.matrix()
        n = numpy.array([state[q] for q in qubits], dtype=int)
        for q, v in zip(qubits, (A @ n + s) % 2):
            state[q] = int(v)
        return state


class TaperedBinary(TaperedLinearEncoding):
    """
    Checksum code (arXiv:1712.07067) on top of the hybrid JordanWigner: the qubits of the last fermionic orbital store
    the alpha and beta electron parities, which are fixed, so they are tapered and stay in |0>.
    The bosonic qubits enter both parities since they hold one alpha and one beta electron.
    """

    def linear_map(self) -> tuple[dict, dict, list]:
        alpha = [self.pos[2 * i] for i in range(self.n_orbitals)]
        beta = [
            self.pos[2 * i + 1]
            if (self.two_qubit or self.select[i] == "F")
            else self.pos[2 * i]
            for i in range(self.n_orbitals)
        ]
        qa, qb = self.FER_SO[-2], self.FER_SO[-1]
        parity = (self.n_electrons // 2) % 2
        return {qa: alpha, qb: beta}, {qa: parity, qb: parity}, [qa, qb]


class TaperedBravyiKitaev(TaperedLinearEncoding):
    """
    Symmetry conserving Bravyi-Kitaev (arXiv:1701.07072) on top of the hybrid JordanWigner: Bravyi-Kitaev tree on the
    alpha and on the beta fermionic qubits, the alpha root stores the alpha parity and the beta root the total parity.
    Both are fixed, so they are tapered and stay in |0>.
    The bosonic qubits enter the alpha parity, they hold two electrons so they do not change the total one.
    """

    def linear_map(self) -> tuple[dict, dict, list]:
        alpha, beta = self.FER_SO[0::2], self.FER_SO[1::2]
        rows = {}
        for modes in [alpha, beta]:
            tree = openfermion.FenwickTree(len(modes))
            for j in range(len(modes)):
                # qubit j stores the sum of its subtree
                subtree, stack = [], [tree.get_node(j)]
                while stack:
                    node = stack.pop()
                    subtree.append(modes[node.index])
                    stack += node.children
                rows[modes[j]] = subtree
        bos = [
            self.pos[2 * i]
            for i in range(self.n_orbitals)
            if self.select[i] == "B" and not self.two_qubit
        ]
        rows[alpha[-1]] = rows[alpha[-1]] + bos
        rows[beta[-1]] = rows[beta[-1]] + alpha
        shift = {
            alpha[-1]: (self.n_electrons // 2) % 2,
            beta[-1]: self.n_electrons % 2,
        }
        return rows, shift, [alpha[-1], beta[-1]]
