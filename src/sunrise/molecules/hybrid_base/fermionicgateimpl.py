"""Gate implementation of fermionic excitations compiled for the mixed fermionic/hard-core boson encoding."""

from __future__ import annotations

import typing
from copy import deepcopy
from numbers import Real

from numpy import pi
from tequila import QCircuit, Variable, assign_variable, compile_circuit
from tequila.circuit import gates


class FermionicGateImpl(gates.QubitExcitationImpl):
    """Fermionic excitation gate, compiled following https://doi.org/10.1103/PhysRevA.102.062612.

    Mainly a gate class so that "FermionicExcitation" is shown when circuits are printed.

    Parameters
    ----------
    generator : QubitHamiltonian
        Qubit generator of the excitation.
    p0 : QubitHamiltonian
        Projector used for the gradient of the gate.
    transformation : str
        Lower-case name of the encoding class, e.g. "jordanwigner". Other encodings are
        compiled by Trotterization.
    select : dict
        "F"/"B" encoding of each spatial orbital.
    indices : list of tuple of int
        Excitation ``[(p0, q0), (p1, q1), ...]`` in spin orbitals, ``(p, q)`` destroys p and creates q.
        A flat list ``[p0, q0, p1, q1, ...]`` is also accepted.
    condense, up_then_down, two_qubit : bool
        Properties of the encoding, see ``JordanWigner``.
    **kwargs
        ``opt`` (bool, default False) replaces the sequence CNOT H CNOT by Rz Rz CNOT Rz in
        ``fermionic_excitation``; the rest is passed to ``QubitExcitationImpl`` (e.g. ``angle``,
        ``control``, ``assume_real``).
    """

    def __init__(
        self,
        generator,
        p0,
        transformation,
        select=None,
        indices=None,
        condense=False,
        up_then_down=False,
        two_qubit=False,
        *args,
        **kwargs,
    ):
        if "opt" in kwargs:
            self.opt = kwargs["opt"]
            kwargs.pop("opt")
        else:
            self.opt = False
        super().__init__(
            *args, generator=generator, target=generator.qubits, p0=p0, **kwargs
        )
        self._name = "FermionicExcitation"
        self.transformation = transformation
        self.indices = indices
        self.select = select
        self.condense = condense
        self.up_then_down = up_then_down
        self.two_qubit = two_qubit
        self.n_orbitals = len(select)
        self.FER_SO, self.pos = self.select_to_list()
        if not hasattr(indices[0], "__len__"):
            self.indices = [
                (indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)
            ]
        self.sign = self.format_excitation_variables(self.indices)
        self.indices = self.format_excitation_indices(self.indices)

    def fermionic_excitation(
        self,
        angle: Real | Variable | typing.Hashable,
        indices: list,
        control: int | list | None = None,
    ) -> QCircuit:
        """Compile the excitation into a circuit of CNOT ladders and a multi-controlled Ry.

        Follows https://doi.org/10.1103/PhysRevA.102.062612. If the gate was created with ``opt=True``,
        the sequence CNOT H CNOT is replaced by Rz Rz CNOT Rz.

        Parameters
        ----------
        angle : float, tq.Variable or hashable
            Angle of the excitation.
        indices : list of tuple of int
            Pairs ``(p, q)`` of spin orbital indices, each destroying ``p`` and creating ``q``.
        control : int or list of int, optional
            Control qubits.

        Returns
        -------
        tq.QCircuit
            The compiled circuit.
        """
        lto = []
        lfrom = []
        angle = assign_variable(angle)
        if isinstance(indices, tuple) and not hasattr(indices[0], "__len__"):
            indices = [
                (indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)
            ]
        for pair in indices:
            if (
                self.select is None
                or self.select[pair[0] // 2] == "F"
                or (self.select[pair[0] // 2] == "B" and not pair[0] % 2)
                or self.two_qubit
            ):
                lfrom.append(self.pos[pair[0]])
            if (
                self.select is None
                or self.select[pair[1] // 2] == "F"
                or (self.select[pair[1] // 2] == "B" and not pair[1] % 2)
                or self.two_qubit
            ):
                lto.append(self.pos[pair[1]])
        Upair = QCircuit()
        for i in range(len(lfrom) - 1):
            Upair += gates.CNOT(lfrom[i + 1], lfrom[i]) + gates.X(lfrom[i])
        for i in range(len(lto) - 1):
            Upair += gates.CNOT(lto[i + 1], lto[i]) + gates.X(lto[i])
        Upair += gates.CNOT(lto[-1], lfrom[-1])
        crt = lfrom[::-1] + lto
        Uladder = QCircuit()
        pairs = lfrom + lto
        pairs.sort()
        idx = []
        for pair in indices:
            idx.append(pair[0])
            idx.append(pair[1])
        idx.sort()
        orbs = []
        for o in range(len(idx) // 2):
            orbs += [*range(idx[2 * o] + 1, idx[2 * o + 1])]
        orbs = [self.pos[i] for i in orbs if self.select[i // 2] == "F"]
        if len(orbs):
            for o in range(len(orbs) - 1):
                Uladder += gates.CNOT(orbs[o], orbs[o + 1])
            Uladder += compile_circuit(gates.CZ(orbs[-1], lto[-1]))
        crt.pop(-1)
        if control is not None and (isinstance(control, int) or len(control) == 1):
            if isinstance(control, int):
                crt.append(control)
            else:
                crt = crt + [*control]
            control = []
        Ur = self.cCRy(target=lto[-1], dcontrol=crt, angle=angle, control=control)
        Upair2 = Upair.dagger()
        if self.opt:
            Ur.gates.pop(-1)
            Ur.gates.pop(-1)
            Upair2.gates.pop(0)
            Ur += gates.Rz(pi / 2, target=lto[-1]) + gates.Rz(-pi / 2, target=lfrom[-1])
            Ur += (
                gates.CNOT(lto[-1], lfrom[-1])
                + gates.Rz(pi / 2, target=lfrom[-1])
                + gates.H(lfrom[-1])
            )
        return Upair + Uladder + Ur + Uladder.dagger() + Upair2

    def compile(self, *args, **kwargs):
        """Compile the gate to a circuit: the exact fermionic excitation circuit with Jordan-Wigner, otherwise one Trotter step of the generator."""
        if self.is_convertible_to_qubit_excitation():
            target = []
            for x in self.indices:
                for y in x:
                    target.append(y.copy)
            U = gates.QubitExcitation(
                target=target, angle=self.parameter, control=self.control
            )
        else:
            if self.transformation.lower().strip("_") == "jordanwigner":
                U = self.fermionic_excitation(
                    angle=self.sign * self.parameter,
                    indices=self.indices,
                    control=self.control,
                )
            else:
                U = gates.Trotterized(
                    generator=self.generator,
                    control=self.control,
                    angle=self.parameter,
                    steps=1,
                )
        return U

    def format_excitation_indices(self, idx):
        """Sort the index pairs of an excitation by their first index (each pair keeps its direction).

        For ``idx = [(p0, q0), (p1, q1), ..., (pn, qn)]`` the pairs are reordered such that
        ``p0 < p1 < ... < pn``; inside each pair the order (direction) is kept.

        Parameters
        ----------
        idx : list of tuple of int
            Pairs ``(p, q)`` describing a single fermionic excitation; ``(p, q)`` destroys p and creates q.

        Returns
        -------
        list of tuple of int
            The sorted pairs.
        """

        idx = sorted(idx, key=lambda x: x[0])
        return list(idx)

    def format_excitation_variables(self, idx):
        """Return the sign of the variable that results from sorting the index pairs.

        Parameters
        ----------
        idx : list of tuple of int
            Pairs ``(p, q)`` describing a single fermionic excitation; ``(p, q)`` destroys p and creates q.

        Returns
        -------
        int
            +1 or -1, the sign to apply to the angle of the compiled circuit with re-ordered indices.
        """
        # prod_k a^dagger_{q_k} a_{p_k} = a^dagger_{q_0} ... a^dagger_{q_n} a_{p_n} ... a_{p_0}, the compiled circuit
        # has the creators sorted: the sign changes for every two pairs whose p and q are in opposite order
        sig = 1
        for k in range(len(idx)):
            for l in range(k + 1, len(idx)):
                if (idx[k][0] > idx[l][0]) != (idx[k][1] > idx[l][1]):
                    sig *= -1
        return sig

    def cCRy(
        self,
        target: int,
        dcontrol: list | int,
        control: list | int,
        angle: Real | Variable | typing.Hashable,
        case: int = 1,
    ) -> QCircuit:
        """Compile a multi-controlled Ry rotation as in https://doi.org/10.1103/PhysRevA.102.062612.

        Parameters
        ----------
        target : int
            Target qubit.
        dcontrol : list of int or int
            Controls of the rotation that are handled by the compilation of the paper.
            If empty, a plain (controlled) Ry is returned.
        control : list of int or int
            Additional controls applied to the whole circuit.
        angle : float, tq.Variable or hashable
            Rotation angle.
        case : int, optional
            1 uses Eq. 12 of the paper, 0 uses Eq. 13. Default is 1.

        Returns
        -------
        tq.QCircuit
            The compiled circuit.
        """
        if control is not None and not len(control):
            control = None
        if isinstance(dcontrol, int):
            dcontrol = [dcontrol]
        if not len(dcontrol):
            return compile_circuit(
                gates.Ry(angle=angle, target=target, control=control)
            )
        else:
            if isinstance(angle, str):
                angle = Variable(angle)
            U = QCircuit()
            aux = dcontrol[0]
            ctr = deepcopy(dcontrol)
            ctr.pop(0)
            if case:
                U += (
                    self.cCRy(
                        target=target,
                        dcontrol=ctr,
                        angle=angle / 2,
                        case=1,
                        control=control,
                    )
                    + gates.H(aux)
                    + gates.CNOT(target, aux)
                )
                U += (
                    self.cCRy(
                        target=target,
                        dcontrol=ctr,
                        angle=-angle / 2,
                        case=0,
                        control=control,
                    )
                    + gates.CNOT(target, aux)
                    + gates.H(aux)
                )
            else:
                U += (
                    gates.H(aux)
                    + gates.CNOT(target, aux)
                    + self.cCRy(
                        target=target,
                        dcontrol=ctr,
                        angle=-angle / 2,
                        case=0,
                        control=control,
                    )
                )
                U += (
                    gates.CNOT(target, aux)
                    + gates.H(aux)
                    + self.cCRy(
                        target=target,
                        dcontrol=ctr,
                        angle=angle / 2,
                        case=1,
                        control=control,
                    )
                )
            return U

    def __str(self):
        """Return a short description with the indices."""
        if self.indices is not None:
            return f"FermionicExcitation({self.indices!s})"
        return "FermionicExcitation"

    def __repr__(self):
        """Return the string representation."""
        return self.__str__()

    def is_convertible_to_qubit_excitation(self):
        """Return whether the gate is a spin-paired double excitation that equals a qubit excitation.

        In Jordan-Wigner these can be compiled more efficiently. Currently disabled, always False.
        """
        return False
        if self.transformation.lower().strip("_") != "jordanwigner":
            return False
        if len(self.indices) != 2:
            return False
        if self.indices[0][0] // 2 != self.indices[1][0] // 2:
            return False
        return self.indices[0][1] // 2 == self.indices[1][1] // 2

    def select_to_list(self):
        """Return the list of fermionic qubits and the dict mapping spin orbitals to qubits."""

        hcb = 0
        FER_SO = []
        sel = self.select
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
        FER_SO.sort()
        return FER_SO, pos
