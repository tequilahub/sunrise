from tequila.circuit import gates
from tequila import assign_variable
from numpy import pi
import typing
from numbers import Real
from tequila import Variable,QCircuit,compile_circuit
from copy import deepcopy




class FermionicGateImpl(gates.QubitExcitationImpl):
    """
        Small helper class for Fermionic Excictation Gates
        Mainly so that "FermionicGate is displayed when circuits are printed
    """
    def __init__(self, generator, p0, transformation,select=None ,indices=None,condense=False,up_then_down=False,two_qubit=False,*args, **kwargs):
        if "opt" in kwargs:
            self.opt = kwargs["opt"]
            kwargs.pop("opt")
        else: self.opt =False
        super().__init__(generator=generator, target=generator.qubits, p0=p0, *args, **kwargs)
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
            self.indices = [(indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)]
        self.sign = self.format_excitation_variables(self.indices)
        self.indices = self.format_excitation_indices(self.indices)
    def fermionic_excitation(self, angle: typing.Union[Real, Variable, typing.Hashable], indices: typing.List,
                             control: typing.Union[int, typing.List] = None,) -> QCircuit:
        '''
            Excitation [(i,j),(k,l)],... compiled following https://doi.org/10.1103/PhysRevA.102.062612
            opt: whether to optimized CNOT H CNOT --> Rz Rz CNOT Rz
        '''
        lto = []
        lfrom = []
        angle = assign_variable(angle)
        if isinstance(indices, tuple) and not hasattr(indices[0],"__len__"):
            indices = [(indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)]
        for pair in indices:
            if self.select is None or self.select[pair[0] // 2] == "F" or (self.select[pair[0] // 2] == "B" and not pair[0] % 2) or self.two_qubit:
                lfrom.append(self.pos[pair[0]])
            if self.select is None or self.select[pair[1] // 2] == "F" or (self.select[pair[1] // 2] == "B" and not pair[1] % 2) or self.two_qubit:
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
        orbs = [self.pos[i] for i in orbs if self.select[i//2]=="F"]
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
            Ur += gates.CNOT(lto[-1], lfrom[-1]) + gates.Rz(pi / 2, target=lfrom[-1]) + gates.H(lfrom[-1])
        return Upair + Uladder + Ur + Uladder.dagger() + Upair2

    def compile(self, *args, **kwargs):
        if self.is_convertable_to_qubit_excitation():
            target = []
            for x in self.indices:
                for y in x:
                    target.append(y)
            U = gates.QubitExcitation(target=target, angle=self.parameter, control=self.control)
        else:
            if self.transformation.lower().strip("_") == "jordanwigner":
                U =  self.fermionic_excitation(angle=self.sign*self.parameter, indices=self.indices, control=self.control)
            else:
                U = gates.Trotterized(generator=self.generator, control=self.control, angle=self.parameter, steps=1)
        return U

    def format_excitation_indices(self, idx):
        """
        Consistent formatting of excitation indices
        idx = [(p0,q0),(p1,q1),...,(pn,qn)]
        sorted as: p0<p1<pn and pi<qi
        :param idx: list of index tuples describing a single(!) fermionic excitation
        :return: list of index tuples
        """

        idx = [tuple(sorted(x)) for x in idx]
        idx = sorted(idx, key=lambda x: x[0])
        return list(idx)

    def format_excitation_variables(self, idx):
        """
        Consistent formatting of excitation variable
        idx = [(p0,q0),(p1,q1),...,(pn,qn)]
        sorted as: pi<qi and p0 < p1 < p2
        :param idx: list of index tuples describing a single(!) fermionic excitation
        :return: sign of the variable with re-ordered indices
        """
        sig = 1
        for pair in idx:
            if pair[1] > pair[0]:
                sig *= -1
        for pair in range(len(idx) - 1):
            if idx[pair + 1][0] > idx[pair][0]:
                sig *= -1
        return sig

    def cCRy(self, target: int, dcontrol: typing.Union[list, int], control: typing.Union[list, int],
             angle: typing.Union[Real, Variable, typing.Hashable], case: int = 1) -> QCircuit:
        '''
        Compilation of CRy as on https://doi.org/10.1103/PhysRevA.102.062612
        If not control passed, Ry returned
        Parameters
        ----------
        case: if 1 employs eq. 12 from the paper, if 0 eq. 13
        '''
        if control is not None and not len(control):
            control = None
        if isinstance(dcontrol, int):
            dcontrol = [dcontrol]
        if not len(dcontrol):
            return compile_circuit(gates.Ry(angle=angle, target=target, control=control))
        else:
            if isinstance(angle, str):
                angle = Variable(angle)
            U = QCircuit()
            aux = dcontrol[0]
            ctr = deepcopy(dcontrol)
            ctr.pop(0)
            if case:
                U += self.cCRy(target=target, dcontrol=ctr, angle=angle / 2, case=1, control=control) + gates.H(aux) + gates.CNOT(target, aux)
                U += self.cCRy(target=target, dcontrol=ctr, angle=-angle / 2, case=0, control=control) + gates.CNOT(target, aux) + gates.H(aux)
            else:
                U += gates.H(aux) + gates.CNOT(target, aux) + self.cCRy(target=target, dcontrol=ctr, angle=-angle / 2,case=0, control=control)
                U += gates.CNOT(target, aux) + gates.H(aux) + self.cCRy(target=target, dcontrol=ctr, angle=angle / 2, case=1, control=control)
            return U
    def __str(self):
        if self.indices is not None:
            return "FermionicExcitation({})".format(str(self.indices))
        return "FermionicExcitation"

    def __repr__(self):
        return self.__str__()
    def is_convertable_to_qubit_excitation(self):
        """
        spin-paired double excitations (both electrons occupy the same spatial orbital and are excited to another spatial orbital)
        in the jordan-wigner representation are identical to 4-qubit excitations which can be compiled more efficient
        this function hels to automatically detect those cases
        Returns
        -------

        """
        return False
        if not self.transformation.lower().strip("_") == "jordanwigner": return False
        if not len(self.indices) == 2: return False
        if not self.indices[0][0] // 2 == self.indices[1][0] // 2: return False
        if not self.indices[0][1] // 2 == self.indices[1][1] // 2: return False
        return True

    def select_to_list(self):
        """
        Internal function
        Read the select string to make the proper Fer and Bos lists
        :return : list of MOs for the Bos, MOs and SOs for the Fer space
        """

        hcb = 0
        FER_SO = []
        sel = self.select
        pos = {}
        up = self.up_then_down
        two = self.two_qubit
        for i in sel:
            if (sel[i] == "B"):
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