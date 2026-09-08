import numpy as np

try:
    from sunrise.expval.fqe_expval import FQEBraKet
except ImportError:
    pass
from tequila.quantumchemistry import QuantumChemistryBase
from sunrise.MCVBT.QulacsBraKet import BraKetQulacs
from sunrise.expval.minimize import simulate,grad
from sunrise.expval import Braket
from  tequila.objective.objective import Objective,identity,Variable,assign_variable
from tequila import TequilaException
from copy import deepcopy

class BigExpVal:

    def __init__(self, circuits, coefficcents, mol:QuantumChemistryBase, solver, **kwargs):
        self.n = len(circuits)
        self.solver = solver
        SS = 0.
        EE = 0.
        for i in range(self.n):
            for j in range(self.n): #oder n ?
                if solver == "TCC":
                    # raise NotImplementedError
                    xEE = Braket(ket=circuits[j], bra=circuits[i], molecule=mol,backend='tcc')
                    xSS = Braket(ket=circuits[j], bra=circuits[i],backend='tcc',molecule=mol,operator='I')
                elif solver == "FQE":
                    xEE = FQEBraKet(ket_fcircuit=circuits[j], bra_fcircuit=circuits[i], molecule=mol)
                    xSS = FQEBraKet(ket_fcircuit=circuits[j], bra_fcircuit=circuits[i],
                                    n_orbitals=mol.n_orbitals, n_ele=mol.n_electrons)
                elif solver == "Qulacs":
                    H = kwargs["H"]
                    xEE = BraKetQulacs(circuits[i], circuits[j], H=H)
                    xSS = BraKetQulacs(circuits[i], circuits[j], H=None)
                else:
                    raise ValueError("Unknown solver {}".format(solver))


                EE += (1*coefficcents[i])*(1*coefficcents[j])*Objective(args=[xEE],transformation=identity)
                SS += (1*coefficcents[i])*(1*coefficcents[j])*Objective(args=[xSS],transformation=identity)

        self.SS:Objective = SS
        self.EE:Objective = EE

        variables = {}
        for U in circuits:
            variables = {**variables, **{x: 0.0 for x in U.extract_variables()}}

        for c in coefficcents:
            variables = {**variables, **{x: 0.0 for x in c.extract_variables()}}
        self.variables = list(variables.keys())


    def __call__(self, variables, *args, **kwargs):

        assert len(variables) <= len(self.variables)
        if self.solver == "Qulacs":
            values = {self.variables[i]: variables[i] for i in range(len(self.variables))}
        else:
            values = {i: variables[i] for i in self.variables}

        A = simulate(self.EE,values)
        B = simulate(self.SS,values)

        f=A.real
        s=B.real
        if np.isclose(s,0):
            r = 1e5
        else:
            r = f/s

        return r

    def extract_variables(self):
        return self.variables

    def grad(self, variable: Variable = None, *args, **kwargs):

        if variable is None:
            # None means that all components are created
            variables = deepcopy(self.extract_variables())
            result = {}

            if len(variables) == 0:
                raise TequilaException("Error in gradient: Objective has no variables")

            for k in variables:
                assert k is not None
                result[k] = self.grad(k)
            return result
        else:
            variable = assign_variable(variable)
        if variable not in self.extract_variables():
            return 0.

        top = grad(self.EE,variable)*self.SS - self.EE*grad(self.SS,variable)
        bottom = self.SS ** 2

        gradient = top/bottom

        return gradient
