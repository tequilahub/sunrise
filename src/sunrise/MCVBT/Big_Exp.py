
try:
    from sunrise.expval.fqe_expval import FQEBraKet
except ImportError:
    pass
from tequila.quantumchemistry import QuantumChemistryBase
from sunrise.MCVBT.QulacsBraKet import BraKetQulacs


class BigExpVal:

    def __init__(self, circuits, coefficcents, mol:QuantumChemistryBase, solver, **kwargs):

        self.n = len(circuits)

        SS = []
        EE = []
        for i in range(self.n):
            tmp1 = []
            tmp2 = []
            for j in range(i+1):
                if solver == "TCC":
                    raise NotImplementedError
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
                tmp1.append(xEE)
                tmp2.append(xSS)

            EE.append(tmp1)
            SS.append(tmp2)

        self.SS = SS
        self.EE = EE
        self.coeffs = coefficcents

        variables = {}
        for U in circuits:
            variables = {**variables, **{x: 0.0 for x in U.extract_variables()}}

        for c in coefficcents:
            variables = {**variables, **{x: 0.0 for x in c.extract_variables()}}
        self.variables = list(variables.keys())


    def __call__(self, x, *args, **kwargs):

        assert len(x) <= len(self.variables)

        values = {self.variables[i]: x[i] for i in range(len(self.variables))}
        c = [self.coeffs[i](values) for i in range(self.n)]

        f = 0.0
        s = 0.0
        for i in range(self.n):
            f+=self.EE[i][i](values)*c[i]**2
            s+=c[i]**2
            for j in range(i):
               f+=2.0*self.EE[i][j](values)*c[i]*c[j]
               s+=2.0*self.SS[i][j](values)*c[i]*c[j]

        f=f.real
        s=s.real
        if s>0.0:
            r=f/s
        else:
            r=1e5
        return r