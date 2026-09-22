from tequila import Variable,Objective,simulate,QubitWaveFunction,TequilaWarning,BitNumbering
from tequila.objective.objective import Variables
from tequila import  BitStringLSB
from sunrise.fermionic_operations import FCircuit
from tequila import Molecule
import numpy as np
from warnings import warn
try:
    from fqe.util import sort_configuration_keys
except ImportError:
    pass
def fqe_circuit_simulatorU(U:FCircuit, variables:Variables, n_orb:int=None, **backend_kwargs) -> QubitWaveFunction:
    from . import Braket
    from .minimize import compile
    if n_orb is None:
        n_orb = int(np.ceil((U.max_qubit()+1)/2))
    res = QubitWaveFunction(n_qubits=2*n_orb,numbering=BitNumbering.MSB,dense=False)

    if U.n_electrons is None:
        warn('FCircuit with no initial_state provided.',TequilaWarning)
        return res

    U = U.to_upthendown(norb=n_orb)

    EV = Braket(ket=U, mol=__generate_dummy_mol(n_orb=n_orb, n_elec=U.n_electrons), backend_kwargs=backend_kwargs, operator=None)
    EV = compile(EV, backend='fqe')
    EV = EV.args[0]
    _ = EV(variables)
    state = EV.ket_time_evolved

    config_in_order = sort_configuration_keys(state.sectors())
    for key in config_in_order:
        self = state._civec[key]
        for inda in range(self._core.lena()):
            alpha_str = self._core.string_alpha(inda)
            for indb in range(self._core.lenb()):
                beta_str = self._core.string_beta(indb)

                wfn = BitStringLSB.from_binary(bin(alpha_str)[2:].zfill(n_orb)+bin(beta_str)[2:].zfill(n_orb),nbits=2*n_orb)
                res[wfn.to_integer(numbering=BitNumbering.LSB)] = self.coeff[inda, indb]

    return res



def __generate_dummy_mol(n_orb:int,n_elec:int):
    h = np.zeros(shape=(n_orb,n_orb))
    g = np.zeros(shape=(n_orb,n_orb,n_orb,n_orb))
    geo = ''.join([f'H 0. 0. {i}\n' for i in range(n_orb)])
    return Molecule(transformation='reordered-jordan-wigner',geometry=geo,basis_set='custom', n_electrons=n_elec, nuclear_repulsion=0, one_body_integrals=h, two_body_integrals=g,units='a')

def map_variables(x:list[Variable,Objective],dvariables:dict):
    if isinstance(x,Variable):
        x = x.map_variables(dvariables)
    elif isinstance(x,Objective):
        x=simulate(x,dvariables)
    return x

