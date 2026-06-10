import fqe
from tequila import Variable,Objective,simulate,QubitWaveFunction,TequilaWarning,BitNumbering
from tequila.objective.objective import Variables
from tequila import  BitString, BitStringLSB
try:
    from sunrise.expval.fqe_expval import FQEBraKet
except ImportError:
    pass
from sunrise.fermionic_operations import FCircuit
from tequila import Molecule
import numpy as np
from warnings import warn
from fqe.util import sort_configuration_keys

def fqe_circuit_simulatorU(U:FCircuit,variables:Variables, n_orb:int,**backend_kwargs)->QubitWaveFunction:

    res = QubitWaveFunction(n_qubits=2*n_orb,numbering=BitNumbering.MSB,dense=False)

    if U.n_electrons is None:
        warn('FCircuit with no initial_state provided.',TequilaWarning)
        return res

    U = U.to_upthendown(norb=n_orb)

    EV = FQEBraKet(ket=U, mol=__generate_dummy_mol(n_orb=n_orb, n_elec=U.n_electrons), backend_kwargs=backend_kwargs)

    variables = [map_variables(x, variables) for x in EV.extract_variables()]

    EV(variables=[i for i in variables])
    EV.print_ket()
    state = EV.ket_time_evolved

    config_in_order = sort_configuration_keys(state.sectors())
    for key in config_in_order:
        self = state._civec[key]
        for inda in range(self._core.lena()):
            alpha_str = self._core.string_alpha(inda)
            for indb in range(self._core.lenb()):
                beta_str = self._core.string_beta(indb)

                wfn = BitStringLSB.from_binary(bin(alpha_str)[2:].zfill(n_orb)+bin(beta_str)[2:].zfill(n_orb),nbits=2*n_orb)
                print(f'{self.coeff[inda, indb]}|{bin(wfn.to_integer(numbering=BitNumbering.MSB))[2:].zfill(2*n_orb)}>')
                res._state[wfn.to_integer(numbering=BitNumbering.LSB)] = self.coeff[inda, indb]

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

