from tequila import Variable,Objective,simulate,QubitWaveFunction,TequilaWarning,BitNumbering
from tequila.objective.objective import Variables
from sunrise.fermionic_operations import FCircuit
from tequila import Molecule
import numpy as np
from warnings import warn

def tcc_circuit_simulator(U:FCircuit, variables:Variables, n_orb:int = None, **backend_kwargs)->QubitWaveFunction:
    '''
    TCC FCircuit simulator based on the TCC.UCC utilities
    :param U: FCircuit to simulate
    :param variables: 
    :param n_orb: number of spatial orbitals
    :param backend_kwargs: Backend kwargs for TCCBraket. Keywords accepted: engine,backend,dtype
    '''
    from sunrise.expval.minimize import compile
    from .. import Braket
    if n_orb is None:
        n_orb = int(np.ceil((U.max_qubit()+1)/2))
    res = QubitWaveFunction(n_qubits=2*n_orb,numbering=BitNumbering.LSB,dense=False)
    if U.n_electrons is None:
        warn('FCircuit with no initial_state provided.',TequilaWarning)
        return res
    U = U.to_upthendown(norb=n_orb)
    BK = Braket(ket=U, mol=__generate_dummy_mol(n_orb=n_orb, n_elec=U.n_electrons), backend_kwargs=backend_kwargs, operator="I")
    tccBK = compile(BK, backend="tcc").args[0]
    variables = [map_variables(x,variables) for x in tccBK.BK.total_variables]
    state = tccBK.BK.statevector(angles=[-0.5*i for i in variables])
    non_zero =  np.argwhere(np.abs(state)>1e-6)
    for i in non_zero:
        res._state[i[0]] = state[i[0]]
    return res


def __generate_dummy_mol(n_orb:int,n_elec:int):
    h = np.zeros(shape=(n_orb,n_orb))
    g = np.zeros(shape=(n_orb,n_orb,n_orb,n_orb))
    geo = ''.join([f'H 0. 0. {i}\n' for i in range(n_orb)])
    charge = n_orb - n_elec
    return Molecule(transformation='reordered-jordan-wigner',geometry=geo,charge = charge, basis_set='sto-3g', n_electrons=n_elec, nuclear_repulsion=0, one_body_integrals=h, two_body_integrals=g,units='a')

def map_variables(x:list[Variable,Objective],dvariables:dict):
    if isinstance(x,Variable):
        x = x.map_variables(dvariables)
    elif isinstance(x,Objective):
        x=simulate(x,dvariables)
    return x
