from sunrise.fermionic_operations import FCircuit
from tequila.objective.objective import Variables
from tequila.objective import format_variable_dictionary
from tequila import QubitWaveFunction, TequilaException, Molecule, QCircuit
from tequila import simulate as tq_simulate
from typing import Union
from numpy import zeros, ceil
from tequila import SUPPORTED_BACKENDS

SUPPORTED_FERMIONIC_CIRCUIT_SIMULATOR = ["fqe", "tcc", "spex"]
INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR = {}

try:
    from .tcc_engine.simulate import tcc_circuit_simulator
    INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR["tcc"] = tcc_circuit_simulator
except ImportError:
    pass

try:
    from sunrise.expval.fqe_circuit_sim import fqe_circuit_simulatorU
    INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR["fqe"] = fqe_circuit_simulatorU
except ImportError:
    pass
try:
    from .spex_expval import spex_circuit_simulator
    INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR["spex"] = spex_circuit_simulator
except (ImportError, AttributeError):
    pass

def simulate_fcircuit(U:FCircuit, variables:Union[Variables,dict], backend:str='tcc',**kwargs) -> QubitWaveFunction:
    '''
    Interface with Fermionic Backends to sumulate FCircuits
    :param U: Fcircuit to simulate
    :param variables: dictionary of {variable:value} to simulate
    :paran backend: name of the desired backend 
    
    If qubit based backend (see tequila.SUPPORTED_BACKENDS) the n_orb/molecule/FCircuit->QCircuit callable is required.
    '''
    if isinstance(U, QCircuit):
        return tq_simulate(U, variables, backend=backend, **kwargs)
    variables = format_variable_dictionary(variables)
    if variables is None and not (len(U.extract_variables()) == 0):
        raise TequilaException(
            "You called simulate for a parametrized type but forgot to pass down the variables: {}".format(
                U.extract_variables()
            )
        )
    backend = backend.strip().lower()
    if 'tequila' in backend.lower() or backend in SUPPORTED_BACKENDS:
        if backend.lower()=='tequila' or backend.lower()=='t':
            backend = None
        elif 'tequila-' in  backend:
            backend = backend[backend.find('tequila-')+8:]
        elif 't-' in backend:
            backend = backend[backend.find('t-')+2:]
        if 'mol' in kwargs:
            mol = kwargs['mol']
            kwargs.pop('mol')
            mol.transformation.up_then_down = True
            U = U.to_qcircuit(molecule=mol)
        elif 'molecule' in kwargs:
            mol = kwargs['molecule']
            kwargs.pop('molecule')
            mol.transformation.up_then_down = True
            U = U.to_qcircuit(molecule=mol)
        elif 'transformation' in kwargs:
            transformation = kwargs['transformation']
            kwargs.pop('transformation')
            U = U.to_qcircuit(transformation=transformation)
        elif 'n_orb' in kwargs:
            n_orb = kwargs['n_orb']
            kwargs.pop('n_orb')
            mol = __generate_dummy_mol(n_orb)
            U = U.to_qcircuit(mol)
        else:
            n_orb = int(ceil(U.max_qubit()+1/2))
            U = U.to_qcircuit(molecule=__generate_dummy_mol(n_orb=n_orb))
        return tq_simulate(U,backend=backend,variables=variables,**kwargs)
    elif backend in SUPPORTED_FERMIONIC_CIRCUIT_SIMULATOR:
        if backend in INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR:
            simulator = INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR[backend]
            return simulator(U=U,variables=variables,**kwargs)
        else:
            raise TequilaException(f'Backend {backend} not installed.')
    else:
        raise TequilaException(f'Not recognised backed: {backend}.')

def __generate_dummy_mol(n_orb:int):
    h = zeros(shape=(n_orb,n_orb))
    g = zeros(shape=(n_orb,n_orb,n_orb,n_orb))
    geo = ''.join([f'H 0. 0. {i}\n' for i in range(n_orb)])
    return Molecule(transformation='reordered-jordan-wigner',geometry=geo,basis_set='custom', n_electrons=n_orb, nuclear_repulsion=0, one_body_integrals=h, two_body_integrals=g,units='a')
