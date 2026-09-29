from tequila.optimizers import minimize as tminimize
import typing
from tequila.objective.objective import Objective, Variable
from tequila.circuit.noise import NoiseModel
from tequila import TequilaException,QCircuit,QubitWaveFunction
from tequila.objective import QTensor,format_variable_dictionary
from tequila.simulators.simulator_api import compile as tq_compile
from tequila.autograd_imports import __AUTOGRAD__BACKEND__
from typing import Dict, Union, Hashable,Callable
from numbers import Number
from numbers import Real as RealNumber
from ..fermionic_operations import FCircuit
from .simulate_circuit import simulate_fcircuit
from tequila.simulators.simulator_base import BackendCircuit
from tequila import grad as tqgrad
from sunrise.expval import INSTALLED_FERMIONIC_BACKENDS, SUPPORTED_FERMIONIC_BACKENDS

def minimize(objective, method:str="bfgs", variables:list=None, initial_values:Union[dict, Number, Callable]=0.0, maxiter:int=None, silent:bool=True, *args, **kwargs):
    if "fbackend" in kwargs:
        fbackend = kwargs["fbackend"]
    elif "backend" in kwargs and kwargs["backend"] in INSTALLED_FERMIONIC_BACKENDS:
        fbackend = kwargs["backend"]
        kwargs.pop("backend")
    else: fbackend = None
    if "backend" not in kwargs:
        kwargs["backend"] = None
    if hasattr(objective,'args') and any([type(arg).__name__ ==  "FermBraketImpl" for arg in objective.args]):
        dE = grad(objective=objective, variable=variables, args=args, kwargs=kwargs)
        dE = {g:compile(dE[g], fbackend=fbackend, backend=kwargs["backend"]) for g in dE.keys()}
        nargs = [compile(Objective(args=[arg]), fbackend=fbackend, backend=kwargs["backend"]) for arg in objective.args]
        objective = Objective(args=nargs, transformation=objective.transformation)
        kwargs['backend'] = None
        return tminimize(objective=objective, gradient=dE, method=method, variables=variables, initial_values=initial_values, maxiter=maxiter, silent=silent, *args,**kwargs)
    else:
        return tminimize(objective=objective,method=method,variables=variables,initial_values=initial_values,maxiter=maxiter,silent=silent,*args,**kwargs)

def grad(objective: Union[Objective, QTensor], variable: Variable = None, no_compile=False, *args, **kwargs):
    """
    function for getting the gradients of directly differentiable gates. Expects precompiled circuits.
    :param unitary: QCircuit: the QCircuit object containing the gate to be differentiated
    :param g: a parametrized: the gate being differentiated
    :param i: Int: the position in unitary at which g appears
    :param variable: Variable or String: the variable with respect to which gate g is being differentiated
    :param hamiltonian: the hamiltonian with respect to which unitary is to be measured, in the case that unitary
        is contained within an ExpectationValue
    :return: an Objective, whose calculation yields the gradient of g w.r.t variable
    """

    if hasattr(objective,'args') and any([type(arg).__name__ == "FermBraketImpl" for arg in objective.args]):
        return tqgrad(objective=objective, variable=variable, no_compile=True, *args, **kwargs)
    else:
        return tqgrad(objective=objective, variable=variable, no_compile=no_compile, *args, **kwargs)

def simulate(
    objective: typing.Union[FCircuit,"Objective", "QCircuit", "QTensor"],
    variables: Dict[Union[Variable, Hashable], RealNumber] = None,
    samples: int = None,
    backend: str = None,
    noise: NoiseModel = None,
    device: str = None,
    initial_state: Union[int, QubitWaveFunction] = 0,
    *args,
    **kwargs,
) -> Union[RealNumber, QubitWaveFunction]:
    """Simulate a tequila objective or circuit

    Parameters
    ----------
    objective: Objective:
        tequila objective or circuit
    variables: Dict:
        The variables of the objective given as dictionary
        with keys as tequila Variables/hashable types and values the corresponding real numbers
    samples : int, optional:
        if None a full wavefunction simulation is performed, otherwise a fixed number of samples is simulated
    backend : str, optional:
        specify the backend or give None for automatic assignment
    noise: NoiseModel, optional:
        specify a noise model to apply to simulation/sampling
    device:
        a device upon which (or in emulation of which) to sample
    initial_state: int or QubitWaveFunction:
        the initial state of the circuit
    *args :

    **kwargs :
        read_out_qubits = list[int] (define the qubits which shall be measured, has only effect on pure QCircuit simulation with samples)

    Returns
    -------
    float or QubitWaveFunction
        the result of simulation.
    """

    variables = format_variable_dictionary(variables)

    if variables is None and not (len(objective.extract_variables()) == 0):
        raise TequilaException(
            "You called simulate for a parametrized type but forgot to pass down the variables: {}".format(
                objective.extract_variables()
            )
        )
    
    if isinstance(objective,list):
        return [simulate(op,variables,samples,backend,noise,device,initial_state,*args,**kwargs) for op in objective] # TODO Multidimensional
    if isinstance(objective,FCircuit):
        return simulate_fcircuit(U=objective,variables=variables,backend=backend,*args,**kwargs)
    
    compiled_objective = compile(
        objective=objective,
        samples=samples,
        variables=variables,
        backend=backend,
        noise=noise,
        device=device,
        *args,
        **kwargs,
    )
    return compiled_objective(variables=variables, samples=samples, initial_state=initial_state, *args, **kwargs)

def compile(
    objective: typing.Union["Objective", "QCircuit", "QTensor", "FCircuit"],
    variables: Dict[Union["Variable", Hashable], RealNumber] = None,
    samples: int = None,
    simulate_density: bool = False,
    backend: str = None,
    noise: NoiseModel = None,
    device: str = None,
    *args,
    **kwargs,
    )-> typing.Union["BackendCircuit", "Objective"]:
    """Compile a tequila objective or circuit to a backend

    Parameters
    ----------
    objective: Objective:
        tequila objective or circuit
    variables: dict, optional:
        The variables of the objective given as dictionary
        with keys as tequila Variables and values the corresponding real numbers
    samples: int, optional:
        if None a full wavefunction simulation is performed, otherwise a fixed number of samples is simulated
    backend : str, optional:
        specify the backend or give None for automatic assignment
    noise: NoiseModel, optional:
        the noise model to apply to the objective or QCircuit.
    device: optional:
        a device on which (or in emulation of which) to sample the circuit.
    Returns
    -------
    simulators.BackendCircuit or Objective
        the compiled object.

    """

    fbackend = None
    if "fbackend" in kwargs:
        fbackend = kwargs["fbackend"]
        kwargs.pop("fbackend")
    if isinstance(fbackend,str):
        fbackend = fbackend.lower()
    if isinstance(backend,str):
        backend = backend.lower()
    if backend in SUPPORTED_FERMIONIC_BACKENDS:
        fbackend = backend
        backend = None
    if fbackend is None:
        fbackend = [*INSTALLED_FERMIONIC_BACKENDS.keys()][0]
    if isinstance(objective,FCircuit):
        return objective # Currently there is not FCircuit compilation since the fermionic backends work with lists of operations instead of actual qcircuit compilation

    if isinstance(objective,Objective):
        argsets = objective.argsets
        compiled_sets = []
        for argset in argsets:
            compiled_args = []
            # avoid double compilations
            expectationvalues = {}
            for arg in argset:
                if type(arg).__name__ == "FermBraketImpl":
                    if id(arg) not in expectationvalues:
                        compiled_expval = compile_fexpval(objective=arg, backend=fbackend, *args, **kwargs)
                        if fbackend == 'tequila':
                            compiled_expval = tq_compile(objective=compiled_expval, variables=variables, samples=samples, 
                                                         simulate_density=simulate_density,backend=backend, noise=noise,
                                                           device=device, *args, **kwargs).args[0]
                        expectationvalues[id(arg)] = compiled_expval
                    else: 
                        compiled_expval = expectationvalues[id(arg)]
                    compiled_args.append(compiled_expval)
                else:
                    compiled_args.append(arg)
            compiled_sets.append(compiled_args)
        objective = type(objective)(args=compiled_sets[0], transformation=objective.transformation)
    res = tq_compile(objective=objective, variables=variables, samples=samples, simulate_density=simulate_density,
                      backend=backend, noise=noise, device=device, *args, **kwargs)
    return res

def compile_fexpval(
    objective = None,
    backend: str = None,
    *args,
    **kwargs
    ): # For now our fermionic backends don't support noise/samples... Variables not passed since already inside FermBraketImpl
    if backend.lower() in INSTALLED_FERMIONIC_BACKENDS:
        return INSTALLED_FERMIONIC_BACKENDS[backend.lower()](objective,*args,**kwargs)
    else:
        raise ImportError(f'Fermionic backend not recognised {backend}')
