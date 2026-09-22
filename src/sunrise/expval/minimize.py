from tequila.optimizers import minimize as tminimize
import typing
# from tequila.circuit.compiler import CircuitCompiler
from tequila.objective.objective import Objective, Variable#, ExpectationValueImpl,assign_variable,identity,FixedVariable
from tequila.circuit.noise import NoiseModel
from tequila import TequilaException,QCircuit,QubitWaveFunction
from tequila.objective import QTensor,format_variable_dictionary
from tequila.simulators.simulator_api import compile as tq_compile
# from numpy import vectorize
from tequila.autograd_imports import __AUTOGRAD__BACKEND__#, jax
from typing import Dict, Union, Hashable,Callable
from numbers import Number
from numbers import Real as RealNumber
from ..fermionic_operations import FCircuit
# from sunrise.expval.simulate_circuit import simulate_fcircuit
from .simulate_circuit import simulate_fcircuit
# from sunrise.expval.fermionic_braket import FermBraketImpl
from tequila.simulators.simulator_base import BackendCircuit# , BackendExpectationValue
from tequila import grad as tqgrad
from sunrise.expval import INSTALLED_FERMIONIC_BACKENDS, SUPPORTED_FERMIONIC_BACKENDS

def minimize(objective, method:str="bfgs", variables:list=None, initial_values:Union[dict, Number, Callable]=0.0, maxiter:int=None, silent:bool=True, *args, **kwargs):
    if "backend" in kwargs and kwargs["backend"] in INSTALLED_FERMIONIC_BACKENDS:
        fbackend = kwargs["backend"]
        kwargs["backend"] = None
    else: fbackend = None
    if "backend" not in kwargs:
        kwargs["backend"] = None
    if hasattr(objective,'args') and any([type(arg).__name__ ==  "FermBraketImpl" for arg in objective.args]):
        dE = grad(objective=objective, variable=variables, args=args, kwargs=kwargs)
        dE = {g:compile(dE[g], backend=fbackend) for g in dE.keys()}
        nargs = [compile(Objective(args=[arg]), fermionic_backend=fbackend, backend=kwargs["backend"]) for arg in objective.args]
        objective = Objective(args=nargs, transformation=objective.transformation)
        kwargs['backend'] = None
        return tminimize(objective=objective, gradient=dE, method=method, variables=variables, initial_values=initial_values, maxiter=maxiter, silent=silent, *args,**kwargs)
    else:
        return tminimize(objective=objective, method=method, variables=variables, initial_values=initial_values, maxiter=maxiter, silent=silent, *args, **kwargs)

# #FIXME: Placeholder functions while not commited to tequila
# def grad(objective: Union[Objective, QTensor], variable: Variable = None, no_compile=False, *args, **kwargs):
#     """
#     wrapper function for getting the gradients of Objectives,ExpectationValues, Unitaries (including single gates), and Transforms.
#     :param obj (QCircuit,ParametrizedGateImpl,Objective,ExpectationValue,Transform,Variable): structure to be differentiated
#     :param variables (list of Variable): parameter with respect to which obj should be differentiated.
#         default None: total gradient.
#     return: dictionary of Objectives, if called on gate, circuit, exp.value, or objective; if Variable or Transform, returns number.
#     """
#     # elif type(objective).__name__ == 'TequilaBraket':
#     #     objective = objective.build()
#     if variable is None:
#         # None means that all components are created
#         variables = objective.extract_variables()
#         result = {}

#         if len(variables) == 0:
#             raise TequilaException("Error in gradient: Objective has no variables")
#         for k in variables:
#             assert k is not None
#             result[k] = grad(objective, k, no_compile=no_compile)
#         return result
#     elif isinstance(variable,list):
#         result = {}
#         if len(variable) == 0:
#             raise TequilaException("Error in gradient: Objective has no variables")

#         for k in variable:
#             assert k is not None
#             result[k] = grad(objective, k, no_compile=no_compile)
#         return result
#     else:
#         variable = assign_variable(variable)

#     if isinstance(objective, QTensor):
#         f = lambda x: grad(objective=x, variable=variable, *args, **kwargs)
#         ff = vectorize(f)
#         return ff(objective)

#     if variable not in objective.extract_variables():
#         return Objective()

#     # objective translation
#     # if the objective was already translated to a backend
#     # we need to reverse that here
#     our = False
#     if any([type(arg).__name__ == "FermBraketImpl" for arg in objective.args]):
#         our = True
#         no_compile = True
#     elif not our and objective.is_translated():
#         raise TequilaException(
#             "\n\ngradient of:{}\ncan not form gradient that was already compiled to a quantum backend\ntq.grad neds to be applied to the abstract - non compiled objective\nE.g. for the (compiled) objective E1 \n\tE1 = tq.compile(E0)\ninstead of doing\n\tdE = tq.grad(E1)\ndo\n\tdE = tq.grad(E0)\nand compile dE afterwards (if wanted) with\n\tdE = tq.compile(dE)\n".format(
#                 str(objective)
#             )
#         )

#     # circuit compilation
#     if no_compile:
#         compiled = objective
#     else:
#         compiler = CircuitCompiler(
#             multitarget=True,
#             trotterized=True,
#             hadamard_power=True,
#             power=True,
#             controlled_phase=True,
#             controlled_rotation=True,
#             gradient_mode=True,
#         )

#         compiled = compiler(objective, variables=[variable])

#     if variable not in compiled.extract_variables():
#         raise TequilaException("Error in taking gradient. Objective does not depend on variable {} ".format(variable))

#     if isinstance(objective, ExpectationValueImpl):
#         return __grad_expectationvalue(E=objective, variable=variable)
#     elif objective.is_expectationvalue():
#         return __grad_expectationvalue(E=compiled.args[-1], variable=variable)
#     elif isinstance(compiled, Objective) or (hasattr(compiled, "args") and hasattr(compiled, "transformation")):
#         return __grad_objective(objective=compiled, variable=variable)
#     else:
#         raise TequilaException("Gradient not implemented for other types than ExpectationValue and Objective.")

# def __grad_objective(objective: Objective, variable: Variable):
#     args = objective.args
#     transformation = objective.transformation
#     dO = None
#     processed_expectationvalues = {}
#     for i, arg in enumerate(args):
#         if __AUTOGRAD__BACKEND__ == "jax":
#             df = jax.grad(transformation, argnums=i)
#         elif __AUTOGRAD__BACKEND__ == "autograd":
#             df = jax.grad(transformation, argnum=i)
#         else:
#             raise TequilaException("Can't differentiate without autograd or jax")

#         # We can detect one simple case where the outer derivative is const=1
#         if transformation is None or transformation == identity:
#             outer = 1.0
#         else:
#             outer = Objective(args=args, transformation=df)

#         if hasattr(arg, "U"):
#             # save redundancies
#             if arg in processed_expectationvalues:
#                 inner = processed_expectationvalues[arg]
#             else:
#                 inner = __grad_inner(arg=arg, variable=variable)
#                 processed_expectationvalues[arg] = inner
#         else:
#             # this means this inner derivative is purely variable dependent
#             inner = __grad_inner(arg=arg, variable=variable)

#         if inner == 0.0:
#             # don't pile up zero expectationvalues
#             continue

#         if dO is None:
#             dO = outer * inner
#         else:
#             dO = dO + outer * inner

#     if dO is None:
#         raise TequilaException("caught None in __grad_objective")
#     return dO

# def __grad_inner(arg, variable):
#     """
#     a modified loop over __grad_objective, which gets derivatives
#     all the way down to variables, return 1 or 0 when a variable is (isnt) identical to var.
#     :param arg: a transform or variable object, to be differentiated
#     :param variable: the Variable with respect to which par should be differentiated.
#     :ivar var: the string representation of variable
#     """

#     assert isinstance(variable, Variable)
#     if isinstance(arg, Variable):
#         if arg == variable:
#             return 1.0
#         else:
#             return 0.0
#     elif isinstance(arg, FixedVariable):
#         return 0.0
#     elif isinstance(arg, ExpectationValueImpl):
#         return __grad_expectationvalue(arg, variable=variable)
#     elif hasattr(arg, "abstract_expectationvalue"):
#         E = arg.abstract_expectationvalue
#         dE = __grad_expectationvalue(E, variable=variable)
#         return compile(dE, **arg._input_args)
#     elif hasattr(arg,'grad'):
#         return arg.grad(variable)
#     else:
#         return __grad_objective(objective=arg, variable=variable)

# def __grad_expectationvalue(E: ExpectationValueImpl, variable: Variable):
#     """
#     implements the analytic partial derivative of a unitary as it would appear in an expectation value. See the paper.
#     :param unitary: the unitary whose gradient should be obtained
#     :param variables (list, dict, str): the variables with respect to which differentiation should be performed.
#     :return: vector (as dict) of dU/dpi as Objective (without hamiltonian)
#     """

#     hamiltonian = E.H
#     unitary = E.U
#     if not (unitary.verify()):
#         raise TequilaException("error in grad_expectationvalue unitary is {}".format(unitary))

#     # fast return if possible
#     if variable not in unitary.extract_variables():
#         return 0.0

#     param_gates = unitary._parameter_map[variable]

#     dO = Objective()
#     for idx_g in param_gates:
#         idx, g = idx_g
#         dOinc = __grad_shift_rule(unitary, g, idx, variable, hamiltonian)
#         dO += dOinc

#     assert dO is not None
#     return dO

# def __grad_shift_rule(unitary, g, i, variable, hamiltonian):
#     """
#     function for getting the gradients of directly differentiable gates. Expects precompiled circuits.
#     :param unitary: QCircuit: the QCircuit object containing the gate to be differentiated
#     :param g: a parametrized: the gate being differentiated
#     :param i: Int: the position in unitary at which g appears
#     :param variable: Variable or String: the variable with respect to which gate g is being differentiated
#     :param hamiltonian: the hamiltonian with respect to which unitary is to be measured, in the case that unitary
#         is contained within an ExpectationValue
#     :return: an Objective, whose calculation yields the gradient of g w.r.t variable
#     """

#     # possibility for overwride in custom gate construction
#     if hasattr(g, "shifted_gates"):
#         inner_grad = __grad_inner(g.parameter, variable)
#         shifted = g.shifted_gates()
#         dOinc = Objective()
#         for x in shifted:
#             w, g = x
#             Ux = unitary.replace_gates(positions=[i], circuits=[g])
#             wx = w * inner_grad
#             Ex = Objective.ExpectationValue(U=Ux, H=hamiltonian)
#             dOinc += wx * Ex
#         return dOinc
#     else:
#         raise TequilaException("No shift found for gate {}\nWas the compiler called?".format(g))

def grad(objective: Union[Objective, QTensor], variable: Variable = None, no_compile=False, *args, **kwargs):
    """
    wrapper function for getting the gradients of Objectives,ExpectationValues, Unitaries (including single gates), and Transforms.
    :param obj (QCircuit,ParametrizedGateImpl,Objective,ExpectationValue,Transform,Variable): structure to be differentiated
    :param variables (list of Variable): parameter with respect to which obj should be differentiated.
        default None: total gradient.
    return: dictionary of Objectives, if called on gate, circuit, exp.value, or objective; if Variable or Transform, returns number.
    """

    if hasattr(objective,'args') and any([type(arg).__name__ == "FermBraketImpl" for arg in objective.args]):
        return tqgrad(objective=objective, variable=variable, no_compile=True, *args, **kwargs) #TODO: What happens if mixture 
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

    if "fermionic_backend" in kwargs:
        fbackend = kwargs['fermionic_backend']
        kwargs.pop('fermionic_backend')
    elif backend in SUPPORTED_FERMIONIC_BACKENDS:
        fbackend = backend
        backend = None
    elif backend is None:
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
                        compiled_expval = compile_fexpval(objective=arg, backend=fbackend)
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
    if backend in INSTALLED_FERMIONIC_BACKENDS:
        return INSTALLED_FERMIONIC_BACKENDS[backend.lower()](objective,*args,**kwargs)
    else:
        raise ImportError(f'Fermionic backend not recognised {backend}')
