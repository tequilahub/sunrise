"""Drop-in replacements for tequila's minimize, grad, simulate and compile.

They accept sunrise Objectives (built from FCircuits) and dispatch them to the
fermionic backends, falling back to tequila for everything else.
"""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Number
from numbers import Real as RealNumber
from typing import Callable

from tequila import SUPPORTED_BACKENDS, QCircuit, QubitWaveFunction, TequilaException
from tequila import grad as tqgrad
from tequila.circuit.noise import NoiseModel
from tequila.objective import QTensor, format_variable_dictionary
from tequila.objective.objective import Objective, Variable, assign_variable
from tequila.optimizers import minimize as tminimize
from tequila.simulators.simulator_api import compile as tq_compile
from tequila.simulators.simulator_base import BackendCircuit

from sunrise.expval import (
    INSTALLED_FERMIONIC_BACKENDS,
    SUPPORTED_FERMIONIC_BACKENDS,
    pick_backend,
)

from ..fermionic_operations import FCircuit
from .simulate_circuit import simulate_fcircuit


def minimize(
    objective,
    method: str = "bfgs",
    variables: list | None = None,
    initial_values: dict | Number | Callable = 0.0,
    maxiter: int | None = None,
    silent: bool = True,
    *args,
    **kwargs,
):
    """Minimize an objective, using fermionic backends for sunrise Objectives.

    Same as ``tequila.minimize``, but if the objective contains fermionic brakets,
    gradients and objectives are compiled with a fermionic backend first. The
    backend is selected with the keyword ``fbackend`` (or ``fermionic_backend``),
    or with ``backend`` if it names an installed fermionic backend. Otherwise
    ``pick_backend`` is used.

    Parameters
    ----------
    objective : tq.Objective
        Objective to minimize.
    method : str
        Optimization method, any of tequila's optimizers. Default is "bfgs".
    variables : list, optional
        Variables to optimize. Default is all variables of the objective.
    initial_values : dict or float or callable
        Initial values of the variables. Default is 0.0.
    maxiter : int, optional
        Maximum number of iterations.
    silent : bool
        If True, do not print optimization progress. Default is True.
    *args, **kwargs
        Passed on to ``tequila.minimize``.

    Returns
    -------
    OptimizerResults
        Tequila optimization result (energy, optimal variables, history).

    See Also
    --------
    simulate : Evaluate an Objective for fixed variable values.
    grad : Gradient of an Objective.
    sunrise.optimize_orbitals : Optimize the orbitals together with the circuit.

    Examples
    --------
    >>> E = sun.Braket(ket=U, mol=mol)
    >>> result = sun.minimize(E, silent=True)
    >>> result.energy, result.variables
    """
    if "fbackend" in kwargs:
        fbackend = kwargs["fbackend"]
    elif "fermionic_backend" in kwargs:
        fbackend = kwargs["fermionic_backend"]
    elif "backend" in kwargs and kwargs["backend"] in INSTALLED_FERMIONIC_BACKENDS:
        fbackend = kwargs["backend"]
        if kwargs["backend"] not in SUPPORTED_BACKENDS:
            kwargs.pop("backend")
    else:
        fbackend = pick_backend()
    if "backend" not in kwargs:
        kwargs["backend"] = None
    if hasattr(objective, "args") and any(
        type(arg).__name__ == "FermBraketImpl" for arg in objective.args
    ):
        dE = grad(objective=objective, variable=variables, args=args, kwargs=kwargs)
        dE = {
            g: compile(dE[g], fbackend=fbackend, backend=kwargs["backend"]) for g in dE
        }
        nargs = [
            compile(Objective(args=[arg]), fbackend=fbackend, backend=kwargs["backend"])
            for arg in objective.args
        ]
        objective = Objective(args=nargs, transformation=objective.transformation)
        kwargs["backend"] = None
        return tminimize(
            *args,
            objective=objective,
            gradient=dE,
            method=method,
            variables=variables,
            initial_values=initial_values,
            maxiter=maxiter,
            silent=silent,
            **kwargs,
        )
    else:
        return tminimize(
            *args,
            objective=objective,
            method=method,
            variables=variables,
            initial_values=initial_values,
            maxiter=maxiter,
            silent=silent,
            **kwargs,
        )


def grad(
    objective: Objective | QTensor | None,
    variable: Variable = None,
    no_compile=False,
    *args,
    **kwargs,
):
    """Compute the gradient of an objective with respect to its variables.

    For objectives with fermionic brakets the gradient is built from the
    braket's own ``grad`` and not compiled here. Otherwise see ``tequila.grad``.

    Parameters
    ----------
    objective : tq.Objective or QTensor
        Objective to differentiate.
    variable : Variable or list of Variable, optional
        Variable(s) to differentiate with respect to. A list returns a dict
        {variable: gradient}.
    no_compile : bool
        If True, do not compile the circuits before differentiating. Always True
        for fermionic brakets.
    *args, **kwargs
        Passed on to ``tequila.grad``.

    Returns
    -------
    tq.Objective or dict
        Gradient Objective, or a dict mapping variables to gradient Objectives.

    Raises
    ------
    TequilaException
        If an empty list of variables is given.

    See Also
    --------
    simulate : Evaluate the gradient Objective.
    minimize : Optimize the variables of an Objective.

    Examples
    --------
    >>> E = sun.Braket(ket=U, mol=mol)
    >>> dE = sun.grad(E, "a")  # gradient Objective w.r.t. variable "a"
    >>> sun.simulate(dE, variables={"a": 0.1})
    """
    if isinstance(variable, (list, tuple)):
        if len(variable) == 0:
            raise TequilaException("Error in gradient: Objective has no variables")
        return {
            assign_variable(k): grad(objective, k, no_compile, *args, **kwargs)
            for k in variable
        }

    if hasattr(objective, "args") and any(
        type(arg).__name__ == "FermBraketImpl" for arg in objective.args
    ):
        return tqgrad(
            *args, objective=objective, variable=variable, no_compile=True, **kwargs
        )
    else:
        return tqgrad(
            *args,
            objective=objective,
            variable=variable,
            no_compile=no_compile,
            **kwargs,
        )


def simulate(
    objective: FCircuit | Objective | QCircuit | QTensor,
    variables: dict[Variable | Hashable, RealNumber] | None = None,
    samples: int | None = None,
    backend: str | None = None,
    noise: NoiseModel | None = None,
    device: str | None = None,
    initial_state: int | QubitWaveFunction = 0,
    *args,
    **kwargs,
) -> RealNumber | QubitWaveFunction | None:
    """Simulate an objective or circuit.

    Same as ``tequila.simulate`` but also accepts sunrise ``FCircuit`` and
    Objectives containing fermionic brakets.

    Parameters
    ----------
    objective : FCircuit or Objective or QCircuit or QTensor
        Object to simulate.
    variables : dict, optional
        Values of the variables, with tequila Variables or hashable names as keys.
    samples : int, optional
        Number of samples. If None, the exact wavefunction/expectation is computed.
    backend : str, optional
        Backend name. If None, it is chosen automatically.
    noise : NoiseModel, optional
        Noise model to apply.
    device : str, optional
        Device on which (or in emulation of which) to sample.
    initial_state : int or QubitWaveFunction
        Initial state of the circuit. Default is 0.
    *args, **kwargs
        Passed on to the backend. ``read_out_qubits`` (list of int) selects the
        measured qubits for sampled QCircuit simulations.

    Returns
    -------
    float or QubitWaveFunction
        Result of the simulation. A list of objectives returns a list of results.

    Raises
    ------
    TequilaException
        If the objective is parametrized and no variables are passed.

    See Also
    --------
    minimize : Optimize the variables of an Objective.
    grad : Gradient of an Objective.

    Examples
    --------
    >>> E = sun.Braket(ket=U, mol=mol)
    >>> energy = sun.simulate(E, variables={"a": 0.1}, backend="spex")
    """

    variables = format_variable_dictionary(variables)

    if variables is None and not (len(objective.extract_variables()) == 0):
        raise TequilaException(
            f"You called simulate for a parametrized type but forgot to pass down the variables: {objective.extract_variables()}"
        )

    if isinstance(objective, list):
        return [
            simulate(
                op,
                variables,
                samples,
                backend,
                noise,
                device,
                initial_state,
                *args,
                **kwargs,
            )
            for op in objective
        ]  # TODO Multidimensional
    if isinstance(objective, FCircuit):
        return simulate_fcircuit(
            *args, U=objective, variables=variables, backend=backend, **kwargs
        )

    compiled_objective = compile(
        *args,
        objective=objective,
        samples=samples,
        variables=variables,
        backend=backend,
        noise=noise,
        device=device,
        **kwargs,
    )
    return compiled_objective(
        *args,
        variables=variables,
        samples=samples,
        initial_state=initial_state,
        **kwargs,
    )


def compile(
    objective: Objective | QCircuit | QTensor | FCircuit,
    variables: dict[Variable | Hashable, RealNumber] | None = None,
    samples: int | None = None,
    simulate_density: bool = False,
    backend: str | None = None,
    noise: NoiseModel | None = None,
    device: str | None = None,
    *args,
    **kwargs,
) -> BackendCircuit | Objective | None:
    """Compile an objective or circuit for a backend.

    Same as ``tequila.compile``, but fermionic brakets inside Objectives are
    compiled with a fermionic backend (see ``pick_backend``). Select it with the
    keyword ``fbackend`` or by passing its name as ``backend``. An ``FCircuit`` is
    returned unchanged, since fermionic backends do not compile circuits.

    Parameters
    ----------
    objective : Objective or QCircuit or QTensor or FCircuit
        Object to compile.
    variables : dict, optional
        Values of the variables.
    samples : int, optional
        Number of samples. If None, the exact wavefunction is simulated.
    simulate_density : bool
        Passed on to tequila's compile. Default is False.
    backend : str, optional
        Backend name. If None, it is chosen automatically.
    noise : NoiseModel, optional
        Noise model to apply.
    device : optional
        Device on which (or in emulation of which) to sample.
    *args, **kwargs
        Passed on to tequila's compile.

    Returns
    -------
    BackendCircuit or Objective or FCircuit
        The compiled object.
    """

    fbackend = None
    if "fbackend" in kwargs:
        fbackend = kwargs["fbackend"]
        kwargs.pop("fbackend")
    if isinstance(fbackend, str):
        fbackend = fbackend.lower()
    if isinstance(backend, str):
        backend = backend.lower()
    if backend in SUPPORTED_FERMIONIC_BACKENDS:
        if fbackend is None:
            fbackend = backend
        if backend not in SUPPORTED_BACKENDS:
            backend = None
    fbackend = pick_backend(
        backend=fbackend, samples=samples, noise=noise, device=device
    )
    if isinstance(objective, FCircuit):
        return objective  # Currently there is not FCircuit compilation since the fermionic backends work with lists of operations instead of actual qcircuit compilation

    if isinstance(objective, Objective):
        argsets = objective.argsets
        compiled_sets = []
        for argset in argsets:
            compiled_args = []
            # avoid double compilations
            expectationvalues = {}
            for arg in argset:
                if type(arg).__name__ == "FermBraketImpl":
                    if id(arg) not in expectationvalues:
                        compiled_expval = compile_fexpval(
                            *args, objective=arg, backend=fbackend, **kwargs
                        )
                        if fbackend == "tequila":
                            compiled_expval = tq_compile(
                                *args,
                                objective=compiled_expval,
                                variables=variables,
                                samples=samples,
                                simulate_density=simulate_density,
                                backend=backend,
                                noise=noise,
                                device=device,
                                **kwargs,
                            ).args[0]
                        expectationvalues[id(arg)] = compiled_expval
                    else:
                        compiled_expval = expectationvalues[id(arg)]
                    compiled_args.append(compiled_expval)
                else:
                    compiled_args.append(arg)
            compiled_sets.append(compiled_args)
        objective = type(objective)(
            args=compiled_sets[0], transformation=objective.transformation
        )
    res = tq_compile(
        *args,
        objective=objective,
        variables=variables,
        samples=samples,
        simulate_density=simulate_density,
        backend=backend,
        noise=noise,
        device=device,
        **kwargs,
    )
    return res


def compile_fexpval(
    objective=None, backend: str | None = None, *args, **kwargs
):  # TODO: For now our fermionic backends don't support noise/samples... Variables not passed since already inside FermBraketImpl
    """Compile a fermionic braket with the chosen fermionic backend.

    Parameters
    ----------
    objective : FermBraketImpl
        Braket to compile. Variables are already stored inside it.
    backend : str
        Name of an installed fermionic backend.
    *args, **kwargs
        Passed on to the backend braket class.

    Raises
    ------
    ImportError
        If the backend is not installed.
    """
    if backend.lower() in INSTALLED_FERMIONIC_BACKENDS:
        return INSTALLED_FERMIONIC_BACKENDS[backend.lower()](objective, *args, **kwargs)
    else:
        raise ImportError(f"Fermionic backend not recognised {backend}")
