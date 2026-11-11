from __future__ import annotations

from collections.abc import Hashable
from numbers import Number
from numbers import Real as RealNumber
from typing import Callable

from numpy.random import binomial
from tequila import SUPPORTED_BACKENDS, QCircuit, QubitWaveFunction, TequilaException
from tequila import grad as tqgrad
from tequila.circuit.noise import NoiseModel
from tequila.objective import QTensor, format_variable_dictionary
from tequila.objective.objective import Objective, Variable, assign_variable
from tequila.optimizers import minimize as tminimize
from tequila.simulators.simulator_api import INSTALLED_SAMPLERS
from tequila.simulators.simulator_api import compile as tq_compile
from tequila.simulators.simulator_base import BackendCircuit

from sunrise.expval import (
    INSTALLED_FERMIONIC_BACKENDS,
    SUPPORTED_FERMIONIC_BACKENDS,
    pick_backend,
)

from ..fermionic_operations import FCircuit
from .fermionic_braket import FermBraketImpl
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
        ]
    if isinstance(objective, FCircuit):
        return simulate_fcircuit(
            *args,
            U=objective,
            variables=variables,
            backend=backend,
            samples=samples,
            **kwargs,
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
    if isinstance(fbackend, str):
        fbackend = fbackend.lower()
    if isinstance(backend, str):
        backend = backend.lower()
    if backend in SUPPORTED_FERMIONIC_BACKENDS:
        if fbackend is None:
            fbackend = backend
        if backend not in SUPPORTED_BACKENDS or (
            samples is not None and backend not in INSTALLED_SAMPLERS
        ):
            # the qubit part goes to the default of tequila (tequila's spex can't sample)
            backend = None
    fbackend = pick_backend(
        backend=fbackend, samples=samples, noise=noise, device=device
    )
    if isinstance(objective, QTensor):
        # compile every element, as tequila does
        compiled = [
            compile(
                o,
                variables=variables,
                samples=samples,
                simulate_density=simulate_density,
                backend=backend,
                noise=noise,
                device=device,
                fbackend=fbackend,
                **kwargs,
            )
            for o in objective.flatten()
        ]
        return QTensor(objective_list=compiled, shape=objective.shape)
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
                        if samples is not None and fbackend != "tequila":
                            compiled_expval = SampledBraket(arg, backend=fbackend)
                        else:
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
):  # Variables not passed since already inside FermBraketImpl
    if backend.lower() in INSTALLED_FERMIONIC_BACKENDS:
        return INSTALLED_FERMIONIC_BACKENDS[backend.lower()](objective, *args, **kwargs)
    else:
        raise ImportError(f"Fermionic backend not recognised {backend}")


class SampledBraket:
    """
    Braket of a fermionic backend estimated from measurements: each Pauli string of the operator
    (Jordan-Wigner, up-then-down, as in the tequila backend) is measured samples times on the
    wavefunctions of the fermionic circuit simulator
    """

    def __init__(self, braket: FermBraketImpl, backend: str):
        self.braket = braket
        self.backend = backend
        # the qubit operator of the tequila backend
        operator_only = FermBraketImpl(
            ket=FCircuit(), operator=braket.operator, molecule=braket.molecule
        )
        self.H = INSTALLED_FERMIONIC_BACKENDS["tequila"](operator_only).args[0].H[0]

    def extract_variables(self) -> list:
        return self.braket.extract_variables()

    def __call__(self, variables=None, samples: int | None = None, *args, **kwargs):
        n_orb = self.braket.molecule.n_orbitals
        ket = simulate_fcircuit(
            U=self.braket.ket, variables=variables, backend=self.backend, n_orb=n_orb
        )
        bra = ket
        if self.braket.bra is not None:
            bra = simulate_fcircuit(
                U=self.braket.bra,
                variables=variables,
                backend=self.backend,
                n_orb=n_orb,
            )
        result = 0.0
        for ps in self.H.paulistrings:
            # a measurement gives +1 with probability (1 + Re<bra|P|ket>)/2
            p = (1 + bra.inner(ket.apply_paulistring(ps.naked())).real) / 2
            p = min(max(p, 0.0), 1.0)
            result += ps.coeff * (2 * binomial(samples, p) / samples - 1)
        return float(result.real)
