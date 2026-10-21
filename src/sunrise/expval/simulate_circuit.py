from __future__ import annotations

from numpy import ceil, zeros
from tequila import SUPPORTED_BACKENDS, QCircuit, QubitWaveFunction, TequilaException
from tequila import simulate as tq_simulate
from tequila.objective import format_variable_dictionary
from tequila.objective.objective import Variables

from sunrise.expval import pick_backend
from sunrise.fermionic_operations import FCircuit
from sunrise.molecules.qubit_base import Molecule

SUPPORTED_FERMIONIC_CIRCUIT_SIMULATOR = ["fqe", "tcc", "spex"]
INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR = {}

try:
    from .tcc.simulate import tcc_circuit_simulator

    INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR["tcc"] = tcc_circuit_simulator
except ImportError:
    pass

try:
    from sunrise.expval.fqe.fqe_circuit_sim import fqe_circuit_simulatorU

    INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR["fqe"] = fqe_circuit_simulatorU
except ImportError:
    pass
try:
    from .spex.spex_expval import spex_circuit_simulator

    INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR["spex"] = spex_circuit_simulator
except (ImportError, AttributeError):
    pass


def simulate_fcircuit(
    U: FCircuit,
    variables: Variables | dict | None,
    backend: str | None = None,
    **kwargs,
) -> QubitWaveFunction:
    """
    Interface with Fermionic Backends to sumulate FCircuits
    :param U: Fcircuit to simulate
    :param variables: dictionary of {variable:value} to simulate
    :param backend: name of the desired backend, if None chosen by pick_backend

    If qubit based backend (see tequila.SUPPORTED_BACKENDS) the n_orb/molecule/FCircuit->QCircuit callable is required.
    """
    if variables is None:
        variables = {}
    if isinstance(U, QCircuit):
        return tq_simulate(U, variables, backend=backend, **kwargs)
    variables = format_variable_dictionary(variables)
    if variables is None and not (len(U.extract_variables()) == 0):
        raise TequilaException(
            f"You called simulate for a parametrized type but forgot to pass down the variables: {U.extract_variables()}"
        )
    if backend is None:
        backend = pick_backend(samples=kwargs.get("samples"), noise=kwargs.get("noise"))
    backend = backend.strip().lower()
    # the fermionic simulators first: tequila also has a (qubit) backend named spex, reachable as tequila-spex
    if backend in SUPPORTED_FERMIONIC_CIRCUIT_SIMULATOR:
        if backend in INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR:
            simulator = INSTALLED_FERMIONIC_CIRCUIT_SIMULATOR[backend]
            return simulator(U=U, variables=variables, **kwargs)
        else:
            raise TequilaException(f"Backend {backend} not installed.")
    elif (
        "tequila" in backend
        or backend == "t"
        or backend.startswith("t-")
        or backend in SUPPORTED_BACKENDS
    ):  # to use the qubit spex call backend='tequila-spex' or 't-spex'
        if backend.lower() == "tequila" or backend.lower() == "t":
            backend = None
        elif "tequila-" in backend:
            backend = backend[backend.find("tequila-") + 8 :]
        elif "t-" in backend:
            backend = backend[backend.find("t-") + 2 :]
        if "mol" in kwargs:
            mol = kwargs["mol"]
            kwargs.pop("mol")
            mol.transformation.up_then_down = True
            U = U.to_qcircuit(molecule=mol)
        elif "molecule" in kwargs:
            mol = kwargs["molecule"]
            kwargs.pop("molecule")
            mol.transformation.up_then_down = True
            U = U.to_qcircuit(molecule=mol)
        elif "transformation" in kwargs:
            transformation = kwargs["transformation"]
            kwargs.pop("transformation")
            U = U.to_qcircuit(transformation=transformation)
        elif "n_orb" in kwargs:
            n_orb = kwargs["n_orb"]
            kwargs.pop("n_orb")
            mol = __generate_dummy_mol(n_orb)
            U = U.to_qcircuit(mol)
        else:
            n_orb = int(ceil(U.max_qubit() + 1 / 2))
            U = U.to_qcircuit(molecule=__generate_dummy_mol(n_orb=n_orb))
        return tq_simulate(U, backend=backend, variables=variables, **kwargs)
    else:
        raise TequilaException(f"Not recognised backed: {backend}.")


def __generate_dummy_mol(n_orb: int):
    h = zeros(shape=(n_orb, n_orb))
    g = zeros(shape=(n_orb, n_orb, n_orb, n_orb))
    geo = "".join([f"H 0. 0. {i}\n" for i in range(n_orb)])
    return Molecule(
        transformation="reordered-jordan-wigner",
        geometry=geo,
        basis_set="custom",
        n_electrons=n_orb,
        nuclear_repulsion=0,
        one_body_integrals=h,
        two_body_integrals=g,
        units="a",
    )
