from __future__ import annotations

from openfermion.ops.operators.fermion_operator import FermionOperator
from tequila import Objective, TequilaException
from tequila.objective.objective import identity

from sunrise.fermionic_operations.circuit import FCircuit
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase

from .fermionic_braket import FermBraketImpl

# in order of preference when no backend is chosen: spex is sunrise's own simulator
SUPPORTED_FERMIONIC_BACKENDS = ["spex", "tcc", "fqe", "tequila"]
INSTALLED_FERMIONIC_BACKENDS = {}

try:
    from .tcc.tcc_expval import TCCBraket

    INSTALLED_FERMIONIC_BACKENDS["tcc"] = TCCBraket
except ImportError:
    pass

try:
    from .fqe.fqe_expval import FQEBraKet

    INSTALLED_FERMIONIC_BACKENDS["fqe"] = FQEBraKet
except ImportError:
    pass
try:
    from .spex.spex_expval import SpexExpval

    INSTALLED_FERMIONIC_BACKENDS["spex"] = SpexExpval
except (ImportError, AttributeError):
    pass

from .tequila_expval import TequilaBraket

INSTALLED_FERMIONIC_BACKENDS["tequila"] = TequilaBraket


def show_available_modules():
    print("Available Fermionic Modules:")
    for k in INSTALLED_FERMIONIC_BACKENDS.items():
        print(k)


def show_supported_modules():
    print(SUPPORTED_FERMIONIC_BACKENDS)


def pick_backend(
    backend: str | None = None, samples: int | None = None, noise=None, device=None
) -> str:
    """
    choose, or verify, a fermionic backend for the user.
    Parameters
    ----------
    backend: str, optional:
        what backend to choose or verify. if None: choose for the user, following the order of SUPPORTED_FERMIONIC_BACKENDS.
        'random' chooses one of the installed backends at random.
    samples: int, optional:
        if int and not None, choose (verify) a backend which supports sampling.
    noise: str or NoiseModel, optional:
        if not None, choose (verify) a backend which supports the specified noise.
    device: optional:
        if not None, choose (verify) a backend which supports the specified device.

    Only the tequila backend, which simulates through the tequila qubit simulators, supports samples, noise and devices.

    Returns
    -------
    str:
        the name of the chosen (or verified) backend.
    """

    if len(INSTALLED_FERMIONIC_BACKENDS) == 0:
        raise TequilaException("No fermionic backends installed on your system")

    # the fermionic simulators only compute wavefunctions
    needs_tequila = samples is not None or noise is not None or device is not None

    if backend is None:
        if needs_tequila:
            return "tequila"
        for f in SUPPORTED_FERMIONIC_BACKENDS:
            if f in INSTALLED_FERMIONIC_BACKENDS:
                return f

    if hasattr(backend, "lower"):
        backend = backend.lower()

    if backend == "random":
        if needs_tequila:
            return "tequila"
        import random

        return random.choice(list(INSTALLED_FERMIONIC_BACKENDS.keys()))

    if backend not in SUPPORTED_FERMIONIC_BACKENDS:
        raise TequilaException(f"Fermionic backend {backend} not supported")
    elif backend not in INSTALLED_FERMIONIC_BACKENDS:
        raise TequilaException(f"Fermionic backend {backend} not installed")
    elif needs_tequila and backend != "tequila":
        raise TequilaException(
            f"Fermionic backend {backend} does not support samples, noise or devices, use the tequila backend"
        )

    return backend


def Fidelity(
    ket: FCircuit,
    bra: FCircuit,
    mol: QuantumChemistryBase | None = None,
    *args,
    **kwargs,
) -> Objective:
    """

    Convenience initialization of an tq.Objective that corresponds to the fidelity |<bra|ket>|^2 between two quantum states
    initialized by the circuits bra and ket

    Notes
    -----

    note, that the fidelity is symmetric: F(a,b) = F(b,a)
    so it does not matter what is bra and what is ket

    Parameters:
    ----------
    bra: FCircuit
    ket: FCircuit
    mol: Sunrise Molecule

    Returns:
    ----------
    A tq.Objective that evaluated to the fidelity between the two states

    """
    return Objective(
        args=[
            Braket(
                ket=ket, bra=bra, molecule=mol, operator="I", args=args, **kwargs
            ).args[0]
        ],
        transformation=lambda x: abs(x) ** 2,
    )


def Overlap(
    ket: FCircuit,
    bra: FCircuit,
    mol: QuantumChemistryBase | None = None,
    *args,
    **kwargs,
) -> Objective:
    """

    Function that calculates the overlap between two quantum states.

    Parameters:
    ----------
    bra: FCircuit
    ket: FCircuit
    mol: Sunrise Molecule

    Returns:
    ----------
    A tq.Objective that evaluated to the overlap between the two states

    """
    return Braket(ket=ket, bra=bra, operator=None, mol=mol, args=args, **kwargs)


def Braket(
    ket: FCircuit,
    bra: FCircuit | None = None,
    operator: str | list[FermionOperator] | FermionOperator | None = None,
    mol: QuantumChemistryBase | None = None,
    *args,
    **kwargs,
) -> Objective:
    """Function that allows to calculate different quantities
       depending on the passed parameters:
       1) If only ket is passed, returns the overlap with itself (1).
       2) If ket and bra are passed, returns the overlap between the two states.
       3) If ket and operator are passed, returns the expectation value of the operator for the given state.
       4) If ket, bra and operator are passed, returns the transition element of the operator.

       returns an instance of tq.Objective

    Args:
        ket (FCircuit): FCircuit corresponding to a state.
        bra (FCircuit, optional): FCircuit corresponding to a second state.
                                  Defaults to None.
        operator (FermionOperator, optional): Operator of which we want to
                                               calculate the transition element.
                                               Defaults to None unless molecule provided, then the fermionic
                                               Hamiltonian is taken as default.
        mol: (QuantumChemistryBase, Optional) Sunrise Molecule to get the information (n electrons, integrals...)
                                                if not enough data provided

    Returns:
        tq.Objective representing the BraKet
    """
    if "molecule" in kwargs:
        if mol is not None:
            raise TequilaException("Two molecules provided?")
        else:
            mol = kwargs["molecule"]
            kwargs.pop("molecule")
    if mol is not None and operator is None:
        operator = "H"
    if bra is not None and ket is None:
        ket = bra
        bra = None
    if isinstance(operator, list):
        return [
            Braket(ket=ket, bra=bra, operator=op, mol=mol, args=args, **kwargs)
            for op in operator
        ]
        # return QTensor(objective_list=arglist, shape=len(operator)) # TODO: create multidimensional
    if bra is None and operator is None:
        return Objective() + 1.0
    return Objective(
        args=[
            FermBraketImpl(
                ket=ket, bra=bra, operator=operator, molecule=mol, args=args, **kwargs
            )
        ],
        transformation=identity,
    )


def ExpectationValue(
    U: FCircuit,
    H: str | list[FermionOperator] | FermionOperator | None = "H",
    mol: QuantumChemistryBase | None = None,
    *args,
    **kwargs,
) -> Objective:
    """
    Function that calculates the Expectation Value of an Hamiltonian operator for a given quantum states.

    Parameters
    ----------
    U : FCircuit sunrise object, corresponding to the quantum state.

    H : QubitHamiltonian tequila object

    Returns
    -------
    Tequila objectives to be simulated or compiled.

    """
    return Braket(ket=U, operator=H, mol=mol, args=args, **kwargs)


def RealBraKet(
    ket: FCircuit,
    bra: FCircuit | None = None,
    operator: str | list[FermionOperator] | FermionOperator | None = None,
    mol: QuantumChemistryBase | None = None,
    *args,
    **kwargs,
):
    """
    Function that calculates the Reak part of a given Expectation Value of an Hamiltonian operator for a given quantum states.

    Parameters
    ----------
    U : FCircuit sunrise object, corresponding to the quantum state.

    H : QubitHamiltonian tequila object

    Returns
    -------
    Tequila objectives to be simulated or compiled.

    """
    return Braket(bra=bra, ket=ket, operator=operator, mol=mol, args=args, **kwargs)


def ImagBraKet(*args, **kwargs):
    """
    Function that calculates the Imaginary part of a given Expectation Value of an Hamiltonian operator for a given quantum states.
    Defined just for consistency with Tequila, since chemistry molecules should be always real

    Parameters
    ----------
    U : FCircuit sunrise object, corresponding to the quantum state.

    H : QubitHamiltonian tequila object

    Returns
    -------
    Tequila objectives to be simulated or compiled.

    """
    return Objective()
