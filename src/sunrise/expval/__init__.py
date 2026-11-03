"""Fermionic expectation values, overlaps and brakets built from FCircuits.

Entry points are Braket, ExpectationValue, Overlap and Fidelity, which return
tequila Objectives that are evaluated by one of the fermionic backends
("spex", "tcc", "fqe", "ffsim") or by the tequila qubit simulators.
"""

from __future__ import annotations

from openfermion.ops.operators.fermion_operator import FermionOperator
from tequila import Objective, TequilaException
from tequila.objective.objective import identity

from sunrise.fermionic_operations.circuit import FCircuit
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase

from .fermionic_braket import FermBraketImpl

# in order of preference when no backend is chosen: spex is sunrise's own simulator
SUPPORTED_FERMIONIC_BACKENDS = ["spex", "tcc", "fqe", "ffsim", "tequila"]
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
try:
    from .ffsim.ffsim_expval import FFS_EXPVAL

    INSTALLED_FERMIONIC_BACKENDS["ffsim"] = FFS_EXPVAL
except ImportError:
    pass

from .tequila_expval import TequilaBraket

INSTALLED_FERMIONIC_BACKENDS["tequila"] = TequilaBraket


def show_available_modules():
    """Print the names of the fermionic backends installed on this system."""
    print("Available Fermionic Modules:")
    for k in INSTALLED_FERMIONIC_BACKENDS:
        print(k)


def show_supported_modules():
    """Print the names of the supported fermionic backends, in order of preference."""
    print(SUPPORTED_FERMIONIC_BACKENDS)


def pick_backend(
    backend: str | None = None, samples: int | None = None, noise=None, device=None
) -> str:
    """Choose or verify a fermionic backend.

    Only the "tequila" backend, which simulates through the tequila qubit
    simulators, supports samples, noise and devices.

    Parameters
    ----------
    backend : str, optional
        Backend to verify. If None, the first installed backend in
        ``SUPPORTED_FERMIONIC_BACKENDS`` is chosen. "random" picks an installed
        backend at random.
    samples : int, optional
        If given, only a backend supporting sampling is accepted.
    noise : str or NoiseModel, optional
        If given, only a backend supporting noise is accepted.
    device : optional
        If given, only a backend supporting devices is accepted.

    Returns
    -------
    str
        Name of the chosen (or verified) backend.

    Raises
    ------
    TequilaException
        If no backend is installed, the requested one is unknown, not installed,
        or cannot handle samples, noise or devices.

    Examples
    --------
    >>> sun.pick_backend()  # name of the default fermionic backend
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
    r"""Create an Objective for the fidelity :math:`|\langle bra|ket \rangle|^2`.

    The fidelity is symmetric, so it does not matter which circuit is bra and
    which is ket.

    Parameters
    ----------
    ket : FCircuit
        Circuit preparing the first state.
    bra : FCircuit
        Circuit preparing the second state.
    mol : QuantumChemistryBase, optional
        Molecule providing the number of orbitals and electrons if they cannot be
        inferred from the circuits.
    *args, **kwargs
        Passed on to ``Braket``.

    Returns
    -------
    tq.Objective
        Objective evaluating to the fidelity of the two states.

    See Also
    --------
    Overlap : Overlap <bra|ket> itself.
    Braket : General overlap, expectation value or transition element.

    Examples
    --------
    >>> F = sun.Fidelity(ket=U1, bra=U2)  # |<U2|U1>|^2
    >>> sun.simulate(F, variables=variables)
    """
    return Objective(
        args=[
            Braket(*args, ket=ket, bra=bra, molecule=mol, operator="I", **kwargs).args[
                0
            ]
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
    r"""Create an Objective for the overlap :math:`\langle bra|ket \rangle`.

    Parameters
    ----------
    ket : FCircuit
        Circuit preparing the ket state.
    bra : FCircuit
        Circuit preparing the bra state.
    mol : QuantumChemistryBase, optional
        Molecule providing the number of orbitals and electrons if they cannot be
        inferred from the circuits.
    *args, **kwargs
        Passed on to ``Braket``.

    Returns
    -------
    tq.Objective
        Objective evaluating to the overlap of the two states.

    See Also
    --------
    Braket : General overlap, expectation value or transition element.
    Fidelity : Squared overlap |<bra|ket>|^2.

    Examples
    --------
    >>> S = sun.Overlap(ket=U1, bra=U2)  # <U2|U1>
    >>> sun.simulate(S, variables=variables)
    """
    return Braket(*args, ket=ket, bra=bra, operator="I", mol=mol, **kwargs)


def Braket(
    ket: FCircuit,
    bra: FCircuit | None = None,
    operator: str | list[FermionOperator] | FermionOperator | None = None,
    mol: QuantumChemistryBase | None = None,
    *args,
    **kwargs,
) -> Objective:
    """Create an Objective for an overlap, expectation value or transition element.

    The result depends on which arguments are given:

    - only ``ket``: overlap of the state with itself (1);
    - ``ket`` and ``bra``: overlap between the two states;
    - ``ket`` and ``operator``: expectation value of the operator;
    - ``ket``, ``bra`` and ``operator``: transition element of the operator.

    Parameters
    ----------
    ket : FCircuit
        Circuit preparing the ket state.
    bra : FCircuit, optional
        Circuit preparing the bra state. If None, the ket is used as bra.
    operator : str or FermionOperator or list of FermionOperator, optional
        Operator to evaluate. A string is one of "H" (molecular Hamiltonian),
        "HCB" (hard-core boson Hamiltonian), "I" (identity) or a tequila
        Pauli string. If None, it defaults to "H" when ``mol`` is given. A list
        returns a list of Objectives, one per operator.
    mol : QuantumChemistryBase, optional
        Sunrise molecule providing the integrals and number of electrons.
        May also be passed as the keyword ``molecule``.
    *args, **kwargs
        Passed on to the backend braket.

    Returns
    -------
    tq.Objective or list of tq.Objective
        Objective to be compiled or simulated with ``sunrise.simulate`` or
        ``sunrise.minimize``.

    Raises
    ------
    TequilaException
        If both ``mol`` and ``molecule`` are passed.

    See Also
    --------
    ExpectationValue : Expectation value <U|H|U> of a single circuit.
    Overlap : Overlap <bra|ket> of two circuits, without an operator.
    Fidelity : Squared overlap |<bra|ket>|^2 of two circuits.
    sunrise.simulate : Evaluate the returned Objective for given variable values.

    Examples
    --------
    >>> E = sun.Braket(ket=U, mol=mol)  # energy <U|H|U>
    >>> H12 = sun.Braket(ket=U1, bra=U2, mol=mol)  # transition element <U2|H|U1>
    >>> S12 = sun.Braket(ket=U1, bra=U2, mol=mol, operator="I")  # overlap <U2|U1>
    >>> energy = sun.simulate(E, variables={"a": 0.1})
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
            Braket(*args, ket=ket, bra=bra, operator=op, mol=mol, **kwargs)
            for op in operator
        ]
        # return QTensor(objective_list=arglist, shape=len(operator)) # TODO: create multidimensional
    if bra is None and operator is None:
        return Objective() + 1.0
    return Objective(
        args=[
            FermBraketImpl(
                *args, ket=ket, bra=bra, operator=operator, molecule=mol, **kwargs
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
    """Create an Objective for the expectation value of an operator in a state.

    Parameters
    ----------
    U : FCircuit
        Circuit preparing the state.
    H : str or FermionOperator or list of FermionOperator, optional
        Operator to evaluate, see ``Braket``. Default is "H", the molecular
        Hamiltonian of ``mol``.
    mol : QuantumChemistryBase, optional
        Sunrise molecule providing the integrals and number of electrons.
    *args, **kwargs
        Passed on to ``Braket``.

    Returns
    -------
    tq.Objective or list of tq.Objective
        Objective to be compiled or simulated.

    See Also
    --------
    Braket : General overlap, expectation value or transition element.
    sunrise.minimize : Minimize the returned Objective.

    Examples
    --------
    >>> E = sun.ExpectationValue(U=U, mol=mol)
    >>> energy = sun.simulate(E, variables={"a": 0.1})
    """
    return Braket(*args, ket=U, operator=H, mol=mol, **kwargs)


def RealBraKet(
    ket: FCircuit,
    bra: FCircuit | None = None,
    operator: str | list[FermionOperator] | FermionOperator | None = None,
    mol: QuantumChemistryBase | None = None,
    *args,
    **kwargs,
):
    """Create an Objective for a braket, see ``Braket``.

    Kept for consistency with tequila. Molecular states are real, so this is the
    same as ``Braket`` and no real part is explicitly taken.

    Parameters
    ----------
    ket : FCircuit
        Circuit preparing the ket state.
    bra : FCircuit, optional
        Circuit preparing the bra state.
    operator : str or FermionOperator or list of FermionOperator, optional
        Operator to evaluate, see ``Braket``.
    mol : QuantumChemistryBase, optional
        Sunrise molecule providing the integrals and number of electrons.
    *args, **kwargs
        Passed on to ``Braket``.

    Returns
    -------
    tq.Objective or list of tq.Objective
        Objective to be compiled or simulated.
    """
    return Braket(*args, bra=bra, ket=ket, operator=operator, mol=mol, **kwargs)


def ImagBraKet(*args, **kwargs):
    """Return an empty Objective (zero).

    Defined only for consistency with tequila: molecular states are real, so the
    imaginary part of a braket is always zero.
    """
    return Objective()
