"""Simulation of FCircuits with the TCC (tencirchem) backend."""

from __future__ import annotations

from warnings import warn

import numpy as np
from tequila import (
    BitNumbering,
    Objective,
    QubitWaveFunction,
    TequilaWarning,
    Variable,
    simulate,
)
from tequila.objective.objective import Variables

from sunrise.fermionic_operations import FCircuit
from sunrise.molecules.qubit_base import Molecule


def tcc_circuit_simulator(
    U: FCircuit, variables: Variables, n_orb: int | None = None, **backend_kwargs
) -> QubitWaveFunction:
    """Simulate an FCircuit with the TCC backend and return the final wavefunction.

    Parameters
    ----------
    U : FCircuit
        Circuit to simulate. It needs an initial state or number of electrons,
        otherwise an empty wavefunction is returned with a warning.
    variables : dict
        Values of the circuit variables.
    n_orb : int, optional
        Number of spatial orbitals. Default is inferred from the circuit.
    **backend_kwargs
        Passed on to ``TCCBraket``. Accepted keywords are ``engine``, ``backend``
        and ``dtype``.

    Returns
    -------
    QubitWaveFunction
        Final state (sparse, amplitudes below 1e-6 dropped).
    """
    from sunrise.expval.minimize import compile

    from .. import Braket

    if n_orb is None:
        n_orb = int(np.ceil((U.max_qubit() + 1) / 2))
    res = QubitWaveFunction(n_qubits=2 * n_orb, numbering=BitNumbering.LSB, dense=False)
    if U.n_electrons is None:
        warn("FCircuit with no initial_state provided.", TequilaWarning)
        return res
    U = U.to_upthendown(norb=n_orb)
    BK = Braket(
        ket=U,
        mol=__generate_dummy_mol(n_orb=n_orb, n_elec=U.n_electrons),
        backend_kwargs=backend_kwargs,
        operator="I",
    )
    tccBK = compile(BK, backend="tcc").args[0]
    variables = [map_variables(x, variables) for x in tccBK.BK.total_variables]
    state = tccBK.BK.statevector(angles=[-0.5 * i for i in variables])
    non_zero = np.argwhere(np.abs(state) > 1e-6)
    for i in non_zero:
        res._state[i[0]] = state[i[0]]
    return res


def __generate_dummy_mol(n_orb: int, n_elec: int):
    """Return a molecule with zero integrals and ``n_elec`` electrons, only used to build the braket."""
    h = np.zeros(shape=(n_orb, n_orb))
    g = np.zeros(shape=(n_orb, n_orb, n_orb, n_orb))
    geo = "".join([f"H 0. 0. {i}\n" for i in range(n_orb)])
    charge = n_orb - n_elec
    return Molecule(
        transformation="reordered-jordan-wigner",
        geometry=geo,
        charge=charge,
        basis_set="sto-3g",
        n_electrons=n_elec,
        nuclear_repulsion=0,
        one_body_integrals=h,
        two_body_integrals=g,
        units="a",
    )


def map_variables(x: list[Variable, Objective], dvariables: dict):
    """Return the value of ``x``, evaluating it with ``dvariables`` if it is a Variable or Objective."""
    if isinstance(x, Variable):
        x = x.map_variables(dvariables)
    elif isinstance(x, Objective):
        x = simulate(x, dvariables)
    return x
