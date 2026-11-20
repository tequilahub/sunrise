"""PySCF-engine energy and gradient routines, extending tencirchem's evolve_pyscf to handle a bra different from the ket."""

from tencirchem.static.evolve_pyscf import *
from tencirchem.static.evolve_pyscf import _get_gradients_pyscf
from tequila import grad, simulate
from tequila.objective.objective import Objective, Variable


def get_expval_and_grad_pyscf(
    angles,
    hamiltonian,
    n_qubits,
    n_elec_s,
    total_variables,
    params,
    ex_ops: Tuple,
    mode: str = "fermion",
    init_state=None,
    params_bra=None,
    ex_ops_bra: Tuple = None,
    init_state_bra=None,
):
    """Return the expectation value, its gradient and the bra-ket overlap using the PySCF engine.

    Parameters
    ----------
    angles : array_like
        Values of ``total_variables`` (tcc angles).
    hamiltonian : callable
        Operator applied to the states.
    n_qubits : int
        Number of spin orbitals.
    n_elec_s : int or tuple of int
        Number of electrons (per spin).
    total_variables : list of Variable
        All variables, in the order of ``angles``.
    params : list
        Gate parameters of the ket (Variables, Objectives or numbers).
    ex_ops : tuple
        Excitation operators of the ket.
    mode : str
        Fermion encoding mode of tencirchem. Default is "fermion".
    init_state : array_like, optional
        Initial ket state.
    params_bra, ex_ops_bra, init_state_bra : optional
        Same as the ket quantities, for the bra. Default is the ket ones.

    Returns
    -------
    tuple
        (<bra|H|ket>, gradient, <bra|ket>).
    """
    if ex_ops_bra is None:
        ex_ops_bra = ex_ops
    if params_bra is None:
        params_bra = params
    if init_state_bra is None:
        init_state_bra = init_state

    # We receive list [x0,x1,...] refering to the total variables
    assert len(angles) == len(total_variables)
    # To build the CIVector we need the already maped params
    map_params = params_to_tcc(params, angles, total_variables)
    map_params_bra = params_to_tcc(params_bra, angles, total_variables)
    theta = tequila_values(angles, total_variables)

    ket = get_civector_pyscf(
        map_params, n_qubits, n_elec_s, ex_ops, [*range(len(ex_ops))], mode, init_state
    )
    bra = get_civector_pyscf(
        map_params_bra,
        n_qubits,
        n_elec_s,
        ex_ops_bra,
        [*range(len(ex_ops_bra))],
        mode,
        init_state_bra,
    )
    hbra = tc.backend.numpy(apply_op(hamiltonian, bra))
    hket = tc.backend.numpy(apply_op(hamiltonian, ket))
    energy = hbra @ ket

    gradients_beforesum = _get_gradients_pyscf(
        bra=hbra,
        ket=ket,
        params=map_params,
        n_qubits=n_qubits,
        n_elec_s=n_elec_s,
        ex_ops=ex_ops,
        param_ids=[*range(len(ex_ops))],
        mode=mode,
    )
    gradients_beforesum_bra = _get_gradients_pyscf(
        ket=bra,
        bra=hket,
        params=map_params_bra,
        n_qubits=n_qubits,
        n_elec_s=n_elec_s,
        ex_ops=ex_ops_bra,
        param_ids=[*range(len(ex_ops_bra))],
        mode=mode,
    )

    ang_grad = np.zeros((len(params), len(total_variables)))
    ang_grad_bra = np.zeros((len(params_bra), len(total_variables)))
    # TODO: Improve this using tequila Objective, and copy paste in the other functions
    # the tcc angle of a gate is -p(theta)/2 with theta = -2*angle, so d(gate angle)/d(angle) = dp/dtheta at theta
    for i, pa in enumerate(params):
        if not isinstance(pa, (Variable, Objective)):
            continue
        for j, an in enumerate(total_variables):
            ang_grad[i, j] = simulate(grad(1 * pa, an), variables=theta)
    for i, pa in enumerate(params_bra):
        if not isinstance(pa, (Variable, Objective)):
            continue
        for j, an in enumerate(total_variables):
            ang_grad_bra[i, j] = simulate(grad(1 * pa, an), variables=theta)
    gradients = np.add(
        gradients_beforesum.dot(ang_grad), gradients_beforesum_bra.dot(ang_grad_bra)
    )
    return energy, gradients, bra @ ket


def get_energy_and_grad_pyscf(
    angles,
    hamiltonian,
    n_qubits,
    n_elec_s,
    total_variables,
    params,
    ex_ops: Tuple,
    mode: str = "fermion",
    init_state=None,
):
    """Return the energy and its gradient using the PySCF engine.

    Parameters
    ----------
    angles : array_like
        Values of ``total_variables`` (tcc angles).
    hamiltonian : callable
        Operator applied to the states.
    n_qubits : int
        Number of spin orbitals.
    n_elec_s : int or tuple of int
        Number of electrons (per spin).
    total_variables : list of Variable
        All variables, in the order of ``angles``.
    params : list
        Gate parameters of the ket (Variables, Objectives or numbers).
    ex_ops : tuple
        Excitation operators of the ket.
    mode : str
        Fermion encoding mode of tencirchem. Default is "fermion".
    init_state : array_like, optional
        Initial ket state.

    Returns
    -------
    tuple
        (energy, gradient).
    """
    assert len(angles) == len(total_variables)
    map_params = params_to_tcc(params, angles, total_variables)
    theta = tequila_values(angles, total_variables)
    ket = get_civector_pyscf(
        map_params, n_qubits, n_elec_s, ex_ops, [*range(len(ex_ops))], mode, init_state
    )
    bra = tc.backend.numpy(apply_op(hamiltonian, ket))
    energy = bra @ ket

    gradients_beforesum = _get_gradients_pyscf(
        bra, ket, map_params, n_qubits, n_elec_s, ex_ops, [*range(len(ex_ops))], mode
    )
    ang_grad = np.zeros((len(params), len(total_variables)))
    for i, pa in enumerate(params):
        if not isinstance(pa, (Variable, Objective)):
            continue
        for j, an in enumerate(total_variables):
            ang_grad[i, j] = simulate(grad(1 * pa, an), variables=theta)
    gradients = gradients_beforesum.dot(ang_grad)

    return energy, 2 * gradients


def map_variables(x: list[Variable, Objective], dvariables: dict):
    """Return the value of x, evaluating it with dvariables if it is a Variable or Objective."""
    if isinstance(x, Variable):
        x = x.map_variables(dvariables)
    elif isinstance(x, Objective):
        x = simulate(x, dvariables)
    return x


def tequila_values(angles, total_variables) -> dict:
    """Return {variable name: value} in tequila units from tcc angles (angle = -theta/2)."""
    return {total_variables[i].name: -2.0 * angles[i] for i in range(len(angles))}


def params_to_tcc(params, angles, total_variables):
    """Return the gate parameters as tcc angles.

    Variables and Objectives are expressions in tequila units (for example
    theta + pi/2 from a shift rule), so they are evaluated in tequila units
    and then converted to -(theta + s)/2. Numbers and FixedVariables are already
    tcc angles.

    Parameters
    ----------
    params : list
        Gate parameters.
    angles : array_like
        Values of total_variables (tcc angles).
    total_variables : list of Variable
        All variables, in the order of angles.

    Returns
    -------
    numpy.ndarray
        Gate angles in tcc convention.
    """
    theta = tequila_values(angles, total_variables)
    pa = [
        -0.5 * map_variables(p, theta) if isinstance(p, (Variable, Objective)) else p
        for p in params
    ]
    return tc.backend.numpy(tc.backend.convert_to_tensor(pa).astype(tc.rdtypestr))
