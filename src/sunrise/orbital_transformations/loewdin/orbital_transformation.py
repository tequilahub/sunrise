"""Native (Löwdin) orbitals: coefficients and molecules in the orthonormalized basis functions."""

from __future__ import annotations

from copy import deepcopy

import numpy

from sunrise.molecules.fermionic_base import FermionicBase
from sunrise.molecules.hybrid_base import HybridBase
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase
from sunrise.orbital_transformations.orbital_transformation import (
    get_active,
    get_core,
    orthogonalize_active_space,
)


def get_native_coefficients(integral_manager) -> numpy.ndarray:
    """Return the coefficients of the native orbitals, the Löwdin-orthonormalized basis functions.

    Parameters
    ----------
    integral_manager : IntegralManager
        Integral manager of the molecule, which holds the overlap matrix of the basis.

    Returns
    -------
    numpy.ndarray
        Coefficient matrix :math:`S^{-1/2}`, the identity if the basis is already orthogonal.

    See Also
    --------
    transform_to_native_orbitals : Put these orbitals into the integral manager.
    """
    S = integral_manager.overlap_integrals
    if integral_manager.basis_is_orthogonal():
        return numpy.eye(S.shape[0])
    # not orthogonalize(eye, S): its rounding differs at 1e-16 and flips ties in get_active
    sv, U = numpy.linalg.eigh(S)
    s = numpy.diag(numpy.asarray([1.0 / numpy.sqrt(x) for x in sv]))
    return U.dot(s.dot(U.transpose()))


def transform_to_native_orbitals(integral_manager) -> None:
    """Replace the orbitals of an integral manager by the native orbitals, in place.

    Parameters
    ----------
    integral_manager : IntegralManager
        Integral manager to change.

    See Also
    --------
    get_native_coefficients : The coefficients that are set.
    generate_loewdin_molecule : Native orbitals for a whole molecule, with core and active space.
    """
    integral_manager.orbital_coefficients = get_native_coefficients(integral_manager)
    integral_manager._orbital_type = (
        f"orthonormalized-{integral_manager._basis_name}-basis"
    )


def generate_loewdin_molecule(
    mol: QuantumChemistryBase,
    inplace: bool = False,
    core: list | None = None,
    **kwargs,
) -> QuantumChemistryBase:
    """Generate a molecule in the native (Löwdin, orthonormalized atomic) orbital basis.

    For standard basis sets these are the orthonormalized Gaussian basis functions. This is what
    ``mol.use_native_orbitals`` calls, for qubit, hybrid and fermionic molecules. For hybrid molecules the F/B
    selection is carried over to the new orbitals, so check that it is still what you want.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Any sunrise molecule.
    inplace : bool, optional
        Update ``mol`` instead of returning a new one. Default is False.
    core : list of int, optional
        Core orbitals (frozen, doubly occupied), given in the currently used orbitals of the molecule
        (see ``print_basis_info``). Default is inherited from the active space of the integral manager.
        If given, the active native orbitals are chosen by their smallest overlap with the core orbitals.
    active : list of int, optional
        Keyword argument. Active orbitals in the native orbital basis. Default is all orbitals, or the
        ``N-len(core)`` orbitals with smallest overlap with ``core`` if ``core`` is given. For H, He, H,
        ``active=[0, 1, 2]`` selects the three atomic 1s orbitals; with ``core=[0]`` and no ``active``,
        ``active=[0, 2]`` is selected, since the He 1s orbital overlaps most with the lowest HF orbital.
    reference_orbitals : list of int, optional
        Keyword argument. Doubly occupied orbitals, ``N//2`` entries (N = total number of electrons).
        Sets the number of electrons in the active space.

    Returns
    -------
    QuantumChemistryBase
        Molecule in the native basis (``mol`` if ``inplace``), of the same class as ``mol``.

    See Also
    --------
    sunrise.molecules.qubit_base.qc_base.QuantumChemistryBase.use_orbitals : Pick any of the orbital bases by name.

    Examples
    --------
    >>> mol = sun.Molecule(geometry=geometry, basis_set="sto-3g")
    >>> mol = generate_loewdin_molecule(mol)
    """
    is_hybrid = isinstance(mol, HybridBase)
    is_fermionic = isinstance(mol, FermionicBase)

    c = mol.integral_manager.orbital_coefficients.copy()
    s = mol.integral_manager.overlap_integrals.copy()
    d = get_native_coefficients(mol.integral_manager)

    active = None
    if not mol.integral_manager.active_space_is_trivial() and core is None:
        core = [i.idx_total for i in mol.integral_manager.orbitals if i.idx is None]
    if "active" in kwargs:
        active = kwargs["active"]
        kwargs.pop("active")
        if core is None:
            core = get_core(c, d, s, active)
    else:
        if active is None:
            if core is None:
                core = []
                active = [i for i in range(len(mol.integral_manager.orbitals))]
            else:
                if isinstance(core, int):
                    core = [core]
                active = get_active(
                    c,
                    d,
                    s,
                    [
                        i
                        for i in range(len(mol.integral_manager.orbitals))
                        if i not in core
                    ],
                )
    assert len(active) + len(core) == len(mol.integral_manager.orbitals)
    reference_orbitals = [i.idx_total for i in mol.integral_manager.reference_orbitals]
    if "reference_orbitals" in kwargs:
        given = kwargs.pop("reference_orbitals")
        assert len(given) == mol.parameters.total_n_electrons // 2, (
            f"Number of  provided reference_orbitals incorrect. Expected {mol.parameters.total_n_electrons // 2}, received {len(given)}"
        )
        # kept from FermionicBase.use_native_orbitals, which ignored it for a new molecule with core orbitals
        if inplace or not is_fermionic:
            reference_orbitals = given
    to_active = [i for i in range(len(mol.integral_manager.orbitals)) if i not in core]
    to_active = {active[i]: to_active[i] for i in range(len(active))}
    if len(core):
        # keep the core orbitals at their own indices (`core` is passed as
        # frozen_idx below) and only replace the active columns
        c_combined = c.copy()
        for act_idx in active:
            c_combined[:, to_active[act_idx]] = d[:, act_idx]
        coeff = orthogonalize_active_space(c_combined, s, core, [*to_active.values()])
        if is_hybrid:
            if (
                not all(i == to_active[i] for i in to_active)
                and len(mol.BOS_MO)
                and len(mol.FER_MO)
            ):
                print("Orbital may be reordered, please double check F/B selection")
            if len(active) == len(mol.select):
                new_select = {i: mol.select[i] for i in range(len(active))}
            else:
                # do not shadow `s` (the overlap integrals) - still needed below
                remaining = {i: mol.select[i] for i in mol.select if i not in core}
                new_select = {
                    i: remaining[[*remaining.keys()][i]] for i in range(len(remaining))
                }
        integral_manager = mol.initialize_integral_manager(
            one_body_integrals=mol.integral_manager.one_body_integrals,
            two_body_integrals=mol.integral_manager.two_body_integrals,
            constant_term=mol.integral_manager.constant_term,
            active_orbitals=[*to_active.values()],
            reference_orbitals=reference_orbitals,
            frozen_orbitals=core,
            orbital_coefficients=coeff,
            overlap_integrals=s,
            orbital_type=f"orthonormalized-{mol.integral_manager._basis_name}-basis",
        )
        if inplace:
            mol.integral_manager = integral_manager
            if is_hybrid:
                mol.update_select(new_select)
            return mol
        parameters = deepcopy(mol.parameters)
        if is_hybrid:
            return HybridBase(
                parameters=parameters,
                integral_manager=integral_manager,
                transformation=mol.transformation,
                select=new_select,
                two_qubit=mol.two_qubit,
                condense=mol.condense,
                integral_tresh=mol.integral_tresh,
                active_orbitals=[*to_active.values()],
            )
        if is_fermionic:
            return FermionicBase(
                parameters=parameters,
                integral_manager=integral_manager,
                active_orbitals=[*to_active.values()],
            )
        return QuantumChemistryBase(
            parameters=parameters,
            integral_manager=integral_manager,
            transformation=mol.transformation,
            active_orbitals=[*to_active.values()],
        )
    # can not be an instance of a specific backend (otherwise we get inconsistencies with classical methods in the backend)
    if inplace:
        transform_to_native_orbitals(mol.integral_manager)
        return mol
    integral_manager = deepcopy(mol.integral_manager)
    transform_to_native_orbitals(integral_manager)
    if is_hybrid:
        return HybridBase(
            parameters=deepcopy(mol.parameters),
            integral_manager=integral_manager,
            transformation=mol.transformation,
            select=mol.select,
            two_qubit=mol.two_qubit,
            condense=mol.condense,
            integral_tresh=mol.integral_tresh,
        )
    if is_fermionic:
        return FermionicBase(
            parameters=mol.parameters,
            integral_manager=integral_manager,
            fermionic_backend=mol.fermionic_backend,
        )
    return QuantumChemistryBase(
        parameters=mol.parameters,
        integral_manager=integral_manager,
        transformation=mol.transformation,
    )
