"""Entry points to generate CLPO and HAO orbitals with JANPA, dispatching to the PySCF or Psi4
implementation.

References
----------
T. Y. Nikolaienko and L. A. Bulavin, Int. J. Quantum Chem. 119, e25798 (2019), doi:10.1002/qua.25798.
T. Y. Nikolaienko, L. A. Bulavin and D. M. Hovorun, Comput. Theor. Chem. 1050, 15 (2014), doi:10.1016/j.comptc.2014.10.002.
"""

from __future__ import annotations

from numbers import Number

from sunrise.molecules.qubit_base import INSTALLED_QCHEMISTRY_BACKENDS
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase

try:
    from .pyscf_janpa_interface import (
        generate_CLPO_molecule_edges_pyscf,
        generate_CLPO_molecule_pyscf,
        generate_HAO_molecule_pyscf,
    )
except ImportError:
    pass

try:
    from .psi4_janpa_interface import (
        generate_CLPO_molecule_edges_psi4,
        generate_CLPO_molecule_psi4,
        generate_HAO_molecule_psi4,
    )
except ImportError:
    pass

HAS_PYSCF = "pyscf" in INSTALLED_QCHEMISTRY_BACKENDS
HAS_PSI4 = "psi4" in INSTALLED_QCHEMISTRY_BACKENDS


def generate_CLPO_molecule_edges(
    mol: QuantumChemistryBase,
    edges: list[tuple[int]] | None = None,
    thres: Number = 1.0e-12,
    silent: bool = True,
    use_active: bool = True,
    return_aviso: bool = False,
    use_mp2: bool = False,
    backend: str | None = None,
    **kwargs,
) -> tuple[QuantumChemistryBase, list]:
    r"""Generate a molecule with CLPO orbitals and the corresponding graph edges.

    CLPOs (chemically localized orbitals) are built with JANPA from a natural population analysis; they
    are bonds, lone pairs and their antibonding partners, and define the graph used by SPA.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Any sunrise molecule.
    edges : list of tuple of int, optional
        Bonds imposed on the CLPOs, given as 0-based atom indices, e.g. ``[(0, 3), (1, 2)]``
        for linear H4 (repeat a pair for a double bond). The bonds
        between the other atom pairs are found automatically.
    thres : float
        Convergence threshold of the JANPA hybrid-orbital optimization.
    silent : bool
        If True, suppress the JANPA output.
    use_active : bool
        If True, keep the active space of ``mol`` (the CLPOs are rotated into it). Otherwise
        all orbitals are replaced by the orthogonalized CLPOs.
    return_aviso : bool
        If True, also return the orbitals that do not have the expected double occupancy
        (the same information as the emitted warnings).
    use_mp2 : bool
        If True, use MP2 to initialize JANPA, otherwise Hartree-Fock.
    backend : {"pyscf", "psi4"}, optional
        Chemistry backend for the HF/MP2 calculation and the MOLDEN file. If None it is guessed
        from the molecule type, the kwargs or the installed backends.
    **kwargs
        Passed to the backend. PySCF: ``pyscf_object`` or ``mo_coeff`` + ``mo_occ`` + ``mo_energy``.
        Psi4: ``wfn``. The result is then the responsibility of the user, JANPA only considers
        HF or MP2 references.

    Returns
    -------
    mol : QuantumChemistryBase
        Molecule with CLPO orbitals.
    edges : list of tuple of int
        Graph edges (pairs of orbital indices in the active space, or single indices for
        lone pairs), ready for SPA.
    aviso : list of tuple, optional
        Only if ``return_aviso`` is True. See Notes.

    See Also
    --------
    generate_CLPO_molecule : Only the molecule, without edges.
    generate_HAO_molecule : Hybrid atomic orbitals, the building blocks of the CLPOs.

    Notes
    -----
    Each aviso entry is ``(mo_type, orb_idx, occupation)``. It lists the orbitals that do not have
    the expected double occupancy: ``mo_type`` 1 is a lone pair with occupation between 0.5
    and 1.5, 2 a bonding/antibonding pair ``(idx, idx + 1)`` whose summed occupation is below 1.7.
    With ``use_active``, ``orb_idx`` is the active-space index (None if the orbital ended up in
    the frozen core), otherwise the index in the full CLPO list.

    Examples
    --------
    >>> mol = sun.Molecule(geometry="N 0 0 0\nN 0 0 1.1", basis_set="sto-3g")
    >>> mol, edges = sun.CLPO.generate_CLPO_molecule_edges(mol, silent=True)  # doctest: +SKIP
    >>> U = mol.make_ansatz("HCB-SPA", edges=edges)  # doctest: +SKIP
    """
    if backend is None:
        backend = __pick_chem_backend(mol, **kwargs)

    if backend.lower() == "psi4":
        return generate_CLPO_molecule_edges_psi4(
            mol=mol,
            edges=edges,
            thres=thres,
            silent=silent,
            use_active=use_active,
            return_aviso=return_aviso,
            use_mp2=use_mp2,
            **kwargs,
        )
    else:
        return generate_CLPO_molecule_edges_pyscf(
            mol=mol,
            edges=edges,
            thres=thres,
            silent=silent,
            use_active=use_active,
            return_aviso=return_aviso,
            use_mp2=use_mp2,
            **kwargs,
        )


def generate_HAO_molecule(
    mol: QuantumChemistryBase,
    thres: Number = 1.0e-9,
    silent: bool = True,
    use_active: bool = True,
    ret2act: bool = False,
    use_mp2: bool = False,
    backend: str = "pyscf",
    **kwargs,
) -> QuantumChemistryBase:
    """Generate a molecule with HAO (hybrid atomic orbital) orbitals.

    HAOs are the atom-centered hybrids that JANPA builds before combining them into CLPOs.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Any sunrise molecule.
    thres : float
        Convergence threshold of the JANPA hybrid-orbital optimization.
    silent : bool
        If True, suppress the JANPA output.
    use_active : bool
        If True, keep the active space of ``mol``. Otherwise all orbitals are replaced by the
        orthogonalized HAOs.
    ret2act : bool
        If True and ``use_active`` is True, also return the dictionary mapping the HAO index
        in the full list to its index after the active-space transformation.
    use_mp2 : bool
        If True, use MP2 to initialize JANPA, otherwise Hartree-Fock.
    backend : {"pyscf", "psi4"}, optional
        Chemistry backend for the HF/MP2 calculation and the MOLDEN file. If None it is guessed
        from the molecule type, the kwargs or the installed backends.
    **kwargs
        Passed to the backend. PySCF: ``pyscf_object`` or ``mo_coeff`` + ``mo_occ`` + ``mo_energy``.
        Psi4: ``wfn``. The result is then the responsibility of the user, JANPA only considers
        HF or MP2 references.

    Returns
    -------
    mol : QuantumChemistryBase
        Molecule with HAO orbitals.
    to_active : dict of int to int, optional
        Only if ``ret2act`` and ``use_active`` are True.

    See Also
    --------
    generate_CLPO_molecule : Chemically localized pair orbitals built from the HAOs.
    generate_CLPO_molecule_edges : CLPOs together with their SPA edges.

    Examples
    --------
    >>> mol = sun.CLPO.generate_HAO_molecule(mol, silent=True)  # doctest: +SKIP
    """
    if backend is None:
        backend = __pick_chem_backend(mol, **kwargs)
    if backend.lower() == "psi4":
        return generate_HAO_molecule_psi4(
            mol=mol,
            thres=thres,
            silent=silent,
            use_active=use_active,
            ret2act=ret2act,
            use_mp2=use_mp2,
            **kwargs,
        )
    else:
        return generate_HAO_molecule_pyscf(
            mol=mol,
            thres=thres,
            silent=silent,
            use_active=use_active,
            ret2act=ret2act,
            use_mp2=use_mp2,
            **kwargs,
        )


def generate_CLPO_molecule(
    mol: QuantumChemistryBase,
    edges: list | None = None,
    thres: Number = 1.0e-12,
    silent: bool = True,
    use_active: bool = True,
    return_aviso: bool = False,
    use_mp2: bool = False,
    backend: str = "pyscf",
    **kwargs,
) -> QuantumChemistryBase:
    """Generate a molecule with CLPO orbitals.

    Same as ``generate_CLPO_molecule_edges`` but the graph edges are not returned.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Any sunrise molecule.
    edges : list of tuple of int, optional
        Bonds imposed on the CLPOs, given as 0-based atom indices, e.g. ``[(0, 3), (1, 2)]``
        for linear H4 (repeat a pair for a double bond). The bonds
        between the other atom pairs are found automatically.
    thres : float
        Convergence threshold of the JANPA hybrid-orbital optimization.
    silent : bool
        If True, suppress the JANPA output.
    use_active : bool
        If True, keep the active space of ``mol`` (the CLPOs are rotated into it). Otherwise
        all orbitals are replaced by the orthogonalized CLPOs.
    return_aviso : bool
        If True, also return the orbitals that do not have the expected double occupancy
        (the same information as the emitted warnings).
    use_mp2 : bool
        If True, use MP2 to initialize JANPA, otherwise Hartree-Fock.
    backend : {"pyscf", "psi4"}, optional
        Chemistry backend for the HF/MP2 calculation and the MOLDEN file. If None it is guessed
        from the molecule type, the kwargs or the installed backends.
    **kwargs
        Passed to the backend. PySCF: ``pyscf_object`` or ``mo_coeff`` + ``mo_occ`` + ``mo_energy``.
        Psi4: ``wfn``. The result is then the responsibility of the user, JANPA only considers
        HF or MP2 references.

    Returns
    -------
    mol : QuantumChemistryBase
        Molecule with CLPO orbitals.
    aviso : list of tuple, optional
        Only if ``return_aviso`` is True. See Notes.

    See Also
    --------
    generate_CLPO_molecule_edges : Also return the SPA edges of the CLPOs.
    generate_HAO_molecule : Hybrid atomic orbitals, the building blocks of the CLPOs.

    Notes
    -----
    Each aviso entry is ``(mo_type, orb_idx, occupation)``. It lists the orbitals that do not have
    the expected double occupancy: ``mo_type`` 1 is a lone pair with occupation between 0.5
    and 1.5, 2 a bonding/antibonding pair ``(idx, idx + 1)`` whose summed occupation is below 1.7.
    With ``use_active``, ``orb_idx`` is the active-space index (None if the orbital ended up in
    the frozen core), otherwise the index in the full CLPO list.

    Examples
    --------
    >>> mol = sun.CLPO.generate_CLPO_molecule(mol)  # doctest: +SKIP
    """
    if backend is None:
        backend = __pick_chem_backend(mol, **kwargs)
    if backend.lower() == "psi4":
        return generate_CLPO_molecule_psi4(
            mol=mol,
            edges=edges,
            thres=thres,
            silent=silent,
            use_active=use_active,
            return_aviso=return_aviso,
            use_mp2=use_mp2,
            **kwargs,
        )
    else:
        return generate_CLPO_molecule_pyscf(
            mol=mol,
            edges=edges,
            thres=thres,
            silent=silent,
            use_active=use_active,
            return_aviso=return_aviso,
            use_mp2=use_mp2,
            **kwargs,
        )


def __pick_chem_backend(mol: QuantumChemistryBase, **kwargs) -> str:
    """Return "pyscf" or "psi4" from the installed backends, the molecule type or the kwargs (default "pyscf")."""
    if not HAS_PSI4 and HAS_PYSCF:
        return "pyscf"
    if not HAS_PYSCF and HAS_PSI4:
        return "psi4"
    if "psi4" in type(mol).__name__.lower():
        return "psi4"
    if "pyscf" in type(mol).__name__.lower():
        return "pyscf"
    if "pyscf_object" in kwargs or "mo_coeff" in kwargs:
        return "pyscf"
    if "wfn" in kwargs:
        return "psi4"
    return "pyscf"  # default
