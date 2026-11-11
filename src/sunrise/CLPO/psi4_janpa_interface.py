"""Generation of CLPO and HAO orbitals with JANPA on top of Psi4."""

from __future__ import annotations

import contextlib
import os
import tempfile
from copy import deepcopy
from numbers import Number
from warnings import warn

import numpy as np
from janpa.clpo.lodesc import LO_TYPE_BD, LO_TYPE_LP, LO_TYPE_NB, LO_TYPE_RY
from janpa.clpo.lpo import CLPOOptions, create_clpos
from janpa.io.molden import MoldenFile
from janpa.npa.npa import run_npa

from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase
from sunrise.utils.orbital_transformation import orthogonalize, transform

try:
    import psi4
except ImportError:
    pass


def extract_clpo_graph(
    npa_res, clpo_res, avisar: bool = True
) -> tuple[list[tuple[int, ...]], str]:
    """Extract the CLPO graph from JANPA results.

    Parameters
    ----------
    npa_res : object
        JANPA natural population analysis result, from ``janpa.npa.npa.run_npa``.
    clpo_res : object
        JANPA CLPO result, from ``janpa.clpo.lpo.create_clpos``.
    avisar : bool
        If True, warn about orbitals that do not have the expected double occupancy.

    Returns
    -------
    nodes : list of tuple of int
        Orbital indices of each graph node: ``(i,)`` for a lone pair, ``(i, i + 1)`` for a bond.
    aviso : list of tuple
        ``(lo_type, index, occupation)`` of the orbitals without the expected double occupancy.
    """
    Q = clpo_res.clpo.nao_to_hybrids @ clpo_res.clpo.lo_to_hybrids.T
    sds_clpo = Q.T @ npa_res.sds_nao @ Q
    occupancies = np.diag(sds_clpo)
    lo_types = clpo_res.clpo.lo_types

    nodes = []
    n_lo = len(occupancies)
    i = 0
    aviso = []
    while i < n_lo:
        lo_type = lo_types[i]
        occ = round(occupancies[i], 5)

        if lo_type == LO_TYPE_LP:
            if 0.5 < occ < 1.5:
                aviso.append((lo_type, i, occ))
            if occ < 1.0:
                i += 1
                continue
            nodes.append((i,))
            i += 1

        elif lo_type == LO_TYPE_RY:
            i += 1

        elif lo_type == LO_TYPE_BD:
            if i + 1 >= n_lo or lo_types[i + 1] != LO_TYPE_NB:
                raise ValueError(f"BD orbital {i} without following NB orbital")
            bd_occ = occ
            nb_occ = round(occupancies[i + 1], 5)
            if bd_occ + nb_occ < 1.7:
                aviso.append((lo_type, i, bd_occ + nb_occ))
            nodes.append((i, i + 1))
            i += 2

        else:
            i += 1

    if avisar:
        __raise_aviso(aviso)
    return nodes, aviso


def _molden_to_psi4_ao_order(wfn) -> np.ndarray:
    """Return ``perm`` where ``perm[k]`` is the Psi4 AO index of the k-th MOLDEN basis function (p shells differ: x, y, z vs z, x, y)."""
    bs = wfn.basisset()
    perm = []
    for sh in range(bs.nshell()):
        shell = bs.shell(sh)
        order = (
            [1, 2, 0] if shell.am == 1 and bs.has_puream() else range(shell.nfunction)
        )
        perm += [shell.function_index + i for i in order]
    return np.array(perm)


def _run_janpa_pipeline(
    wfn,
    thres: Number = 1e-9,
    silent=True,
    custom_edges=None,
    avisar: bool = True,
):
    """Run JANPA (NPA and CLPO) on a Psi4 wavefunction through a MOLDEN file and return the AHO and CLPO coefficients (n_ao, n_orb) in the Psi4 AO order, the graph and the aviso list."""
    bs = wfn.basisset()
    if not bs.has_puream() and bs.max_am() >= 2:
        raise ValueError(
            "JANPA needs spherical basis functions, set the psi4 option 'puream' to True"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        molden_in = os.path.join(tmpdir, "temp.molden")

        with contextlib.ExitStack() as stack:
            if silent:
                devnull = stack.enter_context(open(os.devnull, "w"))
                stack.enter_context(contextlib.redirect_stdout(devnull))

            # 1. Export initial MOLDEN
            psi4.molden(wfn, molden_in)

            # 2. Load and prepare. psi4 writes the raw contraction coefs, which are not normalized for general
            # contractions (e.g. cc-pVXZ), while its MO coefs refer to normalized basis functions
            molden_file = MoldenFile.load(molden_in)
            molden_file.coords_to_au()
            molden_file.to_unnormalized_primitive_coefs()
            molden_file.normalize_contractions()

            # 3. NPA
            npa_res = run_npa(molden_file)

            # 4. CLPO
            clpo_opts = CLPOOptions(hybr_opt_conv_thresh=thres, hybr_opt_max_iter=1000)
            if custom_edges is not None:
                # Bonds between 0-based atoms, a string like "[(1, 2), (1, 2)]"
                clpo_opts.edges = str(custom_edges)

            clpo_res = create_clpos(
                npa_res.sds_nao, npa_res.nao, molden_file.centers, clpo_opts
            )

            # 5. Extract Graph
            graph, aviso = extract_clpo_graph(npa_res, clpo_res, avisar)

            # 6. Get Transformation Matrices
            aho_to_ao = clpo_res.clpo.nao_to_hybrids.T @ npa_res.nao_to_ao
            clpo_to_ao = clpo_res.clpo.lo_to_hybrids @ aho_to_ao

    perm = _molden_to_psi4_ao_order(wfn)
    aho_psi4 = np.zeros_like(aho_to_ao)
    aho_psi4[:, perm] = aho_to_ao
    clpo_psi4 = np.zeros_like(clpo_to_ao)
    clpo_psi4[:, perm] = clpo_to_ao
    # (n_orb, n_ao) -> (n_ao, n_orb), columns are orbitals
    return aho_psi4.T, clpo_psi4.T, graph, aviso


def generate_CLPO_molecule_edges_psi4(
    mol: QuantumChemistryBase,
    edges: list[tuple[int]] | None = None,
    thres: Number = 1.0e-12,
    silent: bool = True,
    use_active: bool = True,
    return_aviso: bool = False,
    use_mp2: bool = False,
    **kwargs,
) -> tuple[QuantumChemistryBase, list]:
    """Generate a molecule with CLPO orbitals and the corresponding graph edges via Psi4.

    CLPOs (Chemist's Localized Property-optimized Orbitals) are built with JANPA from a natural population analysis; they
    are bonds, lone pairs and their antibonding partners, and define the graph used by SPA.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Any sunrise molecule.
    edges : list of tuple of int, optional
        Bonds imposed on the CLPOs, given as 0-based atom indices, e.g. ``[(0, 3), (1, 2)]``
        for linear H4 (repeat a pair for a double bond). The bonds
        between the other atom pairs are found automatically from the Hartree-Fock density
        (MP2 with ``use_mp2``).
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
        If True, run MP2 to initialize JANPA, otherwise use the Hartree-Fock wavefunction.
    **kwargs
        ``wfn`` (custom Psi4 wavefunction) is currently not supported: it fails for Psi4 molecules and is ignored for others.
        With ``use_active``, ``core`` and ``active`` (orbital indices in the full space) are passed
        on to the orbital transformation, see ``sunrise.utils.orbital_transformation.transform``.

    Returns
    -------
    mol : QuantumChemistryBase
        Molecule with CLPO orbitals.
    edges : list of tuple of int
        Graph edges (pairs of orbital indices in the active space, or single indices for
        lone pairs), ready for SPA.
    aviso : list of tuple, optional
        Only if ``return_aviso`` is True. See Notes.

    Notes
    -----
    Each aviso entry is ``(mo_type, orb_idx, occupation)``. It lists the orbitals that do not have
    the expected double occupancy: ``mo_type`` 1 is a lone pair with occupation between 0.5
    and 1.5, 2 a bonding/antibonding pair ``(idx, idx + 1)`` whose summed occupation is below 1.7.
    With ``use_active``, ``orb_idx`` is the active-space index (None if the orbital ended up in
    the frozen core), otherwise the index in the full CLPO list.
    """

    # TODO: support a custom wfn (check kwargs.pop("wfn") first) and fix use_mp2 on psi4 molecules
    # (psi4mol is undefined there, use molecule=wfn.molecule())
    if not hasattr(mol, "ref_wfn"):  # its not a psi4 base
        psi4mol = psi4.geometry(mol.parameters.get_geometry_string(ghost="psi4"))
        psi4mol.set_multiplicity(mol.parameters.multiplicity)
        psi4mol.set_molecular_charge(mol.parameters.charge)
        psi4.activate(psi4mol)
        _, wfn = psi4.energy(name="scf", return_wfn=True, molecule=psi4mol)
    elif "wfn" in kwargs:
        wfn = getattr(kwargs, "wfn", None)
    else:
        wfn = getattr(mol, "ref_wfn", None)

    if use_mp2:
        _, wfn = psi4.energy(
            name="mp2", return_wfn=True, molecule=psi4mol, guess_wfn=wfn
        )

    custom_edges = None
    if edges is not None:
        custom_edges = [tuple([int(e) for e in edge]) for edge in edges]

    _, clpo_to_ao_full, graph_full, aviso = _run_janpa_pipeline(
        wfn,
        thres=thres,
        silent=silent,
        custom_edges=custom_edges,
        avisar=not use_active,
    )

    if use_active:
        nmol_full = deepcopy(mol)
        nmol_full.integral_manager.orbital_coefficients = clpo_to_ao_full
        nmol_full.integral_manager._orbital_type = "CLPO"

        mol_out, to_active = transform(
            original=mol, modified=nmol_full, orbital_type="CLPO"
        )

        ncore = len(mol_out.integral_manager.orbital_coefficients) - mol_out.n_orbitals
        graph = [
            tuple([to_active[i] - ncore for i in edge if i in to_active])
            for edge in graph_full
        ]
        graph = [g for g in graph if len(g)]

        # aviso holds full CLPO indices: warn and only then move them to the active space, mapping them once
        __raise_aviso(aviso, to_active, ncore)
        aviso = __map_aviso(aviso, to_active, ncore)
    else:
        mo_matrix = orthogonalize(
            clpo_to_ao_full, mol.integral_manager.overlap_integrals
        )
        mol_out = deepcopy(mol)
        mol_out.integral_manager.orbital_coefficients = mo_matrix
        mol_out.integral_manager._orbital_type = "CLPO"
        graph = graph_full
    if return_aviso:
        return mol_out, graph, aviso
    return mol_out, graph


def generate_HAO_molecule_psi4(
    mol: QuantumChemistryBase,
    thres: Number = 1.0e-9,
    silent: bool = True,
    use_active: bool = True,
    ret2act: bool = False,
    use_mp2: bool = False,
    **kwargs,
) -> QuantumChemistryBase:
    """Generate a molecule with HAO (hybrid atomic orbital) orbitals via Psi4.

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
        If True, run MP2 to initialize JANPA, otherwise use the Hartree-Fock wavefunction.
    **kwargs
        ``wfn`` (custom Psi4 wavefunction) is currently not supported and is ignored.
        With ``use_active``, ``core`` and ``active`` (orbital indices in the full space) are passed
        on to the orbital transformation, see ``sunrise.utils.orbital_transformation.transform``.

    Returns
    -------
    mol : QuantumChemistryBase
        Molecule with HAO orbitals.
    to_active : dict of int to int, optional
        Only if ``ret2act`` and ``use_active`` are True.
    """

    # TODO: support a custom wfn (check kwargs.pop("wfn") first) and fix use_mp2 on psi4 molecules
    # (psi4mol is undefined there, use molecule=wfn.molecule())
    if not hasattr(mol, "ref_wfn"):  # its not a psi4 base
        psi4mol = psi4.geometry(mol.parameters.get_geometry_string(ghost="psi4"))
        psi4mol.set_multiplicity(mol.parameters.multiplicity)
        psi4mol.set_molecular_charge(mol.parameters.charge)
        psi4.activate(psi4mol)
        _, wfn = psi4.energy(name="scf", return_wfn=True, molecule=psi4mol)
    else:
        wfn = getattr(mol, "ref_wfn", None)

    if use_mp2:
        _, wfn = psi4.energy(
            name="mp2", return_wfn=True, molecule=psi4mol, guess_wfn=wfn
        )

    aho_to_ao_full, _, _, _ = _run_janpa_pipeline(
        wfn,
        thres=thres,
        silent=silent,
        avisar=True,
    )

    if use_active:
        nmol_full = deepcopy(mol)
        nmol_full.integral_manager.orbital_coefficients = aho_to_ao_full
        nmol_full.integral_manager._orbital_type = "HAO"
        mol_out, to_active = transform(
            original=mol, modified=nmol_full, orbital_type="HAO"
        )
        if ret2act:
            return mol_out, to_active
        return mol_out
    else:
        mo_matrix = orthogonalize(
            aho_to_ao_full, mol.integral_manager.overlap_integrals
        )
        mol_out = deepcopy(mol)
        mol_out.integral_manager.orbital_coefficients = mo_matrix
        mol_out.integral_manager._orbital_type = "HAO"
        return mol_out


def generate_CLPO_molecule_psi4(
    mol: QuantumChemistryBase,
    edges: list | None = None,
    thres: Number = 1.0e-12,
    silent: bool = True,
    use_active: bool = True,
    return_aviso: bool = False,
    use_mp2: bool = False,
    **kwargs,
) -> QuantumChemistryBase:
    """Generate a molecule with CLPO orbitals via Psi4.

    Same as ``generate_CLPO_molecule_edges`` but the graph edges are not returned.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Any sunrise molecule.
    edges : list of tuple of int, optional
        Bonds imposed on the CLPOs, given as 0-based atom indices, e.g. ``[(0, 3), (1, 2)]``
        for linear H4 (repeat a pair for a double bond). The bonds
        between the other atom pairs are found automatically from the Hartree-Fock density
        (MP2 with ``use_mp2``).
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
        If True, run MP2 to initialize JANPA, otherwise use the Hartree-Fock wavefunction.
    **kwargs
        ``wfn`` (custom Psi4 wavefunction) is currently not supported: it fails for Psi4 molecules and is ignored for others.
        With ``use_active``, ``core`` and ``active`` (orbital indices in the full space) are passed
        on to the orbital transformation, see ``sunrise.utils.orbital_transformation.transform``.

    Returns
    -------
    mol : QuantumChemistryBase
        Molecule with CLPO orbitals.
    aviso : list of tuple, optional
        Only if ``return_aviso`` is True. See Notes.

    Notes
    -----
    Each aviso entry is ``(mo_type, orb_idx, occupation)``. It lists the orbitals that do not have
    the expected double occupancy: ``mo_type`` 1 is a lone pair with occupation between 0.5
    and 1.5, 2 a bonding/antibonding pair ``(idx, idx + 1)`` whose summed occupation is below 1.7.
    With ``use_active``, ``orb_idx`` is the active-space index (None if the orbital ended up in
    the frozen core), otherwise the index in the full CLPO list.
    """
    if return_aviso:
        mol_out, _, aviso = generate_CLPO_molecule_edges_psi4(
            mol, edges, thres, silent, use_active, return_aviso, use_mp2, **kwargs
        )
        return mol_out, aviso
    else:
        mol_out, _ = generate_CLPO_molecule_edges_psi4(
            mol, edges, thres, silent, use_active, return_aviso, use_mp2, **kwargs
        )
        return mol_out


def __map_aviso(
    aviso: list[tuple[int, int, float]],
    to_active: dict[int, int] | None,
    ncore: int = 0,
) -> list[tuple[int, int, float]]:
    """Move the aviso orbital indices from the full CLPO list to the active space (None for those in the frozen core)."""
    return [
        (lo_type, to_active[idx] - ncore if idx in to_active else None, occ)
        for lo_type, idx, occ in aviso
    ]


def __raise_aviso(
    aviso: list[tuple[int, int, float]],
    to_active: dict[int, int] | None = None,
    ncore: int = 0,
):
    """Warn about the aviso orbitals, given by their index in the full CLPO list. With ``to_active`` the indices are shown in the active space and frozen-core orbitals are reported as such."""
    warnings = {
        LO_TYPE_LP: "Lone Pair {i} found with occupation close to 1 => {occ:.5f}, take care.",
        LO_TYPE_BD: "Bond pair orbitals [{i},{j}] population expected under expected 2e-, predicted: {occ:.5f}, take care with predicted edges.",
    }

    def name(idx):
        if to_active is None:
            return str(idx)
        if idx in to_active:
            return str(to_active[idx] - ncore)
        return f"frozen (CLPO {idx})"

    for lo_type, idx, occ in aviso:
        warn(warnings[lo_type].format(i=name(idx), j=name(idx + 1), occ=occ))
