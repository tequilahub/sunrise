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
    """Extracts the CLPO graph (edges) directly from janpa-py results."""
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
    """
    perm[k] is the psi4 AO index of the k-th MOLDEN basis function. MOLDEN orders the p shells as x, y, z while the
    pure psi4 shells are p0, p+1, p-1 = z, x, y. The other shells are ordered alike (0, +1, -1, +2, -2, ...).
    """
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
    """
    Internal helper to run the JANPA pipeline on a psi4 Wavefunction. The returned matrices have one row per psi4 AO and
    one column per orbital, as the orbital coefficients of the integral manager.
    """
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
    """
    Interface sunrise molecules with janpa for CLPO molecule generation via psi4.

    Parameters:
    ----------
    mol (QuantumChemistryBase): any instance of sunrise molecule
    edges (List[tuple[int]]): Optional. Custom molecule conectivity defined by the
                            atom indices. I.e. for lineral H4. [(0,3),(1,2)] would
                                return 2nd graph orbtials
    thres (Number): Orbital janpa orbital threshold, where orbitals are rotated to accomodate the electrond density
    silent (bool): Whether to print all janpa output
    use_active (bool): Whether to transform the janpa result orbitals to keep the active space defined by the input molecule
    return_aviso: (bool): Wether to return the data of the dangerours orbitals (same info showed on the warnings)
    use_mp2 (bool): Whether to use mp2 or HF wfn to initialize the janpa pipeline. It can be also provided a custom psi4
                    wfn with the wfn keyword.

    Returns:
    ----------
    Sunrise Molecule (same instance as input) with CLPO orbitals.
    Edges already in SPA format.
    If return_aviso a list with content (mo_type, orb_idx, mo_occ) for those orbitals with dangerous occupation.
        mo_type 1 for lone pairs not doubly occupied and mo_type 2 for bonding/antibonding pairs (idx, idx+1)
        whose summed occupation is not close two 2e-. With use_active, orb_idx is the active space index
        (None if the orbital ended up in the frozen core), otherwise the index in the full CLPO list.
    """

    if not hasattr(mol, "ref_wfn"):  # its not a psi4 base
        psi4mol = psi4.geometry(mol.parameters.get_geometry_string(ghost="psi4"))
        psi4mol.set_multiplicity(mol.parameters.multiplicity)
        psi4mol.set_molecular_charge(mol.parameters.charge)
        psi4.activate(psi4mol)
        _, wfn = psi4.energy(name="scf", return_wfn=True, molecule=psi4mol)
    elif hasattr(kwargs, "wfn"):
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
    r"""
    Interface sunrise molecules with janpa for HAO molecule generation via psi4.

    Parameters:
    ----------
    mol (QuantumChemistryBase): any instance of sunrise molecule
    thres (Number): Orbital janpa orbital threshold, where orbitals are rotated to accomodate the electrond density
    silent (bool): Whether to print all janpa output
    use_active (bool): Whether to transform the janpa result orbitals to keep the active space defined by the input molecule
    ret2act: (bool): Wether to return (if use active was true) the dictionary defined by:
                    {idx_before_active: idx_after_active for orbital in active_space}
    use_mp2 (bool): Whether to use mp2 or HF wfn to initialize the janpa pipeline. It can be also provided a custom psi4
                    wfn with the wfn keyword.

    Returns:
    ----------
    Sunrise Molecule (same instance as input) with HAO orbitals.
    If return_aviso a list with content (mo_type, orb_idx, mo_occ) for those orbitals with dangerous occupation.
        mo_type 1 for lone pairs not doubly occupied and mo_type 2 for bonding/antibonding pairs (idx, idx+1)
        whose summed occupation is not close two 2e-. orb_idx is the index in the full CLPO list, not in the HAOs.
    """

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
    """
    Interface sunrise molecules with janpa for CLPO molecule generation.

    Parameters:
    ----------
    mol (QuantumChemistryBase): any instance of sunrise molecule
    edges (List[tuple[int]]): Optional. Custom molecule conectivity defined by the
                            atom indices. I.e. for lineral H4. [(0,3),(1,2)] would
                                return 2nd graph orbtials
    thres (Number): Orbital janpa orbital threshold, where orbitals are rotated to accomodate the electrond density
    silent (bool): Whether to print all janpa output
    use_active (bool): Whether to transform the janpa result orbitals to keep the active space defined by the input molecule
    return_aviso: (bool): Wether to return the data of the dangerours orbitals (same info showed on the warnings)
    use_mp2 (bool): Whether to use mp2 or HF wfn to initialize the janpa pipeline. It can be also provided a custom psi4
                    wfn with the wfn keyword.

    Returns:
    ----------
    Sunrise Molecule (same instance as input) with CLPO orbitals.
    If return_aviso a list with content (mo_type, orb_idx, mo_occ) for those orbitals with dangerous occupation.
        mo_type 1 for lone pairs not doubly occupied and mo_type 2 for bonding/antibonding pairs (idx, idx+1)
        whose summed occupation is not close two 2e-. With use_active, orb_idx is the active space index
        (None if the orbital ended up in the frozen core), otherwise the index in the full CLPO list.
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
    """Moves the aviso orbital indices from the full CLPO list to the active space, None for those in the frozen core."""
    return [
        (lo_type, to_active[idx] - ncore if idx in to_active else None, occ)
        for lo_type, idx, occ in aviso
    ]


def __raise_aviso(
    aviso: list[tuple[int, int, float]],
    to_active: dict[int, int] | None = None,
    ncore: int = 0,
):
    """Warns about the aviso orbitals, given by their index in the full CLPO list. With to_active the indices are shown in
    the active space, and orbitals that ended up in the frozen core are reported as such."""
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
