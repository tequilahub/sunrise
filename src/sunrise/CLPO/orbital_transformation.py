from __future__ import annotations

import contextlib
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from numbers import Number
from warnings import warn

import numpy as np
from janpa.clpo.lodesc import LO_TYPE_BD, LO_TYPE_LP, LO_TYPE_NB, LO_TYPE_RY
from janpa.clpo.lpo import CLPOOptions, create_clpos
from janpa.io.molden import MoldenFile
from janpa.npa.npa import run_npa
from pyscf import mp, scf
from pyscf.tools import molden as pyscf_molden

from sunrise import MoleculeFromPyscf, MoleculeToPyscf
from sunrise.molecules.fermionic_base import FermionicBase
from sunrise.molecules.hybrid_base import HybridBase
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase
from sunrise.molecules.utils_orbital_transformation import orthogonalize, transform


@dataclass
class __custom_mf_conteiner:
    mo_coeff: np.ndarray = None
    mo_occ: list[Number] = None
    mo_energy: list[Number] = None


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


def _run_janpa_pipeline(
    pfmol,
    mo_coeff,
    mo_occ,
    mo_energy,
    thres=1e-9,
    silent=True,
    custom_edges=None,
    avisar: bool = True,
):
    """Internal helper to run the JANPA pipeline entirely in memory/tempfiles."""
    with tempfile.TemporaryDirectory() as tmpdir, open(os.devnull, "w") as devnull:
        molden_in = os.path.join(tmpdir, "temp.molden")
        ctx = (
            contextlib.redirect_stdout(devnull) if silent else contextlib.nullcontext()
        )

        with ctx:
            pyscf_molden.from_mo(pfmol, molden_in, mo_coeff, occ=mo_occ, ene=mo_energy)
            molden_file = MoldenFile.load(molden_in)
            molden_file.coords_to_au()
            molden_file.to_unnormalized_primitive_coefs()

            npa_res = run_npa(molden_file)

            clpo_opts = CLPOOptions(hybr_opt_conv_thresh=thres, hybr_opt_max_iter=1000)
            if custom_edges is not None:
                clpo_opts.edges = str(custom_edges)

            clpo_res = create_clpos(
                npa_res.sds_nao, npa_res.nao, molden_file.centers, clpo_opts
            )
            graph, aviso = extract_clpo_graph(npa_res, clpo_res, avisar)

            # CRITICAL: These matrices have shape (n_orb, n_ao) where ROWS are orbitals.
            # Tequila expects (n_ao, n_orb) where COLUMNS are orbitals.
            # We transpose here so callers can assign directly to integral_manager.
            aho_to_ao = clpo_res.clpo.nao_to_hybrids.T @ npa_res.nao_to_ao
            # janpa keeps the MOLDEN AO order of the file, which differs from the pyscf one for d, f and g shells:
            # MOLDEN position k holds the pyscf AO ao_order[k]
            ao_order = pyscf_molden.order_ao_index(pfmol)
            aho_to_ao = aho_to_ao[:, np.argsort(ao_order)]
            clpo_to_ao = clpo_res.clpo.lo_to_hybrids @ aho_to_ao

            # Transpose: (n_orb, n_ao) -> (n_ao, n_orb) to match Tequila/PySCF convention
            aho_to_ao = aho_to_ao.T
            clpo_to_ao = clpo_to_ao.T

    return aho_to_ao, clpo_to_ao, graph, aviso


def generate_CLPO_molecule_edges(
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
    use_mp2 (bool): Whether to use mp2 mo_occ to initialize the janpa pipeline. It can be also provided a custom pyscf
                    with the keyword pyscf_object, or to provide custom mo_coeff, mo_occ, mo_energy. If not, pyscf
                    HF will be run.

    Returns:
    ----------
    Sunrise Molecule (same instance as input) with CLPO orbitals.
    Edges already in SPA format.
    If return_aviso a list with content (mo_type, orb_idx, mo_occ) for those orbitals with dangerous occupation.
        mo_type 1 for lone pairs not doubly occupied and mo_type 2 for bonding/antibonding pairs (idx, idx+1)
        whose summed occupation is not close two 2e-. With use_active, orb_idx is the active space index
        (None if the orbital ended up in the frozen core), otherwise the index in the full CLPO list.
    """

    pfmol = MoleculeToPyscf(mol)
    if "pyscf_object" in kwargs:
        mf = kwargs.pop("pyscf_object")
    elif "mo_coeff" in kwargs and "mo_occ" in kwargs and "mo_energy" in kwargs:
        mf = __custom_mf_conteiner(
            kwargs["mo_coeff"], kwargs["mo_occ"], kwargs["mo_energy"]
        )
        kwargs.pop("mo_coeff")
        kwargs.pop("mo_occ")
        kwargs.pop("mo_energy")
    else:
        mf = scf.RHF(pfmol).run()
        if use_mp2:
            fr = [2 for _ in range(mol.parameters.get_number_of_core_electrons() // 2)]
            rdm1 = mp.MP2(mf).run().make_rdm1()
            mf = __custom_mf_conteiner(
                mf.mo_coeff, fr + np.diag(rdm1).tolist(), mf.mo_energy
            )
    # the CLPOs come in the AOs of pfmol. Other backends (e.g. psi4) order and orient their AOs differently, then the
    # molecule is built again on the pyscf AOs, keeping its active space
    s = mol.integral_manager.overlap_integrals
    if s.shape != (pfmol.nao, pfmol.nao) or not np.allclose(
        s, pfmol.intor("int1e_ovlp")
    ):
        mol = MoleculeFromPyscf(
            pfmol,
            mo_coeff=mf.mo_coeff,
            transformation=mol.transformation,
            active_orbitals=[o.idx_total for o in mol.integral_manager.active_orbitals],
            nature="fermionic"
            if isinstance(mol, FermionicBase)
            else "hybrid"
            if isinstance(mol, HybridBase)
            else "qubit",
        )
    # edges are bonds imposed on the CLPOs between 0-based atoms, e.g. [(1, 2), (1, 2)] for a double bond between atoms 1
    # and 2; the bonds between the other atom pairs are found automatically. Atoms do not depend on the active space,
    # so they are passed as given. Cast to native int to avoid np.int64 string representation issues
    custom_edges = None
    if edges is not None:
        custom_edges = [tuple([int(e) for e in edge]) for edge in edges]

    # clpo_to_ao is now (n_ao, n_clpo) after the transpose fix in _run_janpa_pipeline
    _, clpo_to_ao_full, graph_full, aviso = _run_janpa_pipeline(
        pfmol,
        mf.mo_coeff,
        mf.mo_occ,
        mf.mo_energy,
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


def generate_HAO_molecule(
    mol: QuantumChemistryBase,
    thres: Number = 1.0e-9,
    silent: bool = True,
    use_active: bool = True,
    ret2act: bool = False,
    return_aviso: bool = False,
    use_mp2: bool = False,
    **kwargs,
) -> QuantumChemistryBase:
    r"""
    Interface sunrise molecules with janpa for HAO molecule generation.

    Parameters:
    ----------
    mol (QuantumChemistryBase): any instance of sunrise molecule
    thres (Number): Orbital janpa orbital threshold, where orbitals are rotated to accomodate the electrond density
    silent (bool): Whether to print all janpa output
    use_active (bool): Whether to transform the janpa result orbitals to keep the active space defined by the input molecule
    ret2act: (bool): Wether to return (if use active was true) the dictionary defined by:
                    {idx_before_active: idx_after_active for orbital in active_space}
    return_aviso: (bool): Wether to return the data of the dangerours orbitals (same info showed on the warnings)
    use_mp2 (bool): Whether to use mp2 mo_occ to initialize the janpa pipeline. It can be also provided a custom pyscf
                    with the keyword pyscf_object, or to provide custom mo_coeff, mo_occ, mo_energy. If not, pyscf
                    HF will be run.

    Returns:
    ----------
    Sunrise Molecule (same instance as input) with HAO orbitals.
    If return_aviso a list with content (mo_type, orb_idx, mo_occ) for those orbitals with dangerous occupation.
        mo_type 1 for lone pairs not doubly occupied and mo_type 2 for bonding/antibonding pairs (idx, idx+1)
        whose summed occupation is not close two 2e-. orb_idx is the index in the full CLPO list, not in the HAOs.
    """

    pfmol = MoleculeToPyscf(mol)
    if "pyscf_object" in kwargs:
        mf = kwargs.pop("pyscf_object")
    elif "mo_coeff" in kwargs and "mo_occ" in kwargs and "mo_energy" in kwargs:
        mf = __custom_mf_conteiner(
            mo_coeff=kwargs["mo_coeff"],
            mo_occ=kwargs["mo_occ"],
            mo_energy=kwargs["mo_energy"],
        )
        kwargs.pop("mo_coeff")
        kwargs.pop("mo_occ")
        kwargs.pop("mo_energy")
    else:
        mf = scf.RHF(pfmol).run()
        if use_mp2:
            fr = [2 for _ in range(mol.parameters.get_number_of_core_electrons() // 2)]
            rdm1 = mp.MP2(mf).run().make_rdm1()
            mf = __custom_mf_conteiner(
                mo_coeff=mf.mo_coeff,
                mo_occ=fr + np.diag(rdm1).tolist(),
                mo_energy=mf.mo_energy,
            )
    # the HAOs come in the AOs of pfmol. Other backends (e.g. psi4) order and orient their AOs differently, then the
    # molecule is built again on the pyscf AOs, keeping its active space
    s = mol.integral_manager.overlap_integrals
    if s.shape != (pfmol.nao, pfmol.nao) or not np.allclose(
        s, pfmol.intor("int1e_ovlp")
    ):
        mol = MoleculeFromPyscf(
            pfmol,
            mo_coeff=mf.mo_coeff,
            transformation=mol.transformation,
            active_orbitals=[o.idx_total for o in mol.integral_manager.active_orbitals],
            nature="fermionic"
            if isinstance(mol, FermionicBase)
            else "hybrid"
            if isinstance(mol, HybridBase)
            else "qubit",
        )

    # aho_to_ao_full is now (n_ao, n_hyb) after the transpose fix in _run_janpa_pipeline
    # aviso indexes the CLPOs, not the HAOs returned here, so it is warned and returned in full CLPO indices
    aho_to_ao_full, _, _, aviso = _run_janpa_pipeline(
        pfmol,
        mf.mo_coeff,
        mf.mo_occ,
        mf.mo_energy,
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
            if return_aviso:
                return mol_out, to_active, aviso
            return mol_out, to_active
        if return_aviso:
            return mol_out, aviso
        return mol_out
    else:
        mo_matrix = orthogonalize(
            aho_to_ao_full, mol.integral_manager.overlap_integrals
        )
        mol_out = deepcopy(mol)
        mol_out.integral_manager.orbital_coefficients = mo_matrix
        mol_out.integral_manager._orbital_type = "HAO"
        if return_aviso:
            return mol_out, aviso
        return mol_out


def generate_CLPO_molecule(
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
    use_mp2 (bool): Whether to use mp2 mo_occ to initialize the janpa pipeline. It can be also provided a custom pyscf
                    with the keyword pyscf_object, or to provide custom mo_coeff, mo_occ, mo_energy. If not, pyscf
                    HF will be run.

    Returns:
    ----------
    Sunrise Molecule (same instance as input) with CLPO orbitals.
    If return_aviso a list with content (mo_type, orb_idx, mo_occ) for those orbitals with dangerous occupation.
        mo_type 1 for lone pairs not doubly occupied and mo_type 2 for bonding/antibonding pairs (idx, idx+1)
        whose summed occupation is not close two 2e-. With use_active, orb_idx is the active space index
        (None if the orbital ended up in the frozen core), otherwise the index in the full CLPO list.
    """
    if return_aviso:
        mol_out, _, aviso = generate_CLPO_molecule_edges(
            mol, edges, thres, silent, use_active, return_aviso, use_mp2, **kwargs
        )
        return mol_out, aviso
    else:
        mol_out, _ = generate_CLPO_molecule_edges(
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
