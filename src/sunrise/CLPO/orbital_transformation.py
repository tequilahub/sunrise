from __future__ import annotations
import os
import contextlib
import tempfile
import numpy as np
from copy import deepcopy
from warnings import warn
from typing import Tuple, List, Optional, Any
from numbers import Number

from pyscf import scf, mp
from pyscf.tools import molden as pyscf_molden

from tequila.quantumchemistry.pyscf_interface import QuantumChemistryPySCF
from tequila.quantumchemistry.qc_base import QuantumChemistryBase
from tequila import TequilaException

from sunrise import from_tequila
from sunrise.molecules.utils_orbital_transformation import transform, orthogonalize

from janpa.io.molden import MoldenFile
from janpa.npa.npa import run_npa
from janpa.clpo.lpo import create_clpos, CLPOOptions
from janpa.clpo.lodesc import LO_TYPE_RY, LO_TYPE_LP, LO_TYPE_BD, LO_TYPE_NB

def __get_MP2_occ(mol: QuantumChemistryBase) -> Tuple[list, list]:
    fr = [2 for _ in range(mol.parameters.get_number_of_core_electrons()//2)]
    molx = QuantumChemistryPySCF.from_tequila(mol)
    hf = molx._get_hf()
    rdm1 = mp.MP2(hf).run().make_rdm1()
    return fr + np.diag(rdm1).tolist(), hf.mo_energy

def extract_clpo_graph(npa_res, clpo_res) -> List[Tuple[int, ...]]:
    """Extracts the CLPO graph (edges) directly from janpa-py results."""
    Q = clpo_res.clpo.nao_to_hybrids @ clpo_res.clpo.lo_to_hybrids.T
    sds_clpo = Q.T @ npa_res.sds_nao @ Q
    occupancies = np.diag(sds_clpo)
    lo_types = clpo_res.clpo.lo_types
    
    nodes = []
    n_lo = len(occupancies)
    i = 0
    while i < n_lo:
        lo_type = lo_types[i]
        occ = round(occupancies[i], 5)
        
        if lo_type == LO_TYPE_LP:
            if 0.5 < occ < 1.5:
                warn(f'Lone Pair {i} found with occupation close to 1 => {occ:.5f}, take care.')
            if occ < 1.0:
                i += 1
                continue
            nodes.append((i,))
            i += 1
            
        elif lo_type == LO_TYPE_RY:
            i += 1
            
        elif lo_type == LO_TYPE_BD:
            if i + 1 >= n_lo or lo_types[i+1] != LO_TYPE_NB:
                raise ValueError(f"BD orbital {i} without following NB orbital")
            bd_occ = occ
            nb_occ = round(occupancies[i+1], 5)
            if bd_occ + nb_occ < 1.7:
                warn(f'Bond pair orbitals [{i},{i+1}] population expected under expected 2e-, predicted: {bd_occ + nb_occ:.5f}, take care with predicted edges.')
            nodes.append((i, i + 1))
            i += 2
            
        else:
            i += 1
            
    return nodes

def _run_janpa_pipeline(pfmol, mo_coeff, mo_occ, mo_energy, thres=1e-9, silent=True, custom_edges=None):
    """Internal helper to run the JANPA pipeline entirely in memory/tempfiles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        molden_in = os.path.join(tmpdir, "temp.molden")
        devnull = open(os.devnull, 'w')
        ctx = contextlib.redirect_stdout(devnull) if silent else contextlib.nullcontext()
        
        try:
            with ctx:
                pyscf_molden.from_mo(pfmol, molden_in, mo_coeff, occ=mo_occ, ene=mo_energy)
                molden_file = MoldenFile.load(molden_in)
                molden_file.coords_to_au()
                molden_file.to_unnormalized_primitive_coefs()
                
                npa_res = run_npa(molden_file)
                
                clpo_opts = CLPOOptions(hybr_opt_conv_thresh=thres, hybr_opt_max_iter=1000)
                if custom_edges is not None:
                    clpo_opts.edges = str(custom_edges) 
                
                clpo_res = create_clpos(npa_res.sds_nao, npa_res.nao, molden_file.centers, clpo_opts)
                graph = extract_clpo_graph(npa_res, clpo_res)
                
                # CRITICAL: These matrices have shape (n_orb, n_ao) where ROWS are orbitals.
                # Tequila expects (n_ao, n_orb) where COLUMNS are orbitals.
                # We transpose here so callers can assign directly to integral_manager.
                aho_to_ao = clpo_res.clpo.nao_to_hybrids.T @ npa_res.nao_to_ao
                clpo_to_ao = clpo_res.clpo.lo_to_hybrids @ aho_to_ao
                
                # Transpose: (n_orb, n_ao) -> (n_ao, n_orb) to match Tequila/PySCF convention
                aho_to_ao = aho_to_ao.T
                clpo_to_ao = clpo_to_ao.T
                
        finally:
            if silent:
                devnull.close()
                
    return aho_to_ao, clpo_to_ao, graph

def generate_molden(mol: QuantumChemistryBase, filename: str = None, output_dir: str = None, 
                    mo_occ: list = None, mo_energy: list = None, use_mp2: bool = False, 
                    option1: bool = True, use_active: bool = True):
    pass

def generate_CLPO_molecule_edges(mol: QuantumChemistryBase, edges: list = None, output_dir: str = None, 
                                 thres: Number = 1.e-12, silent: bool = True, use_active: bool = True, 
                                 rm_files: bool = True, **kwargs) -> Tuple[QuantumChemistryBase, list]:
    if output_dir is None:
        output_dir = os.getcwd()
        
    pfmol = from_tequila(mol)
    mf_full = scf.RHF(pfmol).run()
    
    custom_edges = None
    if edges is not None:
        if use_active:
            _, to_active = generate_HAO_molecule(deepcopy(mol), output_dir=output_dir, thres=thres, silent=True, use_active=True, rm_files=False, to_active=True)
            d = {i.idx: i.idx_total for i in mol.integral_manager.active_orbitals}
            to_active_inv = {v: k for k, v in to_active.items()}
            # Cast to native int to avoid np.int64 string representation issues
            custom_edges = [tuple([int(to_active_inv[d[e]]) for e in edge]) for edge in edges]
        else:
            # Cast to native int to avoid np.int64 string representation issues
            custom_edges = [tuple([int(e) for e in edge]) for edge in edges]

    # clpo_to_ao is now (n_ao, n_clpo) after the transpose fix in _run_janpa_pipeline
    _, clpo_to_ao_full, graph_full = _run_janpa_pipeline(
        pfmol, mf_full.mo_coeff, mf_full.mo_occ, mf_full.mo_energy,
        thres=thres, silent=silent, custom_edges=custom_edges
    )
    
    if use_active:
        nmol_full = deepcopy(mol)
        nmol_full.integral_manager.orbital_coefficients = clpo_to_ao_full
        nmol_full.integral_manager._orbital_type = 'CLPO'
        
        mol_out, to_active = transform(original=mol, modified=nmol_full, orbital_type='CLPO')
        
        ncore = len(mol_out.integral_manager.orbital_coefficients) - mol_out.n_orbitals
        graph = [tuple([to_active[i] - ncore for i in edge if i in to_active.keys()]) for edge in graph_full]
        graph = [g for g in graph if len(g)]
    else:
        mo_matrix = orthogonalize(clpo_to_ao_full, mol.integral_manager.overlap_integrals)
        mol_out = deepcopy(mol)
        mol_out.integral_manager.orbital_coefficients = mo_matrix
        mol_out.integral_manager._orbital_type = 'CLPO'
        graph = graph_full

    return mol_out, graph

def generate_HAO_molecule(mol: QuantumChemistryBase, output_dir: str = None, thres: Number = 1.e-9, 
                          silent: bool = True, use_active: bool = True, rm_files: bool = True, **kwargs) -> QuantumChemistryBase:
    if output_dir is None:
        output_dir = os.getcwd()
        
    pfmol = from_tequila(mol)
    mf_full = scf.RHF(pfmol).run()
    
    ret2act = kwargs.get('to_active', False)
    
    # aho_to_ao_full is now (n_ao, n_hyb) after the transpose fix in _run_janpa_pipeline
    aho_to_ao_full, _, _ = _run_janpa_pipeline(
        pfmol, mf_full.mo_coeff, mf_full.mo_occ, mf_full.mo_energy,
        thres=thres, silent=silent
    )
    
    if use_active:
        nmol_full = deepcopy(mol)
        nmol_full.integral_manager.orbital_coefficients = aho_to_ao_full
        nmol_full.integral_manager._orbital_type = "HAO"
        
        mol_out, to_active = transform(original=mol, modified=nmol_full, orbital_type='HAO')
        if ret2act:
            return mol_out, to_active
        return mol_out
    else:
        mo_matrix = orthogonalize(aho_to_ao_full, mol.integral_manager.overlap_integrals)
        mol_out = deepcopy(mol)
        mol_out.integral_manager.orbital_coefficients = mo_matrix
        mol_out.integral_manager._orbital_type = "HAO"
        return mol_out

def generate_CLPO_molecule(mol: QuantumChemistryBase, edges: list = None, output_dir: str = None, 
                           thres: Number = 1.e-12, silent: bool = True, use_active: bool = True, 
                           rm_files: bool = True, **kwargs) -> QuantumChemistryBase:
    mol_out, _ = generate_CLPO_molecule_edges(mol, edges, output_dir, thres, silent, use_active, rm_files, **kwargs)
    return mol_out