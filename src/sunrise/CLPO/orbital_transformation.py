from __future__ import annotations
from tequila.quantumchemistry.pyscf_interface import QuantumChemistryPySCF
import os
import numpy
from pyscf import scf, mp
from pyscf.tools import molden
from copy import deepcopy
import subprocess
from tequila.quantumchemistry.qc_base import QuantumChemistryBase
import numpy
from copy import deepcopy
from typing import Tuple
from numbers import Number
from .binary_interface import *
from sunrise import from_tequila
from tequila import TequilaException
from sunrise.molecules.utils_orbital_transformation import transform, orthogonalize

def __get_MP2_occ(mol:QuantumChemistryBase) -> Tuple[list[Number], list[Number]]:
    ''''
    Small helper function, given a tequila molecule returns the MP2 orbital occupation and orbital energy
    '''
    fr = [2 for _ in range(mol.parameters.get_number_of_core_electrons()//2)]
    molx = QuantumChemistryPySCF.from_tequila(mol)
    hf = molx._get_hf()
    rdm1 = mp.MP2(hf).run().make_rdm1()
    return fr + numpy.diag(rdm1).tolist(), hf.mo_energy

def generate_molden(mol:QuantumChemistryBase, filename:str = None, output_dir:str = None, mo_occ:list = None, mo_energy:list = None, use_mp2:bool = False, option1:bool = True, use_active:bool = True):
    '''
    Interface with pyscf.tools molden file generation

    :param mol: Any kind of tequila/sunrise Molecule
    :param filename: The moldenfile will be saved as filename.molden. If None, the molecule.parameters.name is employed
    :param output_dir: default None = working file.
    :param mo_occ: Molecular Orbital electronic occupation. If None, hf or mp2 are used depending on use_mp2
    :param mo_energy: Molecular Orbital energy. If None, hf or mp2 are used depending on use_mp2
    :param use_mp2: Whether to us mp2 or hf if no mo_occ and mo_energy are provided
    :param option1: Whether to use the first or second molden generation alternatives propsed by pyscf. See 
                    https://github.com/pyscf/pyscf/blob/master/examples/tools/02-molden.py for more info
    :param use_active: Whether to include only the active orbitals.
    '''

    assert (mo_occ == None) and (mo_energy == None)
    size_basis = len(mol.integral_manager._orbital_coefficients)
    active = mol.integral_manager.active_space.active_orbitals
    pfmol = from_tequila(mol)
    if output_dir is None:
        output_dir = os.getcwd()
    if mo_occ is None:
        if use_mp2:
            mo_occ, mo_energy = __get_MP2_occ(mol)
        else:
            mf = scf.RHF(pfmol).run()
            mo_occ = mf.mo_occ
            mo_energy = mf.mo_energy
    else:
        assert len(mo_occ) == len(mo_energy)
        if (use_active and len(mo_occ) == len(active) or len(mo_occ) == len(size_basis)) or (not use_active and len(mo_occ) == len(size_basis)):
            pass
        else:
            raise TequilaException(f"{len(mo_occ)} Molecular Orbital Occupation but it doesn't fit not the total space size ({size_basis}) nor the active space size ({len(active)}). Use_active={use_active}") 
    
    if filename is None: filename=mol.parameters.name


    mo_coeff = mol.integral_manager.orbital_coefficients.copy()
    if use_active:
        if len(mo_occ) == size_basis:
            mo_occ = [mo_occ[i] for i in active]
            mo_energy = [mo_energy[i] for i in active]
        mo_coeff = mo_coeff[:, active]

    if option1:
        ### OPTION 1
        with open(f'{output_dir}/{filename}.molden', 'w') as f1:
            molden.header(pfmol, f1) 
            molden.orbital_coeff(pfmol, f1, mo_coeff, ene=mo_energy, occ=mo_occ)
    else:
        ### OPTION 2
        try:
            molden.from_mo(pfmol, f'{output_dir}/{filename}.molden',mo_coeff, ene=mo_energy, occ=mo_occ)
        except RuntimeError:
            print('    Found l=5 in basis.')
            molden.from_mo(pfmol, f'{output_dir}/{filename}.molden', mo_coeff, ene=mo_energy, occ=mo_occ, ignore_h=True)

def generate_CLPO_molecule_edges(mol:QuantumChemistryBase, edges:list[tuple[int]] = None, output_dir:str = None, thres:Number = 1.e-12, silent:bool = True, use_active:bool = True, rm_files:bool = True, **kwargs) -> Tuple[QuantumChemistryBase,list]:
    '''
    Temporal function for generating a molecule with CLPO orbitals (10.1002/qua.25798) until integrated in Sunrise molecules
    
    :param mol: Any kind of Tequila/Sunrise Molecules.
    :param edges: optional edges to built the molecule. Default (None) will try to replicate the reference wvf (see generate_molden).
                    not needed to pass all edges, not provided will be get from the reference wvf
    :param output_dir: default None = working file.
    :param thres: -HybrOptOccConvThresh from Janpa. Default 1.e-9.
    :param use_active: Whether to respect input molecule frozen/active space, letting frozen as HF
    :param kwargs: keywords accepted by 'generate_molden', see above.

    Return modified Molecule and SPA edges
    '''
    if 'filename' in kwargs:
        filename = kwargs['filename']
        kwargs.pop('filename')
    else: filename = mol.parameters.name
    if output_dir is None:
        output_dir = os.getcwd()
    generate_molden(mol = mol, filename = filename, output_dir = output_dir, use_active=False, **kwargs) #TODO: Janpa CLPO is bug for active space only, working on 
    call_molden2aim(moldenfile = filename+'.molden', output_dir = output_dir)
    call_molden2molden(command = f'-NormalizeBF -cart2pure  -i {filename}.molden -o {filename}.molden', silent = silent, output_dir = output_dir)
    c = f'-i {filename}.molden -CLPO_Molden_File {filename}_CLPO.molden -HybrOptOccConvThresh {thres} '
    if edges is not None:
        if use_active:
            _, to_active = generate_HAO_molecule(deepcopy(mol), output_dir = output_dir, thres = thres, silent = True, use_active = True, rm_files = False, to_active = True) 
            d = {i.idx:i.idx_total for i in mol.integral_manager.active_orbitals} # We need the correspondence between active space indices and complete basis
            to_active = {v:k for k,v in to_active.items()} # to_active keeps track of reordering on active space (frozen orbitals are kept at the begining) 
            edges = [tuple([to_active[d[e]] for e in edge]) for edge in edges] # Therefore the edges are transformed by: edges_in_active -> edges_in_complete_basis -> edges_in_complete_basis_non_active_space_order
            # to_active is taken between from HAO bcs it may differ from HAO to CLPO to_active, but we want the pairing on the HAO basis  
        else:
            edges = [tuple([e for e in edge]) for edge in edges]
        c += f' -edges {edges}'
    call_janpa(command = c, silent = silent)
    mo_matrix = read_molden_mo_matrix(f"{output_dir}/{filename}_CLPO.molden")
    if not use_active:
        mo_matrix = orthogonalize(mo_matrix, mol.integral_manager.overlap_integrals)
    if rm_files:
        subprocess.call(f'rm {output_dir}/m2a.ini', shell=True)
        subprocess.call(f'rm {output_dir}/{filename}.molden', shell=True) 
        subprocess.call(f'rm {output_dir}/{filename}_new.molden', shell=True)
        subprocess.call(f'rm {output_dir}/{filename}_CLPO.molden', shell=True)
    nmol = deepcopy(mol)
    nmol.integral_manager.orbital_coefficients = mo_matrix
    if use_active:
        mol, to_active = transform(original = mol, modified = nmol, orbital_type = 'CLPO')
    else:
        mol = nmol
        mol.integral_manager._orbital_type = 'CLPO'
    graph = extract_clpo_graph(f"{output_dir}/graph")
    if use_active:
        ncore = len(mol.integral_manager.orbital_coefficients) - mol.n_orbitals
        graph = [tuple([to_active[i] - ncore for i in edge if i in to_active.keys()]) for edge in graph]
        graph = [g for g in graph if len(g)]
    if rm_files:
        subprocess.call(f'rm {output_dir}/graph', shell=True)
    return mol,graph

def generate_HAO_molecule(mol:QuantumChemistryBase, output_dir:str = None, thres:Number = 1.e-9, silent:bool = True, use_active:bool = True, rm_files:bool = True,**kwargs) -> QuantumChemistryBase:
    '''
    Temporal function for generating a molecule with Hybrid Atomic Orbitals via janpa (10.1002/qua.25798) until integrated in Sunrise molecules
    
    :param mol: Any kind of Tequila/Sunrise Molecules.
    :param output_dir: default None = working file.
    :param thres: -HybrOptOccConvThresh from Janpa. Default 1.e-9.
    :param use_active: Whether to respect input molecule frozen/active space, letting frozen as HF
    :param kwargs: keywords accepted by 'generate_molden', see above.

    Return modified Molecule and SPA edges
    '''
    if 'filename' in kwargs:
        filename = kwargs['filename']
        kwargs.pop('filename')
    else: filename = mol.parameters.name
    if 'to_active' in kwargs:  # Internal use, thats why not mentioned on funtion description
        ret2act = kwargs['to_active']
        kwargs.pop('to_active')
    else: ret2act = False

    if output_dir is None:
        output_dir = os.getcwd()

    generate_molden(mol = mol, filename = filename, output_dir = output_dir, use_active = False, **kwargs)
    call_molden2aim(moldenfile = filename+'.molden', output_dir = output_dir)
    call_molden2molden(command = f'-NormalizeBF -cart2pure  -i {filename}.molden -o {filename}.molden', silent = silent, output_dir = output_dir)
    call_janpa(command=f'-i {filename}.molden -AHO_Molden_File {filename}_HAO.molden -HybrOptOccConvThresh {thres}', silent = silent, output_dir = output_dir)
    mo_matrix = read_molden_mo_matrix(f"{output_dir}/{filename}_HAO.molden")
    if not use_active:
        mo_matrix = orthogonalize(mo_matrix, mol.integral_manager.overlap_integrals)
    if rm_files:
        subprocess.call(f'rm {output_dir}/m2a.ini', shell=True)
        subprocess.call(f'rm {output_dir}/{filename}.molden', shell=True)
        subprocess.call(f'rm {output_dir}/{filename}_new.molden', shell=True)
        subprocess.call(f'rm {output_dir}/{filename}_HAO.molden', shell=True)
        subprocess.call(f'rm {output_dir}/graph', shell=True)
    nmol = deepcopy(mol)
    nmol.integral_manager.orbital_coefficients = mo_matrix
    if use_active:
        mol, to_active = transform(original = mol, modified = nmol, orbital_type = 'HAO')
    else:
        mol = nmol
        mol.integral_manager._orbital_type = "HAO"
    if ret2act:
        return mol, to_active
    return mol

def generate_CLPO_molecule(mol:QuantumChemistryBase, edges:list[tuple[int]] = None, output_dir:str = None, thres:Number = 1.e-12, silent:bool = True, use_active:bool = True, rm_files:bool = True, **kwargs) -> QuantumChemistryBase:
    '''
    Temporal function for generating a molecule with CLPO orbitals (10.1002/qua.25798) until integrated in Sunrise molecules
    
    :param mol: Any kind of Tequila/Sunrise Molecules.
    :param edges: optional edges to built the molecule. Default (None) will try to replicate the reference wvf (see generate_molden).
                    not needed to pass all edges, not provided will be get from the reference wvf
    :param output_dir: default None = working file.
    :param thres: -HybrOptOccConvThresh from Janpa. Default 1.e-9.
    :param use_active: Whether to respect input molecule frozen/active space, letting frozen as HF
    :param kwargs: keywords accepted by 'generate_molden', see above.

    Return modified Molecule and SPA edges
    '''
    mol, edges = generate_CLPO_molecule_edges(mol, edges, output_dir, thres, silent, use_active, rm_files, **kwargs)
    return mol