from __future__ import annotations

import contextlib
import io
from copy import deepcopy

import numpy as np
from pyscf import scf
from pyscf.lo import iao, orth
from scipy.linalg import null_space
from tequila import TequilaException

from sunrise import MoleculeFromPyscf, MoleculeToPyscf
from sunrise.molecules.fermionic_base import FermionicBase
from sunrise.molecules.hybrid_base import HybridBase
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase
from sunrise.utils.orbital_transformation import orthogonalize, transform


def generate_MEAO_molecule(
    mol: QuantumChemistryBase,
    minao: str = "minao",
    silent: bool = True,
    use_active: bool = True,
    **kwargs,
) -> QuantumChemistryBase:
    """
    Interface sunrise molecules with the original MEAO code (https://github.com/schilling-group/MEAO) for the
    generation of Maximally Entangled Atomic Orbitals (MEAO, L. Ding, E. Matito, C. Schilling, arXiv:2501.15699) via pyscf.
    MEAO is not installed with sunrise (see the README). TODO: once https://github.com/schilling-group/MEAO/pull/1
    is merged, the install hint in the error below and in the README should point to schilling-group/MEAO instead of
    the branch of that PR.

    Parameters:
    ----------
    mol (QuantumChemistryBase): any instance of sunrise molecule
    minao (str): minimal reference basis of the intrinsic atomic orbitals (IAO) the MEAOs are optimized in. pyscf's
                 minao has no p shell on s-block atoms (e.g. Be in BeH2), "sto-3g" can be used there.
                 For basis sets larger than minao the MEAOs are completed by their orthonormal complement
    silent (bool): Whether to hide the output of the MEAO optimization
    use_active (bool): Whether to transform the MEAOs to keep the active space defined by the input molecule
    Keywords:
        - pyscf_object: pyscf mean field object whose occupied orbitals define the IAOs. If not, pyscf HF will be run.

    Returns:
    ----------
    Sunrise Molecule (same instance as input) with MEAO orbitals.
    """
    try:
        from MEAO.meao import MEAO
    except ImportError as e:
        raise TequilaException(
            "MEAO not found, install it with "
            'pip install "MEAO @ git+https://github.com/davibinco/MEAO_davide.git@pip-installable" '
            "(https://github.com/schilling-group/MEAO/pull/1)"
        ) from e

    pfmol = MoleculeToPyscf(mol)
    if "pyscf_object" in kwargs:
        mf = kwargs.pop("pyscf_object")
    else:
        mf = scf.RHF(pfmol).run()
    # the MEAOs come in the AOs of pfmol. Other backends (e.g. psi4) order and orient their AOs differently, then the
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

    # orthonormal IAOs and the number of them on each atom, as in the examples of MEAO
    s = pfmol.intor("int1e_ovlp")
    c = iao.iao(pfmol, mf.mo_coeff[:, mf.mo_occ > 0], minao=minao)
    norbs_in_atoms = [
        a[3] - a[2] for a in iao.reference_mol(pfmol, minao).aoslice_by_atom()
    ]
    meao = MEAO(pfmol, mf, c @ orth.lowdin(c.T @ s @ c), norbs_in_atoms)
    with (
        contextlib.redirect_stdout(io.StringIO())
        if silent
        else contextlib.nullcontext()
    ):
        meao.meao()
    meao_to_ao_full = np.hstack(
        [meao.mo_meao, orthogonalize(null_space(meao.mo_meao.T @ s), s)]
    )

    if use_active:
        nmol_full = deepcopy(mol)
        nmol_full.integral_manager.orbital_coefficients = meao_to_ao_full
        nmol_full.integral_manager._orbital_type = "MEAO"
        mol_out, _ = transform(original=mol, modified=nmol_full, orbital_type="MEAO")
        return mol_out
    else:
        mo_matrix = orthogonalize(meao_to_ao_full, s)
        mol_out = deepcopy(mol)
        mol_out.integral_manager.orbital_coefficients = mo_matrix
        mol_out.integral_manager._orbital_type = "MEAO"
        return mol_out
