"""Generate MEAO orbitals with the original MEAO code, through PySCF.

References
----------
L. Ding, E. Matito and C. Schilling, arXiv:2501.15699 (2025).
"""

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

try:
    from MEAO.meao import MEAO
except ImportError:
    MEAO = None


def generate_MEAO_molecule(
    mol: QuantumChemistryBase,
    minao: str = "minao",
    silent: bool = True,
    use_active: bool = True,
    **kwargs,
) -> QuantumChemistryBase:
    r"""Generate a molecule with MEAO (maximally entangled atomic orbital) orbitals.

    Interface with the original MEAO code (https://github.com/schilling-group/MEAO) through PySCF.
    The MEAOs are optimized in the space of the intrinsic atomic orbitals (IAOs) of the Hartree-Fock
    occupied orbitals. MEAO is not installed with sunrise, see the README.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Any sunrise molecule.
    minao : str or dict, optional
        Reference basis defining the IAOs, as in the examples of MEAO (the basis of the molecule can be
        larger, e.g. cc-pVDZ). Any basis accepted by PySCF, a name or a dict with one basis per element.
        The PySCF ``"minao"`` has no p shell on s-block atoms (e.g. Be in BeH2), ``"sto-3g"`` can be used
        there. Default is ``"minao"``.
    silent : bool
        If True, suppress the output of the MEAO optimization.
    use_active : bool
        If True, keep the frozen core and the active space of ``mol`` (the MEAOs are rotated into
        it). Otherwise all orbitals are replaced by the MEAOs.
    **kwargs
        ``pyscf_object`` (a PySCF mean-field object) whose occupied orbitals define the IAOs, otherwise
        a PySCF restricted Hartree-Fock is run.

    Returns
    -------
    mol : QuantumChemistryBase
        Molecule with MEAO orbitals. For basis sets larger than ``minao`` the MEAOs are completed by their
        orthonormal complement.

    Raises
    ------
    TequilaException
        If the MEAO package is not installed.

    See Also
    --------
    sunrise.CLPO.generate_HAO_molecule : Hybrid atomic orbitals from JANPA.

    Notes
    -----
    With a frozen core, which is the default of ``Molecule``, the core MEAOs are replaced by the canonical core
    orbitals of ``mol`` (kept frozen) and the valence MEAOs are made orthogonal to them. The valence MEAOs
    (e.g. the bonding hybrids) are then practically the same as those of the original MEAO code (which uses
    all orbitals), but not exactly, and neither are quantities that depend on them, such as orbital entropies
    and mutual information. To obtain exactly the orbitals of the original code, use ``frozen_core=False``
    when creating the molecule: the first orbitals of the result are then the MEAOs, in the order of the
    original, and the rest is the complement.

    Examples
    --------
    >>> mol = sun.Molecule(geometry="Be 0 0 0\nH 0 0 1.3\nH 0 0 -1.3", basis_set="sto-3g")
    >>> mol = sun.MEAO.generate_MEAO_molecule(mol, minao="sto-3g")
    """
    if MEAO is None:
        raise TequilaException(
            "MEAO not found, install it with "
            'pip install "MEAO @ git+https://github.com/schilling-group/MEAO"'
        )

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
