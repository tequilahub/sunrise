from __future__ import annotations

from numbers import Number

from sunrise.molecules.qubit_base import INSTALLED_QCHEMISTRY_BACKENDS
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase

try:
    from .pyscf_janpa_interace import (
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
    use_mp2 (bool): Whether to use mp2 or HF to initialize the janpa pipeline.
    backend (str): backend for possible HF/MP2 calculations, and MOLDEN file generation. Currently supported pyscf and psi4.
                    If not specified, it will be guessed from the provided molecules, kwargs or from the installed backends.
    Keywords:
        if backend is pyscf:
            - pyscf_object: object to get the pyscf_object.mo_coeff, pyscf_object.mo_occ and pyscf_object.mo_energy
            - mo_coeff + mo_occ + mo_energy: they can be also provided separated, but the 3 together.
        if backend is psi4:
            - wfn: Custom wfn object from the resulting energy methd.
        In any case, the user is responsible from the result. Janpa documentation only considers HF or MP2

    Returns:
    ----------
    Sunrise Molecule (same instance as input) with CLPO orbitals.
    Edges already in SPA format.
    If return_aviso a list with content (mo_type, orb_idx, mo_occ) for those orbitals with dangerous occupation.
        mo_type 1 for lone pairs not doubly occupied and mo_type 2 for bonding/antibonding pairs (idx, idx+1)
        whose summed occupation is not close two 2e-. With use_active, orb_idx is the active space index
        (None if the orbital ended up in the frozen core), otherwise the index in the full CLPO list.
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
    use_mp2 (bool): Whether to use mp2 or HF to initialize the janpa pipeline.
    backend (str): backend for possible HF/MP2 calculations, and MOLDEN file generation. Currently supported pyscf and psi4.
                    If not specified, it will be guessed from the provided molecules, kwargs or from the installed backends.
    Keywords:
        if backend is pyscf:
            - pyscf_object: object to get the pyscf_object.mo_coeff, pyscf_object.mo_occ and pyscf_object.mo_energy
            - mo_coeff + mo_occ + mo_energy: they can be also provided separated, but the 3 together.
        if backend is psi4:
            - wfn: Custom wfn object from the resulting energy methd.
        In any case, the user is responsible from the result. Janpa documentation only considers HF or MP2

    Returns:
    ----------
    Sunrise Molecule (same instance as input) with HAO orbitals.
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
    use_mp2 (bool): Whether to use mp2 or HF to initialize the janpa pipeline.
    backend (str): backend for possible HF/MP2 calculations, and MOLDEN file generation. Currently supported pyscf and psi4.
                    If not specified, it will be guessed from the provided molecules, kwargs or from the installed backends.
    Keywords:
        if backend is pyscf:
            - pyscf_object: object to get the pyscf_object.mo_coeff, pyscf_object.mo_occ and pyscf_object.mo_energy
            - mo_coeff + mo_occ + mo_energy: they can be also provided separated, but the 3 together.
        if backend is psi4:
            - wfn: Custom wfn object from the resulting energy methd.
        In any case, the user is responsible from the result. Janpa documentation only considers HF or MP2

    Returns:
    ----------
    Sunrise Molecule (same instance as input) with CLPO orbitals.
    If return_aviso a list with content (mo_type, orb_idx, mo_occ) for those orbitals with dangerous occupation.
        mo_type 1 for lone pairs not doubly occupied and mo_type 2 for bonding/antibonding pairs (idx, idx+1)
        whose summed occupation is not close two 2e-. With use_active, orb_idx is the active space index
        (None if the orbital ended up in the frozen core), otherwise the index in the full CLPO list.
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
    """
    Helper funtion to select a chemistry backend in funtion of the molecule or kwargs
    """
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
