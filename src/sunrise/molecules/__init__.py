from __future__ import annotations

import json
import typing
import warnings
from typing import Callable

import h5py
import pyscf
from numpy import ndarray
from pyscf.gto import Mole
from tequila import TequilaException, TequilaWarning

from sunrise.molecules import fermionic_base, hybrid_base, qubit_base
from sunrise.molecules.qubit_base import Molecule as QubitMolecule
from sunrise.molecules.qubit_base.chemistry_tools import (
    NBodyTensor,
    OrbitalData,
    ParametersQC,
)
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase

from .fermionic_base import FerMolecule
from .hybrid_base import HyMolecule

# installed quantum chemistry backends of each molecule nature, under the names Molecule(nature=...) accepts
NATURE_BACKENDS = {
    **dict.fromkeys(
        ("qubit", "q", "tequila", "t"), qubit_base.INSTALLED_QCHEMISTRY_BACKENDS
    ),
    **dict.fromkeys(("hybrid", "h"), hybrid_base.INSTALLED_QCHEMISTRY_BACKENDS),
    **dict.fromkeys(("fermionic", "f"), fermionic_base.INSTALLED_QCHEMISTRY_BACKENDS),
}


def Molecule(
    geometry: str | None = None,
    basis_set: str | None = None,
    nature: str = "qubit",
    orbital_type: str | None = None,
    backend: str | None = None,
    guess_wfn=None,
    name: str | None = None,
    *args,
    **kwargs,
) -> QuantumChemistryBase:
    """

    Parameters
    ----------
    geometry
        molecular geometry as string or as filename (needs to be in xyz format with .xyz ending)
    basis_set
        quantum chemistry basis set (sto-3g, cc-pvdz, etc)
    nature
        Molecule type (qubit, hybrid, fermionic); 'tequila' is kept as an alias for 'qubit'
    backend
        quantum chemistry backend (psi4, pyscf)
    guess_wfn
        pass down a psi4 guess wavefunction to start the scf cycle from
        can also be a filename leading to a stored wavefunction
    name
        name of the molecule, if not given it's auto-deduced from the geometry
        can also be done vice versa (i.e. geometry is then auto-deduced to name.xyz)
    args
    kwargs

    Returns
    -------
        Molecule object
    """
    if "units" not in kwargs:
        kwargs["units"] = "angstrom"
    if nature.lower() in ("qubit", "q", "tequila", "t"):
        return QubitMolecule(
            *args,
            geometry=geometry,
            basis_set=basis_set,
            orbital_type=orbital_type,
            backend=backend,
            guess_wfn=guess_wfn,
            name=name,
            **kwargs,
        )
    elif nature.lower() == "hybrid" or nature.lower() == "h":
        if "select" in kwargs:
            select = kwargs["select"]
            kwargs.pop("select")
        else:
            select = {}
        return HyMolecule(
            *args,
            geometry=geometry,
            select=select,
            basis_set=basis_set,
            orbital_type=orbital_type,
            backend=backend,
            guess_wfn=guess_wfn,
            name=name,
            **kwargs,
        )
    elif nature.lower() == "fermionic" or nature.lower() == "f":
        return FerMolecule(
            *args,
            geometry=geometry,
            basis_set=basis_set,
            orbital_type=orbital_type,
            backend=backend,
            guess_wfn=guess_wfn,
            name=name,
            **kwargs,
        )
    else:
        raise TequilaException(
            f"Molecule Nature not identified: {nature}. Only qubit, hybrid and fermionic allowed."
        )


def MoleculeFromSunrise(
    mol, transformation=None, backend=None, *args, nature: str = "qubit", **kwargs
):
    """
    Initialize a sunrise Molecule from the integrals of another sunrise molecule in its current orbitals
    Parameters
    ----------
    mol
        The sunrise molecule, of any nature
    transformation
        The Fermion to Qubit Transformation, if not given the one of mol is used when mol has the same nature
    backend
        The quantum chemistry backend, pyscf if installed for the nature, otherwise base
    nature
        Molecule type of the result (qubit, hybrid, fermionic)
    Returns
    -------
        The sunrise molecule
    """
    if nature.lower() not in NATURE_BACKENDS:
        raise TequilaException(
            f"Molecule Nature not identified: {nature}. Only qubit, hybrid and fermionic allowed."
        )
    backends = NATURE_BACKENDS[nature.lower()]
    c, h, g = mol.get_integrals()
    parameters = mol.parameters
    if backend is None:
        if "pyscf" in backends:
            backend = "pyscf"
        else:
            backend = "base"
    if transformation is None and type(mol) in backends.values():
        # the transformation of mol only fits a molecule of the same nature
        transformation = mol.transformation
        if hasattr(mol, "select") and "select" not in kwargs:
            # hybrid molecules set their encoding through select
            kwargs["select"] = mol.select
    return backends[backend.lower()](
        *args,
        parameters=parameters,
        transformation=transformation,
        n_electrons=mol.n_electrons,
        one_body_integrals=h,
        two_body_integrals=g,
        nuclear_repulsion=c,
        **kwargs,
    )


def load_molecule(filename: str) -> QuantumChemistryBase:
    """
    Load a sunrise molecule saved with mol.save(filename), in the nature it was saved with
    It is rebuilt from the stored integrals and orbitals, nothing is recomputed
    Parameters
    ----------
    filename
        The HDF5 file written by mol.save
    Returns
    -------
        The sunrise molecule
    """
    with h5py.File(filename, "r") as f:
        data = json.loads(f["molecule"][()])
        integrals = {
            k: f[k][()]
            for k in (
                "one_body_integrals",
                "two_body_integrals",
                "overlap_integrals",
                "orbital_coefficients",
            )
        }
    integrals["two_body_integrals"] = NBodyTensor(
        elems=integrals["two_body_integrals"], ordering="chem"
    )

    backends = NATURE_BACKENDS[data["nature"]]
    backend = data["backend"]
    if backend not in ("base", "pyscf") or backend not in backends:
        # psi4 and madness would compute the integrals again
        fallback = "pyscf" if "pyscf" in backends else "base"
        warnings.warn(
            f"molecule was saved with the {backend} backend, it is loaded with the {fallback} backend",
            TequilaWarning,
        )
        backend = fallback

    kwargs = {**integrals, **data["settings"]}
    if data["transformation"] is not None:
        kwargs["transformation"] = data["transformation"]["name"]
        kwargs.update(
            {
                "transformation__" + k: v
                for k, v in data["transformation"]["arguments"].items()
            }
        )
    mol = backends[backend](
        parameters=ParametersQC(**data["parameters"]),
        nuclear_repulsion=data["constant_term"],
        active_orbitals=data["active_orbitals"],
        reference_orbitals=data["reference_orbitals"],
        frozen_orbitals=[],  # the active space is given, no core is frozen on top of it
        point_group=data["point_group"],
        **kwargs,
    )

    # the orbital data is set afterwards, on init the orbital type "native" would transform the orbitals again
    mol.integral_manager._basis_name = data["basis_name"]
    mol.integral_manager._orbital_type = data["orbital_type"]
    for o, stored in zip(mol.integral_manager.orbitals, data["orbitals"]):
        # idx is the index in the active space, already set by it
        o.irrep, o.idx_irrep, o.idx_total, o.energy, o.occ = (
            stored[k] for k in ("irrep", "idx_irrep", "idx_total", "energy", "occ")
        )
        o.pair = None if stored["pair"] is None else tuple(stored["pair"])
    if data["irreps"] is not None:
        mol.irreps = data["irreps"]
    if data["pyscf_molecule"] is not None:
        mol.pyscf_molecule = pyscf.gto.loads(data["pyscf_molecule"])
    return mol


def MoleculeFromOpenFermion(
    molecule,
    transformation: str | typing.Callable | None = None,
    backend: str | None = None,
    *args,
    nature: str = "qubit",
    **kwargs,
) -> QuantumChemistryBase:
    """
    Initialize a sunrise Molecule directly from an openfermion molecule object
    Parameters
    ----------
    molecule
        The openfermion molecule
    transformation
        The Fermion to Qubit Transformation (jordan-wigner, bravyi-kitaev, bravyi-kitaev-tree and whatever OpenFermion supports)
    backend
        The quantum chemistry backend, can be None in this case
    nature
        Molecule type (qubit, hybrid, fermionic)
    Returns
    -------
        The sunrise molecule
    """
    if nature.lower() not in NATURE_BACKENDS:
        raise TequilaException(
            f"Molecule Nature not identified: {nature}. Only qubit, hybrid and fermionic allowed."
        )
    if backend is None:
        backend = "base"
    use_integrals = (
        backend == "base"
        and molecule.one_body_integrals is not None
        and "one_body_integrals" not in kwargs
    )
    if use_integrals:
        # the base backend computes no integrals: use the ones of the openfermion molecule as they are,
        # without freezing a core on top of them (the other backends compute them from geometry and basis)
        kwargs["one_body_integrals"] = molecule.one_body_integrals
        kwargs["two_body_integrals"] = NBodyTensor(
            elems=molecule.two_body_integrals, ordering="openfermion"
        )
        kwargs["nuclear_repulsion"] = molecule.nuclear_repulsion
        if "frozen_orbitals" not in kwargs:
            kwargs["frozen_orbitals"] = []
    result = NATURE_BACKENDS[nature.lower()][backend].from_openfermion(
        *args, molecule=molecule, transformation=transformation, **kwargs
    )
    if use_integrals:
        result.parameters.frozen_core = False
    return result


def MoleculeFromPyscf(
    molecule: Mole,
    mo_coeff: ndarray | None = None,
    transformation: str | Callable | None = None,
    active_orbitals: list | None = None,
    frozen_orbitals: list | None = None,
    *args,
    nature: str = "qubit",
    **kwargs,
) -> QuantumChemistryBase:
    """
    Initialize a sunrise Molecule from a pyscf molecule, with its HF orbitals if mo_coeff is not given
    nature selects the Molecule type (qubit, hybrid, fermionic)
    """
    if nature.lower() not in NATURE_BACKENDS:
        raise TequilaException(
            f"Molecule Nature not identified: {nature}. Only qubit, hybrid and fermionic allowed."
        )
    geo = ""
    for i in range(len(molecule.atom_coords())):
        c = molecule.atom_coord(i, unit="UA")
        at = "".join(x for x in molecule.atom_symbol(i) if x.isalpha())
        geo += f"{at} {c[0]} {c[1]} {c[2]}\n"
    if len(molecule.basis):
        basis = molecule.basis
    elif "basis_set" in kwargs:  # when reading molden files, the basis name is lost
        basis = kwargs["basis_set"]
        kwargs.pop("basis_set")
    else:
        basis = "custom"
    parameters = ParametersQC(
        basis_set=basis,
        geometry=geo,
        multiplicity=molecule.multiplicity,
        charge=molecule.charge,
    )
    if "orbital_type" not in kwargs:
        if mo_coeff is None:
            kwargs["orbital_type"] = "hf"
        else:
            kwargs["orbital_type"] = "unknown"
    if "mo_energy" in kwargs:
        orbital_energies = kwargs["mo_energy"]
        kwargs.pop("mo_energy")
    else:
        orbital_energies = [None for _ in range(molecule.nao)]
    if mo_coeff is None:
        mf = pyscf.scf.RHF(molecule)
        mf.kernel()
        mo_coeff = mf.mo_coeff
        orbital_energies = (
            mf.mo_energy
        )  # would be weird to provide orbital energies but not mo_coeff
    irreps = ["a1" for _ in range(len(mo_coeff))]
    if molecule.irrep_name is not None:
        try:
            irreps = pyscf.symm.label_orb_symm(
                molecule, molecule.irrep_name, molecule.symm_orb, mo_coeff
            ).tolist()
        except ValueError:  # orbitals that break the symmetry of the molecule (e.g. localized) have no irrep
            pass

    orbitals = [
        OrbitalData(idx_total=idx, irrep=irr, energy=energy)
        for idx, (irr, energy) in enumerate(zip(irreps, orbital_energies))
    ]
    for irr in {o.irrep for o in orbitals}:
        for i, o in enumerate([o for o in orbitals if o.irrep == irr]):
            o.idx_irrep = i

    h_ao = molecule.intor("int1e_kin") + molecule.intor("int1e_nuc")
    g_ao = molecule.intor("int2e", aosym="s1")
    S = molecule.intor_symmetric("int1e_ovlp")
    g_ao = NBodyTensor(elems=g_ao, ordering="mulliken")
    kwargs["overlap_integrals"] = S
    kwargs["two_body_integrals"] = g_ao
    kwargs["one_body_integrals"] = h_ao
    kwargs["orbital_coefficients"] = mo_coeff

    if "nuclear_repulsion" not in kwargs:
        kwargs["nuclear_repulsion"] = molecule.energy_nuc()
    if "units" in kwargs:
        units = kwargs["units"]
        kwargs.pop("units")
    else:
        units = "a"
    tqmol = NATURE_BACKENDS[nature.lower()]["base"](
        *args,
        parameters=parameters,
        transformation=transformation,
        active_orbitals=active_orbitals,
        frozen_orbitals=frozen_orbitals,
        point_group=molecule.symmetry_subgroup,
        units=units,
        **kwargs,
    )
    tqmol.integral_manager.orbital_coefficients = mo_coeff
    # setting the coefficients resets the orbital data, put back the irreps and energies of mo_coeff
    for o, data in zip(tqmol.integral_manager.orbitals, orbitals):
        o.irrep, o.idx_irrep, o.energy = data.irrep, data.idx_irrep, data.energy
    return tqmol


def MoleculeToPyscf(molecule: QuantumChemistryBase, **kwargs) -> pyscf.gto.Mole:
    geometry = molecule.parameters.get_geometry()
    pyscf_geomstring = ""
    for atom in geometry:
        pyscf_geomstring += f"{atom[0]} {atom[1][0]} {atom[1][1]} {atom[1][2]};"

    point_group = kwargs.get("point_group", None)

    mol = pyscf.gto.M(
        atom=pyscf_geomstring,
        basis=molecule.parameters.basis_set,
        charge=molecule.parameters.charge,
        verbose=0,
        spin=molecule.parameters.multiplicity - 1,
    )
    mol.symmetry = False
    if point_group is not None and point_group.lower() != "c1":
        mol.symmetry = True
        mol.symmetry_subgroup = point_group
    if "cart" in kwargs:
        mol.cart = kwargs["cart"]
    mol.build(parse_arg=False)
    return mol
    # solve restricted HF
    mf = pyscf.scf.RHF(mol)  # .run()
    mf.verbose = False
    if "verbose" in kwargs:
        mf.verbose = kwargs["verbose"]
    mf.mo_coeff = molecule.integral_manager.orbital_coefficients
    # mf.kernel()

    return mf
