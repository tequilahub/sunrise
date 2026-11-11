"""Factory functions to create, convert, save and load sunrise molecules of all natures (qubit, hybrid, fermionic)."""

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
    from_json_data,
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
    r"""Create a sunrise molecule of the chosen nature.

    This is the main entry point (``sun.Molecule``). The molecule holds the integrals and orbitals and
    provides Hamiltonians, ansatz circuits and classical reference energies.

    Parameters
    ----------
    geometry : str, optional
        Molecular geometry as string, or filename of an xyz file (ending ``.xyz``).
    basis_set : str, optional
        Basis set name as understood by Psi4 and PySCF, e.g. ``"sto-3g"`` (minimal), ``"6-31g"``,
        ``"6-31g*"``, ``"cc-pvdz"``, ``"cc-pvtz"``, ``"aug-cc-pvdz"`` or ``"def2-svp"``. Without a
        basis set, MADNESS (basis-set-free PNOs) is used.
    nature : str {"qubit", "hybrid", "fermionic"}, optional
        Type of molecule. Default is ``"qubit"``. Short forms ``"q"``, ``"h"``, ``"f"`` work;
        ``"tequila"`` and ``"t"`` are aliases of ``"qubit"``.
    orbital_type : str, optional
        Orbitals to use, e.g. ``"hf"``, ``"native"``, ``"pno"``. Default depends on the backend.
    backend : str, optional
        Quantum chemistry backend (``"pyscf"``, ``"psi4"``, ``"madness"`` or ``"base"`` if the integrals are given).
        Default is the first installed one that fits.
    guess_wfn : optional
        Psi4 wavefunction (or filename of a stored one) to start the SCF cycle from.
    name : str, optional
        Name of the molecule. Default is deduced from the geometry (and vice versa the geometry is read from ``name.xyz``).
    *args, **kwargs
        Passed to the molecule class, e.g. ``charge``, ``frozen_core``, ``active_orbitals``, ``transformation``,
        ``n_pno``, or the integrals for the base backend. ``units`` defaults to ``"angstrom"``.
        For ``nature="hybrid"`` ``select`` sets the encoding of the orbitals.

    Returns
    -------
    QuantumChemistryBase
        The molecule (``FermionicBase`` for ``"fermionic"``, ``HyMolecule`` for ``"hybrid"``).

    Raises
    ------
    TequilaException
        If the nature is unknown.

    See Also
    --------
    MoleculeFromSunrise : Change the nature of an existing molecule.
    MoleculeFromPyscf : Create a molecule from a PySCF ``Mole``.
    MoleculeFromOpenFermion : Create a molecule from an openfermion molecule.
    load_molecule : Load a molecule saved with ``mol.save``.

    Examples
    --------
    >>> import sunrise as sun
    >>> mol = sun.Molecule(geometry="H 0 0 0\nH 0 0 0.75", basis_set="sto-3g", backend="pyscf")
    >>> fmol = sun.Molecule(geometry="H 0 0 0\nH 0 0 0.75", basis_set="sto-3g", nature="fermionic")
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
    """Create a molecule from the integrals of another sunrise molecule, in its current orbitals.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Sunrise molecule of any nature.
    transformation : str or callable, optional
        Fermion-to-qubit transformation. Default is the one of ``mol`` if the nature is the same.
    backend : str, optional
        Quantum chemistry backend. Default is ``"pyscf"`` if installed for the nature, otherwise ``"base"``.
    nature : str {"qubit", "hybrid", "fermionic"}, optional
        Type of the new molecule. Default is ``"qubit"``.
    *args, **kwargs
        Passed to the molecule class.
        ``select`` defaults to the ``select`` of ``mol`` for hybrid molecules.

    Returns
    -------
    QuantumChemistryBase

    Raises
    ------
    TequilaException
        If the nature is unknown.

    See Also
    --------
    Molecule : Create a molecule from a geometry and basis set.

    Examples
    --------
    >>> fmol = sun.MoleculeFromSunrise(mol, nature="fermionic")
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


def load_molecule(
    filename: str, transformation: str | typing.Callable | None = None
) -> QuantumChemistryBase:
    """Load a molecule saved with ``mol.save(filename)``, in the nature it was saved with.

    The molecule is rebuilt from the stored integrals and orbitals, nothing is recomputed.
    If the saved backend can not be restored (e.g. Psi4 without its wavefunction) the base backend is used with a warning.
    Also available as ``sun.open_molecule``.

    Parameters
    ----------
    filename : str
        HDF5 file written by ``mol.save``.
    transformation : str or callable, optional
        Fermion-to-qubit transformation. Default is the saved one.
        A transformation given as function can not be saved and has to be given here again.

    Returns
    -------
    QuantumChemistryBase

    See Also
    --------
    sunrise.molecules.qubit_base.qc_base.QuantumChemistryBase.save : Save a molecule to a file.

    Examples
    --------
    >>> mol.save("h2.h5")
    >>> mol = sun.load_molecule("h2.h5")
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
        backend_data = (
            from_json_data(json.loads(f["backend_data"][()]))
            if "backend_data" in f
            else None
        )
    integrals["two_body_integrals"] = NBodyTensor(
        elems=integrals["two_body_integrals"], ordering="chem"
    )

    backends = NATURE_BACKENDS[data["nature"]]
    backend = data["backend"]
    if backend not in backends or (
        backend not in ("base", "pyscf") and backend_data is None
    ):
        # the classical methods of psi4 and madness use their own data (psi4 wavefunction, madness pnoinfo),
        # not the integrals, so without it (e.g. files saved before) like in transform_orbitals the molecule can not be an instance of them
        warnings.warn(
            f"molecule was saved with the {backend} backend, it is loaded with the base backend",
            TequilaWarning,
        )
        backend = "base"

    kwargs = {**integrals, **data["settings"]}
    if backend != "base" and backend_data is not None:
        # e.g. the psi4 reference wavefunction or the madness pnoinfo, so the backend is rebuilt without recomputing it
        kwargs.update(backend_data)
    if transformation is not None:
        kwargs["transformation"] = transformation
    elif data["transformation"] is not None:
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
    # files saved before keep the convention of the backend
    mol.integral_manager._basis_convention = data.get(
        "basis_convention", mol.integral_manager._basis_convention
    )
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
    """Create a molecule from an openfermion molecule object.

    With the base backend the integrals stored in the openfermion molecule are used as they are (no frozen core).

    Parameters
    ----------
    molecule : openfermion.MolecularData
        The openfermion molecule.
    transformation : str or callable, optional
        Fermion-to-qubit transformation (``"jordan-wigner"``, ``"bravyi-kitaev"``, ``"bravyi-kitaev-tree"``
        or whatever OpenFermion supports).
    backend : str, optional
        Quantum chemistry backend. Default is ``"base"``.
    nature : str {"qubit", "hybrid", "fermionic"}, optional
        Type of the new molecule. Default is ``"qubit"``.
    *args, **kwargs
        Passed to ``from_openfermion`` of the molecule class.
        ``frozen_orbitals`` defaults to none. Integrals given here (``one_body_integrals``, ...)
        take precedence over those of ``molecule``.

    Returns
    -------
    QuantumChemistryBase

    Raises
    ------
    TequilaException
        If the nature is unknown.

    See Also
    --------
    Molecule : Create a molecule from a geometry and basis set.
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
    """Create a molecule from a PySCF ``Mole``, with its Hartree-Fock orbitals unless ``mo_coeff`` is given.

    Parameters
    ----------
    molecule : pyscf.gto.Mole
        The PySCF molecule.
    mo_coeff : numpy.ndarray, optional
        Orbital coefficients. Default is a restricted HF calculation (``orbital_type="hf"``, otherwise ``"unknown"``).
    transformation : str or callable, optional
        Fermion-to-qubit transformation.
    active_orbitals : list of int, optional
        Active orbitals (total indices).
    frozen_orbitals : list of int, optional
        Frozen orbitals (total indices).
    nature : str {"qubit", "hybrid", "fermionic"}, optional
        Type of the new molecule. Default is ``"qubit"``.
    *args, **kwargs
        Passed to the molecule class. ``mo_energy`` gives the orbital energies and ``basis_set`` the basis name
        if the PySCF molecule has none (e.g. from molden files).
        ``units`` are the units of the geometry, default "a" (angstrom).
        ``nuclear_repulsion`` is computed from the PySCF molecule unless given.

    Returns
    -------
    QuantumChemistryBase

    Raises
    ------
    TequilaException
        If the nature is unknown.

    See Also
    --------
    Molecule : Create a molecule from a geometry and basis set.
    MoleculeToPyscf : Convert a sunrise molecule to PySCF.
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
    kwargs.setdefault("basis_convention", "pyscf")

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
    """Return a PySCF ``Mole`` with the geometry, basis, charge and spin of a sunrise molecule.

    Parameters
    ----------
    molecule : QuantumChemistryBase
        The sunrise molecule.
    **kwargs
        ``point_group`` activates symmetry in that group (default no symmetry), ``cart`` sets cartesian basis functions.
        ``verbose`` sets the verbosity of the PySCF mean-field object.

    Returns
    -------
    pyscf.gto.Mole

    Examples
    --------
    >>> pyscf_mol = sun.MoleculeToPyscf(mol)
    """
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
