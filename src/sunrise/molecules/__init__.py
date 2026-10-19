import typing
import pyscf
from pyscf.gto import Mole
from numpy import ndarray
from typing import Callable
from typing import Union
from tequila import TequilaException
from sunrise.molecules import qubit_base, hybrid_base, fermionic_base
from sunrise.molecules.qubit_base import Molecule as QubitMolecule
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase
from sunrise.molecules.qubit_base.chemistry_tools import (
    ParametersQC,
    NBodyTensor,
    OrbitalData,
)
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
    geometry: str = None,
    basis_set: str = None,
    nature: str = "qubit",
    orbital_type: str = None,
    backend: str = None,
    guess_wfn=None,
    name: str = None,
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
            geometry=geometry,
            basis_set=basis_set,
            orbital_type=orbital_type,
            backend=backend,
            guess_wfn=guess_wfn,
            name=name,
            *args,
            **kwargs,
        )
    elif nature.lower() == "hybrid" or nature.lower() == "h":
        if "select" in kwargs:
            select = kwargs["select"]
            kwargs.pop("select")
        else:
            select = {}
        return HyMolecule(
            geometry=geometry,
            select=select,
            basis_set=basis_set,
            orbital_type=orbital_type,
            backend=backend,
            guess_wfn=guess_wfn,
            name=name,
            *args,
            **kwargs,
        )
    elif nature.lower() == "fermionic" or nature.lower() == "f":
        return FerMolecule(
            geometry=geometry,
            basis_set=basis_set,
            orbital_type=orbital_type,
            backend=backend,
            guess_wfn=guess_wfn,
            name=name,
            *args,
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
        parameters=parameters,
        transformation=transformation,
        n_electrons=mol.n_electrons,
        one_body_integrals=h,
        two_body_integrals=g,
        nuclear_repulsion=c,
        *args,
        **kwargs,
    )


def MoleculeFromOpenFermion(
    molecule,
    transformation: typing.Union[str, typing.Callable] = None,
    backend: str = None,
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
        molecule=molecule, transformation=transformation, *args, **kwargs
    )
    if use_integrals:
        result.parameters.frozen_core = False
    return result


def MoleculeFromPyscf(
    molecule: Mole,
    mo_coeff: Union[ndarray, None] = None,
    transformation: Union[str, Callable, None] = None,
    active_orbitals: list = None,
    frozen_orbitals: list = None,
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
        parameters=parameters,
        transformation=transformation,
        active_orbitals=active_orbitals,
        frozen_orbitals=frozen_orbitals,
        point_group=molecule.symmetry_subgroup,
        units=units,
        *args,
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
        pyscf_geomstring += "{} {} {} {};".format(
            atom[0], atom[1][0], atom[1][1], atom[1][2]
        )

    if "point_group" in kwargs:
        point_group = kwargs["point_group"]
    else:
        point_group = None

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
