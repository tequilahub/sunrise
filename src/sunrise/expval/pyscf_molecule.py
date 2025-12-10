from tequila.quantumchemistry.qc_base import QuantumChemistryBase
from tequila.quantumchemistry.chemistry_tools import ParametersQC,NBodyTensor,OrbitalData
import pyscf
from pyscf.gto import Mole
from numpy import ndarray
from typing import Callable
from typing import Union

def from_tequila(molecule:QuantumChemistryBase,**kwargs)->pyscf.gto.Mole: #TODO: Change to something like MoleFromTequila
    geometry = molecule.parameters.get_geometry()
    pyscf_geomstring = ""
    for atom in geometry:
        pyscf_geomstring += "{} {} {} {};".format(atom[0], atom[1][0], atom[1][1], atom[1][2])

    if "point_group" in kwargs:
        point_group = kwargs["point_group"]
    else:
        point_group = None

    mol = pyscf.gto.M(
        atom=pyscf_geomstring,
        basis=molecule.parameters.basis_set,
        charge=molecule.parameters.charge,
        verbose=0,
        spin=0,
    )
    if point_group is not None:
        if point_group.lower() != "c1":
            mol.symmetry = True
            mol.symmetry_subgroup = point_group
        else:
            mol.symmetry = False
    else:
        mol.symmetry = True
    mol.symmetry = False

    mol.build(parse_arg=False)
    mol.ao2mo(mo_coeffs=molecule.integral_manager.orbital_coefficients)
    mol.build()
    return mol
    # solve restricted HF
    mf = pyscf.scf.RHF(mol)#.run()
    mf.verbose = False
    if "verbose" in kwargs:
        mf.verbose = kwargs["verbose"]
    mf.mo_coeff = molecule.integral_manager.orbital_coefficients
    # mf.kernel()
    
    return mf


def MoleculeFromPyscf(molecule:Mole,mo_coeff:Union[ndarray,None]=None,transformation:Union[str,Callable,None]=None,active_orbitals:list=None,frozen_orbitals:list=None,*args,**kwargs)->QuantumChemistryBase:
    
    geo = ''
    for i in  range(len(molecule.atom_coords())):
            c = molecule.atom_coord(i,unit="UA")
            at = ''.join(x for x in molecule.atom_symbol(i) if x.isalpha())
            geo += f'{at} {c[0]} {c[1]} {c[2]}\n'
    if len(molecule.basis):
        basis = molecule.basis
    elif 'basis_set' in kwargs: #when reading molden files, the basis name is lost
        basis = kwargs['basis_set']
        kwargs.pop('basis_set')
    else:
        basis = 'custom'
    parameters = ParametersQC(
            basis_set=basis,
            geometry=geo,
            multiplicity=molecule.multiplicity,
            charge=molecule.charge,
        )
    if "orbital_type" not in kwargs:
        if mo_coeff is None:
            kwargs["orbital_type"] = 'hf'
        else: kwargs["orbital_type"] = 'unknown'
    if 'mo_energy' in kwargs:
        orbital_energies = kwargs['mo_energy']
        kwargs.pop('mo_energy')
    else: orbital_energies = [*range(molecule.nao)]
    if mo_coeff is None:
        mf =pyscf.scf.RHF(molecule)
        mf.kernel()
        mo_coeff = mf.mo_coeff
        orbital_energies = mf.mo_energy #would be weird to provide orbital energies but not mo_coeff
    if molecule.irrep_name is not None:
        irreps = pyscf.symm.label_orb_symm(molecule, molecule.irrep_name, molecule.symm_orb, mo_coeff).tolist()
    else: irreps = ['a1' for _ in range(len(mo_coeff))]
    

    orbitals = [OrbitalData(idx_total=idx, irrep=irr, energy=energy) for idx, (irr, energy) in enumerate(zip(irreps, orbital_energies))]
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

    tqmol = QuantumChemistryBase(parameters=parameters,transformation=transformation,active_orbitals=active_orbitals,frozen_orbitals=frozen_orbitals,point_group=molecule.symmetry_subgroup,*args,**kwargs)
    tqmol.integral_manager.orbital_coefficients = mo_coeff
    return tqmol