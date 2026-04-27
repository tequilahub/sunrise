from pyscf import gto, scf
from pyscf.tools import cubegen
import tequila as tq
from tequila.quantumchemistry.qc_base import QuantumChemistryBase
from tequila import TequilaException
from sunrise.miscellaneous.bar import giuseppe_bar
import sys

def plot_MO(molecule:QuantumChemistryBase=None,filename:str=None,orbital:list=None,print_orbital:bool=True,density:bool=False,mep:bool=False):
    """
    Small function to save the MOs into Cube files
    Parameters
    ----------
    filename : Cube file will be saved as name+orb_index
    orbital: index of the orbitals to save
    molecule: molecule to plot the orbitals from
    print_orbital: whether to print the MOs
    density: whether to print the electron density
    mep: whether to plot the molecular electrostatic potential
    """
    if molecule is None:
        raise TequilaException("No Molecule to save orbitals from")
    if filename is None:
        filename = molecule.parameters.name +'-'+ molecule.integral_manager._basis_name+'-'+molecule.integral_manager._orbital_type
    if orbital is None:
        orbital = [i.idx_total for i in molecule.integral_manager.orbitals]
    pmol = gto.Mole()
    pmol.build(atom=molecule.parameters.geometry, basis=molecule.parameters.basis_set, charge=molecule.parameters.charge)
    if density or mep:
        mf = scf.RHF(pmol).run()
        mf.mo_coeff=molecule.integral_manager.orbital_coefficients
    if print_orbital:
        for i,idx in enumerate(orbital):
            giuseppe_bar(step=i,total_steps=len(orbital))
            cubegen.orbital(pmol,  str(idx)+"_"+filename+"_MO.cube", molecule.integral_manager.orbital_coefficients[:, idx])
        giuseppe_bar(step=i+1,total_steps=len(orbital))
        sys.stdout.write('\n')
        sys.stdout.flush()
    if density:
        cubegen.density(pmol, filename + '_density.cube', mf.make_rdm1())
    if mep:
        cubegen.mep(pmol, filename + '_mep.cube', mf.make_rdm1())
