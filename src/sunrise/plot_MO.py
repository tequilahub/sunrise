from pyscf import gto, scf
from pyscf.tools import cubegen
from tequila.quantumchemistry.qc_base import QuantumChemistryBase
from sunrise.miscellaneous.bar import giuseppe_bar
import sys

def plot_MO(molecule:QuantumChemistryBase, filename:str = None, orbital:list[int] = None, use_active:bool = True, print_orbital:bool = True, density:bool = False, mep:bool = False):
    """
    Small function to save the MOs into Cube files
    Parameters
    ----------
    filename : Cube file will be saved as name+orb_index
    orbital: index of the orbitals to save
    use_active: Wether to plot only the active orbitals, if orbital list passed and true, its assumed that the indices are w.r.t. active orbitals (mol.integral_manager.active_orbitals[x].idx instead of ...[x].idx_total).
    molecule: molecule to plot the orbitals from
    print_orbital: whether to print the MOs
    density: whether to print the electron density
    mep: whether to plot the molecular electrostatic potential
    """
    
    if filename is None:
        filename = molecule.parameters.name + '-' + molecule.integral_manager._basis_name + '-' + molecule.integral_manager._orbital_type
    if orbital is None and use_active:
        orbital = [i.idx_total for i in molecule.integral_manager.active_orbitals]
        label = [i.idx for i in molecule.integral_manager.active_orbitals]
    elif orbital is None and not use_active:
        orbital = [i.idx_total for i in molecule.integral_manager.orbitals]
        label = orbital
    elif orbital is not None and use_active:
        d = {i.idx:i.idx_total for i in molecule.integral_manager.orbitals}
        label = orbital.copy()
        orbital =  [d[i] for i in orbital]
    else:
        label = orbital

    pmol = gto.Mole()  
    pmol.build(atom = molecule.parameters.geometry, basis = molecule.parameters.basis_set, charge = molecule.parameters.charge)
    if density or mep:
        mf = scf.RHF(pmol).run()
        mf.mo_coeff = molecule.integral_manager.orbital_coefficients
    if print_orbital:
        for i,idx in enumerate(orbital):
            giuseppe_bar(step = i, total_steps = len(orbital))
            cubegen.orbital(pmol, str(label[i])+ "_" + filename + "_MO.cube", molecule.integral_manager.orbital_coefficients[:, idx])
        giuseppe_bar(step = i + 1, total_steps = len(orbital))
        sys.stdout.write('\n')
        sys.stdout.flush()
    if density:
        cubegen.density(pmol, filename + '_density.cube', mf.make_rdm1())
    if mep:
        cubegen.mep(pmol, filename + '_mep.cube', mf.make_rdm1())
