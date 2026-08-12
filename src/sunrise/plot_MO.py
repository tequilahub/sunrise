from pyscf import gto, scf
from pyscf.tools import cubegen
from tequila.quantumchemistry.qc_base import QuantumChemistryBase
from sunrise.miscellaneous.bar import giuseppe_bar
import sys
from numpy import ndarray,zeros,ix_

def plot_MO(molecule:QuantumChemistryBase, filename:str = None, orbital:list[int] = None, use_active:bool = True, print_orbital:bool = True, density:bool = False, mep:bool = False, rdm1:ndarray = None, exclude_core:bool = False):
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
    exclude_core: if custom rdm1 provided with active space shape, whether to include the frozen occupied orbitals on the total space rdm1
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
    pmol.build(atom = molecule.parameters.geometry, basis = molecule.parameters.basis_set, charge = molecule.parameters.charge, verbose=0)
    if density or mep:
        mf = scf.RHF(pmol).run()
        if rdm1 is None:
            rdm1 = mf.make_rdm1(mo_coeff=molecule.integral_manager.orbital_coefficients)
        else:
            mo_coeff = molecule.integral_manager.orbital_coefficients
            if not rdm1.shape[0] == molecule.integral_manager.orbital_coefficients.shape[1]: # already provided on frozen_core = False
                assert rdm1.shape[0] == molecule.n_orbitals, f"RDM1 provided with unexpected shape ({rdm1.shape}), expected either the number of active orbitals ({molecule.n_orbitals})\n or the number of total orbitals ({molecule.integral_manager.orbital_coefficients.shape[1]})"
                rdm = zeros(shape = (mo_coeff.shape[1], mo_coeff.shape[1])) # rectangular mo_coeffs
                if not exclude_core:
                    for i in molecule.integral_manager.active_space.frozen_reference_orbitals:
                        rdm[i,i] = 2 # NOTE: Experimental. Including contribution only from the active orbitals, expected to be more useful on density than mep
                rdm[ix_(molecule.integral_manager.active_space.active_orbitals, molecule.integral_manager.active_space.active_orbitals)] = rdm1
                rdm1 = rdm
            rdm1 = molecule.integral_manager.orbital_coefficients @ rdm1 @ molecule.integral_manager.orbital_coefficients.T
    if print_orbital:
        for i,idx in enumerate(orbital):
            giuseppe_bar(step = i, total_steps = len(orbital))
            cubegen.orbital(pmol, str(label[i])+ "_" + filename + "_MO.cube", molecule.integral_manager.orbital_coefficients[:, idx])
        giuseppe_bar(step = i + 1, total_steps = len(orbital))
        sys.stdout.write('\n')
        sys.stdout.flush()
    if density:
        cubegen.density(pmol, filename + '_density.cube', rdm1)
    if mep:
        cubegen.mep(pmol, filename + '_mep.cube', rdm1)
