import sunrise as sn
from tequila.quantumchemistry.chemistry_tools import NBodyTensor
import numpy as np
import subprocess
try:
    from pyblock2._pyscf.ao2mo import integrals as itg
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
except ImportError:
    pass

geometry = '''
C 0.00000 0.00000 0.00000
C 1.48460 0.00000 0.00000
C -0.76837 0.00000 -1.12008
C 2.25297 0.00000 -1.12008
H -0.47297 -0.00038 0.97971
H 1.95470 -0.00556 0.97953
H -1.85042 -0.00018 -1.03002
H -0.35993 -0.00000 -2.12298
H 3.33365 -0.01193 -1.02751
H 1.85521 0.00845 -2.12564'''
mol = sn.Molecule(geometry=geometry,basis_set='sto-3g',nature='hybrid')
print("Canonical Orb. HF Energy ",mol.compute_energy("HF"))
print("Canonical Orb. CCSD(T) Energy ",mol.compute_energy("CCSD(T)"))
mol = sn.CLPO.generate_CLPO_molecule(mol)
mol.update_select([4,5,10,11]) # Just the pi-orbitals



# In principle, any regular mol.compute_energy() energy reference could be employed, but it would be similar comparison as using it to an active space
# One could also use mol.compute_restricted_energy() which computes the energy setting the initial integrals to zero if not used on the encoding selection.
# Howeverm this works better with canonical orbitals and may break if the fermionic encoding is too small. 
# For this reason, on this example we will show how to use DMRG/py2block to stimate this Hybrid Encoding Error

c,h1,h2 = mol.get_restricted_integrals()
h2 = h2.reorder('chem')
threads = 4
ram_gb = 4
bd_pre = 200
bd_reordered = 200
n_sweeps = 500
verbose = 0
# Take into account that DMRG takes advantage of localized orbitals, if using canonical/delocalized the bond_dim may need to be increased.


#### First DMRG calculation
driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=threads, restart_dir="./restart")
driver.initialize_system(n_sites=mol.n_orbitals, n_elec=mol.n_electrons, spin=0)
mpo = driver.get_qc_mpo(h1e=h1, g2e=h2.elems, ecore=c, iprint=verbose)
ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)

energy = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=[bd_pre], iprint=verbose)
print('First DMRG energy = %20.15f' % energy)

#### Orbital reordering
idx = driver.orbital_reordering(h1, h2.elems)
print('Orbital Reordering ',idx)
h1_new = h1[idx][:, idx]
g2_new = h2.elems[idx][:, idx][:, :, idx][:, :, :, idx]

#### Main DMRG calculation
driver.initialize_system(n_sites=mol.n_orbitals, n_elec=mol.n_electrons, spin=0)
mpo = driver.get_qc_mpo(h1e=h1_new, g2e=g2_new, ecore=c, iprint=verbose)
ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=[bd_reordered], iprint=verbose)
print('Main DMRG energy = %20.15f' % energy)

#### PDM extraction
pdm1 = driver.get_1pdm(ket)
pdm2 = driver.get_2pdm(ket).transpose(0, 3, 1, 2)
print('Energy from pdms = %20.15f' % (np.einsum('ij,ij->', pdm1, h1_new) + 0.5 * np.einsum('ijkl,ijkl->', pdm2, driver.unpack_g2e(g2_new)) + c))

idx_back = np.zeros(len(idx), dtype=int)
for i in range(len(idx)):
    idx_back[idx[i]] = i
    
pdm1 = pdm1[idx_back][:, idx_back]
pdm2 = pdm2[idx_back][:, idx_back][:, :, idx_back][:, :, :, idx_back]
pdm2 = np.swapaxes(pdm2,1,2) #WICHTIG: UMSORTIERUNG VON RDMS IN PYSCF
print('Energy from reordered pdms = %20.15f' % (np.einsum('ij,ij->', pdm1, h1) + 0.5 * np.einsum('ijkl,ikjl->', pdm2, driver.unpack_g2e(h2.elems)) + c))
# OrbOpt_helper.write_rdms(pdm1, pdm2, it_str + "/" + molecule_name)

dmrg_energy = np.einsum('ij,ij->', pdm1, h1) + 0.5 * np.einsum('ijkl,ikjl->', pdm2, driver.unpack_g2e(h2.elems)) + c
print('dmrg_energy ',dmrg_energy)
# iteration_energies.append(dmrg_energy.__str__())
################### DMRG finished #################### 
subprocess.call(f'rm -rf tmp/',shell=True)
subprocess.call(f'rm -rf restart/',shell=True)
