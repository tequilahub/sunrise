''' HOW TO USE IT
1. Install opentrustregion : https://github.com/eriksen-lab/opentrustregion
2. Dowloand their pyscf interface : https://github.com/eriksen-lab/pyscf_opentrustregion/tree/main
3. (At least for me, not sure if bcs too updated pyscf version) Go to this folder/pyscf/opentrustregion/
and comment out:
3.1: path/pyscf_opentrustregion/pyscf/opentrustregion/__init__.py, line 14 (FourthMomentOTR,)
3.2: path/pyscf_opentrustregion/pyscf/opentrustregion/opentrustregion_interface.py, line 169-170: class FourthMomentOTR(...): pass
4. Modify the path variable bellow to the downloaded folder
5. Ready
'''
import os
path = 'PATH/pyscf_opentrustregion'
os.environ['PYSCF_EXT_PATH']=path
import sunrise as sun
import tequila as tq
from sunrise.hybrid_base.HybridBase import HybridBase
from pyscf import gto, scf
from pyscf.opentrustregion import (
    PipekMezeyOTR,
    BoysOTR,
    EdmistonRuedenbergOTR,
)
from time import time

def Pipek_Mezey(mol):
    # perform HF calculation
    hf = scf.RHF(mol).run()
    # orbitals
    orbs = [hf.mo_coeff[:, : ], hf.mo_coeff[:, :]]
    ### Pipek-Mezey localization
    # loop over occupied and virtual subspaces
    for mo_coeff in orbs:
        loc = PipekMezeyOTR(mol, mo_coeff)
        loc.line_search = True
        loc.kernel()
    # transform to tequila
    tqmol = sun.MoleculeFromPyscf(loc.mol,mo_coeff=loc.mo_coeff)
    # sun.plot_MO(tqmol,filename='Pipek_Mezey')
    smol = HybridBase.from_tequila(tqmol)
    edges = smol.get_spa_edges()
    guess = smol.get_spa_guess().T #Obv doesnt work, but dont want to do edges and guess by hand
    U = tqmol.make_ansatz("HCB-SPA",edges=edges)
    opt = tq.chemistry.optimize_orbitals(molecule=tqmol,circuit=U,initial_guess=guess,use_hcb=True,silent=True)
    # sun.plot_MO(opt.molecule,filename='Pipek_Mezey_SPA')
    return opt.energy

def Foster_Boys(mol):
    # perform HF calculation
    hf = scf.RHF(mol).run()
    # orbitals
    orbs = [hf.mo_coeff[:, : ], hf.mo_coeff[:, :]]
    ### Foster-Boys localization
    # loop over occupied and virtual subspaces
    for mo_coeff in orbs:
        loc = BoysOTR(mol, mo_coeff)
        loc.line_search = True
        loc.kernel()
    # transform to tequila
    tqmol = sun.MoleculeFromPyscf(loc.mol,mo_coeff=loc.mo_coeff)
    sun.plot_MO(tqmol,filename='Foster_Boys')
    smol = HybridBase.from_tequila(tqmol)
    edges = smol.get_spa_edges()
    guess = smol.get_spa_guess().T #Obv doesnt work, but dont want to do edges and guess by hand
    U = tqmol.make_ansatz("HCB-SPA",edges=edges)
    opt = tq.chemistry.optimize_orbitals(molecule=tqmol,circuit=U,initial_guess=guess,use_hcb=True,silent=True)
    # sun.plot_MO(opt.molecule,filename='Foster_Boys_SPA')
    return opt.energy

def Edmiston_Ruedenberg(mol):
    # perform HF calculation
    hf = scf.RHF(mol).run()
    # orbitals
    orbs = [hf.mo_coeff[:, : ], hf.mo_coeff[:, :]]
    ### Edmiston-Ruedenberg
    # loop over occupied and virtual subspaces
    for mo_coeff in orbs:
        loc = EdmistonRuedenbergOTR(mol, mo_coeff)
        loc.line_search = True
        loc.kernel()
    # transform to tequila
    tqmol = sun.MoleculeFromPyscf(loc.mol,mo_coeff=loc.mo_coeff)
    sun.plot_MO(tqmol,filename='Edmiston_Ruedenberg')
    smol = HybridBase.from_tequila(tqmol)
    edges = smol.get_spa_edges()
    guess = smol.get_spa_guess().T #Obv doesnt work, but dont want to do edges and guess by hand
    U = tqmol.make_ansatz("HCB-SPA",edges=edges)
    opt = tq.chemistry.optimize_orbitals(molecule=tqmol,circuit=U,initial_guess=guess,use_hcb=True,silent=True)
    # sun.plot_MO(opt.molecule,filename='Edmiston_Ruedenberg_SPA')
    return opt.energy

def Tequila(mol):
    mol = sun.MoleculeFromPyscf(mol)
    mol = mol.use_native_orbitals()
    smol = HybridBase.from_tequila(mol)
    edges = smol.get_spa_edges()
    guess = smol.get_spa_guess().T
    U = mol.make_ansatz("HCB-SPA",edges=edges)
    opt = tq.chemistry.optimize_orbitals(molecule=mol,circuit=U,initial_guess=guess,use_hcb=True,silent=True)
    # sun.plot_MO(opt.molecule,filename='SPA')
    return opt.energy

# define molecule
mol = gto.Mole()
mol.build(
    atom="""
        O    0.000  0.000  0.000
        H    0.000 -0.757  0.587
        H    0.000  0.757  0.587
    """,
    basis="sto-3g",
    symmetry=False,
)
#edges = [(2,), (3,), (0, 4), (1, 5)]
beg = time()
PM = Pipek_Mezey(mol)
tpm = time()
FB = Foster_Boys(mol)
tfm = time()
ER = Edmiston_Ruedenberg(mol)
ter = time()
teq = Tequila(mol)
ttq = time()
print(f"Pipek_Mezey {(PM-teq)*1000} mH with time {tpm-beg} s")
print(f"Foster_Boys {(FB-teq)*1000} mH with time {tfm-tpm} s")
print(f"Edmiston_Ruedenberg {(ER-teq)*1000} mH with time {ter-tfm} s")
print(f"Tequila time {ttq-ter} s")
