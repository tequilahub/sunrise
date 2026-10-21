"""HOW TO USE IT
1. Install opentrustregion : https://github.com/eriksen-lab/opentrustregion
2. Dowloand their pyscf interface : https://github.com/eriksen-lab/pyscf_opentrustregion/tree/main
3. Modify the path variable bellow to the downloaded folder
4. Ready
"""

import os

path = "PATH/pyscf_opentrustregion"
os.environ["PYSCF_EXT_PATH"] = path
from time import time

import numpy as np
from pyscf import gto, mcscf, scf
from pyscf.data import elements
from pyscf.opentrustregion import (
    BoysOTR,
    EdmistonRuedenbergOTR,
    PipekMezeyOTR,
    casscf_to_otr,
    mf_to_otr,
)
from scipy.optimize import linear_sum_assignment

import sunrise as sun


def spa_edges(mol, mo_coeff):
    """
    SPA edges for arbitrary orbitals (get_spa_edges() returns them in the CLPO basis).
    Each active occupied orbital is paired with the virtual orbital whose Mulliken
    atomic populations are most similar, leftover occupied orbitals are single edges.
    Indices are in the active space (core orbitals are frozen by sunrise).
    """
    ncore, nocc = elements.chemcore(mol), mol.nelectron // 2
    S = mol.intor("int1e_ovlp")
    atoms = [slice(a[2], a[3]) for a in mol.aoslice_by_atom()]
    pops = np.array([[(c * (S @ c))[at].sum() for at in atoms] for c in mo_coeff.T])
    occ, vir = range(ncore, nocc), range(nocc, mo_coeff.shape[1])
    cost = [[np.abs(pops[i] - pops[a]).sum() for a in vir] for i in occ]
    rows, cols = linear_sum_assignment(cost)
    pairs = {occ[r]: vir[c] for r, c in zip(rows, cols)}
    return [(i - ncore, pairs[i] - ncore) if i in pairs else (i - ncore,) for i in occ]


def localize(mol, localizer):
    # perform HF calculation
    hf = scf.RHF(mol).run()
    mo_coeff = hf.mo_coeff.copy()
    ncore, nocc = elements.chemcore(mol), mol.nelectron // 2
    # loop over occupied (core excluded) and virtual subspaces
    for space in (slice(ncore, nocc), slice(nocc, None)):
        loc = localizer(mol, mo_coeff[:, space])
        loc.line_search = True
        mo_coeff[:, space] = loc.kernel()
    return mo_coeff


def Pipek_Mezey(mol):
    ### Pipek-Mezey localization
    mo_coeff = localize(mol, PipekMezeyOTR)
    edges = spa_edges(mol, mo_coeff)
    # transform to sunrise
    mol = sun.MoleculeFromPyscf(mol, mo_coeff=mo_coeff, nature="h")
    # sun.plot_MO(mol,filename='Pipek_Mezey')
    U = mol.make_ansatz("HCB-SPA", edges=edges)
    opt = sun.optimize_orbitals(molecule=mol, circuit=U, use_hcb=True, silent=True)
    # sun.plot_MO(opt.molecule,filename='Pipek_Mezey_SPA')
    return opt.energy


def Foster_Boys(mol):
    ### Foster-Boys localization
    mo_coeff = localize(mol, BoysOTR)
    edges = spa_edges(mol, mo_coeff)
    # transform to sunrise
    mol = sun.MoleculeFromPyscf(mol, mo_coeff=mo_coeff, nature="q")
    # sun.plot_MO(mol, filename="Foster_Boys")
    U = mol.make_ansatz("HCB-SPA", edges=edges)
    opt = sun.optimize_orbitals(molecule=mol, circuit=U, use_hcb=True, silent=True)
    # sun.plot_MO(opt.molecule,filename='Foster_Boys_SPA')
    return opt.energy


def Edmiston_Ruedenberg(mol):
    ### Edmiston-Ruedenberg localization
    mo_coeff = localize(mol, EdmistonRuedenbergOTR)
    edges = spa_edges(mol, mo_coeff)
    # transform to sunrise
    mol = sun.MoleculeFromPyscf(mol, mo_coeff=mo_coeff, nature="q")
    sun.plot_MO(mol, filename="Edmiston_Ruedenberg")
    U = mol.make_ansatz("HCB-SPA", edges=edges)
    opt = sun.optimize_orbitals(molecule=mol, circuit=U, use_hcb=True, silent=True)
    # sun.plot_MO(opt.molecule,filename='Edmiston_Ruedenberg_SPA')
    return opt.energy


def CASSCF_OTR(mol, ncas=6, nelecas=8, edges=None):
    # HF with the OpenTrustRegion second-order solver
    mf = mf_to_otr(scf.RHF(mol))
    mf.conv_tol = 1e-8  # set after conversion, its meaning differs from pyscf
    mf.kernel()
    # CASSCF orbital optimization with the OpenTrustRegion solver
    # (default CAS(8,6) for water leaves O 1s as core, matching sunrise's frozen core)
    mc = casscf_to_otr(mcscf.CASSCF(mf, ncas, nelecas))
    mc.conv_tol = 1e-6
    mc.line_search = True
    mc.kernel()
    if edges is None:
        edges = spa_edges(mol, mc.mo_coeff)
    # transform to sunrise
    mol = sun.MoleculeFromPyscf(mol, mo_coeff=mc.mo_coeff, nature="q")
    # sun.plot_MO(mol, filename="CASSCF_OTR")
    U = mol.make_ansatz("HCB-SPA", edges=edges)
    opt = sun.optimize_orbitals(molecule=mol, circuit=U, use_hcb=True, silent=True)
    # sun.plot_MO(opt.molecule,filename='CASSCF_OTR_SPA')
    return opt.energy


def JANPA(mol):
    mol = sun.MoleculeFromPyscf(mol, nature="q")
    mol, edges = mol.use_CLPO_orbitals_and_edges()
    U = mol.make_ansatz("HCB-SPA", edges=edges)
    opt = sun.optimize_orbitals(molecule=mol, circuit=U, use_hcb=True, silent=True)
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
# edges = [(2,), (3,), (0, 4), (1, 5)]
beg = time()
PM = Pipek_Mezey(mol)
tpm = time()
FB = Foster_Boys(mol)
tfm = time()
ER = Edmiston_Ruedenberg(mol)
ter = time()
CAS = CASSCF_OTR(mol)
tcas = time()
teq = JANPA(mol)
ttq = time()
print(f"Pipek_Mezey {(PM - teq) * 1000} mH with time {tpm - beg} s")
print(f"Foster_Boys {(FB - teq) * 1000} mH with time {tfm - tpm} s")
print(f"Edmiston_Ruedenberg {(ER - teq) * 1000} mH with time {ter - tfm} s")
print(f"CASSCF_OTR {(CAS - teq) * 1000} mH with time {tcas - ter} s")
print(f"JANPA time {ttq - tcas} s")
