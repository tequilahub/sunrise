import tequila as tq
from tequila.quantumchemistry.qc_base import QuantumChemistryBase
from tequila.quantumchemistry.pyscf_interface import QuantumChemistryPySCF
import numpy
import sunrise as sun
from pyscf import scf,mp
from pyscf.tools import molden
from time import time
import subprocess
from copy import deepcopy
import pickle
def transform(original:QuantumChemistryBase,modified:QuantumChemistryBase)->QuantumChemistryBase:
    '''
    Procedure similar to what is done in use_native_orbitals but for arbitrary basis (the CLPO orbitals in this case)
    '''
    def inner(a, b, s):
        return numpy.sum(numpy.multiply(numpy.outer(a, b), s))
    core = [i.idx_total for i in original.integral_manager.orbitals if i.idx is None]
    assert len(original.integral_manager.orbitals) == len(modified.integral_manager.orbitals)
    d = deepcopy(modified.integral_manager.orbital_coefficients).T
    c = deepcopy(original.integral_manager.orbital_coefficients).T
    s = original.integral_manager.overlap_integrals
    n_basis = len(d)
    ov = numpy.zeros(shape=(n_basis))
    for i in core:
        for j in range(n_basis):
            ov[j] += numpy.abs(inner(c[i], d[j], s))
    co = {}
    for i in core:
        idx = numpy.argmax(ov)
        co[i] = idx
        ov[idx] = 0
    active = [i for i in range(n_basis) if i not in co.values()]
    to_active =  [i for i in range(n_basis) if  i not in co.keys()]
    to_active = {active[i]:to_active[i] for i in range(len(active))}
    reference_orbitals = [*co.keys()]
    i =0
    while len(reference_orbitals)<original.parameters.total_n_electrons//2:
        if i not in reference_orbitals:
            reference_orbitals.append(i)
        i += 1
    sbar = numpy.zeros(shape=s.shape)
    for k in active:
        for i in core:
            sbar[i][to_active[k]] = inner(c[i], d[k], s)
    dbar = numpy.zeros(shape=s.shape)

    for j in active:
        dbar[to_active[j]] = d[j]
        for i in core:
            temp = sbar[i][to_active[j]] * c[i]
            dbar[to_active[j]] -= temp
    for i in to_active.values():
        norm = numpy.sqrt(inner(dbar[i], dbar[i], s.T))
        if not numpy.isclose(norm, 0):
            dbar[i] = dbar[i] / norm
    for j in to_active.values():
        c[j] = dbar[j]
    sprima = numpy.eye(len(c))
    for idx, i in enumerate(to_active.values()):
        for j in [*to_active.values()][idx:]:
            sprima[i][j] = inner(c[i], c[j], s)
            sprima[j][i] = sprima[i][j]
    lam_s, l_s = numpy.linalg.eigh(sprima)
    lam_s = lam_s * numpy.eye(len(lam_s))
    lam_sqrt_inv = numpy.sqrt(numpy.linalg.inv(lam_s))
    symm_orthog = numpy.dot(l_s, numpy.dot(lam_sqrt_inv, l_s.T))
    jcoef = symm_orthog.dot(c).T
    integral_manager = modified.initialize_integral_manager(one_body_integrals=original.integral_manager.one_body_integrals,
                    two_body_integrals=original.integral_manager.two_body_integrals,constant_term=original.integral_manager.constant_term,
                    active_orbitals= [i for i in range(n_basis) if  i not in co.keys()],frozen_orbitals=[*co.keys()],orbital_coefficients=jcoef,
                    overlap_integrals=original.integral_manager.overlap_integrals,reference_orbitals=reference_orbitals,orbital_type='CLPO')
    parameters = deepcopy(original.parameters)
    return QuantumChemistryBase(parameters=parameters,integral_manager=integral_manager,transformation=original.transformation)

def get_MP2_occ(mol):
    fr = [2 for _ in range(mol.parameters.get_number_of_core_electrons()//2)] #NOTE modify if not traditional
    molx = QuantumChemistryPySCF.from_tequila(mol)
    beg = time()
    hf = molx._get_hf()
    mp2 = mp.MP2(hf)
    rdm1 = mp2.run().make_rdm1()
    end = time()
    return fr + numpy.diag(rdm1).tolist(),mp2.mo_energy ,end-beg

geo = '''
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116'''

basis = 'sto-3g'

threshold = 1.e-12
begining = time()
mp2_occ = False
name = None
so_mac = True #only compiled for mac and linux for now

mol = tq.Molecule(geometry=geo,basis_set=basis,backend='pyscf',units='a')
ref = mol.compute_energy('CCSD(T)') # NOTE May be too expensive for biger molecules, take care
filename = f'{mol.parameters.name}_{basis}.data'
pfmol = mol.pyscf_molecule
mf = scf.RHF(pfmol).run()
if name is None:
    name = mol.parameters.name 
name = '1a-1b_start'
#IDEA: Not really sure when to use each option of pyscf molden generation
### OPTION 1

if mp2_occ:
    mo_occ,mo_energy,mp2_time = get_MP2_occ(mol)
    print(f'MP2 computation overhead: {mp2_time} s')
else:
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy

with open(f'{name}.molden', 'w') as f1:
    molden.header(pfmol, f1) 
    molden.orbital_coeff(pfmol, f1, mf.mo_coeff, ene=mo_energy, occ=mo_occ)
### OPTION 2
# try:
#     molden.from_mo(pfmol, f'{mol.parameters.name}.molden', mf.mo_coeff)
# except RuntimeError:
#     print('    Found l=5 in basis.')
#     molden.from_mo(pfmol, f'{mol.parameters.name}.molden', mf.mo_coeff, ignore_h=True)
# #Pyscf tutorial Molden: https://github.com/pyscf/pyscf/blob/master/examples/tools/02-molden.py

input_answers = "y\nn\nn\nn\n"

#Use Popen to pipe the answers in
p = subprocess.Popen(
    f'./molden2aim.exe -i {name}.molden',
    shell=True,
    stdin=subprocess.PIPE,
    text=True
)
p.communicate(input=input_answers)
#IDEA Instead going with this for bigger basis
if so_mac:
    subprocess.call(f'./JANPA_macos -i {name}.molden -CLPO_Molden_File {name}_CLPO.molden -HybrOptOccConvThresh {threshold}',shell=True) #reading the output molden file
    subprocess.call(f'./replace_mc.sh {name}_CLPO.molden',shell=True) #if it doesnt work here: chmod u+rx replace.sh and run it again
else:
    subprocess.call(f'./JANPA_linux -i {name}.molden -CLPO_Molden_File {name}_CLPO.molden -HybrOptOccConvThresh {threshold}',shell=True) #reading the output molden file
    subprocess.call(f'./replace_lnx.sh {name}_CLPO.molden',shell=True) #if it doesnt work here: chmod u+rx replace.sh and run it again
subprocess.call(f'rm m2a.ini',shell=True) #rm molden2aim input file autogenerated
subprocess.call(f'rm {name}.molden',shell=True) 
subprocess.call(f'rm {name}_new.molden',shell=True) 

fmol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = molden.load(f'{name}_CLPO.molden')
mol = transform(mol,sun.MoleculeFromPyscf(molecule=fmol,mo_coeff=mo_coeff,basis_set=basis))
sun.plot_MO(mol,filename=f'{name}_CLPO')
# NOTE: JANPA orders the orbitals by bonding-antibonding pairs, therefore the edges will be:
# NOTE take care fore basis bigger than minimal, the edges may not be asigned to the lower angular moment basis
edges = [(2*i,2*i+1) for i in range(mol.n_electrons//2)]
with open(filename, 'wb') as file:
        pickle.dump(mol, file)