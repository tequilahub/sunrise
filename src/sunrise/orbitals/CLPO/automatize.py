import tequila as tq
from tequila.quantumchemistry.qc_base import QuantumChemistryBase
import numpy
import sunrise as sun
from pyscf import scf
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


geo = '''
    C	0.0000000	1.3894860	0.0000000
    C	1.2033300	0.6947430	0.0000000
    C	1.2033300	-0.6947430	0.0000000
    C	0.0000000	-1.3894860	0.0000000
    C	-1.2033300	-0.6947430	0.0000000
    C	-1.2033300	0.6947430	0.0000000
    H	0.0000000	2.4702840	0.0000000
    H	2.1393290	1.2351420	0.0000000
    H	2.1393290	-1.2351420	0.0000000
    H	0.0000000	-2.4702840	0.0000000
    H	-2.1393290	-1.2351420	0.0000000
    H	-2.1393290	1.2351420	0.0000000'''
basis = 'aug_cc_pvdz'
threshold = 1.e-9
begining = time()
mol = tq.Molecule(geometry=geo,basis_set=basis,backend='pyscf',units='a')
filename = f'{mol.parameters.name}_{basis}.data'
pfmol = sun.from_tequila(mol)
mf = scf.RHF(pfmol).run()
name = mol.parameters.name 
#IDEA: Not really sure when to use each option of pyscf molden generation
### OPTION 1
with open(f'{name}.molden', 'w') as f1:
    molden.header(pfmol, f1)
    molden.orbital_coeff(pfmol, f1, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)
### OPTION 2
# try:
#     molden.from_mo(pfmol, f'{mol.parameters.name}.molden', mf.mo_coeff)
# except RuntimeError:
#     print('    Found l=5 in basis.')
#     molden.from_mo(pfmol, f'{mol.parameters.name}.molden', mf.mo_coeff, ignore_h=True)
# #Pyscf tutorial Molden: https://github.com/pyscf/pyscf/blob/master/examples/tools/02-molden.py

#it will give you some options, accept only the first: Do you want to generate a new Molden file? ([Yes] / No)
subprocess.call(f'./molden2aim.exe -i {name}.molden',shell=True)
#IDEA This converts the basis to cartesians, and give problems when opening it again with tequila
# subprocess.call(f'java -jar molden2molden.jar -cart2pure -normalizebf -i {name}_new.molden -o {name}.PURE',shell=True) 
# subprocess.call(f'java -jar JANPA.jar -i {name}.PURE -NAO2AO_File {name}_NAO2AO.data -LHO2NAO_File {name}_LHO2NAO.data -CLPO2LHO_File {name}_CLPO2LHO.data  -HybrOptOccConvThresh {threshold}',shell=True) #this would be an alternative using mol.transform_orbitals()
#IDEA Instead going with this for bigger basis
subprocess.call(f'java -jar JANPA.jar -i {name}.molden -CLPO_Molden_File {name}_CLPO.molden -HybrOptOccConvThresh {threshold}',shell=True)

subprocess.call(f'./replace.sh {name}_CLPO.molden',shell=True) #if it doesnt work here: chmod u+rx replace.sh and run it again
subprocess.call(f'rm m2a.ini',shell=True) #rm molden2aim input file autogenerated
subprocess.call(f'rm {name}.PURE',shell=True) 
subprocess.call(f'rm {name}.molden',shell=True) 
subprocess.call(f'rm {name}_new.molden',shell=True) 

fmol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = molden.load(f'{name}_CLPO.molden')
mol = transform(mol,sun.MoleculeFromPyscf(molecule=fmol,mo_coeff=mo_coeff,basis_set=basis))
sun.plot_MO(mol,filename=f'{name}_CLPO')

# JANPA orders the orbitals by bonding-antibonding pairs, therefore the edges will be:
edges = [(2*i,2*i+1) for i in range(mol.n_electrons//2)]
with open(filename, 'wb') as file:
        pickle.dump(mol, file)