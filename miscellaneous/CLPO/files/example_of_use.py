import tequila as tq
from tequila.quantumchemistry.pyscf_interface import QuantumChemistryPySCF
import numpy
from pyscf import scf,mp
from pyscf.tools import molden
from time import time
import subprocess
from copy import deepcopy
import pickle
import sys
from tequila.quantumchemistry.qc_base import QuantumChemistryBase
import numpy
from copy import deepcopy
from typing import Tuple

def transform(original:QuantumChemistryBase,modified:QuantumChemistryBase)->Tuple[QuantumChemistryBase,dict]:
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
    return QuantumChemistryBase(parameters=parameters,integral_manager=integral_manager,transformation=original.transformation),to_active

def get_MP2_occ(mol):
    fr = [2 for _ in range(mol.parameters.get_number_of_core_electrons()//2)] #NOTE modify if not traditional
    molx = QuantumChemistryPySCF.from_tequila(mol)
    beg = time()
    hf = molx._get_hf()
    mp2 = mp.MP2(hf)
    rdm1 = mp2.run().make_rdm1()
    end = time()
    return fr + numpy.diag(rdm1).tolist(),mp2.mo_energy ,end-beg

def read_molden_mo_matrix(filename):
    """
    Reads a Molden file and returns a NumPy array where
    each column is an MO coefficient vector.
    """
    mo_vectors = []
    current_mo = []
    in_mo_section = False

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Detect MO section
            if line == "[MO]":
                in_mo_section = True
                continue

            if not in_mo_section:
                continue

            if line.startswith("Sym="):
                if current_mo:
                    mo_vectors.append(current_mo)
                    current_mo = []
                continue

            # Skip metadata lines
            if (
                line.startswith("Ene=") or
                line.startswith("Spin=") or
                line.startswith("Occup=") or
                not line
            ):
                continue

            parts = line.split()
            if len(parts) == 2:
                try:
                    coeff = float(parts[1])
                    current_mo.append(coeff)
                except ValueError:
                    pass

        if current_mo:
            mo_vectors.append(current_mo)

    matrix = numpy.array(mo_vectors).T
    return matrix

def extract_clpo_graph(graph_file):
    """
    Reads a CLPO 'graph' file and extracts a graph representation.

    Returns:
        List of tuples:
        - (i,)      for lone pairs
        - (i, i+1)  for BD/NB pairs
    """
    nodes = []

    with open(graph_file, "r") as f:
        lines = f.readlines()

    # Skip header
    data_lines = [line.rstrip() for line in lines if line.strip()][1:]

    i = 0
    while i < len(data_lines):
        line = data_lines[i]

        # Lone pair
        if "(LP)" in line:
            nodes.append((i,))
            i += 1
            continue

        # Bonding orbital → must pair with next line
        if "(BD)" in line:
            if i + 1 >= len(data_lines):
                raise ValueError("BD entry without following NB line")

            nodes.append((i, i + 1))
            i += 2
            continue

        i += 1

    return nodes

geo = '''
H          0.00000        0.75545       -0.47116
O          0.00000        0.00000        0.11779
H          0.00000       -0.75545       -0.47116'''

basis = 'sto-3g'
threshold = 1.e-12
begining = time()
mp2_occ = False
name = 'awa'
mol = tq.Molecule(geometry=geo,basis_set=basis,backend='pyscf',units='a',name=name)
# ref = mol.compute_energy('CCSD(T)') # NOTE 
pfmol = mol.pyscf_molecule
filename = f'{name}_{basis}'


if mp2_occ:
    mo_occ,mo_energy,mp2_time = get_MP2_occ(mol)
    print(f'MP2 computation overhead: {mp2_time} s')
else:
    mf = scf.RHF(pfmol).run()
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy

# #IDEA: Not really sure when to use each option of pyscf molden generation
# ### OPTION 1
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
if sys.platform == "darwin":
    p = subprocess.Popen(
    f'./molden2aim_mc.exe -i {name}.molden',
    shell=True,
    stdin=subprocess.PIPE,
    text=True
    )
    p.communicate(input=input_answers)
    subprocess.call(f'./JANPA_macos -i {name}.molden -CLPO_Molden_File {name}_CLPO.molden -HybrOptOccConvThresh {threshold}',shell=True)
    subprocess.call(f'./replace_mc.sh {name}_CLPO.molden',shell=True) #if it doesnt work here: chmod u+rx replace.sh and run it again
elif sys.platform == "linux" or sys.platform == "linux2":
    p = subprocess.Popen(
    f'./molden2aim.exe -i {name}.molden',
    shell=True,
    stdin=subprocess.PIPE,
    text=True
    )
    p.communicate(input=input_answers)
    subprocess.call(f'./JANPA_linux -i {name}.molden -CLPO_Molden_File {name}_CLPO.molden -HybrOptOccConvThresh {threshold}',shell=True)
    subprocess.call(f'./replace_lnx.sh {name}_CLPO.molden',shell=True) #if it doesnt work here: chmod u+rx replace.sh and run it again
elif sys.platform == "win32":
    raise tq.TequilaException('Windows not implemented (yet?)')
else: raise tq.TequilaException('Is this code being run inside Doom?')

subprocess.call(f'rm m2a.ini',shell=True)
subprocess.call(f'rm {name}.molden',shell=True) 
subprocess.call(f'rm {name}_new.molden',shell=True) 
mo_matrix = read_molden_mo_matrix(f"{name}_CLPO.molden")
nmol = deepcopy(mol)
nmol.integral_manager.orbital_coefficients = mo_matrix
mol,to_active = transform(mol,nmol)
graph = extract_clpo_graph("graph")
ncore = len(mol.integral_manager.orbital_coefficients)-mol.n_orbitals
graph = [tuple([to_active[i]-ncore for i in edge if i in to_active.keys()])for edge in graph]
graph = [g for g in graph if len(g)]
subprocess.call(f'rm graph',shell=True)
print('Edges',graph)
# sun.plot_MO(mol,filename=f'{name}_CLPO')
# with open(filename+'.data', 'wb') as file:
#         pickle.dump(mol, file)

# objects = []
# with (open(filename + '.data', "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break
# mol:tq.chemistry.QuantumChemistryBase = objects[0]
# sun.plot_MO(mol,filename=filename)
