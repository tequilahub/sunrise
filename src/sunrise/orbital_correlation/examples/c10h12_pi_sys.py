import sunrise as sun
import tequila as tq
import numpy as np
from sunrise.orbital_correlation.density_matrix_utils import *
from sunrise.orbital_correlation.quantum_info_utils import *
from sunrise.fermionic_operations.givens_rotations import get_givens_circuit
# import sys
# sys.path.insert(1,'spafastprototype')
# from decompose import decompose
# from spa import run_spa

geometry = """
C -5.570819 -0.217743 0.000000
H -5.646184 -1.298410 0.000000
H -6.495732 0.342247 0.000000
C -4.373490 0.405989 0.000000
H -4.347960 1.490582 0.000000
C -3.085856 -0.277393 0.000000
H -3.106017 -1.362443 0.000000
C -1.885150 0.357505 0.000000
H -1.868794 1.442672 0.000000
C -0.600421 -0.318770 0.000000
H -0.615607 -1.403834 0.000000
C 0.600423 0.318772 0.000000
H 0.615610 1.403837 0.000000
C 1.885151 -0.357504 0.000000
H 1.868791 -1.442671 0.000000
C 3.085859 0.277388 0.000000
H 3.106026 1.362438 0.000000
C 4.373489 -0.406002 0.000000
H 4.347952 -1.490594 0.000000
C 5.570825 0.217717 0.000000
H 5.646212 1.298382 0.000000
H 6.495727 -0.342292 0.000000
"""

mol = sun.Molecule(geometry=geometry, basis_set='sto-3g', nature='f', active_orbitals=[30,32,33,34,35,36,37,38,39,40]).use_native_orbitals()
# mol = tq.Molecule(geometry=geometry, basis_set='sto-3g', active_orbitals=[30,32,33,34,35,36,37,38,39,40]).use_native_orbitals() # for fast-spa
edges = [(0,1),(2,3),(4,5),(6,7),(8,9)]
# initial_guess = np.array([
#     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1,-1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1,-1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1,-1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1,-1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1,-1]
# ], dtype=float)
# U = mol.make_ansatz(name="HCB-SPA", edges=edges)
# opt = run_spa(mol=mol, edges=edges,initial_guess=initial_guess.T, decompose=True, silent=False, grouping=12, fast_rdm=True) # optimization through fast-spa
# mo_coeff = opt.mo_coeff
# with open('c10h12_spa_orb.npy', 'wb') as f:
#     np.save(f, opt.mo_coeff)
with open('c10h12_spa_orb.npy', 'rb') as f: # stored mo coeff matrix
    mo_coeff = np.load(f)

U = sun.FCircuit.from_edges(n_orb=mol.n_orbitals, edges=edges)
UR = get_givens_circuit(mo_coeff)
UR_dag = get_givens_circuit(mo_coeff.T) # Transpose because .dagger() doesn't work
U = U + UR_dag
# print(U.extract_indices())
# E = sun.Braket(backend='fqe', molecule=mol, U=U)
# result = sun.minimize(E, silent=False)
# U = U.map_variables(result.variables)
variables = {((0, 1), 'D', None): -0.49211983715024776, ((2, 3), 'D', None): -0.484945590285971, ((4, 5), 'D', None): -0.4822927757317614, ((6, 7), 'D', None): -0.4845345232116803, ((8, 9), 'D', None): -0.4971122459912079}
U = U.map_variables(variables)
# print(U)
# print(sun.simulate(U))

mol = sun.Molecule(geometry=geometry, basis_set='sto-3g', active_orbitals=[30,32,33,34,35,36,37,38,39,40], nature='t', transformation="reordered-jordan-wigner").use_native_orbitals()
U = U.to_qcircuit(mol)
pairs = [(i, j) for i in range(mol.n_orbitals) for j in range(i + 1, mol.n_orbitals)]
for pair in pairs:
    print(f"I({pair[0],pair[1]}) = {mutual_info_2ordm(mol, U, orb_a=pair[0], orb_b=pair[1])}")