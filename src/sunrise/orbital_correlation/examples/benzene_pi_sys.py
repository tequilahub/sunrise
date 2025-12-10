# Benzene (c6h6)
import sunrise as sun
import tequila as tq
import numpy as np
from sunrise.orbital_correlation.density_matrix_utils import *
from sunrise.orbital_correlation.quantum_info_utils import *
from sunrise.fermionic_operations.givens_rotations import get_givens_circuit

geometry = """
C 0.000000 1.396792 0.000000
C 0.000000 -1.396792 0.000000
C 1.209657 0.698396 0.000000
C -1.209657 -0.698396 0.000000
C -1.209657 0.698396 0.000000
C 1.209657 -0.698396 0.000000
H 0.000000 2.484212 0.000000
H 2.151390 1.242106 0.000000
H -2.151390 -1.242106 0.000000
H -2.151390 1.242106 0.000000
H 2.151390 -1.242106 0.000000
H 0.000000 -2.484212 0.000000
"""

# We select only the pi system, so it behaves like H6
# mol = tq.Molecule(geometry=geometry, basis_set='sto-3g')
mol = tq.Molecule(geometry=geometry, basis_set='sto-3g', active_orbitals=[16,19,20,21,22,23]).use_native_orbitals()
# sun.plot_MO(mol)

reorder = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0]
])
mol = mol.transform_orbitals(reorder.T)
H = mol.make_hamiltonian()
U = mol.make_ansatz(name="SPA", edges=[(0,1),(2,3),(4,5)])
guess = np.eye(6)
guess[0] = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
guess[1] = [1.0, -1., 0.0, 0.0, 0.0, 0.0]
guess[2] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
guess[3] = [0.0, 0.0, 1.0, -1., 0.0, 0.0]
guess[4] = [0.0, 0.0, 0.0, 0.0, 1.0, -1.]
guess[5] = [0.0, 0.0, 0.0, 0.0, 1.0, -1.]
opt = tq.chemistry.optimize_orbitals(mol, circuit=U, initial_guess=guess.T, silent=True)
UR = mol.get_givens_circuit(opt.mo_coeff)
U = U + UR.dagger()
E = tq.ExpectationValue(U,H)
result = tq.minimize(E, silent=True)
U = U.map_variables(result.variables)
print(tq.simulate(U))
print("I_01:", mutual_info_simple(mol, U, orb_a=0, orb_b=1))
print("I_02:", mutual_info_simple(mol, U, orb_a=0, orb_b=2))
print("I_03:", mutual_info_simple(mol, U, orb_a=0, orb_b=3))
print("I_04:", mutual_info_simple(mol, U, orb_a=0, orb_b=4))
print("I_05:", mutual_info_simple(mol, U, orb_a=0, orb_b=5))
print("I_12:", mutual_info_simple(mol, U, orb_a=1, orb_b=2))
print("I_13:", mutual_info_simple(mol, U, orb_a=1, orb_b=3))
print("I_14:", mutual_info_simple(mol, U, orb_a=1, orb_b=4))
print("I_15:", mutual_info_simple(mol, U, orb_a=1, orb_b=5))
print("I_23:", mutual_info_simple(mol, U, orb_a=2, orb_b=3))
print("I_24:", mutual_info_simple(mol, U, orb_a=2, orb_b=4))
print("I_25:", mutual_info_simple(mol, U, orb_a=2, orb_b=5))
print("I_34:", mutual_info_simple(mol, U, orb_a=3, orb_b=4))
print("I_35:", mutual_info_simple(mol, U, orb_a=3, orb_b=5))
print("I_45:", mutual_info_simple(mol, U, orb_a=4, orb_b=5))
print("E_0:", one_orb_entanglement(mol, U, orb_a=0))
print("E_1:", one_orb_entanglement(mol, U, orb_a=1))
print("E_2:", one_orb_entanglement(mol, U, orb_a=2))
print("E_3:", one_orb_entanglement(mol, U, orb_a=3))
print("E_4:", one_orb_entanglement(mol, U, orb_a=4))
print("E_5:", one_orb_entanglement(mol, U, orb_a=5))

# I_01: 0.021645242081348437
# I_02: 0.16757141451466895
# I_03: 0.685622272659947
# I_04: 0.42718259623625654
# I_05: 0.2488512557866156
# I_12: 1.0249493248803998
# I_13: 0.09237509757313278
# I_14: 0.19031846159988852
# I_15: 0.26449184372981116
# I_23: 0.006832144361355708
# I_24: 0.013249976281854536
# I_25: 0.02707483227044527
# I_34: 0.037321176133418676
# I_35: 0.08155909050126353
# I_45: 1.6754963912101224
# E_0: 0.9262120791732892
# E_1: 0.9434653103659904
# E_2: 0.7741579610435207
# E_3: 0.6002806162902423
# E_4: 1.3272886464660163
# E_5: 1.3075728902158705