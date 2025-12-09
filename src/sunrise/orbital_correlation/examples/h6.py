import tequila as tq
import numpy as np
from sunrise.orbital_correlation.density_matrix_utils import *
from sunrise.orbital_correlation.quantum_info_utils import *

# canonical orbitals
mol = tq.Molecule("H 0 0 0\nH 0 0 1.5\nH 0 0 3\nH 0 0 4.5\nH 0 0 6\nH 0 0 7.5", "sto-3g")
# print(mol.integral_manager.orbital_coefficients)
H = mol.make_hamiltonian()
U = mol.prepare_reference()
print(tq.simulate(U))
# print("I_01:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=1))
# print("I_02:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=2))
# print("I_03:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=3))
# print("I_04:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=4))
# print("I_05:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=5))
# print("I_12:", mutual_info_2ordm(mol, U, orb_a=1, orb_b=2))
# print("I_13:", mutual_info_2ordm(mol, U, orb_a=1, orb_b=3))
# print("I_14:", mutual_info_2ordm(mol, U, orb_a=1, orb_b=4))
# print("I_15:", mutual_info_2ordm(mol, U, orb_a=1, orb_b=5))
# print("I_23:", mutual_info_2ordm(mol, U, orb_a=2, orb_b=3))
# print("I_24:", mutual_info_2ordm(mol, U, orb_a=2, orb_b=4))
# print("I_25:", mutual_info_2ordm(mol, U, orb_a=2, orb_b=5))
# print("I_34:", mutual_info_2ordm(mol, U, orb_a=3, orb_b=4))
# print("I_35:", mutual_info_2ordm(mol, U, orb_a=3, orb_b=5))
# print("I_45:", mutual_info_2ordm(mol, U, orb_a=4, orb_b=5))
# print("E_0:", pure_state_entanglement(mol, U, orb_a=0))
# print("E_1:", pure_state_entanglement(mol, U, orb_a=1))
# print("E_2:", pure_state_entanglement(mol, U, orb_a=2))
# print("E_3:", pure_state_entanglement(mol, U, orb_a=3))
# print("E_4:", pure_state_entanglement(mol, U, orb_a=4))
# print("E_5:", pure_state_entanglement(mol, U, orb_a=5))
print()

# localized orbitals
mol = mol.use_native_orbitals()
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
print("I_01:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=1))
print("I_02:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=2))
print("I_03:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=3))
print("I_04:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=4))
print("I_05:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=5))
print("I_12:", mutual_info_2ordm(mol, U, orb_a=1, orb_b=2))
print("I_13:", mutual_info_2ordm(mol, U, orb_a=1, orb_b=3))
print("I_14:", mutual_info_2ordm(mol, U, orb_a=1, orb_b=4))
print("I_15:", mutual_info_2ordm(mol, U, orb_a=1, orb_b=5))
print("I_23:", mutual_info_2ordm(mol, U, orb_a=2, orb_b=3))
print("I_24:", mutual_info_2ordm(mol, U, orb_a=2, orb_b=4))
print("I_25:", mutual_info_2ordm(mol, U, orb_a=2, orb_b=5))
print("I_34:", mutual_info_2ordm(mol, U, orb_a=3, orb_b=4))
print("I_35:", mutual_info_2ordm(mol, U, orb_a=3, orb_b=5))
print("I_45:", mutual_info_2ordm(mol, U, orb_a=4, orb_b=5))
print("E_0:", pure_state_entanglement(mol, U, orb_a=0))
print("E_1:", pure_state_entanglement(mol, U, orb_a=1))
print("E_2:", pure_state_entanglement(mol, U, orb_a=2))
print("E_3:", pure_state_entanglement(mol, U, orb_a=3))
print("E_4:", pure_state_entanglement(mol, U, orb_a=4))
print("E_5:", pure_state_entanglement(mol, U, orb_a=5))

# Results
# I_01: 2.266688405294746
# I_02: 0.001313393618814107
# I_03: 0.028487634591789046
# I_04: 0.0016249113613469035
# I_05: 0.004341459066112385
# I_12: 0.02849665696812309
# I_13: 0.0014848131604270343
# I_14: 0.0045712965610862355
# I_15: 0.0016710022518791057
# I_23: 1.977253852752264
# I_24: 0.03364218668183305
# I_25: 0.0884184877764409
# I_34: 0.08848774172757157
# I_35: 0.033186372581380574
# I_45: 2.0046383601958526
# E_0: 1.1874108752029549
# E_1: 1.1876593045257167
# E_2: 1.1475388586024167
# E_3: 1.147438880890895
# E_4: 1.1293014282185432
# E_5: 1.1289191816342492