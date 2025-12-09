import tequila as tq
import numpy as np
from sunrise.orbital_correlation.density_matrix_utils import *
from sunrise.orbital_correlation.quantum_info_utils import *

# canonical orbitals
mol = tq.Molecule("H 0 0 0\nH 0 0 1.5\nH 0 0 3\nH 0 0 4.5", "sto-3g")
# print(mol.integral_manager.orbital_coefficients)
H = mol.make_hamiltonian()
U = mol.prepare_reference()
print(tq.simulate(U))
print("I_01:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=1))
print("I_02:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=2))
print("I_03:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=3))
print("I_12:", mutual_info_2ordm(mol, U, orb_a=1, orb_b=2))
print("I_13:", mutual_info_2ordm(mol, U, orb_a=1, orb_b=3))
print("I_23:", mutual_info_2ordm(mol, U, orb_a=2, orb_b=3))
print("E_0:", pure_state_entanglement(mol, U, orb_a=0))
print("E_1:", pure_state_entanglement(mol, U, orb_a=1))
print("E_2:", pure_state_entanglement(mol, U, orb_a=2))
print("E_3:", pure_state_entanglement(mol, U, orb_a=3),'\n')

"""
# SPA orbitals
mol = mol.use_native_orbitals() # localized orbitals
H = mol.make_hamiltonian()
U = mol.make_ansatz(name="SPA", edges=[(0,1),(2,3)])
guess = np.eye(4)
guess[0] = [1.0, 1.0, 0.0, 0.0]
guess[1] = [1.0, -1., 0.0, 0.0]
guess[2] = [0.0, 0.0, 1.0, 1.0]
guess[3] = [0.0, 0.0, 1.0, -1.]
opt = tq.chemistry.optimize_orbitals(mol, circuit=U, initial_guess=guess.T, silent=True)
mol = opt.molecule
H = mol.make_hamiltonian()
E = tq.ExpectationValue(U,H)
result = tq.minimize(E, silent=True)
U = U.map_variables(result.variables)
print(tq.simulate(U))
print("I_01:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=1))
print("I_02:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=2))
print("I_03:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=3))
print("I_12:", mutual_info_2ordm(mol, U, orb_a=1, orb_b=2))
print("I_13:", mutual_info_2ordm(mol, U, orb_a=1, orb_b=3))
print("I_23:", mutual_info_2ordm(mol, U, orb_a=2, orb_b=3))
print("E_0:", pure_state_entanglement(mol, U, orb_a=0))
print("E_1:", pure_state_entanglement(mol, U, orb_a=1))
print("E_2:", pure_state_entanglement(mol, U, orb_a=2))
print("E_3:", pure_state_entanglement(mol, U, orb_a=3))
print()
"""

# localized orbitals
mol = mol.use_native_orbitals()
H = mol.make_hamiltonian()
U = mol.make_ansatz(name="SPA", edges=[(0,1),(2,3)])
guess = np.eye(4)
guess[0] = [1.0, 1.0, 0.0, 0.0]
guess[1] = [1.0, -1., 0.0, 0.0]
guess[2] = [0.0, 0.0, 1.0, 1.0]
guess[3] = [0.0, 0.0, 1.0, -1.]
opt = tq.chemistry.optimize_orbitals(mol, circuit=U, initial_guess=guess.T, silent=True)
UR = mol.get_givens_circuit(opt.mo_coeff)
U += UR.dagger()
E = tq.ExpectationValue(U,H)
result = tq.minimize(E, silent=True)
U = U.map_variables(result.variables)
print(tq.simulate(U))
print("I_01:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=1))
print("I_02:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=2))
print("I_03:", mutual_info_2ordm(mol, U, orb_a=0, orb_b=3))
print("I_12:", mutual_info_2ordm(mol, U, orb_a=1, orb_b=2))
print("I_13:", mutual_info_2ordm(mol, U, orb_a=1, orb_b=3))
print("I_23:", mutual_info_2ordm(mol, U, orb_a=2, orb_b=3))
print("E_0:", pure_state_entanglement(mol, U, orb_a=0))
print("E_1:", pure_state_entanglement(mol, U, orb_a=1))
print("E_2:", pure_state_entanglement(mol, U, orb_a=2))
print("E_3:", pure_state_entanglement(mol, U, orb_a=3))