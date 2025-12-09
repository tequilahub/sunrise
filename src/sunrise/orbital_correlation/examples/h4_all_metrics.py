import tequila as tq
import numpy as np
from sunrise.orbital_correlation.density_matrix_utils import *
from sunrise.orbital_correlation.quantum_info_utils import *

mol = tq.Molecule("H 0 0 0\nH 0 0 1\nH 0 0 2\nH 0 0 3", "sto-3g")
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
print("wavefunction:", tq.simulate(U))

PSSR = False
NSSR = False

# Two orbitals metrics
for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
    print("orbs:", a, b)
    print("I:", two_orbs_mutual_info(mol, U, orb_a=a, orb_b=b, PSSR=PSSR, NSSR=NSSR))
    # print("I (with entropy):", mutual_info_simple(mol, U, orb_a=a, orb_b=b, PSSR=PSSR, NSSR=NSSR)) # This should be equal to I
    print("Q:", two_orbs_quantum_correlation(mol, U, orb_a=a, orb_b=b, PSSR=PSSR, NSSR=NSSR))
    print("E:", two_orbs_entanglement(mol, U, orb_a=a, orb_b=b, PSSR=PSSR, NSSR=NSSR))
    print("C:", two_orbs_classical_correlation(mol, U, orb_a=a, orb_b=b, PSSR=PSSR, NSSR=NSSR))
    # print("Q+C:", two_orbs_quantum_correlation(mol, U, orb_a=a, orb_b=b, PSSR=PSSR, NSSR=NSSR) + two_orbs_classical_correlation(mol, U, orb_a=a, orb_b=b, PSSR=PSSR, NSSR=NSSR)) # This should be close to I
    print()

# One orbital metrics
for a in list(range(mol.n_orbitals)):
    print("orb:", a)
    print("I:", one_orb_mutual_info(mol, U, orb_a=a, PSSR=PSSR, NSSR=NSSR))
    print("E:", one_orb_entanglement(mol, U, orb_a=a, PSSR=PSSR, NSSR=NSSR))
    print()