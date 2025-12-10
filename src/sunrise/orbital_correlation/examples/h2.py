import tequila as tq
import numpy as np
from sunrise.orbital_correlation.density_matrix_utils import *
from sunrise.orbital_correlation.quantum_info_utils import *

mol = tq.Molecule("H 0 0 0\nH 0 0 0.7", "sto-3g")
mol = mol.use_native_orbitals()
H = mol.make_hamiltonian()
U = mol.make_ansatz(name="SPA", edges=[(0,1)])
guess = np.eye(2)
guess[0] = [1.0, 1.0]
guess[1] = [1.0, -1.]
opt = tq.chemistry.optimize_orbitals(mol, circuit=U, initial_guess=guess.T, silent=True)
UR = mol.get_givens_circuit(opt.mo_coeff)
U += UR.dagger()
E = tq.ExpectationValue(U,H)
result = tq.minimize(E, silent=True)
U = U.map_variables(result.variables)
print("localized orbitals")
print(tq.simulate(U))

print("I_01:", mutual_info_simple(mol, U, orb_a=0, orb_b=1))
print("E_0:", one_orb_entanglement(mol, U, orb_a=0))
print("E_1:", one_orb_entanglement(mol, U, orb_a=1))