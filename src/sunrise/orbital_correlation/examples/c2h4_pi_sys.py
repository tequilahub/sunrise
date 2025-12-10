import sunrise as sun
import tequila as tq
import numpy as np
from sunrise.orbital_correlation.density_matrix_utils import *
from sunrise.orbital_correlation.quantum_info_utils import *
from sunrise.fermionic_operations.givens_rotations import get_givens_circuit

# my geometry
# geometry = """
# C	0.0000000	0.0000000	0.6724790
# C	0.0000000	0.0000000	-0.6724790
# H	0.0000000	0.9344670	1.2552450
# H	0.0000000	-0.9344670	1.2552450
# H	0.0000000	-0.9344670	-1.2552450
# H	0.0000000	0.9344670	-1.2552450
# """

# Schilling geometry
geometry = """
C 0.669500 0.000000 0.000000
C -0.669500 0.000000 0.000000
H 1.232100 0.928900 0.000000
H 1.232100 -0.928900 0.000000
H -1.232100 0.928900 0.000000
H -1.232100 -0.928900 0.000000
"""

# We select only orbitals 7 and 8 (pi system), so it behaves like H2
mol = tq.Molecule(geometry=geometry, basis_set='sto-3g', active_orbitals=[7,8]).use_native_orbitals()
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

# I_01: 2.523046605642772
# E_0: 1.261523302821387
# E_1: 1.261523302821387