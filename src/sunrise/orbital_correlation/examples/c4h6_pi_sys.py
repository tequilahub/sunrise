import sunrise as sun
import tequila as tq
import numpy as np
from sunrise.orbital_correlation.density_matrix_utils import *
from sunrise.orbital_correlation.quantum_info_utils import *
from sunrise.fermionic_operations.givens_rotations import get_givens_circuit

geometry = """
C	-0.0000030	0.7547250	0.5821770
C	0.0000030	-0.7547250	0.5821770
C	0.0000030	1.5440700	-0.5171380
C	-0.0000030	-1.5440700	-0.5171380
H	0.0000060	1.2292030	1.5765230
H	-0.0000060	-1.2292030	1.5765230
H	0.0000150	2.6390550	-0.4323650
H	-0.0000090	1.1293150	-1.5343900
H	-0.0000150	-2.6390550	-0.4323650
H	0.0000090	-1.1293150	-1.5343900
"""

# We select only the pi system, so it behaves like H4
mol = tq.Molecule(geometry=geometry, basis_set='sto-3g', active_orbitals=[13,14,15,16]).use_native_orbitals()
reorder = np.array([ # reorder the orbitals because when doing natives they get mixed
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])
mol = mol.transform_orbitals(reorder.T)
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
print("I_01:", mutual_info_simple(mol, U, orb_a=0, orb_b=1))
print("I_02:", mutual_info_simple(mol, U, orb_a=0, orb_b=2))
print("I_03:", mutual_info_simple(mol, U, orb_a=0, orb_b=3))
print("I_12:", mutual_info_simple(mol, U, orb_a=1, orb_b=2))
print("I_13:", mutual_info_simple(mol, U, orb_a=1, orb_b=3))
print("I_23:", mutual_info_simple(mol, U, orb_a=2, orb_b=3))
print("E_0:", one_orb_entanglement(mol, U, orb_a=0))
print("E_1:", one_orb_entanglement(mol, U, orb_a=1))
print("E_2:", one_orb_entanglement(mol, U, orb_a=2))
print("E_3:", one_orb_entanglement(mol, U, orb_a=3))

#            standard              reordered           reordered transpose
# I_01: 1.4857069199214674     2.4092481765539833      2.489804417842751
# I_02: 0.5199083273507705     0.01687001688015588     0.0005862860106118362
# I_03: 0.3354593129962691     0.12753165303786718     0.013639630436490968
# I_12: 0.3371251452889372     0.12727361318661012     0.01458612451583452
# I_13: 0.5168240142828093     0.0164646194527025      0.0005732102587865384
# I_23: 1.3920017056289882     2.4095315866381513      2.483671474528226