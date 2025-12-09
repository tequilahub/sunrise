# From https://iopscience.iop.org/article/10.1088/2058-9565/aca4ee/meta
import tequila as tq
import numpy as np
from sunrise.orbital_correlation.density_matrix_utils import *
from sunrise.orbital_correlation.quantum_info_utils import *

### Prove Eq.36 and Table 2
mol = tq.Molecule("H 0 0 0\nH 0 0 0.7","sto-3g").use_native_orbitals()
# Hartree-Fock state in atomic orbitals (Eq.35)
HF = mol.prepare_reference() + mol.make_excitation_gate((0,2), -np.pi/2) + mol.make_excitation_gate((1,3), -np.pi/2)
# Mutual information with subsystems (I)
I = mutual_info_2ordm(mol, HF, orb_a=0, orb_b=1, PSSR=False, NSSR=False)
print(f"Mutual information: {(I/np.log(2)).round(2)} ln2")
# No PSSR: I = 4ln2
# P-SSR:   I = 3ln2
# N-SSR:   I = 5/2ln2
# Entanglement with subsystems (E)
E = pure_state_entanglement(mol, HF, orb_a=0, orb_b=1, PSSR=False, NSSR=False)
print(f"Entanglement: {(E/np.log(2)).round(2)} ln2")
# No SSR: E = 2ln2
# P-SSR:  E = ln2
# N-SSR:  E = 1/2ln2
print()

### Prove Table 3
# Bond order = 1/2 (N_bond - N_antibond)
# Different bonding orders
U1 = tq.gates.X(0) + mol.make_excitation_gate((0,2), -np.pi/2) # Psi1 (Eq.43)
U2 = mol.prepare_reference() + mol.make_excitation_gate((0,2), -np.pi/2) + mol.make_excitation_gate((1,3), -np.pi/2) # Psi2 (Eq.43)
U3 = tq.gates.X([0,1,2]) + mol.make_excitation_gate((1,3), -np.pi/2) # Psi3 (Eq.43)
U4 = tq.gates.X([0,1,2,3])
# Compute entanglement (E)
E = pure_state_entanglement(mol, U1, orb_a=0, orb_b=1, PSSR=False, NSSR=False)
print(f"Entanglement: {(E/np.log(2)).round(2)} ln2")
# E(Psi1) = ln2
# E(Psi2) = 2ln2
# E(Psi3) = ln2
# E(Psi4) = 0