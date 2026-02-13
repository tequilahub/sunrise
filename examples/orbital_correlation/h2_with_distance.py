# From https://iopscience.iop.org/article/10.1088/2058-9565/aca4ee/meta Table 2
import numpy as np
from sunrise.orbital_correlation.density_matrix_utils import *
from sunrise.orbital_correlation.quantum_info_utils import *

mol = tq.Molecule("H 0 0 0\nH 0 0 1", "sto-3g").use_native_orbitals()

vac    = np.array([1, 0, 0, 0]) # |00> (Index 0)
down   = np.array([0, 1, 0, 0]) # |01> (Index 1)
up     = np.array([0, 0, 1, 0]) # |10> (Index 2)
double = np.array([0, 0, 0, 1]) # |11> (Index 3)

# Ground state (rho)
HF = 1/2 * (np.kron(double,vac) + np.kron(vac,double) + np.kron(up,down) - np.kron(down,up))

cases = [
    (False, False),
    (True, False),
    (False, True)
]

for PSSR, NSSR in cases:

    print(f"PSSR = {PSSR}, NSSR = {NSSR}")

    ######### Compute Hartree-Fock state and rho #########
    # Method 1
    # rho = outer(HF)
    # Method 2
    # rho = create_general_mixed_state([(1, HF)])
    # Method 3
    U = mol.prepare_reference() + mol.make_excitation_gate((0,2), -np.pi/2) + mol.make_excitation_gate((1,3), -np.pi/2)
    rho = compute_two_orb_rdm(mol, U, p_orb=0, q_orb=1, PSSR=PSSR, NSSR=NSSR)

    ######### Uncorrelated state (pi) and Mutual information (I) #########
    o1 = compute_one_orb_rdm(mol, U, one_orb=0) # 1/4 I
    o2 = compute_one_orb_rdm(mol, U, one_orb=1) # 1/4 I
    pi = np.kron(o1,o2) # 1/16 I
    I = quantum_relative_entropy(rho, pi)

    ######### Separable state (sigma) and Entanglement (E) #########
    # Analytical in this case, in general we need optimization
    states = [
        (0.25, np.kron(double,vac)),
        (0.25, np.kron(vac,double)),
        (0.25, np.kron(up,down)),
        (0.25, np.kron(down,up))
    ]
    sigma = create_general_mixed_state(states)
    sigma = change_basis(sigma, 'to_molecular')
    E = quantum_relative_entropy(rho, sigma)

    ######### Classical state (chi) and Quantum correlation (Q) #########
    chi = np.diag(np.diag(rho))
    Q = quantum_relative_entropy(rho, chi)

    ######### Classical state (chi) and Classical correlation (C) #########
    C = quantum_relative_entropy(chi, pi)
    
    # All results
    print(f"I = {I / math.log(2):.1f} × ln(2)")
    print(f"C = {C / math.log(2):.1f} × ln(2)")
    print(f"Q = {Q / math.log(2):.1f} × ln(2)")
    print(f"E = {E / math.log(2):.1f} × ln(2)")

    print()

# Results of Table 2
# PSSR = False, NSSR = False
# I = 4.0 × ln(2)
# C = 2.0 × ln(2)
# Q = 2.0 × ln(2)
# E = 2.0 × ln(2)

# PSSR = True, NSSR = False
# I = 3.0 × ln(2)
# C = 2.0 × ln(2)
# Q = 1.0 × ln(2)
# E = 1.0 × ln(2)

# PSSR = False, NSSR = True
# I = 2.5 × ln(2)
# C = 2.0 × ln(2)
# Q = 0.5 × ln(2)
# E = 0.5 × ln(2)