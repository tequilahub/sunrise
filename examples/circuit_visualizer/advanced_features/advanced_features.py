import tequila as tq
import sunrise as sun
import numpy as np
import random
from datetime import datetime

# --- 1. Chemistry Ansatz & Personalized Styling ---
geom_BeH2 = "H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"
snmol_BeH2 = sun.Molecule(geometry=geom_BeH2, basis_set='sto-3g', nature='f', units='a')

random.seed(datetime.now().timestamp())
snU = snmol_BeH2.make_ansatz('UpCCSD', spin_adapt_singles=False)

# Plot with personalized gate styling
snU.export_to('advanced_upccsd_styled.pdf', style={'single': 'red', 'generic': "#AC61AC"})

# --- 2. Manual Initial States ---
manual_circuit = sun.FCircuit()
# Manually define an initial state using raw X gates
manual_circuit.initial_state = tq.gates.X([0, 2, 4, 6])
manual_circuit += sun.gates.UR(0, 1, variables="a") + sun.gates.UR(2, 3, variables="a")
manual_circuit += sun.gates.UC(1, 2, variables="a")
manual_circuit.export_to("advanced_manual_state.pdf")

# --- 3. Mixed Ordering Circuits ---
geom = "H 0 0 0\nH 0 0 1\nH 0 0 2\nH 0 0 3"
snmol = sun.Molecule(geometry=geom, basis_set='sto-3g', nature='f', units='a')
mixed_circuit = sun.FCircuit()
mixed_circuit += snmol.prepare_reference()

# Segment 1: udud (default)
mixed_circuit += sun.gates.FermionicExcitation(indices=[(0, 4), (1, 7)], variables=2)
mixed_circuit += sun.gates.UR(0, 1, variables=3)

# Segment 2: uudd (reordered=True) - simulates post-fswap or convention change
mixed_circuit += sun.gates.FermionicExcitation(indices=[(0, 1), (4, 5)], variables=5, reordered=True)

# The export_to method automatically detects the mixed ordering and unifies it!
mixed_circuit.export_to("advanced_mixed_spin.pdf", show_spatial_orbitals=False)
mixed_circuit.export_to("advanced_mixed_spatial.pdf", show_spatial_orbitals=True)

# --- 4. Complex Raw Gates & Consecutive Merging ---
mol_H2 = tq.Molecule(geometry="H 0. 0. 0. \n H 0. 0. 1. ", basis_set="sto-3g")
raw_circuit = sun.FCircuit()

# Consecutive raw gates merge into single generic boxes
raw_circuit += tq.gates.X([0, 1, 2, 3])
raw_circuit += tq.gates.Y([2, 3, 4, 5])

# Trotterized gate kept as a generic box
raw_circuit += tq.gates.Trotterized(generator=mol_H2.make_excitation_generator(indices=[(0, 2)]), angle="d")

raw_circuit.export_to("advanced_raw_gates.pdf")