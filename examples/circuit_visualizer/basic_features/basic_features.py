import tequila as tq
import sunrise as sun
import numpy as np
from math import pi

# --- 1. Basic Circuit Setup & Reference State ---
geom = "H 0 0 0\nH 0 0 1\nH 0 0 2\nH 0 0 3"
snmol = sun.Molecule(geometry=geom, basis_set='sto-3g', nature='f', units='a')
circuit = sun.FCircuit()
circuit += snmol.prepare_reference()

# --- 2. Gate Types Showcase ---
# Single, Paired Double, and Unpaired Double Excitations
circuit += sun.gates.FermionicExcitation(indices=[(1, 7)], variables="a")          # Single
circuit += sun.gates.FermionicExcitation(indices=[(0, 4), (1, 7)], variables=2.0)  # Paired Double
circuit += sun.gates.FermionicExcitation(indices=[(0, 2), (3, 5)], variables="b")  # Unpaired Double

# Special Fermionic Gates
circuit += sun.gates.UR(0, 1, variables="c")  # Orbital Rotator
circuit += sun.gates.UC(1, 3, variables="d")  # Pair Correlator

# Raw Tequila Gate
circuit += tq.gates.Y([0, 3])

# --- 3. Basic Spatial Orbitals View ---
circuit.export_to("basic_spatial_view.pdf", show_spatial_orbitals=True)

# --- 4. Hybrid Wire Selection (Bosonic vs Fermionic) ---
select = {0: "B", 1: "F", 2: "F", 3: "B"}
circuit.export_to("basic_hybrid_spatial.pdf", show_spatial_orbitals=True, select=select)

# --- 5. Variable Assignment & Color Ranges ---
# Plot the symbolic circuit (triggers parametrized marking)
circuit.export_to("basic_symbolic.pdf", style={'parametrized': 'blue'})

# Map variables to numeric values and plot with a color range
variables = {"a": 0, "b": pi / 4, "c": pi / 2, "d": pi}
numeric_circuit = circuit.map_variables(variables)
numeric_circuit.export_to("basic_numeric.pdf", style={'color_range': ["#00FF6A", 'red']})

# --- 6. Spatial vs Spin Views ---
circuit.export_to("basic_spin_view.pdf", show_spatial_orbitals=False)