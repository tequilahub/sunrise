

import tequila as tq
import sunrise as sun
from sunrise.MCVBT.GNM import mcvbt
import time
import warnings

warnings.filterwarnings("ignore", category=tq.TequilaWarning)

solver = "qulacs"
# solver = "spex"
silent = False

#define molecule
geometry = "H 1.5 0.0 0.0\nH 0.0 0.0 0.0\nH 1.5 0.0 1.5\nH 0.0 0.0 1.5"
mol = sun.chemistry.Molecule(geometry=geometry, basis_set="sto-6g")
mol = mol.use_native_orbitals()

#define edges
h4_local = [[(0, 1), (2, 3)], [(0, 3), (1, 2)], [(0, 2), (1, 3)]]
h4_delocal = [[(0, 1), (2, 3)], [(0, 3), (1, 2)], [(0, 2), (1, 3)]]

#run MCVBT with FQE solver
start = time.time()
filename = "MCVBT_test_local"
mcvbt_loc = mcvbt(mol=mol, graphs=h4_local, solver=solver, strategy=None, filename=filename, silent=silent)
mcvbt_loc.calculate_groundstate(init_strategy="pre-optimize")
end = time.time()
print(f"{solver} Time: {end-start}")
mcvbt_loc.compare_to_fci()
#run again with delocalization
start = time.time()
filename = "MCVBT_test_delocal"
mcvbt_deloc = mcvbt(mol=mol, graphs=h4_delocal, solver=solver, strategy="shift", filename=filename, silent=silent)
mcvbt_deloc.calculate_groundstate(init_strategy="pre-optimize")
end = time.time()
print(f"{solver} Time: {end-start}")
mcvbt_deloc.compare_to_fci()

#run with different initialization strategy
start = time.time()
filename = "MCVBT_test_local"
mcvbt_loc = mcvbt(mol=mol, graphs=h4_local, solver=solver, strategy=None, filename=filename, silent=silent)
mcvbt_loc.calculate_groundstate(init_strategy="random")
end = time.time()
print(f"{solver} Time: {end-start}")
mcvbt_loc.compare_to_fci()
