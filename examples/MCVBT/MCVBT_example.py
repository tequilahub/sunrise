

import tequila as tq
from sunrise.MCVBT.GNM import mcvbt

import time

import warnings
warnings.filterwarnings("ignore", category=tq.TequilaWarning)


#define molecule
geometry = "H 1.5 0.0 0.0\nH 0.0 0.0 0.0\nH 1.5 0.0 1.5\nH 0.0 0.0 1.5"
mol = tq.Molecule(geometry=geometry, basis_set="sto-6g")
mol = mol.use_native_orbitals()

#define edges
h4_local = [[(0, 1), (2, 3)], [(0, 3), (1, 2)], [(0, 2), (1, 3)]]
h4_delocal = [[(0, 1), (2, 3)], [(0, 3), (1, 2)], [(0, 2), (1, 3)]]

#run MCVBT with FQE solver
start = time.time()
filename = "MCVBT_test_local"
fqe_loc = mcvbt(mol=mol, graphs=h4_local, solver="FQE",strategy=None, filename=filename)
fqe_loc.calculate_groundstate(init_strategy="pre-optimize")
end = time.time()
print(f"FQE Time: {end-start}")
fqe_loc.compare_to_fci()

#run again with delocalization
start = time.time()
filename = "MCVBT_test_delocal"
fqe_deloc = mcvbt(mol=mol, graphs=h4_delocal, solver="FQE",strategy="shift", filename=filename)
fqe_deloc.calculate_groundstate(init_strategy="pre-optimize")
end = time.time()
print(f"FQE Time: {end-start}")
fqe_deloc.compare_to_fci()

#run with different initialization strategy
start = time.time()
filename = "MCVBT_test_local"
fqe_loc = mcvbt(mol=mol, graphs=h4_local, solver="FQE",strategy=None, filename=filename)
fqe_loc.calculate_groundstate(init_strategy="random")
end = time.time()
print(f"FQE Time: {end-start}")
fqe_loc.compare_to_fci()

