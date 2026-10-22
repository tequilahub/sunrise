import sunrise as sn
from sunrise.CLPO.orbital_transformation import (
    generate_CLPO_molecule_edges,
    generate_HAO_molecule,
)

basis = "sto-3g"
geo = "N 0. 0. 0. \n N 0. 0. 1.098"
nmol = sn.Molecule(geometry=geo, basis_set=basis, backend="pyscf", units="a")

nmol, edges = generate_CLPO_molecule_edges(nmol, silent=True)
sn.plot_MO(molecule=nmol, filename="N2_CLPO")
print("N2 Edges ", edges)

U = nmol.make_ansatz("HCB-SPA", edges=edges)
opt = sn.optimize_orbitals(nmol, circuit=U, silent=True, use_hcb=True)
sn.plot_MO(opt.molecule, filename="N2_SPA")

geo = "H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2. \n H 0. 0. 3."
mol = sn.Molecule(geometry=geo, basis_set=basis, backend="pyscf", units="a")

# If no edges are provided, they are extracted from the Hartree Fock calculation and the geometry
mol1, edges = generate_CLPO_molecule_edges(mol, silent=True)
print("Automatic Edges: ", edges)
sn.plot_MO(mol1, "1st_graph")

# However, one can also generate custom orbitals by providing the desired molecule conectivity.
# Please NOTE that the eddges you provide (based on the atom conectivity) and the ones returned
# (based on orbital pairing) may be different
mol2, edges = generate_CLPO_molecule_edges(mol, edges=[(1, 2), (0, 3)], silent=True)
print("Custom Edges: ", edges)
sn.plot_MO(mol2, "2n_graph")

geometry = """
C 0.00000 0.00000 0.00000
C 1.48460 0.00000 0.00000
C -0.76837 0.00000 -1.12008
C 2.25297 0.00000 -1.12008
H -0.47297 -0.00038 0.97971
H 1.95470 -0.00556 0.97953
H -1.85042 -0.00018 -1.03002
H -0.35993 -0.00000 -2.12298
H 3.33365 -0.01193 -1.02751
H 1.85521 0.00845 -2.12564"""

mol = sn.Molecule(geometry=geometry, basis_set="sto-3g", backend="pyscf", nature="q")

mol3 = generate_HAO_molecule(mol, silent=True)
sn.plot_MO(mol3, filename="HAO", orbital=[2, 6, 10, 14])

# One can also provide only part of the molecule, the rest will be autocomplete, similar as if
# no edges are provided.
mol1, edges = generate_CLPO_molecule_edges(
    mol, edges=[(2, 3), (0, 1), (0, 1)], silent=True, use_active=True
)
sn.plot_MO(mol1, filename="CLPO", orbital=[4, 5, 14, 15])
print("Edges ", edges)

import sunrise.CLPO.orbital_transformation as ot

print("Loading Sunrise from:", ot.__file__)
