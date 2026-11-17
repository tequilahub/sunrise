import numpy as np
import openfermion as of
import scipy
import tequila as tq

import sunrise as sun

# Create the molecule
mol = sun.chemistry.Molecule(
    geometry="h 0.0 0.0 0.0\nh 0.0 0.0 1.5\nh 0.0 0.0 3.0\nh 0.0 0.0 4.5",
    basis_set="sto-3g",
).use_native_orbitals()
fci = mol.compute_energy("fci")
H = mol.make_hamiltonian()

# Create true wave function
Hof = H.to_openfermion()
Hsparse = of.linalg.get_sparse_operator(Hof)
v, vv = scipy.sparse.linalg.eigsh(Hsparse, sigma=fci)
wfn = tq.QubitWaveFunction.from_array(vv[:, 0])
energy = wfn.inner(H * wfn).real

# Create rotators
graphs = [[(0, 1), (2, 3)], [(0, 3), (1, 2)], [(0, 2), (1, 3)]]
rotators = []
for graph in graphs:
    UR = tq.QCircuit()
    for edge in graph:
        UR += mol.UR(edge[0], edge[1], angle=np.pi / 2)
    rotators.append(UR)

# Apply the measurement protocol
result = sun.measurement.rotate_and_hcb(
    molecule=mol, rotators=rotators, target=fci, initial_state=wfn, silent=False
)
print(
    result
)  # the HCB molecules, the discarded residual and the circuits to measure them with

# Compute the energy
energy = 0
hcb_mols, res_mol, circuits = result
for hcb_mol, circuit in zip(hcb_mols, circuits):
    expval = tq.ExpectationValue(U=circuit, H=hcb_mol.make_hamiltonian())
    energy += tq.simulate(expval, initial_state=wfn)

print(f"Energy of the accumulated HCB contributions: {energy}")
print(f"Error: {energy - fci}")
