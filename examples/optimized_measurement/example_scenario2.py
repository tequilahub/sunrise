import tequila as tq
import sunrise as sun
import numpy as np

# Create the molecule
mol = sun.chemistry.Molecule(geometry="h 0.0 0.0 0.0\nh 0.0 0.0 1.5\nh 0.0 0.0 3.0\nh 0.0 0.0 4.5", basis_set="sto-3g").use_native_orbitals()
fci = mol.compute_energy("fci")
H = mol.make_hamiltonian()

# Create circuit
U0 = mol.make_ansatz(name="SPA", edges=[(0,1),(2,3)])
UR1 = mol.UR(0,1,angle=np.pi/2) + mol.UR(2,3,angle=np.pi/2) + mol.UR(0,3,angle=-0.2334) + mol.UR(1,2,angle=-0.2334)

UR2 = mol.UR(1,2,angle=np.pi/2) + mol.UR(0,3,angle=np.pi/2)
UR2+= mol.UR(0,1,angle="x") + mol.UR(0,2,angle="y") + mol.UR(1,3,angle="xx") + mol.UR(2,3,angle="yy") + mol.UR(1,2,angle="z") + mol.UR(0,3,angle="zz")
UC2 = mol.UC(1,2,angle="b") + mol.UC(0,3,angle="c")
U = U0 + UR1.dagger() + UR2 + UC2 + UR2.dagger()

variables = {((0, 1), 'D', None): -0.644359150621798, ((2, 3), 'D', None): -0.644359150621816, "x": 0.4322931478168998, "y": 4.980327764918099e-14,
             "xx": -3.07202211218271e-14, "yy": 0.7167447375727501, "z": -3.982666230146327e-14, "zz": 1.2737831353027637e-13, "c": -0.011081251246998072,
             "b": 0.49719805420976604}
E = tq.ExpectationValue(H=H, U=U)
full_energy = tq.simulate(E, variables=variables)
print(f"Energy error: {(full_energy-fci)*1000:.2f} mE_h\n")

# Create rotators
graphs = [
    [(0,1),(2,3)],
    [(0,3),(1,2)],
    [(0,2),(1,3)]
]
rotators = []
for graph in graphs:
    UR = tq.QCircuit()
    for edge in graph:
        UR += mol.UR(edge[0], edge[1], angle=np.pi/2)
    rotators.append(UR)

# Apply the measurement protocol
result = sun.measurement.rotate_and_hcb(molecule=mol, circuit=U, variables=variables, rotators=rotators, target=full_energy, silent=False)
print(result) # the list of HCB molecules to measure and the residual element discarded

# Compute the energy
energy = 0
for i,hcb_mol in enumerate(result[0]):
    expval = tq.ExpectationValue(U=U+rotators[i], H=hcb_mol.make_hamiltonian())
    energy += tq.simulate(expval, variables=variables)

print(f"Energy of the accumulated HCB contributions: {energy}")
print(f"Error: {energy-full_energy}")