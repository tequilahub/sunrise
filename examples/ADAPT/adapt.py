import sunrise as sun
from sunrise.ADAPT.adapt import MolecularPool,Adapt

geo = 'H 0. 0. 0. \n Li 0. 0. 1.'
mol = sun.Molecule(geometry=geo,basis_set='sto-3g',nature='f')
pool = MolecularPool(molecule=mol,indices='UpCCSD')
solver = Adapt(operator_pool=pool,backend='tcc',molecule=mol, optimizer_args={'method':'adam','silent': True})
result = solver()
final_circuit = mol.prepare_reference() + result.U
print(f"{final_circuit.depth = }") 
print(f"{result.energy = }")
print('Steps', len(result.histories))
