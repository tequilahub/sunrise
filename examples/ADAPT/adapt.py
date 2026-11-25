import sunrise as sun
from sunrise.ADAPT.adapt import Adapt, MolecularPool

geo = "H 0. 0. 0. \n Li 0. 0. 1."
mol = sun.Molecule(
    geometry=geo, basis_set="sto-3g", nature="fermionic"
)  # it works with all sunrise molecules (qubit, fermionic and hybrid)
pool = MolecularPool(molecule=mol, indices="UpCCSD")
solver = Adapt(
    operator_pool=pool,
    backend="spex",
    molecule=mol,
    optimizer_args={"method": "adam", "silent": True},
)
result = solver()
final_circuit = result.U
print(f"{final_circuit.depth = }")
print(f"{result.energy = }")
print("Steps", len(result.histories))
print("Final circuit:\n ", final_circuit)
