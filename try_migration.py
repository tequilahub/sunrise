"""Quick manual check of the qubit-chemistry migration.

Run with:  python try_migration.py
(scratch file - not part of the package, delete when done)
"""

import numpy

import sunrise as sun

GEOM = "H 0.0 0.0 0.0\nHe 0.0 0.0 1.3\nH 0.0 0.0 2.6"


def section(title):
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


section("1. the module moved into sunrise")
print(f"sun.chemistry        -> {sun.chemistry.__name__}")
print(f"sun.quantumchemistry -> {sun.quantumchemistry.__name__}   (back-compat alias)")
print(f"installed backends   -> {list(sun.chemistry.INSTALLED_QCHEMISTRY_BACKENDS)}")
print("\ntop-level names now exported by sunrise:")
for name in ("Molecule", "MoleculeFromOpenFermion", "MoleculeFromTequila",
             "QuantumChemistryBase", "ParametersQC", "NBodyTensor"):
    print(f"  sun.{name:<24} {'ok' if hasattr(sun, name) else 'MISSING'}")

print("\nnothing is imported from tequila.quantumchemistry any more:")
import tequila  # noqa: E402
print(f"  tequila still provides the core: {tequila.QCircuit.__module__}")


section("2. nature= dispatch ('qubit' is canonical, 'tequila' still works)")
for nature in ("qubit", "q", "tequila", "t", "hybrid", "fermionic"):
    mol = sun.Molecule(geometry=GEOM, basis_set="sto-3g", nature=nature, backend="pyscf")
    print(f"  nature={nature:<9} -> {type(mol).__module__.split('.')[-1]}.{type(mol).__name__}")

try:
    sun.Molecule(geometry=GEOM, basis_set="sto-3g", nature="nonsense")
except Exception as exc:
    print(f"  nature='nonsense' -> {type(exc).__name__}: {exc}")


section("3. use_native_orbitals is now shared by all three molecule types")
print("reference = a molecule built with the orbitals frozen explicitly;")
print("we check use_native_orbitals(core=...) reproduces its spectrum.\n")
print(f"  {'core':<10}{'qubit spectrum':<18}{'n_orbitals (qubit/fermionic/hybrid)'}")

for core in ([], [0], [1], [2], [0, 1], [1, 2], [0, 2]):
    reference = sun.chemistry.Molecule(
        geometry=GEOM, units="angstrom", basis_set="sto-3g",
        frozen_core=False, frozen_orbitals=core,
    )
    expected = numpy.linalg.eigvalsh(reference.make_hamiltonian().to_matrix())

    counts = []
    match = None
    for nature in ("qubit", "fermionic", "hybrid"):
        mol = sun.Molecule(
            geometry=GEOM, units="angstrom", basis_set="sto-3g",
            frozen_core=False, nature=nature,
        ).use_native_orbitals(core=core)
        counts.append(str(mol.n_orbitals))
        if nature == "qubit":
            got = numpy.linalg.eigvalsh(mol.make_hamiltonian().to_matrix())
            match = numpy.allclose(expected, got)

    flag = "match" if match else "MISMATCH"
    note = "" if core == sorted(core)[: len(core)] == list(range(len(core))) else "  <- non-contiguous"
    print(f"  {str(core):<10}{flag:<18}{'/'.join(counts)}{note}")

print("\n(core=[1], [2], [1,2], [0,2] are the non-contiguous cases that used to")
print(" raise LinAlgError in fermionic/hybrid and were never reachable at all")
print(" because an assertion fired first.)")


section("4. the other code paths")
mol = sun.Molecule(geometry=GEOM, basis_set="sto-3g", frozen_orbitals=[0])
res = mol.use_native_orbitals()
frozen = [o.idx_total for o in res.integral_manager.orbitals if o.idx is None]
print(f"  inherited active space      -> n_orb={res.n_orbitals}, frozen={frozen}")

mol = sun.Molecule(geometry=GEOM, basis_set="sto-3g", frozen_core=False)
res = mol.use_native_orbitals(active=[0, 2])
frozen = [o.idx_total for o in res.integral_manager.orbitals if o.idx is None]
print(f"  active=[0,2] (core derived) -> n_orb={res.n_orbitals}, frozen={frozen}")

mol = sun.Molecule(geometry=GEOM, basis_set="sto-3g", frozen_core=False)
print(f"  inplace=True returns self   -> {mol.use_native_orbitals(core=[0], inplace=True) is mol}")


section("5. a real calculation, to be sure nothing is subtly broken")
h2 = sun.Molecule(geometry="H 0.0 0.0 0.0\nH 0.0 0.0 0.7", basis_set="sto-3g", backend="pyscf")
print(f"  HF  = {h2.compute_energy('hf'):.10f}")
print(f"  FCI = {h2.compute_energy('fci'):.10f}")

print("\nAll done.\n")
