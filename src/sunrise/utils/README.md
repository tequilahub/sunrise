# utils – internal helpers

Internal helpers used across sunrise (not part of the main user API):

- `orbital_transformation.py` – orthogonalization (`orthogonalize`), active/core space extraction, basis transformations
- `transformation_helpers.py` – `FermionicSWAP` for rearranging orbitals in circuits
- `fermionic_utils.py` – fermionic Hamiltonians and excitation generators as OpenFermion operators, and the restricted integrals of the hard-core boson Hamiltonian
- `timing.py` – lightweight timing utilities
- `giuseppe.py` – progress bar 🐟
