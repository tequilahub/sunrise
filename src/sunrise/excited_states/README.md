# excited_states – variational quantum deflation

Excited states with variational quantum deflation ([VQD, Higgott et al., Quantum 2019](https://doi.org/10.22331/q-2019-07-01-156)): after the ground state $|\psi_0\rangle$ is found, the next state is optimized with the energy penalized by its overlap with the states already found,

$$E = \langle\psi|H|\psi\rangle + \sum_k \beta_k |\langle\psi_k|\psi\rangle|^2 .$$

The weights $\beta_k$ must be larger than the energy gap to the state you look for, otherwise the minimum is the lower state plus the penalty.

- `vqd_projection_method(H, U, circuit_list, constant_list)` – the VQD objective for the circuit `U`, with the circuits of the previous states and their weights. Qubit circuits only.

```python
import tequila as tq
import sunrise as sun

mol = sun.Molecule(geometry="H 0 0 0\nH 0 0 0.75", basis_set="sto-3g")
H = mol.make_hamiltonian()

U0 = mol.make_ansatz("UpCCD", label=0)
E0 = tq.minimize(tq.ExpectationValue(H=H, U=U0), silent=True)
U0 = U0.map_variables(E0.variables)  # fixed ground state

U1 = mol.make_ansatz("UpCCD", label=1)
E1 = sun.excited_states.vqd_projection_method(
    H, U1, circuit_list=[U0], constant_list=[5.0]
)
E1 = tq.minimize(E1, silent=True, initial_values=1.0)
print(E0.energy, E1.energy)  # -1.137, 0.460 (doubly excited singlet)
```
