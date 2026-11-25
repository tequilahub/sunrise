# measurement – optimized measurements with hard-core bosons

Measuring a molecular Hamiltonian on hardware normally needs many circuit repetitions. Here the Hamiltonian is split into parts that become **hard-core-boson (HCB)**-like after an orbital rotation: each part can be measured in the computational basis with a cheap rotation circuit appended (the *rotators*, typically `UR` layers defined by SPA-type graphs).

Main functions:
- `rotate_and_hcb` – rotate the Hamiltonian and keep its HCB-measurable part
- `get_hcb_part`, `fold_rotators`, `gates_to_orb_rot` – building blocks
- `compute_num_meas` – estimate the number of shots needed

Workflow: pick a set of graphs → build one rotator per graph → measure each rotated HCB part → sum up the residual.

**Background:** [Bincoletto & Kottmann, *A physics-informed measurement protocol for expectation values of fermionic observables*, Digital Discovery 2026](https://doi.org/10.1039/D5DD00251F) (reports 30–80 % fewer measurements than state-of-the-art grouping) · **See also:** [scenario 1](../../../examples/optimized_measurement/example_scenario1.py) · [scenario 2](../../../examples/optimized_measurement/example_scenario2.py) · [tests](../../../tests/test_hcb_measurement.py)
