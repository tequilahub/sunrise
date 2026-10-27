# orbital_correlation – orbital entanglement and correlation

Quantifies how strongly orbitals are correlated in a molecular wavefunction, using **quantum information** measures. From a circuit, the module builds one- and two-orbital reduced density matrices (RDMs) and evaluates entropies, entanglement and mutual information on them. These numbers help to read bonding, pick active spaces and compare localized vs. delocalized orbitals. *(Work in progress.)*

**Main functions**
- `compute_one_orb_rdm`, `compute_two_orb_rdm`, `compute_one_spin_rdm` – orbital RDMs from a molecule + circuit (`density_matrix_utils.py`)
- `one_orb_entanglement`, `two_orbs_entanglement` – orbital entanglement
- `mutual_info_simple`, `two_orbs_mutual_info`, `total_mutual_info` – mutual information \$I_{ab}\$
- `two_orbs_quantum_correlation`, `two_orbs_classical_correlation` – split of the correlation into quantum and classical parts
- `PSSR=True` – apply the particle-number superselection rule, i.e. count only physically accessible entanglement

```python
import sunrise as sun
from sunrise.orbital_correlation.quantum_info_utils import mutual_info_simple, one_orb_entanglement

mol = sun.chemistry.Molecule("H 0 0 0\nH 0 0 0.7", "sto-3g").use_native_orbitals()
U = mol.make_ansatz(name="SPA", edges=[(0, 1)], label=None)
U = U.map_variables({v: 0.2 for v in U.extract_variables()})
print("I_01:", mutual_info_simple(mol, U, orb_a=0, orb_b=1))
print("E_0: ", one_orb_entanglement(mol, U, orb_a=0))
```

**Background:** [Ding et al., *Concept of Orbital Entanglement and Correlation in Quantum Chemistry*, JCTC 2020](https://doi.org/10.1021/acs.jctc.0c00559) · [Tenti et al., *Entanglement and Mutual Information in Molecules: Comparing Localized and Delocalized Orbitals*, JCTC 2024](https://doi.org/10.1021/acs.jctc.4c01101) · [Rissler et al., *Measuring orbital interaction using quantum information theory*, Chem. Phys. 2006](https://doi.org/10.1016/j.chemphys.2005.10.018)

**See also:** [H₂ example](../../../examples/orbital_correlation/h2.py) · [bond stretching](../../../examples/orbital_correlation/h2_with_distance.py) · [π systems (ethylene, butadiene, benzene)](../../../examples/orbital_correlation/benzene_pi_sys.py) · [all metrics on H₄](../../../examples/orbital_correlation/h4_all_metrics.py) · [mutual information plots (notebook)](../../../examples/orbital_correlation/mutual_info_plots.ipynb)
