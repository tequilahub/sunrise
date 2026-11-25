# Scenario: Spin Symmetrization for H2 in the canonical basis

import sunrise as sun
from sunrise.symmetry_adaptation import (
    FockSpaceState,
    IrrepAssigner,
    SpinSymmetrizationProcedure,
)

# Create the molecule, its point group (D2h) is detected from the geometry
mol = sun.chemistry.Molecule(
    geometry="H 0. 0. 0. \n H 0. 0. 0.74804", basis_set="sto-3g", backend="pyscf"
)

# Create an irrep assigner to indicate that the FockSpaceStates
# are in the canonical orbital basis and to provide the irrep labels for the states
assigner_pyscf = IrrepAssigner(mol, "pyscf")

# Create a list of non-ionic states for the molecule
states_list: list[FockSpaceState] = FockSpaceState.non_ionic_states(mol, assigner_pyscf)

# Symmetrize the states with respect to spin
symm_list: list[FockSpaceState] = SpinSymmetrizationProcedure(mol).symmetrize(
    states_list
)

# Display the states as a DataFrame
FockSpaceState.dataframe(symm_list)
