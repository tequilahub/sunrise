# Scenario: Spin Symmetrization for H2 in the canonical basis

import tequila as tq
import sunrise as sun

from sunrise.symmetry_adaptation import PointGroup, FockSpaceState, PySCFCanonicalIrrepProvider,SpinSymmetrizationProcedure

# Create the molecule and the point group
mol = tq.Molecule(geometry="H 0. 0. 0. \n H 0. 0. 0.74804",basis_set='sto-3g', backend="pyscf")
pg: PointGroup = sun.symmetry_adaptation.PointGroup.from_pyscf("D2h")

# Create an irrep provider to indicate that the FockSpaceStates
# are in the canonical orbital basis and to provide the irrep labels for the states
provider_canonical = PySCFCanonicalIrrepProvider(mol, pg)

# Create a list of non-ionic states for the molecule
states_list: list[FockSpaceState] = FockSpaceState.non_ionic_states(mol, provider_canonical)

# Symmetrize the states with respect to spin
symm_list: list[FockSpaceState] = SpinSymmetrizationProcedure(mol).symmetrize(states_list)

# Display the states as a DataFrame
FockSpaceState.dataframe(symm_list)
