# Scenario: SALC + Spin Symmetrization for H2 in the localized basis

from pandas import DataFrame
import tequila as tq
import sunrise as sun

from sunrise.symmetry_adaptation import FockSpaceState, LocalizedIrrepProvider, SymmetryAdaptedLinearCombintationSymmetrization, SpinSymmetrizationProcedure

# Create the molecule and the point group
mol = tq.Molecule(geometry="H 0. 0. 0. \n H 0. 0. 0.74804",basis_set='sto-3g', backend="pyscf")
pg = sun.symmetry_adaptation.PointGroup.from_pyscf("D2h")

# Create an irrep provider to indicate that the FockSpaceStates
# are in the localized/naturalized orbital basis corresponding to
# mol.use_native_orbitals() to provide the irrep labels for the states
provider_canonical = LocalizedIrrepProvider(mol, pg)

# Create a list of non-ionic states for the molecule
states_list: list[FockSpaceState] = FockSpaceState.non_ionic_states(mol, provider_canonical)

# Symmetrize the states with respect to SALCs
SALC_symm_list: list[FockSpaceState] = SymmetryAdaptedLinearCombintationSymmetrization(mol, pg).symmetrize(states_list)

# Symmetrize the states with respect to spin
# Note that the spin symmetrization is intended to
# be applied after the SALC symmetrization
spin_symm_list: list[FockSpaceState] = SpinSymmetrizationProcedure(mol).symmetrize(SALC_symm_list)

# Display the states as a DataFrame
FockSpaceState.dataframe(spin_symm_list)