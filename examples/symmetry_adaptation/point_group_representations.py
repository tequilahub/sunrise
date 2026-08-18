# Scenario: Building different Point Group Representations for H2 in the canonical basis

import tequila as tq
import sunrise as sun
import numpy as np

from sunrise.symmetry_adaptation import PointGroup, QCircuitRepresentationBuilder, FockSpaceState, IrrepProvider, SpinSymmetrizationProcedure

# Create the molecule and the point group
mol = tq.Molecule(geometry="H 0. 0. 0. \n H 0. 0. 0.74804",basis_set='sto-3g', backend="pyscf")
pg: PointGroup = sun.symmetry_adaptation.PointGroup.from_pyscf("D2h")
provider_canonical = IrrepProvider(mol, pg, "canon")
states = FockSpaceState.non_ionic_states(mol, provider_canonical)

# Build the AO orbital permutation representation where each
# operation is represented as a permutation matrix acting on spatial orbitals
# The results are matrices of size mol.n_orbitals x mol.n_orbitals
rep_ao = QCircuitRepresentationBuilder(mol, pg).build_ao_permutation_representation()

# Build the QCircuit representation where each operation is represented as a quantum circuit
# The resulting circuit can be applied to a QubitWaveFunction representing a state in the fock state
# with 2 * mol.n_orbitals
rep_qcircuit = QCircuitRepresentationBuilder(mol, pg).build_qcircuit_representation()


# There is an abstraction to apply an operation to a state abstracted
# of the representation they are in, but if you know their types you
# can apply the operation directly
rep_ao.apply(rep_ao.operations["i"], np.ones(2)) # is the same as
rep_ao.operations["i"] @  np.ones(2)

rep_qcircuit.apply(rep_qcircuit.operations["i"], states[0].wavefunction) # is the same as
tq.simulate(rep_qcircuit.operations["i"], initial_state=states[0].wavefunction)
