import tequila as tq
import numpy as np

class input_state:
    """
    Wrapper class to hold the input state to measure the molecule Hamiltonian
    The input state can be a quantum circuit or a wavefunction
    """
    circuit = None
    variables = None
    wavefunction = None

    def __init__(self, circuit=None, variables=None, wavefunction=None):
        if circuit is None and wavefunction is None:
            raise ValueError("Either a circuit or a wavefunction must be provided")
        if circuit is not None and wavefunction is not None:
            raise ValueError("Only one of circuit or wavefunction must be provided")
        
        self.circuit = circuit
        self.variables = variables
        self.wavefunction = wavefunction

    def get_circuit(self):
        """
        Get the circuit from the wavefunction or the circuit
        """
        if self.wavefunction is not None and self.circuit is None:
            self.circuit = tq.gates.Rz(angle=0, target=range(self.wavefunction.n_qubits)) # dummy empty circuit
        return self.circuit
        
    def get_wavefunction(self, variables=None):
        """
        Get the wavefunction from the circuit or the wavefunction
        """
        if self.circuit is not None and self.wavefunction is None:
            if variables is None:
                variables = self.variables
            self.wavefunction = tq.compile(self.circuit)(variables)
        return self.wavefunction

def gates_to_orb_rot(rotation_circuit:tq.QCircuit, dim:int, isReordered:bool=False,core:int=0)->np.array:
    """
    rotation_circuit: a circuit with ONLY UR gates
    dim: the dimension of the spatial orbital space, i.e., the number of orbitals
    core: number of core orbitals, already counted on dim. This is bcs these core orbitals are not included on the UR qubit count
    Output: a matrix that rotates the orbital coefficients
    """

    # Extract parameters and indices from gates
    params = []
    indices = []
    for gate in reversed(rotation_circuit.gates):
        params.append(gate.parameter)
        g = []
        for idx in gate.indices:
            g.append(tuple([i+core+core*(not isReordered) for i in idx]))
        indices.append(g)

    # Select only odd values
    # Because UR gates rotate molecular orbitals 
    params = params[1::2]
    indices = indices[1::2]

    # Refactor as a list
    tmp = []
    for i in indices:
        if isReordered:
            tmp.append((int(i[0][0]),int(i[0][1])))
        else:
            tmp.append((int(i[0][0]/2),int(i[0][1]/2)))
    indices = tmp

    # Create a matrix for each UR gate
    rot_matrices = []
    for k,index in enumerate(indices):
        tmp = np.eye(dim) # dim is the dimension of orbital coefficients
        if isinstance(params[k], tq.Objective):
            params[k] = params[k]()
        tmp[index[0]][index[0]] = np.cos(params[k]/2)
        tmp[index[0]][index[1]] = np.sin(params[k]/2)
        tmp[index[1]][index[0]] = -np.sin(params[k]/2)
        tmp[index[1]][index[1]] = np.cos(params[k]/2)
        rot_matrices.append(tmp)

    # Multiply all matrices
    tmp = rot_matrices[0]
    for matrix in rot_matrices[1:]:
        tmp = np.dot(tmp,matrix)
    transformation_matrix = tmp

    return transformation_matrix

def fold_rotators(mol, UR, *args, **kwargs):
    """
    Rotate the one- and two-body integrals of a molecule using the UR gates
    """
    # Get the transformation matrix from the circuit
    UR_matrix = gates_to_orb_rot(UR, mol.n_orbitals, *args, **kwargs)

    # Rotate one- and two-body part
    c,h,g = mol.get_integrals()
    g = g.elems

    th = np.einsum("ix, jx -> ij",  h, UR_matrix, optimize='greedy')
    th = np.einsum("xj, ix -> ij", th, UR_matrix, optimize='greedy')
    # same as th = (UR_matrix.dot(h)).dot(UR_matrix.T)

    tg = np.einsum("ijkx, lx -> ijkl",  g, UR_matrix, optimize='greedy')
    tg = np.einsum("ijxl, kx -> ijkl", tg, UR_matrix, optimize='greedy')
    tg = np.einsum("ixkl, jx -> ijkl", tg, UR_matrix, optimize='greedy')
    tg = np.einsum("xjkl, ix -> ijkl", tg, UR_matrix, optimize='greedy')

    tmol = tq.Molecule(parameters=mol.parameters, transformation=mol.transformation, n_electrons=mol.n_electrons,
                       nuclear_repulsion=c, one_body_integrals=th, two_body_integrals=tg, ordering='openfermion')

    return tmol

def get_hcb_part(mol):
    """
    Extract the hcb part and the residual part of a molecule
    """
    c,h,g = mol.get_integrals()

    # Get the HCB part of the one-body integral
    hcb_h = np.zeros(shape=(h.shape[0], h.shape[1]))
    res_h = np.zeros(shape=(h.shape[0], h.shape[1]))
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            if i==j:
                hcb_h[i][j] = h[i][j] # ii
            else:
                res_h[i][j] = h[i][j]

    # Get the HCB part of the two-body integral
    hcb_g = np.zeros(shape=(g.shape[0], g.shape[1], g.shape[2], g.shape[3]))
    res_g = np.zeros(shape=(g.shape[0], g.shape[1], g.shape[2], g.shape[3]))
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            for k in range(g.shape[2]):
                for l in range(g.shape[3]):
                    if (i==j and k==l) or (i==l and j==k) or (i==k and j==l): # iikk or ikki or ijij
                        hcb_g[i][j][k][l] = g.elems[i][j][k][l]
                    else:
                        res_g[i][j][k][l] = g.elems[i][j][k][l]

    # Create the HCB and residual molecule
    hcb_mol = tq.Molecule(parameters=mol.parameters, transformation=mol.transformation, n_electrons=mol.n_electrons,
                          nuclear_repulsion=c, one_body_integrals=hcb_h, two_body_integrals=hcb_g, ordering='openfermion')
    res_mol = tq.Molecule(parameters=mol.parameters, transformation=mol.transformation, n_electrons=mol.n_electrons,
                          nuclear_repulsion=0, one_body_integrals=res_h, two_body_integrals=res_g, ordering='openfermion')

    return hcb_mol, res_mol

def rotate_and_hcb(molecule, rotators, circuit=None, variables=None, initial_state=None, target=None, silent=True, *args, **kwargs):
    """
    Rotate the molecule using the UR gates and extract the hcb part and the residual part of the transformed molecule.
    Iterate for all the elements of the rotators list.

    molecule: the molecule that holds the Hamiltonian to be processed
    rotators: the UR gates to be applied

    circuit: the quantum circuit to be measured
    variables: the parameters of the circuit
    initial_state: if you want to use a wavefunction without a quantum circuit

    approx: the initial approximation of the energy
    target: the target energy

    Return a list of HCB molecules and the last residual molecule
    """

    # Initialize the input state in a wrapper class
    state = input_state(circuit=circuit, variables=variables, wavefunction=initial_state)
    if initial_state is not None and circuit is None:
        circuit = state.get_circuit()
        initial_state = state.get_wavefunction()
    elif circuit is None and initial_state is None:
        raise ValueError("Either a circuit or a wavefunction must be provided")

    # General procedure, loop over the list of rotators
    hcb_mols = []
    approx = 0.0
    old_UR = tq.QCircuit()
    for i,UR in enumerate(rotators):
        # Rotate the two body part
        tmol = fold_rotators(molecule, old_UR.dagger()+UR, *args, **kwargs)
        tcircuit = circuit + UR

        # Extract the hcb part and the residual part of the transformed molecule
        hcb_mol, res_mol = get_hcb_part(tmol)

        if not silent:
            # Compute expectation values and error
            EX = tq.ExpectationValue(U=tcircuit, H=hcb_mol.make_hamiltonian())
            incr = tq.simulate(EX, variables=variables, initial_state=initial_state, *args, **kwargs)
            EY = tq.ExpectationValue(U=tcircuit, H=res_mol.make_hamiltonian())
            rest = tq.simulate(EY, variables=variables, initial_state=initial_state, *args, **kwargs)
            approx += incr

            print(f"Iteration {i+1}/{len(rotators)}:")

            print(f"Approximation:         {approx}")
            print(f"Increment:             {incr}")
            print(f"Error in new basis:    {target-approx}")
            # print(f"test:                  {target-approx-rest}")

            M_tot = compute_num_meas(circuit=circuit, variables=variables, initial_state=initial_state, is_hcb=True, hcb_mol=hcb_mol)
            print(f"Number of measurements: {M_tot:e}\n")

        # Update values
        hcb_mols.append(hcb_mol)
        molecule = res_mol
        old_UR = UR

    return hcb_mols, res_mol

def compute_num_meas(circuit=None, variables=None, initial_state=None, is_hcb=False, hcb_mol=None, meas_groups=None, transformations=None, eps=1e-3):
    """
    Compute the number of measurements needed to estimate the expectation value of a Hamiltonian.
    
    circuit: the quantum circuit to be measured
    variables: the parameters of the circuit
    initial_state: if you want to use a wavefunction without a quantum circuit

    is_hcb: if True, we measure the HCB Hamiltonian exploiting the three measurement groups
    hcb_mol: the molecule that holds the HCB Hamiltonian
    meas_groups: the measurement groups, in case the operator is not the HCB Hamiltonian
    transformations: the transformations to be applied to the circuit, in case the operator is not the HCB Hamiltonian
    eps: the error tolerance
    """
    
    # Initialize the input state in a wrapper class
    state = input_state(circuit=circuit, variables=variables, wavefunction=initial_state)
    if initial_state is not None and circuit is None:
        circuit = state.get_circuit()
        initial_state = state.get_wavefunction()
    elif circuit is None and initial_state is None:
        raise ValueError("Either a circuit or a wavefunction must be provided")

    # If hcb then split in the three measurement groups
    if is_hcb:
        H1 = tq.QubitHamiltonian()
        H2 = tq.QubitHamiltonian()
        H3 = tq.QubitHamiltonian()
        for p in hcb_mol.make_hamiltonian().paulistrings:
            q = p.naked().qubits
            if p.is_all_z():
                H1 += tq.QubitHamiltonian().from_paulistrings(p)
            else:
                if (p.naked()[q[0]] == "X" and p.naked()[q[1]] == "X") or (p.naked()[q[0]] == "Y" and p.naked()[q[1]] == "Y"):
                    H2 += tq.QubitHamiltonian().from_paulistrings(p)
                else:
                    H3 += tq.QubitHamiltonian().from_paulistrings(p)
        meas_groups = [H1, H2, H3]

    # Compute group number of measurements
    M_tot = 0
    for i, group in enumerate(meas_groups):

        if is_hcb:
            # Diagonalize each of the three hcb measurement groups
            fc_groups_and_unitaries, _ = tq.grouping.compile_groups.compile_commuting_parts(group, unitary_circuit='improved')
            group = fc_groups_and_unitaries[0][0]
            transformation = fc_groups_and_unitaries[0][1]
        else:
            transformation = transformations[i]

        # Compute individual number of measurements
        M_group = 1
        for op in group.paulistrings:
            # Compute individual Pauli number of measurements
            pauli_exp = tq.ExpectationValue(circuit+transformation, tq.QubitHamiltonian().from_paulistrings(op.naked()))
            var = 1.0 - tq.simulate(pauli_exp, variables=variables, initial_state=initial_state)**2
            M_l = ((abs(op.coeff) * np.sqrt(var)) / eps) ** 2

            # Pick the largest number of measurements
            if M_l > M_group:
                M_group = M_l
        
        M_tot += M_group

    return M_tot