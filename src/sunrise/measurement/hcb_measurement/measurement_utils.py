"""Tools to measure a molecular Hamiltonian efficiently: rotate the orbitals, split off the hard-core boson (HCB) part, and estimate the number of measurements."""

import numpy as np
import tequila as tq

from sunrise.molecules.qubit_base import Molecule as QubitMolecule


class input_state:
    """Container for the state whose energy is measured: either a circuit or a wavefunction.

    Parameters
    ----------
    circuit : tq.QCircuit, optional
        Circuit preparing the state.
    variables : dict, optional
        Values of the circuit variables.
    wavefunction : tq.QubitWaveFunction, optional
        State given directly. Exactly one of ``circuit`` and ``wavefunction`` must be given.

    Raises
    ------
    ValueError
        If both or neither of ``circuit`` and ``wavefunction`` are given.
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
        """Return the circuit; for a wavefunction input return an empty (dummy) circuit on the same qubits."""
        if self.wavefunction is not None and self.circuit is None:
            self.circuit = tq.gates.Rz(
                angle=0, target=range(self.wavefunction.n_qubits)
            )  # dummy empty circuit
        return self.circuit

    def get_wavefunction(self, variables=None):
        """Return the wavefunction; for a circuit input simulate it with ``variables`` (default: the stored ones)."""
        if self.circuit is not None and self.wavefunction is None:
            if variables is None:
                variables = self.variables
            self.wavefunction = tq.compile(self.circuit)(variables)
        return self.wavefunction


def gates_to_orb_rot(
    rotation_circuit: tq.QCircuit, dim: int, isReordered: bool = False, core: int = 0
) -> np.array:
    """Return the orbital rotation matrix represented by a circuit of ``UR`` gates.

    Parameters
    ----------
    rotation_circuit : tq.QCircuit
        Circuit containing only ``UR`` gates (each acts on spin up and down; only one of the pair is used).
    dim : int
        Number of spatial orbitals, including core orbitals.
    isReordered : bool, optional
        True if the circuit uses up-then-down qubit ordering. Default is False.
    core : int, optional
        Number of core orbitals; they are counted in ``dim`` but not in the circuit qubits. Default is 0.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(dim, dim)`` transforming the orbital coefficients.

    Examples
    --------
    >>> rot = sun.measurement.gates_to_orb_rot(UR, mol.n_orbitals)
    """

    # Extract parameters and indices from gates
    params = []
    indices = []
    for gate in reversed(rotation_circuit.gates):
        params.append(gate.parameter)
        g = []
        for idx in gate.indices:
            g.append(tuple([i + core + core * (not isReordered) for i in idx]))
        indices.append(g)

    # Select only odd values
    # Because UR gates rotate molecular orbitals
    params = params[1::2]
    indices = indices[1::2]

    # Refactor as a list
    tmp = []
    for i in indices:
        if isReordered:
            tmp.append((int(i[0][0]), int(i[0][1])))
        else:
            tmp.append((int(i[0][0] / 2), int(i[0][1] / 2)))
    indices = tmp

    # Create a matrix for each UR gate
    rot_matrices = []
    for k, index in enumerate(indices):
        tmp = np.eye(dim)  # dim is the dimension of orbital coefficients
        if isinstance(params[k], tq.Objective):
            params[k] = params[k]()
        # UR(i, j) moves an electron from i to j for a positive angle
        tmp[index[0]][index[0]] = np.cos(params[k] / 2)
        tmp[index[0]][index[1]] = -np.sin(params[k] / 2)
        tmp[index[1]][index[0]] = np.sin(params[k] / 2)
        tmp[index[1]][index[1]] = np.cos(params[k] / 2)
        rot_matrices.append(tmp)

    # Multiply all matrices
    tmp = rot_matrices[0]
    for matrix in rot_matrices[1:]:
        tmp = np.dot(tmp, matrix)
    transformation_matrix = tmp

    return transformation_matrix


def fold_rotators(mol, UR, *args, **kwargs):
    """Return a new molecule whose integrals are rotated by the orbital rotation of the ``UR`` circuit.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule to rotate.
    UR : tq.QCircuit
        Circuit with only ``UR`` gates.
    *args, **kwargs
        Passed to ``gates_to_orb_rot`` (e.g. ``isReordered``, ``core``).

    Returns
    -------
    QubitMolecule
        Molecule with transformed one- and two-body integrals.

    Examples
    --------
    >>> rotated_mol = sun.measurement.fold_rotators(mol, rotators[0])  # integrals in the rotated basis
    """
    # Get the transformation matrix from the circuit
    UR_matrix = gates_to_orb_rot(UR, mol.n_orbitals, *args, **kwargs)

    # Rotate one- and two-body part
    c, h, g = mol.get_integrals()
    g = g.elems

    th = np.einsum("ix, jx -> ij", h, UR_matrix, optimize="greedy")
    th = np.einsum("xj, ix -> ij", th, UR_matrix, optimize="greedy")
    # same as th = (UR_matrix.dot(h)).dot(UR_matrix.T)

    tg = np.einsum("ijkx, lx -> ijkl", g, UR_matrix, optimize="greedy")
    tg = np.einsum("ijxl, kx -> ijkl", tg, UR_matrix, optimize="greedy")
    tg = np.einsum("ixkl, jx -> ijkl", tg, UR_matrix, optimize="greedy")
    tg = np.einsum("xjkl, ix -> ijkl", tg, UR_matrix, optimize="greedy")

    tmol = QubitMolecule(
        parameters=mol.parameters,
        transformation=mol.transformation,
        n_electrons=mol.n_electrons,
        nuclear_repulsion=c,
        one_body_integrals=th,
        two_body_integrals=tg,
        ordering="openfermion",
    )

    return tmol


def get_hcb_part(mol):
    """Split a molecule into its hard-core boson (HCB) part and the remainder.

    The HCB part keeps the diagonal one-body terms and the two-body integrals of type
    ``(ii|kk)``, ``(ik|ki)`` and ``(ij|ij)``-like index patterns (``i==j and k==l``,
    ``i==l and j==k``, ``i==k and j==l``). Everything else is the residual.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule to split.

    Returns
    -------
    hcb_mol : QubitMolecule
        Molecule with the HCB integrals (includes the nuclear repulsion).
    res_mol : QubitMolecule
        Molecule with the remaining integrals (zero nuclear repulsion).

    Examples
    --------
    >>> hcb_mol, res_mol = sun.measurement.get_hcb_part(mol)
    """
    c, h, g = mol.get_integrals()

    # Get the HCB part of the one-body integral
    hcb_h = np.zeros(shape=(h.shape[0], h.shape[1]))
    res_h = np.zeros(shape=(h.shape[0], h.shape[1]))
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            if i == j:
                hcb_h[i][j] = h[i][j]  # ii
            else:
                res_h[i][j] = h[i][j]

    # Get the HCB part of the two-body integral
    hcb_g = np.zeros(shape=(g.shape[0], g.shape[1], g.shape[2], g.shape[3]))
    res_g = np.zeros(shape=(g.shape[0], g.shape[1], g.shape[2], g.shape[3]))
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            for k in range(g.shape[2]):
                for l in range(g.shape[3]):
                    if (
                        (i == j and k == l)
                        or (i == l and j == k)
                        or (i == k and j == l)
                    ):  # iikk or ikki or ijij
                        hcb_g[i][j][k][l] = g.elems[i][j][k][l]
                    else:
                        res_g[i][j][k][l] = g.elems[i][j][k][l]

    # Create the HCB and residual molecule
    hcb_mol = QubitMolecule(
        parameters=mol.parameters,
        transformation=mol.transformation,
        n_electrons=mol.n_electrons,
        nuclear_repulsion=c,
        one_body_integrals=hcb_h,
        two_body_integrals=hcb_g,
        ordering="openfermion",
    )
    res_mol = QubitMolecule(
        parameters=mol.parameters,
        transformation=mol.transformation,
        n_electrons=mol.n_electrons,
        nuclear_repulsion=0,
        one_body_integrals=res_h,
        two_body_integrals=res_g,
        ordering="openfermion",
    )

    return hcb_mol, res_mol


def rotate_and_hcb(
    molecule,
    rotators,
    circuit=None,
    variables=None,
    initial_state=None,
    target=None,
    silent=True,
    *args,
    **kwargs,
):
    """Iteratively rotate the orbitals and extract the HCB part of the Hamiltonian at each step.

    For each rotator the current residual molecule is rotated, its HCB part is stored and the
    remainder is passed on to the next rotator. Measuring each HCB molecule on the state
    followed by the corresponding rotator reconstructs the energy.

    Parameters
    ----------
    molecule : QuantumChemistryBase
        Molecule holding the Hamiltonian.
    rotators : list of tq.QCircuit
        Circuits of ``UR`` gates, one per step.
    circuit : tq.QCircuit, optional
        Circuit preparing the state.
    variables : dict, optional
        Variables of the circuit.
    initial_state : tq.QubitWaveFunction, optional
        State given as a wavefunction instead of a circuit.
    target : float, optional
        Reference energy; only used to print the error when ``silent`` is False.
    silent : bool, optional
        If False, print the running approximation of the energy at each step (requires
        ``target``). Default is True.
    *args, **kwargs
        Passed to ``fold_rotators`` and, when not silent, to ``tq.simulate``.

    Returns
    -------
    hcb_mols : list of QubitMolecule
        One HCB molecule per rotator.
    res_mol : QubitMolecule
        Residual molecule after the last rotator.

    Raises
    ------
    ValueError
        If neither ``circuit`` nor ``initial_state`` is given.

    References
    ----------
    D. Bincoletto and J. S. Kottmann, Digital Discovery (2026), doi:10.1039/D5DD00251F.

    Examples
    --------
    >>> hcb_mols, res = sun.measurement.rotate_and_hcb(
    ...     molecule=mol, rotators=rotators, circuit=U, variables=variables
    ... )
    """

    # Initialize the input state in a wrapper class
    state = input_state(
        circuit=circuit, variables=variables, wavefunction=initial_state
    )
    if initial_state is not None and circuit is None:
        circuit = state.get_circuit()
        initial_state = state.get_wavefunction()
    elif circuit is None and initial_state is None:
        raise ValueError("Either a circuit or a wavefunction must be provided")

    # General procedure, loop over the list of rotators
    hcb_mols = []
    approx = 0.0
    old_UR = tq.QCircuit()
    for i, UR in enumerate(rotators):
        # Rotate the two body part
        tmol = fold_rotators(molecule, old_UR.dagger() + UR, *args, **kwargs)
        tcircuit = circuit + UR

        # Extract the hcb part and the residual part of the transformed molecule
        hcb_mol, res_mol = get_hcb_part(tmol)

        if not silent:
            # Compute expectation values and error
            EX = tq.ExpectationValue(U=tcircuit, H=hcb_mol.make_hamiltonian())
            incr = tq.simulate(
                EX,
                *args,
                variables=variables,
                initial_state=initial_state,
                **kwargs,
            )
            approx += incr

            print(f"Iteration {i + 1}/{len(rotators)}:")

            print(f"Approximation:         {approx}")
            print(f"Increment:             {incr}")
            print(f"Error in new basis:    {target - approx}")
            # print(f"test:                  {target-approx-tq.simulate(tq.ExpectationValue(U=tcircuit, H=res_mol.make_hamiltonian()), variables=variables, initial_state=initial_state, args=args, **kwargs)}")

        # Update values
        hcb_mols.append(hcb_mol)
        molecule = res_mol
        old_UR = UR

    return hcb_mols, res_mol


def compute_num_meas(
    circuit=None,
    variables=None,
    initial_state=None,
    is_hcb=False,
    hcb_mol=None,
    meas_groups=None,
    transformations=None,
    eps=1e-3,
):
    """Estimate the number of measurements needed to reach error ``eps`` on an expectation value.

    For each measurement group the Pauli strings are measured after the group's diagonalizing
    transformation; the number of shots for a string is ``(|c| * std / eps)**2`` and the group
    needs the largest of these (at least 1). The group counts are summed.

    Parameters
    ----------
    circuit : tq.QCircuit, optional
        Circuit preparing the state.
    variables : dict, optional
        Variables of the circuit.
    initial_state : tq.QubitWaveFunction, optional
        State given as a wavefunction instead of a circuit.
    is_hcb : bool, optional
        If True, ``hcb_mol`` is split into its three measurement groups
        (diagonal, XX/YY, and the rest). Default is False.
    hcb_mol : QubitMolecule, optional
        Molecule with the HCB Hamiltonian (used if ``is_hcb``).
    meas_groups : list of tq.QubitHamiltonian, optional
        Measurement groups of commuting terms (used if not ``is_hcb``).
    transformations : list of tq.QCircuit, optional
        Circuit diagonalizing each group (used if not ``is_hcb``).
    eps : float, optional
        Target statistical error. Default is 1e-3.

    Returns
    -------
    float
        Total number of measurements.

    Raises
    ------
    ValueError
        If neither ``circuit`` nor ``initial_state`` is given.
    """

    # Initialize the input state in a wrapper class
    state = input_state(
        circuit=circuit, variables=variables, wavefunction=initial_state
    )
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
                if (p.naked()[q[0]] == "X" and p.naked()[q[1]] == "X") or (
                    p.naked()[q[0]] == "Y" and p.naked()[q[1]] == "Y"
                ):
                    H2 += tq.QubitHamiltonian().from_paulistrings(p)
                else:
                    H3 += tq.QubitHamiltonian().from_paulistrings(p)
        meas_groups = [H1, H2, H3]

    # Compute group number of measurements
    M_tot = 0
    for i, group in enumerate(meas_groups):
        if is_hcb:
            # Diagonalize each of the three hcb measurement groups
            fc_groups_and_unitaries, _ = (
                tq.grouping.compile_groups.compile_commuting_parts(
                    group, unitary_circuit="improved"
                )
            )
            group = fc_groups_and_unitaries[0][0]
            transformation = fc_groups_and_unitaries[0][1]
        else:
            transformation = transformations[i]

        # Compute individual number of measurements
        M_group = 1
        for op in group.paulistrings:
            # Compute individual Pauli number of measurements
            pauli_exp = tq.ExpectationValue(
                circuit + transformation,
                tq.QubitHamiltonian().from_paulistrings(op.naked()),
            )
            var = (
                1.0
                - tq.simulate(
                    pauli_exp, variables=variables, initial_state=initial_state
                )
                ** 2
            )
            var = round(var, 10)
            M_l = ((abs(op.coeff) * np.sqrt(var)) / eps) ** 2

            # Pick the largest number of measurements
            M_group = max(M_group, M_l)

        M_tot += M_group

    return M_tot
