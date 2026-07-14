from sunrise import fermionic_operations as fe
from tequila import QTensor,Variable,Objective,TequilaException,simulate
import numpy
import numbers
from typing import Union
# Adaptation of get_gives_decompostion from tequila qc_base at version 1.9.9dev at August 2025

OPTIMIZED_ORDERING = "Optimized"

def n_rotation(i:int, phi)->fe.FCircuit:
        """
        Creates a quantum circuit that applies a phase rotation based on phi to both components (up and down) of a given spatial orbital.

        Parameters:
        - i (int): The index of the qubit to which the rotation will be applied.
        - phi (float): The rotation angle. The actual rotation applied will be multiplied with -2 for both components.

        Returns:
        - FCircuit: A quantum circuit object containing the sequence of rotations applied to the up and down components of the specified spatial orbital.
        """

        # Start a new circuit and apply rotations to each component.
        circuit = fe.Phase(2*i, angle=-2 * phi)
        circuit += fe.Phase(2*i+1, angle=-2 * phi)
        return circuit

def get_givens_circuit(unitary:numpy.ndarray, tol:float=1e-6, ordering:Union[list,tuple,str]=OPTIMIZED_ORDERING, fix: bool = True, label=None,)->fe.FCircuit:
    """
    Constructs a quantum circuit from a given real unitary matrix using Givens rotations.

    This method decomposes a unitary matrix into a series of Givens and Phase rotations,
    then constructs and returns a quantum circuit that implements this sequence of rotations.

    Parameters:
        - unitary (numpy.array): A real unitary matrix representing the transformation to implement.
        - tol (float): A tolerance threshold below which matrix elements are considered zero.
        - ordering (list of tuples or 'Optimized'): Custom ordering of indices for Givens rotations or 'Optimized' to generate them automatically.
        - fix (bool): whether to set the angle as fixed or as inial value as: angle=tq.Variable(idx) + value. Useful to let further relaxation to the basis change
        - label: can be passed instead of angle to have auto-naming with label ("R",i,j,label) useful for repreating gates with individual variables

    Returns:
    - FCircuit: A quantum circuit implementing the series of rotations decomposed from the unitary.
    """
    # Decompose the unitary matrix into Givens and phase (Rz) rotations.
    theta_list, phi_list = get_givens_decomposition(unitary, tol, ordering)

    # Initialize an empty quantum circuit.
    circuit = fe.FCircuit()

    # Add all Rz (phase) rotations to the circuit.
    for phi in phi_list:
        if  isinstance(phi[0],numbers.Number) and abs(phi[0])%(2*numpy.pi)<tol:
                continue
        if fix:
            circuit += n_rotation(i=phi[1], phi=phi[0])
        else:
            circuit += n_rotation(
                i=phi[1],
                phi=phi[0]
                + Variable(
                    f"Ph({phi[1]}" + ("," + str(label)) * (label is not None) + ")"
                ),
            )

    # Add all Givens rotations to the circuit.
    for theta in reversed(theta_list):
        if  isinstance(theta[0],numbers.Number) and abs(theta[0])%numpy.pi<tol:
                continue
        if fix:
            circuit += fe.UR(i=theta[1], j=theta[2], angle=theta[0] * -2)
        else:
            circuit += fe.UR(
                i=theta[1],
                j=theta[2],
                angle=(theta[0] * -2)
                + Variable(
                    f"UR({theta[1]},{theta[2]}"
                    + ("," + str(label)) * (label is not None)
                    + ")"
                ),
            )

    return circuit

def givens_matrix(n, p, q, theta)->QTensor:
    """
    Construct a complex Givens rotation matrix of dimension n by theta between rows/columns p and q.
    """
    """
    Generates a Givens rotation matrix of size n x n to rotate by angle theta in the (p, q) plane. This matrix can be complex

    Parameters:
    - n (int): The size of the Givens rotation matrix.
    - p (int): The first index for the rotation plane.
    - q (int): The second index for the rotation plane.
    - theta (float): The rotation angle.

    Returns:
    - numpy.array: The Givens rotation matrix.
    """
    matrix = QTensor(shape=(n, n), objective_list=numpy.eye(n).reshape(n * n))  # Matrix to hold complex numbers
    
    if isinstance(theta, (Variable, Objective)):
        cos_theta = theta.apply(numpy.cos)
        sin_theta = theta.apply(numpy.sin)
    else:
        cos_theta = numpy.cos(theta)
        sin_theta = numpy.sin(theta)

    # Directly assign cosine and sine without complex phase adjustment
    matrix[p, p] = cos_theta
    matrix[q, q] = cos_theta
    matrix[p, q] = sin_theta
    matrix[q, p] = -sin_theta

    return matrix

def get_givens_decomposition(unitary:numpy.ndarray, tol:float=1e-6, ordering:Union[list,tuple,str]=OPTIMIZED_ORDERING, return_diagonal:bool=False):
    """
    Decomposes a real unitary matrix into Givens rotations (theta) and Rz rotations (phi).

    Parameters:
    - unitary (numpy.array): A real unitary matrix to decompose. It cannot be complex.
    - tol (float): Tolerance for considering matrix elements as zero. Elements with absolute value less than tol are treated as zero.
    - ordering (list of tuples or 'Optimized'): Custom ordering of indices for Givens rotations or 'Optimized' to generate them automatically.
    - return_diagonal (bool): If True, the function also returns the diagonal matrix as part of the output.

    Returns:
    - list: A list of tuples, each representing a Givens rotation. Each tuple contains the rotation angle theta and indices (i,j) of the rotation.
    - list: A list of tuples, each representing an Rz rotation. Each tuple contains the rotation angle phi and the index (i) of the rotation.
    - numpy.array (optional): The diagonal matrix after applying all Givens rotations, returned if return_diagonal is True.
    """
    U = unitary  # no need to copy as we don't modify the original
    # U[abs(U) < tol] = 0 # Zeroing out the small elements as per the tolerance level. #comented out, its being considered latter again
    n = U.shape[0]

    # Determine optimized ordering if specified.
    if ordering == OPTIMIZED_ORDERING:
        ordering = depth_eff_order_mf(n)

    theta_list = []
    phi_list = []

    def calcTheta(U, c, r):
        """Calculate and apply the Givens rotation for a specific matrix element."""
        t = arctan2(-U[r, c], U[r - 1, c])
        theta_list.append((t, r, r - 1))
        g = givens_matrix(n, r, r - 1, t)  # is a QTensor
        U = g.dot(U)
        return U

    # Apply and store Givens rotations as per the given or computed ordering.
    if ordering is None:
        for c in range(n):
            for r in range(n - 1, c, -1):
                U = calcTheta(U, c, r)
    else:
        for r, c in ordering:
            U = calcTheta(U, c, r)
    # Calculating the Rz rotations based on the phases of the diagonal elements.
    # For real elements this means a 180 degree shift, i.e. a sign change.
    for i in range(n):
        if isinstance(U[i, i], (Variable, Objective)):
            if len(U[i, i].args):
                phi_list.append((U[i, i].apply(numpy.angle), i))
        else:
            phi_list.append((numpy.angle(U[i, i]), i))
    # Filtering out rotations without significance.
    theta_list_new = []
    for i, theta in enumerate(theta_list):
        if isinstance(theta[0], (Variable, Objective)):
            if len(theta[0].args):
                theta_list_new.append(theta)
        elif (abs(theta[0])%( 2 * numpy.pi)) > tol:
            theta_list_new.append(theta)
    phi_list_new = []
    for i, phi in enumerate(phi_list):
        if isinstance(phi[0], (Variable, Objective)):
            if len(phi[0].args):
                phi_list_new.append(phi)
        elif abs(phi[0]) > tol:
            phi_list_new.append(phi)
    if return_diagonal:
        # Optionally return the resulting diagonal
        return theta_list_new, phi_list_new, U
    else:
        return theta_list_new, phi_list_new

def reconstruct_matrix_from_givens(n:int, theta_list:Union[list,tuple], phi_list:Union[list,tuple], to_real_if_possible:bool=True, tol:float=1e-12)->numpy.ndarray:
    """
    Reconstructs a matrix from given Givens rotations and Phase diagonal rotations.
    This function is effectively an inverse of get_givens_decomposition, and therefore only works with data in the same format as its output.

    Parameters:
    - n (int): The size of the unitary matrix to be reconstructed.
    - theta_list (list of tuples): Each tuple contains (angle, i, j) representing a Givens rotation of `angle` radians, applied to rows/columns `i` and `j`.
    - phi_list (list of tuples): Each tuple contains (angle, i), representing an Rz rotation by `angle` radians applied to the `i`th diagonal element.
    - to_real_if_possible (bool): If True, converts the matrix to real if its imaginary part is effectively zero.
    - tol (float): The tolerance whether to swap a complex rotation for a sign change.

    Returns:
    - numpy.ndarray: The reconstructed complex or real matrix, depending on the `to_real_if_possible` flag and matrix composition.
    """
    # Start with an identity matrix
    # reconstructed = numpy.eye(norb, dtype=complex)
    reconstructed =QTensor(shape=(n, n), objective_list=numpy.eye(n, dtype=complex).reshape(n * n))
    
    # Apply Rz rotations for diagonal elements
    for phi in phi_list:
        angle, i = phi
        if isinstance(angle, (Variable, Objective)):
            reconstructed[i, i] = reconstructed[i, i]*angle.apply(numpy.cos) + 1j*reconstructed[i, i]*angle.apply(numpy.sin)
        else:
            # Directly apply a sign flip if the rotation angle is π
            if numpy.isclose(angle, numpy.pi, atol=tol):
                reconstructed[i, i] *= -1
            else:
                reconstructed[i, i] *= numpy.exp(1j * angle)
    
    
    # Apply Givens rotations in reverse order
    for theta in reversed(theta_list):
        angle, i, j = theta
        g = givens_matrix(n, i, j, angle)
        reconstructed = reconstructed.dot(g)  

    # Convert matrix to real if its imaginary part is negligible unless disabled via to_real_if_possible
    if to_real_if_possible:
        # Directly apply a sign flip if the rotation angle is π
        if numpy.all(reconstructed.imag == 0):
            # Convert to real by taking the real part
            reconstructed = reconstructed.real

    return reconstructed.T

def reconstruct_matrix_from_circuit(U:fe.FCircuit,norb:int=None, to_real_if_possible:bool=True, tol:float=1e-12)->QTensor:#numpy.ndarray:
    """
    Reconstructs a matrix from a UR + Phase FCircuit.
    This function is effectively an inverse of get_givens_circuit.

    Parameters:
    - norb (int): The size of the unitary matrix to be reconstructed. If None, the number of qubits//2 in U is used.
    - U (FCircuit): The quantum circuit containing only UR (optionally Excitations gates) and Phase rotations.
    - to_real_if_possible (bool): If True, converts the matrix to real if its imaginary part is effectively zero.
    - tol (float): The tolerance whether to swap a complex rotation for a sign change.

    Returns:
    - numpy.ndarray: The reconstructed complex or real matrix, depending on the `to_real_if_possible` flag and matrix composition.
    """
    U = U.to_udud(norb)
    if norb is None:
        norb = U.n_qubits // 2
    phi_list = []
    theta_list = []  
    first = True
    for i,gate in enumerate(U.gates):
        if gate.name =='UR':
            theta_list.append((0.5*gate.variables, gate.indices[0][0][0]//2, gate.indices[0][0][1]//2))
        elif gate.name == "Ph":
            if not first:
                first = True
                continue
            if i+1!=len(U.gates) and U.gates[i+1].name == "Ph"  and U.gates[i+1].indices[0][0][0]//2 == gate.indices[0][0][0]//2:
                if isinstance(gate.variables, (Variable, Objective)) and isinstance(U.gates[i+1].variables, (Variable, Objective)): # Not sure if I can check the to Objectives to be the same, so I just go on
                    phi_list.append((-0.5*gate.variables, gate.indices[0][0][0]//2))
                elif isinstance(gate.variables, numbers.Number) and isinstance(U.gates[i+1].variables, numbers.Number) and numpy.isclose(gate.variables,U.gates[i+1].variables, atol=tol):
                    phi_list.append((-0.5*gate.variables, gate.indices[0][0][0]//2))
                else:
                    raise TequilaException("Phase gates must come in pairs to be converted to matrix.")
                first = False
            else:
                raise TequilaException("Phase gates must come in pairs to be converted to matrix.")
        elif gate.name == "FermionicExcitation":
            if not first:
                first = True
                continue
            if i+1!=len(U.gates) and U.gates[i+1].name == "FermionicExcitation" and len(U.gates[i+1].indices[0])==len(gate.indices[0])==1 and gate.indices[0][0][0]//2 == U.gates[i+1].indices[0][0][0]//2 and gate.indices[0][0][1]//2 == U.gates[i+1].indices[0][0][1]//2:
                if isinstance(gate.variables, (Variable, Objective)) and isinstance(U.gates[i+1].variables, (Variable, Objective)): # Not sure if I can check the to Objectives to be the same, so I just go on
                    theta_list.append((0.5*gate.variables, gate.indices[0][0][0]//2, gate.indices[0][0][1]//2))
                elif isinstance(gate.variables, numbers.Number) and isinstance(U.gates[i+1].variables, numbers.Number) and numpy.isclose(gate.variables, U.gates[i+1].variables, atol=tol):
                    theta_list.append((0.5*gate.variables, gate.indices[0][0][0]//2, gate.indices[0][0][1]//2))
                else:
                    raise TequilaException("FermionicExcitation gates must come in pairs to be converted to UR gates.")
                first = False
            else:
                raise TequilaException("FermionicExcitation gates must come in pairs to be converted to UR gates.")
        else:
            raise TequilaException(f"Reconstruction from circuits gate {gate} not implented.")
    return reconstruct_matrix_from_givens(norb, theta_list[::-1], phi_list[::-1], to_real_if_possible, tol)



def arctan2(x1, x2, *args, **kwargs):
    if isinstance(x1, (Variable, Objective)) or isinstance(x2, (Variable, Objective)):
        return Objective().binary_operator(left=1 * x1, right=1 * x2, op=numpy.arctan2)
    elif not isinstance(x1, numbers.Complex) and not isinstance(x2, numbers.Complex):
        return numpy.arctan2(x1, x2)
    else:
        return numpy.arctan2(x1.imag, x2.imag) + numpy.arctan2(x1.real, x2.real)


#This one copied from tequila.grouping.fermionic_functions, same date and patch as above
def depth_eff_order_mf(N:int)->list:
    """
    Returns index ordering for linear depth circuit

    For example N = 6 gives elimination order
    [ 0.  0.  0.  0.  0.  0.]
    [ 7.  0.  0.  0.  0.  0.]
    [ 5. 10.  0.  0.  0.  0.]
    [ 3.  8. 12.  0.  0.  0.]
    [ 2.  6. 11. 14.  0.  0.]
    [ 1.  4.  9. 13. 15.  0.]
    """
    l = []
    for c in range(0, N - 1):
        for r in range(1, N):
            if r - c > 0:
                l.append([r, c, 2 * c - r + N])
    l.sort(key=lambda x: x[2])
    return [(a[0], a[1]) for a in l]