"""Decomposition of orbital-basis changes (real orthogonal matrices) into Givens rotations (``UR`` gates) and phases, and the inverse reconstruction."""

from __future__ import annotations

import numbers

import numpy
from tequila import Objective, QTensor, TequilaException, Variable

from sunrise import fermionic_operations as fe

# Adaptation of get_gives_decompostion from tequila qc_base at version 1.9.9dev at August 2025

OPTIMIZED_ORDERING = "Optimized"


def n_rotation(i: int, phi) -> fe.FCircuit:
    """Create the phase gates on the up and down spin orbitals of spatial orbital ``i``.

    Parameters
    ----------
    i : int
        Spatial orbital index.
    phi : float or tq.Variable
        Rotation angle; the gates use the angle ``-2 * phi``.

    Returns
    -------
    FCircuit
    """

    # Start a new circuit and apply rotations to each component.
    circuit = fe.Phase(2 * i, variables=-2 * phi)
    circuit += fe.Phase(2 * i + 1, variables=-2 * phi)
    return circuit


def get_givens_circuit(
    unitary: numpy.ndarray,
    tol: float = 1e-6,
    ordering: list | tuple | str | None = OPTIMIZED_ORDERING,
    fix: bool = True,
    label=None,
) -> fe.FCircuit:
    """Build the FCircuit implementing a real unitary matrix with Givens rotations.

    The matrix is decomposed with ``get_givens_decomposition`` into ``UR`` and ``Phase`` gates.

    Parameters
    ----------
    unitary : numpy.ndarray
        Real unitary matrix to implement.
    tol : float, optional
        Elements and angles below this value are treated as zero. Default is 1e-6.
    ordering : list of tuple of int or str or None, optional
        Custom ``(row, column)`` elimination order, or ``"Optimized"`` (default) for a linear-depth order.
    fix : bool, optional
        If True (default) the angles are fixed numbers. If False each angle is a
        variable ``UR(i,j[,label])`` / ``Ph(i[,label])`` with the decomposition value as offset,
        so the basis change can be relaxed later.
    label : optional
        Label added to the variable names when ``fix`` is False.

    Returns
    -------
    FCircuit

    Examples
    --------
    >>> from sunrise.fermionic_operations.givens_rotations import get_givens_circuit
    >>> UR = get_givens_circuit(mo_coeff)  # mo_coeff: real orthogonal matrix
    """
    # Decompose the unitary matrix into Givens and phase (Rz) rotations.
    theta_list, phi_list = get_givens_decomposition(unitary, tol, ordering)

    # Initialize an empty quantum circuit.
    circuit = fe.FCircuit()

    # Add all Rz (phase) rotations to the circuit.
    for phi in phi_list:
        if isinstance(phi[0], numbers.Number) and abs(phi[0]) % (2 * numpy.pi) < tol:
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
        # theta = pi is not the identity: it flips the sign of both orbitals
        if (
            isinstance(theta[0], numbers.Number)
            and abs(theta[0]) % (2 * numpy.pi) < tol
        ):
            continue
        # (UR(i, j) moves an electron from i to j for a positive angle)
        if fix:
            circuit += fe.UR(i=theta[1], j=theta[2], variables=theta[0] * 2)
        else:
            circuit += fe.UR(
                i=theta[1],
                j=theta[2],
                variables=(theta[0] * 2)
                + Variable(
                    f"UR({theta[1]},{theta[2]}"
                    + ("," + str(label)) * (label is not None)
                    + ")"
                ),
            )

    return circuit


def givens_matrix(n, p, q, theta) -> QTensor:
    """Return the ``n x n`` Givens rotation matrix by angle ``theta`` in the (p, q) plane.

    Parameters
    ----------
    n : int
        Dimension of the matrix.
    p, q : int
        Indices of the rotation plane.
    theta : float, tq.Variable or tq.Objective
        Rotation angle.

    Returns
    -------
    tq.QTensor
        The matrix, which can be complex.
    """
    matrix = QTensor(
        shape=(n, n), objective_list=numpy.eye(n).reshape(n * n)
    )  # Matrix to hold complex numbers

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


def get_givens_decomposition(
    unitary: numpy.ndarray,
    tol: float = 1e-6,
    ordering: list | tuple | str | None = OPTIMIZED_ORDERING,
    return_diagonal: bool = False,
):
    """Decompose a real unitary matrix into Givens rotations and phases.

    Parameters
    ----------
    unitary : numpy.ndarray
        Real unitary matrix.
    tol : float, optional
        Elements and angles with absolute value below this are treated as zero. Default is 1e-6.
    ordering : list of tuple of int or str or None, optional
        Custom ``(row, column)`` elimination order, or ``"Optimized"`` (default) for a linear-depth order.
    return_diagonal : bool, optional
        Also return the remaining diagonal matrix. Default is False.

    Returns
    -------
    theta_list : list of tuple
        Givens rotations ``(theta, i, j)``.
    phi_list : list of tuple
        Phase rotations ``(phi, i)``.
    diagonal : numpy.ndarray
        Diagonal matrix left after the eliminations (only if ``return_diagonal``).

    Examples
    --------
    >>> theta_list, phi_list = get_givens_decomposition(unitary)
    >>> R = reconstruct_matrix_from_givens(unitary.shape[0], theta_list, phi_list)  # equals unitary
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
        if isinstance(t, numbers.Number) and numpy.isclose(
            numpy.abs(t) % numpy.pi, 0, atol=tol
        ):
            return U
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
        elif (abs(theta[0]) % (2 * numpy.pi)) > tol:
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


def reconstruct_matrix_from_givens(
    n: int,
    theta_list: list | tuple | None,
    phi_list: list | tuple | None,
    to_real_if_possible: bool = True,
    tol: float = 1e-12,
) -> numpy.ndarray:
    """Reconstruct a matrix from Givens rotations and Phase diagonal rotations.

    This is effectively the inverse of ``get_givens_decomposition`` and therefore only works with
    data in the same format as its output: the phases are applied to the identity first, then the
    Givens rotations in reverse order.

    Parameters
    ----------
    n : int
        Dimension of the matrix to reconstruct.
    theta_list : list of tuple
        Givens rotations ``(angle, i, j)``: a rotation by ``angle`` radians acting on rows/columns
        ``i`` and ``j``. Angles may be variables.
    phi_list : list of tuple
        Phase rotations ``(angle, i)``: the ``i``-th diagonal element is multiplied by
        ``exp(1j * angle)``. Angles may be variables.
    to_real_if_possible : bool, optional
        If True (default), return a real matrix when its imaginary part is zero.
    tol : float, optional
        Tolerance to detect an angle of pi, which is applied as an exact sign change instead of a
        complex phase. Default is 1e-12.

    Returns
    -------
    numpy.ndarray or tq.QTensor
        The reconstructed matrix, real or complex depending on ``to_real_if_possible`` and on
        the rotations; a QTensor if some angles are variables.

    Examples
    --------
    >>> theta_list, phi_list = get_givens_decomposition(unitary)
    >>> R = reconstruct_matrix_from_givens(unitary.shape[0], theta_list, phi_list)  # equals unitary
    """
    # Start with an identity matrix
    # reconstructed = numpy.eye(norb, dtype=complex)
    reconstructed = QTensor(
        shape=(n, n), objective_list=numpy.eye(n, dtype=complex).reshape(n * n)
    )

    # Apply Rz rotations for diagonal elements
    for phi in phi_list:
        angle, i = phi
        if isinstance(angle, (Variable, Objective)):
            reconstructed[i, i] = reconstructed[i, i] * angle.apply(
                numpy.cos
            ) + 1j * reconstructed[i, i] * angle.apply(numpy.sin)
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

    # givens_matrix returns a QTensor, so with numeric angles the entries are numbers in an object array
    if all(isinstance(x, numbers.Number) for x in numpy.ravel(reconstructed)):
        reconstructed = numpy.asarray(reconstructed, dtype=complex)

    # Convert matrix to real if its imaginary part is negligible unless disabled via to_real_if_possible
    if to_real_if_possible and numpy.all(reconstructed.imag == 0):
        # Directly apply a sign flip if the rotation angle is π
        # Convert to real by taking the real part
        reconstructed = reconstructed.real

    return reconstructed.T


def reconstruct_matrix_from_circuit(
    U: fe.FCircuit,
    norb: int | None = None,
    to_real_if_possible: bool = True,
    tol: float = 1e-12,
) -> QTensor:  # numpy.ndarray:
    """Reconstruct the matrix implemented by a circuit of ``UR`` and ``Phase`` gates.

    This is effectively the inverse of ``get_givens_circuit``: the gates are translated back into
    Givens rotations and phases, which ``reconstruct_matrix_from_givens`` turns into a matrix.
    ``Phase`` gates must come in pairs acting on both spin orbitals of the same spatial orbital
    with the same angle; single-excitation ``FermionicExcitation`` gates are accepted in the
    same paired form instead of ``UR`` gates.

    Parameters
    ----------
    U : FCircuit
        Circuit containing only ``UR`` (optionally paired excitation) and ``Phase`` gates.
    norb : int, optional
        Dimension of the matrix, i.e. the number of spatial orbitals. Default is
        ``U.n_qubits // 2``.
    to_real_if_possible : bool, optional
        If True (default), return a real matrix when its imaginary part is zero.
    tol : float, optional
        Tolerance to compare the angles of paired gates and to detect an angle of pi, which is
        applied as an exact sign change instead of a complex phase. Default is 1e-12.

    Returns
    -------
    numpy.ndarray or tq.QTensor
        The reconstructed matrix, real or complex depending on ``to_real_if_possible`` and on
        the rotations; a QTensor if some angles are variables.

    Raises
    ------
    TequilaException
        If the circuit contains other gates, or ``Phase`` or excitation gates do not come in
        matching pairs.
    """
    U = U.to_udud(norb)
    if norb is None:
        norb = U.n_qubits // 2
    phi_list = []
    theta_list = []
    first = True
    for i, gate in enumerate(U.gates):
        if gate.name == "UR":
            theta_list.append(
                (
                    -0.5 * gate.variables,
                    gate.indices[0][0][0] // 2,
                    gate.indices[0][0][1] // 2,
                )
            )
        elif gate.name == "Ph":
            if not first:
                first = True
                continue
            if (
                i + 1 != len(U.gates)
                and U.gates[i + 1].name == "Ph"
                and U.gates[i + 1].indices[0][0][0] // 2 == gate.indices[0][0][0] // 2
            ):
                if (
                    isinstance(gate.variables, (Variable, Objective))
                    and isinstance(U.gates[i + 1].variables, (Variable, Objective))
                ):  # Not sure if I can check the to Objectives to be the same, so I just go on
                    phi_list.append((-0.5 * gate.variables, gate.indices[0][0][0] // 2))
                elif (
                    isinstance(gate.variables, numbers.Number)
                    and isinstance(U.gates[i + 1].variables, numbers.Number)
                    and numpy.isclose(
                        gate.variables, U.gates[i + 1].variables, atol=tol
                    )
                ):
                    phi_list.append((-0.5 * gate.variables, gate.indices[0][0][0] // 2))
                else:
                    raise TequilaException(
                        "Phase gates must come in pairs to be converted to matrix."
                    )
                first = False
            else:
                raise TequilaException(
                    "Phase gates must come in pairs to be converted to matrix."
                )
        elif gate.name == "FermionicExcitation":
            if not first:
                first = True
                continue
            if (
                i + 1 != len(U.gates)
                and U.gates[i + 1].name == "FermionicExcitation"
                and len(U.gates[i + 1].indices[0]) == len(gate.indices[0]) == 1
                and gate.indices[0][0][0] // 2 == U.gates[i + 1].indices[0][0][0] // 2
                and gate.indices[0][0][1] // 2 == U.gates[i + 1].indices[0][0][1] // 2
            ):
                if (
                    isinstance(gate.variables, (Variable, Objective))
                    and isinstance(U.gates[i + 1].variables, (Variable, Objective))
                ):  # Not sure if I can check the to Objectives to be the same, so I just go on
                    theta_list.append(
                        (
                            -0.5 * gate.variables,
                            gate.indices[0][0][0] // 2,
                            gate.indices[0][0][1] // 2,
                        )
                    )
                elif (
                    isinstance(gate.variables, numbers.Number)
                    and isinstance(U.gates[i + 1].variables, numbers.Number)
                    and numpy.isclose(
                        gate.variables, U.gates[i + 1].variables, atol=tol
                    )
                ):
                    theta_list.append(
                        (
                            -0.5 * gate.variables,
                            gate.indices[0][0][0] // 2,
                            gate.indices[0][0][1] // 2,
                        )
                    )
                else:
                    raise TequilaException(
                        "FermionicExcitation gates must come in pairs to be converted to UR gates."
                    )
                first = False
            else:
                raise TequilaException(
                    "FermionicExcitation gates must come in pairs to be converted to UR gates."
                )
        else:
            raise TequilaException(
                f"Reconstruction from circuits gate {gate} not implented."
            )
    return reconstruct_matrix_from_givens(
        norb, theta_list[::-1], phi_list[::-1], to_real_if_possible, tol
    )


def arctan2(x1, x2, *args, **kwargs):
    """Element-wise ``arctan2`` that also accepts tequila variables/objectives and complex numbers.

    Parameters
    ----------
    x1, x2 : float, complex, tq.Variable or tq.Objective
        Arguments as in ``numpy.arctan2(x1, x2)``.
    """
    if isinstance(x1, (Variable, Objective)) or isinstance(x2, (Variable, Objective)):
        return Objective().binary_operator(left=1 * x1, right=1 * x2, op=numpy.arctan2)
    elif not isinstance(x1, numbers.Complex) and not isinstance(x2, numbers.Complex):
        return numpy.arctan2(x1, x2)
    else:
        return numpy.arctan2(x1.imag, x2.imag) + numpy.arctan2(x1.real, x2.real)


# This one copied from tequila.grouping.fermionic_functions, same date and patch as above
def depth_eff_order_mf(N: int) -> list:
    """Return the elimination order of a linear-depth Givens network.

    For example N = 6 gives the elimination order (numbers show the step)::

        [ 0.  0.  0.  0.  0.  0.]
        [ 7.  0.  0.  0.  0.  0.]
        [ 5. 10.  0.  0.  0.  0.]
        [ 3.  8. 12.  0.  0.  0.]
        [ 2.  6. 11. 14.  0.  0.]
        [ 1.  4.  9. 13. 15.  0.]

    Parameters
    ----------
    N : int
        Dimension of the matrix.

    Returns
    -------
    list of tuple of int
        ``(row, column)`` pairs in elimination order.
    """
    l = []
    for c in range(N - 1):
        for r in range(1, N):
            if r - c > 0:
                l.append([r, c, 2 * c - r + N])
    l.sort(key=lambda x: x[2])
    return [(a[0], a[1]) for a in l]
