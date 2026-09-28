import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import orthogonal_procrustes

class VectorUtils:
    # Unused for now
    @staticmethod
    def roughly_coplanar(vectors, threshold=1e-5):
        """
        Check if a list of vectors are roughly in the same plane, within a given threshold.

        :param vectors: A list of vectors, where each vector is a list/array of length 3.
                        Example: vectors = [[x1, y1, z1], [x2, y2, z2], ...]
        :param threshold: Acceptable deviation from complete coplanarity (default is 1e-5).
        :return: True if all vectors are roughly coplanar, False otherwise.
        """
        if len(vectors) < 3:
            # Fewer than 3 vectors are always coplanar by definition.
            return True

        vectors = [np.array(v) for v in vectors]

        # Compute the normal vector of the first two vectors via cross product
        normal = np.cross(vectors[0], vectors[1])

        # If the normal vector is zero (parallel vectors), return False unless all vectors are collinear
        if np.linalg.norm(normal) == 0:
            # Check if all vectors are collinear instead of coplanar
            for i in range(2, len(vectors)):
                # Cross product should still be near zero with collinear vectors
                if np.linalg.norm(np.cross(vectors[0], vectors[i])) >= threshold:
                    return False
            return True

        # Check subsequent vectors to see if they are in the same plane
        for i in range(2, len(vectors)):
            # Compute the normal vector for the plane formed by one of the original vectors and a new vector
            new_normal = np.cross(vectors[0], vectors[i])

            # Measure the deviation between the original plane normal and the new normal
            # Specifically, take the norm of the new cross-product and ensure it's near-zero
            if np.linalg.norm(new_normal) == 0:
                continue  # perfectly coplanar with this vector
            angle_difference = np.dot(normal, new_normal) / (np.linalg.norm(normal) * np.linalg.norm(new_normal))

            # If the angle difference is significant (outside threshold), their normals are not aligned -> non-coplanar
            # Clamp the value within the valid range of arccos (-1 to 1) to avoid numerical issues
            angle_difference = np.clip(angle_difference, -1.0, 1.0)

            # Use cosine similarity to ensure they are nearly parallel
            angle = np.arccos(angle_difference)

            # When the angle is too large (meaning normals differ significantly), the vectors are not coplanar
            if abs(angle) > threshold:
                return False

        return True
class MatrixUtils:
    @staticmethod
    def is_unitary(m):
        return np.allclose(np.eye(m.shape[0]), np.dot(m.T, m))

    @staticmethod
    def givens_rotation_matrix(size, i, j, theta):
        """Constructs a Givens rotation matrix that rotates between rows/columns i and j."""
        G = np.eye(size)
        G[i, i] = np.cos(theta)
        G[j, j] = np.cos(theta)
        G[i, j] = -np.sin(theta)
        G[j, i] = np.sin(theta)
        return G

    @staticmethod
    def gram_schmidt(vectors, tolerance=1e-8):
        """
        Modified Gram-Schmidt process to orthogonalize vectors, handling collinear cases.

        :param vectors: List of input vectors (Nx3).
        :param tolerance: Tolerance for determining zero vectors after projection.
        :return: Orthogonalized vectors, guaranteed to have the same number of rows as input.
        """
        orthogonal = []
        for i, v in enumerate(vectors):
            # Subtract projections onto previous orthogonal vectors
            for u in orthogonal:
                v -= np.dot(u, v) * u  # Subtract the component of v along u

            norm_v = np.linalg.norm(v)
            if norm_v < tolerance:
                # Handle collinear case: If two vectors are collinear, restore the second as the opposite
                v = -orthogonal[0]  # Flip the direction of the first vector
                norm_v = np.linalg.norm(v)

            orthogonal.append(v / norm_v)  # Normalize the vector

        return np.array(orthogonal)

    @staticmethod
    def kabsch_algorithm(P, Q):
        """
        Computes the optimal rotation matrix to align point clouds P and Q using the Kabsch algorithm.

        :param P: Original points (Nx3 array)
        :param Q: Target points (Nx3 array)
        :return: Rotation matrix R (3x3)
        """
        # Center the point clouds at the origin
        P_mean = np.mean(P, axis=0)
        Q_mean = np.mean(Q, axis=0)
        P_centered = P - P_mean
        Q_centered = Q - Q_mean

        # Compute the covariance matrix
        C = np.dot(P_centered.T, Q_centered)

        # Singular value decomposition (SVD) of the covariance matrix
        U, S, Vt = np.linalg.svd(C)

        # Compute the rotation matrix (enforcing proper rotation if necessary)
        R = np.dot(U, Vt)
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1  # Flip the last column of U
            R = np.dot(U, Vt)

        return R

    @staticmethod
    def kabsch_algorithm_with_sign_correction(P, Q):
        """
        Extended Kabsch Algorithm to include sign correction.
        :param P: Source point set (Nx3).
        :param Q: Target point set (Nx3).
        :return: Rotation matrix R.
        """
        H = np.dot(P.T, Q)  # Covariance matrix
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(np.dot(Vt.T, U.T))  # Determinant of rotation matrix
        D = np.eye(3)  # Identity matrix
        if d < 0:
            D[2, 2] = -1  # Correct for reflection
        R = np.dot(np.dot(Vt.T, D), U.T)  # Adjusted rotation matrix
        return R

    @staticmethod
    def align_with_quaternions(P, Q):
        """
        Aligns vector sets P and Q using quaternions to find the optimal rotation matrix.

        :param P: Source vectors (Nx3 array)
        :param Q: Target vectors (Nx3 array)
        :return: Rotation matrix R (3x3)
        """
        assert len(P) == len(Q), "Both vector sets must have the same number of elements"

        # Find the quaternion that best aligns P to Q
        rotation, rmsd = R.align_vectors(Q, P)
        return rotation.as_matrix()

    @staticmethod
    def loewdin_orthogonalization(C, S):
        """
        Apply Loewdin orthogonalization to ensure the matrix C is unitary in the basis defined by overlap matrix S.

        Parameters:
        ----------
        C : ndarray
            Initial (non-unitary) guess matrix.
        S : ndarray
            Overlap matrix from the molecular basis.

        Returns:
        -------
        C_orth : ndarray
            Orthonormalized coefficient matrix.
        """
        # Eigen decomposition of S
        sv, U = np.linalg.eigh(S)

        # Construct S^(-1/2)
        S_inv_sqrt = U @ np.diag(1.0 / np.sqrt(sv)) @ U.T

        # Apply transformation to orthogonalize C
        C_orth = S_inv_sqrt @ C

        return C_orth

    @staticmethod
    def make_unitary(C):
        """
        Ensure that a matrix C is unitary by performing QR decomposition.

        Parameters:
        ----------
        C : ndarray
            Input (possibly non-unitary) matrix.

        Returns:
        -------
        Q : ndarray
            Strictly unitary matrix.
        """
        Q, R = np.linalg.qr(C)
        return Q
class HybridizationUtils:
    @staticmethod
    def procrustes_alignment(align_source, align_target, gram_correction=True):
        """
        Aligns vectors using the orthogonal Procrustes algorithm.
        :param align_source: Source vector set (Nx3).
        :param align_target: Target vector set (Nx3).
        :param gram_correction: Apply Gram-Schmidt orthogonalization after alignment.
        :return: Rotated source vector set.
        """
        R, _ = orthogonal_procrustes(
            align_source,
            align_target
        )

        # Rotate the px, py, pz components of the hybrid orbitals
        rotated = np.dot(
            align_source,
            R.T
        )

        if gram_correction: rotated = MatrixUtils.gram_schmidt(rotated)  # Re-orthogonalize/normalize directions
        return rotated

    @staticmethod
    def kabsch_alignment(align_source, align_target, gram_correction=False):
        """
        Aligns vectors using the Kabsch algorithm (pure rotation).
        :param align_source: Source vector set (Nx3).
        :param align_target: Target vector set (Nx3).
        :param gram_correction: Apply Gram-Schmidt orthogonalization after alignment.
        :return: Rotated source vector set.
        """
        R = MatrixUtils.kabsch_algorithm_with_sign_correction(align_source, align_target)  # Get pure rotation matrix
        rotated = np.dot(align_source, R.T)  # Apply transformation

        if gram_correction:
            rotated = MatrixUtils.gram_schmidt(rotated)  # Re-orthogonalize/normalize directions

        return rotated
    
    @staticmethod
    def get_kabsch_alignment_matrix(align_source, align_target, gram_correction=False)->np.ndarray:
        """
        Aligns vectors using the Kabsch algorithm (pure rotation).
        :param align_source: Source vector set (Nx3).
        :param align_target: Target vector set (Nx3).
        :return: Rotation Matrix to apply as np.dot(align_source, Rotation_matrix.T).
        """
        return  MatrixUtils.kabsch_algorithm_with_sign_correction(align_source, align_target) 

    @staticmethod
    def quaternion_alignment(align_source, align_target, gram_correction=True):
        """
        Aligns vectors using quaternions (via scipy's R.align_vectors).
        :param align_source: Source vector set (Nx3).
        :param align_target: Target vector set (Nx3).
        :param gram_correction: Apply Gram-Schmidt orthogonalization after alignment.
        :return: Rotated source vector set.
        """
        assert len(align_source) == len(align_target), "Source and target must have the same number of points."

        # Find the rotation matrix using scipy's quaternion-based vector alignment
        rotation, rmsd = R.align_vectors(align_target, align_source)  # align_vectors aligns src -> tgt
        R_matrix = rotation.as_matrix()

        # Rotate the source vectors
        rotated = np.dot(align_source, R_matrix.T)

        if gram_correction:
            rotated = MatrixUtils.gram_schmidt(rotated)  # Re-orthogonalize/normalize directions

        return rotated

    def correct_signs(rotated, target):
        # Loop over each pair of vectors
        for i in range(len(rotated)):
            if np.dot(rotated[i], target[i]) < 0:  # Check if signs are flipped
                rotated[i] *= -1  # Flip the sign of the rotated vector
        return rotated
class AtomUtils:
    max_electrons_by_shell = [2, 8, 18, 32, 32, 18, 8]
class XYZUtils:
    def cleanup(xyz_string):
        lines = [line.strip() for line in xyz_string.strip().splitlines()]
        for i, line in enumerate(lines):
            if len(line) != 0:  # Found the first line with content
                if ' ' in line:
                    # Join the remaining lines
                    return '\n'.join(lines[i:])
                else:  # Starts with number and comment lines
                    # Remove the first 2 lines with content, and join the remaining lines
                    return '\n'.join(lines[i + 2:])
        return ''  # If no lines with spaces exist, return an empty string