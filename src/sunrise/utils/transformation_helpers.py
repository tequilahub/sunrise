from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy
import tequila
from numpy.typing import NDArray

if TYPE_CHECKING:
    # only for type hints, importing it at runtime would be circular
    from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase


class GeometricTransformationHelper:
    """A helper class for transformations from symmetry operations in 3D space on molecules around a given center."""

    # This function is specific to the STO-3g base
    @classmethod
    def num_active_spatial_orbitals(cls, charges: list[int]) -> list[int]:
        """
        Returns an ordered list of active spin orbital counts, given only
        the list of atomic charges (atomic numbers) of the molecule's atoms.

        """
        core_per_period = {1: 0, 2: 1, 3: 5, 4: 9, 5: 18}

        def get_period(charge):
            if charge <= 2:
                return 1
            if charge <= 10:
                return 2
            if charge <= 18:
                return 3
            if charge <= 36:
                return 4
            if charge <= 54:
                return 5
            return 6

        def minimal_basis_ao_count(charge):
            """
            Number of spatial AOs in the minimal (STO-3G) basis for a given element.
            Equals the number of occupied shells in the neutral ground-state atom.
            """
            if charge <= 2:
                return 1  # 1s
            if charge <= 4:
                return 2  # 1s 2s        (Li, Be)
            if charge <= 10:
                return 5  # 1s 2s 2p     (B–Ne)
            if charge <= 12:
                return 6  # + 3s          (Na, Mg)
            if charge <= 18:
                return 9  # + 3s 3p      (Al–Ar)
            if charge <= 20:
                return 10  # + 4s          (K, Ca)
            if charge <= 30:
                return 15  # + 3d          (Sc–Zn)
            if charge <= 36:
                return 18  # + 4p          (Ga–Kr)
            return 18  # extend as needed

        result = []
        for charge in charges:
            period = get_period(charge)
            n_core = core_per_period.get(period, 0)
            n_total = minimal_basis_ao_count(charge)
            n_active = max(n_total - n_core, 0)
            result.append(n_active)

        return result

    @classmethod
    def homogeneous_operation(
        cls, M: NDArray[numpy.float64], center: NDArray[numpy.float64] | None = None
    ) -> NDArray[numpy.float64]:
        """
        Construct the homogeneous representation of a linear operation.

        Homogeneous coordinates embed a 3D affine transformation in a
        higher-dimensional linear space by representing points as
        ``[x, y, z, 1]``. This allows rotations, reflections, and the
        translations required to perform them about an arbitrary center to
        be expressed as a single 4×4 matrix.

        Parameters
        ----------
        M
                3×3 matrix representing a linear operation in Cartesian space.
        center
                Point about which the operation is applied.
        """
        if center is None:
            center = numpy.zeros(3)
        M_h = numpy.eye(4)
        M_h[:3, :3] = M
        M_h[:3, 3] = center - (M @ center)
        return M_h

    @classmethod
    def homogeneous_coordinates(
        cls, v: NDArray[numpy.float64], is_row_vector: bool = True
    ) -> NDArray[numpy.float64]:
        """
        Convert Cartesian coordinates to homogeneous coordinates.

        Homogeneous coordinates represent a point ``[x, y, z]`` as
        ``[x, y, z, 1]``, embedding 3D affine geometry in a 4D linear
        space. In this representation, affine transformations (including
        translations) can be applied using matrix multiplication, just as
        ordinary linear transformations are applied in Cartesian space.

        Parameters
        ----------
        v
                A single 3-vector or an array of 3-vectors.
        is_row_vector
                If ``True``, vectors are stored as rows ``(N, 3)``; otherwise
                vectors are stored as columns ``(3, N)``.
        """

        v = numpy.asarray(v)

        # Single vector: [x, y, z] -> [x, y, z, 1]
        if v.ndim == 1:
            return numpy.append(v, 1.0)

        # Multiple vectors:
        if v.ndim == 2:
            if is_row_vector:
                # Treat rows as vectors: (N, 3) -> (N, 4)
                if v.shape[1] != 3:
                    raise ValueError(
                        f"Expected shape (N, 3) for row vectors, got {v.shape}."
                    )
                return numpy.hstack([v, numpy.ones((v.shape[0], 1))])
            else:
                # Treat columns as vectors: (3, N) -> (4, N)
                if v.shape[0] != 3:
                    raise ValueError(
                        f"Expected shape (3, N) for column vectors, got {v.shape}."
                    )
                return numpy.vstack([v, numpy.ones((1, v.shape[1]))])

        raise ValueError("Input must be a 1D 3-vector or a 2D array.")

    @classmethod
    def to_col_permutation(
        cls, A: NDArray[numpy.float64], B: NDArray[numpy.float64], tol: float = 1e-8
    ) -> NDArray[numpy.int_]:
        """
        Compute the column permutation mapping from ``A`` to ``B``.

        For each column in ``B``, this method finds the index of the matching
        column in ``A`` (within absolute tolerance ``tol``), and returns those
        indices as a permutation vector.

        Parameters
        ----------
        A : NDArray[numpy.float64]
                Reference matrix whose columns define the source ordering.
        B : NDArray[numpy.float64]
                Target matrix whose columns are matched against columns of ``A``.
        tol : float, optional
                Absolute tolerance used for element-wise column comparison, by default
                ``1e-8``.

        Returns
        -------
        NDArray[numpy.int_]
                Integer permutation vector ``p`` such that, column-wise,
                ``B[:, j] == A[:, p[j]]`` up to tolerance.

        Notes
        -----
        - This implementation assumes each column of ``B`` has at least one match
          in ``A`` within ``tol``.
        - In the special case where either ``A`` or ``B`` is the identity matrix
          (and the other is a compatible permutation matrix), this acts as the
          inverse relation of ``to_col_permutation``.
        """
        result = []
        for col in B.T:
            matches = numpy.where(numpy.all(numpy.abs(A - col[:, None]) < tol, axis=0))[
                0
            ]
            if len(matches) == 0:
                raise ValueError(
                    "No matching column found in A for column in B. Is the point group for the molecule correct?"
                )
            result.append(matches[0])
        return numpy.array(result)

    @classmethod
    def to_permutation_matrix(cls, perm: list[int]) -> NDArray[numpy.int_]:
        """
        Convert a permutation vector to a permutation matrix.

        This is the inverse operation of :meth:`to_col_permutation`: given the
        column-index permutation returned by ``to_col_permutation``, this method
        builds the corresponding permutation matrix.

        Parameters
        ----------
        perm : list[int]
                Permutation vector of length ``n`` where ``perm[i]`` is the target
                column index for row ``i``.

        Returns
        -------
        NDArray[numpy.int_]
                Permutation matrix ``P`` of shape ``(n, n)`` with exactly one ``1`` in
                each row and zeros elsewhere.
        """
        n = len(perm)
        P = numpy.zeros((n, n), dtype=int)
        P[numpy.arange(n), perm] = 1
        return P

    @classmethod
    def build_blockwise_permutation_matrix(
        cls, element_labels: list[int], label_perm: list[int]
    ) -> NDArray[numpy.int_]:
        """Build a block-wise permutation matrix from a permutation of block labels.

        In a block-wise permutation, elements are grouped into blocks by label, and
        entire blocks are reordered while preserving order within each block.

        Examples
        --------
        Given AO basis functions grouped by atom::

                labels = [0,0,0,0,0, 1,1,1,1,1, 2, 3]
                perm = [1, 0, 3, 2]

        the resulting matrix reorders rows so atom 1's block comes first, then
        atom 0, then atom 3, then atom 2.

        Parameters
        ----------
        element_labels : array-like
                Label per element indicating block membership, e.g.
                ``[0,0,0,0,0,1,1,1,1,1,2,3]``. Blocks are expected to be contiguous.
        label_perm : array-like
                Permutation of unique block labels, e.g. ``[1, 0, 3, 2]``.
                ``label_perm[i] = k`` means output block position ``i`` is filled by
                inumpyut block label ``k``.

        Returns
        -------
        numpy.ndarray
                Integer permutation matrix ``P`` with shape ``(n, n)``.
                Applying ``P @ v`` reorders a vector, and ``P @ C`` reorders matrix rows
                according to the block-wise permutation.
        """
        element_labels = numpy.asarray(element_labels)
        n = len(element_labels)

        # Map each label to its element indices (preserving intra-block order)
        label_to_indices = {}
        for idx, label in enumerate(element_labels):
            label_to_indices.setdefault(label, []).append(idx)

        # Concatenate blocks in the new label order to get the flat reordering
        new_order = []
        for label in label_perm:
            new_order.extend(label_to_indices[label])

        # P[new_pos, old_pos] = 1
        P = numpy.zeros((n, n), dtype=int)
        for new_pos, old_pos in enumerate(new_order):
            P[new_pos, old_pos] = 1

        return P

    @classmethod
    def perm_matrix_to_swaps(cls, P: NDArray[numpy.int_]) -> list[tuple[int, int]]:
        """
        Decompose an n×n permutation matrix into an ordered list of swaps.

        Parameters
        ----------
        P : numpy.ndarray
                P[i][j] = 1 -> output slot i draws from inumpyut slot j.

        Returns
        -------
        list[tuple[int, int]]
                [(i0, j0), (i1, j1), ...] such that applying the swaps in
                that order to [0, 1, ..., n-1] reproduces the permutation of P.
                Produces exactly n − (number of cycles) swaps — the minimum possible.
        """
        n = P.shape[0]
        assert P.shape == (n, n)
        assert numpy.allclose(P.sum(0), 1) and numpy.allclose(P.sum(1), 1), (
            "Not a valid permutation matrix"
        )

        perm = numpy.argmax(P, axis=1).tolist()  # which slot to pull from

        arr = list(range(n))  # arr[i]  = original element currently at slot i
        pos = list(range(n))  # pos[x]  = current slot of original element x
        swaps = []

        for i in range(n):
            target = perm[i]  # original element that belongs at slot i
            if arr[i] != target:
                j = pos[target]  # where that element currently sits
                swaps.append((i, j))
                old = arr[i]
                arr[i], arr[j] = arr[j], arr[i]
                pos[old] = j
                pos[target] = i

        return swaps


# This is really inefficient but will work for now
@dataclass
class FermionicSWAP:
    mol: QuantumChemistryBase

    def f(self, i: int, j: int) -> tequila.QCircuit:
        """
        This is a helper function for fermionic_swap.

        The fermionic swap (SWAP followed by CZ) of the two adjacent qubit
        modes (i, j), built from 2 CNOTs:
                |00> -> |00>, |01> -> |10>, |10> -> |01>, |11> -> -|11>

        Parameters
        ----------
        i : int
                First qubit index.
        j : int
                Second qubit index.

        Author
        ------
        Francisco Javier Del Arco Santos
        """
        return (
            tequila.gates.H(target=i)
            + tequila.gates.CNOT(target=j, control=i)
            + tequila.gates.CNOT(target=i, control=j)
            + tequila.gates.H(target=j)
        )

    def f_swap(self, i: int, j: int) -> tequila.QCircuit:
        """
        A function that swaps two orbitals (i, j) while accounting for fermionic antisymmetry.

        Note
        ----
        This implementation assumes that the orbitals are ordered as
        [up0, down0, up1, down1, ...] so that the spin-up and spin-down qubits
        for each orbital are adjacent.

        Parameters
        ----------
        i : int
                Index of the first orbital.
        j : int
                Index of the second orbital.

        Returns
        -------
        tequila.QCircuit
                A circuit that will perform the specified fermionic swap.

        Examples
        --------
        For example fermionic_swap(0,1) would perform the following transformations:
                +|0011> -> +|1100>
                +|0101> -> -|0101>

        Author
        ------
        Francisco Javier Del Arco Santos
        """
        U = tequila.QCircuit()
        for k in range(i, j):
            U += (
                self.f(2 * k + 1, 2 * (k + 1))
                + self.f(2 * k, 2 * k + 1)
                + self.f(2 * (k + 1), 2 * (k + 1) + 1)
                + self.f(2 * k + 1, 2 * (k + 1))
            )
        for k in range(i, j - 1)[::-1]:
            U += (
                self.f(2 * k + 1, 2 * (k + 1))
                + self.f(2 * k, 2 * k + 1)
                + self.f(2 * (k + 1), 2 * (k + 1) + 1)
                + self.f(2 * k + 1, 2 * (k + 1))
            )
        return U
