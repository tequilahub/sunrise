from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy
from numpy.typing import NDArray

from .point_group import PointGroup

T = TypeVar(
    "T"
)  # The type of the symmetry operation representation, e.g. a 3x3 matrix.
S = TypeVar(
    "S"
)  # The corresponding state that the symmetry operation acts on, e.g. a 3D vector.


@dataclass(frozen=True)
class PointGroupRepresentation(Generic[T, S]):
    operations: dict[str, T]
    application_function: Callable[[T, S], S] = lambda op, state: op @ state
    is_close_function: Callable[[S, S], bool] = lambda state1, state2: numpy.isclose(
        state1, state2
    ).all()

    def apply(self, operation: T, state: S) -> S:
        """Apply a symmetry operation to a state."""

        return self.application_function(operation, state)

    def character_vector(self, state: S) -> NDArray[numpy.complex128]:
        """
        Returns the character vector of the representation, i.e. the characters of the symmetry
        operations in the representation. These are assumed to be either 1 or -1 here.
        """

        # Compute the expectation value <state | op | state> for each operation.
        # For 1D irreps the eigenvalues are ±1, so rounding cleanly handles
        # numerical noise and global phases.
        return numpy.array(
            [
                numpy.round(state.inner(self.apply(op, state)).real)
                for op in self.operations.values()
            ]
        )

    @classmethod
    def get_geometric_representation(
        cls, pg: PointGroup
    ) -> PointGroupRepresentation[NDArray[numpy.complex128], NDArray[numpy.complex128]]:
        """Returns the geometric representation of a point group, i.e. where the symmetry operations are represented as 3x3 matrices and the states are 3D vectors."""

        operation_dict: dict[str, NDArray[numpy.float64]] = {
            "E": numpy.eye(3),
            "i": -1 * numpy.eye(3),
            "C2z": numpy.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "C2x": numpy.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            "C2y": numpy.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            "sz": numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            "sy": numpy.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "sx": numpy.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            # --- operations beyond the Abelian groups, needed for D4h ---
            # Fourfold rotations about z, and the S4 improper rotations.
            "C4": numpy.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            "C4^3": numpy.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
            "S4": numpy.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]]),
            "S4^3": numpy.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]]),
            # Twofold axes along the diagonals y = x and y = -x.
            "C2a": numpy.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
            "C2b": numpy.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
            # Mirror planes containing z and the diagonals.
            "sda": numpy.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            "sdb": numpy.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
            # --- D6h and D8h: rotation by theta about z, followed by the signs of (x, y, z):
            # C_n [1, 1, 1], S_n [1, 1, -1], C2 about the in-plane axis at angle theta / 2
            # [1, -1, -1], mirror plane containing z and the direction at theta / 2 [1, -1, 1] ---
            **{
                label: numpy.array(
                    [
                        [numpy.cos(theta), -numpy.sin(theta), 0.0],
                        [numpy.sin(theta), numpy.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                * signs
                for label, theta, signs in [
                    # D6h: for a hexagon whose atoms sit at 0, 60, ... 300 degrees, C2' passes
                    # through opposite vertices (0, 60, 120 degrees) and C2'' through opposite
                    # edge midpoints (30, 90, 150 degrees); sigma_v contains the atoms, sigma_d
                    # bisects them
                    ("C6", numpy.pi / 3, [1, 1, 1]),
                    ("C6^5", -numpy.pi / 3, [1, 1, 1]),
                    ("C3", 2 * numpy.pi / 3, [1, 1, 1]),
                    ("C3^2", -2 * numpy.pi / 3, [1, 1, 1]),
                    ("S6", numpy.pi / 3, [1, 1, -1]),
                    ("S6^5", -numpy.pi / 3, [1, 1, -1]),
                    ("S3", 2 * numpy.pi / 3, [1, 1, -1]),
                    ("S3^5", -2 * numpy.pi / 3, [1, 1, -1]),
                    ("C2'1", 0.0, [1, -1, -1]),
                    ("C2'2", 2 * numpy.pi / 3, [1, -1, -1]),
                    ("C2'3", 4 * numpy.pi / 3, [1, -1, -1]),
                    ("C2''1", numpy.pi / 3, [1, -1, -1]),
                    ("C2''2", numpy.pi, [1, -1, -1]),
                    ("C2''3", 5 * numpy.pi / 3, [1, -1, -1]),
                    ("sv1", 0.0, [1, -1, 1]),
                    ("sv2", 2 * numpy.pi / 3, [1, -1, 1]),
                    ("sv3", 4 * numpy.pi / 3, [1, -1, 1]),
                    ("sd1", numpy.pi / 3, [1, -1, 1]),
                    ("sd2", numpy.pi, [1, -1, 1]),
                    ("sd3", 5 * numpy.pi / 3, [1, -1, 1]),
                    # D8h: for an octagon whose atoms sit at 0, 45, ... 315 degrees, C2' and
                    # sigma_v are C2x, C2a, C2y, C2b and sy, sda, sx, sdb from D4h; C2'' and
                    # sigma_d lie between the atoms (22.5, 67.5, 112.5, 157.5 degrees)
                    ("C8", numpy.pi / 4, [1, 1, 1]),
                    ("C8^7", -numpy.pi / 4, [1, 1, 1]),
                    ("C8^3", 3 * numpy.pi / 4, [1, 1, 1]),
                    ("C8^5", -3 * numpy.pi / 4, [1, 1, 1]),
                    ("S8", numpy.pi / 4, [1, 1, -1]),
                    ("S8^7", -numpy.pi / 4, [1, 1, -1]),
                    ("S8^3", 3 * numpy.pi / 4, [1, 1, -1]),
                    ("S8^5", -3 * numpy.pi / 4, [1, 1, -1]),
                    ("C2c", numpy.pi / 4, [1, -1, -1]),
                    ("C2d", 3 * numpy.pi / 4, [1, -1, -1]),
                    ("C2e", 5 * numpy.pi / 4, [1, -1, -1]),
                    ("C2f", 7 * numpy.pi / 4, [1, -1, -1]),
                    ("sdc", numpy.pi / 4, [1, -1, 1]),
                    ("sdd", 3 * numpy.pi / 4, [1, -1, 1]),
                    ("sde", 5 * numpy.pi / 4, [1, -1, 1]),
                    ("sdf", 7 * numpy.pi / 4, [1, -1, 1]),
                ]
            },
            # --- Td: threefold axes along the body diagonals (1, 1, 1), (-1, -1, 1),
            # (-1, 1, -1) and (1, -1, -1), S4 about x and y ---
            # Mirror planes containing y (x = z, x = -z) and x (y = z, y = -z).
            "C3a": numpy.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
            "C3a^2": numpy.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
            "C3b": numpy.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
            "C3b^2": numpy.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
            "C3c": numpy.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
            "C3c^2": numpy.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
            "C3d": numpy.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
            "C3d^2": numpy.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
            "S4x": numpy.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            "S4x^3": numpy.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]]),
            "S4y": numpy.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]),
            "S4y^3": numpy.array([[0, 0, -1], [0, -1, 0], [1, 0, 0]]),
            "sdy1": numpy.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
            "sdy2": numpy.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]]),
            "sdx1": numpy.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            "sdx2": numpy.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]]),
            # --- Oh: fourfold axes along x and y, twofold face diagonals in the
            # xz- and yz-planes (perpendicular to sdy1, sdy2, sdx1, sdx2) and the S6
            # about the body diagonals (the C3 of Td times the inversion) ---
            "C4x": numpy.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            "C4x^3": numpy.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
            "C4y": numpy.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            "C4y^3": numpy.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
            "C2dy1": numpy.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),
            "C2dy2": numpy.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
            "C2dx1": numpy.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]),
            "C2dx2": numpy.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            "S6a": numpy.array([[0, -1, 0], [0, 0, -1], [-1, 0, 0]]),
            "S6a^5": numpy.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]]),
            "S6b": numpy.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]]),
            "S6b^5": numpy.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]]),
            "S6c": numpy.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]]),
            "S6c^5": numpy.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]]),
            "S6d": numpy.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]),
            "S6d^5": numpy.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]]),
        }

        operations: dict[str, NDArray[numpy.complex128]] = {}
        for op in pg.character_table.operation_symbols:
            if op not in operation_dict:
                raise ValueError(
                    f"Operation {op} not found. The supported operations are {list(operation_dict.keys())}."
                )

            operations[op] = operation_dict[op]

        return PointGroupRepresentation(operations)
