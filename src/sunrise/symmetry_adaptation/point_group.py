from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy
from numpy.typing import NDArray
from pandas import DataFrame


@dataclass(frozen=True)
class IrreducibleRepresentation:
    """An irreducible representation of a group."""

    mulliken_symbol: str
    characters: NDArray[numpy.complex128]


@dataclass(frozen=True)
class CharacterTable:
    """The character table of a point group."""

    irrep_data: list[IrreducibleRepresentation]
    operation_symbols: list[str]

    @property
    def dict(self) -> dict[str, NDArray[numpy.complex128]]:
        """A dictionary mapping Mulliken symbols to irreducible representations."""

        return {irrep.mulliken_symbol: irrep.characters for irrep in self.irrep_data}

    @property
    def dataframe(self) -> DataFrame:
        """A pandas DataFrame representation of the character table."""

        return DataFrame(
            data=[irrep.characters for irrep in self.irrep_data],
            index=[irrep.mulliken_symbol for irrep in self.irrep_data],
            columns=self.operation_symbols,
        )

    @property
    def matrix(self) -> NDArray[numpy.complex128]:
        """A NumPy array representation of the character table."""

        return numpy.array([irrep.characters for irrep in self.irrep_data])

    @property
    def irreps(self) -> list[str]:
        """A list of the irreducible representations of the character table."""

        return [irrep.mulliken_symbol for irrep in self.irrep_data]

    def vec_to_str(self, vec: NDArray[numpy.complex128]) -> str | None:
        """Given a vector of characters, return the corresponding Mulliken symbol."""

        for irrep in self.irrep_data:
            if numpy.allclose(irrep.characters, vec):
                return irrep.mulliken_symbol

        return None


@dataclass
class PointGroup:
    schoenflies_label: str
    character_table: CharacterTable

    @property
    def order(self) -> int:
        """The order of the point group, i.e. the number of symmetry operations."""

        return len(self.character_table.operation_symbols)

    @classmethod
    def from_pyscf(cls, point_group_name: str) -> PointGroup:
        """Construct a PointGroup from with PySCF as source of truth."""

        from pyscf.symm import param

        ops = param.OPERATOR_TABLE[
            point_group_name
        ]  # column headers, e.g. ['E', 'C2z', 'sz', 'sx']
        ct = param.CHARACTER_TABLE[
            point_group_name
        ]  # rows: [irrep_name, char1, char2, ...]

        return PointGroup(
            schoenflies_label=point_group_name,
            character_table=CharacterTable(
                [IrreducibleRepresentation(x[0], [*x[1:]]) for x in ct],
                operation_symbols=ops,
            ),
        )


# The Abelian point groups are taken from PySCF with PointGroup.from_pyscf, the
# non-Abelian ones below are not in PySCF and are defined here. They are written as
# the published class table plus the class sizes, and expanded to one column per
# operation, which is what SymmetryAdaptedLinearCombintationSymmetrization expects:
# it forms
#     |psi_irrep> = sum_operations chi_irrep(op) . O_op |psi>
# and every operation in a class carries that class's character.
#
# The degenerate irreps (E, T) work there too - the character projector lands on
# the whole carrier space of the irrep, and the symmetrizer orthonormalizes within
# each irrep sector afterwards. Note that LocalizedIrrepProvider rounds <psi|O|psi>
# assuming +/-1, so it can only LABEL one-dimensional irreps.

# D4h. Convention: the C4 axis is z, the molecule lies in the xy-plane. C2' are taken
# along x and y, C2'' along the diagonals y = +/- x. For a square whose atoms sit
# on the diagonals the usual convention is the opposite assignment, which swaps
# the B1g/B2g and B1u/B2u labels - the projected states are the same either way,
# only the names differ.
_D4H_CLASS_SIZES = [1, 2, 1, 2, 2, 1, 2, 1, 2, 2]
#                     E 2C4 C2 2C2' 2C2'' i 2S4 sh 2sv 2sd
_D4H_CLASS_CHARACTERS = {
    "A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "A2g": [1, 1, 1, -1, -1, 1, 1, 1, -1, -1],
    "B1g": [1, -1, 1, 1, -1, 1, -1, 1, 1, -1],
    "B2g": [1, -1, 1, -1, 1, 1, -1, 1, -1, 1],
    "Eg": [2, 0, -2, 0, 0, 2, 0, -2, 0, 0],
    "A1u": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
    "A2u": [1, 1, 1, -1, -1, -1, -1, -1, 1, 1],
    "B1u": [1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
    "B2u": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
    "Eu": [2, 0, -2, 0, 0, -2, 0, 2, 0, 0],
}
PointGroup.D4h = PointGroup(
    schoenflies_label="D4h",
    character_table=CharacterTable(
        [
            # every operation of a class has the character of the class
            IrreducibleRepresentation(x, numpy.repeat(y, _D4H_CLASS_SIZES))
            for (x, y) in _D4H_CLASS_CHARACTERS.items()
        ],
        operation_symbols=[
            "E",
            "C4",
            "C4^3",
            "C2z",
            "C2x",
            "C2y",
            "C2a",
            "C2b",
            "i",
            "S4",
            "S4^3",
            "sz",
            "sy",
            "sx",
            "sda",
            "sdb",
        ],
    ),
)

# D6h. Convention: the C6 axis is z and the atoms lie at 0, 60, ... 300 degrees in
# the xy-plane, which is what `vbsetup.ring(6, R)` produces. C2' then passes
# through opposite atoms and C2'' through opposite edge midpoints; sigma_v
# contains the atoms and sigma_d bisects them. Choosing the opposite convention
# swaps the B1/B2 labels - the projected subspaces are identical either way.
#
# Six electrons on a ring is the 4n+2 (aromatic) case, so unlike the square the
# SYMMETRIC Kekule combination is the totally symmetric one here.
_D6H_CLASS_SIZES = [1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3]
#                              E 2C6 2C3  C2 3C2' 3C2''  i 2S3 2S6  sh 3sd 3sv
_D6H_CLASS_CHARACTERS = {
    "A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "A2g": [1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
    "B1g": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
    "B2g": [1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
    "E1g": [2, 1, -1, -2, 0, 0, 2, 1, -1, -2, 0, 0],
    "E2g": [2, -1, -1, 2, 0, 0, 2, -1, -1, 2, 0, 0],
    "A1u": [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
    "A2u": [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
    "B1u": [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1],
    "B2u": [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1],
    "E1u": [2, 1, -1, -2, 0, 0, -2, -1, 1, 2, 0, 0],
    "E2u": [2, -1, -1, 2, 0, 0, -2, 1, 1, -2, 0, 0],
}
PointGroup.D6h = PointGroup(
    schoenflies_label="D6h",
    character_table=CharacterTable(
        [
            # every operation of a class has the character of the class
            IrreducibleRepresentation(x, numpy.repeat(y, _D6H_CLASS_SIZES))
            for (x, y) in _D6H_CLASS_CHARACTERS.items()
        ],
        operation_symbols=[
            "E",
            "C6",
            "C6^5",
            "C3",
            "C3^2",
            "C2z",
            "C2'1",
            "C2'2",
            "C2'3",
            "C2''1",
            "C2''2",
            "C2''3",
            "i",
            "S3",
            "S3^5",
            "S6",
            "S6^5",
            "sz",
            "sd1",
            "sd2",
            "sd3",
            "sv1",
            "sv2",
            "sv3",
        ],
    ),
)

# D8h. Convention: the C8 axis is z and the atoms lie at 0, 45, ... 315 degrees in
# the xy-plane. C2' then passes through opposite atoms (along x, y and the
# diagonals) and C2'' through opposite edge midpoints (22.5, 67.5, ... degrees);
# sigma_v contains the atoms and sigma_d bisects them. Choosing the opposite
# convention swaps the B1/B2 labels.
_D8H_CLASS_SIZES = [1, 2, 2, 2, 1, 4, 4, 1, 2, 2, 2, 1, 4, 4]
_R2 = numpy.sqrt(2)
#                         E   2C8  2C8³ 2C4  C2 4C2' 4C2''  i 2S8³  2S8 2S4  sh 4sv 4sd
_D8H_CLASS_CHARACTERS = {
    "A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "A2g": [1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1],
    "B1g": [1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1],
    "B2g": [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1],
    "E1g": [2, _R2, -_R2, 0, -2, 0, 0, 2, _R2, -_R2, 0, -2, 0, 0],
    "E2g": [2, 0, 0, -2, 2, 0, 0, 2, 0, 0, -2, 2, 0, 0],
    "E3g": [2, -_R2, _R2, 0, -2, 0, 0, 2, -_R2, _R2, 0, -2, 0, 0],
    "A1u": [1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
    "A2u": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
    "B1u": [1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1],
    "B2u": [1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1],
    "E1u": [2, _R2, -_R2, 0, -2, 0, 0, -2, -_R2, _R2, 0, 2, 0, 0],
    "E2u": [2, 0, 0, -2, 2, 0, 0, -2, 0, 0, 2, -2, 0, 0],
    "E3u": [2, -_R2, _R2, 0, -2, 0, 0, -2, _R2, -_R2, 0, 2, 0, 0],
}
PointGroup.D8h = PointGroup(
    schoenflies_label="D8h",
    character_table=CharacterTable(
        [
            # every operation of a class has the character of the class
            IrreducibleRepresentation(x, numpy.repeat(y, _D8H_CLASS_SIZES))
            for (x, y) in _D8H_CLASS_CHARACTERS.items()
        ],
        operation_symbols=[
            "E",
            "C8",
            "C8^7",
            "C8^3",
            "C8^5",
            "C4",
            "C4^3",
            "C2z",
            "C2x",
            "C2a",
            "C2y",
            "C2b",
            "C2c",
            "C2d",
            "C2e",
            "C2f",
            "i",
            "S8^3",
            "S8^5",
            "S8",
            "S8^7",
            "S4",
            "S4^3",
            "sz",
            "sy",
            "sda",
            "sx",
            "sdb",
            "sdc",
            "sdd",
            "sde",
            "sdf",
        ],
    ),
)

# Td (e.g. CH4). Convention: the three C2 (and S4) axes are x, y and z, so the four outer atoms
# sit on alternating corners of a cube, e.g. (1, 1, 1), (-1, -1, 1), (-1, 1, -1)
# and (1, -1, -1). The C3 axes are then the body diagonals through these four
# corners (a, b, c, d in this order) and the six mirror planes contain one axis
# and bisect the other two: sda, sdb from D4h (x = +/-y), sdy1, sdy2 (x = +/-z)
# and sdx1, sdx2 (y = +/-z).
_TD_CLASS_SIZES = [1, 8, 3, 6, 6]
#                 E 8C3 3C2 6S4 6sd
_TD_CLASS_CHARACTERS = {
    "A1": [1, 1, 1, 1, 1],
    "A2": [1, 1, 1, -1, -1],
    "E": [2, -1, 2, 0, 0],
    "T1": [3, 0, -1, 1, -1],
    "T2": [3, 0, -1, -1, 1],
}
PointGroup.Td = PointGroup(
    schoenflies_label="Td",
    character_table=CharacterTable(
        [
            # every operation of a class has the character of the class
            IrreducibleRepresentation(x, numpy.repeat(y, _TD_CLASS_SIZES))
            for (x, y) in _TD_CLASS_CHARACTERS.items()
        ],
        operation_symbols=[
            "E",
            "C3a",
            "C3a^2",
            "C3b",
            "C3b^2",
            "C3c",
            "C3c^2",
            "C3d",
            "C3d^2",
            "C2x",
            "C2y",
            "C2z",
            "S4x",
            "S4x^3",
            "S4y",
            "S4y^3",
            "S4",
            "S4^3",
            "sda",
            "sdb",
            "sdy1",
            "sdy2",
            "sdx1",
            "sdx2",
        ],
    ),
)


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
        }

        operations: dict[str, NDArray[numpy.complex128]] = {}
        for op in pg.character_table.operation_symbols:
            if op not in operation_dict:
                raise ValueError(
                    f"Operation {op} not found. The supported operations are {list(operation_dict.keys())}."
                )

            operations[op] = operation_dict[op]

        return PointGroupRepresentation(operations)
