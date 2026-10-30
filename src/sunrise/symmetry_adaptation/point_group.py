from __future__ import annotations

from dataclasses import dataclass

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

# Oh (e.g. SF6). Oh = Td x Ci, so its operations are the ones of Td and their
# products with the inversion.
#
# Convention: the C4 axes are x, y and z, so the six outer atoms of an octahedron
# sit on the axes. The C3 axes are the body diagonals of Td, the C2 axes in the
# 6C2 class are the face diagonals (C2a, C2b in the xy-plane and the axes
# perpendicular to the Td mirror planes sdy1, sdy2, sdx1, sdx2).
_OH_CLASS_SIZES = [1, 8, 6, 6, 3, 1, 6, 8, 3, 6]
#                  E 8C3 6C2 6C4 3C2  i 6S4 8S6 3sh 6sd
_OH_CLASS_CHARACTERS = {
    "A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "A2g": [1, 1, -1, -1, 1, 1, -1, 1, 1, -1],
    "Eg": [2, -1, 0, 0, 2, 2, 0, -1, 2, 0],
    "T1g": [3, 0, -1, 1, -1, 3, 1, 0, -1, -1],
    "T2g": [3, 0, 1, -1, -1, 3, -1, 0, -1, 1],
    "A1u": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
    "A2u": [1, 1, -1, -1, 1, -1, 1, -1, -1, 1],
    "Eu": [2, -1, 0, 0, 2, -2, 0, 1, -2, 0],
    "T1u": [3, 0, -1, 1, -1, -3, -1, 0, 1, 1],
    "T2u": [3, 0, 1, -1, -1, -3, 1, 0, 1, -1],
}
PointGroup.Oh = PointGroup(
    schoenflies_label="Oh",
    character_table=CharacterTable(
        [
            # every operation of a class has the character of the class
            IrreducibleRepresentation(x, numpy.repeat(y, _OH_CLASS_SIZES))
            for (x, y) in _OH_CLASS_CHARACTERS.items()
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
            "C2a",
            "C2b",
            "C2dy1",
            "C2dy2",
            "C2dx1",
            "C2dx2",
            "C4",
            "C4^3",
            "C4x",
            "C4x^3",
            "C4y",
            "C4y^3",
            "C2z",
            "C2x",
            "C2y",
            "i",
            "S4",
            "S4^3",
            "S4x",
            "S4x^3",
            "S4y",
            "S4y^3",
            "S6a",
            "S6a^5",
            "S6b",
            "S6b^5",
            "S6c",
            "S6c^5",
            "S6d",
            "S6d^5",
            "sz",
            "sy",
            "sx",
            "sda",
            "sdb",
            "sdy1",
            "sdy2",
            "sdx1",
            "sdx2",
        ],
    ),
)

# C3v (e.g. NH3, CH3Cl), D3h (e.g. BF3, H3+, PF5) and D3d (e.g. staggered ethane,
# chair cyclohexane) reuse the threefold operations of D6h.
#
# Convention: the C3 axis is z and the atoms lie at 0, 120 and 240 degrees around it
# (seen from above), so one sigma_v is the xz-plane. For D3h the C2' axes then pass
# through the atoms. For D3d the upper atoms sit at 0, 120, 240 degrees and the lower
# ones at 60, 180, 300 degrees: the sigma_d contain the atoms (sv1, sv2, sv3 of D6h)
# and the C2' axes lie between them (C2''1, C2''2, C2''3 of D6h).
_C3V_CLASS_SIZES = [1, 2, 3]
#                  E 2C3 3sv
_C3V_CLASS_CHARACTERS = {
    "A1": [1, 1, 1],
    "A2": [1, 1, -1],
    "E": [2, -1, 0],
}
PointGroup.C3v = PointGroup(
    schoenflies_label="C3v",
    character_table=CharacterTable(
        [
            # every operation of a class has the character of the class
            IrreducibleRepresentation(x, numpy.repeat(y, _C3V_CLASS_SIZES))
            for (x, y) in _C3V_CLASS_CHARACTERS.items()
        ],
        operation_symbols=["E", "C3", "C3^2", "sv1", "sv2", "sv3"],
    ),
)

_D3H_CLASS_SIZES = [1, 2, 3, 1, 2, 3]
#                    E 2C3 3C2' sh 2S3 3sv
_D3H_CLASS_CHARACTERS = {
    "A1'": [1, 1, 1, 1, 1, 1],
    "A2'": [1, 1, -1, 1, 1, -1],
    "E'": [2, -1, 0, 2, -1, 0],
    "A1''": [1, 1, 1, -1, -1, -1],
    "A2''": [1, 1, -1, -1, -1, 1],
    "E''": [2, -1, 0, -2, 1, 0],
}
PointGroup.D3h = PointGroup(
    schoenflies_label="D3h",
    character_table=CharacterTable(
        [
            # every operation of a class has the character of the class
            IrreducibleRepresentation(x, numpy.repeat(y, _D3H_CLASS_SIZES))
            for (x, y) in _D3H_CLASS_CHARACTERS.items()
        ],
        operation_symbols=[
            "E",
            "C3",
            "C3^2",
            "C2'1",
            "C2'2",
            "C2'3",
            "sz",
            "S3",
            "S3^5",
            "sv1",
            "sv2",
            "sv3",
        ],
    ),
)

_D3D_CLASS_SIZES = [1, 2, 3, 1, 2, 3]
#                    E 2C3 3C2'  i 2S6 3sd
_D3D_CLASS_CHARACTERS = {
    "A1g": [1, 1, 1, 1, 1, 1],
    "A2g": [1, 1, -1, 1, 1, -1],
    "Eg": [2, -1, 0, 2, -1, 0],
    "A1u": [1, 1, 1, -1, -1, -1],
    "A2u": [1, 1, -1, -1, -1, 1],
    "Eu": [2, -1, 0, -2, 1, 0],
}
PointGroup.D3d = PointGroup(
    schoenflies_label="D3d",
    character_table=CharacterTable(
        [
            # every operation of a class has the character of the class
            IrreducibleRepresentation(x, numpy.repeat(y, _D3D_CLASS_SIZES))
            for (x, y) in _D3D_CLASS_CHARACTERS.items()
        ],
        operation_symbols=[
            "E",
            "C3",
            "C3^2",
            "C2''1",
            "C2''2",
            "C2''3",
            "i",
            "S6",
            "S6^5",
            "sv1",
            "sv2",
            "sv3",
        ],
    ),
)

# C4v (e.g. BrF5, XeOF4) and D2d (e.g. allene) are subgroups of D4h and Td built
# from their operations.
#
# Convention for C4v: the C4 axis is z, sigma_v are the xz- and yz-planes and
# sigma_d the planes x = +/-y, as in D4h.
# Convention for D2d: the S4 axis is z, the C2' axes are x and y and sigma_d the
# planes x = +/-y, so for allene the CH2 groups lie in the sigma_d planes. Choosing
# the opposite convention swaps the B1/B2 labels.
_C4V_CLASS_SIZES = [1, 2, 1, 2, 2]
#                  E 2C4 C2 2sv 2sd
_C4V_CLASS_CHARACTERS = {
    "A1": [1, 1, 1, 1, 1],
    "A2": [1, 1, 1, -1, -1],
    "B1": [1, -1, 1, 1, -1],
    "B2": [1, -1, 1, -1, 1],
    "E": [2, 0, -2, 0, 0],
}
PointGroup.C4v = PointGroup(
    schoenflies_label="C4v",
    character_table=CharacterTable(
        [
            # every operation of a class has the character of the class
            IrreducibleRepresentation(x, numpy.repeat(y, _C4V_CLASS_SIZES))
            for (x, y) in _C4V_CLASS_CHARACTERS.items()
        ],
        operation_symbols=["E", "C4", "C4^3", "C2z", "sy", "sx", "sda", "sdb"],
    ),
)

_D2D_CLASS_SIZES = [1, 2, 1, 2, 2]
#                  E 2S4 C2 2C2' 2sd
_D2D_CLASS_CHARACTERS = {
    "A1": [1, 1, 1, 1, 1],
    "A2": [1, 1, 1, -1, -1],
    "B1": [1, -1, 1, 1, -1],
    "B2": [1, -1, 1, -1, 1],
    "E": [2, 0, -2, 0, 0],
}
PointGroup.D2d = PointGroup(
    schoenflies_label="D2d",
    character_table=CharacterTable(
        [
            # every operation of a class has the character of the class
            IrreducibleRepresentation(x, numpy.repeat(y, _D2D_CLASS_SIZES))
            for (x, y) in _D2D_CLASS_CHARACTERS.items()
        ],
        operation_symbols=["E", "S4", "S4^3", "C2z", "C2x", "C2y", "sda", "sdb"],
    ),
)
