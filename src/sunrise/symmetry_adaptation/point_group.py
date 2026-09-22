from typing import TypeVar, Generic, Callable
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy
from pandas import DataFrame


# =====================================================================
# Geometric primitives for the higher-order axes (D4h, D6h)
#
# All are 3x3 matrices acting on (x, y, z) with the principal axis along z,
# so the molecule is expected to lie in the xy-plane. `_c2_in_plane` and
# `_mirror_v` take the in-plane angle of the C2 axis / of the direction the
# mirror plane contains, measured from +x.
# =====================================================================

def _rot_z(theta: float) -> NDArray[numpy.float64]:
	"""Proper rotation by `theta` about z."""
	c, s = numpy.cos(theta), numpy.sin(theta)
	return numpy.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _improper_z(theta: float) -> NDArray[numpy.float64]:
	"""S_n: rotation by `theta` about z followed by the horizontal mirror."""
	c, s = numpy.cos(theta), numpy.sin(theta)
	return numpy.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, -1.0]])


def _c2_in_plane(phi: float) -> NDArray[numpy.float64]:
	"""C2 about the in-plane axis at angle `phi` from +x."""
	c, s = numpy.cos(2 * phi), numpy.sin(2 * phi)
	return numpy.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def _mirror_v(phi: float) -> NDArray[numpy.float64]:
	"""Mirror plane containing z and the in-plane direction `phi`."""
	c, s = numpy.cos(2 * phi), numpy.sin(2 * phi)
	return numpy.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, 1.0]])


def _expand_class_table(class_chars: dict[str, list], class_sizes: list[int]) -> list:
	"""Repeat each class's character once per operation in that class.

	Published character tables list one column per CLASS, but
	SymmetryAdaptedLinearCombintationSymmetrization sums over individual
	OPERATIONS. Every operation in a class carries that class's character, so
	expanding by the class sizes gives the correct projector."""
	return [
		IrreducibleRepresentation(
			name,
			numpy.array([c for c, size in zip(chars, class_sizes) for _ in range(size)]))
		for name, chars in class_chars.items()
	]


# =====================================================================
# Irreps, character tables, point groups
# =====================================================================

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

		return { irrep.mulliken_symbol: irrep.characters for irrep in self.irrep_data }
	

	@property
	def dataframe(self) -> DataFrame:
		"""A pandas DataFrame representation of the character table."""
	
		return DataFrame(
			data=[ irrep.characters for irrep in self.irrep_data ],
			index=[ irrep.mulliken_symbol for irrep in self.irrep_data ],
			columns=self.operation_symbols
		)


	@property
	def matrix(self) -> NDArray[numpy.complex128]:
		"""A NumPy array representation of the character table."""

		return numpy.array([ irrep.characters for irrep in self.irrep_data ])

	@property
	def irreps(self) -> list[str]:
		"""A list of the irreducible representations of the character table."""

		return [ irrep.mulliken_symbol for irrep in self.irrep_data ]

	
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
	def from_pyscf(cls, point_group_name: str) -> 'PointGroup':
		"""Construct a PointGroup from with PySCF as source of truth."""

		from pyscf.symm import param

		ops = param.OPERATOR_TABLE[point_group_name]          # column headers, e.g. ['E', 'C2z', 'sz', 'sx']
		ct  = param.CHARACTER_TABLE[point_group_name]         # rows: [irrep_name, char1, char2, ...]

		return PointGroup(
			schoenflies_label=point_group_name,
			character_table=CharacterTable([IrreducibleRepresentation(x[0], [*x[1:]]) for x in ct],	operation_symbols=ops))

	
# These point group definitions follow the database given at
# http://gernot-katzers-spice-pages.com/character_tables/.
# There are differences in the ordering of the operations (columns)
# and the following differences in notations
#	DB	PySCF
#	C2	C2z
#	C2'	C2x
#	C2"	C2y
#	sh	sz
#	sv	sy
#	sd	sx

PointGroup.D2h = PointGroup(
		schoenflies_label="D2h",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("Ag",  [1,  1,  1,  1,  1,  1,  1,  1]),
				("B1g", [1,  1, -1, -1,  1,  1, -1, -1]),
				("B2g", [1, -1, -1,  1,  1, -1,  1, -1]),
				("B3g", [1, -1,  1, -1,  1, -1, -1,  1]),
				("Au",  [1,  1,  1,  1, -1, -1, -1, -1]),
				("B1u",	[1,  1, -1, -1, -1, -1,  1,  1]),
				("B2u",	[1, -1, -1,  1, -1,  1, -1,  1]),
				("B3u",	[1, -1,  1, -1, -1,  1,  1, -1]),
			]],	operation_symbols=["E", "C2", "C2\'", "C2\"", "i", "sh", "sv", "sd"]))

# D4h is non-Abelian, so PySCF cannot supply it - it is defined here in full.
# The table is written out one column PER OPERATION rather than per class, which
# is what SymmetryAdaptedLinearCombintationSymmetrization expects: it forms
#     |psi_irrep> = sum_operations chi_irrep(op) . O_op |psi>
# and every operation in a class carries that class's character, so summing over
# operations gives the correct projector.
#
# The two-dimensional irreps Eg and Eu work here too - the character projector
# lands on the whole carrier space of the irrep rather than a single vector, and
# the symmetrizer orthonormalizes within each irrep sector afterwards. Note that
# `PointGroupRepresentation.character_vector` rounds <psi|O|psi> assuming +/-1,
# so it can only LABEL one-dimensional irreps; degenerate Eg/Eu states will not
# be labelled correctly by that path.
#
# Convention: the C4 axis is z, the molecule lies in the xy-plane. C2' are taken
# along x and y, C2'' along the diagonals y = +/- x. For a square whose atoms sit
# on the diagonals the usual convention is the opposite assignment, which swaps
# the B1g/B2g and B1u/B2u labels - the projected states are the same either way,
# only the names differ.
PointGroup.D4h = PointGroup(
		schoenflies_label="D4h",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				#       E  C4 C4³ C2  C2x C2y C2a C2b   i  S4 S4³ sh  sv  sv' sda sdb
				("A1g", [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]),
				("A2g", [1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1]),
				("B1g", [1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1]),
				("B2g", [1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1]),
				("Eg",  [2,  0,  0, -2,  0,  0,  0,  0,  2,  0,  0, -2,  0,  0,  0,  0]),
				("A1u", [1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1]),
				("A2u", [1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1]),
				("B1u", [1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1, -1, -1,  1,  1]),
				("B2u", [1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1, -1]),
				("Eu",  [2,  0,  0, -2,  0,  0,  0,  0, -2,  0,  0,  2,  0,  0,  0,  0]),
			]],	operation_symbols=["E", "C4", "C4^3", "C2z", "C2x", "C2y", "C2a", "C2b",
							"i", "S4", "S4^3", "sz", "sy", "sx", "sda", "sdb"]))

# D6h, also non-Abelian and also absent from PySCF. Written as the published
# class table plus class sizes, and expanded to one column per operation.
#
# Convention: the C6 axis is z and the atoms lie at 0, 60, ... 300 degrees in
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
	"A1g": [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
	"A2g": [1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1],
	"B1g": [1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
	"B2g": [1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1],
	"E1g": [2,  1, -1, -2,  0,  0,  2,  1, -1, -2,  0,  0],
	"E2g": [2, -1, -1,  2,  0,  0,  2, -1, -1,  2,  0,  0],
	"A1u": [1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1],
	"A2u": [1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1],
	"B1u": [1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1],
	"B2u": [1, -1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1],
	"E1u": [2,  1, -1, -2,  0,  0, -2, -1,  1,  2,  0,  0],
	"E2u": [2, -1, -1,  2,  0,  0, -2,  1,  1, -2,  0,  0],
}
PointGroup.D6h = PointGroup(
		schoenflies_label="D6h",
		character_table=CharacterTable(
			_expand_class_table(_D6H_CLASS_CHARACTERS, _D6H_CLASS_SIZES),
			operation_symbols=[
				"E", "C6", "C6^5", "C3", "C3^2", "C2z",
				"C2'1", "C2'2", "C2'3", "C2''1", "C2''2", "C2''3",
				"i", "S3", "S3^5", "S6", "S6^5", "sz",
				"sd1", "sd2", "sd3", "sv1", "sv2", "sv3"]))

PointGroup.C2h = PointGroup(
		schoenflies_label="C2h",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("Ag",	[1,  1,  1,  1]),
				("Bg",	[1, -1,  1, -1]),
				("Au",	[1,  1, -1, -1]),
				("Bu",	[1, -1, -1,  1]),
			]],	operation_symbols=["E", "C2", "i", "sh"]))

PointGroup.C2v = PointGroup(
		schoenflies_label="C2v",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("A1",	[1,  1,  1,  1]),
				("A2",	[1,  1, -1, -1]),
				("B1",	[1, -1,  1, -1]),
				("B2",	[1, -1, -1,  1]),
			]],	operation_symbols=["E", "C2", "sv", "sd"]))

PointGroup.D2 = PointGroup(
		schoenflies_label="D2",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("A",	[1, 1, 1, 1]),
				("B1",	[1,  1, -1, -1]),
				("B2",	[1, -1, -1,  1]),
				("B3",	[1, -1,  1, -1]),
			]],	operation_symbols=["E", "C2", "C2\'", "C2\""]))

PointGroup.Cs = PointGroup(
		schoenflies_label="Cs",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("A'", [1, 1]),
				("A\"", [1, -1]),
			]],	operation_symbols=["E", "sh"]))

PointGroup.Ci = PointGroup(
		schoenflies_label="Ci",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("Ag", [1, 1]),
				("Au", [1, -1]),
			]],	operation_symbols=["E", "i"]))

PointGroup.C2 = PointGroup(
		schoenflies_label="C2",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("A", [1, 1]),
				("B", [1, -1]),
			]],	operation_symbols=["E", "C2"]))

PointGroup.C1 = PointGroup(
		schoenflies_label="C1",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("A", [1]),
			]],	operation_symbols=["E"]))


# =====================================================================
# Representations of a point group
# =====================================================================

T = TypeVar('T') # The type of the symmetry operation representation, e.g. a 3x3 matrix.
S = TypeVar('S') # The corresponding state that the symmetry operation acts on, e.g. a 3D vector.


@dataclass(frozen=True)
class PointGroupRepresentation(Generic[T, S]):
	
	operations: dict[str, T]
	application_function: Callable[[T, S], S] = lambda op, state: op @ state
	is_close_function: Callable[[S, S], bool] = lambda state1, state2: numpy.isclose(state1, state2).all()

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
		# numerical noise and global phases from compiled circuits.
		return numpy.array([ numpy.round(state.inner(self.apply(op, state)).real) for op in self.operations.values() ])


	@classmethod
	def get_geometric_representation(cls, pg: PointGroup) -> 'PointGroupRepresentation[NDArray[numpy.complex128], NDArray[numpy.complex128]]':
		"""Returns the geometric representation of a point group, i.e. where the symmetry operations are represented as 3x3 matrices and the states are 3D vectors."""

		operation_dict: dict[str, NDArray[numpy.float64]] = {
			"E": numpy.eye(3),
			"i": -1 * numpy.eye(3),
			**dict.fromkeys(["C2","C2z"], numpy.array([[-1,  0,  0],
													[ 0, -1,  0],
													[ 0,  0,  1]])),
			**dict.fromkeys(["C2'","C2x"], numpy.array([[ 1,  0,  0],
													[ 0, -1,  0],
													[ 0,  0, -1]])),
			**dict.fromkeys(["C2\"","C2y"], numpy.array([[-1,  0,  0],
													[ 0,  1,  0],
													[ 0,  0, -1]])),
			**dict.fromkeys(["sh","sz"], numpy.array([[ 1,  0,  0],
													[ 0,  1,  0],
													[ 0,  0, -1]])),
			**dict.fromkeys(["sv","sy"], numpy.array([[ 1,  0,  0],
													[ 0, -1,  0],
													[ 0,  0,  1]])),
			**dict.fromkeys(["sd","sx"], numpy.array([[-1,  0,  0],
													[ 0, 1,  0],
													[ 0,  0,  1]])),
			# --- operations beyond the Abelian groups, needed for D4h ---
			# Fourfold rotations about z, and the S4 improper rotations.
			"C4": numpy.array([[ 0, -1,  0],
							[ 1,  0,  0],
							[ 0,  0,  1]]),
			"C4^3": numpy.array([[ 0,  1,  0],
							[-1,  0,  0],
							[ 0,  0,  1]]),
			"S4": numpy.array([[ 0, -1,  0],
							[ 1,  0,  0],
							[ 0,  0, -1]]),
			"S4^3": numpy.array([[ 0,  1,  0],
							[-1,  0,  0],
							[ 0,  0, -1]]),
			# Twofold axes along the diagonals y = x and y = -x.
			"C2a": numpy.array([[ 0,  1,  0],
							[ 1,  0,  0],
							[ 0,  0, -1]]),
			"C2b": numpy.array([[ 0, -1,  0],
							[-1,  0,  0],
							[ 0,  0, -1]]),
			# Mirror planes containing z and the diagonals.
			"sda": numpy.array([[ 0,  1,  0],
							[ 1,  0,  0],
							[ 0,  0,  1]]),
			"sdb": numpy.array([[ 0, -1,  0],
							[-1,  0,  0],
							[ 0,  0,  1]]),
			# --- D6h: sixfold and threefold axes about z ---
			"C6":    _rot_z(numpy.pi / 3),
			"C6^5":  _rot_z(-numpy.pi / 3),
			"C3":    _rot_z(2 * numpy.pi / 3),
			"C3^2":  _rot_z(-2 * numpy.pi / 3),
			"S6":    _improper_z(numpy.pi / 3),
			"S6^5":  _improper_z(-numpy.pi / 3),
			"S3":    _improper_z(2 * numpy.pi / 3),
			"S3^5":  _improper_z(-2 * numpy.pi / 3),
			# C2' through opposite vertices (0, 60, 120 degrees) and C2''
			# through opposite edge midpoints (30, 90, 150 degrees), for a
			# hexagon whose atoms sit at 0, 60, ... 300 degrees.
			"C2'1":  _c2_in_plane(0.0),
			"C2'2":  _c2_in_plane(numpy.pi / 3),
			"C2'3":  _c2_in_plane(2 * numpy.pi / 3),
			"C2''1": _c2_in_plane(numpy.pi / 6),
			"C2''2": _c2_in_plane(numpy.pi / 2),
			"C2''3": _c2_in_plane(5 * numpy.pi / 6),
			# sigma_v contains the atoms; sigma_d bisects them.
			"sv1":   _mirror_v(0.0),
			"sv2":   _mirror_v(numpy.pi / 3),
			"sv3":   _mirror_v(2 * numpy.pi / 3),
			"sd1":   _mirror_v(numpy.pi / 6),
			"sd2":   _mirror_v(numpy.pi / 2),
			"sd3":   _mirror_v(5 * numpy.pi / 6),
		}

		operations: dict[str, NDArray[numpy.complex128]] = {}
		for op in pg.character_table.operation_symbols:
			if op not in operation_dict:
				raise ValueError(f"Operation {op} not found. The supported operations are {list(operation_dict.keys())}.")

			operations[op] = operation_dict[op]

		return PointGroupRepresentation(operations)