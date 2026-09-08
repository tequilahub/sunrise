from typing import TypeVar, Generic, Callable
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy
from pandas import DataFrame


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
		}

		operations: dict[str, NDArray[numpy.complex128]] = {}
		for op in pg.character_table.operation_symbols:
			if op not in operation_dict:
				raise ValueError(f"Operation {op} not found. The supported operations are {list(operation_dict.keys())}.")

			operations[op] = operation_dict[op]

		return PointGroupRepresentation(operations)