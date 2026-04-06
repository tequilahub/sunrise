import numpy
from dataclasses import dataclass
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
	def array(self) -> NDArray[numpy.complex128]:
		"""A NumPy array representation of the character table."""

		return numpy.array([ irrep.characters for irrep in self.irrep_data ])

	@property
	def irreps(self) -> list[str]:
		"""A list of the irreducible representations in the character table."""

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

PointGroup.D2h: PointGroup = PointGroup(
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

PointGroup.C2h: PointGroup = PointGroup(
		schoenflies_label="C2h",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("Ag",	[1,  1,  1,  1]),
				("Bg",	[1, -1,  1, -1]),
				("Au",	[1,  1, -1, -1]),
				("Bu",	[1, -1, -1,  1]),
			]],	operation_symbols=["E", "C2", "i", "sh"]))

PointGroup.C2v: PointGroup = PointGroup(
		schoenflies_label="C2v",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("A1",	[1,  1,  1,  1]),
				("A2",	[1,  1, -1, -1]),
				("B1",	[1, -1,  1, -1]),
				("B2",	[1, -1, -1,  1]),
			]],	operation_symbols=["E", "C2", "sv", "sd"]))

PointGroup.D2: PointGroup = PointGroup(
		schoenflies_label="D2",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("A",	[1, 1, 1, 1]),
				("B1",	[1,  1, -1, -1]),
				("B2",	[1, -1, -1,  1]),
				("B3",	[1, -1,  1, -1]),
			]],	operation_symbols=["E", "C2", "C2\'", "C2\""]))

PointGroup.Cs: PointGroup = PointGroup(
		schoenflies_label="Cs",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("A'", [1, 1]),
				("A\"", [1, -1]),
			]],	operation_symbols=["E", "sh"]))

PointGroup.Ci: PointGroup = PointGroup(
		schoenflies_label="Ci",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("Ag", [1, 1]),
				("Au", [1, -1]),
			]],	operation_symbols=["E", "i"]))

PointGroup.C2: PointGroup = PointGroup(
		schoenflies_label="C2",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("A", [1, 1]),
				("B", [1, -1]),
			]],	operation_symbols=["E", "C2"]))

PointGroup.C1: PointGroup = PointGroup(
		schoenflies_label="C1",
		character_table=CharacterTable([IrreducibleRepresentation(x, numpy.array(y)) for (x, y) in [
				("A", [1]),
			]],	operation_symbols=["E"]))