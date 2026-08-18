from typing import TypeVar, Generic, Callable
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy
from .point_group import PointGroup

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