from dataclasses import dataclass
from numpy.typing import NDArray
from tequila import Molecule
import tequila
import numpy
from tunits import F
from .point_group import PointGroup
from .point_group_representation import PointGroupRepresentation


class GeometricTransformationHelper:
	"""A helper class for transformations from symmetry operations in 3D space on molecules around a given center."""

	@classmethod
	def parse_xyz_string(cls, mol: Molecule) -> tuple[list[str], NDArray[numpy.float64]]:
		"""
		Parses molecule geometry into symbols and coordinates.

		Parameters
		----------
		mol : Molecule
			Tequila molecule instance.

		Returns
		-------
		tuple
			(list of chemical symbols, numpy array of coordinates)
		"""

		xyz_string = str(mol.parameters.geometry).strip()
		if not xyz_string:
			raise ValueError("mol.parameters.geometry is empty.")

		lines = xyz_string.splitlines()

		# Support full XYZ format with optional first two header lines
		if lines and lines[0].split()[0].isdigit():
			lines = lines[2:]

		symbols = []
		coordinates = []

		for line in lines:
			parts = line.split()
			if not parts:
				continue
			if len(parts) < 4:
				raise ValueError(f"Invalid geometry line: '{line}'")

			symbols.append(parts[0])
			coordinates.append([float(x) for x in parts[1:4]])

		return symbols, numpy.array(coordinates, dtype=float)


	# This function is specific to the STO-3g base
	@classmethod
	def num_active_spatial_orbitals(cls, charges: list[int]) -> list[int]:
		"""
		Returns an ordered list of active spin orbital counts, given only
		the list of atomic charges (atomic numbers) of the molecule's atoms.

		"""
		core_per_period = {1: 0, 2: 1, 3: 5, 4: 9, 5: 18}

		def get_period(charge):
			if charge <= 2:  return 1
			if charge <= 10: return 2
			if charge <= 18: return 3
			if charge <= 36: return 4
			if charge <= 54: return 5
			return 6

		def minimal_basis_ao_count(charge):
			"""
			Number of spatial AOs in the minimal (STO-3G) basis for a given element.
			Equals the number of occupied shells in the neutral ground-state atom.
			"""
			if charge <= 2:  return 1          # 1s
			if charge <= 4:  return 2          # 1s 2s        (Li, Be)
			if charge <= 10: return 5          # 1s 2s 2p     (B–Ne)
			if charge <= 12: return 6          # + 3s          (Na, Mg)
			if charge <= 18: return 9          # + 3s 3p      (Al–Ar)
			if charge <= 20: return 10         # + 4s          (K, Ca)
			if charge <= 30: return 15         # + 3d          (Sc–Zn)
			if charge <= 36: return 18         # + 4p          (Ga–Kr)
			return 18                          # extend as needed

		result = []
		for charge in charges:
			period        = get_period(charge)
			n_core        = core_per_period.get(period, 0)
			n_total       = minimal_basis_ao_count(charge)
			n_active = max(n_total - n_core, 0)
			result.append(n_active)

		return result


	@classmethod
	def homogeneous_operation(cls, M: NDArray[numpy.float64], center: NDArray[numpy.float64]=numpy.zeros(3)) -> NDArray[numpy.float64]:
		M_h = numpy.eye(4)
		M_h[:3, :3] = M
		M_h[:3, 3] = center - (M @ center)
		return M_h


	@classmethod
	def homogeneous_coordinates(cls, v: NDArray[numpy.float64], is_row_vector: bool = True) -> NDArray[numpy.float64]:
		v = numpy.asarray(v)

		# Single vector: [x, y, z] -> [x, y, z, 1]
		if v.ndim == 1:
			return numpy.append(v, 1.0)

		# Multiple vectors:
		if v.ndim == 2:
			if is_row_vector:
				# Treat rows as vectors: (N, 3) -> (N, 4)
				if v.shape[1] != 3:
					raise ValueError(f"Expected shape (N, 3) for row vectors, got {v.shape}.")
				return numpy.hstack([v, numpy.ones((v.shape[0], 1))])
			else:
				# Treat columns as vectors: (3, N) -> (4, N)
				if v.shape[0] != 3:
					raise ValueError(f"Expected shape (3, N) for column vectors, got {v.shape}.")
				return numpy.vstack([v, numpy.ones((1, v.shape[1]))])

		raise ValueError("Inumpyut must be a 1D 3-vector or a 2D array.")
	

	@classmethod
	def to_col_permutation(cls, A: NDArray[numpy.float64], B: NDArray[numpy.float64], tol: float = 1e-8) -> NDArray[numpy.int_]:
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
			matches = numpy.where(numpy.all(numpy.abs(A - col[:, None]) < tol, axis=0))[0]
			if len(matches) == 0:
				raise ValueError("No matching column found in A for column in B. Is the point group for the molecule correct?")
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
	def build_blockwise_permutation_matrix(cls, element_labels: list[int], label_perm: list[int]) -> NDArray[numpy.int_]:
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
		assert numpy.allclose(P.sum(0), 1) and numpy.allclose(P.sum(1), 1), \
			"Not a valid permutation matrix"

		perm = numpy.argmax(P, axis=1).tolist()   # which slot to pull from

		arr  = list(range(n))   # arr[i]  = original element currently at slot i
		pos  = list(range(n))   # pos[x]  = current slot of original element x
		swaps = []

		for i in range(n):
			target = perm[i]          # original element that belongs at slot i
			if arr[i] != target:
				j = pos[target]       # where that element currently sits
				swaps.append((i, j))
				old = arr[i]
				arr[i], arr[j] = arr[j], arr[i]
				pos[old]    = j
				pos[target] = i

		return swaps
	

# This is really inefficient but will work for now
@dataclass
class FermionicSWAP:
	mol: Molecule

	def f(self, i: int, j: int) -> tequila.QCircuit:
		"""
		This is a helper function for fermionic_swap.

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
		U = self.mol.make_excitation_gate(indices=(i,j),angle=numpy.pi)
		return U 
		#return tequila.gates.CNOT(target=i, control=j) + tequila.gates.CNOT(target=j, control=i) + tequila.gates.Rz(target=i, angle=pi) + tequila.gates.CNOT(target=i, control=j)


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
		for k in range(i,j):
			U += self.f(2*k+1, 2*(k+1))+self.f(2*k, 2*k+1)+self.f(2*(k+1), 2*(k+1)+1)+self.f(2*k+1, 2*(k+1))
		for k in range(i,j-1)[::-1]:
			U += self.f(2*k+1, 2*(k+1))+self.f(2*k, 2*k+1)+self.f(2*(k+1), 2*(k+1)+1)+self.f(2*k+1, 2*(k+1))
		return U


@dataclass
class QCircuitRepresentationBuilder:
	"""A class for constructing various representations of a point group dependant on a specific molecule."""

	mol: Molecule
	pg: PointGroup

	def build_atomic_permutation_representation(self) -> PointGroupRepresentation[NDArray[numpy.complex128], NDArray[numpy.complex128]]:
		"""
		Build a representation of the point group where the symmetry operations are
		represented as permutation matrices and the states are vectors where each
		component represents an atom of the molecule.

		Examples
		--------
		For linear H4 and inversion:

			[-1,  0,  0]    [0, 0, 0, 1]
			[ 0, -1,  0] -> [0, 0, 1, 0]
			[ 0,  0, -1]    [0, 1, 0, 0]
							[1, 0, 0, 0]
		"""

		geometric_representation = PointGroupRepresentation.get_geometric_representation(self.pg)
		
		# This can be presumed as an NDArray
		operations = geometric_representation.operations

		mol_atom_coordinates = GeometricTransformationHelper.parse_xyz_string(self.mol)[1]
		mol_center = numpy.mean(mol_atom_coordinates, axis=0)
		mol_hom_atom_coordinates = GeometricTransformationHelper.homogeneous_coordinates(mol_atom_coordinates, is_row_vector=True).T

		atomic_permutation_representation_operations: dict[str, NDArray[numpy.int_]] = {}
		for l, op in operations.items():
			# Apply the homogeneous operation to the homogeneous coordinates
			op_effect = GeometricTransformationHelper.homogeneous_operation(op, center=mol_center) @ mol_hom_atom_coordinates
			op_permutations = GeometricTransformationHelper.to_col_permutation(mol_hom_atom_coordinates, op_effect)
			atomic_permutation_representation_operations[l] = GeometricTransformationHelper.to_permutation_matrix(op_permutations)

		return PointGroupRepresentation(atomic_permutation_representation_operations)


	def build_ao_permutation_representation(self) -> PointGroupRepresentation[NDArray[numpy.complex128], NDArray[numpy.complex128]]:
		"""
		Build a representation of the point group where the symmetry operations are
		represented as permutation matrices and the states are vectors where each
		component represents an atomic orbital of the molecule.

		This is similar to the atomic permutation representation, but with additional
		structure within each atom's block corresponding to the orbitals on that atom.
		For example, for a molecule with two atoms ``x`` and ``y``, where ``x`` has 3 orbitals,
		``y`` has 3 orbitals, and ``z`` has 2 orbitals, the state vector would have 8 components, e.g. ordered as
		``[x1, x2, x3, y1, y2, y3, z1, z2]``. A symmetry operation that swaps atoms ``x`` and ``y``
		would then be represented by a permutation matrix that swaps the first 3
		rows/columns with the last 3 rows/columns accordingly. Only atomic orbitals of the same atom can
		be permuted with each other, so the permutation matrix will have a block-diagonal structure where
		each block corresponds to an atom and permutes only the orbitals of that atom.

		Notes
		-----
		PySCF is required as this method relies on it to determine the number of atomic orbitals on each atom.
		"""

		from pyscf import gto, scf

		atomic_permutation_representation = self.build_atomic_permutation_representation()

		# This can be presumed as an NDArray
		operations = atomic_permutation_representation.operations

		# This finds out the number of active spatial orbitals by constructing a one-atom PySCF molecule
		# and then creating a tequila molecule with it such that mol.n_orbitals is given by tequila
		atom_num_aos: list[int] = []
		for atom in GeometricTransformationHelper.parse_xyz_string(self.mol)[0]:
			# Build a one-atom pyscf molecule with automatic spin
			pyscf_mol = gto.M(atom=f'{atom} 0 0 0', basis=self.mol.parameters.basis_set, spin=None, verbose=0)
			mf = scf.UHF(pyscf_mol).run()

			# Compute integrals manually
			mo_coeff = mf.mo_coeff[0]  # alpha MOs for UHF
			h_ao = pyscf_mol.intor("int1e_kin") + pyscf_mol.intor("int1e_nuc")
			g_ao = pyscf_mol.intor("int2e", aosym="s1")
			S   = pyscf_mol.intor_symmetric("int1e_ovlp")

			# Pass them to tequila, bypassing the internal mol build
			# the attributes are the same to the constructor in tequila
			tq_mol = tequila.Molecule(
				geometry=f'{atom} 0 0 0',
				basis_set=self.mol.parameters.basis_set,
				one_body_integrals=h_ao,
				two_body_integrals=tequila.quantumchemistry.NBodyTensor(elems=g_ao, ordering="mulliken"),
				overlap_integrals=S,
				orbital_coefficients=mo_coeff,
				nuclear_repulsion=pyscf_mol.energy_nuc(),
			)

			atom_num_aos.append(tq_mol.n_orbitals)

		# Alternative: statically determine the number of active spatial orbitals for STO-3G based on num_active_spatial_orbitals
		#atom_num_aos: list[int] = GeometricTransformationHelper.num_active_spatial_orbitals([gto.charge(atom) for atom in GeometricTransformationHelper.parse_xyz_string(self.mol)[0]])

		ao_blocks: list[int] = []
		for i in range(len(atom_num_aos)):
			ao_blocks.extend([i] * atom_num_aos[i])

		ao_permutation_representation_operations: dict[str, NDArray[numpy.int_]] = {}
		for l, op in operations.items():
			atomic_permutation_op = GeometricTransformationHelper.to_col_permutation(numpy.eye(len(op)), op)
			ao_permutation_representation_operations[l] = GeometricTransformationHelper.build_blockwise_permutation_matrix(ao_blocks, atomic_permutation_op)

		return PointGroupRepresentation(ao_permutation_representation_operations)

	
	def build_qcircuit_representation(self) -> PointGroupRepresentation[tequila.QCircuit, tequila.QubitWaveFunction]:
		ao_permutation_representation = self.build_ao_permutation_representation()

		# This can be presumed as an NDArray
		operations = ao_permutation_representation.operations

		# For each operation, we need to convert the permutation matrix into a sequence of swaps,
		# and then convert those swaps into a QCircuit while maintaining fermionic antisymmetry.
		qcircuit_representation_operations: dict[str, tequila.QCircuit] = {}
		for l, op in operations.items():
			swaps = GeometricTransformationHelper.perm_matrix_to_swaps(op)

			if swaps == []:
				U: tequila.QCircuit = tequila.QCircuit()  # Identity operation, no swaps needed
				U += tequila.gates.I(target=list(range(self.mol.n_orbitals*2)))
				qcircuit_representation_operations[l] = U
				continue

			max_qubit_index = max(max(max(swap) for swap in swaps), self.mol.n_orbitals*2)

			U: tequila.QCircuit = tequila.QCircuit()
			U += tequila.gates.I(target=list(range(max_qubit_index))) # self.mol.n_orbitals*2 might not be enough for the fermionic swaps
			for i, j in swaps:
				assert i < U.n_qubits and j < U.n_qubits, f"Mismatch between the AO permutation representation and the number of qubits in the circuit: swap indices ({i}, {j}) exceed current circuit qubits {U.n_qubits}. This likely indicates an error in the AO permutation representation construction."
				# alternative method with FSWAP
				#U += FSWAP(i=i, j=j, n_orbitals=self.mol.n_electrons, up_then_down=True).construct_circuit()
				U += FermionicSWAP(self.mol).f_swap(i, j)

			qcircuit_representation_operations[l] = U
			

		def _is_close(state1, state2):
			if numpy.allclose((state1 + state2).to_array(), 0):
				return False
			return tequila.QubitWaveFunction.isclose(state1, state2)

		def _apply(op, state):
			assert op.n_qubits == state.n_qubits, f"Operation and state have different number of qubits: {op.n_qubits} vs {state.n_qubits}"
			return tequila.simulate(op, initial_state=state)

		return PointGroupRepresentation(
			application_function=_apply,
			is_close_function=_is_close,
			operations=qcircuit_representation_operations)