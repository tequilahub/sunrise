"""Qubit-level orbital rotation gate that is compiled to Givens rotations."""

from __future__ import annotations

import tequila as tq
from numpy import allclose, array, eye, ndarray, zeros
from tequila import QCircuit, QTensor
from tequila.circuit._gates_impl import QGateImpl

from sunrise.molecules.qubit_base import Molecule as QubitMolecule


def OR(*args, **kwargs):
    """Create a ``tq.QCircuit`` containing one ``OrbitalRotation`` gate; arguments as in ``OrbitalRotation``.

    Examples
    --------
    >>> U = OR(orbitals=[0, 1, 2], matrix=rotation)  # rotation: 3x3 orthogonal matrix
    """
    gate = OrbitalRotation(*args, **kwargs)
    return tq.QCircuit.wrap_gate(gate)


# gate impl
class OrbitalRotation(QGateImpl):
    """Rotation of a set of spatial orbitals by a (unitary) matrix.

    Two rotations can be added; the result is a single rotation on the union of the orbitals.
    Use ``compile`` to obtain the circuit.

    Parameters
    ----------
    orbitals : list of int
        Spatial orbitals the rotation acts on.
    matrix : tq.QTensor or numpy.ndarray
        Square unitary matrix of size ``len(orbitals)``.
    molecule : QuantumChemistryBase, optional
        Molecule used to build the circuit. A dummy hydrogen chain is created if not given.
    """

    def __init__(
        self,
        orbitals: list[int] | None = None,
        matrix: QTensor | ndarray | list | None = None,
        molecule=None,
    ):
        if isinstance(matrix, list):
            matrix = array(list)
        if isinstance(matrix, ndarray) and not isinstance(matrix, QTensor):
            assert allclose(eye(len(matrix)), matrix.dot(matrix.T.conj()))  # normalized
            matrix = QTensor(
                shape=matrix.shape,
                objective_list=matrix.reshape(matrix.shape[0] * matrix.shape[1]),
            )
        self.orbital = orbitals
        self.coeff = matrix
        if molecule is None:
            geom = ""
            for k in range(2 * (len(self.coeff) // 2 + 1)):
                geom += f"h 0.0 0.0 {1.5 * k}\n"

            dummy = QubitMolecule(geometry=geom, basis_set="sto-3g", units="a")
            self.molecule = dummy
        else:
            self.molecule = molecule
        assert len(self.orbital) == len(self.coeff)
        assert self.coeff.ndim == 2
        assert self.coeff.shape[0] == self.coeff.shape[1]  # is square
        taget = []
        for i in self.orbital:
            taget.append(2 * i)
            taget.append(2 * i + 1)
        super().__init__(name="OrbRot", target=taget, control=None)
        # assert allclose(eye(len(self.coeff)), self.coeff.dot(self.coeff.T.conj())) #not normalization can be enforced, let to the user

    def is_parametrized(self) -> bool:
        """Return True (the gate is always considered parametrized)."""
        return True

    def extract_variables(self):
        """Return the variables contained in the rotation matrix."""
        return self.coeff.extract_variables()

    def __add__(self, other):
        """Compose with another rotation (merged on the union of orbitals) or prepend to a ``tq.QCircuit`` after compiling."""
        if isinstance(other, QCircuit):
            return self.compile() + other
        else:
            ndx = list(set(self.orbital + other.orbital))
            pos_idx = [ndx.index(i) for i in self.orbital]
            pos_jdx = [ndx.index(i) for i in other.orbital]
            new_a = self.pad_eye(self.coeff, size=len(ndx))
            new_b = self.pad_eye(other.coeff, size=len(ndx))
            ra = self._rot_mat(len(ndx), pos_idx)
            rb = self._rot_mat(len(ndx), pos_jdx)
            ap = ra.T.dot(new_a.dot(ra))
            bp = rb.T.dot(new_b.dot(rb))
            return OrbitalRotation(matrix=bp.dot(ap), orbitals=ndx)

    def __iadd__(self, other):
        """Compose in place with another rotation on the union of the orbitals."""
        ndx = list(set(self.orbital + other.orbital))
        pos_idx = [ndx.index(i) for i in self.orbital]
        pos_jdx = [ndx.index(i) for i in other.orbital]
        new_a = self.pad_eye(self.coeff, size=len(ndx))
        new_b = self.pad_eye(other.coeff, size=len(ndx))
        ra = self._rot_mat(len(ndx), pos_idx)
        rb = self._rot_mat(len(ndx), pos_jdx)
        ap = ra.T.dot(new_a.dot(ra))
        bp = rb.T.dot(new_b.dot(rb))
        self.coeff = bp.dot(ap)
        self.orbital = ndx
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def compile(self, **kwargs) -> QCircuit:
        """Compile the rotation into a circuit of Givens rotations.

        Parameters
        ----------
        tol : float, optional
            Angles below this are dropped. Default is 1e-12.
        ordering : list or str, optional
            Givens elimination order. Default is ``"Optimized"``.

        Returns
        -------
        tq.QCircuit
            Circuit acting on the qubits of ``orbitals``.
        """
        if "tol" in kwargs:
            tol = kwargs["tol"]
            kwargs.pop("tol")
        else:
            tol = 1e-12
        if "ordering" in kwargs:
            ordering = kwargs["ordering"]
            kwargs.pop("ordering")
        else:
            ordering = "Optimized"
        U = self.molecule.get_givens_circuit(
            unitary=self.coeff, tol=tol, ordering=ordering
        )
        d = {2 * i: 2 * idx for i, idx in enumerate(self.orbital)}
        d.update({2 * i + 1: 2 * idx + 1 for i, idx in enumerate(self.orbital)})
        U = U.map_qubits(qubit_map=d)
        return U

    # def __eq__(self, other):
    #     return self.orbital==other.orbital and allclose(self.coeff,other.coeff)
    def _rot_mat(self, n: int, pos_dx):
        """Return the permutation matrix that moves ``pos_dx`` to the leading positions."""
        m = QTensor(shape=(n, n), objective_list=zeros(shape=(n, n)).reshape(n * n))
        for i in range(len(pos_dx)):
            m[i, pos_dx[i]] = 1
        rest = [i for i in [*range(n)] if i not in pos_dx]
        for i in range(len(rest)):
            m[len(pos_dx) + i, rest[i]] = 1
        return m

    def __str__(self):
        result = "Orbital Rotation \n"
        result += f"acting on MO: {self.orbital}\n"
        result += f"with matrix: \n {self.coeff}"
        return result

    def __repr__(self):
        return self.__str__()

    def pad_eye(self, matrix: QTensor, size) -> QTensor:
        """Enlarge a square matrix to ``(size, size)`` filling new diagonal entries with one.

        Parameters
        ----------
        matrix : tq.QTensor
        size : int
            New dimension.

        Returns
        -------
        tq.QTensor
        """
        new = QTensor(shape=(size, size), objective_list=eye(size).reshape(size**2))
        for i in range(len(matrix)):
            for j in range(i, len(matrix)):
                new[i, j] = matrix[i, j]
                new[j, i] = matrix[j, i]
        return new
