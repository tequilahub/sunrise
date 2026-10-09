import tequila as tq
from sunrise.molecules.qubit_base import Molecule as QubitMolecule
from tequila import QCircuit,QTensor
from numpy import zeros,eye,allclose,ndarray,array
from typing import Union,List
from tequila.circuit._gates_impl import ParametrizedGateImpl, QGateImpl


def OR(*args, **kwargs):
    gate = OrbitalRotation(*args, **kwargs)
    return tq.QCircuit.wrap_gate(gate)

# gate impl
class OrbitalRotation(QGateImpl):

    def __init__(self,orbitals:List[int]=None,matrix:Union[QTensor,ndarray,list]=None,molecule=None):
        if isinstance(matrix,list):
            matrix = array(list)
        if isinstance(matrix,ndarray) and not isinstance(matrix,QTensor):
            assert allclose(eye(len(matrix)), matrix.dot(matrix.T.conj())) #normalized
            matrix = QTensor(shape=matrix.shape,objective_list=matrix.reshape(matrix.shape[0]*matrix.shape[1]))
        self.orbital = orbitals
        self.coeff = matrix
        if molecule is None:
            geom = ""
            for k in range(2 * (len(self.coeff) // 2 + 1)):
                geom += f"h 0.0 0.0 {1.5 * k}\n"
            dummy = QubitMolecule(geometry=geom, basis_set='sto-3g')
            self.molecule = dummy
        else: self.molecule = molecule
        assert len(self.orbital) == len(self.coeff)
        assert self.coeff.ndim == 2
        assert self.coeff.shape[0] == self.coeff.shape[1] #is square
        taget = []
        for i in self.orbital:
            taget.append(2*i)
            taget.append(2*i+1)
        super().__init__(name="OrbRot", target=taget, control=None)
        # assert allclose(eye(len(self.coeff)), self.coeff.dot(self.coeff.T.conj())) #not normalization can be enforced, let to the user

    def is_parametrized(self) -> bool:
        return True
    def extract_variables(self):
        return self.coeff.extract_variables()

    def __add__(self, other):
        if isinstance(other,QCircuit):
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

    def compile(self, **kwargs)->QCircuit:
        if "tol" in kwargs:
            tol = kwargs['tol']
            kwargs.pop('tol')
        else:
            tol = 1e-12
        if 'ordering' in kwargs:
            ordering = kwargs['ordering']
            kwargs.pop('ordering')
        else:
            ordering = 'Optimized'
        U = self.molecule.get_givens_circuit(unitary=self.coeff,tol=tol,ordering=ordering)
        d = {2*i:2*idx for i,idx in enumerate(self.orbital)}
        d.update({2*i+1:2*idx+1 for i,idx in enumerate(self.orbital)})
        U = U.map_qubits(qubit_map=d)
        return U
    # def __eq__(self, other):
    #     return self.orbital==other.orbital and allclose(self.coeff,other.coeff)
    def _rot_mat(self,n: int, pos_dx):
        m = QTensor(shape=(n,n),objective_list=zeros(shape=(n,n)).reshape(n*n))
        for i in range(len(pos_dx)):
            m[i, pos_dx[i]] = 1
        rest = [i for i in [*range(n)] if i not in pos_dx]
        for i in range(len(rest)):
            m[len(pos_dx) + i, rest[i]] = 1
        return m
    def __str__(self):
        result = 'Orbital Rotation \n'
        result += f'acting on MO: {self.orbital}\n'
        result += f'with matrix: \n {self.coeff}'
        return result
    def __repr__(self):
        return self.__str__()
    def pad_eye(self,matrix:QTensor,size)->QTensor:
        '''
        Increases the square Qtensor size to (size,size) setting new diag entries to one.
        Useful for Qtensor  tensor product.
        '''
        new = QTensor(shape=(size,size),objective_list=eye(size).reshape(size**2))
        for i in range(len(matrix)):
            for j in range(i,len(matrix)):
                new[i,j] = matrix[i,j]
                new[j,i] = matrix[j,i]
        return new