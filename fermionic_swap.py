import copy
import numpy as np

import tequila as tq
from typing import List, Optional

def fSWAP(mol, i: int, j: int) -> tq.QCircuit:
    """
    Performs a fermionic swap between spin-orbitals i and j
    """

    return (tq.gates.I(list(range(2*mol.n_orbitals)))
            + tq.gates.CNOT(target=i, control=j)
            + tq.gates.CNOT(target=j, control=i)
            + tq.gates.CNOT(target=i, control=j)
            + tq.gates.CZ(target=i, control=j))

def reordering_gate(n_orbitals):
    """
    Reorders qubits from up-down-up-down to up-up-down-down
    """
    qubits = n_orbitals * 2
    depth = qubits // 2

    # U = tq.gates.I(list(range(qubits)))
    U = tq.QCircuit()
    for d in range(depth):
        for qubit in range(d + 1, qubits - d - 1, 2):
            U += fSWAP(qubit, qubit + 1)

    return U

def pair_fSWAP(mol, i: int, j: int) -> tq.QCircuit:
    # U = tq.gates.I(list(range(2*self.n_orbitals+1)))
    U = tq.QCircuit()
    
    start = min(i, j)
    end = max(i, j)
    
    # If they are the same orbital, return empty circuit
    if start == end:
        return U
        
    # If they are nearest neighbors, apply a single fSWAP pair (original logic)
    if end - start == 1:
        return fSWAP(self.up(start), self.up(end)) + self.swap(self.down(start), self.down(end))

    # Rightward sweep: bubble `start` to `end`
    # The last step swap the targeted orbitals
    for k in range(start, end):
        U += fSWAP(self.up(k), self.up(k+1))
        U += fSWAP(self.down(k), self.down(k+1))
        
    # Leftward sweep: bubble the displaced element back to `start`
    for k in range(end - 1, start, -1):
        U += fSWAP(self.up(k-1), self.up(k))
        U += fSWAP(self.down(k-1), self.down(k))
        
    return U

class Reorder():
    """
    Reorders qubits from up-down-up-down to up-up-down-down
    """

    def __init__(self, n_orbitals: int):
        self.n_orbitals = n_orbitals

    # https://arxiv.org/pdf/2111.04572 III. A. & C.
    def swap(self, i: int, j: int) -> tq.QCircuit:
        return (tq.gates.I(list(range(2*self.n_orbitals)))
                + tq.gates.CNOT(target=i, control=j)
                + tq.gates.CNOT(target=j, control=i)
                + tq.gates.CNOT(target=i, control=j)
                + tq.gates.CZ(target=i, control=j))

    def construct_circuit(self):
        qubits = self.n_orbitals * 2
        depth = qubits // 2

        # U = tq.gates.I(list(range(qubits)))
        U = tq.QCircuit()
        for d in range(depth):
            for qubit in range(d + 1, qubits - d - 1, 2):
                U += self.swap(qubit, qubit + 1)

        return U

    def map_variables(self, variables):
        return copy.deepcopy(self)

    def render_circuit(self) -> str:
        gates = ""
        for i in range(self.n_orbitals):
            gates += "a" + str(i) + " "
        return gates + "G:width=30 $\\uparrow\\uparrow\\downarrow\\downarrow$"

    def used_wires(self) -> List[int]:
        return list(range(self.n_orbitals))

class FSWAP():
    """
    Performs a fermionic swap between orbitals i and j
    """

    i: int
    j: int
    n_orbitals: int
    up_then_down: bool

    def __init__(self, i: int, j: int, n_orbitals: int, up_then_down: bool):
        self.i = i
        self.j = j
        self.n_orbitals = n_orbitals
        self.up_then_down = up_then_down

    def down(self, x: int) -> int:
        if self.up_then_down:
            return x + self.n_orbitals
        else:
            return x + 1

    def up(self, x: int) -> int:
        if self.up_then_down:
            return x
        else:
            return x * 2

    # https://arxiv.org/pdf/2111.04572 III. A. & C.
    def swap(self, x: int, y: int) -> tq.QCircuit:
        return (tq.gates.CNOT(target=x, control=y)
                + tq.gates.CNOT(target=y, control=x)
                + tq.gates.CNOT(target=x, control=y)
                + tq.gates.CZ(target=x, control=y))

    def construct_circuit(self) -> tq.QCircuit:
        # U = tq.gates.I(list(range(2*self.n_orbitals+1)))
        U = tq.QCircuit()
        
        start = min(self.i, self.j)
        end = max(self.i, self.j)
        
        # If they are the same orbital, return empty circuit
        if start == end:
            return U
            
        # If they are nearest neighbors, apply a single fSWAP pair (original logic)
        if end - start == 1:
            return self.swap(self.up(start), self.up(end)) + self.swap(self.down(start), self.down(end))

        # Rightward sweep: bubble `start` to `end`
        # The last step swap the targeted orbitals
        for k in range(start, end):
            U += self.swap(self.up(k), self.up(k+1))
            U += self.swap(self.down(k), self.down(k+1))
            
        # Leftward sweep: bubble the displaced element back to `start`
        for k in range(end - 1, start, -1):
            U += self.swap(self.up(k-1), self.up(k))
            U += self.swap(self.down(k-1), self.down(k))
            
        return U

    def map_variables(self, variables):
        return copy.deepcopy(self)

    def render_circuit(self) -> str:
        if self.i < self.j:
            touched = range(self.i + 1, self.j)
        else:
            touched = range(self.j + 1, self.i)

        strTouch = ""
        for i in touched:
            strTouch += "a" + str(i) + " "

        return "a{} a{} SWAP ".format(self.i, self.j) + strTouch

    def used_wires(self) -> List[int]:
        # For non-nearest neighbor, the swap touches all wires between i and j inclusive.
        start = min(self.i, self.j)
        end = max(self.i, self.j)
        return list(range(start, end + 1))

def parametrized_fSWAP(phi_var: str = 'phi') -> tq.QCircuit:
    """
    Builds a 2-qubit parameterized circuit implementing exp(-i*phi/2 * ZZ)
    followed by single-qubit phase rotations and a global phase.
    
    Args:
        phi_var: Name of the tequila Variable to use for the angle parameter.
    
    Returns:
        A tequila QCircuit object.
    """
    phi = tq.Variable(phi_var)
    ZZ = tq.paulis.Z(0) * tq.paulis.Z(1)

    return (
        # XX rotation block
        tq.gates.H(0) + tq.gates.H(1) +
        tq.gates.GeneralizedRotation(generator=ZZ, angle=phi / 2) +
        tq.gates.H(0) + tq.gates.H(1) +
        # YY rotation block
        tq.gates.Rx(angle=np.pi / 2, target=0) +
        tq.gates.Rx(angle=np.pi / 2, target=1) +
        tq.gates.GeneralizedRotation(generator=ZZ, angle=phi / 2) +
        tq.gates.Rx(angle=-np.pi / 2, target=0) +
        tq.gates.Rx(angle=-np.pi / 2, target=1) +
        # ZZ single-qubit phase block
        tq.gates.Rz(angle=phi / 2, target=0) +
        tq.gates.Rz(angle=phi / 2, target=1) +
        # Fixing phi=np.pi here returns the correct global phase for the standard fSWAP
        tq.gates.GlobalPhase(angle=phi / 2)
    )

def generator_fSWAP(mol, start, end, total_orbs=None):
    """
    Generates a fermionic SWAP gate between two orbitals using a fermionic generator.
    
    Args:
        mol (tq.Molecule): The Tequila molecule object (used to generate operators).
        start (int): Index of the first orbital.
        end (int): Index of the second orbital.
        total_orbs (list): List of all orbital indices (used to pad with Identity). 
                           If None, it infers them from the molecule.
                           
    Returns:
        tq.QCircuit: The unitary circuit for the fSWAP operation.
    """
    if total_orbs is None:
        total_orbs = list(range(2 * mol.n_orbitals))
        
    # Generate the creation and annihilation operators for the specific modes
    a_start = mol.make_annihilation_op(start)
    a_end = mol.make_annihilation_op(end)
    a_dag_start = mol.make_creation_op(start)
    a_dag_end = mol.make_creation_op(end)
    
    # Construct the exact fermionic generator
    G = a_dag_start * a_end + a_dag_end * a_start - a_dag_start * a_start - a_dag_end * a_end
    
    # Exponentiate the generator, pad with identity, and apply the missing +i global phase
    U = tq.gates.Trotterized(angle=np.pi, generator=G, steps=1) 
    U += tq.gates.I(total_orbs) 
    U += tq.gates.GlobalPhase(angle=np.pi/2)
    
    return U

def simulate_all(gate, states, label=""):
    print(f"=== {label} ===")
    for s in states:
        init   = tq.QubitWaveFunction.from_string(s)
        result = tq.simulate(gate, initial_state=init)
        print(f"  {s} -> {result}")
    print()

def get_n_electron_states(n_qubits, n_electrons):
    return [
        f"|{format(i, f'0{n_qubits}b')}>"
        for i in range(2**n_qubits)
        if format(i, f'0{n_qubits}b').count('1') == n_electrons
    ]

if __name__ == "__main__":

    # Nearest neighbours case
    # 2 qubits 1 electrons
    # states_2 = get_n_electron_states(2, 2)
    # print(states_2)
    # fswap2 = FSWAP(0, 1, 1, up_then_down=True).swap(0, 1)
    # print(fswap2)
    # simulate_all(fswap2, states_2, "nearest neighbours fswap")

    # 4 qubits 2 electrons
    states_4 = get_n_electron_states(4, 2)
    print(states_4)
    R4 = Reorder(2).construct_circuit()
    # print(R4)
    simulate_all(R4, states_4, "reorder ud->uu")
    simulate_all(R4 + R4.dagger(), states_4, "reorder and back")
    fswap4 = FSWAP(0, 1, 2, up_then_down=True).construct_circuit()
    simulate_all(R4 + fswap4 + R4.dagger(), states_4, "reorder, fswap and back (0,1)")

    # # 6 qubits 2 electrons
    # states_6 = get_n_electron_states(6, 2)
    # print(states_6)
    # R6 = Reorder(3).construct_circuit()
    # # print(R6)
    # simulate_all(R6, states_6, "reorder ud->uu")
    # simulate_all(R6 + R6.dagger(), states_6, "reorder and back")
    # fswap6 = FSWAP(0, 1, 3, up_then_down=True).construct_circuit()
    # simulate_all(R6 + fswap6 + R6.dagger(), states_6, "reorder, fswap and back (0,1)")
    # fswap6 = FSWAP(0, 2, 3, up_then_down=True).construct_circuit()
    # simulate_all(R6 + fswap6 + R6.dagger(), states_6, "reorder, fswap and back (0,2)")