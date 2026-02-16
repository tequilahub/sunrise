import tequila as tq
from sunrise import optimize_orbitals
import numpy
import time
import warnings
from .fast_rdm import fast_rdm
from .fast_qtensor import fast_qtensor
from .decompose import decompose, make_decomposed_clusters
from tequila.quantumchemistry.orbital_optimizer import OptimizeOrbitalsResult
from tequila import TequilaWarning
from typing import Union

def run_spa(mol, edges, initial_guess=None, decompose=True, silent=True,grouping:Union[int,list[int],None]=None,backend:str='qulacs', fast_rdm=True, **kwargs)->OptimizeOrbitalsResult:

    if edges is None:
        U = mol.make_ansatz(name="HCB-SPA")
    else:
        U = mol.make_ansatz(name="HCB-SPA", edges=edges)
    if initial_guess is None:
        initial_guess = make_initial_orbital_guess(edges)
        mol = mol.use_native_orbitals()
    if isinstance(initial_guess,bool):
        if initial_guess:
            initial_guess = make_initial_orbital_guess(edges)
            mol = mol.use_native_orbitals()
        else: initial_guess = None
    grouping = make_decomposed_clusters(U, grouping)

    vqe_solver = SPASolver(decompose=decompose,grouping=grouping,backend=backend, fast_rdm=fast_rdm)
    opt = optimize_orbitals(circuit=U, molecule=mol, initial_guess=initial_guess, vqe_solver=vqe_solver, silent=silent, use_hcb=True,vqe_solver_arguments={"optimizer_arguments":{"backend":backend}}, **kwargs)
    return opt

class SPASolver:

    def __init__(self, decompose=False,grouping:Union[int,list[int],None]=None,backend:str='qulacs', fast_rdm=True, restrict_to_hcb=True):
        self.decompose=decompose
        self.rdm_qtensors = None
        self.variables = None
        self.grouping =grouping
        self.backend=backend
        self.restart_from_previous_runs=True
        self.restrict_to_hcb = restrict_to_hcb # need to change to "use_hcb" for consistency, but also needs to be changted in tequila
        self.fast_rdm=fast_rdm
    def __call__(self, *args, **kwargs):
        return self.vqe(*args, **kwargs)
    
    def vqe(self, H, circuit, molecule, *args, **kwargs):

        if "restrict_to_hcb" in kwargs:
            self.restrict_to_hcb = kwargs
        # failsave:
        if self.restrict_to_hcb:
            n = molecule.n_orbitals
            if len(circuit.qubits)>n:
                warnings.warn(f"VQE: restrict_to_hcb is True but more than {n} qubits in the circuit: {circuit.qubits}", TequilaWarning)
            if len(H.qubits)>n:
                warnings.warn(f"VQE: restrict_to_hcb is True but more than {n} qubits in the Hamiltonian: {circuit.qubits}", TequilaWarning)

        if len(circuit.qubits) != len(H.qubits):
            warnings.warn(f"VQE: Hamiltonian with {H.qubits} qubits and circuit with {circuit.qubits} qubits?!? restrict_to_hcb={self.restrict_to_hcb}", TequilaWarning)

        if self.decompose:
            E = decompose(H=H, U=circuit,grouping=self.grouping)
            result = tq.minimize(E, silent=True, initial_values=self.variables, gradient="2-point",method_options={"finite_diff_rel_step":1.e-4},backend=self.backend,*args,**kwargs)
        else:
            E = tq.ExpectationValue(H=H, U=circuit)
            result = tq.minimize(E, silent=True, initial_values=self.variables,backend=self.backend,*args,**kwargs)
        if self.restart_from_previous_runs: self.variables = result.variables
        #analyse()
        return result

    def compute_rdms(self, U, variables, molecule, use_hcb=False ):
        start = time.time()
        if self.fast_rdm:
            # needs hacked tq version where molecule.compute_rdms gives back the raw list of qops
            rdm1, rdm2 = fast_rdm(U=U, mol=molecule, clusters=self.grouping, variables=variables,backend=self.backend)
        else:
            if self.rdm_qtensors is None or str(U)!=self.rdm_qtensors[-1]:
                rdm1, rdm2 = molecule.compute_rdms(U=U, evaluate=False, use_hcb=use_hcb)
                rdm1, shape1 = fast_qtensor(rdm1, variables=variables, do_decompose=self.decompose, evaluate=False,grouping=self.grouping,backend=self.backend)
                rdm2, shape2 = fast_qtensor(rdm2, variables=variables, do_decompose=self.decompose, evaluate=False,grouping=self.grouping,backend=self.backend)
                self.rdm_qtensors = ((rdm1,shape1),(rdm2,shape2),str(U))

            else:
                rdm1, rdm2, checksum = self.rdm_qtensors
                rdm1, shape1 = rdm1
                rdm2, shape2 = rdm2

            rdm1 = [x(variables) for x in rdm1]
            rdm1 = numpy.asarray(rdm1).reshape(shape1)
            rdm2 = [x(variables) for x in rdm2]
            rdm2 = numpy.asarray(rdm2).reshape(shape2)

        #print("RDMs took {}s".format(time.time()-start))

        return rdm1, rdm2



def make_initial_orbital_guess(edges):

    result = numpy.eye(2*len(edges))
    for i,j in edges:
        result[j][i] = 1.0
        result[i][j] = -1.0
    return result.T
