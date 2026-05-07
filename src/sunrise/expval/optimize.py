from sunrise import FCircuit
from tequila import QCircuit,TequilaException
from . import Braket
from tequila.quantumchemistry import optimize_orbitals as tq_opt_orbs
from tequila.quantumchemistry.orbital_optimizer import OptimizeOrbitalsResult
from ..molecules.fermionic_base.fer_base import FermionicBase
from ..molecules.hybrid_base.HybridBase import HybridBase
from .minimize import minimize
from typing import Union


def optimize_orbitals(molecule,circuit=Union[FCircuit,QCircuit],backend:str='tequila',use_hcb=False,pyscf_arguments=None,silent=False,vqe_solver_arguments:dict=None,initial_guess=None,return_mcscf=False,
    molecule_factory=None,molecule_arguments=None,restrict_to_active_space=True,*args,**kwargs)->OptimizeOrbitalsResult:
    """

    Parameters
    ----------
    molecule: The tequila molecule whose orbitals are to be optimized
    circuit: The FCircuit that defines the ansatz to the wavefunction in the VQE
             can be None, if a customized vqe_solver is passed that can construct a circuit
    backend: Fermionic Backend, will be created a tequila.chemistry.optimize_orbitals with custom vqe_solver
    pyscf_arguments: Arguments for the MCSCF structure of PySCF, if None, the defaults are {"max_cycle_macro":10, "max_cycle_micro":3} (see here https://pyscf.org/pyscf_api_docs/pyscf.mcscf.html)
    silent: silence printout
    vqe_solver_arguments: Optional arguments for a customized selected backed
                          for the default solver: vqe_solver_arguments={"optimizer_arguments":A, "restrict_to_hcb":False} where A holds the kwargs for tq.minimize
                          restrict_to_hcb keyword controls if the standard (in whatever encoding the molecule structure has) Hamiltonian is constructed or the hardcore_boson hamiltonian
    initial_guess: Initial guess for the MCSCF module of PySCF (Matrix of orbital rotation coefficients)
                   The default (None) is a unit matrix
                   predefined commands are
                        initial_guess="random"
                        initial_guess="random_loc=X_scale=Y" with X and Y being floats
                        This initialized a random guess using numpy.random.normal(loc=X, scale=Y) with X=0.0 and Y=0.1 as defaults
    return_mcscf: return the PySCF MCSCF structure after optimization
    molecule_arguments: arguments to pass to molecule_factory or default molecule constructor | only change if you know what you are doing
    args: just here for convenience
    kwargs: just here for conveniece

    Returns
    -------
        Optimized Tequila Molecule
    """
    class solver:
        def __init__(self,backend:str='tequila',circuit:FCircuit=None):
            self.backend = backend
            self.U = circuit
        def __call__(self, H, circuit, molecule, **vqe_solver_arguments):
            if 'silent' in vqe_solver_arguments:
                silent = vqe_solver_arguments['silent']
                vqe_solver_arguments.pop('silent')
            else: silent = True
            return minimize(Braket(backend=backend,molecule=molecule,circuit=self.U,**vqe_solver_arguments),silent=silent)
    
    if isinstance(molecule,HybridBase):
        use_hcb = False
        if molecule_arguments is None:
            molecule_arguments = {"select": molecule.select, "condense": molecule.condense,
                                  "two_qubit": molecule.two_qubit,
                                  "integral_tresh": molecule.integral_tresh,"parameters": molecule.parameters,
                                  "transformation": molecule.transformation,"backend":'pyscf'}
        else:
            mol_args = {"select": molecule.select, "condense": molecule.condense,
                                  "two_qubit": molecule.two_qubit,
                                  "integral_tresh": molecule.integral_tresh,"parameters": molecule.parameters,
                                  "transformation": molecule.transformation,"backend":'pyscf'}
            mol_args.update(molecule_arguments)
            molecule_arguments = mol_args
            if isinstance(circuit,FCircuit):
                circuit = circuit.to_qcircuit(molecule=molecule)
        if molecule_factory is None:
            molecule_factory = HybridBase
    elif isinstance(molecule,FermionicBase):
        if molecule_factory is None:
            molecule_factory = FermionicBase
        molecule.fermionic_backend = backend
        if molecule_arguments is None:
            molecule_arguments = {'fermionic_backend':backend,'parameters':molecule.parameters}
        else: 
            molecule_arguments['fermionic_backend']=backend
            if 'parameters' not in molecule_arguments:
                molecule_arguments['parameters'] = molecule.parameters
        if use_hcb:
            if vqe_solver_arguments is None:
                vqe_solver_arguments = {}
            vqe_solver_arguments['restrict_to_hcb'] = True
        if isinstance(circuit,QCircuit):
            raise TequilaException('No Qubit based circuit allowed with Fermionic Molecule')
        if 'vqe_solver' not in kwargs:
            kwargs['vqe_solver'] = solver(backend=backend,circuit=circuit)
    else: #Plain tequila molecule
        if isinstance(circuit,FCircuit):
            circuit.to_qcircuit(molecule=molecule)
    result = tq_opt_orbs(molecule=molecule,use_hcb=use_hcb,circuit=circuit,pyscf_arguments=pyscf_arguments,silent=silent,initial_guess=initial_guess,return_mcscf=return_mcscf,molecule_factory=molecule_factory,molecule_arguments=molecule_arguments,restrict_to_active_space=restrict_to_active_space,vqe_solver_arguments=vqe_solver_arguments,*args,**kwargs)
    if isinstance(molecule,HybridBase):
        result.molecule = HybridBase(**molecule_arguments, integral_manager=result.molecule.integral_manager)
    elif isinstance(molecule,FermionicBase):
        result.molecule = FermionicBase(parameters=result.molecule.parameters, integral_manager=result.molecule.integral_manager)
    return result