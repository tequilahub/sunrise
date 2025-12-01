import warnings
from tequila import TequilaException, BitString, TequilaWarning
from tequila.hamiltonian import QubitHamiltonian
from tequila.hamiltonian.paulis import  Zero

from tequila.circuit import QCircuit, gates
from tequila.objective.objective import Variable, Variables, ExpectationValue

from tequila.simulators.simulator_api import simulate
from tequila.utils import to_float
from tequila.quantumchemistry.chemistry_tools import prepare_product_state, \
    ParametersQC, NBodyTensor
from tequila.quantumchemistry import optimize_orbitals
from tequila.quantumchemistry.qc_base import QuantumChemistryBase as qc_base
import typing, numpy
from itertools import product
from sunrise.molecules.hybrid_base.encodings import known_encodings
from tequila.quantumchemistry.encodings import EncodingBase
from sunrise.molecules.hybrid_base.FermionicGateImpl import FermionicGateImpl
from openfermion import FermionOperator
import copy
from sunrise.hybridization.hybridization import Graph
from typing import Union
class HybridBase(qc_base):
    def __init__(self, parameters: ParametersQC,select: typing.Union[str,dict]={},transformation: typing.Union[str, typing.Callable] = None, active_orbitals: list = None,
                 frozen_orbitals: list = None, orbital_type: str = None,reference_orbitals: list = None, orbitals: list = None, *args, **kwargs):
        '''
        Parameters
        ----------
        select: codification of the transformation for each MO.
        parameters: the quantum chemistry parameters handed over as instance of the ParametersQC class (see there for content)
        transformation1: the fermion to qubit transformation (default is JordanWigner).
        transformation2: the boson to qubit transformation (default is Hard-Core Boson).
        active_orbitals: list of active orbitals (others will be frozen, if we have N-electrons then the first N//2 orbitals will be considered occpied when creating the active space)
        frozen_orbitals: convenience (will be removed from list of active orbitals)
        reference_orbitals: give list of orbitals that shall be considered occupied when creating a possible active space (default is the first N//2). The indices are expected to be total indices (including possible frozen orbitals in the counting)
        orbitals: information about the orbitals (should be in OrbitalData format, can be a dictionary)
        args
        kwargs
        '''
        self._select = {}
        self._rdm1 = None
        self._rdm2 = None
        if ("condense" in kwargs):
            self.condense = kwargs["condense"]
            kwargs.pop("condense")
        else:
            self.condense = True
        if ("two_qubit" in kwargs):
            self.two_qubit = kwargs["two_qubit"]
            kwargs.pop("two_qubit")
            if self.two_qubit: self.condense=False
        else:
            self.two_qubit = False
        if ("integral_tresh" in kwargs):
            self.integral_tresh = kwargs["integral_tresh"]
            kwargs.pop('integral_tresh')
        else:
            self.integral_tresh = 1.e-6
        self.parameters = parameters
        n_electrons = parameters.total_n_electrons
        if "n_electrons" in kwargs:
            n_electrons = kwargs["n_electrons"]

        if reference_orbitals is None:
            reference_orbitals = [i for i in range(n_electrons // 2)]
        self._reference_orbitals = reference_orbitals

        if orbital_type is None:
            orbital_type = "unknown"
        # no frozen core with native orbitals (i.e. atomics)
        overriding_freeze_instruction = orbital_type is not None and orbital_type.lower() == "native"
        # determine frozen core automatically if set
        # only if molecule is computed from scratch and not passed down from above
        overriding_freeze_instruction = overriding_freeze_instruction or n_electrons != parameters.total_n_electrons
        overriding_freeze_instruction = overriding_freeze_instruction or frozen_orbitals is not None
        if not overriding_freeze_instruction and self.parameters.frozen_core:
            n_core_electrons = self.parameters.get_number_of_core_electrons()
            if frozen_orbitals is None:
                frozen_orbitals = [i for i in range(n_core_electrons // 2)]

        # initialize integral manager
        if "integral_manager" in kwargs:
            self.integral_manager = kwargs["integral_manager"]
        else:
            self.integral_manager = self.initialize_integral_manager(active_orbitals=active_orbitals,
                                                                     reference_orbitals=reference_orbitals,
                                                                     orbitals=orbitals, frozen_orbitals=frozen_orbitals,
                                                                     orbital_type=orbital_type,
                                                                     basis_name=self.parameters.basis_set,condense=self.condense,
                                                                     two_qubit=self.two_qubit,  *args,**kwargs)

        if orbital_type is not None and orbital_type.lower() == "native":
            self.integral_manager.transform_to_native_orbitals()
        #tq init overwriten bcs I need the number of orbitals for select->transformation
        if not isinstance(transformation,typing.Callable):
            self.transformation:EncodingBase = self._initialize_transformation(transformation=transformation,*args,**kwargs) #here select set only to full-HCB. changed afterwards
        else: self.transformation:EncodingBase = transformation
        self.update_select(select,n_orb=self.n_orbitals)
        self.up_then_down = self.transformation.up_then_down

    #Select Related Functions
    def update_select(self,select:typing.Union[str,dict,list,tuple],n_orb:int=None):
        '''
        Parameters
        ----------
        select: codification of the transformation for each MO.

        Returns
        -------
        Updates the MO cofication data. Returns Instance of the class
        '''
        if n_orb is None: n_orb = self.n_orbitals
        def verify_selection_str(select:str,n_orb:int)->dict:
            """
            Internal function
            Checks if the orbital selection string has the appropiated lenght, otherwise corrects it
            :return : corrected selection dict
            """
            sel = {}
            if (len(select) >= n_orb):
                select = select[:n_orb]
            else:
                select += (n_orb-len(select))*"B"
            for i in range(len(select)):
                if select[i] in ["F","B"]:
                    sel.update({i: select[i]})
                else:
                    raise TequilaException(f"Warning, encoding character not recognised on position {i}: {select[i]}.\n Please choose between F (Fermionic) and B (Bosonic).")
            return sel
        def verify_selection_dict(select:dict,n_orb:int)->dict:
            """
            Internal function
            Checks if the orbital selection dictionary has the appropiated lenght, otherwise corrects it
            :return : corrected selection dict
            """
            sel = {}
            for i in range(n_orb):
                if i in [*select.keys()]:
                    if select[i] in ["F","B"]:
                        sel.update({i:select[i]})
                    else: raise TequilaException("Warning, encoding character not recognised on entry {it}.\n Please choose between F (Fermionic) and B (Bosonic).".format(it={i:sel[i]}))
                else:
                    sel.update({i:'B'})
            return sel
        def verify_selection_list(select:typing.Union[list,tuple],n_orb:int)->dict:
            """
            Internal function
            Checks if the orbital selection string has the appropiated lenght, otherwise corrects it
            :return : corrected selection dict
            """
            select = [*select]
            sel = {}
            if (len(select) >= n_orb):
                select = select[:n_orb]
            else:
                select= select + (n_orb-len(select))*["B"]
            for i in range(len(select)):
                if select[i] in ["F","B"]:
                    sel.update({i: select[i]})
                else:
                    TequilaException(f"Warning, encoding character not recognised on position {i}: {select[i]}.\n Please choose between F (Fermionic) and B (Bosonic).")
            return sel
        def select_to_list(select:dict):
            """
            Internal function
            Read the select string to make the proper Fer and Bos lists
            :return : list of MOs for the Bos, MOs and SOs for the Fer space
            """

            hcb = 0
            jworb = 0
            BOS_MO = []
            FER_MO = []
            FER_SO = []
            for i in select:
                if (select[i] == "B"):
                    BOS_MO.append(i)
                    hcb += 1
                elif (select[i] == "F"):
                    FER_MO.append(i)
                    FER_SO.append(2 * i)
                    FER_SO.append(2 * i + 1)
                    jworb += 1
                else:
                    print("Warning, codification not recognized: ,", i, " returning void lists")
                    return [], [],[]
            self.bos_orb = hcb
            self.fer_orb = jworb
            return BOS_MO, FER_MO, FER_SO
        if type(select) is dict:
            select = verify_selection_dict(select=select,n_orb=n_orb)
        elif type(select) is str:
            select = verify_selection_str(select=select,n_orb=n_orb)
        elif type(select) is list or type(select) is tuple:
            select = verify_selection_list(select=select,n_orb=n_orb)
        else:
            try:
                select = verify_selection_list(select=select,n_orb=n_orb)
            except:
                raise TequilaException(f"Warning, encoding format not recognised: {type(select)}.\n Please choose either a Str, Dict, List or Tuple.")
        self.BOS_MO, self.FER_MO, self.FER_SO= select_to_list(select)
        self._select = select
        self.transformation.update_select(select)

    @classmethod
    def from_tequila(cls, molecule=None, transformation=None, *args, **kwargs):
        c = molecule.integral_manager.constant_term
        h1 = molecule.integral_manager.one_body_integrals
        h2 = molecule.integral_manager.two_body_integrals
        S = molecule.integral_manager.overlap_integrals
        if "active_orbitals" not in kwargs:
            active_orbitals = [o.idx_total for o in molecule.integral_manager.active_orbitals]
        else:
            active_orbitals = kwargs["active_orbitals"]
            kwargs.pop("active_orbitals")
        if transformation is None:
            transformation = molecule.transformation
        if not hasattr(transformation,'select'):
            transformation = cls._new_transformation(self=cls,transformation=type(transformation).__name__,n_electrons=molecule.n_electrons,n_orbitals=molecule.n_orbitals)
        parameters = molecule.parameters

        return cls(
            nuclear_repulsion=c,
            one_body_integrals=h1,
            two_body_integrals=h2,
            overlap_integrals=S,
            orbital_coefficients=molecule.integral_manager.orbital_coefficients,
            active_orbitals=active_orbitals,
            transformation=transformation,
            orbital_type=molecule.integral_manager._orbital_type,
            parameters=parameters,
            reference_orbitals=molecule.integral_manager.active_space.reference_orbitals,
            *args,
            **kwargs,
        )
    
    def use_native_orbitals(self, inplace=False, core: list = None, *args, **kwargs):
        """
        Parameters
        ----------
        inplace: update current molecule or return a new instance
        core: list of the core orbitals indices (they will be frozen) on the current orbital schema. If not provided, they will be employed the
        indices on the integral manager. Indices interpreted with respect to the complete basis. If core provided but not active, will be
        chosen the ones with the lowest overlap on the Native schema.
        active(in kwargs): list of the active orbital indices on the Native Orbs schema. If not provided they will be chosen the ones
        with the lowest overlap with the core orbitals. For an example where it may necesary, check the H-He-H molecule
        with pyscf; the fist HF orbital (at high interatomic distances) correspond to |\phi_0> = 1.|\chi_1>, being identical to the native \varphi_1>.
        Returns
        -------
        New molecule in the native (orthonormalized) basis given
        e.g. for standard basis sets the orbitals are orthonormalized Gaussian Basis Functions
        """
        c = copy.deepcopy(self.integral_manager.orbital_coefficients)
        s = self.integral_manager.overlap_integrals
        d = self.integral_manager.get_orthonormalized_orbital_coefficients()

        def inner(a, b, s):
            return numpy.sum(numpy.multiply(numpy.outer(a, b), s))

        def orthogonalize(c, d, s):
            '''
            :return: orthogonalized orbitals with core HF orbitals and active Orthongonalized Native orbitals.
            '''
            ### Computing Core-Active overlap Matrix
            # sbar_{ki} = \langle \phi_k | \varphi_i \rangle = \sum_{m,n} c_{nk}d_{mi}\langle \chi_n | \chi_m \rangle
            # c_{nk} = HF coeffs, d_{mi} = nat orb coef s_{mn} = Atomic Overlap Matrix
            # k \in active orbs, i \in core orbs, m,n \in basis coeffs
            # sbar = np.einsum('nk,mi,nm->ki', c, d, s) #only works if active == to_active
            c = c.T
            d = d.T
            sbar = numpy.zeros(shape=s.shape)
            for k in active:
                for i in core:
                    sbar[i][to_active[k]] = inner(c[i], d[k], s)
            ### Projecting out Core orbitals from the Native ones
            # dbar_{ji} = d_{ji} - \sum_k sbar_{ki}c_{jk}
            # k \in active, i \in core, j in basis coeffs
            dbar = numpy.zeros(shape=s.shape)

            for j in active:
                dbar[to_active[j]] = d[j]
                for i in core:
                    temp = sbar[i][to_active[j]] * c[i]
                    dbar[to_active[j]] -= temp
            ### Projected-out Nat Orbs Normalization
            for i in to_active.values():
                norm = numpy.sqrt(numpy.sum(numpy.multiply(numpy.outer(dbar[i], dbar[i]), s.T)))
                if not numpy.isclose(norm, 0):
                    dbar[i] = dbar[i] / norm
            ### Reintroducing the New Coeffs on the HF coeff matrix
            for j in to_active.values():
                c[j] = dbar[j]
            ### Compute new orbital overlap matrix:
            sprima = numpy.eye(len(c))
            for idx, i in enumerate(to_active.values()):
                for j in [*to_active.values()][idx:]:
                    sprima[i][j] = inner(c[i], c[j], s)
                    sprima[j][i] = sprima[i][j]
            ### Symmetric orthonormalization
            lam_s, l_s = numpy.linalg.eigh(sprima)
            lam_s = lam_s * numpy.eye(len(lam_s))
            lam_sqrt_inv = numpy.sqrt(numpy.linalg.inv(lam_s))
            symm_orthog = numpy.dot(l_s, numpy.dot(lam_sqrt_inv, l_s.T))
            return symm_orthog.dot(c).T

        def get_active(core):
            ov = numpy.zeros(shape=(len(self.integral_manager.orbitals)))
            for i in core:
                for j in range(len(d)):
                    ov[j] += numpy.abs(inner(c.T[i], d.T[j], s))
            act = []
            for i in range(len(self.integral_manager.orbitals) - len(core)):
                idx = numpy.argmin(ov)
                act.append(idx)
                ov[idx] = 1 * len(core)
            act.sort()
            return act

        def get_core(active):
            ov = numpy.zeros(shape=(len(self.integral_manager.orbitals)))
            for i in active:
                for j in range(len(d)):
                    ov[j] += numpy.abs(inner(d.T[i], c.T[j], s))
            co = []
            for i in range(len(self.integral_manager.orbitals) - len(active)):
                idx = numpy.argmin(ov)
                co.append(idx)
                ov[idx] = 1 * len(active)
            co.sort()
            return co

        def active_to_active(active):
            '''
            translates active indices from canonical/the original basis to the native coeffs
            '''
            ov = numpy.zeros(shape=(len(self.integral_manager.orbitals)))
            for i in active:
                for j in range(len(d)):
                    ov[j] += numpy.abs(inner(c.T[i], d.T[j], s))
            act = []
            for i in range(len(active)):
                idx = numpy.argmax(ov)
                act.append(idx)
                ov[idx] = 0.
            act.sort()
            return act

        active = None
        if "active" in kwargs:
            active = kwargs["active"]
            kwargs.pop("active")
            if core is None:
                core = get_core(active)
        else:
            if core is None:
                if not self.integral_manager.active_space_is_trivial():
                    active = [i.idx_total for i in self.integral_manager.orbitals if i.idx is not None]
                    active = active_to_active(active)
                    core = [i.idx_total for i in self.integral_manager.orbitals if i.idx is None]
                else:
                    core = []
                    active = [i for i in range(len(self.integral_manager.orbitals))]
            else:
                if isinstance(core, int):
                    core = [core]
                active = get_active(core)
        assert len(active) + len(core) == len(self.integral_manager.orbitals)
        to_active = [i for i in range(len(self.integral_manager.orbitals)) if i not in core]
        to_active = {active[i]: to_active[i] for i in range(len(active))}
        if len(core):
            coeff = orthogonalize(c, d, s)
            if not all([i == to_active[i] for i in to_active]) and len(self.BOS_MO) and len(self.FER_MO):
                print("Orbital may be reordered, please double check F/B selection")
            if len(active) == len(self.select):
                new_select = {i: self.select[i] for i in range(len(active))}
            else:
                s = {i: self.select[i] for i in self.select.keys() if i not in core}
                new_select = {i: s[[*s.keys()][i]] for i in range(len(s))}
            if inplace:
                self.integral_manager = self.initialize_integral_manager(
                    one_body_integrals=self.integral_manager.one_body_integrals,
                    two_body_integrals=self.integral_manager.two_body_integrals,
                    constant_term=self.integral_manager.constant_term,
                    active_orbitals=[*to_active.values()],
                    reference_orbitals=[i.idx_total for i in self.integral_manager.reference_orbitals]
                    , frozen_orbitals=core, orbital_coefficients=coeff, overlap_integrals=s)
                self.integral_manager._orbital_type = 'native'
                self.update_select(new_select)
                return self
            else:
                integral_manager = self.initialize_integral_manager(
                    one_body_integrals=self.integral_manager.one_body_integrals,
                    two_body_integrals=self.integral_manager.two_body_integrals,
                    constant_term=self.integral_manager.constant_term
                    , active_orbitals=[*to_active.values()],
                    reference_orbitals=[i.idx_total for i in self.integral_manager.reference_orbitals]
                    , frozen_orbitals=core, orbital_coefficients=coeff, overlap_integrals=s)
                integral_manager._orbital_type = 'native'
                parameters = copy.deepcopy(self.parameters)
                result = HybridBase(parameters=parameters, integral_manager=integral_manager,
                                                    transformation=self.transformation,
                                                    select=new_select,
                                                    two_qubit=self.two_qubit, condense=self.condense
                                                    , integral_tresh=self.integral_tresh,
                                                    active_orbitals=[*to_active.values()])
                return result
        # can not be an instance of a specific backend (otherwise we get inconsistencies with classical methods in the backend)
        if inplace:
            self.integral_manager.transform_to_native_orbitals()
            return self
        else:
            integral_manager = copy.deepcopy(self.integral_manager)
            integral_manager.transform_to_native_orbitals()
            parameters = copy.deepcopy(self.parameters)
            result = HybridBase(parameters=parameters, integral_manager=integral_manager,
                                                transformation=self.transformation, select=self.select,
                                                two_qubit=self.two_qubit, condense=self.condense,
                                                integral_tresh=self.integral_tresh)
            return result

    @property
    def select(self):
        if self._select is not None:
            return self._select
        else: return {}

    @select.setter
    def select(self,select):
        self.update_select(select=select)
    
    @property
    def condense(self):
        if self._condense is not None:
            return self._condense
        else: return True

    @condense.setter
    def condense(self,condense):
        self._condense=condense

    @property
    def two_qubit(self):
        if self._two_qubit is not None:
            return self._two_qubit
        else: return False

    @two_qubit.setter
    def two_qubit(self,two_qubit):
        self._two_qubit=two_qubit
    
        
    # Tranformation Related Function
    def _initialize_transformation(self, transformation=None, *args, **kwargs)->EncodingBase:
        """
        Helper Function to initialize the Fermion-to-Qubit Transformation
        Parameters
        ----------
        transformation: name of the transformation (passed down from __init__
        args
        kwargs

        Returns
        -------

        """

        if transformation is None:
            transformation = "JordanWigner"

        # filter out arguments to the transformation
        trafo_args = {k.split("__")[1]: v for k, v in kwargs.items() if
                      (hasattr(k, "lower") and "transformation__" in k.lower())}

        trafo_args["n_electrons"] = self.n_electrons
        trafo_args["n_orbitals"] = self.n_orbitals
        trafo_args["select"]= self.select
        trafo_args["condense"]=self.condense
        trafo_args["two_qubit"] =self.two_qubit
        if hasattr(transformation, "upper"):
            # format to conventions
            transformation = transformation.replace("_", "").replace("-", "").upper()
            encodings = known_encodings()
            if transformation in encodings:
                transformation = encodings[transformation](**trafo_args)
            else:
                raise TequilaException(
                    "Unkown Fermion-to-Qubit encoding {}. Try something like: {}".format(transformation,
                                                                                         list(encodings.keys())))
        return transformation
    
    def _new_transformation(self, transformation=None,n_electrons:int|None=None,
                            n_orbitals:Union[int,None]=None,select:typing.Union[str,dict,list,tuple]={},
                            condense:bool=True,two_qubit:bool=False, *args, **kwargs)->EncodingBase:
        """
        Helper Function to initialize the Fermion-to-Qubit Transformation
        But providing all required parameters as input
        Parameters
        ----------
        transformation: name of the transformation (passed down from __init__
        args
        kwargs

        Returns
        -------

        """

        if transformation is None:
            transformation = "JordanWigner"

        # filter out arguments to the transformation
        trafo_args = {k.split("__")[1]: v for k, v in kwargs.items() if
                      (hasattr(k, "lower") and "transformation__" in k.lower())}

        trafo_args["n_electrons"] = n_electrons
        trafo_args["n_orbitals"] = n_orbitals
        trafo_args["select"]= select
        trafo_args["condense"]=condense
        trafo_args["two_qubit"] =two_qubit
        if hasattr(transformation, "upper"):
            # format to conventions
            transformation = transformation.replace("_", "").replace("-", "").upper()
            encodings = known_encodings()
            if transformation in encodings:
                transformation = encodings[transformation](**trafo_args)
            else:
                raise TequilaException(
                    "Unkown Fermion-to-Qubit encoding {}. Try something like: {}".format(transformation,
                                                                                         list(encodings.keys())))
        return transformation
    
    # Hamiltonian Related Funcions
    def make_hamiltonian(self)->QubitHamiltonian:
        '''
        Returns
        -------
        Qubit Hamiltonian in the Fermion/Bosont-to-Qubit transformations defined in self.parameters
        '''
        def make_fermionic_hamiltonian() -> QubitHamiltonian:
            '''
            Internal function
            Returns
            -------
            Fermionic part of the total Hamiltonian
            '''
            H_FER = []
            for i in self.FER_SO:
                for j in self.FER_SO:
                    if ((i % 2) == (j % 2)):
                        H_FER.append((((j,1),(i,0)),self.h1[j // 2][i // 2]))
                    for k in self.FER_SO:
                        for l in self.FER_SO:
                            if ((i % 2) == (l % 2) and (k % 2) == (j % 2)):
                                H_FER.append((((l,1),(k,1),(j,0),(i,0)),0.5 * self._h2.elems[l // 2][k // 2][j // 2][i // 2]))
            return self.transformation(H_FER)
        def make_bosonic_hamiltonian() -> QubitHamiltonian:
            '''
            Internal function
            Returns
            -------
            Bosonic part of the total Hamiltonian
            '''
            H_BOS = []
            for i in self.BOS_MO:
                for j in self.BOS_MO:
                    e1 = self._h2.elems[j][j][i][i]
                    if (i == j):
                        e1 += 2 * self.h1[i][i]
                    if not self.two_qubit:
                        H_BOS.append((((2 * j, 1), (2 * i, 0)), e1))
                    else: H_BOS.append((((2 * j, 1),(2 * j+1, 1),(2 * i+1, 0),(2 * i, 0)), e1))
                    if (i != j):
                        e2 = 2 * self._h2.elems[j][i][i][j] - self._h2.elems[j][i][j][i]
                        if not self.two_qubit:
                            H_BOS.append((((2 * i, 1), (2 * i, 0), (2 * j, 1), (2 * j, 0)), e2))
                        else:
                            H_BOS.append((((2*i,1),(2*i+1,1),(2*i,0),(2*i+1,0),(2*j,1),(2*j+1,1),(2*j,0),(2*j+1,0)),e2))
            return self.transformation(H_BOS)
        def make_interaction_hamiltonian() -> QubitHamiltonian:
            '''
            Returns
            -------
            Fermionic-Bosonic Interaction part of the total Hamiltonian
            '''
            H_INT = []
            h2 = self._h2.elems
            for j in self.BOS_MO:
                for k in self.FER_SO:
                    for l in self.FER_SO:
                        if (k % 2) != (l % 2):  # hhjj
                            if not self.two_qubit:
                                H_INT.append((((2*j,1),(l,0),(k,0)),((l % 2) - (k % 2)) * 0.5 * h2[j][j][l // 2][k // 2]))
                            else:
                                H_INT.append((((2*j+k%2,1),(2*j+l%2,1),(l,0),(k,0)),0.5 * h2[j][j][l // 2][k // 2]))
                        if (k % 2) != (l % 2):  # jjhh
                            if not self.two_qubit:
                                H_INT.append((((l,1),(k,1),(2*j,0)),((k % 2) - (l % 2)) * 0.5 * h2[l // 2][k // 2][j][j]))
                            else:
                                H_INT.append((((l,1),(k,1),(2*j+k%2,0),(2*j+l%2,0)),0.5 * h2[l // 2][k // 2][j][j]))
                        if ((k % 2) == (l % 2)):
                            if not self.two_qubit:
                                e2 = 2 * h2[k // 2][j][j][l // 2] + 2 * h2[j][k // 2][l // 2][j] - h2[k // 2][j][l // 2][j] - h2[j][k // 2][j][l // 2]
                                H_INT.append((((2*j,1),(2*j,0),(k,1),(l,0)),0.5 * e2))
                            else:
                                e2 = h2[j][k // 2][l // 2][j]+h2[k // 2][j][j][l // 2]
                                H_INT.append((((j*2,1),(j*2,0),(k,1),(l,0)),0.5*e2))
                                H_INT.append((((j*2+1,1),(j*2+1,0),(k,1),(l,0)),0.5*e2))
                                H_INT.append((((k,1),(l,0),(j*2+k%2,1),(j*2+k%2,0)),-0.5*(h2[k // 2][j][l // 2][j]+h2[j][k // 2][j][l // 2])))

            return self.transformation(H_INT)
        self.C, self.h1, self._h2 = self.get_integrals(ordering="openfermion")
        H = make_fermionic_hamiltonian() + make_bosonic_hamiltonian() + make_interaction_hamiltonian() + self.C
        return H.simplify(self.integral_tresh)

    def make_hardcore_boson_hamiltonian(self):
        '''
        Just for consistency.
        '''
        if len(self.FER_MO):
            print("Warning HCB Hamiltonian called but the encoding is not full HCB, with encoding: ",self.encoding)
        return self.make_hamiltonian()

    def compute_rdms(self, U: QCircuit = None, variables: Variables = None, spin_free: bool = True,
                     get_rdm1: bool = True, get_rdm2: bool = True, ordering="dirac", use_hcb: bool = False,
                     rdm_trafo: QubitHamiltonian = None, evaluate=True):
        """
        Computes the one- and two-particle reduced density matrices (rdm1 and rdm2) given
        a unitary U. This method uses the standard ordering in physics as denoted below.
        Note, that the representation of the density matrices depends on the qubit transformation
        used. The Jordan-Wigner encoding corresponds to 'classical' second quantized density
        matrices in the occupation picture.

        We only consider real orbitals and thus real-valued RDMs.
        The matrices are set as private members _rdm1, _rdm2 and can be accessed via the properties rdm1, rdm2.

        .. math :
            \\text{rdm1: } \\gamma^p_q = \\langle \\psi | a^p a_q | \\psi \\rangle
                                     = \\langle U 0 | a^p a_q | U 0 \\rangle
            \\text{rdm2: } \\gamma^{pq}_{rs} = \\langle \\psi | a^p a^q a_s a_r | \\psi \\rangle
                                             = \\langle U 0 | a^p a^q a_s a_r | U 0 \\rangle

        Parameters
        ----------
        U :
            Quantum Circuit to achieve the desired state \\psi = U |0\\rangle, non-optional
        variables :
            If U is parametrized, then need to hand over a set of fixed variables
        spin_free :
            Set whether matrices should be spin-free (summation over spin) or defined by spin-orbitals
        get_rdm1, get_rdm2 :
            Set whether either one or both rdm1, rdm2 should be computed. If both are needed at some point,
            it is recommended to compute them at once.
        rdm_trafo :
            The rdm operators can be transformed, e.g., a^dagger_i a_j -> U^dagger a^dagger_i a_j U,
            where U represents the transformation. The default is set to None, implying that U equas the identity.
        evaluate :
            if true, the tequila expectation values are evaluated directly via the tq.simulate command.
            the protocol is optimized to avoid repetation of wavefunction simulation
            if false, the rdms are returned as tq.QTensors
        Returns
        -------
        """
        # Check whether unitary circuit is not 0
        if U is None:
            raise TequilaException('Need to specify a Quantum Circuit.')
        # Check whether transformation is BKSF.
        # Issue here: when a single operator acts only on a subset of qubits, BKSF might not yield the correct
        # transformation, because it computes the number of qubits incorrectly in this case.
        # A hotfix such as for symmetry_conserving_bravyi_kitaev would require deeper changes, thus omitted for now
        if type(self.transformation).__name__ == "BravyiKitaevFast":
            raise TequilaException(
                "The Bravyi-Kitaev-Superfast transformation does not support general FermionOperators yet.")
        def _build_1bdy_operators_mix() -> list:
            """ Returns BOS one-body operators as a symmetry-reduced list of QubitHamiltonians """
            # Exploit symmetry pq = qp (not changed by spin-summation)
            ops = []
            for p in range(self.n_orbitals):
                for q in range(p + 1):
                    if (self.select[p] == "F" and self.select[q] == "F"):
                        op_tuple = [(((2 * p, 1), (2 * q, 0)),0.5)]
                        op = self.transformation(op_tuple)
                        op+= op.dagger()
                        op_tuple = [(((2 * p + 1, 1), (2 * q + 1, 0)),0.5)]
                        opa = self.transformation(op_tuple)
                        op += opa + opa.dagger()
                    elif (p == q and self.select[p] == "B"):
                        if not self.two_qubit:
                            op_tuple = [(((2 * p, 1), (2 * q, 0)),1)]
                            op = self.transformation(op_tuple)
                            op += op.dagger()
                        else:
                            op_tuple = [(((2 * p, 1), (2 * p, 0)), 0.5)]
                            op = self.transformation(op_tuple)
                            op += op.dagger()
                            op_tuple = [(((2 * p + 1, 1), (2 * p + 1, 0)), 0.5)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                    else:
                        op = Zero()
                    ops += [op]
            return ops

        def __case_2bdy(i: int, j: int, k: int, l: int) -> int:
            ''' Returns 1 if allowed term all in J, 2 if all H, 3 mixed, 0 else'''
            list = [self.select[i], self.select[j], self.select[k], self.select[l]]
            b = list.count("B")
            f = list.count("F")
            if (f == 4):
                return 1
            elif (b == 4):
                return 2
            elif (f == b):
                return 3
            else:
                return 0

        def _build_2bdy_operators_mix() -> list:
            """ Returns BOS two-body operators as a symmetry-reduced list of QubitHamiltonians """
            # Exploit symmetries pqrs = qpsr (due to spin summation, '-pqsr = -qprs' drops out)
            #                and      = rspq
            ops = []
            for p, q, r, s in product(range(self.n_orbitals), repeat=4):
                if p * self.n_orbitals + q >= r * self.n_orbitals + s and (p >= q or r >= s):
                    case = __case_2bdy(p, q, r, s)
                    if (case == 1):  # case JJJJ
                        # Spin aaaa
                        op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * s, 0), (2 * r, 0)),0.5)] if (p != q and r != s) else [((),0)]
                        op = self.transformation(op_tuple)
                        op += op.dagger()
                        # Spin abba
                        op_tuple = [(((2 * p, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r, 0)),0.5)] if (2 * p != 2 * q + 1 and 2 * s + 1 != 2 * r) else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa + opa.dagger()
                        # Spin baab
                        op_tuple = [(((2 * p + 1, 1), (2 * q, 1), (2 * s, 0), (2 * r + 1, 0)),0.5)] if (2 * p + 1 != 2 * q and 2 * s != 2 * r + 1) else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa + opa.dagger()
                        # Spin bbbb
                        op_tuple = [(((2 * p + 1, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r + 1, 0)),0.5)] if (p != q and r != s) else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa + opa.dagger()
                        ops += [op]
                    elif (case == 2):  # case HHHH
                        if not self.two_qubit:
                            # Spin aaaa+bbbb dont allow p=q=r=s bcs self-interaction orb ijji
                            op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * q, 0), (2 * p, 0)),-1)] if (p != q and p == s and q == r) else [((),0)]
                            opa = self.transformation(op_tuple)
                            op = opa + opa.dagger()
                            # Spin abba+ baab allow p=q=r=s orb iijj
                            op_tuple = [(((2 * p, 1), (2 * s, 0)),1)] if (p == q and s == r) else [((),0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # Spin abba+ baab dont allow p=q=r=s orb ijij
                            op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * p, 0), (2 * q, 0)),2)] if (p != q and p == r and s == q) else [((),0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                        else:
                            # Spin aaaa+ bbbb dont allow p=q=r=s  orb ijji
                            op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * q, 0), (2 * p, 0)), -0.5)] if (p != q and r != s and p == s and q == r) else [((),0)]
                            opa = self.transformation(op_tuple)
                            op = opa + opa.dagger()
                            op_tuple = [(((2 * p + 1, 1), (2 * q + 1, 1), (2 * q + 1, 0), (2 * p + 1, 0)), -0.5)] if (p != q and r != s and p == s and q == r) else [((),0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # Spin abba+ baab allow p=q=r=s orb iijj
                            sign = numpy.sign(p-s)
                            if not sign: sign = 1
                            op_tuple = [(((2 * p, 1),(2 * p + 1, 1),(2 * s + 1, 0), (2 * s, 0)), sign*1)] if (p == q and s == r) else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # Spin aaaa+ bbbb dont allow p=q=r=s orb ijij
                            sign = numpy.sign(q - p)
                            if not sign: sign = 1
                            op_tuple = [(((2 * p + 1, 1), (2 * q, 1), (2 * p, 0), (2 * q + 1 , 0)), 2*sign)] if (p != q and p == r and s == q) else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = [(((2 * p + 1, 1), (2 * q + 1, 1), (2 * p + 1 , 0), (2 * q + 1, 0)), 2*sign)] if (p != q and p == r and s == q) else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                        ops += [op]
                    elif (case == 3):  # case HJJH+JHHJ+HJHJ+JHJH+HHJJ+JJHH
                        if not self.two_qubit:
                            # uddu+duud hhjj
                            op_tuple = [(((2 * p, 1), (2 * r + 1, 0), (2 * s, 0)),0.5)] if (p == q and self.select[p] == "B") else [((),0)]
                            opa = self.transformation(op_tuple)
                            op = opa + opa.dagger()
                            op_tuple = [(((2 * p, 1), (2 * r, 0), (2 * s + 1, 0)),-0.5)] if (p == q and self.select[p] == "B") else [((),0)] #-0.5 bcs sign(2r,2s+1)
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # uddu+duud jjhh
                            op_tuple = [(((2 * p, 1), (2 * q + 1, 1), (2 * r, 0)),0.5)] if r == s and self.select[r] == "B" else [((),0)]
                            opa = self.transformation(op_tuple)
                            op += opa +  opa.dagger()
                            op_tuple = [(((2 * p + 1, 1), (2 * q, 1), (2 * r, 0)),-0.5)] if r == s and self.select[r] == "B" else [((),0)] #-0.5 bcs sign(2r,2s+1)
                            opa = self.transformation(op_tuple)
                            op += opa +  opa.dagger()
                            # uddu+duud+uuuu+dddd hjjh
                            op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * p, 0)),-0.5)] if (p == s and self.select[p] == "B") else [((),0)]
                            opa = self.transformation(op_tuple)
                            op += opa +  opa.dagger()
                            op_tuple = [(((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * p, 0)),-0.5)] if (p == s and self.select[p] == "B") else [((),0)]
                            opa = self.transformation(op_tuple)
                            op += opa +  opa.dagger()
                            # uddu+duud+uuuu+dddd jhhj
                            op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * q, 0), (2 * s, 0)),-0.5)] if (r == q and self.select[r] == "B") else [((),0)]
                            opa = self.transformation(op_tuple)
                            op += opa +  opa.dagger()
                            op_tuple = [(((2 * p + 1, 1), (2 * q, 1), (2 * q, 0), (2 * s + 1, 0)),-0.5)] if (r == q and self.select[r] == "B") else [((),0)]
                            opa = self.transformation(op_tuple)
                            op += opa +  opa.dagger()
                            # dddd+uuuu hjhj
                            op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * p, 0), (2 * s, 0)),1)] if p == r and self.select[p] == "B" else [((),0)]
                            opa = self.transformation(op_tuple)
                            op += opa +  opa.dagger()
                            op_tuple = [(((2 * p, 1), (2 * q + 1, 1), (2 * p, 0), (2 * s + 1, 0)),1)] if p == r and self.select[p] == "B" else [((),0)]
                            opa = self.transformation(op_tuple)
                            op += opa +  opa.dagger()
                            # dddd+uuuu jhjh
                            op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * q, 0)),1)] if (s == q and self.select[s] == "B") else [((),0)]
                            opa = self.transformation(op_tuple)
                            op += opa +  opa.dagger()
                            op_tuple = [(((2 * p + 1, 1), (2 * q, 1), (2 * r + 1, 0), (2 * q, 0)),1)] if (s == q and self.select[s] == "B") else [((),0)]
                            opa = self.transformation(op_tuple)
                            op += opa +  opa.dagger()
                        else:
                            # uddu+duud hhjj
                            op_tuple = [(((2 * p, 1),(2 * p + 1, 1), (2 * r + 1, 0), (2 * s, 0)), 0.5)] if (p == q and self.select[p] == "B") else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op = opa + opa.dagger()
                            op_tuple = [(((2 * p + 1, 1),(2 * p, 1), (2 * r, 0), (2 * s + 1, 0)), 0.5)] if (p == q and self.select[p] == "B") else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # uddu+duud jjhh
                            op_tuple = [(((2 * p, 1), (2 * q + 1, 1),(2 * r + 1, 0) ,(2 * r, 0)), 0.5)] if r == s and self.select[r] == "B" else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = [(((2 * p + 1, 1), (2 * q, 1), (2 * r, 0), (2 * r + 1, 0)), 0.5)] if r == s and self.select[r] == "B" else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # uddu+duud+uuuu+dddd hjjh
                            op_tuple = [(((2 * p + 1, 1), (2 * q, 1), (2 * r, 0), (2 * p + 1, 0)), -.5)] if (p == s and self.select[p] == "B") else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = [(((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * p, 0)), -.5)] if (p == s and self.select[p] == "B") else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # uddu+duud+uuuu+dddd jhhj
                            op_tuple = [(((2 * p, 1), (2 * q + 1, 1), (2 * q + 1, 0), (2 * s, 0)), -0.5)] if (r == q and self.select[r] == "B") else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = [(((2 * p + 1, 1), (2 * q, 1), (2 * q, 0), (2 * s + 1, 0)), -0.5)] if (r == q and self.select[r] == "B") else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # dddd+uuuu hjhj
                            op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * p, 0), (2 * s, 0)), -1)] if p == r and self.select[p] == "B" else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = [(((2 * p + 1, 1), (2 * q + 1, 1), (2 * p + 1, 0), (2 * s + 1, 0)), -1)] if p == r and self.select[p] == "B" else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # dddd+uuuu jhjh
                            op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * q, 0)), -1)] if (s == q and self.select[s] == "B") else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = [(((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * q + 1, 0)), -1)] if (s == q and self.select[s] == "B") else [((), 0)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                        ops += [op]
                    else:
                        ops += [Zero()]
            return ops

        def _assemble_rdm1_mix(evals) -> numpy.ndarray:
            """
            Returns BOS one-particle RDM built by symmetry conditions
            """
            N = self.n_orbitals
            rdm1 = numpy.zeros([N, N])
            ctr: int = 0
            for p in range(N):
                for q in range(p + 1):
                    rdm1[p, q] = evals[ctr]
                    # Symmetry pq = qp
                    rdm1[q, p] = rdm1[p, q]
                    ctr += 1
            return rdm1

        def _assemble_rdm2_mix(evals) -> numpy.ndarray:
            """ Returns spin-free two-particle RDM built by symmetry conditions """
            ctr: int = 0
            rdm2 = numpy.zeros([self.n_orbitals, self.n_orbitals, self.n_orbitals, self.n_orbitals])
            for p, q, r, s in product(range(self.n_orbitals), repeat=4):
                if p * self.n_orbitals + q >= r * self.n_orbitals + s and (p >= q or r >= s):
                    rdm2[p, q, r, s] = evals[ctr]
                    # Symmetry pqrs = rspq
                    rdm2[r, s, p, q] = rdm2[p, q, r, s]
                    ctr += 1
            # Further permutational symmetry: pqrs = qpsr
            for p, q, r, s in product(range(self.n_orbitals), repeat=4):
                if p >= q or r >= s:
                    rdm2[q, p, s, r] = rdm2[p, q, r, s]
            return rdm2

        # Build operator lists
        qops = []
        qops += _build_1bdy_operators_mix() if get_rdm1 else []
        qops += _build_2bdy_operators_mix() if get_rdm2 else []

        # Compute expected values
        rdm1 = None
        rdm2 = None
        from tequila import QTensor
        if evaluate:
            if rdm_trafo is None:
                evals = simulate(ExpectationValue(H=qops, U=U, shape=[len(qops)]), variables=variables)
            else:
                qops = [rdm_trafo.dagger() * qops[i] * rdm_trafo for i in range(len(qops))]
                evals = simulate(ExpectationValue(H=qops, U=U, shape=[len(qops)]), variables=variables)
        else:
            if rdm_trafo is None:
                evals = [ExpectationValue(H=x, U=U) for x in qops]
                N = self.n_orbitals # if spin_free else 2*self.n_orbitals
                rdm1 = QTensor(shape=[N, N])
                rdm2 = QTensor(shape=[N, N, N, N])
            else:
                raise TequilaException("compute_rdms: rdm_trafo was set but evaluate flag is False (not supported)")
            
        # Split expectation values in 1- and 2-particle expectation values
        if get_rdm1:
            len_1 = self.n_orbitals * (self.n_orbitals + 1) // 2
        else:
            len_1 = 0
        evals_1, evals_2 = evals[:len_1], evals[len_1:]
        rdm1 = []
        rdm2 = []
        # Build matrices using the expectation values
        rdm1 = _assemble_rdm1_mix(evals_1) if get_rdm1 else rdm1
        rdm2 = _assemble_rdm2_mix(evals_2) if get_rdm2 else rdm2
        if get_rdm2:
            rdm2_ = NBodyTensor(elems=rdm2, ordering="dirac")
            rdm2_.reorder(to=ordering)
            rdm2 = rdm2_.elems
        if get_rdm1:
            if get_rdm2:
                self._rdm2 = rdm2
                self._rdm1 = rdm1
                return rdm1, rdm2
            else:
                self._rdm1 = rdm1
                return rdm1
        elif get_rdm2:
            self._rdm2 = rdm2
            return rdm2
        else:
            warnings.warn("compute_rdms called with instruction to not compute?", TequilaWarning)
    
    def optimize_orbitals(self,molecule, circuit:QCircuit=None, vqe_solver=None, pyscf_arguments=None, silent=False, vqe_solver_arguments=None, initial_guess=None, return_mcscf=False, use_hcb = False, molecule_factory=None,molecule_arguments=None ,restrict_to_active_space = True,*args, **kwargs):
        """
        Interface with tq.quantumchemistry.optimize_orbitals
        Parameters
        ----------
        molecule: The tequila molecule whose orbitals are to be optimized
        circuit: The circuit that defines the ansatz to the wavefunction in the VQE
                 can be None, if a customized vqe_solver is passed that can construct a circuit
        vqe_solver: The VQE solver (the default - vqe_solver=None - will take the given circuit and construct an expectationvalue out of molecule.make_hamiltonian and the given circuit)
                    A customized object can be passed that needs to be callable with the following signature: vqe_solver(H=H, circuit=self.circuit, molecule=molecule, **self.vqe_solver_arguments)
        pyscf_arguments: Arguments for the MCSCF structure of PySCF, if None, the defaults are {"max_cycle_macro":10, "max_cycle_micro":3} (see here https://pyscf.org/pyscf_api_docs/pyscf.mcscf.html)
        silent: silence printout
        vqe_solver_arguments: Optional arguments for a customized vqe_solver or the default solver
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
        molecule_factory: callable function creates the molecule class
        args: just here for convenience
        kwargs: just here for conveniece

        Returns
        -------
            Optimized Tequila Hybrid Molecule
        """
        hybrid = hasattr(molecule,'select')
        if molecule_arguments is None:
            if hybrid:
                molecule_arguments = {"select": molecule.select, "condense": molecule.condense,
                                  "two_qubit": molecule.two_qubit,
                                  "integral_tresh": molecule.integral_tresh,"parameters": molecule.parameters,
                                  "transformation": molecule.transformation,"backend":'pyscf'}
            else: molecule_arguments = {"parameters": molecule.parameters,"transformation": molecule.transformation,"backend":'pyscf'}
        else:
            if hybrid:
                mol_args = {"select": molecule.select, "condense": molecule.condense,
                                  "two_qubit": molecule.two_qubit,
                                  "integral_tresh": molecule.integral_tresh,"parameters": molecule.parameters,
                                  "transformation": molecule.transformation,"backend":'pyscf'}
                mol_args.update(molecule_arguments)
                molecule_arguments = mol_args
        if molecule_factory is None:
            result = optimize_orbitals(molecule=molecule, circuit=circuit, vqe_solver=vqe_solver,
                                       pyscf_arguments=pyscf_arguments, silent=silent,
                                       vqe_solver_arguments=vqe_solver_arguments,
                                       initial_guess=initial_guess, return_mcscf=return_mcscf,
                                       use_hcb=use_hcb, molecule_factory=HybridBase,
                                       molecule_arguments=molecule_arguments, *args, **kwargs)
        else:
            result = optimize_orbitals(molecule=molecule, circuit=circuit, vqe_solver=vqe_solver,
                                                           pyscf_arguments=pyscf_arguments, silent=silent,
                                                           vqe_solver_arguments=vqe_solver_arguments,
                                                           initial_guess=initial_guess, return_mcscf=return_mcscf,
                                                           use_hcb=use_hcb, molecule_factory=molecule_factory,
                                                           molecule_arguments=molecule_arguments, *args, **kwargs)
        result.molecule = HybridBase(**molecule_arguments, integral_manager=result.molecule.integral_manager)
        return result
    
    def transform_orbitals(self, orbital_coefficients, ignore_active_space=False, name=None, *args, **kwargs):
        """
        Parameters
        ----------
        orbital_coefficients: second index is new orbital indes, first is old orbital index (summed over), indices are assumed to be defined on the active space
        ignore_active_space: if true orbital_coefficients are not assumed to be given in the active space
        name: str, name the new orbitals
        args
        kwargs

        Returns
        -------
        New molecule with transformed orbitals
        """

        U = numpy.eye(self.integral_manager.orbital_coefficients.shape[0])
        # mo_coeff by default only acts on the active space
        active_indices = [o.idx_total for o in self.integral_manager.active_orbitals]

        if ignore_active_space:
            U = orbital_coefficients
        else:
            for kk,k in enumerate(active_indices):
                for ll,l in enumerate(active_indices):
                    U[k][l] = orbital_coefficients[kk][ll]

        # can not be an instance of a specific backend (otherwise we get inconsistencies with classical methods in the backend)
        integral_manager = copy.deepcopy(self.integral_manager)
        integral_manager.transform_orbitals(U=U, name=name)
        result = HybridBase(parameters=self.parameters, integral_manager=integral_manager, transformation=self.transformation, select=self.select, two_qubit=self.two_qubit, condense=self.condense, integral_tresh=self.integral_tresh)
        return result
    
    # Cicuit Related Functions
    def verify_excitation(self, indices: typing.Iterable[typing.Tuple[int, int]], warning:bool=True)->bool:
        """
        Checks if the Bosonic restriction are accomplished by the excitation
        Parameters
        ----------
        :param indices: turple of pair turples (a_i,b_j) where the electron is excited from a_i to b_j
        :warning bool: change the action in case of forbiden excitations, from raising and Exception to return False
        :return : True if the excitations are allowed, raises an exception otherwise
        Returns
        -------
            Optimized Tequila Hybrid Molecule
        """
        sel = self.select
        if not isinstance(indices[0], typing.Iterable):
            converted = [(indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)]
            indices = converted
        froml = []
        tol = []
        for op in indices:
            if sel[op[0] // 2] == "B":#we dont care of any Fermionic possibility
                froml.append(op[0])
            if sel[op[1] // 2] == 'B':
                tol.append(op[1])
        for t in froml:
            if t in tol:
                froml.remove(t)
                tol.remove(t)
            elif 2*(t//2)+(not t%2) in froml:
                froml.remove(2*(t//2))
                froml.remove(2*(t//2)+1)
        for t in tol:
            if 2*(t//2)+(not t%2) in tol:
                tol.remove(2*(t//2))
                tol.remove(2*(t//2)+1)
        if (len(froml) or len(tol)):
            if (warning):
                raise TequilaException("Excitations not allowed for BOSONIC restrictions")
            else:
                return False
        return True

    def UR(self, i, j, angle=None, label=None, control=None, assume_real=True, *args, **kwargs):
        """
        Convenience function for orbital rotation circuit (rotating spatial orbital i and j) with standard naming of variables
        See arXiv:2207.12421 Eq.6 for UR(0,1)
        Parameters:
        ----------
            indices:
                tuple of two spatial(!) orbital indices
            angle:
                Numeric or hashable type or tequila objective. Default is None and results
                in automatic naming as ("R",i,j)
            label:
                can be passed instead of angle to have auto-naming with label ("R",i,j,label)
                useful for repreating gates with individual variables
            control:
                List of possible control qubits
            assume_real:
                Assume that the wavefunction will always stay real.
                Will reduce potential gradient costs by a factor of 2
        """
        i, j = self.format_excitation_indices([(i, j)])[0]
        if not (self.select[i] == "F" and self.select[j] == "F"):
            raise TequilaException("Rotation not allowed, try Correlator")
        if angle is None:
            if label is None:
                angle = Variable(name=("R", i, j)) * numpy.pi
            else:
                angle = Variable(name=("R", i, j, label)) * numpy.pi

        circuit = self.make_excitation_gate(indices=[(2 * i, 2 * j)], angle=angle, assume_real=assume_real,
                                            control=control, *args, **kwargs)
        circuit += self.make_excitation_gate(indices=[(2 * i + 1, 2 * j + 1)], angle=angle, assume_real=assume_real,
                                             control=control, *args, **kwargs)
        return circuit

    def UC(self, i, j, angle=None, label=None, control=None, assume_real=True, *args, **kwargs):
        """
        Convenience function for orbital correlator circuit (correlating spatial orbital i and j through a spin-paired double excitation) with standard naming of variables
        See arXiv:2207.12421 Eq.22 for UC(1,2)

        Parameters:
        ----------
            indices:
                tuple of two spatial(!) orbital indices
            angle:
                Numeric or hashable type or tequila objective. Default is None and results
                in automatic naming as ("R",i,j)
            label:
                can be passed instead of angle to have auto-naming with label ("R",i,j,label)
                useful for repreating gates with individual variables
            control:
                List of possible control qubits
            assume_real:
                Assume that the wavefunction will always stay real.
                Will reduce potential gradient costs by a factor of 2
        """
        i, j = self.format_excitation_indices([(i, j)])[0]
        if angle is None:
            if label is None:
                angle = Variable(name=("C", i, j)) * numpy.pi
            else:
                angle = Variable(name=("C", i, j, label)) * numpy.pi
        if "jordanwigner" in self.transformation.name.lower() and not self.transformation.up_then_down:
            # for JW we can use the optimized form shown in arXiv:2207.12421 Eq.22
            target = [(2 * i, 2 * j), (2 * i + 1, 2 * j + 1)]
            G = self.make_excitation_generator(indices=target)
            P0 = self.make_excitation_generator(indices=target, form="p0")
            return gates.GeneralizedRotation(angle=angle, generator=G, p0=P0, assume_real=assume_real, **kwargs)
        else:
            return self.make_excitation_gate(indices=[(2 * i, 2 * j), (2 * i + 1, 2 * j + 1)], angle=angle,
                                             control=control, assume_real=assume_real, *args, **kwargs)

    def make_excitation_gate(self, indices: typing.Iterable[typing.Tuple[int, int]], angle, control=None, assume_real=True, **kwargs)->QCircuit:
        """
        Initialize a fermionic excitation gate defined as

        .. math::
            e^{-i\\frac{a}{2} G}
        with generator defines by the indices [(p0,q0),(p1,q1),...] or [p0,q0,p1,q1 ...]
        .. math::
            G = i(\\prod_{k} a_{p_k}^\\dagger a_{q_k} - h.c.)

        Parameters
        ----------
            indices:
                List of tuples that define the generator
            angle:
                Numeric or hashable type or tequila objective
            control:
                List of possible control qubits
            assume_real:
                Assume that the wavefunction will always stay real.
                Will reduce potential gradient costs by a factor of 2
        """
        if "warning" in kwargs:
            warning = kwargs["warning"]
            kwargs.pop("warning")
            allowed = self.verify_excitation(indices, warning=warning)
        else:
            self.verify_excitation(indices)
            allowed = True  # if it was False, previous line exception
        if (allowed):
            if "opt" in kwargs:
                opt = kwargs["opt"]
                kwargs.pop("opt")
            else:
                opt = True
            if not self.supports_ucc():
                raise TequilaException("Molecule with transformation {} does not support general UCC operations".format(
                    self.transformation))
            generator = self.make_excitation_generator(indices=indices, remove_constant_term=control is None)
            p0 = self.make_excitation_generator(indices=indices, form="P0", remove_constant_term=control is None)
            return QCircuit.wrap_gate(
                FermionicGateImpl(angle=angle, generator=generator, p0=p0,
                                  transformation=type(self.transformation).__name__.lower(), indices=indices,
                                  select=self.select, condense=self.condense, up_then_down=self.up_then_down,two_qubit=self.two_qubit,
                                  assume_real=assume_real, opt=opt, control=control, **kwargs))

    def make_excitation_generator(self, indices: typing.Iterable[typing.Tuple[int, int]], form: str = None, remove_constant_term: bool = True) -> QubitHamiltonian:
        """
        Notes
        ----------
        Creates the transformed hermitian generator of UCC type unitaries:
              M(a^\dagger_{a_0} a_{i_0} a^\dagger{a_1}a_{i_1} ... - h.c.)
              where the qubit map M depends is self.transformation

        Parameters
        ----------
        indices : typing.Iterable[typing.Tuple[int, int]] :
            List of tuples [(a_0, i_0), (a_1, i_1), ... ] - recommended format, in spin-orbital notation (alpha odd numbers, beta even numbers)
            can also be given as one big list: [a_0, i_0, a_1, i_1 ...]
        form : str : (Default value None):
            Manipulate the generator to involution or projector
            set form='involution' or 'projector'
            the default is no manipulation which gives the standard fermionic excitation operator back
        remove_constant_term: bool: (Default value True):
            by default the constant term in the qubit operator is removed since it has no effect on the unitary it generates
            if the unitary is controlled this might not be true!
        Returns
        -------
        type
            1j*Transformed qubit excitation operator, depends on self.transformation
        """
        if not self.supports_ucc():
            raise TequilaException(
                "Molecule with transformation {} does not support general UCC operations".format(self.transformation))
        # check indices and convert to list of tuples if necessary
        if len(indices) == 0:
            raise TequilaException("make_excitation_operator: no indices given")
        elif not isinstance(indices[0], typing.Iterable):
            if len(indices) % 2 != 0:
                raise TequilaException("make_excitation_generator: unexpected input format of indices\n"
                                       "use list of tuples as [(a_0, i_0),(a_1, i_1) ...]\n"
                                       "or list as [a_0, i_0, a_1, i_1, ... ]\n"
                                       "you gave: {}".format(indices))
            converted = [(indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)]
        else:
            converted = indices
        # convert everything to native python int
        # otherwise openfermion will complain
        converted = [(int(pair[0]), int(pair[1])) for pair in converted]
        # convert to openfermion input format
        ofi = []
        dag = []
        for pair in converted:
            assert (len(pair) == 2)
            ofi += [(int(pair[0]), 1),
                    (int(pair[1]), 0)]  # openfermion does not take other types of integers like numpy.int64
            dag += [(int(pair[0]), 0),
                    (int(pair[1]), 1)]  # to distinguis from HCB to JW, JW 0,1 HCB 2,3 for creation anihilation
        op = FermionOperator(tuple(ofi), 1.j)  # 1j makes it hermitian
        op += FermionOperator(tuple(reversed(dag)), -1.j)

        if isinstance(form, str) and form.lower() != 'fermionic':
            # indices for all the Na operators
            Na = [x for pair in converted for x in [(pair[0], 1), (pair[0], 0)]]
            # indices for all the Ma operators (Ma = 1 - Na)
            Ma = [x for pair in converted for x in [(pair[0], 0), (pair[0], 1)]]
            # indices for all the Ni operators
            Ni = [x for pair in converted for x in [(pair[1], 1), (pair[1], 0)]]
            # indices for all the Mi operators
            Mi = [x for pair in converted for x in [(pair[1], 0), (pair[1], 1)]]

            # can gaussianize as projector or as involution (last is default)
            if form.lower() == "p+":
                op *= 0.5
                op += FermionOperator(Na + Mi, 0.5)
                op += FermionOperator(Ni + Ma, 0.5)
            elif form.lower() == "p-":
                op *= 0.5
                op += FermionOperator(Na + Mi, -0.5)
                op += FermionOperator(Ni + Ma, -0.5)
            elif form.lower() == "g+":
                op += FermionOperator([], 1.0)  # Just for clarity will be subtracted anyway
                op += FermionOperator(Na + Mi, -1.0)
                op += FermionOperator(Ni + Ma, -1.0)
            elif form.lower() == "g-":
                op += FermionOperator([], -1.0)  # Just for clarity will be subtracted anyway
                op += FermionOperator(Na + Mi, 1.0)
                op += FermionOperator(Ni + Ma, 1.0)
            elif form.lower() == "p0":
                # P0: we only construct P0 and don't keep the original generator
                op = FermionOperator([], 1.0)  # Just for clarity will be subtracted anyway
                op += FermionOperator(Na + Mi, -1.0)
                op += FermionOperator(Ni + Ma, -1.0)
            else:
                raise TequilaException(
                    "Unknown generator form {}, supported are G, P+, P-, G+, G- and P0".format(form))
        qop = self.transformation(op)
        # remove constant terms
        # they have no effect in the unitary (if not controlled)
        if remove_constant_term:
            qop.qubit_operator.terms[tuple()] = 0.0

        # check if the operator is hermitian and cast coefficients to floats
        # in order to avoid trouble with the simulation backends
        assert qop.is_hermitian()
        for k, v in qop.qubit_operator.terms.items():
            qop.qubit_operator.terms[k] = to_float(v)
        if len(qop) == 0:
            warnings.warn("Excitation generator is a unit operator.\n"
                          "Non-standard transformations might not work with general fermionic operators\n"
                          "indices = " + str(indices), category=TequilaWarning)
        return qop.simplify()

    #Ansatzs and Algorithm
    def prepare_reference(self, state=None, *args, **kwargs):
        """
        Returns
        -------
        A tequila circuit object which prepares the reference of this molecule in the chosen transformation
        """
        if state is None:
            state = self._reference_state()
        reference_state = BitString.from_array(self.transformation.map_state(state=state))
        U = prepare_product_state(reference_state)
        # prevent trace out in direct wfn simulation
        if(not self.condense or self.two_qubit):
            U.n_qubits = self.n_orbitals * 2  # adapt when tapered transformations work
        else:
            U.n_qubits = len(self.FER_SO) + len(self.BOS_MO)   # adapt when tapered transformations work
        return U
    
    def prepare_hardcore_boson_reference(self):
        """
        Prepare reference state in the Hardcore-Boson approximation (eqch qubit represents two spin-paired electrons)
        Returns
        -------
        tq.QCircuit that prepares the HCB reference
        """
        pos = self.transformation.pos
        U = gates.X(target=[pos[2*i.idx] for i in self.reference_orbitals])
        U.n_qubits = self.n_orbitals
        return U
    
    def make_ansatz(self, name: str, *args, **kwargs)->QCircuit:
        """
        Automatically calls the right subroutines to construct ansatze implemented in tequila.chemistry
        name: namne of the ansatz, examples are: UpCCGSD, UpCCD, SPA, UCCSD, SPA+UpCCD, SPA+GS
        """
        name = name.lower()
        if name.strip() == "":
            return QCircuit()
        if "+" in name:
            U = QCircuit()
            subparts = name.split("+")
            U = self.make_ansatz(name=subparts[0], *args, **kwargs)
            # making sure the there are is no undesired behaviour in layers after +
            # reference should not be included since we are not starting from |00...0> anymore
            if "include_reference" in kwargs:
                kwargs.pop("include_reference")
            # hcb optimization can also not be used (in almost all cases)
            if "hcb_optimization" in kwargs:
                kwargs.pop("hcb_optimization")
            # making sure that we have no repeating variable names
            label = None
            if "label" in kwargs:
                label = kwargs["label"]
                kwargs.pop("label")
            for i,subpart in enumerate(subparts[1:]):
                U += self.make_ansatz(name=subpart, *args, label=(label,i), include_reference=False, hcb_optimization=False, **kwargs)
            return U

        if name == "uccsd":
            return self.make_uccsd_ansatz(*args, **kwargs)
        elif "spa" in name.lower():
            if "hcb" not in kwargs:
                hcb = False
                if "hcb" in name.lower():
                    hcb = True
                kwargs["hcb"]=hcb
            return self.make_spa_ansatz(*args, **kwargs)
        elif "d" in name or "s" in name:
            return self.make_upccgsd_ansatz(name=name, *args, **kwargs)
        else:
            raise TequilaException("unknown ansatz with name={}".format(name))

    def make_spa_ansatz(self, edges=None, hcb=False, use_units_of_pi=False, label=None, optimize=None, ladder=True):
        """
        Separable Pair Ansatz (SPA) for general molecules
        see arxiv:
        edges: a list of tuples that contain the orbital indices for the specific pairs
               one example: edges=[(0,), (1,2,3), (4,5)] are three pairs, one with a single orbital [0], one with three orbitals [1,2,3] and one with two orbitals [4,5]
        hcb: spa ansatz in the hcb (hardcore-boson) space without transforming to current transformation (e.g. JordanWigner), use this for example in combination with the self.make_hardcore_boson_hamiltonian() and see the article above for more info
        use_units_of_pi: circuit angles in units of pi
        label: label the variables in the circuit
        optimize: optimize the circuit construction (see article). Results in shallow circuit from Ry and CNOT gates
        ladder: if true the excitation pattern will be local. E.g. in the pair from orbitals (1,2,3) we will have the excitations 1->2 and 2->3, if set to false we will have standard coupled-cluster style excitations - in this case this would be 1->2 and 1->3
        """
        if edges is None:
            # raise TequilaException(
            #     "SPA ansatz within a standard orbital basis needs edges. Please provide with the keyword edges.\nExample: edges=[(0,1,2),(3,4)] would correspond to two edges created from orbitals (0,1,2) and (3,4), note that orbitals can only be assigned to a single edge")
            edges = self.get_spa_edges()
        # sanity checks
        # current SPA implementation needs even number of electrons
        if self.n_electrons % 2 != 0:
            raise TequilaException(
                "need even number of electrons for SPA ansatz.\n{} active electrons".format(self.n_electrons))
        # making sure that enough edges are assigned
        if len(edges) != self.n_electrons // 2:
            raise TequilaException(
                "number of edges need to be equal to number of active electrons//2\n{} edges given\n{} active electrons\nfrozen core is {}".format(
                    len(edges), self.n_electrons, self.parameters.frozen_core))
        # making sure that orbitals are uniquely assigned to edges
        for edge in edges:
            for orbital in edge:
                for edge2 in edges:
                    if edge2 == edge:
                        continue
                    elif orbital in edge2:
                        raise TequilaException(
                            "make_spa_ansatz: faulty list of edges, orbitals are overlapping e.g. orbital {} is in edge {} and edge {}".format(
                                orbital, edge, edge2))

        # auto assign if the circuit construction is optimized
        # depending on the current qubit encoding (if hcb_to_me is implemnented we can optimize)
        if optimize is None:
            try:
                have_hcb_to_me = self.hcb_to_me() is not None
            except:
                have_hcb_to_me = False
            if have_hcb_to_me or not len(self.FER_SO):
                optimize = True
            else:
                optimize = False
        pos = self.transformation.pos
        U = QCircuit()
        # construction of the optimized circuit
        if optimize:
            # circuit in HCB representation
            # depends a bit on the ordering of the spin-orbitals in the encoding
            # here we transform it to the qubits representing the up-spins
            # the hcb_to_me sequence will then transfer to the actual encoding later
            for edge_orbitals in edges:
                edge_qubits = [pos[2*i] for i in edge_orbitals]
                U += gates.X(edge_qubits[0])
                if len(edge_qubits) == 1:
                    continue
                for i in range(1, len(edge_qubits)):
                    q1 = edge_qubits[i]
                    c = edge_qubits[i - 1]
                    if not ladder:
                        c = edge_qubits[0]
                    angle = Variable(name=((edge_orbitals[i - 1], edge_orbitals[i]), "D", label))
                    if use_units_of_pi:
                        angle = angle * numpy.pi
                    if i - 1 == 0:
                        U += gates.Ry(angle=angle, target=q1, control=None)
                    else:
                        U += gates.Ry(angle=angle, target=q1, control=c)
                    U += gates.CNOT(q1, c)
            if not hcb:
                U += self.transformation.hcb_to_me()
        else:
            # construction of the non-optimized circuit
            if hcb:
                U = self.prepare_hardcore_boson_reference()
            else:
                U = self.prepare_reference()
            # will only work if the first orbitals in the edges are the reference orbitals
            sane = True
            reference_orbitals = self.reference_orbitals
            for edge_qubits in edges:
                if self.orbitals[edge_qubits[0]] not in reference_orbitals:
                    sane = False
                if len(edge_qubits) > 1:
                    for orbital in edge_qubits[1:]:
                        if self.orbitals[orbital] in reference_orbitals:
                            sane = False
            if not sane:
                raise TequilaException(
                    "Non-Optimized SPA (e.g. with encodings that are not JW) will only work if the first orbitals of all SPA edges are occupied reference orbitals and all others are not. You gave edges={} and reference_orbitals are {}".format(
                        edges, reference_orbitals))

            for edge_qubits in edges:
                previous = edge_qubits[0]
                if len(edge_qubits) > 1:
                    for q1 in edge_qubits[1:]:
                        c = previous
                        if not ladder:
                            c = edge_qubits[0]
                        angle = Variable(name=((c, q1), "D", label))
                        if use_units_of_pi:
                            angle = angle * numpy.pi
                        if hcb:
                            U += self.make_hardcore_boson_excitation_gate(indices=[(q1, c)], angle=angle)
                        else:
                            U += self.make_excitation_gate(indices=[(2 * c, 2 * q1), (2 * c + 1, 2 * q1 + 1)],angle=angle)
                        previous = q1
        if not self.condense:
            U.n_qubits = 2*self.n_orbitals
        return U

    def make_upccgsd_ansatz(self, include_reference: bool = True, name: str = "UpCCGSD",label: str = None,order: int = None,
                            assume_real: bool = True,hcb_optimization: bool = None,spin_adapt_singles: bool = True,
                            neglect_z:bool=False, mix_sd:bool=False,firts_double:bool=True,*args, **kwargs)->QCircuit:
        """
        UpGCCSD Ansatz similar as described by Lee et. al.

        Parameters
        ----------
        include_reference
            include the HF reference state as initial state
        indices
            pass custom defined set of indices from which the ansatz will be created
            List of tuples of tuples spin-indices e.g. [((2*p,2*q),(2*p+1,2*q+1)), ...]
        label
            An additional label that is set with the variables
            default is None and no label will be set: variables names will be
            (x, (p,q)) for x in range(order)
            with a label the variables will be named
            (label, (x, (p,q)))
        order
            Order of the ansatz (default is 1)
            determines how often the ordering gets repeated
            parameters of repeating layers are independent
        assume_real
            assume a real wavefunction (that is always the case if the reference state is real)
            reduces potential gradient costs from 4 to 2
        mix_sd
            Changes the ordering from first all doubles and then all singles excitations (DDDDD....SSSS....) to
            a mixed order (DS-DS-DS-DS-...) where one DS pair acts on the same MOs. Useful to consider when systems
            with high electronic correlation and system high error associated with the no Trotterized UCC.
        firts_double
            changes the excitation order from  Doubles then Singles if True, to Singles then Doubles if False
        Returns
        -------
            UpGCCSD ansatz
        """

        name = name.upper()

        if ("A" in name) and neglect_z is None:
            neglect_z = True
        else:
            neglect_z = False

        if order is None:
            try:
                if "-" in name:
                    order = int(name.split("-")[0])
                else:
                    order = 1
            except:
                order = 1

        indices = self.make_upccgsd_indices(key=name)
        # check if the used qubit encoding has a hcb transformation
        have_hcb_trafo = self.transformation.hcb_to_me() is not None

        # consistency checks for optimization
        if have_hcb_trafo and hcb_optimization is None and include_reference:
            hcb_optimization = True
        if "HCB" in name:
            hcb_optimization = True
        if hcb_optimization and not have_hcb_trafo and "HCB" not in name:
            raise TequilaException(
                "use_hcb={} but transformation={} has no \'hcb_to_me\' function. Try transformation=\'ReorderedJordanWigner\'".format(
                    hcb_optimization, self.transformation))
        if "S" in name and "HCB" in name:
            if "HCB" in name and "S" in name:
                raise Exception(
                    "name={}, Singles can't be realized without mapping back to the standard encoding leave S or HCB out of the name".format(
                        name))
        if hcb_optimization and mix_sd and (order==1):
            raise TequilaException("Mixed SD can not be employed together with HCB Optimization with only one UCC layer")
        if hcb_optimization and not firts_double and (order==1):
            raise TequilaException("First Doubles can not be employed together with HCB Optimization with only one UCC layer")
        # convenience
        S = "S" in name.upper()
        D = "D" in name.upper()
        # first layer
        if not hcb_optimization:
            U = QCircuit()
            if include_reference:
                U = self.prepare_reference()

            U += self.make_upccgsd_layer(include_singles=S, include_doubles=D, indices=indices, assume_real=assume_real,
                                         label=(label, 0), spin_adapt_singles=spin_adapt_singles, mix_sd=mix_sd,
                                         firts_double=firts_double, *args, **kwargs)
        else:
            U = QCircuit()
            if include_reference:
                U = self.prepare_hardcore_boson_reference()
            if D:
                U += self.make_hardcore_boson_upccgd_layer(indices=indices, assume_real=assume_real, label=(label, 0),*args, **kwargs)
            U += self.transformation.hcb_to_me()
            if S:
                U += self.make_upccgsd_singles(indices=indices, assume_real=assume_real, label=(label, 0),
                                               spin_adapt_singles=spin_adapt_singles, neglect_z=neglect_z, *args,**kwargs)

        for k in range(1, order):
            U += self.make_upccgsd_layer(include_singles=S, include_doubles=D, indices=indices, label=(label, k),
                                         spin_adapt_singles=spin_adapt_singles, neglect_z=neglect_z, mix_sd=mix_sd,firts_double=firts_double)
        if not self.condense:
            U.n_qubits = 2*self.n_orbitals
        return U

    def make_upccgsd_layer(self, indices, include_singles=True, include_doubles=True, assume_real=True, label=None,
                           spin_adapt_singles: bool = True, angle_transform=None, mix_sd=False, neglect_z=False,
                           firts_double:bool=True,*args,**kwargs):
        U = QCircuit()
        if include_singles and not mix_sd and not firts_double:
            U += self.make_upccgsd_singles(indices=indices, assume_real=assume_real, label=label,
                                           spin_adapt_singles=spin_adapt_singles, angle_transform=angle_transform,
                                           neglect_z=neglect_z)
        for idx in indices:
            assert len(idx) == 1
            idx = idx[0]
            angle = (tuple([idx]), "D", label)
            if include_singles and mix_sd and not firts_double:
                U += self.make_upccgsd_singles(indices=[(idx,)], assume_real=assume_real, label=label,
                                               spin_adapt_singles=spin_adapt_singles, angle_transform=angle_transform,
                                               neglect_z=neglect_z)
            if include_doubles:
                if "jordanwigner" in self.transformation.name.lower() and not self.transformation.up_then_down:
                    # we can optimize with qubit excitations for the JW representation
                    target = [(2*idx[0], 2*idx[1]),(2*idx[0]+1, 2*idx[1]+1)]
                    G = self.make_excitation_generator(indices=target)
                    P0 = self.make_excitation_generator(indices=target,form="p0")
                    U += gates.GeneralizedRotation(angle=angle,generator=G, p0=P0,assume_real=assume_real, **kwargs)
                else:
                    U += self.make_excitation_gate(angle=angle,indices=((2 * idx[0], 2 * idx[1]), (2 * idx[0] + 1, 2 * idx[1] + 1)),
                                                   assume_real=assume_real, **kwargs)
            if include_singles and mix_sd and firts_double:
                U += self.make_upccgsd_singles(indices=[(idx,)], assume_real=assume_real, label=label,
                                               spin_adapt_singles=spin_adapt_singles, angle_transform=angle_transform,
                                               neglect_z=neglect_z)

        if include_singles and not mix_sd and firts_double:
            U += self.make_upccgsd_singles(indices=indices, assume_real=assume_real, label=label,
                                           spin_adapt_singles=spin_adapt_singles, angle_transform=angle_transform,
                                           neglect_z=neglect_z)
        return U
    
    def make_upccgsd_singles(self, indices="UpCCGSD", spin_adapt_singles=True, label=None, angle_transform=None,
                             assume_real=True, neglect_z=False,*args, **kwargs):
        if neglect_z and "jordanwigner" not in self.transformation.name.lower():
            raise TequilaException(
                "neglegt-z approximation in UpCCGSD singles needs the (Reversed)JordanWigner representation")
        if hasattr(indices, "lower"):
            indices = self.make_upccgsd_indices(key=indices)
        U = QCircuit()
        for idx in indices:
            assert len(idx) == 1
            idx = idx[0]
            if (self.select[idx[0]]=="F" and self.select[idx[1]]=="F"):
                if spin_adapt_singles:
                    angle = (idx, "S", label)
                    if angle_transform is not None:
                        angle = angle_transform(angle)
                    if neglect_z:
                        targeta = [self.transformation.pos[2*idx[0]], self.transformation.pos[2*idx[1]]]
                        targetb = [self.transformation.pos[2*idx[0]+1], self.transformation.pos[2*idx[1]+1]]
                        U += gates.QubitExcitation(angle=angle, target=targeta, assume_real=assume_real, **kwargs)
                        U += gates.QubitExcitation(angle=angle, target=targetb, assume_real=assume_real, **kwargs)
                    else:
                        U += self.make_excitation_gate(angle=angle, indices=[(2 * idx[0], 2 * idx[1])],
                                                       assume_real=assume_real, **kwargs)
                        U += self.make_excitation_gate(angle=angle, indices=[(2 * idx[0] + 1, 2 * idx[1] + 1)],
                                                       assume_real=assume_real, **kwargs)
                else:
                    angle1 = (idx, "SU", label)
                    angle2 = (idx, "SD", label)
                    if angle_transform is not None:
                        angle1 = angle_transform(angle1)
                        angle2 = angle_transform(angle2)
                    if neglect_z:
                        targeta = [self.transformation.up(idx[0]), self.transformation.up(idx[1])]
                        targetb = [self.transformation.down(idx[0]), self.transformation.down(idx[1])]
                        U += gates.QubitExcitation(angle=angle1, target=targeta, assume_real=assume_real, *kwargs)
                        U += gates.QubitExcitation(angle=angle2, target=targetb, assume_real=assume_real, *kwargs)
                    else:
                        U += self.make_excitation_gate(angle=angle1, indices=[(2 * idx[0], 2 * idx[1])],
                                                       assume_real=assume_real, **kwargs)
                        U += self.make_excitation_gate(angle=angle2, indices=[(2 * idx[0] + 1, 2 * idx[1] + 1)],
                                                       assume_real=assume_real, **kwargs)

        return U
    
    def make_hardcore_boson_excitation_gate(self, indices, angle, control=None, assume_real=True,
                                            compile_options="optimize"):
        """
        Make excitation generator in the hardcore-boson approximation (all electrons are forced to spin-pairs)
        use only in combination with make_hardcore_boson_hamiltonian()

        Parameters
        ----------
        indices
        angle
        control
        assume_real
        compile_options

        Returns
        -------

        """
        target = []
        for pair in indices:
            assert len(pair) == 2
            target += [self.transformation.pos[2*pair[0]], self.transformation.pos[2*pair[1]]]
        if self.transformation.up_then_down:
            consistency = [x < self.n_orbitals for x in target]
        else:
            consistency = [True]
        if not all(consistency):
            raise TequilaException(
                "make_hardcore_boson_excitation_gate: Inconsistencies in indices={} for encoding: {}".format(indices, self.transformation))
        return gates.QubitExcitation(angle=angle, target=target, assume_real=assume_real, control=control,compile_options=compile_options)
    
    def hcb_to_me(self,**kwargs):
        return self.transformation.hcb_to_me(**kwargs)

    def compute_energy(self, method:str, *args, **kwargs):
        """
        Call classical methods over PySCF (needs to be installed) or
        use as a shortcut to calculate quantum energies (see make_upccgsd_ansatz)

        Parameters
        ----------
        method: method name
                classical: HF, MP2, CCSD, CCSD(T), FCI -- with pyscf
                quantum: UpCCD, UpCCSD, UpCCGSD, k-UpCCGSD, UCCSD,
                see make_upccgsd_ansatz of the this class for more information
        args
        kwargs: for quantum methods, keyword arguments for minimizer

        Returns
        -------

        """
        method = method.lower()
        if 'hybrid' in method:
            method = method.replace('hybrid','').strip()
            if method[0] == '-':
                method = method[1:]
            return self.compute_restricted_energy(method,*args,**kwargs)
        else:
            return super().compute_energy(method,*args,**kwargs)
    
    def compute_restricted_energy(self, method:str, *args, **kwargs):
        """
            Call classical methods over PySCF (needs to be installed) or
            use as a shortcut to calculate quantum energies (see make_upccgsd_ansatz)
            Difference with self.compute_energy() is that here the inteaction restriction
            is considered
            Parameters
            ----------
            method: method name
                    classical: HF, MP2, CCSD, CCSD(T), FCI -- with pyscf
                    quantum: UpCCD, UpCCSD, UpCCGSD, k-UpCCGSD, UCCSD,
                    see make_upccgsd_ansatz of the this class for more information
            args
            kwargs: for quantum methods, keyword arguments for minimizer

            Returns
            -------
        """
        from tequila import Molecule as mol
        c, h, g = self.get_integrals()
        BOS_L = self.BOS_MO
        NBOS_L = self.FER_MO
        for i in BOS_L:
            for j in BOS_L:
                if i != j:
                    h[i][j] = 0.
        for i in BOS_L:
            for j in NBOS_L:
                h[i][j] = 0.
                h[j][i] = 0.
        new_g = numpy.zeros(shape=(len(g.elems), len(g.elems), len(g.elems), len(g.elems)))
        for i in NBOS_L:
            for j in NBOS_L:
                for k in NBOS_L:
                    for l in NBOS_L:
                        new_g[i][j][k][l] = g.elems[i][j][k][l]
        for i in BOS_L:
            for j in BOS_L:
                new_g[i][i][j][j] = g.elems[i][i][j][j]
                new_g[i][j][j][i] = g.elems[i][j][j][i]
                new_g[i][j][i][j] = g.elems[i][j][i][j]
        for i in NBOS_L:
            for j in NBOS_L:
                for k in BOS_L:
                    new_g[i][j][k][k] = g.elems[i][j][k][k]
                    new_g[k][k][i][j] = g.elems[k][k][i][j]
                    new_g[i][k][k][j] = g.elems[i][k][k][j]
                    new_g[k][i][j][k] = g.elems[k][i][j][k]
                    new_g[i][k][j][k] = g.elems[i][k][j][k]
                    new_g[k][i][k][j] = g.elems[k][i][k][j]
        g.elems = new_g
        parameters = copy.deepcopy(self.parameters)
        return mol(parameters=parameters,one_body_integrals=h,two_body_integrals=g,nuclear_repulsion=c,backend='pyscf',transformation=self.transformation.name,n_electrons=self.n_electrons).compute_energy(method=method, *args,**kwargs)

    def get_restricted_integrals(self, *args, **kwargs):
        c, h, g = self.get_integrals()
        BOS_L = self.BOS_MO
        NBOS_L = self.FER_MO
        for i in BOS_L:
            for j in BOS_L:
                if i != j:
                    h[i][j] = 0.
        for i in BOS_L:
            for j in NBOS_L:
                h[i][j] = 0.
                h[j][i] = 0.
        new_g = numpy.zeros(shape=(len(g.elems), len(g.elems), len(g.elems), len(g.elems)))
        for i in NBOS_L:
            for j in NBOS_L:
                for k in NBOS_L:
                    for l in NBOS_L:
                        new_g[i][j][k][l] = g.elems[i][j][k][l]
        for i in BOS_L:
            for j in BOS_L:
                new_g[i][i][j][j] = g.elems[i][i][j][j]
                new_g[i][j][j][i] = g.elems[i][j][j][i]
                new_g[i][j][i][j] = g.elems[i][j][i][j]
        for i in NBOS_L:
            for j in NBOS_L:
                for k in BOS_L:
                    new_g[i][j][k][k] = g.elems[i][j][k][k]
                    new_g[k][k][i][j] = g.elems[k][k][i][j]
                    new_g[i][k][k][j] = g.elems[i][k][k][j]
                    new_g[k][i][j][k] = g.elems[k][i][j][k]
                    new_g[i][k][j][k] = g.elems[i][k][j][k]
                    new_g[k][i][k][j] = g.elems[k][i][k][j]
        g.elems = new_g
        return c,h,g

    def get_xyz(self)->str:
        geom = self.parameters.get_geometry()
        f = ''
        f += f'{len(geom)}\n'
        f += f'{self.parameters.name}\n'
        for at in geom:
            f += f'{at[0]} {at[1][0]} {at[1][1]} {at[1][2]}\n'
        return f

    def graph(self):
        return Graph.parse_xyz(self.get_xyz())
    
    def get_spa_edges(self,collapse:bool=True,strip_orbitals:bool=None):
        if strip_orbitals  is None:
            strip_orbitals = not self.integral_manager.active_space_is_trivial()
        return self.graph().get_spa_edges(collapse=collapse,strip_orbitals=strip_orbitals)
    
    def get_spa_guess(self,strip_orbitals:bool=None):
        if strip_orbitals  is None:
            strip_orbitals = not self.integral_manager.active_space_is_trivial()
        return self.graph().get_orbital_coefficient_matrix(strip_orbitals=strip_orbitals)

    def get_spa_edges_and_guess(self,collapse:bool=True,strip_orbitals:bool=None):
        if strip_orbitals  is None:
            strip_orbitals = not self.integral_manager.active_space_is_trivial()
        g = self.graph()
        return g.get_spa_edges(collapse=collapse,strip_orbitals=strip_orbitals),g.get_orbital_coefficient_matrix(strip_orbitals=strip_orbitals)
