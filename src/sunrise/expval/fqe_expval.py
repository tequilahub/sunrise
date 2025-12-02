from typing import Union, Tuple, List, Any, Dict

import tequila as tq
from numpy.ma.core import shape
from tequila import QubitWaveFunction,TequilaException
from tequila.objective.objective import Objective,Variables,assign_variable

import fqe

from sunrise.expval.fqe_utils import *
from sunrise.expval.fermionic_utils import *
from sunrise.fermionic_operations.circuit import FCircuit




class FQEBraKet:

    def __init__(self,
                 ket: FCircuit = None, bra: FCircuit = None,
                 one_body_integrals: Any = None, two_body_integrals = None, constant: int = None,
                 mol: QuantumChemistryBase= None,
                 *args, **kwargs
                 ):
        """

        :param ket: Fcircuit representing the ket state. 
        :param bra (optional): Fcircuit representing the bra state. If None bra is assumed to be the same as the ket
        :param one_body_integrals (optional): one body integral used for calculating the Hamiltonian. 
                                            If None Hamiltonian is not constructed.
        :param two_body_integrals (optional): two body integral used for calculating the Hamiltonian
                                            If None Hamiltonian is not constructed.
        :param constant (optional): constant term used for calculating the Hamiltonian.
                                            If None Hamiltonian is not constructed.
        :param mol (optional): QuantumChemistryBase molecule object containing molecule specidic information.
                    If None, number of electrons must be defined in the kwargs and Hamiltonian is not constructed.
        :param args:
        :param kwargs:
                    ket: FCircuit representing the ket state.
                    ket_fcircuit: FCircuit representing the ket state.

                    bra: FCircuit representing the bra state.
                    bra_fcircuit: FCircuit representing the bra state.

                    molecule: QuantumChemistryBase molecule object containing molecule specidic information.
                    mol: QuantumChemistryBase molecule object containing molecule specidic information.

                    one_body_integrals: one body integral used for calculating the Hamiltonian.
                    h: one body integral used for calculating the Hamiltonian.
                    init1e: one body integral used for calculating the Hamiltonian.

                    two_body_integrals: two body integral used for calculating the Hamiltonian.
                    g: two body integral used for calculating the Hamiltonian.
                    init2e: two body integral used for calculating the Hamiltonian.

                    constant: constant term used for calculating the Hamiltonian.
                    c: constant term used for calculating the Hamiltonian.

                    operator: string or openfermion FermionOperator defining a custom operator to be used.
                              If string is "h" or "hamiltonian" Hamiltonian is constructed from integrals or molecule.
                              If string is "i" or "identity" the identity operator is used

                    n_orbitals: number of orbitals in the system. Needed if no molecule or integrals are provided.
                    n_ele: number of electrons in the system. Needed if no molecule is provided.


        """
        if "ket_fcircuit" in kwargs:
            ket = kwargs["ket_fcircuit"]
            kwargs.pop("ket_fcircuit")
        elif "ket" in kwargs:
            ket = kwargs["ket"]
            kwargs.pop("ket")
        elif "U" in kwargs:
            ket = kwargs["U"]
            kwargs.pop("U")
        elif "circuit" in kwargs:
            ket = kwargs["circuit"]
            kwargs.pop("circuit")

        if ket is None:
            raise ValueError("No ket fcircuit provided")

        if "bra_fcircuit" in kwargs:
            bra = kwargs["bra_fcircuit"]
            kwargs.pop("bra_fcircuit")
        elif "bra" in kwargs:
            bra = kwargs["bra"]
            kwargs.pop("bra")

        molecule_flag = False
        if mol is not None:
            molecule_flag = True
        elif "molecule" in kwargs:
            mol = kwargs["molecule"]
            kwargs.pop("molecule")
            molecule_flag = True
        elif "mol" in kwargs:
            mol = kwargs["mol"]
            kwargs.pop("mol")
            molecule_flag = True

        if "one_body_integrals" in kwargs:
            one_body_integrals = kwargs["one_body_integrals"]
            kwargs.pop("one_body_integrals")
        elif "h" in kwargs:
            one_body_integrals = kwargs["h"]
            kwargs.pop("h")
        elif "init1e" in kwargs:
            one_body_integrals = kwargs["init1e"]
            kwargs.pop("init1e")

        if "two_body_integrals" in kwargs:
            two_body_integrals = kwargs["two_body_integrals"]
            kwargs.pop("two_body_integrals")
        elif "g" in kwargs:
            two_body_integrals = kwargs["g"]
            kwargs.pop("g")
        elif "init2e" in kwargs:
            two_body_integrals = kwargs["init2e"]
            kwargs.pop("init2e")


        if "constant" in kwargs:
            constant = kwargs["constant"]
            kwargs.pop("constant")
        elif "c" in kwargs:
            constant = kwargs["c"]
            kwargs.pop("c")


        operator_flag_h = False
        operator_flag_custom = False
        if 'H' in kwargs and kwargs['H'] is not None:
            if 'operator' in kwargs and kwargs['operator'] is not None:
                raise TequilaException('Two operators provided?')
            kwargs['operator'] = kwargs['H']
            kwargs.pop('H')
        if "operator" in kwargs and kwargs["operator"] is not None :
            operator = kwargs["operator"]
            kwargs.pop("operator")
            if isinstance(operator, str):
                if operator.lower() == "h" or operator.lower() == "hamiltonian":
                    operator_flag_h = True
                elif operator.lower() == "i" or operator.lower() == "identity":
                   pass
                else:
                    raise TequilaException("Not implemented operator {}".format(operator))

            elif isinstance(operator,openfermion.ops.operators.fermion_operator.FermionOperator):
                operator_flag_custom = True

            else:
                raise TequilaException("Not recognized format {}".format(operator))


        if (one_body_integrals is None and two_body_integrals is not None) \
                or one_body_integrals is not None and two_body_integrals is None:
            raise TequilaException("Both integrals are needed two conunstract a Hamiltonian")
        if (one_body_integrals is not None) and (two_body_integrals is not None) and (constant is None):
            raise TequilaException("Constant not defined")

        if one_body_integrals is not None and two_body_integrals is not None and constant is not None:
            integral_flag = True
        else:
            integral_flag = False

        construct_ham= True
        if integral_flag is False and molecule_flag is False and operator_flag_custom is False:
            construct_ham = False
            if operator_flag_h is True:
                raise TequilaException("No integrals or molecule provided to construct Hamiltonian")

        if construct_ham:
            if integral_flag is True:
                self.h_of = make_fermionic_hamiltonian(one_body_integrals, two_body_integrals, constant,)
                self.n_orbitals = one_body_integrals.shape[0]
                self.h_fqe = fqe.get_hamiltonian_from_openfermion(self.h_of,norb=self.n_orbitals)
                n_ele = kwargs.get("n_ele")

            elif operator_flag_custom is True:
                self.h_of = operator
                if mol is None:
                    self.n_orbitals = kwargs.get("n_orbitals")
                    n_ele = kwargs.get("n_ele")
                else:
                    c,h,g = mol.get_integrals()
                    self.n_orbitals = h.shape[0]
                    n_ele = mol.n_electrons
                self.h_fqe = fqe.get_hamiltonian_from_openfermion(self.h_of,norb=self.n_orbitals)
            elif molecule_flag is True:
                c,h,g = mol.get_integrals()
                self.h_of = make_fermionic_hamiltonian(one_body_integrals=h, two_body_integrals=g.elems, constant=c)
                self.n_orbitals = h.shape[0]
                self.h_fqe = fqe.get_hamiltonian_from_openfermion(self.h_of,norb=self.n_orbitals)
                n_ele = mol.n_electrons
        else:
            self.h_fqe = None
            self.n_orbitals = kwargs.get("n_orbitals")
            n_ele = kwargs.get("n_ele")

        if self.n_orbitals is None:
            raise TequilaException("n_orbitals not defined in the kwargs")
        if n_ele is None:
            raise TequilaException("Number of electrons not defined in the kwargs")

        if n_ele > self.n_orbitals:
            raise TequilaException("number of electrons must not be greater than number of orbitals")

        self.n_ele = n_ele
        ket = ket.to_udud(norb=self.n_orbitals)

        self.parameter_map_ket = []
        for x in ket.variables:
            self.parameter_map_ket.append(tq.assign_variable(x))
        self.parameter_map_ket = list(dict.fromkeys(self.parameter_map_ket))

        # self.parameter_map_ket = [tq.assign_variable(x) for x in ket_fcircuit.variables]
        self._constant_dict_ket = {}
        ket_instructions = ket.extract_indices()
        self.ket_instructions = ket_instructions
        ket_angles = ket.variables
        self.ket_angles = ket_angles
        self.non_fixed_variables_ket = ket.extract_variables()

        self.ket_generator = create_fermionic_generators(ket_instructions, ket_angles)
        self.ket_generator_idx_map={}
        self.bra_generator_idx_map={}
        for i, gen in enumerate(self.ket_generator.keys()):
            self.ket_generator_idx_map[gen] = self.ket_instructions[i]

        bin_dict = generate_of_binary_dict(self.n_orbitals, self.n_ele // 2)

        self.ket = fqe.Wavefunction(param=[[self.n_ele, 0, self.n_orbitals]])  # probably only works for H

        if ket.initial_state is None:
            self.ket.set_wfn(strategy='hartree-fock')
        else:
            set_init_state(wfn = self.ket, n_ele=self.n_ele, n_orb=self.n_orbitals, init_state=ket.initial_state, bin_dict=bin_dict)


        bra_instructions = None
        bra_angles = None
        init_bra = None
        non_fixed_variables_bra = None
        self.constant_dict_bra = {}
        if bra is not None:
            bra = bra.to_udud(norb=self.n_orbitals)

            self.parameter_map_bra = []
            for x in bra.variables:
                    self.parameter_map_bra.append(tq.assign_variable(x))
            self.parameter_map_bra = list(dict.fromkeys(self.parameter_map_bra))
            bra_instructions = bra.extract_indices()
            bra_angles = bra.variables
            non_fixed_variables_bra = bra.extract_variables()

            self.bra_generator = create_fermionic_generators(bra_instructions, bra_angles)
            for i, gen in enumerate(self.bra_generator.keys()):
                self.bra_generator_idx_map[gen] = bra_instructions[i]
            init_bra = bra.initial_state

        self.bra_instructions = bra_instructions
        self.bra_angles = bra_angles
        self.init_bra = init_bra
        self.non_fixed_variables_bra = non_fixed_variables_bra


        if bra is None:
            self.bra = None
        else:
            self.bra = fqe.Wavefunction(param=[[self.n_ele, 0, self.n_orbitals]])

            if self.init_bra is None:
                self.bra.set_wfn(strategy='hartree-fock')
            else:
                set_init_state(wfn=self.bra, n_ele=self.n_ele,n_orb=self.n_orbitals, init_state=init_bra, bin_dict=bin_dict)

        self.ket_time_evolved = None
        self.bra_time_evolved = None


    def __call__(self, variables, *args, **kwargs) -> float:
        """

        :param variables: Variables to be used on the time evolution. Can be a list, dict or tequila.Variables object
        :param args:
        :param kwargs:
        :return: Expectation value <bra|H|ket> or <ket|ket> if no Hamiltonian is provided
        """
        internal_variables = variables
        if self.bra is not None:
            parameter_map = self.parameter_map_bra + self.parameter_map_ket
        else:
            parameter_map = self.parameter_map_ket

        if isinstance(internal_variables, Variables):
            pass
        else:
            if type(internal_variables) is not dict and internal_variables is not None:
                    internal_variables = {parameter_map[i]: internal_variables[i] for i in range(len(internal_variables))}
            internal_variables = tq.format_variable_dictionary(internal_variables)

        if self.constant_dict_ket != {}:
            for const in self.constant_dict_ket.keys():

                internal_variables[const] = internal_variables[const] + self.constant_dict_ket[const]
            internal_variables["p0sign_ket"] = -np.pi # const


        parameters_ket = [x(internal_variables) for x in self.parameter_map_ket]

        zip_ket = zip(parameters_ket, self.ket_generator.values())
        ket_t = self.ket


        for argument in zip_ket:
            ket_t = ket_t.time_evolve(-0.5 * argument[0], argument[1])


        #bra time evolution
        if self.bra_instructions is  None:
            bra_t = None
        else:
            if self.constant_dict_bra != {}:
                for const in self.constant_dict_bra.keys():
                    internal_variables[const] = internal_variables[const] + self.constant_dict_bra[
                        const]
                internal_variables["p0sign_bra"] = -np.pi  # const

            parameters_bra = [x(internal_variables) for x in self.parameter_map_bra]
            zip_bra = zip(parameters_bra, self.bra_generator.values())
            bra_t = self.bra
            for argument in zip_bra:
                bra_t = bra_t.time_evolve(-0.5 * argument[0], argument[1])

        self.ket_time_evolved = ket_t
        self.bra_time_evolved = bra_t

        if self.h_fqe is not None:
            result = fqe.expectationValue(wfn=ket_t, ops=self.h_fqe, brawfn=bra_t)
        else:
            result = fqe.dot(bra_t, ket_t)

        return result.real


    def print_ket(self):
        self.ket.print_wfn()

    def print_bra(self):
        self.bra.print_wfn()

    def print_ket_time_evolved(self):
        self.ket_time_evolved.print_wfn()

    def print_bra_time_evolved(self):
        self.bra_time_evolved.print_wfn()

    def print_ket_generator(self):
        print(self.ket_generator)

    def print_bra_generator(self):
        if self.bra_instructions is not None:
            raise ValueError("No bra circuit provided")
        else:
            print(self.bra_generator)

    def extract_ket_variables(self):
        return self.non_fixed_variables_ket

    def extract_bra_variables(self):
        return self.non_fixed_variables_bra

    def extract_variables(self):
        ket_v = self.extract_ket_variables()

        bra_v=[]
        if self.bra_instructions is not None:
            bra_v = self.extract_bra_variables()
            ket_v.append(bra_v)
        return ket_v

    @property
    def constant_dict_ket(self):
        # print('Called,',self._constant_dict_ket)
        return  self._constant_dict_ket

    @constant_dict_ket.setter
    def constant_dict_ket(self, constant_dict_ket):
        self._constant_dict_ket = constant_dict_ket
        # print("Setted",self._constant_dict_ket)

    @property
    def U(self):
        return None

    def count_measurements(self) -> int:
        mes = 0
        if self.h_fqe is not None:
            mes = self.h_fqe.dim()
        return mes

    def __str__(self):
        res = ''
        # if self.is_diagonal:
        res += f" Ket with indices: {self.ket_instructions} with variables {self.ket_angles}"
        res += f" Bra with indices: {self.bra_instructions} with variables {self.bra_angles}"
        # else:
        #     res += f"{self._name} with Bra= {self.bra} with variables {self.params_bra}\n"
        #     res += f"{len(self._name)*' '} with Ket= {self.ket} with variables {self.params_ket}"
        return res

    def __repr__(self):
        return self.__str__()



    def grad(self, variable: Variable = None, *args, **kwargs):
        # print("============== Start grad")
        def apply_phase(braket: FQEBraKet, exct: List[Tuple[int]], variable, ket: bool = True, p0sign: bool = True) \
                -> Objective:
            # print("============= Start phase",ket,p0sign)
            s = {True: +1, False: -1}
            length=0
            ph = tq.grad(variable, variable) if not isinstance(variable, Variable) else 1.

            p0 = make_excitation_generator_op(exct, form="p0")
            for key in p0.terms:
                length += len(key)

            if ket:
                name = "p0sign_ket"
                c = deepcopy(braket.constant_dict_ket)
                if p0sign:
                    p0 = p0
                    c[variable] =  0.5*np.pi

                else:
                    p0 = -p0
                    c[variable] = -  np.pi


                index = braket.parameter_map_ket.index(variable)
                aux_dict = {}
                i = 0
                for stuff in braket.ket_generator:

                    aux_dict[stuff] = braket.ket_generator[stuff]
                    if i == index:
                        aux_dict["p0"] = p0
                    i+=1
                braket.ket_generator = aux_dict
                braket.constant_dict_ket = c
                braket.parameter_map_ket.insert(index+1, tq.Variable(name))
            else:
                name = "p0sign_bra"

                if p0sign:
                    braket.ket_generator[name] = p0
                else:
                    braket.ket_generator[name] = -p0

                braket.constant_dict_bra[variable] = - np.pi
                print(braket.parameter_map_bra)
                index = braket.parameter_map_bra.index(variable)
                braket.parameter_map_bra.insert(index + 1, tq.Variable(name))

            return  -1*s[ket]*s[(length//2)%2]*s[p0sign]*ph*Objective([braket])  # TODO: Check this correct



        if variable is None:
            # None means that all components are created
            variables = self.extract_variables()
            result = {}

            if len(variables) == 0:
                raise TequilaException("Error in gradient: Objective has no variables")

            for k in variables:
                assert k is not None
                result[k] = self.grad(k)
            return result
        else:
            variable = assign_variable(variable)

        if variable not in self.extract_variables():

            return 0.

        erw=self.ket_generator_idx_map[variable]
        # print("Variable",variable,'->',erw)
        g = 0
        for stuff in [erw]:
            if variable in self.parameter_map_ket:
                g += apply_phase(deepcopy(self), stuff, variable, ket=True, p0sign=True)
                g += apply_phase(deepcopy(self), stuff, variable, ket=True, p0sign=False)
        if self.bra_instructions is not None:
            if variable in self.parameter_map_bra:
                erw2 = self.bra_generator_idx_map[variable]
                for stuff in [erw2]:
                    g += apply_phase(deepcopy(self), stuff, variable, ket=False, p0sign=True)
                    g += apply_phase(deepcopy(self), stuff, variable, ket=False, p0sign=False)

        return 0.5*g


    def minimize(self):
        return None


def set_init_state( wfn: fqe.Wavefunction, n_ele, n_orb,
                    init_state: Union[List[Union[Tuple[str, int], QubitWaveFunction, np.array]],QubitWaveFunction],
                    bin_dict: dict) -> None:

    coeff = wfn.get_coeff((n_ele, 0))

    if isinstance(init_state, Tuple):
        for state in init_state:
            if len(state[0]) != n_ele:
                raise TequilaException("initial state is to long")
            n_ones = 0
            for binary in state[0]:
                if binary == "1":
                    n_ones += 1
            if n_ones != n_ele // 2:
                raise TequilaException("initial state has to many ones for the reordrerd JW")

        for state in init_state:
            i = bin_dict[state[0]]
            coeff[i][i] += state[1]
        wfn.set_wfn(strategy="from_data", raw_data={(n_ele, 0): coeff})
        wfn.normalize()

    elif isinstance(init_state, QubitWaveFunction):
        indices, values = init_state_from_wavefunction(wvf=init_state,n_orb=n_orb, bin_dict=bin_dict)
        for i,index in enumerate(indices):
            coeff[index][index] = values[i]

        wfn.set_wfn(strategy="from_data", raw_data={(n_ele, 0): coeff})
        wfn.normalize()

    elif isinstance(init_state, np.ndarray):
        wfn.set_wfn(strategy="from_data", raw_data={(n_ele, 0): init_state[0]})
        wfn.normalize()

    else:
        raise TequilaException("unkown intitial state type {}".format(type(init_state[0])))



def init_state_from_wavefunction(wvf:QubitWaveFunction, n_orb:int, bin_dict:dict):

    indices=[]
    values=[]
    for idx, i in enumerate(wvf._state):
        if abs(i) > 1e-3:
            vec = (bin(idx)[2:])
            if len(vec) < n_orb:
                vec = '0'*(n_orb-len(vec))+vec
            if len(vec) > n_orb:
                vec = vec[:n_orb]
            vec = vec[::-1]
            # vec = vec[len(vec)//2:]
            indices.append(bin_dict[vec])
            values.append(abs(i)) #todo not sure about this


    return indices, values





