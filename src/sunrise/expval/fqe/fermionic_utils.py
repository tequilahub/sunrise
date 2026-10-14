import openfermion
from sympy.codegen.rewriting import powm1_opt
from tequila import TequilaException, QCircuit, Objective
from sunrise.molecules.qubit_base import Molecule
import typing
import numpy as np
from sunrise.fermionic_operations.gates import *

def make_fermionic_hamiltonian(one_body_integrals, two_body_integrals, constant, *args, **kwargs):

    one_body_coefficients, two_body_coefficients = openfermion.chem.molecular_data.spinorb_from_spatial(
        one_body_integrals, two_body_integrals)

    molecular_hamiltonian = openfermion.ops.representations.InteractionOperator(
        constant, one_body_coefficients, 1 / 2 * two_body_coefficients)
    fop = openfermion.transforms.get_fermion_operator(molecular_hamiltonian)

    return fop


def make_excitation_generator_op(indices: typing.Iterable[typing.Tuple[int, int]], form: str = 'fermionic')-> openfermion.FermionOperator:
    r"""
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
    Returns
    -------
    type
        1j*Transformed qubit excitation operator, depends on self.transformation
    """


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
        dag += [(int(pair[0]), 0), (int(pair[1]), 1)]
        if pair[0] == pair[1]:
            number_op = True
        else: 
            number_op = False
    op = openfermion.FermionOperator(tuple(ofi), -1.j)  # 1j makes it hermitian
    op += openfermion.FermionOperator(tuple(reversed(dag)), 1.j)

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
            op += openfermion.FermionOperator(Na + Mi, 0.5)
            op += openfermion.FermionOperator(Ni + Ma, 0.5)
        elif form.lower() == "p-":
            op *= 0.5
            op += openfermion.FermionOperator(Na + Mi, -0.5)
            op += openfermion.FermionOperator(Ni + Ma, -0.5)

        elif form.lower() == "g+":
            op += openfermion.FermionOperator([], 1.0)  # Just for clarity will be subtracted anyway
            op += openfermion.FermionOperator(Na + Mi, -1.0)
            op += openfermion.FermionOperator(Ni + Ma, -1.0)
        elif form.lower() == "g-":
            op += openfermion.FermionOperator([], -1.0)  # Just for clarity will be subtracted anyway
            op += openfermion.FermionOperator(Na + Mi, 1.0)
            op += openfermion.FermionOperator(Ni + Ma, 1.0)
        elif form.lower() == "p0":
            # P0: we only construct P0 and don't keep the original generator
            # op = openfermion.FermionOperator([], 1.0)  # Just for clarity will be subtracted anyway <- biggest lie i have ever heard it is useless and destroys everything why the fuck is this there
            op = openfermion.FermionOperator(Na + Mi, -1.0)
            op += openfermion.FermionOperator(Ni + Ma, -1.0)
        else:
            raise TequilaException(
                "Unknown generator form {}, supported are G, P+, P-, G+, G- and P0".format(form))
    
    if number_op:
        op = openfermion.FermionOperator(tuple(ofi), 1)
    return op

