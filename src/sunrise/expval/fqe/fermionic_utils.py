import typing

import openfermion

from sunrise.fermionic_operations.gates import *


def make_fermionic_hamiltonian(
    one_body_integrals, two_body_integrals, constant, *args, **kwargs
):

    one_body_coefficients, two_body_coefficients = (
        openfermion.chem.molecular_data.spinorb_from_spatial(
            one_body_integrals, two_body_integrals
        )
    )

    molecular_hamiltonian = openfermion.ops.representations.InteractionOperator(
        constant, one_body_coefficients, 1 / 2 * two_body_coefficients
    )
    fop = openfermion.transforms.get_fermion_operator(molecular_hamiltonian)

    return fop


def make_excitation_generator_op(
    indices: typing.Iterable[tuple[int, int]], form: str = "fermionic"
) -> openfermion.FermionOperator:
    r"""
    Notes
    ----------
    Creates the transformed hermitian generator of UCC type unitaries:
          M(a^\dagger_{a_0} a_{i_0} a^\dagger{a_1}a_{i_1} ... - h.c.)
          where the qubit map M depends is self.transformation

    Parameters
    ----------
    indices : typing.Iterable[typing.Tuple[int, int]] :
        List of tuples [(i_0, a_0), (i_1, a_1), ... ], each pair destroys i and creates a - recommended format, in spin-orbital notation (alpha odd numbers, beta even numbers)
        can also be given as one big list: [i_0, a_0, i_1, a_1 ...]
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
            raise TequilaException(
                "make_excitation_generator: unexpected input format of indices\n"
                "use list of tuples as [(i_0, a_0),(i_1, a_1) ...]\n"
                "or list as [i_0, a_0, i_1, a_1, ... ]\n"
                f"you gave: {indices}"
            )
        converted = [
            (indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)
        ]
    else:
        converted = indices

    # convert everything to native python int
    # otherwise openfermion will complain
    converted = [(int(pair[0]), int(pair[1])) for pair in converted]
    # (p, q) destroys p and creates q: the operator below is a^dagger_q a_p
    converted = [(q, p) for p, q in converted]

    # convert to openfermion input format
    ofi = []
    dag = []
    for pair in converted:
        assert len(pair) == 2
        ofi += [
            (int(pair[0]), 1),
            (int(pair[1]), 0),
        ]  # openfermion does not take other types of integers like numpy.int64
        dag += [(int(pair[0]), 0), (int(pair[1]), 1)]
        if pair[0] == pair[1]:
            number_op = True
        else:
            number_op = False
    op = openfermion.FermionOperator(tuple(ofi), -1.0j)  # 1j makes it hermitian
    op += openfermion.FermionOperator(tuple(reversed(dag)), 1.0j)

    if isinstance(form, str) and form.lower() != "fermionic":
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
            op += openfermion.FermionOperator(
                [], 1.0
            )  # Just for clarity will be subtracted anyway
            op += openfermion.FermionOperator(Na + Mi, -1.0)
            op += openfermion.FermionOperator(Ni + Ma, -1.0)
        elif form.lower() == "g-":
            op += openfermion.FermionOperator(
                [], -1.0
            )  # Just for clarity will be subtracted anyway
            op += openfermion.FermionOperator(Na + Mi, 1.0)
            op += openfermion.FermionOperator(Ni + Ma, 1.0)
        elif form.lower() == "p0":
            # P0: we only construct P0 and don't keep the original generator
            # op = openfermion.FermionOperator([], 1.0)  # Just for clarity will be subtracted anyway <- biggest lie i have ever heard it is useless and destroys everything why the fuck is this there
            op = openfermion.FermionOperator(Na + Mi, -1.0)
            op += openfermion.FermionOperator(Ni + Ma, -1.0)
        else:
            raise TequilaException(
                f"Unknown generator form {form}, supported are G, P+, P-, G+, G- and P0"
            )

    if number_op:
        op = openfermion.FermionOperator(tuple(ofi), 1)
    return op
