"""Helpers to build fermionic Hamiltonians and excitation generators as OpenFermion operators."""

import typing
from collections import defaultdict
from itertools import combinations

import numpy
import openfermion

from sunrise.fermionic_operations.gates import *


def hcb_restricted_integrals(h, g):
    r"""Keep only the integrals a seniority-zero state sees.

    These are :math:`h_{ii}`, :math:`(ii|jj)`, :math:`(ij|ji)` and :math:`(ij|ij)`. On
    seniority-zero states the restricted Hamiltonian gives the same expectation values as the
    hard-core boson Hamiltonian, with 2-body terms only.

    Parameters
    ----------
    h : numpy.ndarray
        One-body integrals.
    g : numpy.ndarray
        Two-body integrals in chemists' (Mulliken) ordering.

    Returns
    -------
    h : numpy.ndarray
        Diagonal of ``h``.
    g : numpy.ndarray
        Restricted two-body integrals, same ordering.
    """
    i, j = numpy.indices(h.shape)
    g_hcb = numpy.zeros_like(g)
    g_hcb[i, i, j, j] = g[i, i, j, j]
    g_hcb[i, j, j, i] = g[i, j, j, i]
    g_hcb[i, j, i, j] = g[i, j, i, j]
    return numpy.diag(numpy.diag(h)), g_hcb


def make_fermionic_hamiltonian(
    one_body_integrals, two_body_integrals, constant, *args, **kwargs
):
    """Build the molecular Hamiltonian as an OpenFermion FermionOperator.

    Parameters
    ----------
    one_body_integrals : numpy.ndarray
        One-body integrals in the spatial-orbital basis.
    two_body_integrals : numpy.ndarray
        Two-body integrals in the spatial-orbital basis (OpenFermion ordering).
    constant : float
        Constant energy term (nuclear repulsion plus core energy).
    *args, **kwargs
        Ignored.

    Returns
    -------
    openfermion.FermionOperator
        Hamiltonian in spin orbital basis.
    """
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
    r"""Build the anti-Hermitian generator of a fermionic excitation.

    For pairs ``(i_k, a_k)`` the generator is
    :math:`-i\,a^\dagger_{a_0} a_{i_0} a^\dagger_{a_1} a_{i_1} \cdots + \text{h.c.}`,
    so that :math:`\exp(\theta G)` is a UCC-type unitary.

    Parameters
    ----------
    indices : iterable of tuple of int
        Pairs ``[(i_0, a_0), (i_1, a_1), ...]`` in spin orbital notation, each
        pair destroying ``i`` and creating ``a``. A flat list
        ``[i_0, a_0, i_1, a_1, ...]`` is also accepted.
    form : str, optional
        "fermionic" (default) returns the plain generator. Other values transform
        it into an involution or projector: "g+", "g-" (involutions), "p+", "p-",
        "p0" (projectors).

    Returns
    -------
    openfermion.FermionOperator
        Generator of the excitation. If it is a number operator (``i == a``), the
        number operator itself is returned.

    Raises
    ------
    TequilaException
        If no indices are given, the flat list has odd length or ``form`` is unknown.
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


def create_fermionic_generators(instructions, angles, form: str = "fermionic") -> dict:
    """Build the excitation generators for a list of excitations.

    Parameters
    ----------
    instructions : list of list of tuple of int
        Excitations, each a list of ``(i, a)`` pairs.
    angles : list
        Variable of each excitation.
    form : str, optional
        Form of the generators, see ``make_excitation_generator_op``.

    Returns
    -------
    dict
        Maps each variable of ``angles`` to the list of generators it multiplies.
    """
    generators = defaultdict(list)
    new_instructions = []

    for exct in instructions:
        aux = []
        for indicies in exct:
            aux.append(indicies[0])
            aux.append(indicies[1])
        aux = [tuple(aux)]
        new_instructions.append(aux)
    instructions = new_instructions

    for angle_idx, fermionic_circuit in enumerate(instructions):
        generators[angles[angle_idx]].append(
            make_excitation_generator_op(indices=fermionic_circuit[0], form=form)
        )

    return generators


def generate_of_binary_dict(n_orb: int, n_e: int) -> dict:
    """Map each occupation bitstring to a running index.

    Parameters
    ----------
    n_orb : int
        Length of the bitstrings.
    n_e : int
        Number of ones in each bitstring.

    Returns
    -------
    dict
        Keys are the (reversed) binary strings with ``n_e`` ones, values their
        index in order of generation.
    """
    result = {}
    for index, positions in enumerate(combinations(range(n_orb), n_e)):
        s = ["0"] * n_orb
        for pos in positions:
            s[pos] = "1"
        binary_str = "".join(s)

        result[binary_str[::-1]] = index
    return result
