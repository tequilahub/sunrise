from itertools import combinations
from .fermionic_utils import make_excitation_generator_op
from collections import defaultdict

def create_fermionic_generators(instructions, angles, form: str = 'fermionic') -> dict:


    generators=defaultdict(list)
    new_instructions = []

    for exct in instructions:
        aux =[]
        for indicies in exct:
            aux.append(indicies[0])
            aux.append(indicies[1])
        aux = [tuple(aux)]
        new_instructions.append(aux)
    instructions = new_instructions


    for angle_idx, fermionic_circuit in enumerate(instructions):
        generators[angles[angle_idx]].append(make_excitation_generator_op(indices=fermionic_circuit[0], form=form))

    return generators


def generate_of_binary_dict(n_orb: int, n_e: int) -> dict:

    """
    create a dictionary of all possibilities of a binary string of length n_orb with n_e number of 1s as a key and
    the integer value of the binary string as the value
    Parameters
    ----------
    n_orb: number of orbitals
    n_e: number of electrons

    Returns
    -------
    Dictionary with the key being the binary possibilities and the value being the integer value of the binary string
    """
    result = {}
    for index, positions in enumerate(combinations(range(n_orb), n_e)):
        s = ['0'] * n_orb
        for pos in positions:
            s[pos] = '1'
        binary_str = ''.join(s)

        result[binary_str[::-1]] = index
    return result


