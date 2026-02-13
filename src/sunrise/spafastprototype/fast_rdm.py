import tequila as tq
from spafastprototype.decompose import make_decomposed_circuits
import numpy
from spafastprototype._compute_rdms import compute_rdms, _assemble_rdm2, _assemble_rdm1


def fast_rdm(U, mol, clusters, variables, test=False,backend='qulacs'):
    """
    This function builds the 1-RDM and 2-RDM by exploiting the separability of 
    the circuit to speed up the procedure. 

    It decomposes each global RDM operator into local Pauli strings for each 
    circuit cluster, simulates these small local expectation values, 
    and reassembles them to reconstruct the full RDM element.
    To further optimize performance, evaluated Pauli strings are stored in a 
    dictionary to avoid redundant simulations.

    Each RDM element is the expectation value of an operator O:
    <O>_U = \sum_k c_k <P_k>_U
    where each global Pauli string is a tensor product of local Pauli strings:
    P_k = P_k,0 * P_k,1 * P_k,2 ...
    In the case where U is a tensor product: U = U_0 * U_1 * U_2 ... 
    Each RDM element factorizes: 
    <O>_U = \sum_k (c_k <P_k,0>_U0 * <P_k,1>_U1 * <P_k,2>_U2 ...) 
    """
    assert clusters is not None

    n_MOs = mol.n_orbitals
    circuits = make_decomposed_circuits(U, clusters=clusters)

    # needs hackend version that gives back the raw qops
    qops, asd = compute_rdms(self=mol, U=U, use_hcb=True, evaluate=False)

    all_ps = sum([op.paulistrings for op in qops], [])
    
    all_ps_evaluated = {}
    all_psi_evaluated = {}
    for ps in all_ps:
        assert numpy.isclose(ps.coeff.imag, 0.0)
        # all ps are evaluated and stored with ps.coeff=1.0 to ensure:
        # - dictionary key consistency
        # - identical ps with different ps.coeff are only simulated once
        ps_key = str(tq.PauliString(ps._data, coeff=1.0))
        ps_evaluated = 1.0

        if len(ps.qubits) == 0:
            all_ps_evaluated[ps_key] = float(ps_evaluated)
            continue

        for i, u in enumerate(circuits):
            psi = tq.PauliString({k: v for k, v in ps.items() if k in u.qubits}, coeff=1.0)
            if len(psi) == 0: 
                continue

            psi_key = (str(psi), i) 
            if psi_key not in all_psi_evaluated:
                Hi = tq.QubitHamiltonian.from_paulistrings(psi)
                psi_evaluated = tq.simulate(tq.ExpectationValue(H=Hi, U=u), variables=variables,backend=backend)
                all_psi_evaluated[psi_key] = psi_evaluated
            else:
                psi_evaluated = all_psi_evaluated[psi_key]
            ps_evaluated *= psi_evaluated

        all_ps_evaluated[ps_key] = float(ps_evaluated)

    evaluated = []
    if test: testus = tq.simulate(tq.ExpectationValue(H=qops, U=U, shape=[len(qops)]), variables=variables,backend=backend)

    for x,op in enumerate(qops):
        tmp = 0.0
        trace = {}
        for ps in op.paulistrings:
            ps_key = str(tq.PauliString(ps._data, coeff=1.0))
            tmp += float(ps.coeff.real)*all_ps_evaluated[ps_key]
            trace[str(ps)] = all_ps_evaluated[ps_key]
        evaluated.append(tmp)

        if test and not numpy.isclose(tmp, testus[x]):
            print(x, " " , op)
            print("{} vs {}".format(tmp, testus[x]))
            print(trace)

    len_1 = n_MOs * (n_MOs + 1) // 2
    evals = [float(x.real) for x in evaluated]
    evals_1, evals_2 = evals[:len_1], evals[len_1:]
    rdm1 = _assemble_rdm1(evals_1, n_MOs)
    rdm2 = _assemble_rdm2(evals_2, n_MOs)
    rdm2 = tq.quantumchemistry.NBodyTensor(elems=rdm2, ordering="dirac", verify=False)
    rdm2 = rdm2.reorder(to="dirac").elems

    return rdm1, rdm2


if __name__ == "__main__":

    for n_atoms in [8, 12, 16, 20]:
        edges = [(2 * i, 2 * i + 1) for i in range(n_atoms // 2)]
        geometry = ''
        for i in range(n_atoms):
            geometry += f'H 0. 0. {i}.\n'

        mol = tq.Molecule(geometry=geometry, basis_set='sto-3g', backend='pyscf', transformation="ReorderedJordanWigner").use_native_orbitals()
        clusters = [[i for i in range(n_atoms // 2)], [i for i in range(n_atoms // 2, n_atoms)]]
        if n_atoms == 16: clusters = [[0, 1, 2, 3],[ 4, 5, 6, 7],[8,9,10,11], [12,13,14,15]]

        U = mol.make_ansatz(name="HCB-SPA", edges=edges)
        variables = {k: 1.0 for k in U.extract_variables()}
        rdm1, rdm2 = mol.compute_rdms(U=U, use_hcb=True, variables=variables)
        xrdm1, xrdm2 = fast_rdm(U=U, clusters=clusters, mol=mol, variables=variables)
        print(numpy.linalg.norm(rdm1-xrdm1))
        print(numpy.linalg.norm(rdm2-xrdm2))




