from tequila import TequilaException, TequilaWarning
from tequila.hamiltonian import QubitHamiltonian
from tequila.hamiltonian.paulis import Sp, Sm, Zero
from tequila.objective.objective import ExpectationValue
from tequila.simulators.simulator_api import simulate
from tequila.quantumchemistry import NBodyTensor
import warnings
import numpy
from itertools import product
from numbers import Number

def compute_rdms(self, U = None, variables = None, spin_free: bool = True,
                 get_rdm1: bool = True, get_rdm2: bool = True, ordering="dirac", use_hcb: bool = False,
                 rdm_trafo = None, evaluate=True):
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
    # Set up number of spin-orbitals and molecular orbitals respectively
    n_SOs = 2 * self.n_orbitals
    n_MOs = self.n_orbitals

    # Check whether unitary circuit is not 0
    if U is None:
        raise TequilaException('Need to specify a Quantum Circuit.')

    def _get_hcb_op(op_tuple):
        '''Build the hardcore boson operators: b^\dagger_ib_j + h.c. in qubit encoding '''
        if (len(op_tuple) == 2):
            return 2 * Sm(op_tuple[0][0]) * Sp(op_tuple[1][0])
        elif (len(op_tuple) == 4):
            if ((op_tuple[0][0] == op_tuple[1][0]) and (op_tuple[2][0] == op_tuple[3][0])):  # iijj uddu+duud
                return Sm(op_tuple[0][0]) * Sp(op_tuple[2][0]) + Sm(op_tuple[2][0]) * Sp(op_tuple[0][0])
            if ((op_tuple[0][0] == op_tuple[2][0]) and (op_tuple[1][0] == op_tuple[3][0]) and (
                    op_tuple[0][0] != op_tuple[1][0]) and (op_tuple[2][0] != op_tuple[3][0])):  # ijij uuuu+dddd
                return 4 * Sm(op_tuple[0][0]) * Sm(op_tuple[1][0]) * Sp(op_tuple[2][0]) * Sp(op_tuple[3][0])
            if ((op_tuple[0][0] == op_tuple[3][0]) and (op_tuple[1][0] == op_tuple[2][0]) and (
                    op_tuple[0][0] != op_tuple[1][0]) and (op_tuple[2][0] != op_tuple[3][0])):  # ijji abba
                return -2 * Sm(op_tuple[0][0]) * Sm(op_tuple[1][0]) * Sp(op_tuple[2][0]) * Sp(op_tuple[3][0])
        else:
            return Zero()

    def _get_of_op(operator_tuple):
        """ Returns operator given by a operator tuple as OpenFermion - Fermion operator """
        import openfermion
        op = openfermion.FermionOperator(operator_tuple)
        return op

    def _get_qop_hermitian(of_operator) -> QubitHamiltonian:
        """ Returns Hermitian part of Fermion operator as QubitHamiltonian """
        qop = self.transformation(of_operator)
        # qop = QubitHamiltonian(self.transformation(of_operator))
        real, imag = qop.split(hermitian=True)
        if real:
            return real
        elif not real:
            raise TequilaException(
                "Qubit Hamiltonian does not have a Hermitian part. Operator ={}".format(of_operator))

    def _build_1bdy_operators_spinful() -> list:
        """ Returns spinful one-body operators as a symmetry-reduced list of QubitHamiltonians """
        # Exploit symmetry pq = qp
        ops = []
        for p in range(n_SOs):
            for q in range(p + 1):
                op_tuple = ((p, 1), (q, 0))
                op = _get_of_op(op_tuple)
                ops += [op]

        return ops

    def _build_2bdy_operators_spinful() -> list:
        """ Returns spinful two-body operators as a symmetry-reduced list of QubitHamiltonians """
        # Exploit symmetries pqrs = -pqsr = -qprs = qpsr
        #                and      =  rspq
        ops = []
        for p in range(n_SOs):
            for q in range(p):
                for r in range(n_SOs):
                    for s in range(r):
                        if p * n_SOs + q >= r * n_SOs + s:
                            op_tuple = ((p, 1), (q, 1), (s, 0), (r, 0))
                            op = _get_of_op(op_tuple)
                            ops += [op]

        return ops

    def _build_1bdy_operators_spinfree() -> list:
        """ Returns spinfree one-body operators as a symmetry-reduced list of QubitHamiltonians """
        # Exploit symmetry pq = qp (not changed by spin-summation)
        ops = []
        for p in range(n_MOs):
            for q in range(p + 1):
                # Spin aa
                op_tuple = ((2 * p, 1), (2 * q, 0))
                op = _get_of_op(op_tuple)
                # Spin bb
                op_tuple = ((2 * p + 1, 1), (2 * q + 1, 0))
                op += _get_of_op(op_tuple)
                ops += [op]

        return ops

    def _build_2bdy_operators_spinfree() -> list:
        """ Returns spinfree two-body operators as a symmetry-reduced list of QubitHamiltonians """
        # Exploit symmetries pqrs = qpsr (due to spin summation, '-pqsr = -qprs' drops out)
        #                and      = rspq
        ops = []
        for p, q, r, s in product(range(n_MOs), repeat=4):
            if p * n_MOs + q >= r * n_MOs + s and (p >= q or r >= s):
                # Spin aaaa
                op_tuple = ((2 * p, 1), (2 * q, 1), (2 * s, 0), (2 * r, 0)) if (p != q and r != s) else '0.0 []'
                op = _get_of_op(op_tuple)
                # Spin abab
                op_tuple = ((2 * p, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r, 0)) if (
                        2 * p != 2 * q + 1 and 2 * r != 2 * s + 1) else '0.0 []'
                op += _get_of_op(op_tuple)
                # Spin baba
                op_tuple = ((2 * p + 1, 1), (2 * q, 1), (2 * s, 0), (2 * r + 1, 0)) if (
                        2 * p + 1 != 2 * q and 2 * r + 1 != 2 * s) else '0.0 []'
                op += _get_of_op(op_tuple)
                # Spin bbbb
                op_tuple = ((2 * p + 1, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r + 1, 0)) if (
                        p != q and r != s) else '0.0 []'
                op += _get_of_op(op_tuple)
                ops += [op]
        return ops

    def _assemble_rdm1(evals, rdm1=None) -> numpy.ndarray:
        """
        Returns spin-ful or spin-free one-particle RDM built by symmetry conditions
        Same symmetry with or without spin, so we can use the same function
        """
        N = n_MOs if spin_free else n_SOs
        if rdm1 is None:
            rdm1 = numpy.zeros([N, N])
        ctr: int = 0
        for p in range(N):
            for q in range(p + 1):
                rdm1[p, q] = evals[ctr]
                # Symmetry pq = qp
                rdm1[q, p] = rdm1[p, q]
                ctr += 1

        return rdm1

    def _assemble_rdm2_spinful(evals, rdm2=None) -> numpy.ndarray:
        """ Returns spin-ful two-particle RDM built by symmetry conditions """
        ctr: int = 0
        if rdm2 is None:
            rdm2 = numpy.zeros([n_SOs, n_SOs, n_SOs, n_SOs])
        for p in range(n_SOs):
            for q in range(p):
                for r in range(n_SOs):
                    for s in range(r):
                        if p * n_SOs + q >= r * n_SOs + s:
                            rdm2[p, q, r, s] = evals[ctr]
                            # Symmetry pqrs = rspq
                            rdm2[r, s, p, q] = rdm2[p, q, r, s]
                            ctr += 1

        # Further permutational symmetries due to anticommutation relations
        for p in range(n_SOs):
            for q in range(p):
                for r in range(n_SOs):
                    for s in range(r):
                        rdm2[p, q, s, r] = -1 * rdm2[p, q, r, s]  # pqrs = -pqsr
                        rdm2[q, p, r, s] = -1 * rdm2[p, q, r, s]  # pqrs = -qprs
                        rdm2[q, p, s, r] = rdm2[p, q, r, s]  # pqrs =  qpsr

        return rdm2

    def _assemble_rdm2_spinfree(evals, rdm2=None) -> numpy.ndarray:
        """ Returns spin-free two-particle RDM built by symmetry conditions """
        ctr: int = 0
        if rdm2 is None:
            rdm2 = numpy.zeros([n_MOs, n_MOs, n_MOs, n_MOs])
        for p, q, r, s in product(range(n_MOs), repeat=4):
            if p * n_MOs + q >= r * n_MOs + s and (p >= q or r >= s):
                rdm2[p, q, r, s] = evals[ctr]
                # Symmetry pqrs = rspq
                rdm2[r, s, p, q] = rdm2[p, q, r, s]
                ctr += 1

        # Further permutational symmetry: pqrs = qpsr
        for p, q, r, s in product(range(n_MOs), repeat=4):
            if p >= q or r >= s:
                rdm2[q, p, s, r] = rdm2[p, q, r, s]

        return rdm2

    def _build_1bdy_operators_hcb() -> list:
        """ Returns hcb one-body operators as a symmetry-reduced list of QubitHamiltonians """
        # Exploit symmetry pq = qp (not changed by spin-summation)
        ops = []
        for p in range(n_MOs):
            for q in range(p + 1):
                if (p == q):
                    if (self.transformation.up_then_down):
                        op_tuple = ((p, 1), (p, 0))
                        op = _get_hcb_op(op_tuple)
                    else:
                        op_tuple = ((2 * p, 1), (2 * p, 0))
                        op = _get_hcb_op(op_tuple)
                    ops += [op]
                else:
                    ops += [Zero()]
        return ops

    def _build_2bdy_operators_hcb() -> list:
        """ Returns hcb two-body operators as a symmetry-reduced list of QubitHamiltonians """
        # Exploit symmetries pqrs = qpsr (due to spin summation, '-pqsr = -qprs' drops out)
        #                and      = rspq
        ops = []
        scale = 2
        if self.transformation.up_then_down:
            scale = 1
        for p, q, r, s in product(range(n_MOs), repeat=4):
            if p * n_MOs + q >= r * n_MOs + s and (p >= q or r >= s):
                # Spin abba+ baab allow p=q=r=s orb iijj
                op_tuple = ((scale * p, 1), (scale * q, 1), (scale * r, 0), (scale * s, 0)) if (
                        p == q and s == r) else '0.0 []'
                op = _get_hcb_op(op_tuple)
                # Spin abba+ baab dont allow p=q=r=s orb ijij
                op_tuple = ((scale * p, 1), (scale * q, 1), (scale * r, 0), (scale * s, 0)) if (
                        p != q and r != s and p == r and s == q) else '0.0 []'
                op += _get_hcb_op(op_tuple)
                # Spin aaaa+ bbbb dont allow p=q=r=s  orb ijji
                op_tuple = ((scale * p, 1), (scale * q, 1), (scale * r, 0), (scale * s, 0)) if (
                        p != q and r != s and p == s and q == r) else '0.0 []'
                op += _get_hcb_op(op_tuple)
                ops += [op]
        return ops

    # Build operator lists
    qops = []
    if spin_free and not use_hcb:
        qops += _build_1bdy_operators_spinfree() if get_rdm1 else []
        qops += _build_2bdy_operators_spinfree() if get_rdm2 else []
    elif use_hcb:
        qops += _build_1bdy_operators_hcb() if get_rdm1 else []
        qops += _build_2bdy_operators_hcb() if get_rdm2 else []
    else:
        if use_hcb:
            raise TequilaException(
                "compute_rdms: spin_free={} and use_hcb={} are not compatible".format(spin_free, use_hcb))
        qops += _build_1bdy_operators_spinful() if get_rdm1 else []
        qops += _build_2bdy_operators_spinful() if get_rdm2 else []

    # Transform operator lists to QubitHamiltonians
    if (not use_hcb):
        qops = [_get_qop_hermitian(op) for op in qops]

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
            return qops, None
            evals = [ExpectationValue(H=x, U=U) for x in qops]
            N = n_MOs if spin_free else n_SOs
            rdm1 = QTensor(shape=[N, N])
            rdm2 = QTensor(shape=[N, N, N, N])
        else:
            raise TequilaException("compute_rdms: rdm_trafo was set but evaluate flag is False (not supported)")

    # Assemble density matrices
    # If self._rdm1, self._rdm2 exist, reset them if they are of the other spin-type
    def _reset_rdm(rdm):
        if rdm is not None:
            if (spin_free or use_hcb) and rdm.shape[0] != n_MOs:
                return None
            if not spin_free and rdm.shape[0] != n_SOs:
                return None
        return rdm

    self._rdm1 = _reset_rdm(self._rdm1)
    self._rdm2 = _reset_rdm(self._rdm2)
    # Split expectation values in 1- and 2-particle expectation values
    if get_rdm1:
        len_1 = n_MOs * (n_MOs + 1) // 2 if (spin_free or use_hcb) else n_SOs * (n_SOs + 1) // 2
    else:
        len_1 = 0
    evals_1, evals_2 = evals[:len_1], evals[len_1:]
    # Build matrices using the expectation values
    self._rdm1 = _assemble_rdm1(evals_1, rdm1=rdm1) if get_rdm1 else self._rdm1
    if spin_free or use_hcb:
        self._rdm2 = _assemble_rdm2_spinfree(evals_2, rdm2=rdm2) if get_rdm2 else self._rdm2
    else:
        self._rdm2 = _assemble_rdm2_spinful(evals_2, rdm2=rdm2) if get_rdm2 else self._rdm2

    if get_rdm2:
        rdm2 = NBodyTensor(elems=self.rdm2, ordering="dirac", verify=False)
        rdm2.reorder(to=ordering)
        rdm2 = rdm2.elems
        self._rdm2 = rdm2

    if get_rdm1:
        if get_rdm2:
            return self.rdm1, self.rdm2
        else:
            return self.rdm1
    elif get_rdm2:
        return self.rdm2
    else:
        warnings.warn("compute_rdms called with instruction to not compute?", TequilaWarning)


def _assemble_rdm2(evals, n_MOs, rdm2=None) -> numpy.ndarray:
    """ Returns spin-free two-particle RDM built by symmetry conditions """
    ctr: int = 0
    if rdm2 is None:
        rdm2 = numpy.zeros([n_MOs, n_MOs, n_MOs, n_MOs])
    for p, q, r, s in product(range(n_MOs), repeat=4):
        if p * n_MOs + q >= r * n_MOs + s and (p >= q or r >= s):
            if isinstance(evals[ctr],Number) and numpy.isclose(evals[ctr],0):
                ctr += 1
                continue
            rdm2[p, q, r, s] = evals[ctr]
            # Symmetry pqrs = rspq
            rdm2[r, s, p, q] = rdm2[p, q, r, s]
            ctr += 1

    # Further permutational symmetry: pqrs = qpsr
    for p, q, r, s in product(range(n_MOs), repeat=4):
        if p >= q or r >= s:
            rdm2[q, p, s, r] = rdm2[p, q, r, s]

    return rdm2


def _assemble_rdm1(evals, n_MOs,rdm1=None):
    """
    Returns spin-ful or spin-free one-particle RDM built by symmetry conditions
    Same symmetry with or without spin, so we can use the same function
    """
    N = n_MOs
    if rdm1 is None:
        rdm1 = numpy.zeros([N, N])
    ctr: int = 0
    for p in range(N):
        for q in range(p + 1):
            if isinstance(evals[ctr],Number) and numpy.isclose(evals[ctr],0):
                ctr += 1
                continue
            rdm1[p, q] = evals[ctr]
            # Symmetry pq = qp
            rdm1[q, p] = rdm1[p, q]
            ctr += 1

    return rdm1
