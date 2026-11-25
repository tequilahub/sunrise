"""Base class for the explicitly correlated (F12) energy correction of the spin-free [2]_R12 method.

The use of [2]_R12 on top of VQE is described in P. Schleich, J. S. Kottmann and A. Aspuru-Guzik,
Phys. Chem. Chem. Phys. 24, 13550 (2022), arXiv:2110.06812.
"""

from __future__ import annotations

import numpy
from tequila import TequilaException

from ..chemistry_tools import NBodyTensor


class ExplicitCorrelationCorrection:
    """Explicitly correlated (F12) correction to an energy, using only the 1- and 2-RDM.

    Implements the spin-free [2]_R12 correction of Kong and Valeev
    (J. Chem. Phys. 135, 214105 (2011)), applied to VQE references as in Schleich,
    Kottmann and Aspuru-Guzik, Phys. Chem. Chem. Phys. 24, 13550 (2022). It estimates the
    part of the correlation energy missing from a finite orbital basis (OBS) through
    excitations into a complementary auxiliary basis (CABS), with the Slater-type
    correlation factor f_12 = -1/gamma * exp(-gamma * r_12). Only the 1- and 2-RDM of the
    reference state are needed, so no additional quantum resources are used. The integrals
    are provided by subclasses or from external files.

    Index notation (Latin letters for the Greek indices of the papers):

    - p,q,r,s,t,u,v,w,x,y: active space, equivalent to the orbital basis set (OBS).
      x,y denote the orbitals correlated by the geminals (geminal-generating orbitals). The method
      allows correlating only a subset of them; this implementation correlates all active orbitals.
    - k,l,m,n: full space, the resolution-of-identity (RI) approximation of the complete basis
      set (CBS), built as the union of OBS and CABS.
    - a,b,c,d: CABS (complementary auxiliary basis set) space.

    Tensors are written e.g. O^kl_rs = <rs|O|kl>: upper indices refer to the ket, lower indices
    to the bra. In variable names the upper indices are written in capitals, e.g. ``r_PQrs``.

    Parameters
    ----------
    mol : QuantumChemistryBase
        The molecule.
    rdm1 : numpy.ndarray, optional
        One-electron reduced density matrix.
    rdm2 : numpy.ndarray, optional
        Two-electron reduced density matrix (spin-free), in the physics (Dirac) ordering as
        returned by ``compute_rdms``.
    gamma : float, optional
        Exponent of the correlation factor. Default is 1.4.
    n_ri : int, optional
        Dimension of the resolution-of-identity (RI) basis. Use it only to truncate the
        available RI basis; by default the largest available one is used. It must
        not exceed the available RI basis and must be larger than the OBS,
        otherwise the correction is zero.
    external_info : dict, optional
        Where to find the f12 tensor <rs|f_12|pq>, as
        ``{"f12_filename": path to a .npy file, "ordering": ordering of the tensor}``.
    **kwargs
        Needed if ``rdm1`` and ``rdm2`` are not given: ``U`` (tq.QCircuit) and
        optionally ``variables`` (optimal angles), used to compute the RDMs, or
        ``rdm__psi4_method`` to compute them later with Psi4. ``silent`` (bool)
        suppresses printing of the intermediates.

    Raises
    ------
    TequilaException
        If neither the RDMs nor a way to compute them is given.
    """

    def __init__(
        self,
        mol=None,
        rdm1: numpy.ndarray | None = None,
        rdm2: numpy.ndarray | None = None,
        gamma: float = 1.4,
        n_ri: int | None = None,
        external_info: dict | None = None,
        **kwargs,
    ):
        """See the class docstring for the parameters."""
        # Associated molecule
        self.mol = mol

        # F12-exponent
        self.gamma = gamma

        # Check if rdm"s are provided (only check for both, do not care for providing only one)
        if (rdm1 is not None) and (rdm2 is not None):
            self.rdm1, self.rdm2 = rdm1, rdm2
        # If not provided, need to specify a circuit "U" (if parametrized, including a set of "variables")
        else:
            if "U" in kwargs:
                if "variables" in kwargs:
                    self.mol.compute_rdms(U=kwargs["U"], variables=kwargs["variables"])
                else:
                    self.mol.compute_rdms(U=kwargs["U"])
                self.rdm1, self.rdm2 = self.mol.rdm1, self.mol.rdm2
            # Check whether a psi4_method for rdms is specified, to be computed later
            elif "rdm__psi4_method" in kwargs:
                self.rdm1, self.rdm2 = None, None
            else:
                raise TequilaException(
                    "Need to either specify rdm's or provide a way how to compute them."
                )

        # Set dimensionalities and active space if applicable
        self.n_ri = n_ri
        if self.mol.active_space:
            # (does not need to correspond to a chemical active space)
            self.active = self.mol.active_space.active_orbitals
            self.n_obs = len(self.active)  # this might get overwritten in child classes
        else:
            self.active = None
            self.n_obs = -1

        # External info to read data from external files
        self.external_info = external_info

        # Set silent if in kwargs
        self.silent = False
        if "silent" in kwargs:
            self.silent = kwargs["silent"]

        # Amplitudes to be defined later
        self.t_PQrs = None

    def _sp_ansatz(self) -> numpy.ndarray:
        r"""Return the geminal amplitudes t_PQrs of the Hylleraas functional.

        There are two options to set the amplitudes:

        1. Minimize the functional, :math:`t^* = \operatorname{argmin}_t H^{(2)}`. The resulting
           linear system is often ill-conditioned.
        2. Fix them such that the singlet and triplet electron-electron cusp conditions are
           fulfilled (SP ansatz of S. Ten-no, Explicitly correlated second order perturbation
           theory: Introduction of a rational generator and numerical quadratures, J. Chem. Phys.
           121, 117-129 (2004)):
           :math:`t^{rs}_{pq} = \frac{3}{8} \delta^r_p \delta^s_q + \frac{1}{8} \delta^r_q \delta^s_p`.

        Option 2 is used here, as it is computationally far less expensive.

        Returns
        -------
        numpy.ndarray
            Amplitudes of shape (n_active,) * 4.
        """
        kron_d = numpy.eye(len(self.active))
        t_PQrs = 3 / 8 * numpy.einsum(
            "pr,qs -> pqrs", kron_d, kron_d, optimize="greedy"
        ) + 1 / 8 * numpy.einsum("pr,qs -> qprs", kron_d, kron_d, optimize="greedy")
        return t_PQrs

    def build_fock_operator(self, h: NBodyTensor, g: NBodyTensor) -> NBodyTensor:
        r"""Build the generalized (spin-free) Fock operator.

        .. math::
            f^k_l = h^k_l + \Gamma^s_r ( g^{kr}_{ls} - \tfrac{1}{2} g^{kr}_{sl} )

        with k, l in OBS and CABS, and Gamma the one-electron RDM.

        Parameters
        ----------
        h : NBodyTensor
            One-body (core Hamiltonian) tensor.
        g : NBodyTensor
            Two-body (Coulomb) tensor in physicist ordering.

        Returns
        -------
        NBodyTensor
            Generalized Fock operator.
        """
        g_1 = numpy.einsum(
            "sr, krls -> kl", self.rdm1, g.sub_str("fafa"), optimize="greedy"
        )
        g_2 = numpy.einsum(
            "sr, krsl -> kl",
            self.rdm1,
            numpy.einsum("krls->krsl", g.sub_str("fafa"), optimize="greedy"),
            optimize="greedy",
        )
        fock = NBodyTensor(
            elems=h.sub_str("ff") + g_1 - 1 / 2 * g_2,
            active_indices=self.active,
            size_full=self.n_ri,
        )

        return fock

    def size_check(self, n_ri_max: int):
        """Check that ``n_ri`` is valid.

        Parameters
        ----------
        n_ri_max : int
            Size of the available RI basis.

        Raises
        ------
        TequilaException
            If the OBS is not smaller than ``n_ri`` (the correction would be zero) or
            if ``n_ri`` exceeds ``n_ri_max``.
        """
        # If dimensionality of OBS is same as dimensionality of (formally-)CBS (or its approx),
        # then correction does not make sense and returns 0
        # intermediates V, B, X would be all 0
        if self.n_obs >= self.n_ri:
            raise TequilaException("Correction ineffective. Returns 0.")
        # n_ri needs to be smaller than dim(C_CBS)
        if self.n_ri > n_ri_max:
            raise TequilaException("Dimensionality of CBS too large.")

    def setup_tensors_external(self):
        """Set up the tensors from integrals provided externally.

        The f12 integrals r_PQrs = <rs|f_12|pq> are loaded from the ``.npy`` file in
        ``external_info`` and must already include the factor f_12 = -1/gamma * exp(-gamma*r_12).
        The other integrals are taken from the molecule.

        Returns
        -------
        tuple of NBodyTensor
            One-body tensor ``h``, Coulomb tensor ``g`` and f12 tensor ``r``.

        Raises
        ------
        TequilaException
            If the molecule has no active space or ``external_info`` is incomplete.
        """

        if not self.active:
            raise TequilaException(
                "Need to use an active space ~ subset of integrals for OBS"
                " in order to compute correction using external integrals."
            )

        # Load f12-tensor
        if ("f12_filename" not in self.external_info) and (
            "ordering" not in self.external_info
        ):
            raise TequilaException(
                "Need to specify information where to find f12-integrals and ordering scheme."
            )
        r_elems = numpy.load(self.external_info["f12_filename"])
        # Get n_ri via size of tensors, if has not been specified
        n_ri_max = r_elems.shape[0]
        if self.n_ri is None:
            self.n_ri = n_ri_max
        self.size_check(n_ri_max=n_ri_max)
        # This assumes a f12-operator of the kind -1/gamma*exp(-gamma*r12), adjust your integrals if necessary
        r = NBodyTensor(
            elems=r_elems,
            ordering=self.external_info["ordering"],
            active_indices=self.active,
            size_full=self.n_ri,
        )
        r.reorder(to="phys")

        # # Load coulomb-tensor elements
        # if "two_body_ordering" in self.mol.kwargs:
        #     ordering = self.mol.kwargs["two_body_ordering"]
        # else:
        #     ordering = "openfermion"
        # g = NBodyTensor(elems=self.mol.molecule.two_body_integrals, ordering=ordering,
        #                 active_indices=self.active, size_full=self.n_ri)
        _c, h, g = self.mol.integral_manager.get_integrals(ignore_active_space=True)
        g = NBodyTensor(
            elems=g.elems,
            ordering=g.ordering,
            active_indices=self.active,
            size_full=self.n_ri,
        )
        g.reorder(to="phys")

        # Load one-body tensor
        h = NBodyTensor(elems=h, active_indices=self.active, size_full=self.n_ri)

        return h, g, r

    def _compute_intermediate_V(self, g: NBodyTensor, r: NBodyTensor):
        """Compute intermediate V, the mixed geminal-reference block."""
        rdm1, rdm2 = self.rdm1, self.rdm2
        gKLxy_rRSkl = numpy.einsum(
            "klxy, rskl -> rsxy",
            g.sub_str("ffaa"),
            r.sub_str("aaff"),
            optimize="greedy",
        )
        gTUxy_rRStu = numpy.einsum(
            "tuxy, rstu -> rsxy",
            g.sub_str("aaaa"),
            r.sub_str("aaaa"),
            optimize="greedy",
        )

        gATxy_rdm1Ut_rRSau = numpy.einsum(
            "atxy,ut,rsau -> rsxy",
            g.sub_str("paaa"),
            rdm1,
            r.sub_str("aapa"),
            optimize="greedy",
        )
        V_mid = gKLxy_rRSkl - gTUxy_rRStu - gATxy_rdm1Ut_rRSau

        V = numpy.einsum(
            "pqrs, xypq, rsxy", self.t_PQrs, rdm2, V_mid, optimize="greedy"
        )

        return V

    def _compute_intermediate_B(self, r: NBodyTensor, fock: NBodyTensor):
        """Compute intermediate B, the geminal-geminal block with the Fock operator."""
        rdm1, rdm2 = self.rdm1, self.rdm2
        rZYpq_fockXy_rTUzx = numpy.einsum(
            "zypq, xy, tuzx -> tupq",
            r.sub_str("aaaa"),
            fock.sub_str("aa"),
            r.sub_str("aaaa"),
            optimize="greedy",
        )
        rAYpq_fockXa_rTUxy = numpy.einsum(
            "aypq, xa, tuxy -> tupq",
            r.sub_str("paaa"),
            fock.sub_str("ap"),
            r.sub_str("aaaa"),
            optimize="greedy",
        )
        rYXpq_fockAx_rTUya = numpy.einsum(
            "yxpq, ax, tuya -> tupq",
            r.sub_str("aaaa"),
            fock.sub_str("pa"),
            r.sub_str("aaap"),
            optimize="greedy",
        )
        rMLpq_fockKl_rTUmk = numpy.einsum(
            "mlpq, kl, tumk -> tupq",
            r.sub_str("ffaa"),
            fock.sub_str("ff"),
            r.sub_str("aaff"),
            optimize="greedy",
        )

        rBYpq_rdm1Xy_fockAb_rTUax = numpy.einsum(
            "bypq, xy, ab, tuax -> tupq",
            r.sub_str("paaa"),
            rdm1,
            fock.sub_str("pp"),
            r.sub_str("aapa"),
            optimize="greedy",
        )
        rAYpq_rdm1Xy_fockKx_rTUak = numpy.einsum(
            "aypq, xy, kx, tuak -> tupq",
            r.sub_str("paaa"),
            rdm1,
            fock.sub_str("fa"),
            r.sub_str("aapf"),
            optimize="greedy",
        )

        B_mid = (
            rMLpq_fockKl_rTUmk
            - rZYpq_fockXy_rTUzx
            - rAYpq_fockXa_rTUxy
            - rYXpq_fockAx_rTUya
            - 1 / 2 * rBYpq_rdm1Xy_fockAb_rTUax
            - 1 / 2 * rAYpq_rdm1Xy_fockKx_rTUak
        )
        B = numpy.einsum(
            "pqrs, vwtu, rsvw, tupq",
            self.t_PQrs,
            self.t_PQrs,
            rdm2,
            B_mid,
            optimize="greedy",
        )

        return B

    def _compute_intermediate_X(self, r: NBodyTensor, fock: NBodyTensor):
        """Compute intermediate X, the geminal-geminal overlap."""
        rdm1, rdm2 = self.rdm1, self.rdm2

        rTUkl_rKLpq = numpy.einsum(
            "tukl, klpq -> tupq",
            r.sub_str("aaff"),
            r.sub_str("ffaa"),
            optimize="greedy",
        )
        rTUyz_rYZpq = numpy.einsum(
            "tuyz, yzpq -> tupq",
            r.sub_str("aaaa"),
            r.sub_str("aaaa"),
            optimize="greedy",
        )

        # rTUya_rdm1Yz_rZApq = numpy.einsum("tuya, yz, zapq -> tupq", r.sub_str("aaap"), rdm1,\
        # r.sub_str("apaa"), optimize="greedy")
        rUTya_rdm1Yz_rAZpq = numpy.einsum(
            "utya, yz, azpq -> tupq",
            r.sub_str("aaap"),
            rdm1,
            r.sub_str("paaa"),
            optimize="greedy",
        )
        rTUay_rdm1Yz_rAZqp = numpy.einsum(
            "tuay, yz, azqp -> tupq",
            r.sub_str("aapa"),
            rdm1,
            r.sub_str("paaa"),
            optimize="greedy",
        )

        # X_mid = rTUkl_rKLpq - rTUyz_rYZpq - 1/2*rTUya_rdm1Yz_rZApq # in paper
        X_mid = (
            rTUkl_rKLpq
            - rTUyz_rYZpq
            - 1 / 2 * rUTya_rdm1Yz_rAZpq
            - 1 / 2 * rTUay_rdm1Yz_rAZqp
        )  # adjusted to python script
        X = -1 * numpy.einsum(
            "pqrs, vwtu, rsvx, xw, tupq",
            self.t_PQrs,
            self.t_PQrs,
            rdm2,
            fock.sub_str("aa"),
            X_mid,
            optimize="greedy",
        )

        return X

    def _compute_intermediate_Delta_paper(self, r: NBodyTensor, fock: NBodyTensor):
        """Compute intermediate Delta with the formulas of the SF-[2]_R12 paper."""
        # Delta intermediate
        # Delta from paper:
        rdm1, rdm2 = self.rdm1, self.rdm2
        tPQrs_rAYpq = numpy.einsum(
            "pqrs, aypq -> ayrs", self.t_PQrs, r.sub_str("paaa"), optimize="greedy"
        )
        # build cumulant-like terms (call such bcs look a little bit alike, but are not same)
        cum_like_1 = numpy.einsum(
            "sv, xrwy -> sxrvwy",
            rdm1,
            (
                -1 / 2 * rdm2
                + 1 / 2 * numpy.einsum("xw, ry -> xrwy", rdm1, rdm1, optimize="greedy")
                - 1 / 2 * numpy.einsum("rw, xy -> xrwy", rdm1, rdm1, optimize="greedy")
            ),
            optimize="greedy",
        )

        cum_like_2 = numpy.einsum(
            "sw, xryv -> sxrvwy",
            rdm1,
            (
                -1 / 2 * rdm2
                + numpy.einsum("rv, xy -> xryv", rdm1, rdm1, optimize="greedy")
                - 1 / 4 * numpy.einsum("ry, xv -> xryv", rdm1, rdm1)
            ),
            optimize="greedy",
        )

        cum_like_3 = numpy.einsum(
            "rv, xswy -> sxrvwy",
            rdm1,
            (rdm2 - numpy.einsum("sy, xw -> xswy", rdm1, rdm1, optimize="greedy")),
            optimize="greedy",
        )

        cum_like_4 = numpy.einsum(
            "rw, xsvy -> sxrvwy",
            rdm1,
            (
                -1 / 2 * rdm2
                + 1 / 2 * numpy.einsum("sy, xv -> xsvy", rdm1, rdm1, optimize="greedy")
            ),
            optimize="greedy",
        )

        Delta_mid = numpy.einsum(
            "ayrs, sxrvwy -> axvw",
            tPQrs_rAYpq,
            (cum_like_1 + cum_like_2 + cum_like_3 + cum_like_4),
            optimize="greedy",
        )

        Delta = -1 * numpy.einsum(
            "tuak, kx, vwtu, axvw",
            r.sub_str("aapf"),
            fock.sub_str("fa"),
            self.t_PQrs,
            Delta_mid,
            optimize="greedy",
        )
        return Delta

    def _compute_intermediate_Delta_MBeq(self, r: NBodyTensor, fock: NBodyTensor):
        """Compute intermediate Delta with formulas generated by the MBEq tool of the Valeev group."""
        rdm1, rdm2, t_PQrs = self.rdm1, self.rdm2, self.t_PQrs
        # Delta from MBeq-tool
        Delta1 = (
            -1
            / 2
            * numpy.einsum(
                "pqrs, aypq, vwtu, xrvy, kx, sw, utak",
                t_PQrs,
                r.sub_str("paaa"),
                t_PQrs,
                rdm2,
                fock.sub_str("fa"),
                rdm1,
                r.sub_str("aapf"),
                optimize="greedy",
            )
            - 1
            / 2
            * numpy.einsum(
                "pqrs, aypq, vwtu, xryv, kx, sw, tuak",
                t_PQrs,
                r.sub_str("paaa"),
                t_PQrs,
                rdm2,
                fock.sub_str("fa"),
                rdm1,
                r.sub_str("aapf"),
                optimize="greedy",
            )
            - 1
            / 2
            * numpy.einsum(
                "pqrs, aypq, vwtu, kx, rv, sw, xy, utak",
                t_PQrs,
                r.sub_str("paaa"),
                t_PQrs,
                fock.sub_str("fa"),
                rdm1,
                rdm1,
                rdm1,
                r.sub_str("aapf"),
                optimize="greedy",
            )
            + numpy.einsum(
                "pqrs, aypq, vwtu, kx, rv, sw, xy, tuak",
                t_PQrs,
                r.sub_str("paaa"),
                t_PQrs,
                fock.sub_str("fa"),
                rdm1,
                rdm1,
                rdm1,
                r.sub_str("aapf"),
                optimize="greedy",
            )
            + 1
            / 2
            * numpy.einsum(
                "pqrs, aypq, vwtu, kx, ry, sv, xw, tuak",
                t_PQrs,
                r.sub_str("paaa"),
                t_PQrs,
                fock.sub_str("fa"),
                rdm1,
                rdm1,
                rdm1,
                r.sub_str("aapf"),
                optimize="greedy",
            )
            - 1
            / 4
            * numpy.einsum(
                "pqrs, aypq, vwtu, kx, ry, sv, xw, utak",
                t_PQrs,
                r.sub_str("paaa"),
                t_PQrs,
                fock.sub_str("fa"),
                rdm1,
                rdm1,
                rdm1,
                r.sub_str("aapf"),
                optimize="greedy",
            )
        )
        Delta2 = (
            numpy.einsum(
                "pqrs, ayqp, vwtu, xrvy, kx, sw, utak",
                self.t_PQrs,
                r.sub_str("paaa"),
                self.t_PQrs,
                rdm2,
                fock.sub_str("fa"),
                rdm1,
                r.sub_str("aapf"),
                optimize="greedy",
            )
            - 1
            / 2
            * numpy.einsum(
                "pqrs, ayqp, vwtu, xrvy, kx, sw, tuak",
                self.t_PQrs,
                r.sub_str("paaa"),
                self.t_PQrs,
                rdm2,
                fock.sub_str("fa"),
                rdm1,
                r.sub_str("aapf"),
                optimize="greedy",
            )
            - numpy.einsum(
                "pqrs, ayqp, vwtu, kx, ry, sv, xw, tuak",
                self.t_PQrs,
                r.sub_str("paaa"),
                self.t_PQrs,
                fock.sub_str("fa"),
                rdm1,
                rdm1,
                rdm1,
                r.sub_str("aapf"),
                optimize="greedy",
            )
            + 1
            / 2
            * numpy.einsum(
                "pqrs, ayqp, vwtu, kx, ry, sv, xw, utak",
                self.t_PQrs,
                r.sub_str("paaa"),
                self.t_PQrs,
                fock.sub_str("fa"),
                rdm1,
                rdm1,
                rdm1,
                r.sub_str("aapf"),
                optimize="greedy",
            )
        )
        Delta = Delta1 + Delta2

        return Delta

    def _compute_intermediates(
        self, g: NBodyTensor, r: NBodyTensor, fock: NBodyTensor
    ) -> list:
        """Return the intermediates V, B, X and Delta as a list."""
        V = self._compute_intermediate_V(g, r)
        B = self._compute_intermediate_B(r, fock)
        X = self._compute_intermediate_X(r, fock)
        Delta = self._compute_intermediate_Delta_MBeq(r, fock)
        # alternative: Delta = self._compute_intermediate_Delta_paper(r, fock), seems to perform a little worse
        if not self.silent:
            print("Intermediates:")
            print("\tV\t= ", V)
            print("\tB\t= ", B)
            print("\tX\t= ", X)
            print("\tDelta\t= ", Delta)

        return [V, B, X, Delta]

    def compute_correction(
        self, g: NBodyTensor, r: NBodyTensor, fock: NBodyTensor
    ) -> float:
        """Compute the correction from the Coulomb tensor, f12 tensor and generalized Fock operator.

        Uses the SP ansatz for the amplitudes and sums the intermediates V, B, X and Delta.

        Parameters
        ----------
        g : NBodyTensor
            Coulomb tensor.
        r : NBodyTensor
            f12 tensor.
        fock : NBodyTensor
            Generalized Fock operator, see ``build_fock_operator``.

        Returns
        -------
        float
            The explicitly correlated correction.
        """

        # Here, if len(active_!=n_obs, then active is effective OBS, and n_ri-len(active) is effective CABS
        print(
            "Computing with dim(OBS): "
            + str(self.n_obs)
            + ", of which "
            + str(len(self.active))
            + " active and dim(RI): "
            + str(self.n_ri)
            + "."
        )
        print(
            "Therefore effective OBS: "
            + str(len(self.active))
            + " and effective CABS: "
            + str(self.n_ri - len(self.active))
        )

        # Set amplitudes via SP-ansatz by Ten-No (fix s.th. cusp condition is fulfilled)
        self.t_PQrs = self._sp_ansatz()

        # Compute intermediates
        intermediates = self._compute_intermediates(g, r, fock)
        # Sum up intermediates for correction
        correction = numpy.sum(intermediates)

        return correction

    def compute(self) -> float:
        """Compute the correction with tensors read from the file given in ``external_info``.

        Returns
        -------
        float
            The explicitly correlated correction.
        """
        print("Set up universal f12-correction using external integrals.")
        h, g, r = self.setup_tensors_external()
        fock = self.build_fock_operator(h=h, g=g)

        correction = self.compute_correction(g=g, r=r, fock=fock)

        return correction
