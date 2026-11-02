"""F12 correction using the tensors computed by MADNESS."""

from __future__ import annotations

import numpy
from tequila import TequilaException

from ._f12_correction_base import ExplicitCorrelationCorrection


class ExplicitCorrelationCorrectionMadness(ExplicitCorrelationCorrection):
    """Explicitly correlated correction using PNO orbitals and integrals from MADNESS.

    Parameters
    ----------
    mol : QuantumChemistryMadness
        The molecule.
    rdm1 : numpy.ndarray, optional
        One-electron reduced density matrix.
    rdm2 : numpy.ndarray, optional
        Two-electron reduced density matrix (spin-free), in the physics (Dirac) ordering as
        returned by ``compute_rdms``.
    n_ri : int, optional
        Dimension of the RI basis. By default the largest available one is used.
    gamma : float, optional
        Exponent of the correlation factor f_12 = -1/gamma * exp(-gamma*r_12). Default is 1.4.
    f12_filename : str, optional
        File with the f12 tensor <rs|f_12|pq> in Mulliken ordering, either ``.bin``
        (from MADNESS, converted to ``.npy`` on the fly) or ``.npy``.
        Default is "molecule_f12tensor.bin".
    **kwargs
        Same as for ``ExplicitCorrelationCorrection``, e.g. ``U`` and ``variables``
        to compute the RDMs.

    Raises
    ------
    TequilaException
        If ``f12_filename`` is neither a ``.bin`` nor a ``.npy`` file.
    """

    def __init__(
        self,
        mol=None,
        rdm1: numpy.ndarray | None = None,
        rdm2: numpy.ndarray | None = None,
        n_ri: int | None = None,
        gamma: float = 1.4,
        f12_filename: str = "molecule_f12tensor.bin",
        **kwargs,
    ):
        """See the class docstring for the parameters."""
        # Set up known external info for madness
        if f12_filename.endswith(".bin"):
            try:
                f12_data = numpy.fromfile(f12_filename)
                sd = int(numpy.power(f12_data.size, 0.25))
                assert sd**4 == f12_data.size
                sds = [sd] * 4
                f12 = f12_data.reshape(sds)
                f12_filename = f12_filename[:-3] + "npy"
                numpy.save(f12_filename, arr=f12)
            except Exception:  # noqa: BLE001
                print(f"Error while trying to read {f12_filename}!")
        elif not f12_filename.endswith(".npy"):
            raise TequilaException
        external_info = {"f12_filename": f12_filename, "ordering": "chem"}
        super().__init__(
            mol=mol,
            rdm1=rdm1,
            rdm2=rdm2,
            n_ri=n_ri,
            gamma=gamma,
            external_info=external_info,
            **kwargs,
        )

    def setup_tensors_madness(self):
        """Set up the tensors from the MADNESS integrals.

        Like ``setup_tensors_external``, but the f12 integrals <rs|f_12|pq> in the file are
        assumed to be in Mulliken ("chem") ordering and to use f_12 = exp(-gamma*r_12);
        they are rescaled by -1/gamma to obtain the factor used in the correction.

        Returns
        -------
        tuple of NBodyTensor
            One-body tensor ``h``, Coulomb tensor ``g`` and f12 tensor ``r``.
        """

        h, g, r = super().setup_tensors_external()
        # Assuming a f12-operator of the kind exp(-gamma*r12)
        r.elems *= -1 / self.gamma

        return h, g, r

    def compute(self) -> float:
        """Compute the correction using the MADNESS tensors read from the external file.

        Returns
        -------
        float
            The explicitly correlated correction.
        """
        print("Set up universal f12-correction using external integrals from madness.")
        # Prepare tensors
        h, g, r = self.setup_tensors_madness()
        fock = super().build_fock_operator(h=h, g=g)

        # Compute correction using base class method
        correction = super().compute_correction(g=g, r=r, fock=fock)

        return correction
