"""Hybrid molecule with integrals and classical methods from PySCF."""

from __future__ import annotations

import typing

from sunrise.molecules.qubit_base import ParametersQC
from sunrise.molecules.qubit_base.pyscf_interface import PySCFMixin

from .hybridbase import HybridBase


class PyscfHybridBase(PySCFMixin, HybridBase):
    """Hybrid molecule with integrals and classical methods from PySCF, see ``PySCFMixin`` for the
    backend options.

    Parameters
    ----------
    parameters : ParametersQC
        Geometry, basis set and charge of the molecule.
    select : str or dict, optional
        Orbital types of the hybrid encoding, e.g. "BFFB".
    transformation : str or callable, optional
        Fermion-to-qubit transformation.
    *args, **kwargs
        See ``PySCFMixin``.
    """

    def __init__(
        self,
        parameters: ParametersQC,
        select: str | dict | None = None,
        transformation: str | typing.Callable | None = None,
        *args,
        **kwargs,
    ):
        """See the class docstring."""
        super().__init__(
            parameters, *args, select=select, transformation=transformation, **kwargs
        )

    def compute_energy(self, method: str, *args, **kwargs) -> float:
        """Compute the energy of a classical or quantum method.

        Methods with "hybrid" in the name are handled by ``HybridBase``, all others by ``PySCFMixin.compute_energy``.

        Parameters
        ----------
        method : str
            Name of the method.
        *args, **kwargs
            Passed on to the method.

        Returns
        -------
        float
        """
        if "hybrid" in method.lower():
            # restricted to the hybrid encoding, done by HybridBase instead of PySCF
            return super(PySCFMixin, self).compute_energy(method, *args, **kwargs)
        return super().compute_energy(method, *args, **kwargs)
