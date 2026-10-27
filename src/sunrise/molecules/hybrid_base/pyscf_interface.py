from __future__ import annotations

import typing

from sunrise.molecules.qubit_base import ParametersQC
from sunrise.molecules.qubit_base.pyscf_interface import PySCFMixin

from .hybridbase import HybridBase


class PyscfHybridBase(PySCFMixin, HybridBase):
    """
    Hybrid molecule with the integrals and the classical methods of PySCF
    see sunrise.molecules.qubit_base.pyscf_interface for all options
    """

    def __init__(
        self,
        parameters: ParametersQC,
        select: str | dict | None = None,
        transformation: str | typing.Callable | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            parameters, *args, select=select, transformation=transformation, **kwargs
        )

    def compute_energy(self, method: str, *args, **kwargs) -> float:
        if "hybrid" in method.lower():
            # restricted to the hybrid encoding, done by HybridBase instead of PySCF
            return super(PySCFMixin, self).compute_energy(method, *args, **kwargs)
        return super().compute_energy(method, *args, **kwargs)
