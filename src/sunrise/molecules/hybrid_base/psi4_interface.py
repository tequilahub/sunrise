import typing

from sunrise.molecules.qubit_base.psi4_interface import Psi4Mixin

from .hybridbase import HybridBase


class QuantumChemistryPsi4(Psi4Mixin, HybridBase):
    """
    Hybrid molecule with the integrals and the classical methods of psi4
    see sunrise.molecules.qubit_base.psi4_interface for all options
    """

    def _update_after_active_space(self, *args, **kwargs):
        """
        HybridBase built the select and the transformation for all orbitals, psi4 set the active space afterwards
        select can be given for all orbitals (then the entries of the frozen ones are dropped) or for the active ones
        """
        n_total = self.integral_manager.one_body_integrals.shape[0]
        if self.n_orbitals == n_total:
            return  # no active space

        select = kwargs.get("select")
        if select is None:
            select = {}
        if len(select) == n_total:
            # self.select is the dict of select for all orbitals
            select = {
                o.idx: self.select[o.idx_total]
                for o in self.integral_manager.active_orbitals
            }

        transformation = kwargs.get("transformation")
        if not isinstance(transformation, typing.Callable):
            # the transformation needs the number of active orbitals
            self._select = {}
            self.transformation = self._initialize_transformation(*args, **kwargs)
        self.update_select(select, n_orb=self.n_orbitals)
        self.up_then_down = self.transformation.up_then_down
