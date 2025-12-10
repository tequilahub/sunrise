import numbers

from ..core import CircuitState, CircuitStyle
from . import PairCorrelatorGate
from ..core import Polygon
from typing import List

class SingleExcitation(PairCorrelatorGate):
    """
        i,j correspond to Spin Orbital index --> ((i,j))
    """
    def __init__(self, i, j, angle, control=None, assume_real=True, encoding="JordanWigner", unit_of_pi: bool = False,n_qubits_is_double:bool=False,*args, **kwargs):
        super().__init__(i=i, j=j, angle=angle, control=control, assume_real=assume_real, encoding=encoding,unit_of_pi=unit_of_pi,*args,**kwargs)
        self.n_qubits_is_double = n_qubits_is_double

    def construct_circuit(self, *args, **kwargs):
        return self._dummy.make_excitation_gate(indices=(self.i, self.j), angle=self.angle,
                                                assume_real=self.assume_real, control=self.control, *args, **kwargs)

    def _render(self, state: CircuitState, style: CircuitStyle) -> str:
        spin = {0: Polygon(-3), 1: Polygon(3)}

        # styling
        shape1 = spin[self.i % 2]
        shape2 = spin[self.j % 2]
        tcol = style.primary.text_color
        gcol = style.primary.gate_color

        # angle
        gcol, tcol = self.render_angle(style, gcol, tcol)
        if 'single' in style.personalized:
            gcol = style.personalized['single'].gate_color
            tcol = style.personalized['single'].text_color
        # TODO: param is not used again in the original??
        param = self.angle
        if isinstance(param, numbers.Number):
            param = "{:1.2f}".format(param)

        # rendering
        result = " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.i // ((not self.n_qubits_is_double)+1),shape=shape1, gcol=gcol,tcol="{" + tcol.name + "}",op="")
        result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.j // ((not self.n_qubits_is_double)+1),shape=shape2, gcol=gcol,tcol="{" + tcol.name + "}",op="")
        return result
    
    def used_wires(self) -> List[int]:

        xi = self.i// ((not self.n_qubits_is_double)+1)
        xj = self.j// ((not self.n_qubits_is_double)+1)

        used = list(range(min(xi, xj), max(xi, xj) + 1))
        if self.control is not None:
            used.append(self.control// ((not self.n_qubits_is_double)+1))

        return used