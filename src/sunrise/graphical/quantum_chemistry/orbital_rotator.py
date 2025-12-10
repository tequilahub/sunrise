import numbers

from ..core import CircuitState, CircuitStyle
from ..quantum_chemistry import PairCorrelatorGate


class OrbitalRotatorGate(PairCorrelatorGate):
    """
        i,j correspond to Molecular Orbital index --> ((2*i,2*j),(2*i+1,2*j+1))
        only employed if n_qubits_is_double = False
    """

    def construct_circuit(self):
        return self._dummy.UR(self.i, self.j, control=self.control, assume_real=self.assume_real, angle=self.angle)


    def _render(self, state: CircuitState, style: CircuitStyle) -> str:
        # styling
        shape = 2
        tcol = style.primary.text_color
        gcol = style.primary.gate_color

        # angle
        gcol, tcol = self.render_angle(style, gcol, tcol)
        if 'UR' in style.personalized:
            gcol = style.personalized['UR'].gate_color
            tcol = style.personalized['UR'].text_color
        # TODO: param is not used again in the original??
        param = self.angle
        if isinstance(param, numbers.Number):
            param = "{:1.2f}".format(param)

        # rendering
        result = " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.i, shape=shape,gcol=gcol,tcol="{" + tcol.name + "}",op="")
        result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.j, shape=shape,gcol=gcol,tcol="{" + tcol.name + "}",op="")

        return result
