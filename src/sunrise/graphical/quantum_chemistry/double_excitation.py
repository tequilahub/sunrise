import numbers
from typing import List

from ..core import CircuitState, CircuitStyle
from ..quantum_chemistry import PairCorrelatorGate
from ..core import Polygon


class DoubleExcitation(PairCorrelatorGate):
    """
            i,j,k,l correspond to Spin Orbital index --> ((i,j),(k,l))
    """

    def __init__(self, i, j, k, l, angle, control=None, assume_real=True, encoding="JordanWigner", n_qubits_is_double:bool=False, *args, **kwargs):
        super().__init__(i=i, j=j, angle=angle, control=control, assume_real=assume_real, encoding=encoding, *args,**kwargs)
        self.n_qubits_is_double = n_qubits_is_double
        self.k = k
        self.l = l
        self.type = 3
        if not self.n_qubits_is_double:
            if self.i // 2 == self.k // 2 and self.j // 2 == self.l // 2:
                self.type = 0  # Actually its a UC
            elif self.i // 2 == self.k // 2 and self.j // 2 != self.l // 2:
                self.type = 1  # Origin Orbital Paired, Destination Unpaired
            elif self.i // 2 != self.k // 2 and self.j // 2 == self.l // 2:
                self.type = 2  # Origin Orbital Unpaired, Destination Paired
            else:
                self.type = 3  # Completely Unpaired


    def used_wires(self) -> List[int]:
        used = list(range(min(self.i, self.j, self.k, self.l) // (1+(not self.n_qubits_is_double)), max(self.i, self.j, self.k, self.l) // (1+(not self.n_qubits_is_double)) + 1))

        if self.control is not None:
            used.append(self.control // (1+(not self.n_qubits_is_double)))
        return used

    def construct_circuit(self, *args, **kwargs):
        if not self.type:
            return PairCorrelatorGate(i=self.i // 2, j=self.j // 2, assume_real=self.assume_real, angle=self.angle,encoding=self.encoding).construct_circuit(*args, **kwargs)
        else:
            return self._dummy.make_excitation_gate(indices=((self.i, self.j), (self.k, self.l)),assume_real=self.assume_real, angle=self.angle,control=self.control, *args, **kwargs)

    def _render(self, state: CircuitState, style: CircuitStyle) -> str:
        spin = {0: Polygon(-3), 1: Polygon(3)}

        # styling
        tcol = style.secondary.text_color
        gcol = style.secondary.gate_color

        # angle
        gcol, tcol = self.render_angle(style, gcol, tcol)
        if 'double' in style.personalized:
            gcol = style.personalized['double'].gate_color
            tcol = style.personalized['double'].text_color
        # TODO: param is not used again in the original??
        param = self.angle
        if isinstance(param, numbers.Number):
            param = "{:1.2f}".format(param)

        # render
        result = ""
        if not self.type:
            shape = Polygon(6)
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.i // 2,shape=shape,gcol=gcol,tcol="{" + tcol.name + "}",op="")
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.j // 2,shape=shape,gcol=gcol,tcol="{" + tcol.name + "}",op="")
        elif self.type == 1:
            shape1 = Polygon(6)
            shape2 = spin[self.j % 2]
            shape3 = spin[self.l % 2]
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.i // 2,shape=shape1,gcol=gcol,tcol="{" + tcol.name + "}",op="")
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.j // 2,shape=shape2,gcol=gcol,tcol="{" + tcol.name + "}",op="")
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.l // 2,shape=shape3,gcol=gcol,tcol="{" + tcol.name + "}",op="")
        elif self.type == 2:
            shape3 = Polygon(6)
            shape1 = spin[self.i % 2]
            shape2 = spin[self.k % 2]
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.i // 2,shape=shape1,gcol=gcol,tcol="{" + tcol.name + "}",op="")
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.k // 2,shape=shape2,gcol=gcol,tcol="{" + tcol.name + "}",op="")
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.j // 2,shape=shape3,gcol=gcol,tcol="{" + tcol.name + "}",op="")
        else:
            shape1 = spin[self.i % 2]
            shape2 = spin[self.j % 2]
            shape3 = spin[self.k % 2]
            shape4 = spin[self.l % 2]
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.i // (1+(not self.n_qubits_is_double)),shape=shape1,gcol=gcol,tcol="{" + tcol.name + "}",op="")
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.j // (1+(not self.n_qubits_is_double)),shape=shape2,gcol=gcol,tcol="{" + tcol.name + "}",op="")
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.k // (1+(not self.n_qubits_is_double)),shape=shape3,gcol=gcol,tcol="{" + tcol.name + "}",op="")
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.l // (1+(not self.n_qubits_is_double)),shape=shape4,gcol=gcol,tcol="{" + tcol.name + "}",op="")
        return result