import numbers
from abc import ABC
from math import pi
from typing import List

from ..core import Color
from ..core import CircuitStyle
from ..core import Gate
import tequila as tq
import numpy
import copy


class PairCorrelatorGate(Gate, ABC):
    """
        i,j correspond to Molecular Orbital index --> ((2*i,2*j),(2*i+1,2*j+1))
        only employed if n_qubits_is_double = False
    """

    def __init__(self, i, j, angle, control=None, assume_real=True, encoding="JordanWigner", unit_of_pi: bool = False,*args, **kwargs):
        self.i = i
        self.j = j
        self.angle = tq.assign_variable(angle)
        self.unit_of_pi = unit_of_pi
        self.control = control
        self.assume_real = assume_real
        if "molecule" not in kwargs:
            k = max(self.i, self.j)
            x = numpy.zeros(k ** 2).reshape([k, k])
            y = numpy.zeros(k ** 4).reshape([k, k, k, k])
            if encoding is None:
                encoding = "jordanwigner"
            self.encoding = encoding.lower()
            if "molecule_factory" not in kwargs:
                self._dummy = tq.Molecule(geometry="", one_body_integrals=x, two_body_integrals=y,
                                          nuclear_repulsion=0.0,
                                          transformation=self.encoding, *args, **kwargs)
            else:
                mol_fact = kwargs["molecule_factory"]
                kwargs.pop("molecule_factory")
                self._dummy = mol_fact(geometry="", one_body_integrals=x, two_body_integrals=y, nuclear_repulsion=0.0,
                                       transformation=self.encoding, *args, **kwargs)
        else:
            self._dummy = kwargs["molecule"]
            self.encoding = self._dummy.transformation
            kwargs.pop("molecule")

    def construct_circuit(self, *args, **kwargs):
        if self.encoding == "jordanwigner":
            return tq.gates.QubitExcitation(angle=self.angle, control=self.control,
                                            target=[2 * self.i, 2 * self.j, 2 * self.i + 1, 2 * self.j + 1],
                                            assume_real=self.assume_real)
        else:
            return self._dummy.UC(self.i, self.j, control=self.control, assume_real=self.assume_real, angle=self.angle,
                                  *args, **kwargs)

    def map_variables(self, variables):
        mapped_gate = copy.deepcopy(self)
        mapped_gate.angle = mapped_gate.angle.map_variables(variables)
        return mapped_gate

    def used_wires(self) -> List[int]:

        xi = self.i
        xj = self.j

        used = list(range(min(xi, xj), max(xi, xj) + 1))
        if self.control is not None:
            used.append(self.control // 2)

        return used

    def dagger(self) -> "Gate":
        raise NotImplemented

    def render_angle(self, style: CircuitStyle, gcol: Color, tcol: Color) -> [Color, Color]:
        if not isinstance(self.angle, numbers.Number) and style.parametrized_marking is not None:
            tcol = style.parametrized_marking.text_color
            gcol = style.parametrized_marking.gate_color

        if isinstance(self.angle, numbers.Number) and style.color_range is not None:
            if self.unit_of_pi:
                angle = int(((self.angle % 2) / 2) * 100)
            else:
                angle = int(abs((self.angle / (2 * pi)) * 100)) % 100
            gcol = style.color_range.interpolate(angle)

        return gcol, tcol

    def _render(self, state, style):
        # styling
        shape = 6
        tcol = style.primary.text_color
        gcol = style.primary.gate_color

        # angle
        gcol, tcol = self.render_angle(style, gcol, tcol)
        if 'UC' in style.personalized:
            gcol = style.personalized['UC'].gate_color
            tcol = style.personalized['UC'].text_color
        # TODO: param is not used again in the original??
        param = self.angle
        if isinstance(param, numbers.Number):
            param = "{:1.2f}".format(param)

        # rendering
        result = " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.i, shape=shape,gcol=gcol,tcol="{" + tcol.name + "}",op="")
        result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=self.j, shape=shape,gcol=gcol,tcol="{" + tcol.name + "}",op="")

        return result