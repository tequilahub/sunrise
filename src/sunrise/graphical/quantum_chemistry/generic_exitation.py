import numbers
from typing import List

from ..core import CircuitState, CircuitStyle
from ..quantum_chemistry import PairCorrelatorGate
from ..core import Polygon
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

class GenericExcitation(Gate, ABC):
    """
            indices [(i,j),(k,l),...] correspond to the n electron excitation i -> j, k -> l,...
    """

    def __init__(self, indices, angle, control=None, assume_real=True, encoding="JordanWigner",n_qubits_is_double:bool=False, unit_of_pi: bool = False,*args, **kwargs):
        assert isinstance(indices,(list,tuple))
        if isinstance(indices[0],numbers.Number):
            indices = [indices]
        self.indices = indices
        self.idx = []
        for i in self.indices:
            self.idx.extend(i)
        self.angle = tq.assign_variable(angle)
        self.unit_of_pi = unit_of_pi
        self.control = control
        self.assume_real = assume_real
        self.n_qubits_is_double = n_qubits_is_double
        if "molecule" not in kwargs:
            k = max(self.idx)
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
        return self._dummy.make_excitation_gate(angle=self.angle, control=self.control,indices=self.indices,assume_real=self.assume_real)

    def map_variables(self, variables):
        mapped_gate = copy.deepcopy(self)
        mapped_gate.angle = mapped_gate.angle.map_variables(variables)
        return mapped_gate

    def used_wires(self) -> List[int]:

        used = list(range(min(self.idx) // (1+(not self.n_qubits_is_double)), max(self.idx) // (1+(not self.n_qubits_is_double)) + 1))
        if self.control is not None:
            used.append(self.control // (1+(not self.n_qubits_is_double)))

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
        spin = {0: Polygon(-3), 1: Polygon(3)}
        # styling
        tcol = style.primary.text_color
        gcol = style.primary.gate_color

        # angle
        gcol, tcol = self.render_angle(style, gcol, tcol)
        if 'generic' in style.personalized:
            gcol = style.personalized['generic'].gate_color
            tcol = style.personalized['generic'].text_color
        param = self.angle
        if isinstance(param, numbers.Number):
            param = "{:1.2f}".format(param)

        # rendering
        result = ""
        for exct in self.indices:
            shape1 = spin[exct[0] % 2]
            shape2 = spin[exct[1] % 2]
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=exct[0] // (1+(not self.n_qubits_is_double)),shape=shape1,gcol=gcol,tcol="{" + tcol.name + "}",op="")
            result += " a{qubit} P:fill={gcol}:shape={shape} \\textcolor{tcol}{{{op}}} ".format(qubit=exct[1] // (1+(not self.n_qubits_is_double)),shape=shape2,gcol=gcol,tcol="{" + tcol.name + "}",op="")

        return result