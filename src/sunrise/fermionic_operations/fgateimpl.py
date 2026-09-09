import numbers
import typing
from tequila.objective.objective import FixedVariable,Variable
from .circuit import FCircuit
from tequila import TequilaException
from tequila import assign_variable
from copy import deepcopy
import numbers
from math import pi
from sunrise.graphical.core.shape import Polygon

# Spin orientation convention: up = ▲ (tip up), down = ▼ (tip down)
SPIN_SHAPES = {0: Polygon(3), 1: Polygon(-3)}

class FGateImpl:
    def __init__(self,indices:typing.Union[list,tuple],variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None,reordered:bool=False):
        self.reordered:bool=reordered
        self._indices:list = indices
        self.variables=variables
        self._name:str = 'GenericFermionic'
        self.verify()
        return FCircuit.wrap_gate(gate=self)

    @property
    def name(self):
        return self._name
    
    def is_controlled(self):
        return False #TODO: at some point we should

    def to_upthendown(self,norb:int):
        if not self.reordered:
            self._indices = [[(idx[0]//2+(idx[0]%2)*norb,idx[1]//2+(idx[1]%2)*norb) for idx in gate] for gate in self._indices]
            self.reordered = True
        return self
    
    def to_udud(self,norb:int):
        if self.reordered:
            self._indices = [[(2*(idx[0]%norb)+(idx[0]>=norb),2*(idx[1]%norb)+(idx[1]>=norb)) for idx in gate] for gate in self._indices]
            self.reordered = False
        return self
    
    def __str__(self):
        return f'{self.name}(indices = {self.indices} ,variables = {repr(self.variables)})'

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        if self._name != other._name:
            return False
        elif self._indices != other._indices:
            return False
        elif self.extract_variables() != other.extract_variables():
            return False
        return True
            
    def is_parameterized(self)->bool:
        return not isinstance(self.variables,(FixedVariable,numbers.Number))

    @property
    def indices(self):
        return self._indices
     
    @indices.setter
    def indices(self,indices):
        self._indices = indices
    
    def extract_variables(self)->list[Variable]:
        if self.is_parameterized() and hasattr(self.variables, "extract_variables"):
            return self.variables.extract_variables()
        else:
            return []

    @property
    def variables(self)->Variable:
        return self._variables
    
    @variables.setter
    def variables(self,variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]):
        self._variables:Variable = assign_variable(variable=variables)

    def verify(self):
        assert isinstance(self._variables,(typing.Hashable, numbers.Real, Variable, FixedVariable))
        if isinstance(self._indices[0],numbers.Number): #[1,3]
            self._indices = [[tuple(self._indices),],] #->[[(1,3)]]
        elif isinstance(self._indices[0][0],numbers.Number): #[(0,2),(1,2)]
            self._indices = [self._indices,] #->[[(0,2),(1,3)],]
        elif not isinstance(self._indices[0][0][0],numbers.Number): #[[(0,2),(1,2)]]
            raise TequilaException(f'Indices formating not recognized, received {self._indices}')
        if isinstance(self._variables,FixedVariable):
            assert 1 == len(self._indices) 
        else:
            assert len(self._variables) == len(self._indices)

    @property
    def qubits(self):
        if self._indices is not None:
            q = []
            for gate in self._indices:
                for exct in gate:
                    q.extend(exct)
            return sorted(list(set(q)))
        else: return 0

    @property
    def n_qubits(self)->int:
        return len(self.qubits)
    
    def map_qubits(self,qubit_map:dict={}):
        self._indices =[[(qubit_map[idx[0]],qubit_map[idx[1]]) for idx in gate] for gate in self._indices]
        return self

    def map_variables(self,var_map:dict={}):
        if self.is_parameterized() and hasattr(self.variables, "extract_variables"):
            self.variables = self.variables.map_variables(var_map)
        return self 

    def dagger(self):
        indinces = []
        cir = deepcopy(self)
        for gate in reversed(cir._indices):
            indinces.extend([[tuple([*idx][::-1]) for idx in gate]])
        cir.indices = indinces
        return cir

    @property
    def max_qubit(self):
        if self.qubits:
            return self.qubits[-1]
        else: return 0

    def __eq__(self, other) -> bool:
        if self.name != other.name:
            return False
        if self.reordered != other.reordered:
            return False
        if self.variables != other.variables:
            return False
        if self.indices != other.indices:
            return False
        return True

    def _render(self, state, style, show_spatial_orbitals=True):
        """Fallback rendering for generic/unhandled gates."""
        norb = state.n_orbitals
        
        def spatial_of(q):
            if not show_spatial_orbitals: return q
            return (q if q < norb else q - norb) if self.reordered else (q // 2)

        result = "".join(f" a{spatial_of(q)} " for q in self.qubits)
        return result + " G G "

    def render_angle(self, style, gcol, tcol):
        """Color symbolic gates with parametrized_marking, or numeric gates with color_range."""
        # Inlined numeric extraction
        try:
            value = float(self.variables)
        except (TypeError, ValueError):
            try: value = float(self.variables())
            except Exception: value = None
            
        if value is None and style.parametrized_marking is not None:
            return style.parametrized_marking.gate_color, style.parametrized_marking.text_color
        if value is not None and style.color_range is not None:
            angle = int(abs((value / (2 * pi)) * 100)) % 100
            return style.color_range.interpolate(angle), tcol
        return gcol, tcol

class FermionicExcitationImpl(FGateImpl):
    def __init__(self, indices:typing.Union[list,tuple], variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable,None]=None, reordered:bool=False):
        super().__init__(indices, variables, reordered)
        self._name = 'FermionicExcitation'

    def _render(self, state, style, show_spatial_orbitals=True):
        norb = state.n_orbitals
        spin = SPIN_SHAPES
        hexagon = Polygon(6)

        def spatial_of(q):
            if not show_spatial_orbitals: return q
            return (q if q < norb else q - norb) if self.reordered else (q // 2)
        def spin_of(q):
            return (0 if q < norb else 1) if self.reordered else (q % 2)

        def group_kind(group):
            if len(group) == 1: return "single"
            elif len(group) == 2: return "double"
            else: return "generic"

        def base_colors(kind):
            if kind == "double":
                tcol = style.secondary.text_color
                gcol = style.secondary.gate_color
            else:
                tcol = style.primary.text_color
                gcol = style.primary.gate_color
            gcol, tcol = self.render_angle(style, gcol, tcol)
            if kind in style.personalized:
                gcol = style.personalized[kind].gate_color
                tcol = style.personalized[kind].text_color
            return gcol, tcol

        # Brighter (Start) and Darker (End) using TikZ color mixing
        def place_start(q, shape, gcol, tcol):
            return f" a{q} P:fill={gcol}!80!white:shape={shape} \\textcolor{{black}}{{}} "

        def place_end(q, shape, gcol, tcol):
            return f" a{q} P:fill={gcol}!90!black:shape={shape} \\textcolor{{{tcol.name}}}{{}} "

        result = ""
        for group in self._indices:
            kind = group_kind(group)
            gcol, tcol = base_colors(kind)

            if len(group) == 2 and show_spatial_orbitals:
                (i, j), (k, l) = group
                si, sj, sk, sl = spatial_of(i), spatial_of(j), spatial_of(k), spatial_of(l)
                spi_i, spi_j, spi_k, spi_l = spin_of(i), spin_of(j), spin_of(k), spin_of(l)

                if si == sk and sj == sl:
                    result += place_start(si, hexagon, gcol, tcol) + place_end(sj, hexagon, gcol, tcol)
                elif si == sk:
                    result += place_start(si, hexagon, gcol, tcol) + place_end(sj, spin[spi_j], gcol, tcol) + place_end(sl, spin[spi_l], gcol, tcol)
                elif sj == sl:
                    result += place_start(si, spin[spi_i], gcol, tcol) + place_start(sk, spin[spi_k], gcol, tcol) + place_end(sj, hexagon, gcol, tcol)
                else:
                    result += place_start(si, spin[spi_i], gcol, tcol) + place_end(sj, spin[spi_j], gcol, tcol)
                    result += place_start(sk, spin[spi_k], gcol, tcol) + place_end(sl, spin[spi_l], gcol, tcol)
            else:
                for i, j in group:
                    result += place_start(spatial_of(i), spin[spin_of(i)], gcol, tcol)
                    result += place_end(spatial_of(j), spin[spin_of(j)], gcol, tcol)
        return result

class URImpl(FGateImpl):
    def __init__(self, i:int,j:int, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable,None]=None):
        super().__init__([[(2*i,2*j)],[(2*i+1,2*j+1)]], variables, False)
        self._name = 'UR'
    def verify(self):
        assert isinstance(self._variables,(typing.Hashable, numbers.Real, Variable, FixedVariable))
        if isinstance(self._indices[0],numbers.Number): #[1,3]
            self._indices = [[tuple(self._indices),],] #->[[(1,3)]]
        elif isinstance(self._indices[0][0],numbers.Number): #[(0,2),(1,2)]
            self._indices = [self._indices,] #->[[(0,2),(1,3)],]
        elif not isinstance(self._indices[0][0][0],numbers.Number): #[[(0,2),(1,2)]]
            raise TequilaException(f'Indices formating not recognized, received {self._indices}')
        if isinstance(self._variables,FixedVariable):
            assert 2 == len(self._indices) 
        else:
            assert 2*len(self._variables) == len(self._indices)

    def _render(self, state, style, show_spatial_orbitals=True):
        norb = state.n_orbitals  
        
        def spatial_of(q):
            if not show_spatial_orbitals: return q
            return (q if q < norb else q - norb) if self.reordered else (q // 2)
        def spin_of(q):
            return (0 if q < norb else 1) if self.reordered else (q % 2)

        tcol = style.primary.text_color
        gcol = style.primary.gate_color
        gcol, tcol = self.render_angle(style, gcol, tcol)
        if 'UR' in style.personalized:
            gcol = style.personalized['UR'].gate_color
            tcol = style.personalized['UR'].text_color

        if show_spatial_orbitals:
            i, j = self._indices[0][0]
            return (f" a{spatial_of(i)} P:fill={gcol}!80!white:shape=2 \\textcolor{{black}}{{}} "
                    f" a{spatial_of(j)} P:fill={gcol}!90!black:shape=2 \\textcolor{{{tcol.name}}}{{}} ")
                    
        spin = SPIN_SHAPES
        lines = []
        for group in self._indices:
            line = ""
            for i, j in group:
                line += f" a{i} P:fill={gcol}!80!white:shape={spin[spin_of(i)]} \\textcolor{{black}}{{}} "
                line += f" a{j} P:fill={gcol}!90!black:shape={spin[spin_of(j)]} \\textcolor{{{tcol.name}}}{{}} "
            lines.append(line)
        return "\n".join(lines)

class UCImpl(FGateImpl):
    def __init__(self,i,j, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None):
        super().__init__([[(2*i,2*j),(2*i+1,2*j+1)]], variables, False)
        self._name = 'UC'

    def _render(self, state, style, show_spatial_orbitals=True):
        norb = state.n_orbitals  
        
        def spatial_of(q):
            if not show_spatial_orbitals: return q
            return (q if q < norb else q - norb) if self.reordered else (q // 2)
        def spin_of(q):
            return (0 if q < norb else 1) if self.reordered else (q % 2)

        tcol = style.secondary.text_color
        gcol = style.secondary.gate_color
        gcol, tcol = self.render_angle(style, gcol, tcol)
        if 'UC' in style.personalized:
            gcol = style.personalized['UC'].gate_color
            tcol = style.personalized['UC'].text_color
            
        result = ""
        if show_spatial_orbitals:
            shape = Polygon(6)
            i, j = self._indices[0][0]
            result += f" a{spatial_of(i)} P:fill={gcol}!80!white:shape={shape} \\textcolor{{black}}{{}} "
            result += f" a{spatial_of(j)} P:fill={gcol}!90!black:shape={shape} \\textcolor{{{tcol.name}}}{{}} "
        else:
            spin = SPIN_SHAPES
            for group in self._indices:
                for pair in group:
                    i, j = pair
                    result += f" a{spatial_of(i)} P:fill={gcol}!80!white:shape={spin[spin_of(i)]} \\textcolor{{black}}{{}} "
                    result += f" a{spatial_of(j)} P:fill={gcol}!90!black:shape={spin[spin_of(j)]} \\textcolor{{{tcol.name}}}{{}} "
        return result
    
class PhaseImpl(FGateImpl):
    def __init__(self,i, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None, reordered:bool=False):
        super().__init__([[(i,i)]], variables, reordered)
        self._name = 'Ph'

    def __str__(self):
        return f'{self.name}(target = {(self.indices[0][0][0],)}, variable = {repr(self.variables)})'

    def _render(self, state, style, show_spatial_orbitals=True):
        norb = state.n_orbitals  # Moved to top
        
        def spatial_of(q):
            if not show_spatial_orbitals: return q
            return (q if q < norb else q - norb) if self.reordered else (q // 2)

        shape = Polygon(4)
        tcol = style.primary.text_color
        gcol = style.primary.gate_color
        gcol, tcol = self.render_angle(style, gcol, tcol)
        if 'Phase' in style.personalized:
            gcol = style.personalized['Phase'].gate_color
            tcol = style.personalized['Phase'].text_color
            
        i, j = self._indices[0][0] 
        result = f" a{spatial_of(i)} P:fill={gcol}:shape={shape} \\textcolor{{{tcol.name}}}{{}} "
        return result