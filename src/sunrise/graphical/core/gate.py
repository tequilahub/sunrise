import abc
from os import path
from typing import Optional, List,Union,Tuple

import tequila as tq

from .color import RGB, DefaultColors, Color,Colors,ColorRange
from .state import CircuitState, CircuitStyle
from ... import graphical #import qpic_to_pdf,qpic_to_png
class Gate(abc.ABC):
    """
    Gate that can generate it's qpic visualization on its own as well as construct the matching tequila circuit.
    """

    @abc.abstractmethod
    def construct_circuit(self) -> tq.QCircuit:
        """Constructs the matching QCircuit"""
        pass

    @abc.abstractmethod
    def map_variables(self, variables) -> "Gate":
        pass


    def render(self, state: CircuitState, style: CircuitStyle) -> str:
        """
        This method renders the gate to a qpic string.
        This is the generic wrapper adding styling which is the same for each gate.
        When rendering subgates ALWAYS use this method.
        """
        out = self._render(state, style)
        if style.group_together:
            out += "\n" + state.all_names() + " TOUCH\n"
        return out


    @abc.abstractmethod
    def _render(self, state: CircuitState, style: CircuitStyle) -> str:
        """
        This method should return the qpic string for this gate, tailing \n is not needed!
        Do NOT call this method directly, instead use the generic render() method!
        """
        pass


    def export_qpic(self, filename: str, filepath: Optional[str] = None, style: dict = {}, labels: dict[int, str] = {},colors: dict[str, RGB] = {}, wire_colors: dict[int, Color] = {} , select:dict=None):
        """
        This method renders this gate/circuit into a complete qpic document and stores it in the given filepath (or cwd if omitted).
        Parameters:
            filename -- Name of the resulting file
            filepath -- If given, the qpic file will be stored in this directory.
            style -- Defines the styling that should be used to render the circuit.
            labels -- Dictionary mapping wire indices to names. These well be displayed instead of the index.
            colors -- Dictionary of additional colors that should be defined. The DefaultColors are prepended and can thus be overridden.
            wire_colors -- Dictionary mapping wire indices to a Color. If set the wire will be colored instead of defaulting to black.
        """
        result = ""
        style,colors = self._parse_CircuitStyle(style,colors)
        if not len(wire_colors) and select is not None:
            for s in select:
                if select[s] == 'B':
                    wire_colors[s] = "red"
        # add colors
        for color, rgb in DefaultColors.items():
            result += f"COLOR {color} {rgb.r} {rgb.g} {rgb.b}\n"

        for color, rgb in colors.items():
            result += f"COLOR {color} {rgb.r} {rgb.g} {rgb.b}\n"

        # add wires
        wires = self.used_wires()
        for wire in range(0, max(wires) + 1):
            name = "a" + str(wire)

            if wire in labels:
                label = labels[wire]
            else:
                label = str(wire)

            if wire in wire_colors:
                color = wire_colors[wire]
            else:
                color = "black"
            result += f"color={color} {name} W {label} \n"

        for wire in range(0, max(wires) + 1):
            result += f"a{wire} /\n"
        # add gates
        state = CircuitState(max(wires))
        result += self.render(state, style) + "\n"

        if filename is not None:
            extendedFilename = filename
            if not extendedFilename.endswith(".qpic"):
                extendedFilename = filename + ".qpic"
            if filepath is not None:
                extendedFilename = path.join(filepath, extendedFilename)
            with open(extendedFilename, "w") as file:
                file.write(result)

    def export_to(self,filename: str,**kwargs):
        """
        Shortcut to render this gate/circuit into a complete qpic/png/pdf document.
        """

        filename_tmp = filename.split(".")
        if 'filepath' in kwargs:
            filepath=kwargs['filepath']
            kwargs.pop('filepath')
        else: filepath = None
        if len(filename_tmp) == 1:
            ftype = "pdf"
            fname = filename
        else:
            ftype = filename_tmp[-1]
            fname = "".join(filename_tmp[:-1])
        if ftype == 'qpic':
            self.export_qpic(fname,**kwargs)
        elif ftype == 'pdf':
            self.export_qpic(fname,**kwargs)
            graphical.qpic_to_pdf(filename=fname,filepath=filepath)
        elif ftype == 'png':
            self.export_qpic(fname,**kwargs)
            graphical.qpic_to_png(filename=fname,filepath=filepath)
        else:
            raise tq.TequilaException(f'Extension {ftype} not supported directly. Try exporting to qpic and compiling to {ftype} yourself')
    @abc.abstractmethod
    def used_wires(self) -> List[int]:
        """
        This method should return all qubits that are used by this gate.
        """
        pass

    @abc.abstractmethod
    def dagger(self) -> "Gate":
        """
        This method should return the daggered gate.
        """
        pass

    def __str__(self):
        res = type(self).__name__
        U = self.used_wires()
        res += f': qubits {U}'
        return res
    
    def _parse_CircuitStyle(self,style:dict={},color_dict:dict={})->Tuple[CircuitStyle,dict]:
        def hex_to_three_letters(hex_color: str) -> str:
            """
            Deterministically converts a 6-digit hex color code to a 3-letter string.

            The string is generated by:
            1. Parsing the Red, Green, and Blue components.
            2. Using the value of each component to index into a custom character set.

            Args:
                hex_color: A string representing the hex color (e.g., '#FF5733' or 'FF5733').

            Returns:
                A 3-letter string derived from the hex color.
            """
            # 1. Clean and validate the input hex
            hex_color = hex_color.lstrip('#').upper()
            if len(hex_color) != 6:
                raise ValueError("Hex color must be 6 digits (e.g., FF5733).")

            # 2. Define the character pools (vowels/consonants/digits for variety)
            # The length of each pool must evenly divide 256 (the maximum R, G, or B value).
            # 256 / 8 = 32. This gives 32 possible indices for each pool.
            POOL_1 = "AEIOULMN"  # Vowels and common liquids/nasals
            POOL_2 = "BCDFGHKR"  # Common consonants
            POOL_3 = "PSTWXYZ1"  # Other consonants and a digit

            # Combine the pools for easy indexing (256/8 = 32 indices, so pool length should be 8)
            character_pools = [POOL_1, POOL_2, POOL_3]

            # 3. Split the hex into R, G, B components
            red_hex = hex_color[0:2]
            green_hex = hex_color[2:4]
            blue_hex = hex_color[4:6]

            hex_components = [red_hex, green_hex, blue_hex]
            
            result_letters = []

            # 4. Process each component
            for i in range(3):
                # Convert the 2-digit hex component to an integer (0-255)
                value = int(hex_components[i], 16)
                
                # Calculate the index using the modulo operator based on the pool size (8)
                # 0-255 mapped to 0-7
                index = value % len(character_pools[i])
                
                # Select the character
                letter = character_pools[i][index]
                result_letters.append(letter)

            return "".join(result_letters)
        def hex_to_rgb(hex:str)->Tuple:
            rgb = []
            for i in (0, 2, 4):
                decimal = int(hex[i:i+2], 16)/255
                rgb.append(decimal)
            return tuple(rgb)
        def parse_color(color:Union[Color,Tuple,List,str],color_dict:dict={})->Tuple[Color,dict]:
            if isinstance(color,Color):
                return color
            if isinstance(color,str):
                if len(color) and (len(color)==6 or color[0]=='#'): #it is a Hex RBG  
                    if color[0]=='#':
                        (r,g,b) = hex_to_rgb(color[1:])
                        name = hex_to_three_letters(color)
                    else:
                        (r,g,b) = hex_to_rgb(color)
                        name = hex_to_three_letters('#'+color)
                    color_dict[name]=RGB(r,g,b)
                    color = Color(name)
                else:
                    color = Color(color)
            elif isinstance(color,Union[List,Tuple]):
                assert len(color)==3, f"Three RGB components expected, received {len(color)}"
                color = RGB(color[0],color[1],color[2])
            else: raise Exception('No color format indified')
            return color,color_dict
        circut_style = CircuitStyle()
        if 'group_together' in style.keys():
            circut_style.group_together = style['group_together']
            style.pop('group_together')
        if 'parametrized' in style.keys():
            parametrized = style['parametrized']
            parametrized,color_dict = parse_color(parametrized,color_dict)
            if 'parametrized_text' in style:
                parametrized_text = style['parametrized_text']
                style.pop('parametrized_text')
            else: parametrized_text = 'black'
            parametrized_text,color_dict = parse_color(parametrized_text,color_dict)
            circut_style.parametrized_marking = Colors(parametrized_text,parametrized)
            style.pop('parametrized')
        if 'color_range' in style.keys():
            color_range = style['color_range']
            if isinstance(color_range,Union[List,Tuple]):
                if not len(color_range)==2:
                    raise Exception("Only 2 colors accepted")
                c0,color_dict = parse_color(color_range[0],color_dict)
                c1,color_dict = parse_color(color_range[1],color_dict)
                color_range = ColorRange(c0,c1)
            circut_style.color_range = color_range
            style.pop('color_range')
        if 'primary' in style.keys():
            primary = style['primary']
            primary,color_dict = parse_color(primary,color_dict)
            if 'primary_text' in style:
                primary_text = style['primary_text']
                style.pop('primary_text')
            else: primary_text = 'while'
            primary_text,color_dict = parse_color(primary_text,color_dict)
            circut_style.primary = Colors(primary_text,primary)
            style.pop('primary')
        if 'secondary' in style.keys():
            secondary = style['secondary']
            secondary,color_dict = parse_color(secondary,color_dict)
            if 'secondary_text' in style:
                secondary_text = style['secondary_text']
                style.pop('secondary_text')
            else: primary_text = 'while'
            secondary_text,color_dict = parse_color(secondary_text,color_dict)
            circut_style.secondary = Colors(secondary_text,secondary)
            style.pop('secondary')
        per = {}
        if 'personalize' in style.keys():
            personalize = style['personalize']
            style.pop('personalize')
            for gate_name,gate_color in personalize.items():
                gate_color,color_dict = parse_color(gate_color,color_dict)
                per[gate_name] = gate_color
        possibes = ['double','single','generic','UC','UR']
        for g in possibes:
            if g in style.keys():
                gate_color,color_dict = parse_color(style[g],color_dict)
                style.pop(g)
                if g+'_text' in style.keys():
                    text_color,color_dict = parse_color(style[g+'_text'],color_dict)
                else: text_color = Color('white')
                per[g] = Colors(text_color,gate_color)
        circut_style.personalized =  per
        return circut_style,color_dict