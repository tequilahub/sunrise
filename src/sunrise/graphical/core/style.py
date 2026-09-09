from typing import Tuple, Union, List
from .state import CircuitStyle
from .color import Color, Colors, ColorRange, RGB

def hex_to_three_letters(hex_color: str) -> str:
    hex_color = hex_color.lstrip('#').upper()
    if len(hex_color) != 6:
        raise ValueError("Hex color must be 6 digits (e.g., FF5733).")
    POOL_1 = "AEIOULMN"
    POOL_2 = "BCDFGHKR"
    POOL_3 = "PSTWXYZ1"
    character_pools = [POOL_1, POOL_2, POOL_3]
    red_hex = hex_color[0:2]
    green_hex = hex_color[2:4]
    blue_hex = hex_color[4:6]
    hex_components = [red_hex, green_hex, blue_hex]
    result_letters = []
    for i in range(3):
        value = int(hex_components[i], 16)
        index = value % len(character_pools[i])
        letter = character_pools[i][index]
        result_letters.append(letter)
    return "".join(result_letters)

def hex_to_rgb(hex_str: str) -> Tuple:
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex_str[i:i+2], 16) / 255
        rgb.append(decimal)
    return tuple(rgb)

def parse_color(color: Union[Color, Tuple, List, str], color_dict: dict = {}) -> Tuple[Color, dict]:
    if isinstance(color, Color):
        return color, color_dict
    if isinstance(color, str):
        if len(color) and (len(color) == 6 or color[0] == '#'):
            if color[0] == '#':
                (r, g, b) = hex_to_rgb(color[1:])
                name = hex_to_three_letters(color)
            else:
                (r, g, b) = hex_to_rgb(color)
                name = hex_to_three_letters('#' + color)
            color_dict[name] = RGB(r, g, b)
            color = Color(name)
        else:
            color = Color(color)
    elif isinstance(color, (List, Tuple)):
        assert len(color) == 3, f"Three RGB components expected, received {len(color)}"
        color = RGB(color[0], color[1], color[2])
    else:
        raise Exception('No color format indicated')
    return color, color_dict

def parse_circuit_style(style: dict = {}, color_dict: dict = {}) -> Tuple[CircuitStyle, dict]:
    style = dict(style) if style else {}
    color_dict = dict(color_dict) if color_dict else {}
    
    circuit_style = CircuitStyle()
    
    if 'group_together' in style:
        circuit_style.group_together = style['group_together']
        style.pop('group_together')
        
    if 'parametrized' in style:
        parametrized = style['parametrized']
        parametrized, color_dict = parse_color(parametrized, color_dict)
        if 'parametrized_text' in style:
            parametrized_text = style['parametrized_text']
            style.pop('parametrized_text')
        else:
            parametrized_text = 'black'
        parametrized_text, color_dict = parse_color(parametrized_text, color_dict)
        circuit_style.parametrized_marking = Colors(parametrized_text, parametrized)
        style.pop('parametrized')
        
    if 'color_range' in style:
        color_range = style['color_range']
        if isinstance(color_range, (List, Tuple)):
            if not len(color_range) == 2:
                raise Exception("Only 2 colors accepted")
            c0, color_dict = parse_color(color_range[0], color_dict)
            c1, color_dict = parse_color(color_range[1], color_dict)
            color_range = ColorRange(c0, c1)
        circuit_style.color_range = color_range
        style.pop('color_range')
        
    if 'primary' in style:
        primary = style['primary']
        primary, color_dict = parse_color(primary, color_dict)
        if 'primary_text' in style:
            primary_text = style['primary_text']
            style.pop('primary_text')
        else:
            primary_text = 'white'
        primary_text, color_dict = parse_color(primary_text, color_dict)
        circuit_style.primary = Colors(primary_text, primary)
        style.pop('primary')
        
    if 'secondary' in style:
        secondary = style['secondary']
        secondary, color_dict = parse_color(secondary, color_dict)
        if 'secondary_text' in style:
            secondary_text = style['secondary_text']
            style.pop('secondary_text')
        else:
            secondary_text = 'white'
        secondary_text, color_dict = parse_color(secondary_text, color_dict)
        circuit_style.secondary = Colors(secondary_text, secondary)
        style.pop('secondary')
        
    per = {}
    if 'personalize' in style:
        personalize = style['personalize']
        style.pop('personalize')
        for gate_name, gate_color in personalize.items():
            gate_color, color_dict = parse_color(gate_color, color_dict)
            per[gate_name] = gate_color
            
    possibles = ['double', 'single', 'generic', 'UC', 'UR']
    for g in possibles:
        if g in style:
            gate_color, color_dict = parse_color(style[g], color_dict)
            style.pop(g)
            if g + '_text' in style:
                text_color, color_dict = parse_color(style[g + '_text'], color_dict)
            else:
                text_color = Color('white')
            per[g] = Colors(text_color, gate_color)

    # Backwards-compatible aliases
    if "single" in per and "UR" not in per:
        per["UR"] = per["single"]
    if "double" in per and "UC" not in per:
        per["UC"] = per["double"]
    
    circuit_style.personalized = per
    return circuit_style, color_dict