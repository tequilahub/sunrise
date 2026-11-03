"""Conversion of user-supplied color and style options into qpic style objects."""

from __future__ import annotations

from tequila import TequilaException

from .color import RGB, Color, ColorRange, Colors
from .state import CircuitStyle


def hex_to_three_letters(hex_color: str) -> str:
    """Map a hex color ``"#RRGGBB"`` to a three-letter name usable as a qpic color name."""
    hex_color = hex_color.lstrip("#").upper()
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


def hex_to_rgb(hex_str: str) -> tuple:
    """Convert a 6-digit hex string (no ``#``) to an ``(r, g, b)`` tuple in [0, 1]."""
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex_str[i : i + 2], 16) / 255
        rgb.append(decimal)
    return tuple(rgb)


def parse_color(
    color: Color | tuple | list | str, color_dict: dict | None = None
) -> tuple[Color, dict]:
    """Convert a user color specification to a ``Color``.

    Hex strings (``"#RRGGBB"`` or ``"RRGGBB"``) are registered in ``color_dict``
    under a generated name; other strings are taken as qpic color names.

    Parameters
    ----------
    color : Color, str, tuple or list
        Color to convert. A tuple or list must hold three RGB components.
    color_dict : dict, optional
        Custom color definitions (name to RGB), extended in place.

    Returns
    -------
    tuple
        The color and the updated ``color_dict``.

    Raises
    ------
    TequilaException
        If the color format is not recognized.
    """
    if color_dict is None:
        color_dict = {}
    if isinstance(color, Color):
        return color, color_dict
    if isinstance(color, str):
        if len(color) and (len(color) == 6 or color[0] == "#"):
            if color[0] == "#":
                (r, g, b) = hex_to_rgb(color[1:])
                name = hex_to_three_letters(color)
            else:
                (r, g, b) = hex_to_rgb(color)
                name = hex_to_three_letters("#" + color)
            color_dict[name] = RGB(r, g, b)
            color = Color(name)
        else:
            color = Color(color)
    elif isinstance(color, (list, tuple)):
        assert len(color) == 3, f"Three RGB components expected, received {len(color)}"
        color = RGB(color[0], color[1], color[2])
    else:
        raise TequilaException("No color format indicated")
    return color, color_dict


def parse_circuit_style(
    style: dict | None = None, color_dict: dict | None = None
) -> tuple[CircuitStyle, dict]:
    """Build a ``CircuitStyle`` from a plain dictionary of options.

    Recognized keys are ``group_together``, ``parametrized``, ``color_range``,
    ``primary``, ``secondary`` (each with an optional ``<key>_text`` color),
    ``personalize`` (gate name to color) and per-gate colors ``double``,
    ``single``, ``generic``, ``UC``, ``UR`` (``single`` and ``double`` are
    aliases of ``UR`` and ``UC``). Colors can be anything accepted by ``parse_color``.

    Parameters
    ----------
    style : dict, optional
        Style options. Not modified.
    color_dict : dict, optional
        Existing custom color definitions (name to RGB).

    Returns
    -------
    tuple
        The style and the color dictionary including newly defined colors.

    Raises
    ------
    TequilaException
        If ``color_range`` does not hold exactly two colors.
    """
    if style is None:
        style = {}
    if color_dict is None:
        color_dict = {}
    style = dict(style) if style else {}
    color_dict = dict(color_dict) if color_dict else {}

    circuit_style = CircuitStyle()

    if "group_together" in style:
        circuit_style.group_together = style["group_together"]
        style.pop("group_together")

    if "parametrized" in style:
        parametrized = style["parametrized"]
        parametrized, color_dict = parse_color(parametrized, color_dict)
        if "parametrized_text" in style:
            parametrized_text = style["parametrized_text"]
            style.pop("parametrized_text")
        else:
            parametrized_text = "black"
        parametrized_text, color_dict = parse_color(parametrized_text, color_dict)
        circuit_style.parametrized_marking = Colors(parametrized_text, parametrized)
        style.pop("parametrized")

    if "color_range" in style:
        color_range = style["color_range"]
        if isinstance(color_range, (list, tuple)):
            if not len(color_range) == 2:
                raise TequilaException("Only 2 colors accepted")
            c0, color_dict = parse_color(color_range[0], color_dict)
            c1, color_dict = parse_color(color_range[1], color_dict)
            color_range = ColorRange(c0, c1)
        circuit_style.color_range = color_range
        style.pop("color_range")

    if "primary" in style:
        primary = style["primary"]
        primary, color_dict = parse_color(primary, color_dict)
        if "primary_text" in style:
            primary_text = style["primary_text"]
            style.pop("primary_text")
        else:
            primary_text = "white"
        primary_text, color_dict = parse_color(primary_text, color_dict)
        circuit_style.primary = Colors(primary_text, primary)
        style.pop("primary")

    if "secondary" in style:
        secondary = style["secondary"]
        secondary, color_dict = parse_color(secondary, color_dict)
        if "secondary_text" in style:
            secondary_text = style["secondary_text"]
            style.pop("secondary_text")
        else:
            secondary_text = "white"
        secondary_text, color_dict = parse_color(secondary_text, color_dict)
        circuit_style.secondary = Colors(secondary_text, secondary)
        style.pop("secondary")

    per = {}
    if "personalize" in style:
        personalize = style["personalize"]
        style.pop("personalize")
        for gate_name, gate_color in personalize.items():
            gate_color, color_dict = parse_color(gate_color, color_dict)
            per[gate_name] = gate_color

    possibles = ["double", "single", "generic", "UC", "UR"]
    for g in possibles:
        if g in style:
            gate_color, color_dict = parse_color(style[g], color_dict)
            style.pop(g)
            if g + "_text" in style:
                text_color, color_dict = parse_color(style[g + "_text"], color_dict)
            else:
                text_color = Color("white")
            per[g] = Colors(text_color, gate_color)

    # Backwards-compatible aliases
    if "single" in per and "UR" not in per:
        per["UR"] = per["single"]
    if "double" in per and "UC" not in per:
        per["UC"] = per["double"]

    circuit_style.personalized = per
    return circuit_style, color_dict
