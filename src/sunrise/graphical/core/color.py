"""Color types used by the qpic circuit renderer."""

from __future__ import annotations

from numbers import Number


class RGB:
    """Color defined by RGB values.

    Parameters
    ----------
    r, g, b : float
        Red, green and blue components in the range [0, 1].
    """

    r: float
    g: float
    b: float

    def __init__(self, r: float, g: float, b: float):
        self.r = r
        self.g = g
        self.b = b


# these colors are added by default and can thus always be references
DefaultColors: dict[str, RGB] = {
    "tq": RGB(0.03137254901960784, 0.1607843137254902, 0.23921568627450981),
    "guo": RGB(0.988, 0.141, 0.757),
    "unia": RGB(0.678, 0.0, 0.486),
    "fai": RGB(0.282, 0.576, 0.141),
}


class Color:
    """Reference to a named color.

    The RGB definition of a custom color is passed separately to the qpic export;
    names in ``DefaultColors`` are always available.

    Parameters
    ----------
    name : str
        Color name as understood by qpic (e.g. ``"red"`` or a defined custom name).
    """

    def __init__(self, name: str):
        self.name = name

    def __format__(self, format_spec):
        return self.name


class ColorRange:
    """Pair of colors between which a gate color is interpolated by angle.

    Parameters
    ----------
    start, end : Color, optional
        End points of the range. Defaults are blue and red.
    """

    def __init__(self, start: Color | None = None, end: Color | None = None):
        if start is None:
            start = Color("blue")
        if end is None:
            end = Color("red")
        self.start = start
        self.end = end

    def interpolate(self, angle: Number) -> str:
        """Return the qpic color expression ``start!angle!end``.

        Parameters
        ----------
        angle : numbers.Number
            Mixing percentage of ``end`` as used by qpic (0 to 100).

        Returns
        -------
        str
            The qpic color string.
        """
        return f"{self.start}!{angle}!{self.end}"


class Colors:
    """Text color and gate (background) color of a gate.

    Parameters
    ----------
    text_color : Color
        Color of the gate label.
    gate_color : Color
        Background color of the gate.
    """

    def __init__(self, text_color: Color, gate_color: Color):
        self.text_color = text_color
        self.gate_color = gate_color
