from numbers import Number


class RGB:
    """This class defines a color by its RGB values."""

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
    """This class is a reference to a defined color.
    Colors can be defined by passing their RGB representation to the export_qpic method."""

    def __init__(self, name: str):
        self.name = name

    def __format__(self, format_spec):
        return self.name


class ColorRange:
    """This class holds two colors and can interpolate between them when given an angle."""

    def __init__(self, start: Color = Color("blue"), end: Color = Color("red")):
        self.start = start
        self.end = end

    def interpolate(self, angle: Number) -> str:
        """Returns the matching qpic string to interpolate between the colors."""
        return f"{self.start}!{angle}!{self.end}"


class Colors:
    """Contains both a text and gate color.
    The gate color is used for a gates background."""

    def __init__(self, text_color: Color, gate_color: Color):
        self.text_color = text_color
        self.gate_color = gate_color
