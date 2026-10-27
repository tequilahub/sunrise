import abc


class Shape(abc.ABC):
    """Abstract base class for all qpic shapes."""

    @abc.abstractmethod
    def to_qpic(self) -> str:
        """Returns the qpic string identifying this shape."""

    def __format__(self, format_spec) -> str:
        return self.to_qpic()


class NoBoundary(Shape):
    """This draws a gate without a boundary."""

    def to_qpic(self):
        return "0"


class Control(Shape):
    """This draws the gate as a control. It can optionally be drawn as a negated control."""

    def __init__(self, negated: bool = False):
        self.negated = negated

    def to_qpic(self):
        if self.negated:
            return "-1"
        else:
            return "1"


class Circle(Shape):
    """This draws a circle."""

    def to_qpic(self):
        return "2"


class Polygon(Shape):
    """
    This draws any polygon with n >= 3 vertices.
    If bottom_vertex is true, odd numbered polygons will draw their tip at the bottom instead of at the top. (Basically rotating by 180°)
    """

    def __init__(self, vertices: int, bottom_vertex: bool = False):
        if vertices < 0:
            vertices = vertices * -1
            bottom_vertex = not bottom_vertex

        assert vertices >= 3
        self.vertices = vertices
        self.bottom_vertex = bottom_vertex

    def to_qpic(self):
        if self.bottom_vertex:
            return str(-self.vertices)
        else:
            return str(self.vertices)


class Triangle(Shape):
    """This draws a triangle pointing to the right, or left when backward is True."""

    def __init__(self, backward: bool = False):
        self.backward = backward

    def to_qpic(self):
        if self.backward:
            return "<"
        else:
            return ">"
