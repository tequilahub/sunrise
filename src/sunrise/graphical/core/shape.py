"""Gate shapes of the qpic circuit renderer."""

import abc


class Shape(abc.ABC):
    """Abstract base class for all qpic shapes."""

    @abc.abstractmethod
    def to_qpic(self) -> str:
        """Return the qpic code of this shape."""

    def __format__(self, format_spec) -> str:
        return self.to_qpic()


class NoBoundary(Shape):
    """Draw a gate without a boundary."""

    def to_qpic(self):
        return "0"


class Control(Shape):
    """Draw the gate as a control, optionally negated.

    Parameters
    ----------
    negated : bool, optional
        If True, draw a negated control. Default is False.
    """

    def __init__(self, negated: bool = False):
        self.negated = negated

    def to_qpic(self):
        if self.negated:
            return "-1"
        else:
            return "1"


class Circle(Shape):
    """Draw a circle."""

    def to_qpic(self):
        return "2"


class Polygon(Shape):
    """Draw a regular polygon.

    Parameters
    ----------
    vertices : int
        Number of vertices, at least 3. A negative value is equivalent to the
        positive one with ``bottom_vertex`` toggled.
    bottom_vertex : bool, optional
        If True, polygons with an odd number of vertices point their tip down
        instead of up (rotation by 180 degrees). Default is False.
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
    """Draw a triangle pointing right, or left if ``backward`` is True.

    Parameters
    ----------
    backward : bool, optional
        Point left instead of right. Default is False.
    """

    def __init__(self, backward: bool = False):
        self.backward = backward

    def to_qpic(self):
        if self.backward:
            return "<"
        else:
            return ">"
