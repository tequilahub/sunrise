"""Style and state containers for the qpic circuit renderer."""

from __future__ import annotations

from .color import Color, ColorRange, Colors


class CircuitStyle:
    """Style used when rendering a circuit.

    Parameters
    ----------
    group_together : bool, optional
        If True, touch all wires after each gate so that qpic cannot rearrange
        gates. Default is False.
    parametrized_marking : Colors, optional
        Colors of parametrized gates. Default is black text on the "unia" color.
    color_range : ColorRange, optional
        Colors parametrized gates according to their angle. Default is blue to red.
    primary, secondary : Colors, optional
        Basic text and gate colors. Defaults are white on "tq" and white on "fai".
    personalized : dict of str to Colors, optional
        Colors for specific gate names (e.g. ``"UR"``, ``"UC"``). Default is empty.
    """

    def __init__(
        self,
        group_together: bool = False,
        parametrized_marking: Colors | None = None,
        color_range: ColorRange | None = None,
        primary: Colors | None = None,
        secondary: Colors | None = None,
        personalized: dict[str, Colors] | None = None,
    ):
        if parametrized_marking is None:
            parametrized_marking = Colors(Color("black"), Color("unia"))
        if color_range is None:
            color_range = ColorRange(Color("blue"), Color("red"))
        if primary is None:
            primary = Colors(Color("white"), Color("tq"))
        if secondary is None:
            secondary = Colors(Color("white"), Color("fai"))
        if personalized is None:
            personalized = {}
        self.group_together = group_together
        self.parametrized_marking = parametrized_marking
        self.color_range = color_range
        self.primary = primary
        self.secondary = secondary
        self.personalized = personalized


class CircuitState:
    """Current state of a circuit being rendered.

    Stores the number of orbitals and the current order of the wires, which
    changes in swap networks.

    Parameters
    ----------
    n_orbitals : int
        Number of orbitals (wires).
    """

    def __init__(self, n_orbitals: int):
        self.n_orbitals = n_orbitals
        self.wires = list(range(n_orbitals))

    @staticmethod
    def names(wires: list[int] | int | None) -> str:
        """Return the space-separated qpic names (``a<i>``) of the given wires.

        Parameters
        ----------
        wires : list of int
            Wire indices. A bare int is not handled (the check ``wires is int``
            never triggers), so pass a list.
        """
        if wires is int:
            wires = [wires]

        return " ".join(f"a{i}" for i in wires)

    def all_names(self) -> str:
        """Return the qpic names of all wires, in their current order."""
        return self.names(self.wires)
