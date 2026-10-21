from __future__ import annotations

from .color import Color, ColorRange, Colors


class CircuitStyle:
    """
    This class defines the style that should be used when rendering the circuit.
    Parameters:
        group_together -- If true, this touches all wires after each gate, preventing gates from being rearranged.
        parametrized_marking -- If set the given Marking is used to color parametrized gates.
        color_range -- If set the given ColorRange is used to color parametrized gates according to their angle.

        primary, secondary -- Basic colors to be used by the gates, each contains a text and gate color.
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
    """
    This class contains the current state of the circuit.
    This is mostly needed for swap networks, to store the order of the wires.
    It also stores the number of orbitals.
    """

    def __init__(self, n_orbitals: int):
        self.n_orbitals = n_orbitals
        self.wires = list(range(n_orbitals))

    @staticmethod
    def names(wires: list[int] | int | None) -> str:
        """Generates the name of a single or multiple wires by their indices."""
        if wires is int:
            wires = [wires]

        return " ".join(f"a{i}" for i in wires)

    def all_names(self) -> str:
        """Generates names for all wires known to this state"""
        return self.names(self.wires)
