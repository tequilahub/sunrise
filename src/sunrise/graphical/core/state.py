from optparse import Option
from typing import List, Union, Optional,Dict

from .color import ColorRange, Colors, Color


class CircuitStyle:
    """
    This class defines the style that should be used when rendering the circuit.
    Parameters:
        group_together -- If true, this touches all wires after each gate, preventing gates from being rearranged.
        parametrized_marking -- If set the given Marking is used to color parametrized gates.
        color_range -- If set the given ColorRange is used to color parametrized gates according to their angle.

        primary, secondary -- Basic colors to be used by the gates, each contains a text and gate color.
    """

    def __init__(self,
                 group_together: bool = False,
                 parametrized_marking: Optional[Colors] = Colors(Color("black"), Color("unia")),
                 color_range: Optional[ColorRange] = ColorRange(Color("blue"), Color("red")),
                 primary: Colors = Colors(Color("white"), Color("tq")),
                 secondary: Colors = Colors(Color("white"), Color("fai")),
                 personalized:Dict['str',Colors]={}):
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
    def names(wires: Union[List[int], int]) -> str:
        """Generates the name of a single or multiple wires by their indices."""
        if wires is int:
            wires = [wires]

        return " ".join(f"a{i}" for i in wires)

    def all_names(self) -> str:
        """Generates names for all wires known to this state"""
        return self.names(self.wires)
