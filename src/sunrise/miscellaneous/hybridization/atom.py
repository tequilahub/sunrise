from numpy import sqrt,array
from .utils import XYZUtils
from re import split


BOND_DISTANCE_TOLERANCE = 1.3

class AtomData:
    def __init__(self, name, symbol, covalent_radius_pm, max_bonds, number_of_shells, number_of_electrons,
                 electrons_by_shell):
        self.name = name
        self.symbol = symbol
        self.covalent_radius = None if covalent_radius_pm is None else covalent_radius_pm / 100
        self.max_bonds = max_bonds
        self.number_of_shells = number_of_shells
        self.number_of_electrons = number_of_electrons
        self.electrons_by_shell = electrons_by_shell

    @property
    def covalent_radius_pm(self):
        return self.covalent_radius * 100

atom_data_by_number = [
    AtomData("Hydrogen", "H", 31, 1, 1, 1, [1]),
    AtomData("Helium", "He", 28, 0, 1, 2, [2]),
    AtomData("Lithium", "Li", 128, 1, 2, 3, [2, 1]),
    AtomData("Beryllium", "Be", 96, 2, 2, 4, [2, 2]),
    AtomData("Boron", "B", 85, 3, 2, 5, [2, 3]),
    AtomData("Carbon", "C", 76, 4, 2, 6, [2, 4]),
    AtomData("Nitrogen", "N", 71, 3, 2, 7, [2, 5]),
    AtomData("Oxygen", "O", 66, 2, 2, 8, [2, 6]),
    AtomData("Fluorine", "F", 57, 1, 2, 9, [2, 7]),
    AtomData("Neon", "Ne", 58, 0, 2, 10, [2, 8]),
    AtomData("Sodium", "Na", 166, 1, 3, 11, [2, 8, 1]),
    AtomData("Magnesium", "Mg", 141, 2, 3, 12, [2, 8, 2]),
    AtomData("Aluminium", "Al", 121, 3, 3, 13, [2, 8, 3]),
    AtomData("Silicon", "Si", 111, 4, 3, 14, [2, 8, 4]),
    AtomData("Phosphorus", "P", 107, 5, 3, 15, [2, 8, 5]),
    AtomData("Sulfur", "S", 105, 2, 3, 16, [2, 8, 6]),
    AtomData("Chlorine", "Cl", 102, 1, 3, 17, [2, 8, 7]),
    AtomData("Argon", "Ar", 106, 0, 3, 18, [2, 8, 8]),
    AtomData("Potassium", "K", 203, 1, 4, 19, [2, 8, 8, 1]),
    AtomData("Calcium", "Ca", 176, 2, 4, 20, [2, 8, 8, 2]),
    AtomData("Scandium", "Sc", 170, 3, 4, 21, [2, 8, 9, 2]),
    AtomData("Titanium", "Ti", 160, 4, 4, 22, [2, 8, 10, 2]),
    AtomData("Vanadium", "V", 153, 5, 4, 23, [2, 8, 11, 2]),
    AtomData("Chromium", "Cr", 139, 6, 4, 24, [2, 8, 13, 1]),
    AtomData("Manganese", "Mn", 139, 7, 4, 25, [2, 8, 13, 2]),
    AtomData("Iron", "Fe", 132, 6, 4, 26, [2, 8, 14, 2]),
    AtomData("Cobalt", "Co", 126, 5, 4, 27, [2, 8, 15, 2]),
    AtomData("Nickel", "Ni", 124, 4, 4, 28, [2, 8, 16, 2]),
    AtomData("Copper", "Cu", 132, 1, 4, 29, [2, 8, 18, 1]),
    AtomData("Zinc", "Zn", 122, 2, 4, 30, [2, 8, 18, 2]),
    AtomData("Gallium", "Ga", 122, 3, 4, 31, [2, 8, 18, 3]),
    AtomData("Germanium", "Ge", 120, 4, 4, 32, [2, 8, 18, 4]),
    AtomData("Arsenic", "As", 119, 3, 4, 33, [2, 8, 18, 5]),
    AtomData("Selenium", "Se", 120, 2, 4, 34, [2, 8, 18, 6]),
    AtomData("Bromine", "Br", 120, 1, 4, 35, [2, 8, 18, 7]),
    AtomData("Krypton", "Kr", 116, 0, 4, 36, [2, 8, 18, 8]),
    AtomData("Rubidium", "Rb", 220, 1, 5, 37, [2, 8, 18, 8, 1]),
    AtomData("Strontium", "Sr", 195, 2, 5, 38, [2, 8, 18, 8, 2]),
    AtomData("Yttrium", "Y", 190, 3, 5, 39, [2, 8, 18, 9, 2]),
    AtomData("Zirconium", "Zr", 175, 4, 5, 40, [2, 8, 18, 10, 2]),
    AtomData("Niobium", "Nb", 164, 5, 5, 41, [2, 8, 18, 12, 1]),
    AtomData("Molybdenum", "Mo", 154, 6, 5, 42, [2, 8, 18, 13, 1]),
    AtomData("Technetium", "Tc", 147, 7, 5, 43, [2, 8, 18, 13, 2]),
    AtomData("Ruthenium", "Ru", 146, 8, 5, 44, [2, 8, 18, 15, 1]),
    AtomData("Rhodium", "Rh", 142, 9, 5, 45, [2, 8, 18, 16, 1]),
    AtomData("Palladium", "Pd", 139, 10, 5, 46, [2, 8, 18, 18]),
    AtomData("Silver", "Ag", 145, 1, 5, 47, [2, 8, 18, 18, 1]),
    AtomData("Cadmium", "Cd", 144, 2, 5, 48, [2, 8, 18, 18, 2]),
    AtomData("Indium", "In", 142, 3, 5, 49, [2, 8, 18, 18, 3]),
    AtomData("Tin", "Sn", 139, 4, 5, 50, [2, 8, 18, 18, 4]),
    AtomData("Antimony", "Sb", 139, 3, 5, 51, [2, 8, 18, 18, 5]),
    AtomData("Tellurium", "Te", 138, 2, 5, 52, [2, 8, 18, 18, 6]),
    AtomData("Iodine", "I", 139, 1, 5, 53, [2, 8, 18, 18, 7]),
    AtomData("Xenon", "Xe", 140, 0, 5, 54, [2, 8, 18, 18, 8]),
    AtomData("Caesium", "Cs", 244, 1, 6, 55, [2, 8, 18, 18, 8, 1]),
    AtomData("Barium", "Ba", 215, 2, 6, 56, [2, 8, 18, 18, 8, 2]),
    AtomData("Lanthanum", "La", 207, 3, 6, 57, [2, 8, 18, 18, 9, 2]),
    AtomData("Cerium", "Ce", 204, 4, 6, 58, [2, 8, 18, 19, 9, 2]),
    AtomData("Praseodymium", "Pr", 203, 4, 6, 59, [2, 8, 18, 21, 8, 2]),
    AtomData("Neodymium", "Nd", 201, 4, 6, 60, [2, 8, 18, 22, 8, 2]),
    AtomData("Promethium", "Pm", 199, 4, 6, 61, [2, 8, 18, 23, 8, 2]),
    AtomData("Samarium", "Sm", 198, 4, 6, 62, [2, 8, 18, 24, 8, 2]),
    AtomData("Europium", "Eu", 198, 4, 6, 63, [2, 8, 18, 25, 8, 2]),
    AtomData("Gadolinium", "Gd", 196, 4, 6, 64, [2, 8, 18, 25, 9, 2]),
    AtomData("Terbium", "Tb", 194, 4, 6, 65, [2, 8, 18, 27, 8, 2]),
    AtomData("Dysprosium", "Dy", 192, 4, 6, 66, [2, 8, 18, 28, 8, 2]),
    AtomData("Holmium", "Ho", 192, 4, 6, 67, [2, 8, 18, 29, 8, 2]),
    AtomData("Erbium", "Er", 189, 4, 6, 68, [2, 8, 18, 30, 8, 2]),
    AtomData("Thulium", "Tm", 190, 4, 6, 69, [2, 8, 18, 31, 8, 2]),
    AtomData("Ytterbium", "Yb", 187, 2, 6, 70, [2, 8, 18, 32, 8, 2]),
    AtomData("Lutetium", "Lu", 187, 3, 6, 71, [2, 8, 18, 32, 9, 2]),
    AtomData("Hafnium", "Hf", 175, 4, 6, 72, [2, 8, 18, 32, 10, 2]),
    AtomData("Tantalum", "Ta", 170, 5, 6, 73, [2, 8, 18, 32, 11, 2]),
    AtomData("Tungsten", "W", 162, 6, 6, 74, [2, 8, 18, 32, 12, 2]),
    AtomData("Rhenium", "Re", 151, 7, 6, 75, [2, 8, 18, 32, 13, 2]),
    AtomData("Osmium", "Os", 144, 8, 6, 76, [2, 8, 18, 32, 14, 2]),
    AtomData("Iridium", "Ir", 141, 9, 6, 77, [2, 8, 18, 32, 15, 2]),
    AtomData("Platinum", "Pt", 136, 10, 6, 78, [2, 8, 18, 32, 17, 1]),
    AtomData("Gold", "Au", 136, 1, 6, 79, [2, 8, 18, 32, 18, 1]),
    AtomData("Mercury", "Hg", 132, 2, 6, 80, [2, 8, 18, 32, 18, 2]),
    AtomData("Thallium", "Tl", 145, 3, 6, 81, [2, 8, 18, 32, 18, 3]),
    AtomData("Lead", "Pb", 146, 4, 6, 82, [2, 8, 18, 32, 18, 4]),
    AtomData("Bismuth", "Bi", 148, 5, 6, 83, [2, 8, 18, 32, 18, 5]),
    AtomData("Polonium", "Po", 140, 6, 6, 84, [2, 8, 18, 32, 18, 6]),
    AtomData("Astatine", "At", 150, 7, 6, 85, [2, 8, 18, 32, 18, 7]),
    AtomData("Radon", "Rn", 150, 8, 6, 86, [2, 8, 18, 32, 18, 8]),
    AtomData("Francium", "Fr", 260, 1, 7, 87, [2, 8, 18, 32, 18, 8, 1]),
    AtomData("Radium", "Ra", 221, 2, 7, 88, [2, 8, 18, 32, 18, 8, 2]),
    AtomData("Actinium", "Ac", 215, 3, 7, 89, [2, 8, 18, 32, 18, 9, 2]),
    AtomData("Thorium", "Th", 206, 4, 7, 90, [2, 8, 18, 32, 18, 10, 2]),
    AtomData("Protactinium", "Pa", 200, 5, 7, 91, [2, 8, 18, 32, 20, 9, 2]),
    AtomData("Uranium", "U", 196, 6, 7, 92, [2, 8, 18, 32, 21, 9, 2]),
    AtomData("Neptunium", "Np", 190, 5, 7, 93, [2, 8, 18, 32, 22, 9, 2]),
    AtomData("Plutonium", "Pu", 187, 6, 7, 94, [2, 8, 18, 32, 24, 8, 2]),
    AtomData("Americium", "Am", 180, 6, 7, 95, [2, 8, 18, 32, 25, 8, 2]),
    AtomData("Curium", "Cm", 169, 6, 7, 96, [2, 8, 18, 32, 25, 9, 2]),
    AtomData("Berkelium", "Bk", None, 4, 7, 97, [2, 8, 18, 32, 27, 8, 2]),
    AtomData("Californium", "Cf", None, 4, 7, 98, [2, 8, 18, 32, 28, 8, 2]),
    AtomData("Einsteinium", "Es", None, 4, 7, 99, [2, 8, 18, 32, 29, 8, 2]),
    AtomData("Fermium", "Fm", None, 4, 7, 100, [2, 8, 18, 32, 30, 8, 2]),
    AtomData("Mendelevium", "Md", None, 4, 7, 101, [2, 8, 18, 32, 31, 8, 2]),
    AtomData("Nobelium", "No", None, 4, 7, 102, [2, 8, 18, 32, 32, 8, 2]),
    AtomData("Lawrencium", "Lr", None, 3, 7, 103, [2, 8, 18, 32, 32, 8, 3]),
    AtomData("Rutherfordium", "Rf", None, 4, 7, 104, [2, 8, 18, 32, 32, 10, 2]),
    AtomData("Dubnium", "Db", None, 5, 7, 105, [2, 8, 18, 32, 32, 11, 2]),
    AtomData("Seaborgium", "Sg", None, 6, 7, 106, [2, 8, 18, 32, 32, 12, 2]),
    AtomData("Bohrium", "Bh", None, 7, 7, 107, [2, 8, 18, 32, 32, 13, 2]),
    AtomData("Hassium", "Hs", None, 8, 7, 108, [2, 8, 18, 32, 32, 14, 2]),
    AtomData("Meitnerium", "Mt", None, 9, 7, 109, [2, 8, 18, 32, 32, 15, 2]),
    AtomData("Darmstadtium", "Ds", None, 10, 7, 110, [2, 8, 18, 32, 32, 16, 2]),
    AtomData("Roentgenium", "Rg", None, 11, 7, 111, [2, 8, 18, 32, 32, 17, 2]),
    AtomData("Copernicium", "Cn", None, 12, 7, 112, [2, 8, 18, 32, 32, 18, 2]),
    AtomData("Nihonium", "Nh", None, 13, 7, 113, [2, 8, 18, 32, 32, 18, 3]),
    AtomData("Flerovium", "Fl", None, 14, 7, 114, [2, 8, 18, 32, 32, 18, 4]),
    AtomData("Moscovium", "Mc", None, 15, 7, 115, [2, 8, 18, 32, 32, 18, 5]),
    AtomData("Livermorium", "Lv", None, 16, 7, 116, [2, 8, 18, 32, 32, 18, 6]),
    AtomData("Tennessine", "Ts", None, 17, 7, 117, [2, 8, 18, 32, 32, 18, 7]),
    AtomData("Oganesson", "Og", None, 18, 7, 118, [2, 8, 18, 32, 32, 18, 8])
]
atom_data_by_symbol = {}

for atom in atom_data_by_number:
    atom_data_by_symbol[atom.symbol] = atom


class Position:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get_distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    @property
    def coords(self):
        return array([self.x, self.y, self.z])

    @staticmethod
    def from_string(string):
        parts = string.split()
        return Position(float(parts[1]), float(parts[2]), float(parts[3]))
class Atom(AtomData):
    def __init__(self, atom_data, position):
        super().__init__(atom_data.name, atom_data.symbol, atom_data.covalent_radius_pm, atom_data.max_bonds,
                         atom_data.number_of_shells, atom_data.number_of_electrons, atom_data.electrons_by_shell)
        self.position = position

    @property
    def coords(self):
        return self.position.coords

    def distance_to(self, other):
        return self.position.get_distance(other.position)

    def to_string(self):
        x, y, z = self.coords
        return f"{self.symbol} {x:.6f} {y:.6f} {z:.6f}"

    @staticmethod
    def from_string(string):
        parts = split(r'\s+', string.strip())
        symbol = parts[0]
        atom_data = atom_data_by_symbol[symbol]
        position = Position.from_string(string)
        return Atom(atom_data, position)

    @staticmethod
    def parse_xyz(string):
        lines = XYZUtils.cleanup(string).splitlines()
        return [Atom.from_string(line) for line in lines]

    def can_bond_with(self, other):
        if self.covalent_radius is None or other.covalent_radius is None:
            return False
        max_bonding_distance = BOND_DISTANCE_TOLERANCE * (self.covalent_radius + other.covalent_radius)
        return self.distance_to(other) <= max_bonding_distance