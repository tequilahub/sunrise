"""Molecules in maximally entangled atomic orbitals (MEAO), through the original MEAO code."""

from .pyscf_meao_interface import generate_MEAO_molecule

__all__ = [
    "generate_MEAO_molecule",
]
