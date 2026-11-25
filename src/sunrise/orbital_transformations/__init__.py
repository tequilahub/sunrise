"""Orbital bases of a molecule: Löwdin (native), CLPO/HAO and MEAO orbitals.

Each basis lives in its own subpackage. It is available as a molecule method and by name in
``mol.use_orbs``:

- ``loewdin``: ``mol.use_native_orbitals()``, ``mol.use_orbs("loewdin")`` or ``mol.use_orbs("native")``.
- ``CLPO``: ``mol.use_CLPO_orbitals()`` and ``mol.use_CLPO_orbitals_and_edges()``, ``mol.use_orbs("CLPO")``.
  The hybrid atomic orbitals of the same JANPA run are ``mol.use_HAO_orbitals()``, ``mol.use_orbs("HAO")``.
- ``MEAO``: ``mol.use_MEAO_orbitals()``, ``mol.use_orbs("MEAO")``.
"""

from sunrise.orbital_transformations import CLPO, MEAO, loewdin

__all__ = ["CLPO", "MEAO", "loewdin"]
