"""Export molecular orbitals, density and electrostatic potential as cube files."""

from __future__ import annotations

import sys

from numpy import ix_, ndarray, zeros
from pyscf import gto, scf
from pyscf.tools import cubegen

from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase
from sunrise.utils.giuseppe import giuseppe_bar


def plot_MO(
    molecule: QuantumChemistryBase,
    filename: str | None = None,
    orbital: list[int] | None = None,
    use_active: bool = True,
    print_orbital: bool = True,
    density: bool = False,
    mep: bool = False,
    rdm1: ndarray | None = None,
    exclude_core: bool = False,
):
    """Save molecular orbitals, electron density or electrostatic potential as cube files.

    Orbitals are written to ``<label>_<filename>_MO.cube``, the density to
    ``<filename>_density.cube`` and the potential to ``<filename>_mep.cube``.

    Parameters
    ----------
    molecule : QuantumChemistryBase
        Molecule to take the orbitals from.
    filename : str, optional
        Base name of the cube files. Default is ``<name>-<basis>-<orbital type>``.
    orbital : list of int, optional
        Orbital indices to save. Default is all active orbitals if ``use_active``
        else all orbitals. With ``use_active`` the indices refer to the active
        orbitals (``active_orbitals[x].idx``), otherwise to all orbitals (``idx_total``).
    use_active : bool, optional
        Plot only the active orbitals. If ``orbital`` is given, its indices refer to the
        active orbitals (``mol.integral_manager.active_orbitals[x].idx`` instead of
        ``idx_total``). Default is True.
    print_orbital : bool, optional
        Whether to write the orbital cube files. Default is True.
    density : bool, optional
        Whether to write the electron density. Default is False.
    mep : bool, optional
        Whether to write the molecular electrostatic potential. Default is False.
    rdm1 : numpy.ndarray, optional
        Custom 1-RDM in the MO basis, of active-space or full size, used for the
        density and potential. Default is the RHF density.
    exclude_core : bool, optional
        If ``rdm1`` has active-space size, whether to leave out the frozen occupied
        orbitals (otherwise they are added with occupation 2). Default is False.

    Examples
    --------
    >>> sun.plot_MO(molecule=mol, filename="N2_CLPO")
    """

    if filename is None:
        filename = (
            molecule.parameters.name
            + "-"
            + molecule.integral_manager._basis_name
            + "-"
            + molecule.integral_manager._orbital_type
        )
    if orbital is None and use_active:
        orbital = [i.idx_total for i in molecule.integral_manager.active_orbitals]
        label = [i.idx for i in molecule.integral_manager.active_orbitals]
    elif orbital is None and not use_active:
        orbital = [i.idx_total for i in molecule.integral_manager.orbitals]
        label = orbital
    elif orbital is not None and use_active:
        d = {i.idx: i.idx_total for i in molecule.integral_manager.orbitals}
        label = orbital.copy()
        orbital = [d[i] for i in orbital]
    else:
        label = orbital

    pmol = gto.Mole()
    pmol.build(
        atom=molecule.parameters.geometry,
        basis=molecule.parameters.basis_set,
        charge=molecule.parameters.charge,
        verbose=0,
    )
    if density or mep:
        mf = scf.RHF(pmol).run()
        if rdm1 is None:
            rdm1 = mf.make_rdm1(mo_coeff=molecule.integral_manager.orbital_coefficients)
        else:
            mo_coeff = molecule.integral_manager.orbital_coefficients
            if (
                rdm1.shape[0] != molecule.integral_manager.orbital_coefficients.shape[1]
            ):  # already provided on frozen_core = False
                assert rdm1.shape[0] == molecule.n_orbitals, (
                    f"RDM1 provided with unexpected shape ({rdm1.shape}), expected either the number of active orbitals ({molecule.n_orbitals})\n or the number of total orbitals ({molecule.integral_manager.orbital_coefficients.shape[1]})"
                )
                rdm = zeros(
                    shape=(mo_coeff.shape[1], mo_coeff.shape[1])
                )  # rectangular mo_coeffs
                if not exclude_core:
                    for i in (
                        molecule.integral_manager.active_space.frozen_reference_orbitals
                    ):
                        rdm[i, i] = (
                            2  # NOTE: Experimental. Including contribution only from the active orbitals, expected to be more useful on density than mep
                        )
                rdm[
                    ix_(
                        molecule.integral_manager.active_space.active_orbitals,
                        molecule.integral_manager.active_space.active_orbitals,
                    )
                ] = rdm1
                rdm1 = rdm
            rdm1 = (
                molecule.integral_manager.orbital_coefficients
                @ rdm1
                @ molecule.integral_manager.orbital_coefficients.T
            )
    if print_orbital:
        for i, idx in enumerate(orbital):
            giuseppe_bar(step=i, total_steps=len(orbital))
            cubegen.orbital(
                pmol,
                str(label[i]) + "_" + filename + "_MO.cube",
                molecule.integral_manager.orbital_coefficients[:, idx],
            )
        giuseppe_bar(step=i + 1, total_steps=len(orbital))
        sys.stdout.write("\n")
        sys.stdout.flush()
    if density:
        cubegen.density(pmol, filename + "_density.cube", rdm1)
    if mep:
        cubegen.mep(pmol, filename + "_mep.cube", rdm1)
