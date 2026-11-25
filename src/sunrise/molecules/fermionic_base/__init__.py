"""Fermionic molecules: backend selection and the ``FerMolecule`` factory."""

from __future__ import annotations

import warnings

from tequila import TequilaWarning

from sunrise.molecules.qubit_base.chemistry_tools import ParametersQC

from .fer_base import FermionicBase
from .madness_interface import QuantumChemistryMadness

SUPPORTED_QCHEMISTRY_BACKENDS = ["base", "pyscf", "psi4", "madness"]
INSTALLED_QCHEMISTRY_BACKENDS = {
    "base": FermionicBase,
    "madness": QuantumChemistryMadness,
}

try:
    from .psi4_interface import QuantumChemistryPsi4

    INSTALLED_QCHEMISTRY_BACKENDS["psi4"] = QuantumChemistryPsi4
except ImportError:
    pass

try:
    from .pyscf_interface import QuantumChemistryPySCF

    INSTALLED_QCHEMISTRY_BACKENDS["pyscf"] = QuantumChemistryPySCF
except ImportError:
    pass


def FerMolecule(
    geometry: str | None = None,
    basis_set: str | None = None,
    orbital_type: str | None = None,
    backend: str | None = None,
    guess_wfn=None,
    name: str | None = None,
    *args,
    **kwargs,
) -> FermionicBase:
    r"""Create a fermionic molecule with the quantum chemistry backend that fits the arguments.

    Usually called through ``sun.Molecule(..., nature="fermionic")``. Without ``backend`` the integrals
    are taken from ``one_body_integrals`` and ``two_body_integrals`` if given ("base"), from MADNESS
    if no basis set is given, otherwise from Psi4 if installed, else from PySCF.

    Parameters
    ----------
    geometry : str, optional
        Geometry as string, or file name of an xyz file (with ``.xyz`` ending).
    basis_set : str, optional
        Basis set name as understood by Psi4 and PySCF, e.g. "sto-3g" (minimal), "6-31g",
        "6-31g*", "cc-pvdz", "cc-pvtz", "aug-cc-pvdz" or "def2-svp". None (or "madness", "mra",
        "pno") selects MADNESS, which needs no basis set.
    orbital_type : str, optional
        Orbitals to use: "hf" (default) or "native" with a basis set, "pno" with MADNESS.
    backend : str {"base", "psi4", "pyscf", "madness"}, optional
        Quantum chemistry backend. "base" needs ``one_body_integrals`` and ``two_body_integrals``.
    guess_wfn : psi4.core.Wavefunction or str, optional
        Psi4 wavefunction (or the file it is stored in) to start the SCF cycle from. Only for Psi4.
    name : str, optional
        Name of the molecule. Default is deduced from the geometry.
    *args, **kwargs
        Passed to the backend class. Entries of ``ParametersQC`` (e.g. ``charge``, ``frozen_core``)
        are used for the molecular parameters, or give ``parameters`` directly; ``basis`` is accepted
        as alias of ``basis_set``.

    Returns
    -------
    FermionicBase
        Molecule of the class of the chosen backend.

    Raises
    ------
    ValueError
        If the backend is not supported, neither basis set nor integrals are given, or ``guess_wfn``
        is used without Psi4.
    ImportError
        If the backend (or any backend) is not installed.

    Examples
    --------
    >>> import sunrise as sun
    >>> mol = sun.Molecule(geometry="H 0 0 0\nH 0 0 0.75", basis_set="sto-3g", nature="fermionic", backend="pyscf")
    """

    # failsafe for common mistake
    if "basis" in kwargs:
        warnings.warn(
            'called molecule with keyword "basis={0}" converting it to "basis_set={0}"'.format(
                kwargs["basis"]
            ),
            TequilaWarning,
        )
        if basis_set is not None:
            warnings.warn(
                f'did not convert as "basis_set={basis_set}" was already given',
                TequilaWarning,
            )
        basis_set = kwargs["basis"]

    keyvals = {}
    for k, v in kwargs.items():
        if k in ParametersQC.__dict__:
            keyvals[k] = v

    if "parameters" in kwargs:
        parameters = kwargs["parameters"]
        kwargs.pop("parameters")
    else:
        parameters = ParametersQC(
            name=name, geometry=geometry, basis_set=basis_set, multiplicity=1, **keyvals
        )

    integrals_provided = all(
        key in kwargs for key in ["one_body_integrals", "two_body_integrals"]
    )
    if integrals_provided and backend is None:
        backend = "base"
    if backend is None:
        if basis_set is None or basis_set.lower() in ["madness", "mra", "pno"]:
            backend = "madness"
            basis_set = "mra"
            parameters.basis_set = basis_set
            if orbital_type is not None and orbital_type.lower() not in [
                "pno",
                "mra-pno",
            ]:
                warnings.warn(
                    f"only PNOs supported as orbital_type without basis set. Setting to pno - You gave={orbital_type}",
                    TequilaWarning,
                )
            orbital_type = "pno"
        else:
            if orbital_type is not None and orbital_type.lower() not in [
                "hf",
                "native",
            ]:
                warnings.warn(
                    f"only hf and native supported as orbital_type with basis-set. Setting to hf - You gave={orbital_type}",
                    TequilaWarning,
                )
                orbital_type = "hf"
            if orbital_type is None:
                orbital_type = "hf"

            if "psi4" in INSTALLED_QCHEMISTRY_BACKENDS:
                backend = "psi4"
            elif "pyscf" in INSTALLED_QCHEMISTRY_BACKENDS:
                backend = "pyscf"
            else:
                raise ImportError(
                    "No quantum chemistry backends installed on your system"
                )

    elif backend == "base":
        if not integrals_provided:
            raise ValueError(
                "No quantum chemistry backends installed on your system\n"
                "To use the base functionality you need to pass the following tensors via keyword\n"
                "one_body_integrals, two_body_integrals\n"
            )
        else:
            backend = "base"

    if backend not in SUPPORTED_QCHEMISTRY_BACKENDS:
        raise ValueError(str(backend) + " is not (yet) supported by Fermionic Molecule")
    if backend not in INSTALLED_QCHEMISTRY_BACKENDS:
        raise ImportError(str(backend) + " was not found on your system")

    if guess_wfn is not None and backend != "psi4":
        raise ValueError("guess_wfn only works for psi4")

    if (
        basis_set is None
        and backend.lower() not in ["base", "madness"]
        and not integrals_provided
    ):
        raise ValueError(f"no basis_set or integrals provided for backend={backend}")
    elif basis_set is None:
        basis_set = "custom"
        parameters.basis_set = basis_set

    return INSTALLED_QCHEMISTRY_BACKENDS[backend.lower()](
        *args,
        parameters=parameters,
        orbital_type=orbital_type,
        guess_wfn=guess_wfn,
        **kwargs,
    )
