from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import tempfile
import typing
import warnings

import h5py
import numpy
from tequila import ExpectationValue

from .chemistry_tools import NBodyTensor, OrbitalData, ParametersQC
from .qc_base import (
    QCircuit,
    QuantumChemistryBase,
    TequilaException,
    TequilaWarning,
    gates,
)

# frayedends is only imported in the worker process, but the backend is only registered if it is installed
if importlib.util.find_spec("frayedends") is None:
    raise ImportError("frayedends is not installed")

# basis_set names the Molecule factories give to a molecule without basis: MRA-PNOs
_PNO_BASIS_NAMES = ("custom", "mra", "madness", "pno", "frayed", "frayedends")
_WORLD_OPTIONS = [
    "L",
    "k",
    "thresh",
    "initial_level",
    "truncate_mode",
    "refine",
    "n_threads",
]

# Examples how to initialize the madness backend
# sun.Molecule(geometry="...") will initialize a molecule with n_qubits=n_electrons (n_electrons//2 HF orbitals + n_electrons//2 PNOs)
# sun.Molecule(geometry="...", n_pno="read") will read in the molecule saved by a previous calculation (name is auto-generated from geometry)
# sun.Molecule(geometry="...", name="X", n_pno="read") will read in X_pno.h5
# sun.Molecule(geometry="...", name="X", datadir="asd/Y/", n_pno="read") reads in asd/Y/X_pno.h5
# control madness input sections with dictionaries
# sun.Molecule(geometry="...", pno={"maxrank":10, "freeze":0}, dft={"k":9, "L":25.0})
# compute more orbitals
# sun.Molecule(geometry="...", n_pno=10) # computes 10 PNOs additional to the occupied HF orbitals


class SunriseFrayedEndsException(TequilaException):
    def __str__(self):
        return "Error in FrayedEnds backend:" + self.message


class SunriseFrayedEndWarning(TequilaWarning):
    def __str__(self):
        return "Warning in FrayedEnds backend:" + self.message


class FrayedEndsMixin:
    """
    MRA-PNO functionality shared by the qubit, fermionic and hybrid madness molecules
    Only needs self.orbitals and self.integral_manager (with the pair information of the PNOs)
    Put it first in the bases so that it overrides the methods of the backend base class
    """

    def __init__(
        self,
        parameters: ParametersQC,
        *args,
        mra_orbitals: list[bytes] | None = None,
        mra_parameters: dict | None = None,
        n_pno: int | str | None = None,
        datadir: str | None = None,
        **kwargs,
    ):
        orbitals = None
        pno_file = None
        if "one_body_integrals" not in kwargs:
            world_kwargs = {k: kwargs.pop(k) for k in _WORLD_OPTIONS if k in kwargs}
            # sun.Molecule turns a missing basis_set into "custom"
            if (
                parameters.basis_set is None
                or parameters.basis_set.lower() in _PNO_BASIS_NAMES
            ):
                orbitals, mra_orbitals, mra_parameters, pno_file = self._pno_data(
                    parameters, n_pno, datadir, world_kwargs, kwargs
                )
            else:
                # the madness part runs in the frayedends worker, no madness object is created in this process
                from ._frayed_worker import run_frayed

                integral_kwargs = {
                    k: kwargs.pop(k)
                    for k in ("truncation_tol", "coulomb_lo", "coulomb_eps")
                    if k in kwargs
                }
                try:
                    result = run_frayed(
                        {
                            "task": "project",
                            "world": world_kwargs,
                            "integrals": integral_kwargs,
                            # the projector takes the geometry as string (or file), not as MolecularGeometry
                            "geometry": parameters.get_geometry_string(),
                            "basis": parameters.basis_set,
                            "silent": kwargs.get("silent", True),
                        }
                    )
                except RuntimeError as e:
                    if "Failed loading basis set" not in str(e):
                        raise
                    raise SunriseFrayedEndsException(
                        f"madness could not load the AO basis '{parameters.basis_set}' (the file name, case sensitive).\n"
                        "madness reads it from $MRA_CHEMDATA_DIR if set, otherwise from madness_source/src/madness/chem of its build.\n"
                        "Installed by default: sto-3g, sto-6g, 6-31g. Others from madness_source/src/madness/chem "
                        "(e.g. 3-21g, 6-31gss, aug-cc-pvdz) have to be copied into $MRA_CHEMDATA_DIR.\n"
                        "Each basis file only covers some elements, e.g. aug-cc-pvdz up to Ar."
                    ) from e
                # integrals in the projected (MRA) AO basis, like the gaussian AO integrals of the pyscf backend
                S, h_ao, g_ao = result["S"], result["h"], result["g"]
                if "nuclear_repulsion" not in kwargs:
                    kwargs["nuclear_repulsion"] = float(result["nuclear_repulsion"])
                # the SavedFct files of the AOs are the bytes kept in the molecule
                mra_orbitals = _read_orbitals(result["workdir"], S.shape[0])
                with open(os.path.join(result["workdir"], "mra_parameters.json")) as f:
                    mra_parameters = json.load(f)
                shutil.rmtree(result["workdir"])

                # restricted HF in the projected AO basis (close to, but not the same as the gaussian one)
                mf = _solve_rhf(
                    h_ao,
                    g_ao,
                    S,
                    kwargs.get("n_electrons", parameters.total_n_electrons),
                    kwargs["nuclear_repulsion"],
                )
                orbitals = [
                    OrbitalData(idx_total=i, energy=e)
                    for i, e in enumerate(mf.mo_energy)
                ]

                kwargs["overlap_integrals"] = S
                kwargs["two_body_integrals"] = NBodyTensor(
                    elems=g_ao, ordering="mulliken"
                )
                kwargs["one_body_integrals"] = h_ao
                kwargs["orbital_coefficients"] = mf.mo_coeff
                kwargs["orbital_type"] = "hf"

        # the MRA orbitals are kept as bytes (SavedFct content): they do not need a madness world, so the molecule
        # can live in a process without madness. mra_from_bytes gives them back as functions inside a world
        # mra_parameters (L, k, ...) of the world they were made in, they are only valid in a world with the same ones
        self.mra_orbitals = mra_orbitals
        self.mra_parameters = mra_parameters
        self.datadir = datadir
        super().__init__(*args, parameters=parameters, orbitals=orbitals, **kwargs)
        if pno_file is not None:
            # as the old madness interface kept its files: the next sun.Molecule(..., n_pno="read") reads it
            if datadir is not None:
                os.makedirs(datadir, exist_ok=True)
            self.save(pno_file)

    def _pno_data(
        self, parameters: ParametersQC, n_pno, datadir, world_kwargs: dict, kwargs: dict
    ):
        """
        MRA-PNOs as the old madness interface: n_pno PNOs on top of the occupied HF orbitals (default: one per active pair)
        computed by frayedends MadPNO in the worker, n_pno="read" reads the molecule saved by a previous calculation
        ({datadir}/{name}_pno.h5). madness input sections as dictionaries: dft={...}, pno={...}, nemo={...}, plot={...}
        Fills kwargs with the integrals, returns the orbitals (with the PNO pair structure), the mra orbitals and parameters
        and the file the molecule is saved to (None if it was read)
        """
        base = (
            parameters.name
            if datadir is None
            else os.path.join(datadir, parameters.name)
        )
        path = base + "_pno.h5"

        if (
            isinstance(n_pno, str)
            and n_pno.lower() == "read"
            and not os.path.exists(path)
            and os.path.exists(base + "_htensor.npy")
        ):
            # files of the old madness interface: no MRA orbitals, so no transform_orbitals, refinement or cube plots
            h, g, nuclear_repulsion, pairs, occs = _read_madness_files(base)
            kwargs["one_body_integrals"] = h
            kwargs["two_body_integrals"] = NBodyTensor(elems=g, ordering="mulliken")
            kwargs.setdefault("nuclear_repulsion", nuclear_repulsion)
            mra_orbitals, mra_parameters, path = None, None, None
        elif isinstance(n_pno, str) and n_pno.lower() == "read":
            from sunrise.molecules import load_molecule

            loaded = load_molecule(path)
            manager = loaded.integral_manager
            kwargs["one_body_integrals"] = manager.one_body_integrals
            kwargs["two_body_integrals"] = manager.two_body_integrals
            kwargs.setdefault("nuclear_repulsion", manager.constant_term)
            pairs = [o.pair for o in manager.orbitals]
            occs = [o.occ for o in manager.orbitals]
            mra_orbitals, mra_parameters, path = (
                loaded.mra_orbitals,
                loaded.mra_parameters,
                None,
            )
        else:
            from ._frayed_worker import run_frayed

            if parameters.multiplicity != 1:
                raise SunriseFrayedEndsException(
                    f"MRA-PNOs only for closed shells, multiplicity={parameters.multiplicity}"
                )
            n_electrons = parameters.total_n_electrons
            n_active_pairs = (
                n_electrons
                - (
                    parameters.get_number_of_core_electrons()
                    if parameters.frozen_core
                    else 0
                )
            ) // 2
            if n_pno is None:
                n_pno = n_active_pairs  # minimal correlated: one PNO per active pair
            madness = {
                k: kwargs.pop(k) for k in ("dft", "pno", "nemo", "plot") if k in kwargs
            }
            madness["dft"] = {"charge": parameters.charge, **madness.get("dft", {})}
            integral_kwargs = {
                k: kwargs.pop(k)
                for k in ("truncation_tol", "coulomb_lo", "coulomb_eps")
                if k in kwargs
            }
            result = run_frayed(
                {
                    "task": "pno",
                    "world": world_kwargs,
                    "integrals": integral_kwargs,
                    "geometry": parameters.get_geometry_string(),
                    "n_orbitals": n_electrons // 2 + n_pno,
                    "frozen_core": parameters.frozen_core,
                    "silent": kwargs.get("silent", True),
                    "madness": madness,
                }
            )
            kwargs["one_body_integrals"] = result["h"]
            kwargs["two_body_integrals"] = NBodyTensor(
                elems=result["g"], ordering="mulliken"
            )
            kwargs.setdefault("nuclear_repulsion", float(result["nuclear_repulsion"]))
            # HF orbitals first (occ=2, pair (k, k)), then the PNOs of the pairs (i, j) of HF orbitals
            pairs = [tuple(int(x) for x in p) for p in result["pair"]]
            occs = [float(x) for x in result["occ"]]
            mra_orbitals = _read_orbitals(result["workdir"], len(pairs))
            with open(os.path.join(result["workdir"], "mra_parameters.json")) as f:
                mra_parameters = json.load(f)
            shutil.rmtree(result["workdir"])

        orbitals = [
            OrbitalData(idx_total=i, pair=p, occ=o)
            for i, (p, o) in enumerate(zip(pairs, occs))
        ]
        kwargs["orbital_type"] = "pno"
        if kwargs.get("active_orbitals") is None:
            # as the old madness interface ("auto"): reference orbitals without PNOs (the frozen core) are not active
            kwargs["active_orbitals"] = [
                o.idx_total
                for o in orbitals
                if o.occ != 2.0 or sum(o.idx_total in x.pair for x in orbitals) > 1
            ]
            kwargs["frozen_orbitals"] = []
        return orbitals, mra_orbitals, mra_parameters, path

    def save_mra_orbitals(self, prefix: str = "orbital", datadir: str = "."):
        """
        Write the MRA orbitals as frayedends files {datadir}/{prefix}{i}.data (e.g. to use them in a frayedends script)
        read them back with fe.SavedFct3D(filename), only valid in a MadWorld with the same mra_parameters (L, k)
        """
        os.makedirs(datadir, exist_ok=True)
        _write_orbitals(datadir, self.mra_orbitals, pattern=prefix + "{}.data")

    def cube_plot(
        self,
        filename: str | None = None,
        orbital: list[int] | None = None,
        use_active: bool = True,
        print_orbital: bool = True,
        density: bool = False,
        rdm1: numpy.ndarray | None = None,
        outdir: str = ".",
        **kwargs,
    ):
        """
        DRAFT. Same input as sunrise.plot_MO, but the MOs are built from the MRA orbitals instead of the gaussian AOs
        The plotting runs in a frayedends worker process (see _frayed_worker), this process does not need a madness world
        Parameters
        ----------
        filename: Cube file will be saved as orb_index+filename
        orbital: index of the orbitals to save
        use_active: Wether to plot only the active orbitals, if orbital list passed and true, its assumed that the indices are w.r.t. active orbitals (mol.integral_manager.active_orbitals[x].idx instead of ...[x].idx_total).
        print_orbital: whether to print the MOs
        density: whether to print the electron density (frozen reference orbitals doubly occupied + active rdm1)
        rdm1: active space rdm1 for the density (default: reference occupation)
        outdir: folder for the cube files
        kwargs: further arguments for MadWorld.cube_plot (zoom, datapoints, origin)
        """
        from ._frayed_worker import run_frayed

        if self.mra_orbitals is None:
            raise SunriseFrayedEndsException("molecule has no MRA orbitals to plot")
        if filename is None:
            filename = (
                self.parameters.name
                + "-"
                + self.integral_manager._basis_name  # TODO
                + "-"
                + self.integral_manager._orbital_type  # TODO
            )

        if orbital is None and use_active:
            orbital = [i.idx_total for i in self.integral_manager.active_orbitals]
            label = [i.idx for i in self.integral_manager.active_orbitals]
        elif orbital is None and not use_active:
            orbital = [i.idx_total for i in self.integral_manager.orbitals]
            label = orbital
        elif orbital is not None and use_active:
            d = {i.idx: i.idx_total for i in self.integral_manager.orbitals}
            label = orbital.copy()
            orbital = [d[i] for i in orbital]
        else:
            label = orbital
        if density and rdm1 is None:
            reference = self.integral_manager.active_space.reference_orbitals
            rdm1 = numpy.diag(
                [
                    2.0 if i in reference else 0.0
                    for i in self.integral_manager.active_space.active_orbitals
                ]
            )

        # the worker gets the molecule as file, the cube files are written straight into outdir
        with tempfile.TemporaryDirectory() as tmp:
            self.save(os.path.join(tmp, "mol.h5"))
            result = run_frayed(
                {
                    "task": "cube_plot",
                    "molecule": os.path.join(tmp, "mol.h5"),
                    "outdir": os.path.abspath(outdir),
                    "filename": filename,
                    "orbital": [int(i) for i in orbital] if print_orbital else [],
                    "label": [str(i) for i in label],
                    "rdm1": numpy.asarray(rdm1).tolist() if density else None,
                    "plot_kwargs": kwargs,
                }
            )
        shutil.rmtree(result["workdir"])

    def transform_orbitals(
        self,
        orbital_coefficients: numpy.ndarray | None = None,
        ignore_active_space: bool = False,
        name: str | None = None,
        source: QuantumChemistryBase | None = None,
        *args,
        **kwargs,
    ):
        """
        Same as QuantumChemistryBase.transform_orbitals, but the new orbitals are built as MRA functions
        (MO[i] = sum_j AO[j] C[j, i], symmetric orthonormalized) and their integrals are computed by frayedends
        The madness part runs in the frayedends worker (see _frayed_worker), as in examples/VB_embedding/embedding.py::orbital_refinement
        Parameters
        ----------
        orbital_coefficients: second index is new orbital index, first is old orbital index (summed over), indices are assumed to be defined on the active space (default: no transformation)
        ignore_active_space: if true orbital_coefficients are not assumed to be given in the active space
        name: str, name the new orbitals
        source: molecule (same geometry and basis) the orbitals come from, default is this molecule
            pyscf like molecules use mo_coeff_from_pyscf_base, psi4 molecules mo_coeff_from_psi4_base
            its active space is used for orbital_coefficients and kept in the new molecule
        Returns
        -------
        New frayedends molecule with the transformed orbitals as its MRA orbitals
        """
        from ._frayed_worker import run_frayed

        if self.mra_orbitals is None:
            raise SunriseFrayedEndsException(
                "molecule has no MRA orbitals to transform"
            )
        if source is None:
            source = self
            C = self.integral_manager.orbital_coefficients
        elif hasattr(source, "ref_wfn"):
            C = mo_coeff_from_psi4_base(source)
        else:
            C = mo_coeff_from_pyscf_base(source)
        if C.shape[0] != len(self.mra_orbitals):
            raise SunriseFrayedEndsException(
                f"orbital coefficients of the source have {C.shape[0]} AOs, the MRA basis has {len(self.mra_orbitals)}"
            )

        # same as QuantumChemistryBase.transform_orbitals: U acts on the active space of the source
        U = numpy.eye(C.shape[1])
        if orbital_coefficients is not None:
            if ignore_active_space:
                U = orbital_coefficients
            else:
                active_indices = [
                    o.idx_total for o in source.integral_manager.active_orbitals
                ]
                U[numpy.ix_(active_indices, active_indices)] = orbital_coefficients
        C = C @ U

        # the worker gets the AOs as SavedFct files
        with tempfile.TemporaryDirectory() as tmp:
            _write_orbitals(tmp, self.mra_orbitals)
            result = run_frayed(
                {
                    "task": "transform_orbitals",
                    "input_dir": tmp,
                    "n_orbitals": len(self.mra_orbitals),
                    "C": C.tolist(),
                    "geometry": self.parameters.get_geometry_string(),
                    "mra_parameters": self.mra_parameters,
                }
            )
        return self._from_worker_orbitals(
            result,
            n_orbitals=C.shape[1],
            active_orbitals=[
                o.idx_total for o in source.integral_manager.active_orbitals
            ],
            reference_orbitals=source.integral_manager.active_space.reference_orbitals,
            orbital_type="mra" if name is None else name,
            **kwargs,
        )

    def _from_worker_orbitals(
        self,
        result: dict,
        n_orbitals: int,
        active_orbitals,
        reference_orbitals,
        orbital_type,
        **kwargs,
    ):
        # new molecule from the orbital_{i} files and the integrals (h, mulliken g) of a frayedends worker task
        mra_orbitals = _read_orbitals(result["workdir"], n_orbitals)
        shutil.rmtree(result["workdir"])
        if hasattr(self, "select"):
            # hybrid molecules keep their encoding, the active space is the same
            kwargs.setdefault("select", self.select)

        # the new orbitals are the basis now: orthonormal, no coefficients
        return type(
            self
        )(
            parameters=self.parameters,
            transformation=self.transformation,
            one_body_integrals=result["h"],
            two_body_integrals=NBodyTensor(elems=result["g"], ordering="mulliken"),
            nuclear_repulsion=self.integral_manager.constant_term,
            active_orbitals=active_orbitals,
            reference_orbitals=reference_orbitals,
            frozen_orbitals=[],  # the active space is given, no core is frozen on top of it
            orbital_type=orbital_type,
            mra_orbitals=mra_orbitals,
            mra_parameters=self.mra_parameters,
            **kwargs,
        )

    def refine_orbitals(
        self,
        solver: str | typing.Callable = "spa",
        solver_kwargs: dict | None = None,
        refinement_kwargs: dict | None = None,
        maxiter: int = 10,
        econv: float = 1.0e-6,
        silent: bool = False,
    ):
        """
        Orbital refinement as in examples/VB_embedding/embedding.py::orbital_refinement, alternates until the energy changes less than econv:
            1. solver on the molecule -> energy, rdm1, rdm2 (in this process)
            2. refinement of the active MRA orbitals with these rdms -> new molecule (in the frayedends worker)
        Parameters
        ----------
        solver: name in frayed_solvers.SOLVERS (spa, upccgsd, upccd, fci), a callable f(mol, **solver_kwargs) -> (energy, rdm1, rdm2)
            or "path/to/file.py:function" with such a function. rdms spin summed in dirac ordering, as mol.compute_rdms
        solver_kwargs: keyword arguments of the solver, e.g. {"edges": [(0, 1), (2, 3)]} for spa
        refinement_kwargs: for frayedends OrbitalRefinement: opt_thresh, occ_thresh, maxiter, refine_core, use_nonlinear_solver,
            orthonormalization_method, degeneracy_tol, truncation_tol, coulomb_lo, coulomb_eps, BSH_lo, BSH_eps
        maxiter: maximum number of refinements
        econv: energy convergence
        silent: print the energy of every iteration
        Returns
        -------
        Molecule with the refined orbitals
        """
        from .frayed_solvers import get_solver

        solver = get_solver(solver)
        mol = self
        C = self.integral_manager.orbital_coefficients
        if C.shape != (len(self.mra_orbitals),) * 2 or not numpy.allclose(
            C, numpy.eye(C.shape[0])
        ):
            # the MRA orbitals have to be the molecular orbitals (e.g. not the AOs of the projector)
            mol = self.transform_orbitals()

        energy = None
        for iteration in range(maxiter + 1):
            new_energy, rdm1, rdm2 = solver(mol, **(solver_kwargs or {}))
            if not silent:
                print(f"orbital refinement iteration {iteration} energy {new_energy}")
            if energy is not None and abs(new_energy - energy) < econv:
                return mol
            energy = new_energy
            if iteration < maxiter:
                mol = mol._refine_step(rdm1, rdm2, refinement_kwargs or {})
        warnings.warn(
            f"orbital refinement not converged in {maxiter} iterations",
            SunriseFrayedEndWarning,
        )
        return mol

    def _refine_step(self, rdm1, rdm2, refinement_kwargs: dict):
        # one frayedends OrbitalRefinement of the active orbitals with the rdms, the frozen core is kept (unless refine_core)
        from ._frayed_worker import run_frayed

        space = self.integral_manager.active_space
        with tempfile.TemporaryDirectory() as tmp:
            _write_orbitals(tmp, self.mra_orbitals)
            numpy.save(os.path.join(tmp, "rdm1.npy"), numpy.asarray(rdm1))
            numpy.save(os.path.join(tmp, "rdm2.npy"), numpy.asarray(rdm2))
            result = run_frayed(
                {
                    "task": "refine_orbitals",
                    "input_dir": tmp,
                    "n_orbitals": len(self.mra_orbitals),
                    "core": list(space.frozen_reference_orbitals),
                    "active": list(space.active_orbitals),
                    "nuclear_repulsion": self.integral_manager.constant_term,
                    "geometry": self.parameters.get_geometry_string(),
                    "mra_parameters": self.mra_parameters,
                    "refinement": refinement_kwargs,
                }
            )
        return self._from_worker_orbitals(
            result,
            n_orbitals=len(self.mra_orbitals),
            active_orbitals=list(space.active_orbitals),
            reference_orbitals=space.reference_orbitals,
            orbital_type="refined",
        )

    def state_average_refinement(
        self,
        n_roots: int,
        solver: str = "fci",
        solver_kwargs: dict | None = None,
        refinement_kwargs: dict | None = None,
        maxiter: int = 10,
        econv: float = 1.0e-6,
        silent: bool = False,
    ):
        """
        State average orbital refinement for excited states, as FrayedEnds examples/excited_states/h4_mra_pno_{fci,dmrg}.py
        The whole loop runs in the frayedends worker (solver and refinement), until the mean energy changes less than econv
        Parameters
        ----------
        n_roots: number of states averaged (ground state + n_roots - 1 excited states)
        solver: name in _frayed_worker._SA_SOLVERS (fci, dmrg)
        solver_kwargs: keyword arguments of the solver, e.g. {"bond_dim": 200} for dmrg
        refinement_kwargs: for frayedends OrbitalRefinement, as in refine_orbitals
        Returns
        -------
        Molecule with the refined orbitals, energies of the roots in every iteration (n_iterations, n_roots)
        """
        from ._frayed_worker import run_frayed

        mol = self
        C = self.integral_manager.orbital_coefficients
        if C.shape != (len(self.mra_orbitals),) * 2 or not numpy.allclose(
            C, numpy.eye(C.shape[0])
        ):
            # the MRA orbitals have to be the molecular orbitals (e.g. not the AOs of the projector)
            mol = self.transform_orbitals()

        space = mol.integral_manager.active_space
        c, h, g = mol.get_integrals(
            ordering="chem"
        )  # active space, with the frozen core folded in
        with tempfile.TemporaryDirectory() as tmp:
            _write_orbitals(tmp, mol.mra_orbitals)
            result = run_frayed(
                {
                    "task": "state_average_exct_state_refinement",
                    "input_dir": tmp,
                    "n_orbitals": len(mol.mra_orbitals),
                    "core": list(space.frozen_reference_orbitals),
                    "active": list(space.active_orbitals),
                    "nuclear_repulsion": mol.integral_manager.constant_term,
                    "geometry": mol.parameters.get_geometry_string(),
                    "mra_parameters": mol.mra_parameters,
                    "c": float(c),
                    "h": h.tolist(),
                    "g": g.elems.tolist(),
                    "n_electrons": mol.n_electrons,
                    "n_roots": n_roots,
                    "solver": solver,
                    "solver_kwargs": solver_kwargs or {},
                    "refinement": refinement_kwargs or {},
                    "maxiter": maxiter,
                    "econv": econv,
                }
            )
        energies = result["energies"]
        if not silent:
            for iteration, e in enumerate(energies):
                print(f"state average refinement iteration {iteration} energies {e}")
        if not result["converged"]:
            warnings.warn(
                f"state average refinement not converged in {maxiter} iterations",
                SunriseFrayedEndWarning,
            )
        new = mol._from_worker_orbitals(
            result,
            n_orbitals=len(mol.mra_orbitals),
            active_orbitals=list(space.active_orbitals),
            reference_orbitals=space.reference_orbitals,
            orbital_type="refined",
        )
        return new, energies

    def get_pair_orbitals(
        self, i: OrbitalData, j: OrbitalData, exclude: list[OrbitalData] | None = None
    ):
        if isinstance(i, int):
            i = self.orbitals[i]
        if isinstance(j, int):
            j = self.orbitals[j]
        if isinstance(exclude, int):
            exclude = [self.orbitals[exclude]]
        if exclude is None or isinstance(exclude, OrbitalData):
            exclude = [exclude]
        return [
            x
            for x in self.orbitals
            if (i.idx_total, j.idx_total) == x.pair and x not in exclude
        ]

    def get_virtual_orbitals(self):
        return [x for x in self.orbitals if len(x.pair) == 1 and x.pair[0] < 0]

    # mol.save stores the PNO pair structure, sun.load_molecule gives it back to the qubit madness interface
    def _backend_data(self):
        # the mra orbitals themselves are too large for the json, save writes them as raw bytes
        return {"mra_parameters": self.mra_parameters}

    def save(self, filename: str):
        super().save(filename)
        if self.mra_orbitals is not None:
            with h5py.File(filename, "a") as f:
                for i, b in enumerate(self.mra_orbitals):
                    f[f"mra_orbitals/{i}"] = numpy.frombuffer(b, dtype=numpy.uint8)

    def use_CLPO_orbitals_and_edges(self, *args, **kwargs):
        """
        Currently no janpa - frayedends interface is implemented. However it can be interfaced via pyscf. See `examples/VB_embedding/embeddings.py`
        """
        raise NotImplementedError

    def use_CLPO_orbitals(self, *args, **kwargs):
        """
        Currently no janpa - frayedends interface is implemented. However it can be interfaced via pyscf. See `examples/VB_embedding/embeddings.py`
        """
        raise NotImplementedError

    def use_HAO_orbitals(self, *args, **kwargs):
        """
        Currently no janpa - frayedends interface is implemented. However it can be interfaced via pyscf. See `examples/VB_embedding/embeddings.py`
        """
        raise NotImplementedError

    def make_spa_ansatz(self, label=None, hcb=False, edges=None, **kwargs):
        """
        Shortcut for convenience
        Parameters
        ----------
        label:
           label for the angles
        hcb:
           if True the circuit will not map from HCB to JW (or other encodings supported)
        edges, kwargs:
           if given, the SPA of the backend base class is used (e.g. edges=mol.get_spa_edges())
        Returns
        -------
        Default SPA ansatz (equivalent to PNO-UpCCD with madness PNOs)

        """
        if edges is not None or kwargs:
            # the fermionic SPA has no hcb option
            if hcb:
                kwargs["hcb"] = hcb
            return super().make_spa_ansatz(edges=edges, label=label, **kwargs)
        if not any(o.pair is not None for o in self.orbitals):
            # TODO: without PNOs the edges could come from the CLPO module (see examples/VB_embedding/embedding.py)
            raise SunriseFrayedEndsException(
                "SPA ansatz without PNOs needs edges. Please provide with the keyword edges, or use MRA-PNOs (no basis_set).\nExample: edges=[(0,1,2),(3,4)] would correspond to two edges created from orbitals (0,1,2) and (3,4), note that orbitals can only be assigned to a single edge"
            )
        name = "SPA-UpCCD"  # NOTE: This is due to PNOs, will change
        if hcb and "HCB" not in name.upper():
            name = "HCB-" + name
        return self.make_upccgsd_ansatz(name=name, label=label)

    # def perturbative_f12_correction(
    #     self,
    #     rdm1: numpy.ndarray | None = None,
    #     rdm2: numpy.ndarray | None = None,
    #     n_ri: int | None = None,
    #     f12_filename: str = "molecule_f12tensor.bin",
    #     **kwargs,
    # ) -> float:
    #     """
    #     Computes the spin-free [2]_R12 correction, needing only the 1- and 2-RDM of a reference method
    #     Requires either 1-RDM, 2-RDM or information to compute them in kwargs

    #     Parameters
    #     ----------
    #     rdm1 :
    #         1-electron reduced density matrix
    #     rdm2 :
    #         2-electron reduced density matrix
    #     gamma :
    #         f12-exponent, for a correlation factor f_12 = -1/gamma * exp[-gamma*r_12]
    #     n_ri :
    #         dimensionality of RI-basis; if None, then the maximum available via tensors / basis-set is used
    #     f12_filename :
    #         when using madness_interface, <q|h|p> and <rs|1/r_12|pq> already available;
    #         need to provide f12-tensor <rs|f_12|pq> as ".bin" from madness or ".npy", assuming Mulliken ordering
    #     kwargs :
    #         e.g. RDM-information via {"U": QCircuit, "variables": optimal angles}, needs to be passed if rdm1,rdm2 not
    #         yet computed

    #     Returns
    #     -------
    #         the f12 correction for the energy
    #     """
    #     from .f12_corrections._f12_correction_frayed_ends import (
    #         ExplicitCorrelationCorrectionFrayedends,
    #     )

    #     correction = ExplicitCorrelationCorrectionFrayedends(
    #         mol=self,
    #         rdm1=rdm1,
    #         rdm2=rdm2,
    #         n_ri=n_ri,
    #         f12_filename=f12_filename,
    #         **kwargs,
    #     )

    #     return correction.compute()

    def __str__(self):
        info = super().__str__()
        info += "\nMRA\n"
        info += f"{'orbital type':15} : {self.integral_manager._orbital_type}\n"
        if self.mra_orbitals is not None:
            size = sum(len(b) for b in self.mra_orbitals) / 1.0e6
            info += f"{'mra orbitals':15} : {len(self.mra_orbitals)} ({size:.1f} MB)\n"
        for k in ("L", "k", "thresh"):
            if self.mra_parameters is not None and k in self.mra_parameters:
                info += f"{k:15} : {self.mra_parameters[k]}\n"
        if self.integral_manager._orbital_type == "pno":
            path = self.parameters.name + "_pno.h5"
            if self.datadir is not None:
                path = os.path.join(self.datadir, path)
            info += f"{'pno file':15} : {path}\n"

        return info

    def __repr__(self):
        return self.__str__()

    def compute_energy(self, method, *args, **kwargs):
        """
        Call classical methods over PySCF (needs to be installed) or
        use as a shortcut to calculate quantum energies (see make_upccgsd_ansatz)

        Parameters
        ----------
        method: method name
                classical: HF, MP2, CCSD, CCSD(T), FCI
                quantum: SPA-GASD (SPA can be dropped as well as letters in GASD)
                examples: GSD is the same as UpCCGSD, SPA alone is equivalent to SPA-D
                see make_upccgsd_ansatz of the this class for more information
        args
        kwargs

        Returns
        -------

        """

        if any(x in method.upper() for x in ["U", "SPA", "PNO", "HCB"]):
            # simulate entirely in HCB representation if no singles are involved
            if "S" not in method.upper().split("-")[-1] and "HCB" not in method.upper():
                method = "HCB-" + method
            U = self.make_upccgsd_ansatz(name=method)
            if "hcb" in method.lower():
                H = self.make_hardcore_boson_hamiltonian()
            else:
                H = self.make_hamiltonian()
            E = ExpectationValue(H=H, U=U)
            from sunrise import minimize

            return minimize(*args, objective=E, **kwargs).energy
        else:
            return super().compute_energy(*args, method=method, **kwargs)

    @classmethod
    def from_sunrise(
        cls, other: QuantumChemistryBase, name: str | None = None, **kwargs
    ):
        """
        Frayedends molecule with the orbitals of another sunrise molecule (pyscf or psi4, gaussian basis)
        The AOs of the same basis are projected on MRA (AtomicBasisProjector), the coefficients of other are reordered
        to the madness AO order and its active space is kept, see transform_orbitals(source=other)
        The basis has to be in the madness AO library (e.g. sto-3g, sto-6g, 6-31g)
        kwargs: for the projector molecule, e.g. the MadWorld options (L, k, thresh) or transformation (default: the one of other)
        """
        if hasattr(other, "transformation"):
            kwargs.setdefault("transformation", other.transformation)
        projector = cls(parameters=other.parameters, **kwargs)
        return projector.transform_orbitals(source=other, name=name)

    @classmethod
    def from_tequila(
        cls, other: QuantumChemistryBase, name: str | None = None, **kwargs
    ):
        """
        Short cut for from_sunrise
        Frayedends molecule with the orbitals of another sunrise molecule (pyscf or psi4, gaussian basis)
        The AOs of the same basis are projected on MRA (AtomicBasisProjector), the coefficients of other are reordered
        to the madness AO order and its active space is kept, see transform_orbitals(source=other)
        The basis has to be in the madness AO library (e.g. sto-3g, sto-6g, 6-31g)
        kwargs: for the projector molecule, e.g. the MadWorld options (L, k, thresh) or transformation (default: the one of other)
        """
        return cls.from_sunrise(other=other, name=name, **kwargs)

class QuantumChemistryFrayed(FrayedEndsMixin, QuantumChemistryBase):
    def __init__(
        self,
        parameters: ParametersQC,
        transformation: str | typing.Callable | None = None,
        active_orbitals: list | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            parameters,
            *args,
            transformation=transformation,
            active_orbitals=active_orbitals,
            **kwargs,
        )

    def local_qubit_map(self, hcb=False, up_then_down=False):
        # re-arrange orbitals to result in more local circuits
        # does not make the circuit more local, but rather will show locality better in pictures
        # transform circuits and Hamiltonians with this map
        # H = H.map_qubits(qubit_map), U = U.map_qubits(qubit_map)
        # hcb: same for the harcore_boson representation
        ordered_qubits = []
        pairs = [i for i in range(self.n_electrons // 2)]
        for i in pairs:
            pnos = [i] + [a.idx for a in self.get_pair_orbitals(i=i, j=i, exclude=i)]
            if hcb:
                up = [i for i in pnos]
                ordered_qubits += up
            else:
                if up_then_down:
                    up = [self.transformation.up(i) for i in pnos]
                    down = [self.transformation.down(i) for i in pnos]
                    ordered_qubits += up + down
                else:
                    for i in pnos:
                        ordered_qubits.append(self.transformation.up(i))
                        ordered_qubits.append(self.transformation.down(i))

        qubit_map = {x: i for i, x in enumerate(ordered_qubits)}
        return qubit_map

    def make_upccgsd_ansatz(
        self,
        name="UpCCGSD",
        label=None,
        direct_compiling=None,
        order=None,
        neglect_z=None,
        hcb_optimization=None,
        include_reference=True,
        indices: list | None = None,
        *args,
        **kwargs,
    ):
        """
        Overwriting baseclass to allow names like : PNO-UpCCD etc
        Parameters
        ----------
        label: label the variables of the ansatz ( variables will be labelled (indices, X, (label, layer) witch X=D/S)
        direct_compiling: Directly compile the first layer (works only for transformation that implement the hcb_to_me function)
        name: ansatz name (SPA-UpCCD or SPA-D, SPA-UpCCGD or SPA-GD, SPA-UpCCGSD or SPA-GSD, UpCCGSD or GSD ..., in general {HCB}-{SPA}-{Excitations}
              if HCB is included in name: do not map from hard-core Boson to qubit encoding of this molecule
              if SPA is included in name: Use the separable pair ansatz (excitations will be restricted to the PNO structure of the surrogate model)
              Excitations: can be "S" (use only singles), "D" (use only doubles), "GSD" (generalized singles and doubles), "GASD" (approximate singles, neglecting Z terms in JW)
        neglect_z: neglect all Z terms in singles excitations generators
        order: repetition of layers, can be given over the name as well, the order needs to be the first in the name then (i.e. 2-UpCCGSD, 2-SPA-GSD, etc)
        indices: excitation indices in the format of format_excitation_indices, e.g. [((0, 3),), ((1, 4),)]
                 (see make_pno_upccgsd_ansatz), they replace the ones of the name in every layer
        args
        kwargs

        Returns
        -------

        """
        # check if the used qubit encoding has a hcb transformation
        try:
            have_hcb_trafo = self.transformation.hcb_to_me() is not None
        except Exception:  # noqa: BLE001
            have_hcb_trafo = False
        name = name.upper()

        # Default Method
        if "SPA" in name and name.split("-")[-1] in ["SPA", "HCB"]:
            name += "-D"

        excitations = name.split("-")[-1]

        if "HCB" in name and "S" in excitations:
            raise SunriseFrayedEndsException(
                f"name={name}, HCB + Singles can't be realized"
            )

        if "HCB" in name and "D" not in excitations:
            warnings.warn(
                f"name={name}, HCB without Doubles has no result ",
                TequilaWarning,
            )

        if "S" not in excitations and "D" not in excitations:
            warnings.warn(
                f"name={name}, neither singles nor doubles requested",
                TequilaWarning,
            )

        if "T" in excitations or "Q" in excitations:
            warnings.warn(
                f"name={name}, only singles and doubles supported",
                TequilaWarning,
            )

        if (
            (have_hcb_trafo and "D" in excitations or "HCB" in name)
            and include_reference
            and hcb_optimization is None
        ):
            hcb_optimization = True

        if hcb_optimization and direct_compiling is None:
            direct_compiling = True

        if ("A" in excitations) and neglect_z is None:
            neglect_z = True
            # spin adaption does not work with z neglected
            if "spin_adapt_singles" not in kwargs:
                kwargs["spin_adapt_singles"] = False
        else:
            neglect_z = False
            # switch on by default
            if "spin_adapt_singles" not in kwargs:
                kwargs["spin_adapt_singles"] = True

        if direct_compiling and not have_hcb_trafo and "HCB" not in name:
            raise SunriseFrayedEndsException(
                f"direct_compiling={direct_compiling} demanded but no hcb_to_me in transformation={self.transformation}\ntry transformation='ReorderedJordanWigner' "
            )

        name = name.upper()
        if order is None:
            try:
                if "-" in name:
                    order = int(name.split("-")[0])
                else:
                    order = 1
            except Exception:  # noqa: BLE001
                order = 1

        # first layer
        U = QCircuit()
        given_indices = indices
        if indices is not None:
            # raw pairs (p, q) as built by make_pno_upccgsd_ansatz, in the format of the layers ((p, q),)
            given_indices = [
                x if isinstance(x[0], tuple) else self.format_excitation_indices([x])
                for x in indices
            ]
        if hcb_optimization:
            if "D" in excitations and given_indices is None:
                U = self.make_hardcore_boson_pno_upccd_ansatz(
                    include_reference=include_reference,
                    direct_compiling=direct_compiling,
                    label=(label, 0),
                )
            elif include_reference:
                U = self.prepare_hardcore_boson_reference()

            indices0 = [k.name[0] for k in U.extract_variables()]
            if given_indices is None:
                indices1 = self.make_upccgsd_indices(
                    *args, label=label, name=name, exclude=indices0, **kwargs
                )
            else:
                indices1 = [x for x in given_indices if x not in indices0]
            if "D" in excitations:
                U += self.make_hardcore_boson_upccgd_layer(
                    *args, indices=indices1, label=(label, 0), **kwargs
                )
            indices = indices0 + indices1
            if "HCB" not in name and len(U.gates) > 0:
                U = self.hcb_to_me(U=U)
            else:
                assert "S" not in excitations
            if "S" in excitations:
                U += self.make_upccgsd_singles(
                    *args,
                    indices=indices,
                    label=(label, 0),
                    neglect_z=neglect_z,
                    **kwargs,
                )
        else:
            if given_indices is None:
                indices = self.make_upccgsd_indices(
                    *args, label=(label, 0), name=name, **kwargs
                )
            if include_reference:
                U = self.prepare_reference()
            U += self.make_upccgsd_layer(
                *args,
                indices=indices,
                include_singles="S" in excitations,
                include_doubles="D" in excitations,
                label=(label, 0),
                neglect_z=neglect_z,
                **kwargs,
            )

        if order > 1:
            for layer in range(1, order):
                indices = given_indices
                if given_indices is None:
                    indices = self.make_upccgsd_indices(
                        *args, label=(label, layer), name=name, **kwargs
                    )
                if "HCB" in name:
                    U += self.make_hardcore_boson_upccgd_layer(
                        *args, indices=indices, label=(label, layer), **kwargs
                    )
                else:
                    U += self.make_upccgsd_layer(
                        *args,
                        indices=indices,
                        include_singles="S" in excitations,
                        include_doubles="D" in excitations,
                        label=(label, layer),
                        neglect_z=neglect_z,
                        **kwargs,
                    )
        return U

    def make_hardcore_boson_pno_upccd_ansatz(
        self,
        pairs=None,
        label=None,
        include_reference: bool = True,
        direct_compiling: bool = False,
        ladder: bool = True,
    ):
        if pairs is None:
            pairs = [x for x in self.reference_orbitals]
        U = QCircuit()
        if direct_compiling:
            if not include_reference:
                raise TequilaException(
                    "HCB_PNO_UPCCD: Direct compiling needs reference included"
                )
            up = self.transformation.up
            for x in pairs:
                U += gates.X(up(x.idx))
                c = [None, x.idx]
                for a in self.get_pair_orbitals(i=x, j=x):
                    if a == x:
                        continue
                    idx = self.format_excitation_indices([(x.idx, a.idx)])
                    if (
                        "TAPEREDBINARY" in self.transformation.name.upper()
                        and a.idx == self.n_orbitals - 1
                    ):
                        # the tapered orbital is implicit: move the pair from c[1] into it with the
                        # HCB excitation gate, which is what Ry + CNOT do in the other encodings
                        U += self.make_hardcore_boson_excitation_gate(
                            indices=self.format_excitation_indices([(c[1], a.idx)]),
                            angle=(idx, "D", label),
                        )
                    else:
                        U += gates.Ry(
                            angle=(idx, "D", label),
                            target=up(a.idx),
                            control=None if c[0] is None else up(c[0]),
                        )
                        U += gates.X(target=up(c[1]), control=up(a.idx))
                    if ladder:
                        c = [a.idx, a.idx]
                    else:
                        c = [x.idx, x.idx]

        else:
            for x in pairs:
                if include_reference:
                    U += gates.X(self.transformation.up(x.idx))
                for a in self.get_pair_orbitals(i=x, j=x):
                    if x == a:
                        continue
                    idx = self.format_excitation_indices([(x.idx, a.idx)])
                    U += self.make_hardcore_boson_excitation_gate(
                        indices=idx, angle=(idx, "D", label)
                    )

        return U

    def make_upccgsd_indices(
        self, name="UpCCGD", exclude: list | None = None, *args, **kwargs
    ):
        """
        :param label: label the angles
        :param generalized: if true the complement to UpCCGD is created (otherwise UpCCD)
        :param exclude: list of indices to exclude
        :return: All gates missing between PNO-UpCCD and UpCC(G)D
        """

        if exclude is None:
            exclude = []
        name = name.upper()

        indices = []

        # HF-X -- PNO-XX indices
        for i in self.reference_orbitals:
            for a in self.get_pair_orbitals(i=i, j=i, exclude=i):
                idx = self.format_excitation_indices([(i.idx, a.idx)])
                if idx not in exclude and idx not in indices:
                    indices.append(idx)

        if "G" in name:
            for i in self.reference_orbitals:
                for a in self.get_pair_orbitals(i=i, j=i, exclude=i):
                    for b in self.get_pair_orbitals(i=i, j=i, exclude=i):
                        if a.idx <= b.idx:
                            continue

                        idx = self.format_excitation_indices([(a.idx, b.idx)])
                        if idx not in exclude and idx not in indices:
                            indices.append(idx)

        if "PNO" in name or "SPA" in name:
            return indices

        virtuals = [i for i in self.orbitals if len(i.pair) == 2 and i.occ != 2.0]
        virtuals += self.get_virtual_orbitals()  # this is usually empty
        # HF-X -- PNO-XY/PNO-YY indices
        for i in self.reference_orbitals:
            for a in virtuals:
                idx = self.format_excitation_indices([(i.idx, a.idx)])
                if idx not in exclude and idx not in indices:
                    indices.append(idx)
        # HF-HF and PNO-PNO indices
        if "G" in name:
            for i in self.reference_orbitals:
                for j in self.reference_orbitals:
                    if i.idx <= j.idx:
                        continue
                    idx = self.format_excitation_indices([(i.idx, j.idx)])
                    if idx not in exclude and idx not in indices:
                        indices.append(idx)
            for a in virtuals:
                for b in virtuals:
                    if a.idx_total <= b.idx_total:
                        continue
                    idx = self.format_excitation_indices([(a.idx, b.idx)])
                    if idx not in exclude and idx not in indices:
                        indices.append(idx)
        return indices

    def make_pno_upccgsd_ansatz(
        self, generalized=False, include_offdiagonals=False, **kwargs
    ):
        indices = []
        for i in self.reference_orbitals:
            for a in self.get_pair_orbitals(i=i, j=i, exclude=i):
                indices.append((i.idx, a.idx))
            if generalized:
                for a in self.get_pair_orbitals(i=i, j=i, exclude=i):
                    for b in self.get_pair_orbitals(i=i, j=i, exclude=i):
                        if b.idx_total <= a.idx_total:
                            continue
                        indices.append((a.idx, b.idx))

        if include_offdiagonals:
            for i in self.reference_orbitals:
                for j in self.reference_orbitals:
                    pairs_ij = self.get_pair_orbitals(i=i, j=j, exclude=[i, j])
                    if i.idx <= j.idx:
                        continue
                    for a in pairs_ij:
                        indices.append((j.idx, a.idx))

                    if generalized:
                        for a in pairs_ij:
                            for b in pairs_ij:
                                if a.idx <= b.idx:
                                    continue
                                indices.append((a.idx, b.idx))

        return self.make_upccgsd_ansatz(indices=indices, **kwargs)


def get_madness_sort_key(item):
    _original_idx, label_str = item
    parts = label_str.split()
    atom_idx = int(parts[0])
    orbital_part = parts[2]

    match = re.match(r"(\d+)([spdfgh])(.*)", orbital_part)
    n = int(match.group(1))
    l_str = match.group(2)
    comp = match.group(3)

    l_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
    l = l_map[l_str]

    comp_order = {
        "x": 0,
        "y": 1,
        "z": 2,
        "xx": 0,
        "xy": 1,
        "xz": 2,
        "yy": 3,
        "yz": 4,
        "zz": 5,
        "": 0,
    }
    c_idx = comp_order.get(comp, 0)

    return (atom_idx, n, l, c_idx)


# TODO: psi4 if not pyscf
def _solve_rhf(h, g, S, n_electrons, nuclear_repulsion):
    # pyscf RHF with custom AO integrals (h, mulliken g, S), as PySCFMixin._get_hf but in a non orthogonal basis
    import pyscf

    n = h.shape[0]
    mol = pyscf.gto.M(verbose=0, parse_arg=False)
    mol.nelectron = n_electrons
    mol.incore_anyway = True  # ensure that custom integrals are used
    mol.energy_nuc = lambda *args: nuclear_repulsion
    mf = pyscf.scf.RHF(mol)
    mf.get_hcore = lambda *args: h
    mf.get_ovlp = lambda *args: S
    mf._eri = pyscf.ao2mo.restore(8, g, n)
    mf.chkfile = None  # the dummy mol can not be dumped
    mf.kernel()
    if not mf.converged:
        warnings.warn(
            "RHF in the projected AO basis did not converge", SunriseFrayedEndWarning
        )
    return mf


def _write_orbitals(
    directory: str, mra_orbitals: list[bytes], pattern: str = "orbital_{}"
):
    # the SavedFct files orbital_{i} the frayedends worker reads
    for i, b in enumerate(mra_orbitals):
        with open(os.path.join(directory, pattern.format(i)), "wb") as f:
            f.write(b)


def _read_madness_files(base: str):
    # {base}_htensor.npy, {base}_gtensor.npy (mulliken) and {base}_pnoinfo.txt of the old madness interface
    # pnoinfo: nuclear_repulsion=..., pairinfo=i.j,... (a single index for the reference orbitals), occinfo=...
    h = numpy.load(base + "_htensor.npy")
    g = numpy.load(base + "_gtensor.npy")
    nuclear_repulsion, pairs, occs = 0.0, None, None
    with open(base + "_pnoinfo.txt") as f:
        for line in f:
            if "nuclear_repulsion" in line:
                nuclear_repulsion = float(line.split("=")[1])
            elif "pairinfo" in line:
                pairs = [
                    tuple(int(i) for i in x.split("."))
                    for x in line.split("=")[1].split(",")
                ]
            elif "occinfo" in line:
                occs = [float(x) for x in line.split("=")[1].split(",")]
    if pairs is None:
        raise SunriseFrayedEndsException(f"no pairinfo in {base}_pnoinfo.txt")
    return h, g, nuclear_repulsion, pairs, occs


def _read_orbitals(directory: str, n_orbitals: int) -> list[bytes]:
    # the SavedFct files orbital_{i} the frayedends worker writes
    data = []
    for i in range(n_orbitals):
        with open(os.path.join(directory, f"orbital_{i}"), "rb") as f:
            data.append(f.read())
    return data


def mra_from_bytes(data: list[bytes]) -> list:
    # only valid in a world with the same parameters (L, k) as the one the functions were made in
    # worker side only: it imports frayedends
    import frayedends as fe

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f")
        functions = []
        for b in data:
            with open(path, "wb") as fh:
                fh.write(b)
            functions.append(fe.SavedFct3D(path))
    return functions


def mo_coeff_from_pyscf_base(mol: QuantumChemistryBase):
    from sunrise.molecules import (
        MoleculeToPyscf,
    )  # circular on module level, sunrise.molecules imports this file

    ##Pyscf uses spherical coordinates meanwhile madness does on cartesians, important for d shells and further
    mol_sph = MoleculeToPyscf(mol, cart=False)
    c2s = mol_sph.cart2sph_coeff(normalized="sp")
    mo_coeff_cart_pyscf_order = numpy.dot(
        c2s, mol.integral_manager.orbital_coefficients
    )
    ## A further issue is the shell ordering, pyscf orders by angular momentum (l) (first all s orbs, then all p...)
    ## meanwhile madness does following natural filling: (...  3s 3p 3d (if populated) 4s 4p)
    mol_cart = MoleculeToPyscf(mol, cart=True)
    pyscf_labels = mol_cart.ao_labels()
    indexed_labels = list(enumerate(pyscf_labels))
    sorted_labels = sorted(indexed_labels, key=get_madness_sort_key)
    madness_row_order = [idx for idx, label in sorted_labels]
    return mo_coeff_cart_pyscf_order[madness_row_order, :]
    # bp = feAtomicBasisProjector(world, geometry, units="a", aobasis="sto-3g")
    # basis = bp.get_orbitals()

    # integrals = fe.Integrals(world)
    # orbitals = integrals.transform(
    #     basis, mo_coeff_madness
    # )  # transforms orbitals according to: orbtials[i] = sum[j] basis[j]*C[j,i]
    # orbitals = integrals.orthonormalize(orbitals=orbitals)


def mo_coeff_from_psi4_base(mol: QuantumChemistryBase):
    try:
        import psi4

        ## Madness works on cartesians, the coefficients are projected onto the cartesian version of the same basis:
        ## C_cart = S_cc^-1 S_cs C (exact, the spherical functions are linear combinations of the cartesian ones)
        wfn = mol.ref_wfn
        bs = wfn.basisset()
        bs_cart = psi4.core.BasisSet.build(
            wfn.molecule(), "ORBITAL", mol.parameters.basis_set, puream=0, quiet=True
        )
        mints = psi4.core.MintsHelper(bs_cart)
        s_cc = numpy.asarray(mints.ao_overlap())
        s_cs = numpy.asarray(mints.ao_overlap(bs_cart, bs))
        mo_coeff_cart = numpy.linalg.solve(
            s_cc, s_cs @ mol.integral_manager.orbital_coefficients
        )
        ## Same row order as mo_coeff_from_pyscf_base: (atom, n, l, component), n counts the shells of each l on an atom
        ## starting at l+1. Psi4 cartesian components are already ordered as madness (x, y, z / xx, xy, xz, yy, yz, zz, ...)
        ## The l >= 2 rows are scaled to the pyscf normalization of the cartesian functions (cart2sph_coeff(normalized="sp"))
        # TODO check which cartesian normalization madness expects: the scale reproduces mo_coeff_from_pyscf_base, if madness
        #  wants the psi4 one (angular factor included) drop the scale
        keys, scale, count = [], [], {}
        for sh in range(bs_cart.nshell()):
            shell = bs_cart.shell(sh)
            atom, l = shell.ncenter, shell.am
            k = count[(atom, l)] = count.get((atom, l), -1) + 1
            keys += [(atom, l + 1 + k, l, j) for j in range(shell.nfunction)]
            scale += [
                numpy.sqrt((2 * l + 1) / (4 * numpy.pi)) if l >= 2 else 1.0
            ] * shell.nfunction
        madness_row_order = sorted(range(len(keys)), key=keys.__getitem__)
        return (numpy.array(scale)[:, None] * mo_coeff_cart)[madness_row_order, :]
    except ImportError:
        warnings.warn("Failed to import Psi4 for FrayedEnds interface", TequilaWarning)
