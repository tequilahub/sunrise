"""
Runs frayedends (MADNESS) in a separate python process, so its MPI/thread runtime never lives in the sunrise process.
Only plain data crosses the boundary:
    request  -> request.json (geometry string, units, basis, world/integral options)
    result   <- result.npz   (numpy arrays + scalars)
    orbitals <- orbital_{i}  (MADNESS SavedFct files, only valid for the same L and k)
Do not import frayedends at module level: the client side of this file is imported by sunrise.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy


def run_frayed(request: dict) -> dict:
    """Client side: write the request, run the worker in a fresh interpreter (in a temp dir), read the result back.
    The caller removes result["workdir"] once it read the files it needs."""
    workdir = tempfile.mkdtemp(prefix="sunrise_frayed_")
    with open(os.path.join(workdir, "request.json"), "w") as f:
        json.dump(request, f)

    proc = subprocess.run(
        [sys.executable, "-m", "sunrise.molecules.qubit_base._frayed_worker", workdir],
        cwd=workdir,  # MADNESS writes its files into the cwd
        capture_output=True,
        text=True,
        check=False,  # the return code is checked below, with the stderr of the worker in the error
    )
    if proc.returncode != 0:
        shutil.rmtree(
            workdir, ignore_errors=True
        )  # the caller never gets the workdir to remove it
        raise RuntimeError(
            f"frayedends worker failed ({proc.returncode}):\n{proc.stderr[-3000:]}"
        )

    with numpy.load(os.path.join(workdir, "result.npz")) as data:
        result = {k: data[k] for k in data.files}
    result["workdir"] = workdir
    return result


def _project(request: dict, workdir: str) -> dict:
    """Worker side: AO projection and integrals in the projected AO basis, see FrayedEndsMixin.__init__.
    The AOs are written as orbital_{i} (SavedFct files) and the world parameters as mra_parameters.json"""
    import frayedends as fe

    world = fe.MadWorld(ndims=3, **request["world"])
    ao = fe.AtomicBasisProjector(
        madworld=world,
        geometry=request["geometry"],
        units="angstrom",
        silent=request["silent"],
        aobasis=request["basis"],
    )
    functions = ao.get_orbitals()
    vnuc = ao.get_nuclear_potential()

    integrals = fe.Integrals(madworld=world, **request["integrals"])

    result = {
        "nuclear_repulsion": ao.get_nuclear_repulsion(),
        "S": integrals.compute_overlap_integrals(functions),
        "h": integrals.compute_kinetic_integrals(functions)
        + integrals.compute_potential_integrals(functions, vnuc),
        "g": integrals.compute_two_body_integrals(functions, ordering="mulliken").elems,
    }
    for i, f in enumerate(functions):
        f.save_to_file(os.path.join(workdir, f"orbital_{i}"))
    with open(os.path.join(workdir, "mra_parameters.json"), "w") as f:
        json.dump(world.madness_parameters, f)

    # the madness objects have to go before their world, otherwise madness crashes on finalize
    del functions, vnuc, integrals, ao, world
    return result


def _pno(request: dict, workdir: str) -> dict:
    """Worker side: MRA-PNOs (frayedends MadPNO), orthonormalized, and their integrals, see FrayedEndsMixin._pno_data.
    The orbitals are written as orbital_{i} (SavedFct files) and the world parameters as mra_parameters.json"""
    import frayedends as fe

    world = fe.MadWorld(ndims=3, **request["world"])
    pno = fe.MadPNO(
        world,
        request["geometry"],
        units="angstrom",
        silent=request["silent"],
        n_orbitals=request["n_orbitals"],
        frozen_core=request["frozen_core"],
        **request["madness"],
    )

    info = fe.get_function_info(pno.get_orbitals())
    integrals = fe.Integrals(madworld=world, **request["integrals"])
    orbitals = integrals.orthonormalize(orbitals=pno.get_orbitals())
    vnuc = pno.get_nuclear_potential()

    result = {
        "nuclear_repulsion": pno.get_nuclear_repulsion(),
        "h": integrals.compute_kinetic_integrals(orbitals)
        + integrals.compute_potential_integrals(orbitals, vnuc),
        "g": integrals.compute_two_body_integrals(orbitals, ordering="mulliken").elems,
        "occ": [x["occ"] for x in info],
        "pair": [(x["pair1"], x["pair2"]) for x in info],
    }
    for i, f in enumerate(orbitals):
        f.save_to_file(os.path.join(workdir, f"orbital_{i}"))
    with open(os.path.join(workdir, "mra_parameters.json"), "w") as f:
        json.dump(world.madness_parameters, f)

    # the madness objects have to go before their world, otherwise madness crashes on finalize
    del orbitals, vnuc, integrals, pno, world
    return result


def _cube_plot(request: dict, workdir: str) -> dict:
    """Worker side: MOs (and density) of a saved frayedends molecule to cube files, see FrayedEndsMixin.cube_plot."""
    import frayedends as fe

    from sunrise.molecules import load_molecule
    from sunrise.molecules.qubit_base.frayedends_interface import mra_from_bytes

    mol = load_molecule(request["molecule"])
    manager = mol.integral_manager
    world = fe.MadWorld(ndims=3, **mol.mra_parameters)
    integrals = fe.Integrals(world)
    geometry = fe.MolecularGeometry(
        geometry=mol.parameters.get_geometry_string(), units="angstrom", silent=True
    )
    # mo[i] = sum_j ao[j] C[j, i]
    mos = integrals.transform(
        mra_from_bytes(mol.mra_orbitals), manager.orbital_coefficients
    )

    outdir, filename = request["outdir"], request["filename"]
    for idx, label in zip(request["orbital"], request["label"]):
        world.cube_plot(
            filename=os.path.join(
                outdir, f"{label}_{filename}_MO"
            ),  # madness adds .cube
            mra_function=mos[idx],
            molecule=geometry,
            **request["plot_kwargs"],
        )
    if request["rdm1"] is not None:
        space = manager.active_space
        rho = integrals.compute_electron_density(
            [mos[i] for i in space.frozen_reference_orbitals],
            [mos[i] for i in space.active_orbitals],
            numpy.array(request["rdm1"]),
        )
        world.cube_plot(
            filename=os.path.join(outdir, f"{filename}_density"),
            mra_function=rho,
            molecule=geometry,
            **request["plot_kwargs"],
        )

    del mos, integrals, world
    return {}


def _nuclear_potential(fe, request: dict, world, eprec: float = 1.0e-6):
    """Nuclear potential with the smoothing (default eprec=1e-6) of the projector and MadPNO, the integrals of the molecule were computed with
    frayedends versions without the eprec option ignore it and use the madness default 1e-4 (integrals slightly off, ~1e-4)"""
    geometry = fe.MolecularGeometry(
        geometry=request["geometry"], units="angstrom", silent=True, eprec=eprec
    )
    return geometry.get_vnuc(world)


def _transform_orbitals(request: dict, workdir: str) -> dict:
    """Worker side: new MRA orbitals from AOs and coefficients, and their integrals, see FrayedEndsMixin.transform_orbitals.
    The new orbitals are written as orbital_{i} (SavedFct files)"""
    import frayedends as fe

    world = fe.MadWorld(ndims=3, **request["mra_parameters"])
    integrals = fe.Integrals(madworld=world)
    aos = [
        fe.SavedFct3D(os.path.join(request["input_dir"], f"orbital_{i}"))
        for i in range(request["n_orbitals"])
    ]
    # orbitals[i] = sum_j aos[j] C[j, i], orthonormalized as the coefficients may come from another (e.g. gaussian) basis
    orbitals = integrals.orthonormalize(
        orbitals=integrals.transform(aos, numpy.array(request["C"]))
    )
    vnuc = _nuclear_potential(fe, request, world)

    result = {
        "h": integrals.compute_kinetic_integrals(orbitals)
        + integrals.compute_potential_integrals(orbitals, vnuc),
        "g": integrals.compute_two_body_integrals(orbitals, ordering="mulliken").elems,
    }
    for i, f in enumerate(orbitals):
        f.save_to_file(os.path.join(workdir, f"orbital_{i}"))

    # the madness objects have to go before their world, otherwise madness crashes on finalize
    del orbitals, aos, vnuc, integrals, world
    return result


_REFINEMENT_INIT_KEYS = (
    "orthonormalization_method",
    "degeneracy_tol",
    "truncation_tol",
    "coulomb_lo",
    "coulomb_eps",
    "BSH_lo",
    "BSH_eps",
)


def _refine_orbitals(request: dict, workdir: str) -> dict:
    """Worker side: one OrbitalRefinement with the given rdms and the integrals of the new orbitals, see FrayedEndsMixin.refine_orbitals.
    The orbitals (all, in the order of the molecule) are written as orbital_{i} (SavedFct files)"""
    import frayedends as fe

    def path(name):
        return os.path.join(request["input_dir"], name)

    world = fe.MadWorld(ndims=3, **request["mra_parameters"])
    orbitals = [
        fe.SavedFct3D(path(f"orbital_{i}")) for i in range(request["n_orbitals"])
    ]
    vnuc = _nuclear_potential(fe, request, world)

    refinement = request["refinement"]
    opti = fe.OrbitalRefinement(
        world,
        vnuc,
        request["nuclear_repulsion"],
        **{k: v for k, v in refinement.items() if k in _REFINEMENT_INIT_KEYS},
    )
    core, active, converged = opti.refine_orbitals(
        orbitals=[
            [orbitals[i] for i in request["core"]],
            [orbitals[i] for i in request["active"]],
        ],
        rdm1=numpy.load(path("rdm1.npy")),
        rdm2=numpy.load(path("rdm2.npy")),
        **{k: v for k, v in refinement.items() if k not in _REFINEMENT_INIT_KEYS},
    )
    # orbitals neither core nor active (frozen virtuals) are kept as they are
    for i, f in zip(request["core"], core):
        orbitals[i] = f
    for i, f in zip(request["active"], active):
        orbitals[i] = f

    integrals = fe.Integrals(madworld=world)
    result = {
        "h": integrals.compute_kinetic_integrals(orbitals)
        + integrals.compute_potential_integrals(orbitals, vnuc),
        "g": integrals.compute_two_body_integrals(orbitals, ordering="mulliken").elems,
        "converged": converged,
    }
    for i, f in enumerate(orbitals):
        f.save_to_file(os.path.join(workdir, f"orbital_{i}"))

    # the madness objects have to go before their world, otherwise madness crashes on finalize
    del orbitals, core, active, opti, integrals, vnuc, world
    return result


# State average solvers: f(c, h, g, n_electrons, n_roots, **kwargs) -> (energies of the roots, sa rdm1, sa rdm2)
# c, h, g: active space effective hamiltonian (g in chem ordering), energies include c
# rdms spin summed and equally weighted over the roots, in dirac ordering as OrbitalRefinement expects
# New solvers: write the function and add it to _SA_SOLVERS (it runs in the worker, next to madness)
def sa_fci(c, h, g, n_electrons, n_roots, **kwargs):
    """pyscf singlet FCI (as examples/excited_states/h4_mra_pno_fci.py), kwargs go to fci.direct_spin0.kernel"""
    from pyscf import fci

    n = h.shape[0]
    energies, civecs = fci.direct_spin0.kernel(
        h, g, n, n_electrons, nroots=n_roots, **kwargs
    )
    if n_roots == 1:
        energies, civecs = [energies], [civecs]
    rdms = [fci.direct_spin0.make_rdm12(v, n, n_electrons) for v in civecs]
    rdm1 = numpy.mean([r[0] for r in rdms], axis=0)
    rdm2 = numpy.mean([r[1] for r in rdms], axis=0).swapaxes(
        1, 2
    )  # pyscf rdm2[p, q, r, s] = <p^ r^ s q>
    return numpy.asarray(energies) + c, rdm1, rdm2


def sa_dmrg(
    c, h, g, n_electrons, n_roots, bond_dim=100, n_sweeps=10, n_threads=8, **kwargs
):
    """block2 SU2 DMRG (as examples/excited_states/h4_mra_pno_dmrg.py), kwargs go to driver.dmrg"""
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    driver = DMRGDriver(
        scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=n_threads
    )  # the cwd is the worker dir
    driver.initialize_system(n_sites=h.shape[0], n_elec=n_electrons, spin=0)
    mpo = driver.get_qc_mpo(h1e=h, g2e=g, ecore=c, iprint=0)
    ket = driver.get_random_mps(tag="KET", bond_dim=bond_dim, nroots=n_roots)
    dmrg_kwargs = {
        "bond_dims": [bond_dim],
        "noises": [1e-5] * 4 + [0],
        "thrds": [1e-10] * 8,
        "iprint": 0,
        **kwargs,
    }
    energies = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, **dmrg_kwargs)
    kets = (
        [driver.split_mps(ket, i, tag=f"KET-{i}") for i in range(n_roots)]
        if n_roots > 1
        else [ket]
    )
    rdm1 = numpy.mean([driver.get_1pdm(k) for k in kets], axis=0)
    # block2 2pdm -> chem (transpose) -> dirac (swapaxes), as in the example
    rdm2 = (
        numpy.mean([driver.get_2pdm(k) for k in kets], axis=0)
        .transpose(0, 3, 1, 2)
        .swapaxes(1, 2)
    )
    return numpy.atleast_1d(energies), rdm1, rdm2


_SA_SOLVERS = {"fci": sa_fci, "dmrg": sa_dmrg}


def _state_average_exct_state_refinement(request: dict, workdir: str) -> dict:
    """Worker side: state average orbital refinement loop, see FrayedEndsMixin.state_average_refinement
    solver (state average rdms) -> OrbitalRefinement -> effective hamiltonian of the new orbitals, until the mean energy converges
    The orbitals (all, in the order of the molecule) are written as orbital_{i} (SavedFct files)"""
    import frayedends as fe

    world = fe.MadWorld(ndims=3, **request["mra_parameters"])
    orbitals = [
        fe.SavedFct3D(os.path.join(request["input_dir"], f"orbital_{i}"))
        for i in range(request["n_orbitals"])
    ]
    vnuc = _nuclear_potential(fe, request, world)

    solver = _SA_SOLVERS[request["solver"]]
    refinement = request["refinement"]
    init_kwargs = {k: v for k, v in refinement.items() if k in _REFINEMENT_INIT_KEYS}
    refine_kwargs = {
        k: v for k, v in refinement.items() if k not in _REFINEMENT_INIT_KEYS
    }
    core = [orbitals[i] for i in request["core"]]
    active = [orbitals[i] for i in request["active"]]
    c, h, g = request["c"], numpy.array(request["h"]), numpy.array(request["g"])

    history = []
    converged = False
    for iteration in range(request["maxiter"] + 1):
        energies, rdm1, rdm2 = solver(
            c,
            h,
            g,
            request["n_electrons"],
            request["n_roots"],
            **request["solver_kwargs"],
        )
        history.append(energies)
        if (
            iteration > 0
            and abs(numpy.mean(energies) - numpy.mean(history[-2])) < request["econv"]
        ):
            converged = True
            break
        if iteration == request["maxiter"]:
            break
        opti = fe.OrbitalRefinement(
            world, vnuc, request["nuclear_repulsion"], **init_kwargs
        )
        core, active, _ = opti.refine_orbitals(
            orbitals=[core, active], rdm1=rdm1, rdm2=rdm2, **refine_kwargs
        )
        c, h, g = opti.get_effective_hamiltonian(g_ordering="chem")
        del opti

    # orbitals neither core nor active (frozen virtuals) are kept as they are
    for i, f in zip(request["core"], core):
        orbitals[i] = f
    for i, f in zip(request["active"], active):
        orbitals[i] = f

    integrals = fe.Integrals(madworld=world)
    result = {
        "h": integrals.compute_kinetic_integrals(orbitals)
        + integrals.compute_potential_integrals(orbitals, vnuc),
        "g": integrals.compute_two_body_integrals(orbitals, ordering="mulliken").elems,
        "energies": numpy.array(history),
        "converged": converged,
    }
    for i, f in enumerate(orbitals):
        f.save_to_file(os.path.join(workdir, f"orbital_{i}"))

    # the madness objects have to go before their world, otherwise madness crashes on finalize
    del orbitals, core, active, integrals, vnuc, world
    return result


_TASKS = {
    "project": _project,
    "pno": _pno,
    "cube_plot": _cube_plot,
    "transform_orbitals": _transform_orbitals,
    "refine_orbitals": _refine_orbitals,
    "state_average_exct_state_refinement": _state_average_exct_state_refinement,
}


if __name__ == "__main__":
    workdir = sys.argv[1]
    with open(os.path.join(workdir, "request.json")) as f:
        request = json.load(f)
    result = _TASKS[request["task"]](request, workdir)
    numpy.savez(os.path.join(workdir, "result.npz"), **result)
