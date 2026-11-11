"""
Solvers for FrayedEndsMixin.refine_orbitals: f(mol, **kwargs) -> (energy, rdm1, rdm2)
The rdms are spin summed in dirac ordering, as mol.compute_rdms (and frayedends OrbitalRefinement) expect:
    rdm1[p, q] = <p^ q>,  rdm2[p, q, r, s] = <p^ q^ s r>
They run in the sunrise process, only the orbital refinement runs in the frayedends worker, so any method works here
New solvers: write the function and add it to SOLVERS
"""

from __future__ import annotations

import importlib.util
import typing

import tequila as tq
from pyscf import fci

from sunrise.spafastprototype import spa_rdms, spa_solve_graph


def vqe(
    mol,
    ansatz: str,
    ansatz_kwargs: dict | None = None,
    silent: bool = True,
    **minimize_kwargs,
):
    """VQE with any ansatz of mol.make_ansatz, minimize_kwargs go to tq.minimize"""
    if ansatz.upper() in ["HCB-SPA", "SPA"]:
        return spa(mol=mol, **(ansatz_kwargs or {}))
    U = mol.make_ansatz(name=ansatz, **(ansatz_kwargs or {}))
    H = mol.make_hamiltonian()
    result = tq.minimize(
        tq.ExpectationValue(H=H, U=U), silent=silent, **minimize_kwargs
    )
    rdm1, rdm2 = mol.compute_rdms(U=U, variables=result.variables)
    return result.energy, rdm1, rdm2


def spa(mol, edges: list | None = None, **kwargs):
    """
    SPA, edges in active space indices (e.g. [(0, 1), (2, 3)])
    edges of at most 2 orbitals: classical solver of spafastprototype (no circuit, as SPASolver), kwargs are ignored
    otherwise VQE (without edges: PNO structure of the molecule)
    """
    if edges is not None and all(len(edge) < 3 for edge in edges):
        variables, energy = spa_solve_graph(mol=mol, graph=edges)
        rdm1, rdm2 = spa_rdms(
            edges=edges, variables=variables, n_orbitals=mol.n_orbitals
        )
        return energy, rdm1, rdm2
    return vqe(mol, "SPA", {} if edges is None else {"edges": edges}, **kwargs)


def upccgsd(mol, **kwargs):
    return vqe(mol, "UpCCGSD", **kwargs)


def upccd(mol, **kwargs):
    return vqe(mol, "UpCCD", **kwargs)


def fci_solver(mol, **kwargs):
    """pyscf FCI in the active space, kwargs go to fci.direct_spin1.kernel"""
    c, h, g = mol.get_integrals(ordering="chem")
    energy, civec = fci.direct_spin1.kernel(
        h, g.elems, mol.n_orbitals, mol.n_electrons, **kwargs
    )
    rdm1, rdm2 = fci.direct_spin1.make_rdm12(civec, mol.n_orbitals, mol.n_electrons)
    # pyscf: rdm2[p, q, r, s] = <p^ r^ s q>
    return energy + c, rdm1, rdm2.transpose(0, 2, 1, 3)


SOLVERS = {
    "spa": spa,
    "upccgsd": upccgsd,
    "upccd": upccd,
    "fci": fci_solver,
}


def get_solver(solver: str | typing.Callable) -> typing.Callable:
    """
    solver: a name in SOLVERS, a callable, or "path/to/file.py:function" for a function in a custom file
    (the user is responsible that it follows the solver signature above)
    """
    if callable(solver):
        return solver
    if solver.lower() in SOLVERS:
        return SOLVERS[solver.lower()]
    if ":" in solver:
        path, name = solver.rsplit(":", 1)
        spec = importlib.util.spec_from_file_location("sunrise_custom_solver", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, name)
    raise ValueError(
        f"unknown solver {solver}, available: {list(SOLVERS)}, a callable or 'file.py:function'"
    )
