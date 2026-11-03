"""
Fermionic backends benchmark: time to reach the minimum on linear H chains (STO-3G, 0.9 A), spex vs tcc, fqe, ffsim,
qulacs.

Two workloads, every backend doing the same calculation from the same start:
  spa-oo   SPA + orbital optimization: sn.optimize_orbitals on the SPA circuit from the bonding/antibonding
           pair orbitals (the initial guess of arXiv:2606.26393), every VQE from all angles = 0, until
           pyscf's CASSCF converges
  upccgsd  UpCCGSD VQE from all angles = 0 in canonical Hartree-Fock orbitals
Every VQE: BFGS, exact parameter-shift gradients, gradient tolerance 1e-6. spex, tcc, fqe and ffsim run
through sunrise's fermionic backends, qulacs through tequila (Jordan-Wigner).
Each point runs in its own process alone on one physical core with one thread (the hyperthread siblings
stay idle), every backend goes through the sizes one after the other, smallest first, and stops at the
first size that takes longer than LIMIT or fails. Run it on an otherwise idle machine with at least as
many physical cores as backends, and never mix machines in one figure.

    python miscellaneous/benchmark/fbackends_benchmark.py spa-oo          # run, writes fbackends_benchmark_spa-oo.json and .png
    python miscellaneous/benchmark/fbackends_benchmark.py spa-oo ffsim    # run only ffsim, the other backends stay in the json
A run keeps the points of its json that reached the minimum and computes only the missing ones; for a
fair comparison start from an empty json, so that every point comes from the same code and machine.
    python miscellaneous/benchmark/fbackends_benchmark.py upccgsd --plot  # only the figure, from the json
"""

import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKLOADS = {
    "spa-oo": "SPA + orbital optimization from the pair orbitals",
    "upccgsd": "UpCCGSD VQE from angles 0",
}
BACKENDS = ["spex", "tcc", "fqe", "ffsim", "qulacs"]
SIZES = list(range(2, 31, 2))  # H atoms
LIMIT = 3600  # seconds per point
MAXITER = 10000  # BFGS iterations and CASSCF macro-iterations, high enough to always reach the minimum


def point(workload, backend, n):
    """time to the minimum of H_n with one backend, printed as json"""
    import numpy as np
    import tequila as tq
    from openfermion.ops.operators.symbolic_operator import SymbolicOperator

    import sunrise as sn
    from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase
    from sunrise.spafastprototype.spa import make_initial_orbital_guess

    # openfermion drops coefficients below 1e-8 while it builds the qubit Hamiltonian (~3e-8 Ha);
    # 1e-12 as spex uses, so that qulacs gets the same Hamiltonian as the fermionic backends
    SymbolicOperator._issmall.__defaults__ = (1.0e-12,)
    SymbolicOperator.compress.__defaults__ = (1.0e-12,)

    geometry = "\n".join(f"H 0.0 0.0 {0.9 * i}" for i in range(n))
    mol = sn.Molecule(geometry=geometry, basis_set="sto-3g", nature="f")
    if workload == "spa-oo":
        mol = mol.use_native_orbitals()
        edges = [(2 * k, 2 * k + 1) for k in range(n // 2)]
        guess = make_initial_orbital_guess(edges)
        mol = mol.transform_orbitals(guess / np.linalg.norm(guess, axis=0))
        U = mol.make_ansatz("SPA", edges=edges)
    else:
        U = mol.make_ansatz("UpCCGSD")
    qmol = QuantumChemistryBase(
        parameters=mol.parameters,
        transformation="REORDEREDJORDANWIGNER",
        integral_manager=mol.integral_manager,
    )
    vqe = {"initial_values": 0.0, "tol": 1.0e-6, "maxiter": MAXITER}

    start = time.perf_counter()
    if workload == "spa-oo":
        oo = {
            "silent": True,
            "return_mcscf": True,
            "pyscf_arguments": {"max_cycle_macro": MAXITER, "max_cycle_micro": 3},
        }
        if backend == "qulacs":
            r = sn.optimize_orbitals(
                molecule=qmol,
                circuit=U.to_upthendown(n).to_qcircuit(qmol),
                vqe_solver_arguments={
                    "optimizer_arguments": {"backend": "qulacs", **vqe}
                },
                **oo,
            )
        else:  # the fermionic backends: spex, tcc, fqe, ffsim
            r = sn.optimize_orbitals(
                molecule=mol, circuit=U, backend=backend, vqe_solver_arguments=vqe, **oo
            )
        iterations, converged = r.iterations, bool(r.mcscf_object.converged)
    else:
        vqe.update(method="bfgs", silent=True)
        if backend == "qulacs":
            E = tq.ExpectationValue(
                U=U.to_upthendown(n).to_qcircuit(qmol), H=qmol.make_hamiltonian()
            )
            r = tq.minimize(E, backend="qulacs", **vqe)
        else:  # the fermionic backends: spex, tcc, fqe, ffsim
            r = sn.minimize(
                sn.ExpectationValue(U=U, H="H", mol=mol), backend=backend, **vqe
            )
        iterations = len(r.history.energies)
        converged = iterations < MAXITER
    seconds = time.perf_counter() - start
    print(
        json.dumps(
            {
                "energy": r.energy,
                "seconds": seconds,
                "iterations": iterations,
                "converged": converged,
            }
        )
    )


def run(workload, backends):
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONWARNINGS="ignore",
    )

    # one logical CPU per physical core this process may use: the first of its hyperthread siblings
    cores = queue.Queue()
    for cpu in sorted(os.sched_getaffinity(0)):
        siblings = Path(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        ).read_text()
        if int(siblings.replace("-", ",").split(",")[0]) == cpu:
            cores.put(str(cpu))
    # the points that already reached the minimum stay, only the missing ones are computed
    path = HERE / f"fbackends_benchmark_{workload}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    results = {b: [r for r in old.get(b, []) if "seconds" in r] for b in backends}

    def job(backend):
        # the sizes one after the other: a size only starts once the one before reached the minimum
        for n in SIZES:
            if any(r["n"] == n for r in results[backend]):
                continue
            core = cores.get()
            cmd = [
                "taskset",
                "-c",
                core,
                sys.executable,
                __file__,
                "--point",
                workload,
                backend,
                str(n),
            ]
            # no memory cap: a point may use all memory of the machine
            try:
                out = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=LIMIT,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                row = {"n": n, "timeout": True}
            else:
                if out.returncode == 0:
                    row = {"n": n, **json.loads(out.stdout.strip().splitlines()[-1])}
                elif out.returncode in (-9, 137) or "MemoryError" in out.stderr:
                    row = {"n": n, "error": "out of memory"}
                else:
                    row = {
                        "n": n,
                        "error": (out.stderr.strip().splitlines() or ["?"])[-1],
                    }
            finally:
                cores.put(core)
            results[backend].append(row)
            print(backend, row, flush=True)
            if "seconds" not in row:
                return

    with ThreadPoolExecutor(len(backends)) as pool:
        list(pool.map(job, backends))
    # the backends that were not run keep their earlier curves
    results = {
        b: results[b] if b in results else old[b]
        for b in BACKENDS
        if b in results or b in old
    }
    path.write_text(json.dumps(results, indent=1))


def plot(workload):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads((HERE / f"fbackends_benchmark_{workload}.json").read_text())
    colors = {
        "spex": "#d62728",
        "tcc": "#1f77b4",
        "fqe": "#2ca02c",
        "ffsim": "#ff7f0e",
        "qulacs": "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backend, rows in results.items():
        done = [r for r in rows if "seconds" in r]
        ax.plot(
            [r["n"] for r in done],
            [r["seconds"] for r in done],
            "-",
            marker="o",
            color=colors[backend],
            label=backend,
        )
        for r in rows:
            if "seconds" not in r:
                # shifted a little per backend, so that marks at the same size do not hide each other
                x = r["n"] + 0.2 * (BACKENDS.index(backend) - (len(BACKENDS) - 1) / 2)
                ax.plot(x, LIMIT, "x", color=colors[backend], ms=10, mew=2)
                if "error" in r:  # crashed, not over the time limit
                    ax.annotate(
                        "error",
                        (x, LIMIT),
                        xytext=(0, -14),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                        color=colors[backend],
                    )
    ax.axhline(LIMIT, color="k", lw=0.6, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(sorted({r["n"] for rows in results.values() for r in rows}))
    ax.set_xlabel("number of H atoms")
    ax.set_ylabel("time to reach the minimum [s]")
    ax.set_title(
        f"{WORKLOADS[workload]}, linear H chains (x: over {LIMIT // 3600} h or error)",
        fontsize=10,
    )
    ax.grid(alpha=0.3, which="both", lw=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / f"fbackends_benchmark_{workload}.png", dpi=150)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--point"]:
        point(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif (
        sys.argv[1:2] == []
        or sys.argv[1] not in WORKLOADS
        or not set(sys.argv[2:]) <= {"--plot", *BACKENDS}
    ):
        sys.exit(
            f"usage: fbackends_benchmark.py {{{','.join(WORKLOADS)}}} [{' '.join(BACKENDS)}] [--plot]"
        )
    elif sys.argv[2:3] == ["--plot"]:
        plot(sys.argv[1])
    else:
        run(sys.argv[1], [b for b in BACKENDS if b in sys.argv[2:]] or BACKENDS)
        plot(sys.argv[1])
