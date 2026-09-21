import time
import numpy as np
import tequila as tq

from mcvb_readout import solve_mcvb

geometry="""
C 	0.0000 	1.3970 	0.0000
C 	1.2098 	0.6985 	0.0000
C 	1.2098 	-0.6985 	0.0000
C 	0.0000 	-1.3970 	0.0000
C 	-1.2098 	-0.6985 	0.0000
C 	-1.2098 	0.6985 	0.0000
H 	0.0000 	2.4810 	0.0000
H 	2.1486 	1.2405 	0.0000
H 	2.1486 	-1.2405 	0.0000
H 	0.0000 	-2.4810 	0.0000
H 	-2.1486 	-1.2405 	0.0000
H 	-2.1486 	1.2405 	0.0000"""

def unique_edges(n):
    def gen(atoms):
        if not atoms:
            yield []
            return
        a = atoms[0]
        for i in range(1, len(atoms)):
            for rest in gen(atoms[1:i] + atoms[i+1:]):
                yield [(a, atoms[i])] + rest
    return list(gen(list(range(n))))



# the five non-crossing (Rumer) matchings on the chain 0-1-2-3-4-5
GRAPHS = [
    [(0, 1), (2, 3), (4, 5)], # Kekule
    [(0, 5), (1, 2), (3, 4)], #Kekule
    [(0, 1), (2, 5), (3, 4)],
    [(0, 3), (1, 2), (4, 5)],
    [(0, 5), (1, 4), (2, 3)],
]
# GRAPHS = unique_edges(6)

def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main():
    mol = tq.Molecule(geometry=geometry, basis_set="sto-3g", active_orbitals =[16, 19, 20, 21, 22, 23]).use_native_orbitals()


    H = mol.make_hamiltonian()
    e_fci = mol.compute_energy("fci")

    U = mol.make_ansatz(name="HCB-SPA", edges=GRAPHS[0])
    fixed = np.eye(6)
    for edge in GRAPHS[0]:
        fixed[edge[0], edge[1]] = 1.0
        fixed[edge[1], edge[0]] = -1.0

    fixed *= np.sqrt(2) / 2
    fixed = fixed.T # HEre initial guess is similar to fixed UR with pi/2
    opt = tq.quantumchemistry.optimize_orbitals(mol, circuit=U, use_hcb=True,
                                                initial_guess=fixed, silent=True)
    res = tq.minimize(tq.ExpectationValue(U=U, H=opt.molecule.make_hardcore_boson_hamiltonian()), silent=True)

    print("opt SPA", res.energy)
    print("opt SPA error", (res.energy-e_fci)*1000)

    print("benzol, sto-3g, {} orbitals, {} graphs"
          .format(mol.n_orbitals, len(GRAPHS)))
    print("FCI  {:+.10f}\n".format(e_fci))

    # ---- old method, fixed rotations --------------------------------------

    # ---- new method -------------------------------------------------------
    (_, e_0, c_0), t_0 = timed(
        lambda: solve_mcvb(mol, GRAPHS, iterations=0, verbose=False))
    (_, e_1, c_1), t_1 = timed(
        lambda: solve_mcvb(mol, GRAPHS, iterations=1, verbose=False))


    runs = [
            ("read-out only", e_0, c_0, t_0),
            ("+ descent 1", e_1, c_1, t_1),
    ]

    print("{:<16} {:>16} {:>12} {:>10}".format("", "E / Ha", "err / mEh", "t / s"))
    print("-" * 58)
    for name, e, c, t in runs:
        print("{:<16} {:+16.10f} {:12.3f} {:10.2f}"
              .format(name, e, (e - e_fci) * 1e3, t))
    print("-" * 58)


if __name__ == "__main__":
    main()
