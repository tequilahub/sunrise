from __future__ import annotations

import csv
import os
import warnings
from copy import deepcopy

import numpy as np
import tequila as tq
from tequila import SUPPORTED_BACKENDS, QCircuit, TequilaException, TequilaWarning
from tequila.objective.objective import Variables

import sunrise as sn
from sunrise.expval import SUPPORTED_FERMIONIC_BACKENDS, Braket
from sunrise.MCVBT.big_exp import BigExpVal
from sunrise.MCVBT.gem import gem_fast
from sunrise.molecules.fermionic_base import FermionicBase
from sunrise.molecules.qubit_base import QuantumChemistryBase


class mcvbt:
    def __init__(
        self,
        mol: QuantumChemistryBase,
        graphs: list[list[tuple[int]]],
        circuits: list[QCircuit | sn.FCircuit] | None = None,
        strategy: str | None = None,
        solver: str = "FQE",
        solver_args: dict | None = None,
        filename: str = "mcvbt_results.csv",
        overwrite_file: bool = True,
        silent=False,
        tol: float = 1.0e-5,
    ):

        self.mol = mol
        self.graphs = graphs
        self.strategy = strategy
        self.solver = solver
        self.circuits = circuits

        self.variables_preopt = {}
        self.results = {}
        self.csvfile_name = filename
        self.final_variables = {}
        self.silent = silent
        if solver_args is None:
            solver_args = {}
        self.solver_args = solver_args
        self.tol = tol
        if overwrite_file and os.path.isfile(self.csvfile_name):
            os.remove(self.csvfile_name)

        if self.circuits is not None and len(self.circuits) != len(self.graphs):
            raise ValueError("Number of circuits must be equal to number of graphs")

    def calculate_groundstate(self, init_strategy: str = "pre-optimize"):

        if self.circuits is None:
            self.__create_circuits()

        if init_strategy == "pre-optimize":
            if not self.silent:
                print("Start pre-optimization of individual circuits")
            variables_preopt, energies = self.__preoptimize_variables()
            self.results[(1, 0)] = min(energies)
            if not self.silent:
                print("End of pre-optimization")
        elif init_strategy == "zeros":
            variables_preopt = {}
            for circ in self.circuits:
                for v in circ.extract_variables():
                    variables_preopt[v] = 0.01
            self.results[(1, 0)] = 0.0
        elif init_strategy == "random":
            variables_preopt = {}
            for circ in self.circuits:
                for v in circ.extract_variables():
                    variables_preopt[v] = np.random.uniform(0, 2 * np.pi)
            self.results[(1, 0)] = 0.0
        else:
            raise ValueError(f"Unknown init_strategy {init_strategy}")

        # self.strategy="shift" #todo check strategy
        if self.strategy is not None:
            self.__add_delocalization(variables_preopt=variables_preopt)

        if not self.silent:
            with open(self.csvfile_name, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.mol.compute_energy("fci")])

        if not self.silent:
            print("Start G(N,M)  optimization")
        N = len(self.circuits)
        for n in range(2, N + 1):
            if not self.silent:
                print(f"Start G({n},0) optimization")
                print("variables preopt", variables_preopt)
            v, _ = gem_fast(
                circuits=self.circuits[:n],
                solver=self.solver,
                variables=variables_preopt,
                mol=self.mol,
                silent=self.silent,
            )
            if not self.silent:
                print(f"End G({n},0)   optimization")
            self.results[(n, 0)] = v[0]
        self.final_variables = {**variables_preopt}
        self.final_variables[("c", 0)] = 1.0

        variables = {**variables_preopt}
        start = 2
        for m in range(1, N + 1):
            start = max(start, m)
            for n in range(start, N + 1):
                if not self.silent:
                    with open(self.csvfile_name, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([f"G({n},{m})"])
                if not self.silent:
                    print(f"Start G({n},{m}) optimization")
                v, _, variables = GNM(
                    circuits=self.circuits[:n],
                    variables=variables,
                    solver=self.solver,
                    mol=self.mol,
                    filename=self.csvfile_name,
                    M=m,
                    max_iter=10,
                    silent=self.silent,
                    tol=self.tol,
                    **self.solver_args,
                )
                if not self.silent:
                    print(f"End G({n},{m})   optimization")
                self.results[(n, m)] = v[0]
                self.final_variables = variables

        return self.results

    def compare_to_fci(self):

        fci = self.mol.compute_energy("fci")
        print("G(N,M) errors:")
        print(self.results)
        for k, v in self.results.items():
            error = fci - v
            print(f"G({k[0]},{k[1]}): {error:+2.5f} Ha")

    def __preoptimize_variables(self, *args, **kwargs):

        energies = []
        for preopt_circuit in self.circuits:
            if self.solver.lower() in SUPPORTED_FERMIONIC_BACKENDS:
                E_preopt = Braket(ket=preopt_circuit, mol=self.mol, operator="H")
                variables = preopt_circuit.variables
                init_vars = {vs: 0 for vs in variables}

                for gate in preopt_circuit.gates:
                    if gate.name == "UR":
                        init_vars[gate.variables] = np.pi / 2
                r = sn.minimize(
                    E_preopt,
                    *args,
                    silent=self.silent,
                    initial_values=init_vars,
                    fbackend=self.solver,
                    **self.solver_args,
                )
                energies.append(r.energy)

                self.variables_preopt = {**self.variables_preopt, **r.angles}
            elif self.solver.lower() in SUPPORTED_BACKENDS:
                E_preopt = tq.ExpectationValue(
                    U=preopt_circuit, H=self.mol.make_hamiltonian()
                )

                variables = E_preopt.extract_variables()
                init_vars = {vs: 0 for vs in variables}

                for vs in init_vars:
                    if "R" in str(vs):
                        init_vars[vs] = 0.01
                result = sn.minimize(
                    E_preopt,
                    *args,
                    silent=self.silent,
                    initial_values=init_vars,
                    backend=self.solver,
                    **self.solver_args,
                )

                energies.append(result.energy)

                self.variables_preopt = {**self.variables_preopt, **result.variables}

        self.results[(1, 0)] = min(energies)

        return self.variables_preopt, energies

    def __create_circuits(self):
        if self.solver.lower() in SUPPORTED_FERMIONIC_BACKENDS:
            self.circuits = []
            for i, edges in enumerate(self.graphs):
                U = sn.FCircuit.from_edges(
                    edges=edges, label=f"G{i}", n_orb=self.mol.n_orbitals
                )
                for j, e in enumerate(edges):
                    U += sn.gates.FermionicExcitation(
                        indices=[(2 * e[0], 2 * e[1])],
                        variables=f"(R{i}_{j})",
                    )
                    U += sn.gates.FermionicExcitation(
                        indices=[(2 * e[0] + 1, 2 * e[1] + 1)],
                        variables=f"(R{i}_{j})",
                    )
                self.circuits.append(U)
        elif self.solver.lower() in SUPPORTED_BACKENDS:
            mol = self.mol
            if isinstance(mol, FermionicBase):
                warnings.warn(
                    f"Fermionic Base provided while qubit base solver ({self.solver})\n Converting molecule to qubit with reordered-JW",
                    TequilaWarning,
                )
                mol = QuantumChemistryBase(
                    parameters=mol.parameters,
                    integral_manager=mol.integral_manager,
                    transformation="reordered-jordan-wigner",
                )
            self.circuits = []
            for i, edges in enumerate(self.graphs):
                U = mol.make_ansatz(name="SPA", edges=edges, label=f"G{i}")
                for j, e in enumerate(edges):
                    U += mol.UR(i=e[0], j=e[1], label=f"R{i}_{j}")
                self.circuits.append(U)

    def __add_delocalization(self, variables_preopt: dict[tq.Variable, float]):
        aux_circuits = []
        for i, circ in enumerate(self.circuits):
            if isinstance(circ, QCircuit):
                raise TequilaException("Strategies not implemented for QCircuit yet.")
            if self.strategy == "shift":
                flat = [x for tup in self.graphs[i] for x in tup]

                # shift cyclically
                shifted = flat[1:] + flat[:1]

                # regroup into tuples of original size
                tuple_size = len(self.graphs[i][0])
                shifted_graph = [
                    tuple(shifted[i : i + tuple_size])
                    for i in range(0, len(shifted), tuple_size)
                ]

                for edge in shifted_graph:
                    circ += sn.gates.FermionicExcitation(
                        indices=[(2 * edge[0], 2 * edge[1])],
                        variables=f"shift{i}_{edge}",
                    )

                    circ += sn.gates.FermionicExcitation(
                        indices=[(2 * edge[0] + 1, 2 * edge[1] + 1)],
                        variables=f"shift{i}_{edge}",
                    )

            elif self.strategy == "triangle":
                raise NotImplementedError
            else:
                raise NotImplementedError

            aux_circuits.append(circ)
            filtered_variables = [
                x for x in circ.variables if x not in variables_preopt
            ]
            for v in filtered_variables:
                variables_preopt[v] = 0.0

        self.circuits = aux_circuits

        return variables_preopt

    def get_final_variables(self):
        return self.final_variables


def GNM(
    circuits: list[QCircuit | sn.FCircuit],
    variables: Variables,
    solver: str,
    mol: QuantumChemistryBase,
    filename: str,
    silent: bool = True,
    max_iter: int = 10,
    tol: float = 1.0e-5,
    M: int | None = None,
    solver_args: dict | None = None,
):
    if solver_args is None:
        solver_args = {}
    # circuits = [x for x in circuits]
    N = len(circuits)
    if M is None:
        M = len(circuits)

    for i in range(
        M, N
    ):  # map pre opt variables to current circuit set and overrite old circuits
        U = deepcopy(circuits[i])
        U = U.map_variables(variables)
        circuits[i] = U

    vkeys = []
    for U in circuits:
        vkeys += [k for k in U.extract_variables() if k not in vkeys]

    variables = {**{k: 0.0 for k in vkeys if k not in variables}, **variables}

    v, vv = gem_fast(circuits=circuits, variables=variables, mol=mol, solver=solver)
    x0 = {k: variables[k] for k in vkeys}
    angles_keys = deepcopy(vkeys)

    coeffs = []
    for i in range(len(circuits)):
        c = tq.Variable(("c", i))
        coeffs.append(c)
        x0[c] = vv[i, 0]
        vkeys.append(c)

    energy = 1.0

    callback_energies = []

    def callback(x):

        energy = sn.simulate(mcvbt_exp, variables=x, backend=solver)
        if not silent:
            print(f"current energy: {energy:+2.4f}")
        callback_energies.append(energy)

        if not silent:
            with open(filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([callback_energies[-1]])

    # the fermionic brakets build their own hamiltonian, only qubit circuits need it
    H = (
        {}
        if solver.lower() in SUPPORTED_FERMIONIC_BACKENDS
        else {"H": mol.make_hamiltonian()}
    )
    mcvbt_exp = BigExpVal(circuits=circuits, coefficcents=coeffs, mol=mol, **H)
    for i in range(max_iter):
        if not silent:
            print(f"iteration {i}")
        # fermionic first: "spex" is both a fermionic and a tequila qubit backend
        if solver.lower() in SUPPORTED_FERMIONIC_BACKENDS:
            result = sn.minimize(
                mcvbt_exp,
                initial_values=x0,
                method="bfgs",
                silent=silent,
                callback=callback,
                backend=solver,
                variables=angles_keys,
                **solver_args,
            )
        else:
            result = tq.minimize(
                mcvbt_exp,
                initial_values=x0,
                method="bfgs",
                silent=silent,
                gradient="2-point",
                method_options={"disp": not silent, "maxiter": 100},
                callback=callback,
                backend=solver,
                variables=angles_keys,
                **solver_args,
            )
        v, vv = gem_fast(
            circuits=circuits, variables=result.variables, mol=mol, solver=solver
        )

        x0.update({k: result.variables[k] for k in angles_keys})
        for j in range(len(coeffs)):
            x0[coeffs[j]] = vv[j, 0]

        if not np.isclose(energy, v[0], atol=tol):
            if not silent:
                print("not converged")
            if not silent:
                print(energy)
            if not silent:
                print(v[0])
            energy = v[0]
        else:
            if not silent:
                print(f"converged after {i + 1} iterations with energy {energy}")
            energy = v[0]
            break

    for k in vkeys:
        variables[k] = x0[k]

    return v, vv, variables
