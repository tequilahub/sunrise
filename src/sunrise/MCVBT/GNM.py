import tequila as tq
import scipy
import numpy as np
from sunrise.MCVBT.gem import gem_fast
from tequila.quantumchemistry import QuantumChemistryBase

import sunrise as sn
from sunrise.MCVBT.Big_Exp import BigExpVal

from typing import Dict
import csv
import os


class mcvbt:

    def __init__(self, mol: QuantumChemistryBase, graphs: list, circuits=None, strategy=None, solver="FQE",
                 filename="mcvbt_results.csv", overwrite_file=True, silent=True):

        self.mol = mol
        self.graphs = graphs
        self.strategy = strategy
        self.solver = solver
        self.circuits = circuits

        self.variables_preopt = {}
        self.results = {}
        self.csvfile_name = filename
        self.final_variables = []
        self.silent = silent

        if overwrite_file and os.path.isfile(self.csvfile_name):
           os.remove(self.csvfile_name)

        if self.circuits is not None:
            if len(self.circuits) != len(self.graphs):
                raise ValueError("Number of circuits must be equal to number of graphs")


    def calculate_groundstate(self,init_strategy: str = "pre-optimize"):

        if self.circuits is None:
            self.__create_circuits()

        if init_strategy == "pre-optimize":
            if not self.silent: print("Start pre-optimization of individual circuits")
            variables_preopt, energies = self.__preoptimize_variables()
            self.results[(1, 0)] = min(energies)
            if not self.silent: print("End of pre-optimization")
        elif init_strategy == "zeros":
            variables_preopt = {}
            for circ in self.circuits:
                for v in circ.variables:
                    variables_preopt[v] = 0.01
            self.results[(1, 0)] = None
        elif init_strategy == "random":
            variables_preopt = {}
            for circ in self.circuits:
                for v in circ.variables:
                    variables_preopt[v] = np.random.uniform(0, 2*np.pi)
            self.results[(1, 0)] = None
        else:
            raise ValueError("Unknown init_strategy {}".format(init_strategy))


        # self.strategy="shift" #todo check strategy
        if self.strategy is not None:
            self.__add_delocalization(variables_preopt=variables_preopt)

        with open(self.csvfile_name, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.mol.compute_energy("fci")])

        if not self.silent: print("Start G(N,M)  optimization")
        N = len(self.circuits)
        for n in range(2, N+1):
            if not self.silent: print("Start G({},0) optimization".format(n))
            v, _ = gem_fast(circuits=self.circuits[:n], solver=self.solver, variables=variables_preopt, mol=self.mol)
            if not self.silent: print("End G({},0)   optimization".format(n))
            self.results[(n, 0)] = v[0]


        variables = {**variables_preopt}
        start = 2
        for m in range(1,N+1):
            if m > start:
                start = m
            for n in range(start, N+1):
                with open(self.csvfile_name, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["G({},{})".format(n,m)])
                if not self.silent: print("Start G({},{}) optimization".format(n,m))
                v, _, variables = GNM(circuits=self.circuits[:n], variables=variables, solver=self.solver,
                                      mol=self.mol,filename=self.csvfile_name, M=m, max_iter=10)
                if not self.silent: print("End G({},{})   optimization".format(n,m))
                self.results[(n, m)] = v[0]
                self.final_variables = variables

        return self.results


    def compare_to_fci(self):

        fci = self.mol.compute_energy("fci")
        print("G(N,M) errors:")
        print(self.results)
        for k, v in self.results.items():

            error = abs(fci - v)
            print("G({},{}): {:+2.5f} Ha".format(k[0], k[1], error))


    def __preoptimize_variables(self):

        energies = []
        for preopt_circuit in self.circuits:
            if self.solver == "FQE":
                E_preopt = sn.expval.FQEBraKet(ket=preopt_circuit, molecule=self.mol)

                variables = preopt_circuit.variables
                init_vars = {vs: 0 for vs in variables}

                for vs in init_vars:
                    if "R" in str(vs):
                        init_vars[vs] = np.pi/2

                x0 = list(init_vars.values())

                r = scipy.optimize.minimize(fun=E_preopt, x0=x0, jac="2-point", method="l-bfgs-b",
                                            options={"finite_diff_rel_step": 1.e-5, "disp": False})
                energies.append(r.fun)

                r_variabales = {vs: r.x[i].real for i, vs in enumerate(init_vars)}
                self.variables_preopt = {**self.variables_preopt, **r_variabales}

            elif self.solver == "Qulacs":
                E_preopt = tq.ExpectationValue(U=preopt_circuit, H=self.mol.make_hamiltonian())

                result = tq.minimize(E_preopt, silent=True)

                energies.append(result.energy)

                self.variables_preopt = {**self.variables_preopt, **result.variables}



        self.results[(1, 0)] = min(energies)

        return self.variables_preopt, energies


    def __create_circuits(self):

        if (self.solver == "FQE") or (self.solver == "TCC"):
            self.circuits = []
            for i, edges in enumerate(self.graphs):
                U = sn.FCircuit.from_edges(edges=edges, label="G{}".format(i), n_orb=self.mol.n_orbitals)
                for j, e in enumerate(edges):
                    U += sn.gates.FermionicExcitation(indices=[(2 * e[0], 2 * e[1])],
                                                      variables="(R{}_{})".format(i, j))
                    U += sn.gates.FermionicExcitation(indices=[(2 * e[0] + 1, 2 * e[1] + 1)],
                                                      variables="(R{}_{})".format(i, j))
                self.circuits.append(U)

        elif self.solver == "Qulacs":
            self.circuits = []
            for i, edges in enumerate(self.graphs):
                U = self.mol.make_ansatz(name="SPA", edges=edges, label="G{}".format(i))
                for j, e in enumerate(edges):
                    U += self.mol.UR(i=e[0], j=e[1], label="R{}_{}".format(i, j))
                self.circuits.append(U)


    def __add_delocalization(self, variables_preopt:Dict[tq.Variable, float]):

        aux_circuits=[]
        for i, circ in enumerate(self.circuits):
            if self.strategy == "shift":


                flat = [x for tup in self.graphs[i] for x in tup]

                # shift cyclically
                shifted = flat[1:] + flat[:1]

                # regroup into tuples of original size
                tuple_size = len(self.graphs[i][0])
                shifted_graph = [tuple(shifted[i:i + tuple_size]) for i in range(0, len(shifted), tuple_size)]

                for edge in shifted_graph:
                    circ += sn.gates.FermionicExcitation(indices=[(2 * edge[0], 2 * edge[1])],
                                                          variables="shift{}_{}".format(i, edge))

                    circ += sn.gates.FermionicExcitation(indices=[(2 * edge[0] + 1, 2 * edge[1] + 1)],
                                                          variables="shift{}_{}".format(i, edge))


            elif self.strategy == "triangle":
                raise NotImplementedError
            else:
                raise NotImplementedError

            aux_circuits.append(circ)
            filtered_variables= [x for x in circ.variables if x not in variables_preopt]
            for v in filtered_variables:
                variables_preopt[v] = 0.0

        self.circuits = aux_circuits

        return variables_preopt


    def get_final_variables(self):
        return self.final_variables


def GNM(circuits, variables, solver, mol, filename, silent=True, max_iter=10, M=None):

    # circuits = [x for x in circuits]
    N = len(circuits)
    if M is None:
        M = len(circuits)

    for i in range(M, N): #map pre opt variables to current circuit set and overrite old circuits
        U = circuits[i]
        U = U.map_variables(variables)
        circuits[i] = U

    vkeys = []
    for U in circuits:
        vkeys += U.extract_variables()

    variables = {**{k: 0.0 for k in vkeys if k not in variables}, **variables}

    v, vv = gem_fast(circuits=circuits, variables=variables, mol=mol, solver=solver)
    x0 = {k: variables[k] for k in vkeys}


    coeffs = []
    for i in range(len(circuits)):
        c = tq.Variable(("c", i))
        coeffs.append(c)
        x0[c] = vv[i, 0]
        vkeys.append(c)

    energy = 1.0

    callback_energies = []
    def callback(x):

        energy = mcvbt_exp(x)
        if not silent:
            print("current energy: {:+2.4f}".format(energy))
        callback_energies.append(energy)

        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([callback_energies[-1]])


    mcvbt_exp = BigExpVal(circuits=circuits, coefficcents=coeffs, mol=mol, solver=solver, H=mol.make_hamiltonian())
    disp = not silent
    for i in range(max_iter):
        if not silent: print("iteration {}".format(i))
        result = scipy.optimize.minimize(mcvbt_exp, x0=list(x0.values()), jac="2-point", method="slsqp",
                                         options={"finite_diff_rel_step":5.e-5, "disp": disp, "maxiter": 100},
                                         callback=callback)

        x0 = {vkeys[i]: result.x[i] for i in range(len(result.x))}
        v, vv = gem_fast(circuits=circuits, variables=x0, mol=mol, solver=solver)

        for i in range(len(coeffs)):
            x0[coeffs[i]] = vv[i, 0]

        if np.isclose(energy, v[0], atol=1.e-4):
            if not silent: print("not converged")
            if not silent: print(energy)
            if not silent: print(v[0])
            energy = v[0]
        else:
            energy = v[0]
            break

    for k in vkeys:
        variables[k] = x0[k]

    return v, vv, variables