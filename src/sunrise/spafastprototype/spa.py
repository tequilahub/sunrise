"""Orbital-optimized SPA (separable pair approximation) workflow: ``run_spa`` and the ``SPASolver`` used by it."""

from __future__ import annotations

import warnings

import numpy
import tequila as tq
from tequila import TequilaWarning
from tequila.optimizers.optimizer_base import OptimizerResults

from sunrise import FCircuit, optimize_orbitals

from ..expval.orbital_optimizer import OptimizeOrbitalsResult
from .decompose import decompose, make_decomposed_clusters
from .fast_qtensor import fast_qtensor
from .fast_rdm import fast_rdm
from .spa_classical_solver import spa_rdms, spa_solve_graph


def run_spa(
    mol,
    edges,
    initial_guess=None,
    decompose=True,
    silent=True,
    grouping: int | list[int] | None = None,
    backend: str = "qulacs",
    fast_rdm=True,
    **kwargs,
) -> OptimizeOrbitalsResult:
    """Optimize an SPA circuit together with the orbitals of a molecule.

    Wraps ``sunrise.optimize_orbitals`` with an ``SPASolver`` that
    exploits the separability of the SPA circuit.

    Parameters
    ----------
    mol : QuantumChemistryBase
        Molecule to optimize the orbitals of.
    edges : list of tuple of int, tq.QCircuit or FCircuit, optional
        Pairing of orbitals, e.g. ``[(0, 1), (2, 3)]``. A circuit is used directly;
        None builds the default ``"HCB-SPA"`` ansatz.
    initial_guess : numpy.ndarray or bool, optional
        Initial orbital rotation. If None or True, a guess built from ``edges``
        is used together with the native orbitals of ``mol``; if False, none.
    decompose : bool, optional
        Factorize expectation values over circuit clusters. Default is True.
    silent : bool, optional
        Suppress optimizer output. Default is True.
    grouping : int or list of int, optional
        Cluster size or explicit qubit clusters, see ``make_decomposed_clusters``.
    backend : str, optional
        Simulation backend. Default is ``"qulacs"``.
    fast_rdm : bool, optional
        Compute RDMs with ``fast_rdm``. Default is True.
    **kwargs
        Passed to ``sunrise.optimize_orbitals``.

    Returns
    -------
    OptimizeOrbitalsResult
        Optimized energy, orbitals and circuit variables.

    Examples
    --------
    >>> opt = sun.SPAFP.run_spa(mol=mol, edges=edges, initial_guess=False, grouping=12)
    >>> opt.energy
    """
    if edges is None:
        U = mol.make_ansatz(name="HCB-SPA")
    elif isinstance(edges, (tq.QCircuit, FCircuit)):
        U = edges
    else:
        U = mol.make_ansatz(name="HCB-SPA", edges=edges)
    if initial_guess is None:
        initial_guess = make_initial_orbital_guess(edges)
        mol = mol.use_native_orbitals()
    if isinstance(initial_guess, bool):
        if initial_guess:
            initial_guess = make_initial_orbital_guess(edges)
            mol = mol.use_native_orbitals()
        else:
            initial_guess = None
    grouping = make_decomposed_clusters(U, grouping)

    vqe_solver = SPASolver(
        decompose=decompose,
        grouping=grouping,
        backend=backend,
        fast_rdm=fast_rdm,
        edges=edges,
    )
    opt = optimize_orbitals(
        circuit=U,
        molecule=mol,
        initial_guess=initial_guess,
        vqe_solver=vqe_solver,
        silent=silent,
        use_hcb=True,
        vqe_solver_arguments={"optimizer_arguments": {"backend": backend}},
        **kwargs,
    )
    return opt


class SPASolver:
    """VQE-like solver for SPA circuits, to be used as ``vqe_solver`` in orbital optimization.

    Calling the solver minimizes the energy of a circuit, and ``compute_rdms``
    returns the RDMs of the current state. For plain graphs (lists of
    pairs) the classical SPA solution is used and no circuit is simulated.

    Parameters
    ----------
    decompose : bool, optional
        Factorize expectation values over circuit clusters. Default is False.
    grouping : int or list of int, optional
        Cluster size or explicit qubit clusters.
    backend : str, optional
        Simulation backend. Default is ``"qulacs"``.
    fast_rdm : bool, optional
        Compute RDMs with ``fast_rdm``. Default is True.
    restrict_to_hcb : bool, optional
        Warn if circuit or Hamiltonian use more qubits than orbitals (hard-core boson
        encoding). Default is True.
    edges : list of tuple of int, optional
        Orbital pairs of the SPA graph; enables the classical solver if given as pairs.
    """

    def __init__(
        self,
        decompose=False,
        grouping: int | list[int] | None = None,
        backend: str = "qulacs",
        fast_rdm=True,
        restrict_to_hcb=True,
        edges: list[tuple[int]] | None = None,
    ):
        self.decompose = decompose
        self.rdm_qtensors = None
        self.variables = None
        self.grouping = grouping
        self.backend = backend
        self.restart_from_previous_runs = True
        self.restrict_to_hcb = restrict_to_hcb  # need to change to "use_hcb" for consistency, but also needs to be changted in tequila
        self.fast_rdm = fast_rdm
        self.edges = edges

    def __call__(self, *args, **kwargs):
        return self.vqe(*args, **kwargs)

    def vqe(self, H, circuit, molecule, *args, **kwargs) -> OptimizerResults:
        """Minimize the energy of ``circuit`` for the Hamiltonian ``H``.

        Parameters
        ----------
        H : tq.QubitHamiltonian
            Hamiltonian in the current orbital basis.
        circuit : tq.QCircuit
            Parametrized SPA circuit (ignored if ``edges`` is a plain graph).
        molecule : QuantumChemistryBase
            Molecule providing the integrals.
        *args, **kwargs
            Passed to ``tq.minimize``.
            ``restrict_to_hcb`` (bool) overrides the setting of the solver.

        Returns
        -------
        OptimizerResults
            Energy and optimal variables. Variables are reused as initial values
            in later calls.
        """
        if (
            self.edges is not None
            and isinstance(self.edges, list)
            and isinstance(self.edges[0], (tuple, list))
            and isinstance(self.edges[0][0], int)
            and all(len(edge) < 3 for edge in self.edges)
        ):
            variables, energy = spa_solve_graph(mol=molecule, graph=self.edges)
            return OptimizerResults(energy=energy, history=None, variables=variables)

        if "restrict_to_hcb" in kwargs:
            self.restrict_to_hcb = kwargs.pop("restrict_to_hcb")
        # failsave:
        if self.restrict_to_hcb:
            n = molecule.n_orbitals
            if len(circuit.qubits) > n:
                warnings.warn(
                    f"VQE: restrict_to_hcb is True but more than {n} qubits in the circuit: {circuit.qubits}",
                    TequilaWarning,
                )
            if len(H.qubits) > n:
                warnings.warn(
                    f"VQE: restrict_to_hcb is True but more than {n} qubits in the Hamiltonian: {circuit.qubits}",
                    TequilaWarning,
                )

        if len(circuit.qubits) != len(H.qubits):
            warnings.warn(
                f"VQE: Hamiltonian with {H.qubits} qubits and circuit with {circuit.qubits} qubits?!? restrict_to_hcb={self.restrict_to_hcb}",
                TequilaWarning,
            )

        if self.decompose:
            E = decompose(H=H, U=circuit, grouping=self.grouping)
            result = tq.minimize(
                E,
                *args,
                silent=True,
                initial_values=self.variables,
                gradient="2-point",
                method_options={"finite_diff_rel_step": 1.0e-4},
                backend=self.backend,
                **kwargs,
            )
        else:
            E = tq.ExpectationValue(H=H, U=circuit)
            result = tq.minimize(
                E,
                *args,
                silent=True,
                initial_values=self.variables,
                backend=self.backend,
                **kwargs,
            )
        if self.restart_from_previous_runs:
            self.variables = result.variables
        # analyse()
        return result

    def compute_rdms(self, U, variables, molecule, use_hcb=False):
        """Compute the spin-free 1- and 2-RDM of the state ``U(variables)``.

        Parameters
        ----------
        U : tq.QCircuit
            Circuit preparing the state.
        variables : dict
            Values of the circuit variables.
        molecule : QuantumChemistryBase
            Molecule defining the RDM operators.
        use_hcb : bool, optional
            Use hard-core boson RDM operators. Default is False.

        Returns
        -------
        tuple of numpy.ndarray
            ``(rdm1, rdm2)``, the 2-RDM in the physics (Dirac) ordering.
        """
        # start = time.time()
        if self.fast_rdm:
            if (
                self.edges is not None
                and isinstance(self.edges, list)
                and isinstance(self.edges[0], (tuple, list))
                and isinstance(self.edges[0][0], int)
                and all(len(edge) < 3 for edge in self.edges)
            ):
                return spa_rdms(
                    edges=self.edges,
                    variables=variables,
                    n_orbitals=molecule.n_orbitals,
                )
            # needs hacked tq version where molecule.compute_rdms gives back the raw list of qops
            rdm1, rdm2 = fast_rdm(
                U=U,
                mol=molecule,
                clusters=self.grouping,
                variables=variables,
                backend=self.backend,
            )
        else:
            if self.rdm_qtensors is None or str(U) != self.rdm_qtensors[-1]:
                rdm1, rdm2 = molecule.compute_rdms(U=U, evaluate=False, use_hcb=use_hcb)
                rdm1, shape1 = fast_qtensor(
                    rdm1,
                    variables=variables,
                    do_decompose=self.decompose,
                    evaluate=False,
                    grouping=self.grouping,
                    backend=self.backend,
                )
                rdm2, shape2 = fast_qtensor(
                    rdm2,
                    variables=variables,
                    do_decompose=self.decompose,
                    evaluate=False,
                    grouping=self.grouping,
                    backend=self.backend,
                )
                self.rdm_qtensors = ((rdm1, shape1), (rdm2, shape2), str(U))

            else:
                rdm1, rdm2, _checksum = self.rdm_qtensors
                rdm1, shape1 = rdm1
                rdm2, shape2 = rdm2

            rdm1 = [x(variables) for x in rdm1]
            rdm1 = numpy.asarray(rdm1).reshape(shape1)
            rdm2 = [x(variables) for x in rdm2]
            rdm2 = numpy.asarray(rdm2).reshape(shape2)

        # print("RDMs took {}s".format(time.time()-start))

        return rdm1, rdm2


def make_initial_orbital_guess(edges):
    """Return an orbital rotation matrix mixing the two orbitals of each edge (single-orbital entries stay fixed).

    Parameters
    ----------
    edges : list of tuple of int
        SPA edges; length-2 edges mix their two orbitals.
    """

    result = numpy.eye(2 * len(edges))
    for edge in edges:
        if len(edge) - 1:
            result[edge[0]][edge[1]] = 1.0
            result[edge[1]][edge[0]] = -1.0
    return result.T
