import typing
from sunrise.molecules.qubit_base import ParametersQC
from sunrise.molecules.qubit_base.madness_interface import QuantumChemistryMadness as QubitQuantumChemistryMadness
from .HybridBase import HybridBase


class QuantumChemistryMadness(HybridBase):
    """
    Hybrid molecule with MRA-PNO orbitals computed by madness
    The madness calculation (or the read-in of name_htensor.npy, name_gtensor.npy and name_pnoinfo.txt with n_pno="read")
    is done by the qubit madness interface, see sunrise.molecules.qubit_base.madness_interface for all options
    e.g. sn.Molecule(geometry="...", n_pno=4, select="BFFB", nature="hybrid")
    """

    plot2cube = QubitQuantumChemistryMadness.plot2cube
    get_pair_orbitals = QubitQuantumChemistryMadness.get_pair_orbitals
    get_virtual_orbitals = QubitQuantumChemistryMadness.get_virtual_orbitals

    def __init__(self, parameters: ParametersQC, select: typing.Union[str, dict] = {},
                 transformation: typing.Union[str, typing.Callable] = None, *args, **kwargs):
        # hybrid options are not passed to the qubit molecule
        hybrid_kwargs = {k: kwargs.pop(k) for k in ["condense", "two_qubit", "integral_tresh"] if k in kwargs}
        molecule = QubitQuantumChemistryMadness(parameters=parameters, *args, **kwargs)
        self.executable = molecule.executable
        self.datadir = molecule.datadir
        super().__init__(
            parameters=parameters,
            select=select,
            transformation=transformation,
            nuclear_repulsion=molecule.integral_manager.constant_term,
            one_body_integrals=molecule.integral_manager.one_body_integrals,
            two_body_integrals=molecule.integral_manager.two_body_integrals,
            overlap_integrals=molecule.integral_manager.overlap_integrals,
            orbital_coefficients=molecule.integral_manager.orbital_coefficients,
            orbitals=molecule.integral_manager.orbitals,  # keeps the pair information of the PNOs
            active_orbitals=[o.idx_total for o in molecule.integral_manager.active_orbitals],
            reference_orbitals=molecule.integral_manager.active_space.reference_orbitals,
            orbital_type=molecule.integral_manager._orbital_type,
            **hybrid_kwargs,
        )

    def get_spa_edges(self, *args, **kwargs):
        """
        SPA edges from the PNO structure: each occupied orbital together with its PNOs
        """
        return [
            tuple([i.idx] + [a.idx for a in self.get_pair_orbitals(i=i, j=i, exclude=i)])
            for i in self.reference_orbitals
        ]
