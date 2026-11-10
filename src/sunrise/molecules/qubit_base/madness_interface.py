"""MADNESS backend: basis-set-free MRA-PNO orbitals and their integrals (``MadnessPNOMixin``) and
the qubit MADNESS molecule.
"""

from __future__ import annotations

import os
import shutil
import typing
import warnings

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

# Examples how to initialize the madness backend
# sun.Molecule(geometry="...") will initialize a molecule with n_qubits=n_electrons (n_electrons//2 HF orbitals + n_electrons//2 PNOs)
# sun.Molecule(geometry="...", n_pno="read") will read in files (name is auto-generated from geometry)
# sun.Molecule(geometry="...", name="X", n_pno="read") will read in files X_htensor.npy, X_gtensor.npy, X_pnoinfo.txt
# sun.Molecule(geometry="...", name="X", datadir="asd/Y/", n_pno="read") reads in files from directory asd/Y/
# control madness input sections with dictionaries
# sun.Molecule(geometry="...", pno={"maxrank":10, "freeze":0}, dft={"k":9, "L":25.0})
# compute more orbitals
# sun.Molecule(geometry="...", n_pno=10) # computes 10 PNOs additional to the occupied HF orbitals


class TequilaMadnessException(TequilaException):
    """Error raised by the MADNESS backend."""

    def __str__(self):
        """Return the message prefixed with the backend name."""
        return "Error in madness backend:" + self.message


class MadnessPNOMixin:
    """MRA-PNO orbitals and integrals from MADNESS, shared by the qubit, fermionic and hybrid molecules.

    Only needs ``self.orbitals`` and ``self.integral_manager`` (with the pair information of the
    PNOs). Put it first in the bases so that it overrides the methods of the backend base class.

    References
    ----------
    J. S. Kottmann, P. Schleich, T. Tamayo-Mendoza and A. Aspuru-Guzik, J. Phys. Chem. Lett. 12, 663 (2021), doi:10.1021/acs.jpclett.0c03410.
    """

    def plot2cube(self, orbital, filename=None, *args, **kwargs):
        """Plot an orbital to a cube file (needs the madtequila backend installed).

        Parameters
        ----------
        orbital : int or OrbitalData
            Index of the active orbital (starting from 0). Frozen orbitals can be given as
            orbital structure with ``idx_total`` defined.
        filename : str, optional
            Name of the cube file. Default is ``mra_orbital_X.cube`` with X the total index of the orbital.
        *args, **kwargs
            Further arguments of the ``plot2cube`` executable, see
            https://github.com/kottmanj/madness/tree/tequila/src/apps/plot
        """

        plot2cube = shutil.which("plot2cube")
        if plot2cube is None:
            raise TequilaMadnessException(
                "can't plot to cube file. Couldn't find plot2cube executable.\n\nTry installing\n\t conda install madtequila -c kottmann\nand assure the version is >2.3"
            )

        if hasattr(orbital, "idx"):
            idx = orbital.idx
        else:
            idx = self.orbitals[orbital].idx_total

        callist = [plot2cube, f"file=mra_orbital_{idx}"]

        if filename is not None:
            callist.append(f"outfile={filename}")
        for k, v in kwargs.items():
            callist.append(f"{k}={v}")
        for k in args:
            callist.append(f"{k}")

        import subprocess

        try:
            with open(f"plot2cube_{orbital}.log", "w") as logfile:
                subprocess.call(callist, stdout=logfile)
        except Exception:  # noqa: BLE001
            print("plotting failed ....")
            print(f"see plot2cube_{orbital}.log")

    def get_pair_orbitals(
        self, i: OrbitalData, j: OrbitalData, exclude: list[OrbitalData] | None = None
    ):
        """Return the orbitals that belong to the pair ``(i, j)``.

        Parameters
        ----------
        i, j : int or OrbitalData
            The orbitals of the pair.
        exclude : list of OrbitalData, optional
            Orbitals to leave out.

        Returns
        -------
        list of OrbitalData
        """
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
        """Return the virtual orbitals that belong to no pair (usually empty)."""
        return [x for x in self.orbitals if len(x.pair) == 1 and x.pair[0] < 0]

    # mol.save stores the PNO pair structure, sun.load_molecule gives it back to the qubit madness interface
    def _backend_data(self):
        """Return the pair and occupation information of the orbitals (None, with a warning, if the PNO structure was lost)."""
        # the pnoinfo of the madness calculation (the integrals and the nuclear repulsion are stored by mol.save)
        orbitals = self.integral_manager.orbitals
        if any(o.pair is None or o.occ is None for o in orbitals):
            warnings.warn(
                "the orbitals lost their PNO pair structure (e.g. they were transformed), "
                "the molecule is loaded with the base backend",
                TequilaWarning,
            )
            return None
        return {
            "pairinfo": [o.pair for o in orbitals],
            "occinfo": [o.occ for o in orbitals],
        }

    def use_CLPO_orbitals_and_edges(self, *args, **kwargs):
        """Not supported with MADNESS orbitals."""
        raise NotImplementedError

    def use_CLPO_orbitals(self, *args, **kwargs):
        """Not supported with MADNESS orbitals."""
        raise NotImplementedError

    def use_HAO_orbitals(self, *args, **kwargs):
        """Not supported with MADNESS orbitals."""
        raise NotImplementedError

    def make_spa_ansatz(self, label=None, hcb=False, edges=None, **kwargs):
        """Return the default SPA ansatz, equivalent to PNO-UpCCD with the MADNESS PNOs.

        If ``edges`` or further keyword arguments are given, the SPA of the backend base class is used instead.

        Parameters
        ----------
        label : hashable, optional
            Label of the angles.
        hcb : bool, optional
            If True the circuit is not mapped from the hard-core boson encoding to the encoding of the molecule.
        edges : list, optional
            SPA edges, e.g. ``mol.get_spa_edges()``.
        **kwargs
            Passed to the ``make_spa_ansatz`` of the backend base class.

        Returns
        -------
        QCircuit or FCircuit

        Examples
        --------
        >>> U = mol.make_spa_ansatz(edges=mol.get_spa_edges())
        """
        if edges is not None or kwargs:
            # the fermionic SPA has no hcb option
            if hcb:
                kwargs["hcb"] = hcb
            return super().make_spa_ansatz(edges=edges, label=label, **kwargs)
        name = "SPA-UpCCD"
        if hcb and "HCB" not in name.upper():
            name = "HCB-" + name
        return self.make_upccgsd_ansatz(name=name, label=label)

    @staticmethod
    def find_executable(madness_root_dir=None):
        """Return the path of the ``pno_integrals`` executable, or None if not found.

        Parameters
        ----------
        madness_root_dir : str, optional
            Root directory of MADNESS. Default is the environment variable ``MAD_ROOT_DIR``.
        """
        executable = shutil.which("pno_integrals")
        if madness_root_dir is None:
            madness_root_dir = str(os.environ.get("MAD_ROOT_DIR"))
        if executable is None and madness_root_dir is not None:
            executable = shutil.which(f"{madness_root_dir}/src/apps/pno/pno_integrals")
        return executable

    def read_tensors(self, name="molecule", filetype="npy", datadir=None):
        """Read the tensors ``name_htensor.<filetype>`` and ``name_gtensor.<filetype>``.

        Parameters
        ----------
        name : str, optional
            Prefix of the files. Default is "molecule".
        filetype : str, optional
            File ending. Default is "npy".
        datadir : str, optional
            Directory of the files.

        Returns
        -------
        tuple
            One- and two-body tensors, each the string "failed" if it could not be read.
        """

        path = name
        if datadir is not None:
            path = f"{datadir}/{name}"

        try:
            h = numpy.load(f"{path}_htensor.{filetype}")
        except Exception:  # noqa: BLE001
            h = "failed"

        try:
            g = numpy.load(f"{path}_gtensor.{filetype}")
        except Exception:  # noqa: BLE001
            g = "failed"

        return h, g

    def cleanup(self, warn=False, delete_all_files=False):
        """Delete intermediate MADNESS files.

        Parameters
        ----------
        warn : bool, optional
            Warn about files that cannot be deleted. Default is False.
        delete_all_files : bool, optional
            Also delete the tensor, pnoinfo and output files of this molecule. Default is False.
        """
        filenames = ["pnoinfo.txt", "molecule_htensor.bin", "molecule.gtensor.bin"]
        if delete_all_files:
            filenames = [
                f"{self.parameters.name}_htensor.npy",
                f"{self.parameters.name}_gtensor.npy",
                f"{self.parameters.name}_pnoinfo.txt",
                f"{self.parameters.name}_pno_integrals.out",
            ]
        for filename in filenames:
            if os.path.exists(filename):
                if warn:
                    warnings.warn(
                        f"Found file {filename} from previous calculation ... deleting it",
                        TequilaWarning,
                    )
                os.remove(filename)

    def perturbative_f12_correction(
        self,
        rdm1: numpy.ndarray | None = None,
        rdm2: numpy.ndarray | None = None,
        n_ri: int | None = None,
        f12_filename: str = "molecule_f12tensor.bin",
        **kwargs,
    ) -> float:
        r"""Return the spin-free [2]_R12 correction from the 1- and 2-RDM of a reference method.

        Either the RDMs or the information to compute them (``U`` and ``variables`` as keyword
        arguments, see ``compute_rdms``) must be given.

        Parameters
        ----------
        rdm1, rdm2 : numpy.ndarray, optional
            1- and 2-electron reduced density matrices (spin-free), the 2-RDM in the physics
            (Dirac) ordering as returned by ``compute_rdms``.
        n_ri : int, optional
            Dimension of the RI basis. Default is the maximum available from the tensors.
        f12_filename : str, optional
            File with the f12 tensor :math:`\langle rs | f_{12} | pq \rangle`, ".bin" from MADNESS or
            ".npy" (Mulliken ordering). The other tensors, :math:`\langle q | h | p \rangle` and
            :math:`\langle rs | r_{12}^{-1} | pq \rangle`, are already available from MADNESS.
            Default is "molecule_f12tensor.bin".
        **kwargs
            E.g. ``U`` and ``variables`` if the RDMs are not computed yet, or ``gamma`` (float,
            default 1.4), the exponent of the correlation factor
            :math:`f_{12} = -\frac{1}{\gamma} e^{-\gamma r_{12}}`; it must match the exponent used by
            MADNESS for the f12 tensor.

        Returns
        -------
        float
            The correction to the energy.
        """
        from .f12_corrections._f12_correction_madness import (
            ExplicitCorrelationCorrectionMadness,
        )

        correction = ExplicitCorrelationCorrectionMadness(
            mol=self,
            rdm1=rdm1,
            rdm2=rdm2,
            n_ri=n_ri,
            f12_filename=f12_filename,
            **kwargs,
        )

        return correction.compute()

    def __str__(self):
        """Return a summary of the molecule, the executable and the tensor files."""
        path = self.parameters.name
        if hasattr(self, "datadir") and self.datadir is not None:
            path = f"{self.datadir}/{path}"
        info = super().__str__()
        info += "\n"
        info += "{:15} : {}\n".format("executable", self.executable)
        info += "{:15} : {}\n".format("htensor", f"{path}_htensor.npy")
        info += "{:15} : {}\n".format("gtensor", f"{path}_gtensor.npy")

        return info

    def __repr__(self):
        """Return the same summary as ``__str__``."""
        return self.__str__()

    def _init_madness(
        self,
        build: typing.Callable,
        parameters: ParametersQC,
        *args,
        active_orbitals: list = "auto",
        executable: str | None = None,
        n_pno: int | None = None,
        n_virt: int = 0,
        keep_mad_files=False,
        datadir=None,
        **kwargs,
    ):
        """Get the integrals and the PNO pair structure from MADNESS and initialize the molecule with them.

        The data is computed by the executable, read from files (``n_pno="read"``) or taken from the
        keyword arguments of ``sun.load_molecule``. ``build`` is the ``__init__`` of the backend base class
        (QuantumChemistryBase, HybridBase or FermionicBase), the remaining keyword arguments (e.g.
        ``transformation``, ``select``) are passed to it.
        """
        # can not use "geometry"
        # as this is already used for other things furter up
        if "mad_geometry_options" in kwargs:
            self.mad_geometry_options = kwargs["mad_geometry_options"]

        self.datadir = datadir

        # see if MAD_ROOT_DIR is defined
        self.madness_root_dir = os.environ.get("MAD_ROOT_DIR")
        # see if the pno_integrals executable can be found
        if executable is None:
            executable = self.find_executable()
            if executable is None and self.madness_root_dir is not None:
                warnings.warn(
                    f"MAD_ROOT_DIR={self.madness_root_dir} found\nbut couldn't find executable",
                    TequilaWarning,
                )

        else:
            executable = shutil.which(executable)

        self.executable = executable
        self.n_pno = n_pno
        self.n_virt = n_virt
        self.kwargs = kwargs

        # if no n_pno is given, look for MRA data (default)
        name = parameters.name

        # the data of a madness calculation can be given directly (e.g. by sun.load_molecule):
        # one_body_integrals, two_body_integrals, nuclear_repulsion, pairinfo and occinfo, then no files are read
        given = "one_body_integrals" in kwargs and "pairinfo" in kwargs
        if given:
            h = kwargs.pop("one_body_integrals")
            g = kwargs.pop("two_body_integrals")
            if isinstance(g, NBodyTensor):
                g = g.reorder(to="chem").elems
            n_pno = None
        # try to read in data in the following cases
        # - no executable found
        # - executable found but read in explicitly demanded through n_pno="read"
        elif (n_pno is None and executable is None) or (
            hasattr(n_pno, "lower") and n_pno.lower() == "read"
        ):
            h, g = self.read_tensors(name=name, datadir=datadir)
            n_pno = None
        else:
            h = "failed"
            g = "failed"

        if (isinstance(h, str) and "failed" in h) or (
            isinstance(g, str) and "failed" in g
        ):
            status = "found {}_htensor.npy={}\n".format(name, "failed" not in h)
            status += "found {}_gtensor.npy={}\n".format(name, "failed" not in g)
            try:
                # try to run madness
                self.parameters = parameters
                status += "madness="
                madness_status = self.run_madness(*args, **kwargs)
                if int(madness_status) != 0:
                    warnings.warn(
                        f"MADNESS did not terminate as expected! status = {status}",
                        TequilaWarning,
                    )
                status += str(madness_status) + "\n"
            except Exception as E:  # noqa: BLE001 - any failure is reported in the status message below
                status += str(E) + "\n"

            # will read the binary files, convert them and save them with the right name
            h, g, pinfo = self.convert_madness_output_from_bin_to_npy(
                name=name, datadir=datadir
            )
            status += "found {}_htensor.npy={}\n".format(name, "failed" not in h)
            status += "found {}_gtensor.npy={}\n".format(name, "failed" not in g)
            status += "found {}_pnoinfo.txt={}\n".format(name, "failed" not in pinfo)
            status += "h_tensor report:\n"
            status += str(h)
            status += "g_tensor report:\n"
            status += str(g)
            status += "pnoinfo report:\n"
            status += str(pinfo)

            solution = f"Solution 1: Assuming precomputed files are available:\n    provide {name}_gtensor.npy, {name}_htensor.npy and {name}_pnoinfo.txt\n    and call the Molecule constructor with n_pno='read' keyword \n\nSolution 2: Try installing with conda\n    conda install madtequila -c kottmann\n\nSolution 3: Install from source\n    follow instructions on github.com/kottmanj/madness"
            if self.executable is not None:
                solution = f"madness executable was found, but calculation did not succeed, check {name}_pno_integrals.out for clues"

            if "failed" in h or "failed" in g:
                raise TequilaMadnessException(
                    f"Could not initialize the madness interface\nStatus report is\n{status}\n\n"
                    + solution
                )
        # get additional information from madness file
        nuclear_repulsion = kwargs.pop("nuclear_repulsion", 0.0)
        pairinfo = kwargs.pop("pairinfo", None)
        occinfo = kwargs.pop("occinfo", None)
        path = parameters.name
        if datadir is not None:
            path = f"{datadir}/{path}"
        if not given:
            for name in [
                path + "_pnoinfo.txt",
                parameters.name + "_pnoinfo.txt",
                "pnoinfo.txt",
            ]:
                try:
                    with open(name, "r") as f:
                        for line in f:
                            if "nuclear_repulsion" in line:
                                nuclear_repulsion = float(line.split("=")[1])
                            elif "pairinfo" in line:
                                pairinfo = line.split("=")[1].split(",")
                                pairinfo = [
                                    tuple([int(i) for i in x.split(".")])
                                    for x in pairinfo
                                ]
                            elif "occinfo" in line:
                                occinfo = line.split("=")[1].split(",")
                                occinfo = [float(x) for x in occinfo]

                    if pairinfo is not None:
                        break
                except Exception:  # noqa: BLE001, S112
                    continue

        if pairinfo is None:
            raise TequilaMadnessException(
                "Pairinfo from madness calculation not found\nPlease provide pnoinfo.txt"
            )

        n_orbitals_total = h.shape[0]
        if "n_orbitals" in kwargs:
            # this would be the active orbitals
            kwargs.pop("n_orbitals")

        assert h.shape[1] == n_orbitals_total
        assert sum(g.shape) == 4 * n_orbitals_total
        assert len(g.shape) == 4
        assert len(h.shape) == 2

        g = NBodyTensor(elems=g, ordering="mulliken")

        orbitals = []
        if pairinfo is not None:
            orbitals = [
                OrbitalData(idx_total=i, idx=i, pair=p, occ=occinfo[i])
                for i, p in enumerate(pairinfo)
            ]
            reference_orbitals = [x for x in orbitals if x.occ == 2.0]
            if active_orbitals == "auto":
                not_active = [
                    i
                    for i in reference_orbitals
                    if sum([1 for x in orbitals if i.idx_total in x.pair]) < 2
                ]
                active_orbitals = [x.idx_total for x in orbitals if x not in not_active]

            if active_orbitals is not None:
                i = 0
                for x in orbitals:
                    if x.idx_total in active_orbitals:
                        orbitals[x.idx_total].idx = i
                        i += 1
                    else:
                        orbitals[x.idx_total].idx = None
        else:
            raise TequilaMadnessException(
                "No pairinfo given: madness interface needs a file moleculename_pnoinfo.txt"
            )

        # convert to indices only
        # active space data will be set in baseclass constructor
        reference_orbitals = [x.idx_total for x in reference_orbitals]
        if kwargs.get("reference_orbitals") is not None:
            # given (e.g. by sun.load_molecule) instead of the doubly occupied orbitals
            reference_orbitals = kwargs.pop("reference_orbitals")
        kwargs.pop("reference_orbitals", None)
        build(
            *args,
            parameters=parameters,
            active_orbitals=active_orbitals,
            one_body_integrals=h,
            two_body_integrals=g,
            nuclear_repulsion=nuclear_repulsion,
            n_orbitals=n_orbitals_total,
            orbitals=orbitals,
            reference_orbitals=reference_orbitals,
            **kwargs,
        )

        # print warning if read data does not match expectations
        if n_pno is not None:
            nrefs = len(self.reference_orbitals)
            if n_pno + nrefs + n_virt != self.n_orbitals:
                warnings.warn(
                    f"read in data has {self.n_orbitals - nrefs} pnos/virtuals, but n_pno and n_virt where set to {n_pno} and {n_virt}",
                    TequilaWarning,
                )

        # delete *.bin files and pnoinfo.txt form madness calculation (none if the data was given)
        if not keep_mad_files and not given:
            self.cleanup(warn=False, delete_all_files=False)

    def run_madness(self, *args, **kwargs):
        """Run the MADNESS executable and return its status (or a message if there is no executable)."""
        if self.executable is None:
            return "\n\n----> pno_integrals executable not found <----\n\n"
        self.write_madness_input(*args, n_pno=self.n_pno, n_virt=self.n_virt, **kwargs)

        # prevent reading in old files
        self.cleanup(warn=True, delete_all_files=True)

        import subprocess
        import time

        start = time.time()
        filename = f"{self.parameters.name}_pno_integrals.out"
        print("Starting madness calculation with executable: ", self.executable)
        print(f"output redirected to {filename} logfile")
        with open(filename, "w") as logfile:
            madout = subprocess.call([self.executable], stdout=logfile)
        print(f"finished after {time.time() - start}s")

        os.rename("pnoinfo.txt", f"{self.parameters.name}_pnoinfo.txt")

        return madout

    def write_madness_input(
        self,
        n_pno=None,
        n_virt=0,
        filename="input",
        maxrank=None,
        n_orbitals=None,
        *args,
        **kwargs,
    ):
        """Write the MADNESS input file.

        Parameters
        ----------
        n_pno : int, optional
            Number of PNOs per active pair. Cannot be given together with ``n_orbitals``.
        n_virt : int, optional
            Number of additional virtual orbitals. Default is 0.
        filename : str, optional
            Name of the input file. Default is "input".
        maxrank : int, optional
            Maximum rank of the PNOs.
        n_orbitals : int, optional
            Total number of active orbitals. Default is the number of active electrons.
        **kwargs
            Dictionaries to change the sections of the input, e.g. ``pno={"maxrank": 10}``.
        """
        if n_pno is not None and n_orbitals is not None:
            raise TequilaMadnessException(
                f"n_pno={n_pno} and n_orbitals={n_orbitals} given ... please pick one"
            )

        n_electrons = self.parameters.total_n_electrons
        if self.parameters.frozen_core:
            # only count active electrons (will not compute pnos for frozen pairs)
            n_core_electrons = self.parameters.get_number_of_core_electrons()
            n_electrons -= n_core_electrons

        n_pairs = n_electrons // 2
        if n_orbitals is None:
            n_orbitals = n_electrons  # minimal correlated (each active pair will have one virtual)

        if n_pno is None:
            n_pno = n_orbitals - n_pairs

        if maxrank is None:
            # need at least maxrank=1, otherwise no PNOs are computed
            # this was a bug in <=v1.8.5
            maxrank = max(1, int(numpy.ceil(n_pno / n_pairs)))

        if maxrank <= 0:
            warnings.warn(
                f"maxrank={maxrank} in tequila madness backend! No PNOs will be computed. Set the value when initializing the Molecule as sun.Molecule(..., pno={'maxrank':1, ...})",
                TequilaWarning,
            )

        data = {}
        if self.parameters.multiplicity != 1:
            raise TequilaMadnessException(
                f"Currently only closed shell supported for MRA-PNO-MP2, you demanded multiplicity={self.parameters.multiplicity} for the surrogate"
            )
        data["dft"] = {
            "charge": self.parameters.charge,
            "xc": "hf",
            "k": 7,
            "econv": 1.0e-4,
            "dconv": 5.0e-4,
            "localize": "boys",
            "ncf": "( none , 1.0 )",
        }
        data["pno"] = {
            "maxrank": maxrank,
            "f12": "false",
            "thresh": 1.0e-4,
            "diagonal": True,
        }
        if not self.parameters.frozen_core:
            data["pno"]["freeze"] = 0
        data["pnoint"] = {"n_pno": n_pno, "n_virt": n_virt, "orthog": "symmetric"}
        data["plot"] = {}
        data["f12"] = {}
        for key, value in data.items():
            if key in kwargs:
                data[key] = {**value, **kwargs[key]}

        geom = {}
        if hasattr(self, "mad_geometry_options"):
            geom = self.mad_geometry_options

        if "units" not in geom:
            geom["units"] = "angstrom"
        if "eprec" not in geom:
            geom["eprec"] = 1.0e-6
        if "no_orient" not in geom:
            geom["no_orient"] = 1
        if filename is not None:
            with open(filename, "w") as f:
                for k1, v1 in data.items():
                    print(k1, file=f)
                    for k2, v2 in v1.items():
                        print(f"{k2} {v2}", file=f)
                    print("end\n", file=f)

                print("geometry", file=f)
                for k, v in geom.items():
                    print(f"{k} {v}", file=f)
                for line in self.parameters.get_geometry_string().split("\n"):
                    line = line.strip()
                    if line != "":
                        print(line, file=f)
                print("end", file=f)

        return data

    def convert_madness_output_from_bin_to_npy(self, name, datadir=None):
        """Convert the binary MADNESS tensors to ``.npy`` files and read them.

        Parameters
        ----------
        name : str
            Prefix of the files.
        datadir : str, optional
            Directory of the files, created if it does not exist.

        Returns
        -------
        tuple
            One- and two-body tensors and the pno info, each the string "failed" if it could not be read.
        """
        path = name
        if datadir is not None:
            # if the datadir does not exist then tequila will crash
            try:
                import os

                if not os.path.exists(datadir):
                    os.makedirs(datadir)
            except Exception as E:  # noqa: BLE001  # reported in status
                warnings.warn(
                    f"tried to create datadir={datadir} and caught\n{E!s}",
                    TequilaWarning,
                )

            path = f"{datadir}/{name}"
        try:
            g_data = numpy.fromfile("molecule_gtensor.bin".format())
            sd = int(numpy.power(g_data.size, 0.25))
            assert sd**4 == g_data.size
            sds = [sd] * 4
            g = g_data.reshape(sds)
            numpy.save(f"{path}_gtensor.npy", arr=g)
        except Exception as E:  # noqa: BLE001  # reported in status
            g = f"failed\n{E!s}\n"

        try:
            h_data = numpy.fromfile("molecule_htensor.bin")
            sd = int(numpy.sqrt(h_data.size))
            assert sd**2 == h_data.size
            sds = [sd] * 2
            h = h_data.reshape(sds)
            numpy.save(f"{path}_htensor.npy", arr=h)
        except Exception as E:  # noqa: BLE001  # reported in status
            h = f"failed\n{E!s}\n"

        try:
            with open(f"{name}_pnoinfo.txt", "r") as f1:
                pnoinfo = f1.read().strip()
        except Exception as E:  # noqa: BLE001
            pnoinfo = f"failed\n{E!s}\n"

        if datadir is not None:
            try:
                with (
                    open(f"{name}_pnoinfo.txt", "r") as f1,
                    open(f"{path}_pnoinfo.txt", "w") as f2,
                ):
                    f2.write(f1.read().strip())
            except Exception as E:  # noqa: BLE001
                pnoinfo = f"failed\n{E!s}\n"
            try:
                with (
                    open(f"{name}_pno_integrals.out", "r") as f1,
                    open(f"{path}_pno_integrals.out", "w") as f2,
                ):
                    f2.write(f1.read().strip())
            except Exception:  # noqa: BLE001, S110
                pass

        return h, g, pnoinfo


class QuantumChemistryMadness(MadnessPNOMixin, QuantumChemistryBase):
    """Qubit molecule with MRA-PNO orbitals and integrals from MADNESS.

    Initialization is explained in the examples at the top of the module. The integrals are
    either computed with the ``pno_integrals`` executable, read from the files
    ``name_htensor.npy``, ``name_gtensor.npy`` and ``name_pnoinfo.txt`` (``n_pno="read"``), or
    passed directly as ``one_body_integrals``, ``two_body_integrals``, ``pairinfo`` and ``occinfo``.

    Parameters
    ----------
    parameters : ParametersQC
        Molecular parameters.
    transformation : str or callable, optional
        Fermion-to-qubit encoding.
    active_orbitals : list of int or str, optional
        Active orbitals. Default is "auto", which removes reference orbitals without PNOs.
    executable : str, optional
        Name of the ``pno_integrals`` executable. Default is searched in the PATH and ``MAD_ROOT_DIR``.
    n_pno : int or str, optional
        Number of PNOs in addition to the occupied HF orbitals, or "read" to read the files.
        Default is to read the files if there is no executable.
    n_virt : int, optional
        Number of additional virtual orbitals. Default is 0.
    keep_mad_files : bool, optional
        Keep the intermediate MADNESS files. Default is False.
    datadir : str, optional
        Directory of the MADNESS files.
    **kwargs
        Dictionaries to control the MADNESS input sections, e.g. ``pno={"maxrank": 10}`` or
        ``dft={"k": 9}``, and further arguments of the base class.

    References
    ----------
    J. S. Kottmann, P. Schleich, T. Tamayo-Mendoza and A. Aspuru-Guzik, J. Phys. Chem. Lett. 12, 663 (2021), doi:10.1021/acs.jpclett.0c03410.
    """

    def __init__(
        self,
        parameters: ParametersQC,
        transformation: str | typing.Callable | None = None,
        active_orbitals: list = "auto",
        executable: str | None = None,
        n_pno: int | None = None,
        n_virt: int = 0,
        keep_mad_files=False,
        datadir=None,
        *args,
        **kwargs,
    ):
        """See the class docstring."""
        self._init_madness(
            super().__init__,
            parameters,
            *args,
            transformation=transformation,
            active_orbitals=active_orbitals,
            executable=executable,
            n_pno=n_pno,
            n_virt=n_virt,
            keep_mad_files=keep_mad_files,
            datadir=datadir,
            **kwargs,
        )

    def compute_energy(self, method, *args, **kwargs):
        """Compute the energy of a classical or quantum method.

        Classical methods run with PySCF, quantum methods build an ansatz and minimize the energy.

        Parameters
        ----------
        method : str
            Classical: "HF", "MP2", "CCSD", "CCSD(T)", "FCI". Quantum: "SPA-GASD", where "SPA" and
            letters of "GASD" can be dropped, e.g. "GSD" is the same as UpCCGSD and "SPA" alone
            is "SPA-D". See ``make_upccgsd_ansatz``.
        **kwargs
            For quantum methods, keyword arguments of the minimizer.

        Returns
        -------
        float
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
            from tequila import minimize

            return minimize(*args, objective=E, **kwargs).energy
        else:
            return super().compute_energy(*args, method=method, **kwargs)

    def local_qubit_map(self, hcb=False, up_then_down=False):
        """Return a qubit map that orders the orbitals pair by pair.

        It does not make the circuit more local, but shows the locality better in pictures.
        Apply it with ``H = H.map_qubits(qubit_map)`` and ``U = U.map_qubits(qubit_map)``.

        Parameters
        ----------
        hcb : bool, optional
            Map for the hard-core boson representation. Default is False.
        up_then_down : bool, optional
            Order the spin orbitals as all up then all down. Default is False.

        Returns
        -------
        dict
            Maps qubit indices to the new indices.

        Examples
        --------
        >>> qubit_map = mol.local_qubit_map()
        """
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
        """Return a UpCCGSD-type ansatz circuit, with names like "SPA-UpCCD" and "PNO-UpCCD".

        Overrides the base class method. The name has the form ``{order}-{HCB}-{SPA}-{excitations}``.

        Parameters
        ----------
        name : str, optional
            In general ``{HCB}-{SPA}-{excitations}``, e.g. "SPA-UpCCD" or "SPA-D", "SPA-UpCCGD" or
            "SPA-GD", "SPA-UpCCGSD" or "SPA-GSD", "UpCCGSD" or "GSD". "HCB" in the name keeps the
            hard-core boson encoding (no mapping to the qubit encoding of the molecule). "SPA" uses the
            separable pair ansatz: the excitations are restricted to the PNO structure of the surrogate
            model. The excitations are "S" (only singles), "D" (only doubles), "GSD" (generalized
            singles and doubles) or "GASD" (approximate singles, neglecting Z terms in Jordan-Wigner).
            Default is "UpCCGSD".
        label : hashable, optional
            Label of the variables, named ``(indices, X, (label, layer))`` with X = D or S.
        direct_compiling : bool, optional
            Compile the first layer directly (only for encodings with ``hcb_to_me``).
        order : int, optional
            Number of repeated layers. Can also be given as the first part of the name, e.g.
            "2-UpCCGSD" or "2-SPA-GSD".
        neglect_z : bool, optional
            Neglect all Z terms in the generators of the singles. Set automatically if the name
            contains "A" (e.g. "GASD").
        hcb_optimization : bool, optional
            Build the doubles in the hard-core boson space and map back.
        include_reference : bool, optional
            Start from the reference state. Default is True.
        indices : list, optional
            Excitation indices in the format of ``format_excitation_indices``, e.g.
            ``[((0, 3),), ((1, 4),)]`` (see ``make_pno_upccgsd_ansatz``). They replace the indices
            given by the name in every layer.
        **kwargs
            Passed to the base class, e.g. ``spin_adapt_singles`` (default False here).

        Returns
        -------
        tq.QCircuit

        Examples
        --------
        >>> U = mol.make_upccgsd_ansatz(name="HCB-PNO-UpCCD")
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
            raise TequilaMadnessException(
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
            raise TequilaMadnessException(
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
        """Return the hard-core boson PNO-UpCCD circuit, with pair doubles from each reference orbital into its PNOs.

        Parameters
        ----------
        pairs : list of OrbitalData, optional
            Reference orbitals of the pairs. Default is all reference orbitals.
        label : hashable, optional
            Label of the angles.
        include_reference : bool, optional
            Start from the reference state. Default is True.
        direct_compiling : bool, optional
            Compile directly into the encoding of the molecule (needs the reference).

        ladder : bool, optional
            With ``direct_compiling``, chain the CNOTs through the PNOs instead of fanning out from the
            reference orbital. Default is True.

        Returns
        -------
        tq.QCircuit
        """
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
        """Return the excitation indices missing between PNO-UpCCD and UpCC(G)D.

        Parameters
        ----------
        label : hashable, optional
            Not used.
        name : str, optional
            Excitation type. "G" in the name also adds the generalized excitations (otherwise
            UpCCD), "PNO" or "SPA" returns only the PNO-pair excitations. Default is "UpCCGD".
        exclude : list, optional
            Indices to leave out.

        Returns
        -------
        list
            Formatted excitation indices, see ``format_excitation_indices``.
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
        """Return a UpCCGSD-type ansatz restricted to the PNO excitations.

        Parameters
        ----------
        generalized : bool, optional
            Also add excitations between the PNOs of a pair. Default is False.
        include_offdiagonals : bool, optional
            Also add excitations of off-diagonal pairs. Default is False.
        **kwargs
            Passed on to ``make_upccgsd_ansatz``.

        Examples
        --------
        >>> U = mol.make_pno_upccgsd_ansatz(name="UpCCD")
        """
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
