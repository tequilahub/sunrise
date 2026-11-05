from __future__ import annotations

import os
import re
import shutil
import typing
import warnings
import sys

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
from ...utils.giuseppe import giuseppe_bar
import frayedends as fe

_Cached_WORLD = fe.MadWorld(ndims=3, thresh=1.0e-6) # TODO Here just while I work
_world_options = ["l", "wavelet_order", "thresh", "initial_level", "truncate_mode", "refine", "n_threads"]


_Live_Molecules_ID = []
# Examples how to initialize the madness backend
# sun.Molecule(geometry="...") will initialize a molecule with n_qubits=n_electrons (n_electrons//2 HF orbitals + n_electrons//2 PNOs)
# sun.Molecule(geometry="...", n_pno="read") will read in files (name is auto-generated from geometry)
# sun.Molecule(geometry="...", name="X", n_pno="read") will read in files X_htensor.npy, X_gtensor.npy, X_pnoinfo.txt
# sun.Molecule(geometry="...", name="X", datadir="asd/Y/", n_pno="read") reads in files from directory asd/Y/
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

    def cube_plot(self, filename: str | None = None, orbital: list[int] | None = None, use_active: bool = True,*args, **kwargs):
        """
        plot orbitals to cube file
        Parameters
        ----------
        filename: name of the cubefile (default: mra_orbital_X.cube where X is the total index of the active orbital)
        use_active: Wether to plot only the active orbitals, if orbital list passed and true, its assumed that the indices are w.r.t. active orbitals (mol.integral_manager.active_orbitals[x].idx instead of ...[x].idx_total).
        orbital: index of the orbitals to save
        args: further arguments for cube_plot
        kwargs further keyword arguments for cube_plot (or MadWorld if needed)
        """
        if filename is None:
                filename = (
                    self.parameters.name
                    + "-"
                    + self.integral_manager._basis_name # TODO
                    + "-"
                    + self.integral_manager._orbital_type # TODO
                )
        if _Cached_WORLD is None:
            word_kwargs = {k: kwargs.pop(k) for k in _world_options if k in kwargs}
            self._initialize_world(**word_kwargs)

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

        for i, idx in enumerate(orbital):
            name = str(label[i]) + "_" + filename + "_MO.cube",
            giuseppe_bar(step=i, total_steps=len(orbital))
            _Cached_WORLD.cube_plot(*args,
                                    filename=name, 
                                    mra_function=self.integral_manager.orbital[idx],
                                    molecule=self._frayed_geom(),
                                    **kwargs
                                    )
            giuseppe_bar(step=i + 1, total_steps=len(orbital))
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _initialize_world(self,
                    L: float = 50.0,  # half the box length, units: bohr
                    wavelet_order: int = 7,  # wavelet order
                    thresh: float = 1.0e-5,  # threshold for function representation
                    initial_level: int = 3,  # initial projection level
                    truncate_mode: int = 1,
                    refine: bool = True,
                    n_threads: int = -1,  # default is all available threads
                ):
        if _Cached_WORLD is not None:
            warnings.warn("World Was already created on this proces, skiping for safetiness"), SunriseFrayedEndWarning
            return
        _Cached_WORLD = fe.MadWorld(ndims=3, L=L, k=wavelet_order, thresh=thresh, initial_level=initial_level,
                                    truncate_mode=truncate_mode, refine=refine, n_threads=n_threads)

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
        # Extra info that the molecule need
        ...

    def use_CLPO_orbitals_and_edges(self, *args, **kwargs):
        '''
        Currently no janpa - frayedends interface is implemented. However it can be interfaced via pyscf. See `examples/VB_embedding/embeddings.py`
        '''
        raise NotImplementedError

    def use_CLPO_orbitals(self, *args, **kwargs):
        '''
        Currently no janpa - frayedends interface is implemented. However it can be interfaced via pyscf. See `examples/VB_embedding/embeddings.py`
        '''
        raise NotImplementedError

    def use_HAO_orbitals(self, *args, **kwargs):
        '''
        Currently no janpa - frayedends interface is implemented. However it can be interfaced via pyscf. See `examples/VB_embedding/embeddings.py`
        '''
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
        name = "SPA-UpCCD" # NOTE: This is due to PNOs, will change
        if hcb and "HCB" not in name.upper():
            name = "HCB-" + name
        return self.make_upccgsd_ansatz(name=name, label=label)

    def perturbative_f12_correction(
        self,
        rdm1: numpy.ndarray | None = None,
        rdm2: numpy.ndarray | None = None,
        n_ri: int | None = None,
        f12_filename: str = "molecule_f12tensor.bin",
        **kwargs,
    ) -> float:
        """
        Computes the spin-free [2]_R12 correction, needing only the 1- and 2-RDM of a reference method
        Requires either 1-RDM, 2-RDM or information to compute them in kwargs

        Parameters
        ----------
        rdm1 :
            1-electron reduced density matrix
        rdm2 :
            2-electron reduced density matrix
        gamma :
            f12-exponent, for a correlation factor f_12 = -1/gamma * exp[-gamma*r_12]
        n_ri :
            dimensionality of RI-basis; if None, then the maximum available via tensors / basis-set is used
        f12_filename :
            when using madness_interface, <q|h|p> and <rs|1/r_12|pq> already available;
            need to provide f12-tensor <rs|f_12|pq> as ".bin" from madness or ".npy", assuming Mulliken ordering
        kwargs :
            e.g. RDM-information via {"U": QCircuit, "variables": optimal angles}, needs to be passed if rdm1,rdm2 not
            yet computed

        Returns
        -------
            the f12 correction for the energy
        """
        from .f12_corrections._f12_correction_frayed_ends import (
            ExplicitCorrelationCorrectionFrayedends,
        )

        correction = ExplicitCorrelationCorrectionFrayedends(
            mol=self,
            rdm1=rdm1,
            rdm2=rdm2,
            n_ri=n_ri,
            f12_filename=f12_filename,
            **kwargs,
        )

        return correction.compute()

    def __str__(self):
        info = super().__str__()
        # TODO        

        return info

    def __repr__(self):
        return self.__str__()

    def _init_frayed(
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
        
        ## Frayed molecule nature independent init
        ... 

    def _frayed_geom(self):
        return fe.MolecularGeometry(geometry=self.parameters.geometry, units=self.parameters.units, silent=True)

    def compute_energy(): ...
        ## Frayed has its own, other should be called super().X

    def refine_orbitals(): ...
        ## Keep similar to sn.optimize_orbitals
        #   

    def __del__(self): ...

    def __deepcopy__(self): ...

    def __copy__(self): ...

    def __init__(self): ...

    def compile_to_frayed(self): ...
        ## return frayed ends compiled stuff for more experimented users ??

    @staticmethod
    def from_sunrise(other:QuantumChemistryBase): ...
    # reads another sunrise molecule, reorders the orb coeff if needed, and proyects its via AtomicBasisProjector



class QuantumChemistryFrayed(FrayedEndsMixin, QuantumChemistryBase):
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
            from tequila import minimize

            return minimize(*args, objective=E, **kwargs).energy
        else:
            return super().compute_energy(*args, method=method, **kwargs)

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
