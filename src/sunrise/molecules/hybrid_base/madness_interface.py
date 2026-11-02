"""Hybrid molecule with MRA-PNO orbitals and integrals from MADNESS."""

from __future__ import annotations

import typing
import warnings

from tequila import ExpectationValue, QCircuit, TequilaException, TequilaWarning, gates

from sunrise.molecules.qubit_base import ParametersQC
from sunrise.molecules.qubit_base.madness_interface import (
    MadnessPNOMixin,
    TequilaMadnessException,
)

from .hybridbase import HybridBase


class QuantumChemistryMadness(MadnessPNOMixin, HybridBase):
    """Hybrid molecule with MRA-PNO orbitals and integrals from MADNESS.

    The MADNESS calculation (or the read-in of ``name_htensor.npy``, ``name_gtensor.npy`` and
    ``name_pnoinfo.txt`` with ``n_pno="read"``) is shared with the other natures, see the qubit
    ``QuantumChemistryMadness`` for the backend options.

    Examples
    --------
    >>> mol = sun.Molecule(geometry="...", n_pno=4, select="BFFB", nature="hybrid")  # doctest: +SKIP
    """

    def __init__(
        self,
        parameters: ParametersQC,
        select: str | dict | None = None,
        transformation: str | typing.Callable | None = None,
        *args,
        **kwargs,
    ):
        """See the class docstring."""
        self._init_madness(
            super().__init__,
            parameters,
            *args,
            select=select,
            transformation=transformation,
            **kwargs,
        )

    def get_spa_edges(self, *args, **kwargs):
        """Return the SPA edges from the PNO structure: each occupied orbital together with its
        PNOs, as list of tuples of orbital indices.
        """
        return [
            tuple(
                [i.idx] + [a.idx for a in self.get_pair_orbitals(i=i, j=i, exclude=i)]
            )
            for i in self.reference_orbitals
        ]

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
        >>> qubit_map = mol.local_qubit_map()  # doctest: +SKIP
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
                    down = [i for i in down if i is not None]
                    ordered_qubits += up + down
                else:
                    for i in pnos:
                        ordered_qubits.append(self.transformation.up(i))
                        dwn = self.transformation.down(i)
                        if dwn is not None:
                            ordered_qubits.append(dwn)

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
        QCircuit
        """
        if pairs is None:
            pairs = [x for x in self.reference_orbitals]
        U = QCircuit()
        if direct_compiling:
            if not include_reference:
                raise TequilaException(
                    "HCB_PNO_UPCCD: Direct compiling needs reference included"
                )
            for x in pairs:
                U += gates.X(x.idx)
                c = [None, x.idx]
                for a in self.get_pair_orbitals(i=x, j=x):
                    if a == x:
                        continue
                    idx = self.format_excitation_indices([(x.idx, a.idx)])
                    U += gates.Ry(angle=(idx, "D", label), target=a.idx, control=c[0])
                    U += gates.X(target=c[1], control=a.idx)
                    if ladder:
                        c = [a.idx, a.idx]
                    else:
                        c = [x.idx, x.idx]

            alpha_map = {k.idx: self.transformation.up(k.idx) for k in self.orbitals}
            U = U.map_qubits(alpha_map)
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

    def _verify_pair_excitation(self, idx) -> bool:
        """Return whether both electrons of the pair excitation ``idx`` (in the format of ``format_excitation_indices``, e.g. ``((p, q),)``) are allowed by ``verify_excitation``."""
        spin_orbitals = []
        for p, q in idx:
            spin_orbitals += [(2 * p, 2 * q), (2 * p + 1, 2 * q + 1)]
        return self.verify_excitation(spin_orbitals, False)

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
                if (
                    idx not in exclude
                    and idx not in indices
                    and self._verify_pair_excitation(idx)
                ):
                    indices.append(idx)

        if "G" in name:
            for i in self.reference_orbitals:
                for a in self.get_pair_orbitals(i=i, j=i, exclude=i):
                    for b in self.get_pair_orbitals(i=i, j=i, exclude=i):
                        if a.idx <= b.idx:
                            continue

                        idx = self.format_excitation_indices([(a.idx, b.idx)])
                        if (
                            idx not in exclude
                            and idx not in indices
                            and self._verify_pair_excitation(idx)
                        ):
                            indices.append(idx)

        if "PNO" in name or "SPA" in name:
            return indices

        virtuals = [i for i in self.orbitals if len(i.pair) == 2 and i.occ != 2.0]
        virtuals += self.get_virtual_orbitals()  # this is usually empty
        # HF-X -- PNO-XY/PNO-YY indices
        for i in self.reference_orbitals:
            for a in virtuals:
                idx = self.format_excitation_indices([(i.idx, a.idx)])
                if (
                    idx not in exclude
                    and idx not in indices
                    and self._verify_pair_excitation(idx)
                ):
                    indices.append(idx)
        # HF-HF and PNO-PNO indices
        if "G" in name:
            for i in self.reference_orbitals:
                for j in self.reference_orbitals:
                    if i.idx <= j.idx:
                        continue
                    idx = self.format_excitation_indices([(i.idx, j.idx)])
                    if (
                        idx not in exclude
                        and idx not in indices
                        and self._verify_pair_excitation(idx)
                    ):
                        indices.append(idx)
            for a in virtuals:
                for b in virtuals:
                    if a.idx_total <= b.idx_total:
                        continue
                    idx = self.format_excitation_indices([(a.idx, b.idx)])
                    if (
                        idx not in exclude
                        and idx not in indices
                        and self._verify_pair_excitation(idx)
                    ):
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
        >>> U = mol.make_pno_upccgsd_ansatz(name="UpCCD")  # doctest: +SKIP
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
