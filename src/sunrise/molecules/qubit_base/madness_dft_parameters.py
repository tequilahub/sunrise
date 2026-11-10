"""
Keywords of the madness HF/DFT calculation, the "dft" block of the frayedends input (madness/chem/CalculationParameters.h)
name: (type, madness default, description)
Given to frayedends as MadPNO(..., dft={"econv": 1e-6, ...}), they end up as --dft="key=value; ..."
Retired keywords (gopt, restartao, no_compute, nv_factor, ...) and loadbalparts (no description) are left out
"""

# madness derives some defaults (set_derived_values), a user given value is not overridden:
#   aobasis: from the molecule (guess file);  pure_ae: false if any pseudo atom
#   nalpha, nbeta: from the nuclear charge, charge and nopen (spin_restricted=false if they differ)
#   nmo_alpha, nmo_beta: nalpha + nvalpha, nbeta + nvbeta
#   l: bounding cube of the molecule + 50 bohr
#   pointgroup: c1 if localized, otherwise the one of the molecule (symmetry and localization exclude each other)
#   localize: canon if aocc/bocc have a hole (0 entry)
#   population_basis: = aobasis;  dispersion_functional: = xc if dispersion is set and xc != hf
# frayedends overrides (MadPNO / AtomicBasisProjector): xc="hf", L = MadWorld cell_width/2, k = MadWorld k,
#   econv=1e-4, dconv=5e-4, localize="boys" (AtomicBasisProjector also aobasis and ncf)
DFT_PARAMETERS = {
    "prefix": (str, "mad", "prefixes your output/restart/json/plot/etc files"),
    "charge": (float, 0.0, "total molecular charge"),
    "xc": (str, "hf", "XC input line"),
    "hfexalg": (
        str,
        "multiworld_row",
        "hf exchange algorithm; multiworld bounds memory, needs one subworld per rank (choices: multiworld, multiworld_row, fetch_compute, smallmem, largemem)",
    ),
    "hfex_granularity": (
        int,
        1,
        "hfexalg=multiworld: orbital batches per rank; >1 balances better",
    ),
    "hfex_cost_aware": (
        bool,
        True,
        "hfexalg=multiworld: place tasks by measured cost, not by count",
    ),
    "hfex_accumulation": (
        int,
        2,
        "hfexalg=multiworld: gather tile results 1=per subworld, 2=per node",
    ),
    "bsh_apply": (
        str,
        "auto",
        "BSH apply backend; auto=macrotask when multinode or at tight thresh, else tile (choices: auto, tile, macrotask, plain)",
    ),
    "memory": (
        list[str],
        ["storefunction", "nodereplicated", "distributed"],
        "memory algorithm for storing functions (storing,cloud,target)",
    ),
    "smear": (float, 0.0, "smearing parameter"),
    "econv": (float, 1.0e-5, "energy convergence"),
    "dconv": (float, 1.0e-4, "density convergence"),
    "convergence_criteria": (
        list[str],
        ["bsh_residual", "total_energy"],
        "possible values are: bsh_residual, total_energy, each_energy, density",
    ),
    "k": (int, -1, "polynomial order"),
    "l": (float, 20, "user coordinates box size"),
    "deriv": (str, "abgv", "derivative method (choices: abgv, bspline, ble)"),
    "dft_deriv": (
        str,
        "bspline",
        "derivative method for gga potentials (choices: abgv, bspline, ble)",
    ),
    "xc_weak_gga": (bool, False, "weak form of the semilocal xc potential; nemo only"),
    "maxrotn": (float, 0.25, "step restriction used in autoshift algorithm"),
    "nvalpha": (int, 0, "number of alpha virtuals to compute"),
    "nvbeta": (int, 0, "number of beta virtuals to compute"),
    "aocc": (
        list[float],
        [],
        "explicit alpha occupations (0 or 1) by orbital index within the occupied span; a 0 is a hole",
    ),
    "bocc": (
        list[float],
        [],
        "explicit beta occupations (0 or 1) by orbital index within the occupied span; a 0 is a hole",
    ),
    "nopen": (int, 0, "number of unpaired electrons = nalpha-nbeta"),
    "maxiter": (int, 25, "maximum number of iterations"),
    "nio": (int, 1, "no. of io servers to use"),
    "spin_restricted": (bool, True, "true if spin restricted"),
    "plotlo": (int, 0, "range of MOs to print (for both spins if polarized"),
    "plothi": (int, -1, "range of MOs to print (for both spins if polarized"),
    "plotdens": (bool, False, "If true print the density at convergence"),
    "plotcoul": (
        bool,
        False,
        "If true plot the total coulomb potential at convergence",
    ),
    "plotcube": (
        bool,
        False,
        "If true also write Gaussian .cube files (for Avogadro/VMD) alongside .dx",
    ),
    "localize": (
        str,
        "new",
        "localization method (choices: pm, boys, new, new_sys, cholesky, canon)",
    ),
    "pointgroup": (
        str,
        "c1",
        "use point (sub) group symmetry if not localized (choices: c1, c2, ci, cs, c2v, c2h, d2, d2h)",
    ),
    "restart": (
        str,
        "auto",
        "where the initial orbitals come from: auto picks between none/iterate/read_only by what is on disk (choices: auto, none, iterate, read_only, ao, nwchem)",
    ),
    "save": (bool, True, "if true save orbitals to disk"),
    "maxsub": (int, 10, "size of iterative subspace ... set to 0 or 1 to disable"),
    "orbitalshift": (
        float,
        0.0,
        "scf orbital shift: shift the occ orbitals to lower energies",
    ),
    "npt_plot": (int, 101, "no. of points to use in each dim for plots"),
    "plot_cell": (
        list[float],
        [],
        "lo hi in each dimension for plotting (default is all space)",
    ),
    "aobasis": (
        str,
        "6-31g",
        "AO basis used for initial guess (6-31gss, 6-31g, 3-21g, sto-6g, sto-3g)",
    ),
    "derivatives": (bool, False, "if true calculate nuclear derivatives"),
    "dipole": (bool, False, "if true calculate dipole moment"),
    "conv_only_dens": (
        bool,
        False,
        "if true remove bsh_residual from convergence criteria (deprecated)",
    ),
    "psp_calc": (bool, False, "pseudopotential calculation for all atoms"),
    "pcm_data": (str, "none", "do a PCM (solvent) calculation"),
    "dispersion": (
        str,
        "none",
        "DFT-D3 dispersion correction (choices: none, d3bj, d3zero)",
    ),
    "dispersion_functional": (
        str,
        "none",
        "functional whose D3 damping parameters to use",
    ),
    "dispersion_atm": (
        bool,
        False,
        "include the three-body Axilrod-Teller-Muto dispersion term",
    ),
    "pure_ae": (bool, True, "pure all electron calculation with no pseudo-atoms"),
    "print_level": (
        int,
        3,
        "0: no output; 1: final energy; 2: iterations; 3: timings; 10: debug",
    ),
    "population": (
        list[str],
        ["none"],
        "atomic charges (and spin populations) of the final orbitals, printed at the end: none, mulliken, lowdin, iao; several allowed, e.g. [iao,mulliken]",
    ),
    "population_basis": (
        str,
        "6-31g",
        "basis set for the mulliken and lowdin populations (default: the value of aobasis)",
    ),
    "population_minbasis": (str, "sto-3g", "minimal basis set for the iao populations"),
    "molecular_structure": (
        str,
        "inputfile",
        "where to read the molecule from: inputfile or name from the library",
    ),
    "nalpha": (int, -1, "number of alpha spin electrons"),
    "nbeta": (int, -1, "number of beta spin electrons"),
    "nmo_alpha": (int, -1, "number of alpha spin molecular orbitals"),
    "nmo_beta": (int, -1, "number of beta spin molecular orbitals"),
    "lo": (float, 1.0e-10, "smallest length scale we need to resolve"),
    "protocol": (list[float], [1.0e-4, 1.0e-6], "calculation protocol"),
    "nv_extra": (
        int,
        0,
        "extra virtuals converged first and dropped stepwise down to nvalpha",
    ),
    "nv_step": (int, 0, "virtuals dropped per step-down stage (0: all extras at once)"),
    "nv_its": (int, 5, "maximum iterations per intermediate step-down stage"),
    "freeze_occupied": (
        bool,
        False,
        "iterate only the virtuals in the mean field of restarted occupied orbitals, which stay fixed",
    ),
    "vnucextra": (int, 2, "load balance parameter for nuclear pot"),
    "nwfile": (
        str,
        "none",
        "Base name of nwchem output files (.out and .movecs extensions) to read from",
    ),
}
