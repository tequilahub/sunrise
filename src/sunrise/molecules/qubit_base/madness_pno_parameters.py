"""
Keywords of the madness PNO calculation, the "pno" block of the MadPNO input (madness/chem/PNOParameters.h)
name: (type, madness default, description)
Given to frayedends as MadPNO(..., pno={"tpno": 1e-9, ...}), they end up as --pno="key=value; ..."
"""

# PNOParameters::initialize_pno_parameters
# madness derives some defaults (set_derived_values):
#   freeze: 1 per atom Li-Ne, 5 per atom Na-Ar (H, He: 0), beyond set it by hand
#   no_guess, restart: = no_compute;  tpno_tight: = 0.01 * tpno
#   if adaptive_solver: guesstype="exop", exop="multipole", econv_macro=econv_micro=thresh, econv_pairs=0.1*thresh
# frayedends.MadPNO overrides: maxrank=10, f12=false, thresh=MadWorld thresh, diagonal=True, freeze=0 if frozen_core=False
PNO_PARAMETERS = {
    "rank_increase": (int, 15, "maximum rank to increase in every macroiteration"),
    "chunk": (
        int,
        100,
        "chunk of functions operated on in parallel when G or K is applied (prevent memory shortage)",
    ),
    "debug": (bool, False, "debug mode"),
    "freeze": (int, 0, "frozen core approximation"),
    "maxrank": (int, 999, "maximal pno rank for all pairs"),
    "guesstype": (
        str,
        "exop",
        "the guesstype: exop (recommended, multiply polynomial excitation operators onto the occupied orbitals), empty (don't compute a guess), scf (use the ao basis from the SCF solver), partial_wave (create s,p,d ..., type functions and place on atoms), predefined (same as partial_wave but for predefined sets)",
    ),
    "exop": (
        str,
        "multipole",
        "this string defines which excitation operators can be used, use 'dipole', 'dipole+', 'quadrupole', 'ocopole', or 'multipole', need to set guesstype to 'exop' which is the default",
    ),
    "exop_trigo": (bool, True, "use trigonometric excitation operators ( x --> sin(x)"),
    "partial_wave": (
        str,
        "",
        "atom centered partial wave guess of format like 'h-2s1p-o-3s2p1d', need to set guesstype to 'partial_wave' and adaptive_solver to 'none'",
    ),
    "predefined": (
        str,
        "",
        "predefined partial wave guesses of type pvxz with x=d,t,q,5,6, need to set guesstype to 'predefined' and adaptive_solver to 'none'",
    ),
    "maxiter_t": (int, 100, "maximal number of iterations in the amplitude solver"),
    "maxiter_micro": (
        int,
        10,
        "Maximum of iterations for every cycle in the adaptive solver",
    ),
    "maxiter_macro": (
        int,
        10,
        "Maximum of iterations of cycles in the adaptive solver",
    ),
    "tpno": (float, 1.0e-8, "PNO cutoff threshold"),
    "tpno_tight": (float, 1.0e-10, "PNO cutoff for the first iteration"),
    "canonicalize_pno": (
        bool,
        True,
        "canonicalize the pnos before the amplitude solver",
    ),
    "thresh": (float, 1.0e-3, "MRA threshold"),
    "econv_micro": (
        float,
        1.0e-3,
        "Energy convergence for microiterations (Greens function based optimization) in adaptive solver",
    ),
    "econv_pairs": (
        float,
        1.0e-4,
        "ENergy convergence for individual pairs in adaptive solver. Converged pairs are frozen automatically",
    ),
    "econv_macro": (
        float,
        1.0e-3,
        "Energy convergence for macroiterations in adaptive solver, no effect if adaptive_solver is deactivated",
    ),
    "dconv": (float, 1.0e-1, "convergence of every PNO in the Green's function solver"),
    "op_thresh": (float, 1.0e-6, "MRA operator thresh"),
    "restart": (
        str,
        "none",
        "restart pairs of this type, use 'mp2', 'cispd', 'all' or 'none'",
    ),
    "no_compute": (
        str,
        "none",
        "do not compute the pairs of this type, use 'mp2', 'cispd', 'all' or 'none'",
    ),
    "no_opt": (
        str,
        "none",
        "do not optimize the pnos of this type, use 'mp2', 'cispd', 'all' or 'none'",
    ),
    "no_guess": (
        str,
        "none",
        "guess for this type will be empty, use 'mp2', 'cispd', 'all' or 'none'",
    ),
    "adaptive_solver": (
        str,
        "all",
        "Use adaptive solver for those pairs, use 'mp2', 'cispd', 'all' or 'none', works only in combination with guesstype 'exop'",
    ),
    "kain": (bool, True, "use KAIN solver in amplitude solver"),
    "kain_subspace": (int, 5, "subspace size of the KAIN solver (amplitudes)"),
    "f12": (bool, True, "use explicit correlation"),
    "cispd_number": (
        int,
        -1,
        "CIS(D) excitation numbers, to read in correct functions",
    ),
    "cispd_energy": (float, 0.0, "CIS energy"),
    "freeze_pairs": (
        str,
        "none",
        "frozen pairs will not be optimized: Expected format 'a b c d' will freeze pairs ab and cd",
    ),
    "freeze_pairs_of_orbital": (
        list[int],
        [],
        "All pairs which originate from this orbital will not be optimized",
    ),
    "active_pairs_of_orbital": (
        list[int],
        [],
        "All pairs which originate from this orbital will not be frozen all other pairs will, if this vector is not empty",
    ),
    "no_opt_in_first_iteration": (
        bool,
        False,
        "Do not optimize in the first iteration (then the potentials do not have to be evaluated, use this for large guesses)",
    ),
    "exchange": (
        str,
        "full",
        "approximate exchange with 'neglect' or xc functional -> same syntax as moldft",
    ),
    "save_pnos": (
        bool,
        True,
        "Save the OBS-PNOs to a file, before and after orthonormalization.",
    ),
    "diagonal": (bool, False, "Compute only diagonal PNOs"),
}

# F12Parameters::initialize_f12_parameters, same "pno" block, only used with f12=true
F12_PARAMETERS = {
    "abs_c": (
        bool,
        True,
        "use auxilliary basis on f12Q[Kf12] part of energy (only if energytype is HYLLERAAS_ENERGYTYPE). If switched off the part neglected!",
    ),
    "abs_u": (
        bool,
        False,
        "use auxilliary basis on f12QUe part of energy (only if energytype is HYLLERAAS_ENERGYTYPE). If switched off the part is computed in full (recommended)",
    ),
    "cabs_thresh": (float, 1.0e-4, "thresh for auxbasis part in f12 energy"),
    "energytype": (
        str,
        "projected",
        "the energytype is 'projected' or 'hylleraas' functional projected energies do not need auxilliary bases for the evaluation of the f12 energy. It's recommended to use projected_energies! For Hylleraas type you need to specify an auxbas from file OR internal",
    ),
    "gamma": (float, 1.4, "The f12 length scale"),
    "auxbas": (
        str,
        "none",
        "atom centered partial wave guess of format like 'h-2s1p-o-3s2p1d'",
    ),
    "auxbas_file": (
        str,
        "none",
        "use external comp. aux. basis in addition to the pnos as auxbasis. Give the filename as parameter. Give the auxbas in turbomole format. Don't use contractions. If a file is specified the auxbas parameter has no effect",
    ),
}
