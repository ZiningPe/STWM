*! stwm_build.ado  version 1.0.0  27Feb2026
*! Build a Spatial-Temporal Weight Matrix (STWM) via Python stwm package
*! STWM = TWM ⊗ SWM  (Kronecker product)
*!
*! Syntax:
*!   stwm_build depvar [if] [in], SWM(filename) N(#) T(#)
*!              [DECay(string) PARAM(real) SAVEAs(name)
*!               WINSorize(real) MINABS(real)]

program define stwm_build, rclass
    version 16.0

    syntax varname(numeric) [if] [in] ,   ///
        SWM(string)                        ///  path to SWM file (.dta/.csv)
        N(integer)                         ///  number of spatial units
        T(integer)                         ///  number of time periods
        [DECay(string)                     ///  morans|gearyc|getis|gini|exponential|linear|power
         PARAM(real 0.5)                   ///  decay param (for exponential/linear/power)
         SAVEAs(name)                      ///  name to register STWM as (default: STWM)
         WINSorize(real 0.95)              ///  winsorize quantile for ratio TWMs
         MINABS(real 0.001)]               //   min absolute value guard for ratios

    marksample touse

    // ── Defaults ────────────────────────────────────────────────────────────
    if "`decay'"  == "" local decay  "morans"
    if "`saveas'" == "" local saveas "STWM"

    // ── Validate ─────────────────────────────────────────────────────────────
    qui count if `touse'
    local nT = r(N)
    if `nT' != `n' * `t' {
        di as error "Observations in sample (`nT') ≠ n×T = `n'×`t' = " `n'*`t'
        di as error "Ensure data is sorted in time-major order (year × city)."
        exit 198
    }

    local depvar "`varlist'"

    // ── Call Python ──────────────────────────────────────────────────────────
    di as text "Building STWM: decay=`decay', n=`n', T=`t', saveas=`saveas' ..."

    python: _stwm_build(        ///
        "`depvar'",             ///
        "`touse'",              ///
        "`swm'",                ///
        `n', `t',               ///
        "`decay'", `param',     ///
        `winsorize', `minabs',  ///
        "`saveas'"              ///
    )

    // ── Retrieve results stored by Python ────────────────────────────────────
    local spath = "``saveas'_path'"    // global set by Python

    return local  name    "`saveas'"
    return local  path    "`spath'"
    return local  decay   "`decay'"
    return scalar n       = `n'
    return scalar T       = `t'
    return scalar nT      = `nT'
    return scalar spectral_radius = `r_spectral_'
    return scalar twm_passed      = `r_twm_passed_'

    di as result "  STWM `saveas' registered. Shape: " `nT' "×" `nT'
    di as result "  TWM spectral radius: " %6.4f `r_spectral_'
    di as result "  TWM admissibility:   " cond(`r_twm_passed_', "PASSED ✓", "WARNING ✗")
    di as text   "  Access via: w(`saveas')"

end


// ────────────────────────────────────────────────────────────────────────────
// Python bridge — defined once, persists for the entire Stata session
// ────────────────────────────────────────────────────────────────────────────
python:
import os, sys, tempfile
import numpy as np

# ── Persistent STWM file store ───────────────────────────────────────────────
_STWM_DIR   = tempfile.mkdtemp(prefix="stwm_stata_")
_STWM_STORE = {}   # name → numpy file path

def _stwm_save(name, arr):
    path = os.path.join(_STWM_DIR, f"{name}.npy")
    np.save(path, arr)
    _STWM_STORE[name] = path
    return path

def _stwm_load(name):
    if name not in _STWM_STORE:
        raise KeyError(
            f"STWM '{name}' not found in this session.\n"
            f"  Run: stwm_build ..., saveas({name})\n"
            f"  Available: {list(_STWM_STORE.keys())}"
        )
    return np.load(_STWM_STORE[name])

def _stata_vars_to_numpy(varnames_str, touse_var):
    """Read Stata variables into numpy arrays, respecting if/in sample."""
    from sfi import Data
    touse_raw = Data.get(touse_var)
    mask      = np.array([bool(v) for v in touse_raw])
    result    = {}
    for v in varnames_str.split():
        raw = Data.get(v)
        result[v] = np.array(raw, dtype=float)[mask]
    return result, mask

def _load_swm(swm_path):
    """Load spatial weight matrix from .dta or .csv file."""
    import subprocess, os
    from sfi import SFIToolkit
    ext = os.path.splitext(swm_path)[1].lower()
    if ext == ".dta":
        # Load via Stata, convert to numpy
        SFIToolkit.stata(f'quietly preserve')
        SFIToolkit.stata(f'quietly use "{swm_path}", clear')
        from sfi import Data
        nobs = Data.getObsCount()
        ncols = Data.getVarCount()
        W = np.zeros((nobs, ncols))
        for j in range(ncols):
            W[:, j] = np.array(Data.get(j))
        SFIToolkit.stata('quietly restore')
    elif ext in (".csv", ".txt"):
        W = np.loadtxt(swm_path, delimiter=",")
    elif ext == ".npy":
        W = np.load(swm_path)
    else:
        raise ValueError(f"Unsupported SWM file format: {ext}. Use .dta, .csv, or .npy")
    # Row-standardise
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W /= rs
    return W.astype(float)

# ── Main build function ──────────────────────────────────────────────────────
def _stwm_build(depvar, touse_var, swm_path, n, T, decay, param,
                winsorize, minabs, saveas):
    from sfi import Macro, Scalar
    import stwm as pkg

    # 1. Get Y from Stata
    vdata, mask = _stata_vars_to_numpy(depvar, touse_var)
    Y = vdata[depvar]

    # 2. Load SWM
    W = _load_swm(swm_path)
    if W.shape[0] != n:
        raise ValueError(f"SWM rows ({W.shape[0]}) ≠ n ({n})")

    # 3. Build TWM
    decay = decay.lower().strip()
    if decay in ("morans", "moran", "moran_i"):
        seq = [pkg.compute_morans_i(Y[t*n:(t+1)*n], W) for t in range(T)]
        TWM = pkg.build_twm_morans(seq, winsorize_quantile=winsorize, min_abs=minabs)
    elif decay in ("gearyc", "geary_c", "geary"):
        seq = [pkg.compute_geary_c(Y[t*n:(t+1)*n], W) for t in range(T)]
        TWM = pkg.build_twm_gearyc(seq, winsorize_quantile=winsorize, min_abs=minabs)
    elif decay in ("getis", "getis_ord", "getis_ord_g"):
        seq = [pkg.compute_getis_ord_g(Y[t*n:(t+1)*n], W) for t in range(T)]
        TWM = pkg.build_twm_getis_ord(seq, winsorize_quantile=winsorize, min_abs=minabs)
    elif decay in ("gini", "spatial_gini"):
        seq = [pkg.compute_spatial_gini(Y[t*n:(t+1)*n], W) for t in range(T)]
        TWM = pkg.build_twm_spatial_gini(seq, winsorize_quantile=winsorize, min_abs=minabs)
    elif decay in ("exponential", "exp"):
        TWM = pkg.build_twm_decay(T, decay_type="exponential", param=param)
    elif decay in ("linear", "lin"):
        TWM = pkg.build_twm_decay(T, decay_type="linear", param=param)
    elif decay in ("power", "pow"):
        TWM = pkg.build_twm_decay(T, decay_type="power", param=param)
    else:
        raise ValueError(f"Unknown decay type: '{decay}'. "
                         "Choose: morans|gearyc|getis|gini|exponential|linear|power")

    # 4. Admissibility check
    chk = pkg.twm_stability_check(TWM)

    # 5. Build STWM = kron(TWM, SWM)
    STWM = pkg.build_stwm(TWM, W, row_standardize=True)

    # 6. Save to temp file, register name
    path_stwm = _stwm_save(saveas, STWM)
    path_twm  = _stwm_save(f"{saveas}_TWM", TWM)
    path_swm  = _stwm_save(f"{saveas}_SWM", W)

    # 7. Return info to Stata via globals
    Macro.setGlobal(f"{saveas}_path",     path_stwm)
    Macro.setGlobal(f"{saveas}_twm_path", path_twm)
    Macro.setGlobal(f"{saveas}_swm_path", path_swm)
    Macro.setGlobal(f"{saveas}_decay",    decay)
    Macro.setGlobal(f"{saveas}_n",        str(n))
    Macro.setGlobal(f"{saveas}_T",        str(T))

    Scalar.setValue("r_spectral_",   float(chk["spectral_radius"]))
    Scalar.setValue("r_twm_passed_", 1.0 if chk["passed"] else 0.0)

end
