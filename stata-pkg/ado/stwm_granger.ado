*! stwm_granger.ado  version 1.0.0  27Feb2026
*! Granger causality: Moran's I → estimated spillover parameters
*! Tests whether spatial-autocorrelation dynamics predict spillover changes
*! (justifies using Moran's I as the basis for TWM construction)
*!
*! Syntax:
*!   stwm_granger depvar [if] [in], W(name) N(#) T(#) [MAXLAG(#) WINDOW(#)]
*!
*!   Requires: stwm_build already run (to access TWM & SWM)
*!             Computes rolling spillover sequence internally

program define stwm_granger, rclass
    version 16.0

    syntax varname(numeric) [if] [in] , ///
        W(name)                         ///  registered STWM name
        N(integer)                      ///  spatial units
        T(integer)                      ///  time periods
        [MAXLAG(integer 3)              ///  maximum Granger lag order
         WINDOW(integer 4)]             //   rolling-window size for spillover seq

    marksample touse

    if "${`w'_path}" == "" {
        di as error "STWM '`w'' not found. Run: stwm_build ..., saveas(`w')"
        exit 111
    }

    di as text "Granger Causality Test: Moran's I → Spillover Parameters"
    di as text "  STWM     : `w'"
    di as text "  n, T     : `n', `t'"
    di as text "  Window   : `window' years (for rolling spillover sequence)"
    di as text "  Max lag  : `maxlag'"

    python: _stwm_granger_test(                         ///
        "`varlist'", "`touse'", "`w'", `n', `t',        ///
        `maxlag', `window'                              ///
    )

    di as text ""
    di as text " ═══════════════════════════════════════════════════════════"
    di as text "  Granger Causality: Moran's I → Indirect Effects"
    di as text "  H₀: Moran's I does NOT Granger-cause spillover parameters"
    di as text " ───────────────────────────────────────────────────────────"

    forvalues lag = 1/`maxlag' {
        local F   = scalar(r_gr_F_`lag'_)
        local p   = scalar(r_gr_p_`lag'_)
        local rej = "`r_gr_rej_`lag'_'"
        di as text "  Lag `lag':" ///
           as result "  F = " %8.4f `F' "  p = " %8.4f `p' ///
           as text   "  Reject H₀: `rej'"
    }
    di as text " ═══════════════════════════════════════════════════════════"
    di as text "  Reject H₀ → Moran's I dynamics predict spillovers"
    di as text "  → TWM construction is structurally justified (not circular)"

    return scalar gr_F_lag1   = scalar(r_gr_F_1_)
    return scalar gr_p_lag1   = scalar(r_gr_p_1_)
    return local  gr_rej_lag1 "`r_gr_rej_1_'"

end


python:
import numpy as np

def _stwm_granger_test(depvar, touse_var, w_name, n, T, max_lag, window):
    from sfi import Scalar, Macro, Data
    import stwm as pkg

    # 1. Load Y from Stata
    vdata, _ = _stata_vars_to_numpy(depvar, touse_var)
    Y  = vdata[depvar]
    W  = np.load(_STWM_STORE[f"{w_name}_SWM"])
    TWM = np.load(_STWM_STORE[f"{w_name}_TWM"])

    # 2. Compute annual Moran's I sequence
    morans_seq = np.array([
        pkg.compute_morans_i(Y[t*n:(t+1)*n], W) for t in range(T)
    ])

    # 3. Rolling-window spillover estimates (indirect effects, var 0)
    roll = pkg.rolling_effects(Y, None, TWM, W, n, T, window,
                                pkg.SpatialLagModel, var_names=None)
    valid_wins  = [w for w in roll.get("_windows", [])
                   if "error" not in roll.get(w, {})]
    if len(valid_wins) < max_lag + 5:
        print(f"  WARNING: Only {len(valid_wins)} rolling windows; "
              f"Granger test may be unreliable.")

    spillover_seq = np.array([roll[w].get("indirect", [np.nan])[0]
                               for w in valid_wins])
    end_years     = [int(w.split("\u2013t")[1]) for w in valid_wins]
    morans_aligned = morans_seq[end_years]

    print(f"  Moran's I (end-of-window): {np.round(morans_aligned, 4)}")
    print(f"  Spillover sequence:        {np.round(spillover_seq, 4)}")

    # 4. Run Granger test
    gr = pkg.granger_spillover_test(morans_aligned, spillover_seq,
                                    max_lag=max_lag)

    for lag_key, res in gr.items():
        lag_num = int(lag_key.split("_")[1])
        if "error" in res:
            Scalar.setValue(f"r_gr_F_{lag_num}_", float("nan"))
            Scalar.setValue(f"r_gr_p_{lag_num}_", float("nan"))
            Macro.setLocal(f"r_gr_rej_{lag_num}_", "N/A")
        else:
            Scalar.setValue(f"r_gr_F_{lag_num}_", float(res["F_statistic"]))
            Scalar.setValue(f"r_gr_p_{lag_num}_", float(res["p_value"]))
            Macro.setLocal(f"r_gr_rej_{lag_num}_",
                           "Yes" if res["reject_H0"] else "No")

end
