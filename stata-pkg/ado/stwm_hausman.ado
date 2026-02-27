*! stwm_hausman.ado  version 1.0.0  27Feb2026
*! Hausman + Sargan + Redundancy exogeneity battery for STWM
*!
*! Syntax:
*!   stwm_hausman depvar indepvars [if] [in], W(name) WG(name)
*!
*!   W  = STWM under test (from stwm_build)
*!   WG = exogenous benchmark STWM (static block-diagonal or inverse-distance)

program define stwm_hausman, rclass
    version 16.0

    syntax varlist(min=2 numeric) [if] [in] , ///
        W(name)                               ///
        WG(name)

    marksample touse

    tokenize `varlist'
    local depvar "`1'"; macro shift; local indepvars "`*'"

    foreach nm in `w' `wg' {
        if "${`nm'_path}" == "" {
            di as error "STWM '`nm'' not found. Run: stwm_build ..., saveas(`nm')"
            exit 111
        }
    }

    di as text "Running STWM exogeneity test battery ..."
    di as text "  W  (STWM to test) : `w'"
    di as text "  WG (benchmark)    : `wg'"

    python: _stwm_exog_test("`depvar'", "`indepvars'", "`touse'", "`w'", "`wg'")

    di as text ""
    di as text " ═══════════════════════════════════════════════════════════════"
    di as text "  STWM Exogeneity Test Battery"
    di as text " ═══════════════════════════════════════════════════════════════"
    di as text ""
    di as text "  [1] Hausman Test  (H₀: STWM exogenous — estimates ≈ benchmark)"
    di as result "      H = " %8.4f scalar(r_hausman_H_) "   p = " %8.4f scalar(r_hausman_p_)
    di as result "      Reject H₀: `r_hausman_reject_'"
    di as text ""
    di as text "  [2] Sargan J-Test  (H₀: instruments orthogonal to errors)"
    di as result "      J = " %8.4f scalar(r_sargan_J_)  "   p = " %8.4f scalar(r_sargan_p_)
    di as result "      Reject H₀: `r_sargan_reject_'"
    di as text ""
    di as text "  [3] Redundancy F-Test  (H₀: STWM lags redundant given WG lags)"
    di as result "      F = " %8.4f scalar(r_redund_F_)  "   p = " %8.4f scalar(r_redund_p_)
    di as result "      Reject H₀: `r_redund_reject_'"
    di as text " ═══════════════════════════════════════════════════════════════"
    di as text "  Fail to reject H₀ in [1] → STWM is exogenous ✓"

    return scalar hausman_H     = scalar(r_hausman_H_)
    return scalar hausman_p     = scalar(r_hausman_p_)
    return local  hausman_reject "`r_hausman_reject_'"
    return scalar sargan_J      = scalar(r_sargan_J_)
    return scalar sargan_p      = scalar(r_sargan_p_)
    return local  sargan_reject  "`r_sargan_reject_'"
    return scalar redund_F      = scalar(r_redund_F_)
    return scalar redund_p      = scalar(r_redund_p_)
    return local  redund_reject  "`r_redund_reject_'"

end


python:
import numpy as np

def _stwm_exog_test(depvar, indepvars_str, touse_var, w_name, wg_name):
    from sfi import Scalar, Macro
    import stwm as pkg

    vlist = [depvar] + indepvars_str.split()
    vdata, _ = _stata_vars_to_numpy(" ".join(vlist), touse_var)
    Y  = vdata[depvar]
    X  = np.column_stack([vdata[v] for v in indepvars_str.split()])
    W_stwm = _stwm_load(w_name)
    W_g    = _stwm_load(wg_name)

    report = pkg.stwm_exogeneity_report(Y, X, W_stwm, W_g, label_tw=w_name)

    h = report["hausman"]
    s = report["sargan"]
    r = report["redundancy"]

    def _f(v): return float(v) if v is not None and v == v else float("nan")

    Scalar.setValue("r_hausman_H_",     _f(h.get("H_stat")))
    Scalar.setValue("r_hausman_p_",     _f(h["p_value"]))
    Macro.setLocal( "r_hausman_reject_","Yes" if h["reject_H0"] else "No")
    Scalar.setValue("r_sargan_J_",      _f(s.get("J_stat")))
    Scalar.setValue("r_sargan_p_",      _f(s["p_value"]))
    Macro.setLocal( "r_sargan_reject_", "Yes" if s.get("reject_H0", False) else "No")
    Scalar.setValue("r_redund_F_",      _f(r.get("F_stat")))
    Scalar.setValue("r_redund_p_",      _f(r["p_value"]))
    Macro.setLocal( "r_redund_reject_", "Yes" if r.get("reject_H0", False) else "No")

end
