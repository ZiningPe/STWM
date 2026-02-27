*! stwm_sdm.ado  version 1.0.0  27Feb2026
*! Spatial Durbin Model (SDM) using a pre-built STWM
*! Estimation stored as e()-class (compatible with test, lincom, etc.)
*!
*! Syntax:
*!   stwm_sdm depvar indepvars [if] [in],
*!            W(name) [METHOD(string) EFFects(string) N(#) NOCONStant]

program define stwm_sdm, eclass
    version 16.0

    syntax varlist(min=2 numeric) [if] [in] , ///
        W(name)                               ///  STWM name (from stwm_build)
        [METHOD(string)                       ///  ml|qml|iv|gmm|bayes
         EFFects(string)                      ///  none|fe|re
         N(integer 0)                         ///  n units (required for fe/re)
         NOCONStant                           ///  suppress intercept
         NDRaws(integer 2000)                 ///  MCMC draws (bayes only)
         NBUrn(integer 500)]                  //   MCMC burn-in (bayes only)

    marksample touse

    // ── Defaults ─────────────────────────────────────────────────────────────
    if "`method'"  == "" local method  "ml"
    if "`effects'" == "" local effects "none"
    local method  = lower("`method'")
    local effects = lower("`effects'")

    // ── Parse varlist ─────────────────────────────────────────────────────────
    tokenize `varlist'
    local depvar "`1'"
    macro shift
    local indepvars "`*'"
    local k : word count `indepvars'

    // ── Check STWM exists ─────────────────────────────────────────────────────
    if "${`w'_path}" == "" {
        di as error "STWM '`w'' not found. Run: stwm_build ..., saveas(`w')"
        exit 111
    }
    if "`effects'" != "none" & `n' == 0 {
        di as error "Option n(#) required when effects(fe/re) is specified."
        exit 198
    }

    di as text "Fitting SDM: method=`method', effects=`effects' ..."

    // ── Call Python ───────────────────────────────────────────────────────────
    python: _stwm_fit(                ///
        "`depvar'",                   ///
        "`indepvars'",                ///
        "`touse'",                    ///
        "`w'",                        ///
        "sdm",                        ///
        "`method'",                   ///
        "`effects'", `n',             ///
        `ndraws', `nburn',            ///
        "`nocons'" != ""              ///
    )

    // ── Post results to e() ───────────────────────────────────────────────────
    _stwm_post_eresults "`depvar'" "`indepvars'" "`w'" "SDM" "`method'" "`effects'"

    di as text ""
    di as result "SDM estimation complete."
    di as text   "  Spatial parameter (δ): " %8.4f e(rho) ///
                 "  SE: " %8.4f e(rho_se)
    di as text   "  Use estat effects  to display direct/indirect/total"
    di as text   "  Use test, lincom   as usual"

end


// ── Post e()-class results ───────────────────────────────────────────────────
program define _stwm_post_eresults, eclass
    args depvar indepvars wname model method effects

    // Retrieve coefficient vector + VCV stored as Stata matrices by Python
    // (Python stored them as r(b_mat_) and r(V_mat_))
    matrix b_post = r(b_mat_)
    matrix V_post = r(V_mat_)

    // Post to e()
    ereturn post b_post V_post, esample(`_touse_marker_') depname(`depvar')

    // Scalars
    ereturn scalar rho    = scalar(e_rho_)
    ereturn scalar rho_se = scalar(e_rho_se_)
    ereturn scalar sigma2 = scalar(e_sigma2_)
    ereturn scalar N      = scalar(e_N_)
    ereturn scalar k      = scalar(e_k_)
    if scalar(e_theta_re_) != . {
        ereturn scalar theta_re = scalar(e_theta_re_)
    }

    // Matrices: effects decomposition
    ereturn matrix direct   = r(direct_mat_)
    ereturn matrix indirect = r(indirect_mat_)
    ereturn matrix total    = r(total_mat_)
    ereturn matrix direct_se   = r(direct_se_mat_)
    ereturn matrix indirect_se = r(indirect_se_mat_)
    ereturn matrix total_se    = r(total_se_mat_)

    // Macros
    ereturn local  cmd       "stwm_sdm"
    ereturn local  model     "`model'"
    ereturn local  method    "`method'"
    ereturn local  effects   "`effects'"
    ereturn local  W_name    "`wname'"
    ereturn local  depvar    "`depvar'"
    ereturn local  indepvars "`indepvars'"

end


// ── estat effects — post-estimation effects table ────────────────────────────
program define estat
    version 16.0
    if "`e(cmd)'" == "" {
        di as error "No estimation results found."
        exit 301
    }
    if !inlist("`e(cmd)'", "stwm_sdm","stwm_sar","stwm_sem","stwm_slx") {
        di as error "estat effects only available after stwm_* commands."
        exit 301
    }
    _stwm_effects_table
end


program define _stwm_effects_table
    version 16.0
    local indepvars = e(indepvars)
    local k : word count `indepvars'

    di as text ""
    di as text " Effect Decomposition — `e(model)' (`e(method)')"
    di as text " ─────────────────────────────────────────────────────────────────"
    di as text %14s "Variable" %12s "Direct" %12s "Indirect" %12s "Total" ///
               %10s "D.SE" %10s "I.SE"
    di as text " ─────────────────────────────────────────────────────────────────"

    forvalues i = 1/`k' {
        local vn : word `i' of `indepvars'
        local d   = el(e(direct),   1, `i')
        local ind = el(e(indirect), 1, `i')
        local tot = el(e(total),    1, `i')
        local dse = el(e(direct_se), 1, `i')
        local ise = el(e(indirect_se),1,`i')
        di as text %14s "`vn'" ///
           as result %12.4f `d' %12.4f `ind' %12.4f `tot' ///
           as text   %10.4f `dse' %10.4f `ise'
    }
    di as text " ─────────────────────────────────────────────────────────────────"
    di as text "  δ (spatial) = " as result %8.4f e(rho) ///
               as text "  SE = " as result %8.4f e(rho_se)
end


// ────────────────────────────────────────────────────────────────────────────
// Python bridge — shared fitting function for all 4 model types
// ────────────────────────────────────────────────────────────────────────────
python:
import numpy as np

def _stwm_fit(depvar, indepvars_str, touse_var, w_name,
              model_type, method, effects, n_units,
              n_draws, n_burn, noconst):
    """
    Fit any STWM spatial model and post results back to Stata.

    Parameters
    ----------
    model_type : 'sdm'|'sar'|'sem'|'slx'
    method     : 'ml'|'qml'|'iv'|'gmm'|'bayes'
    effects    : 'none'|'fe'|'re'
    """
    from sfi import Data, Scalar, Matrix, Macro
    import stwm as pkg

    # 1. Load data from Stata
    vlist = [depvar] + indepvars_str.split()
    vdata, mask = _stata_vars_to_numpy(" ".join(vlist), touse_var)
    Y = vdata[depvar]
    X = np.column_stack([vdata[v] for v in indepvars_str.split()])
    N, k = X.shape

    # 2. Load STWM
    W = _stwm_load(w_name)
    if W.shape[0] != N:
        raise ValueError(
            f"STWM rows ({W.shape[0]}) ≠ observations ({N}). "
            f"Check n and T or that the same if/in applies."
        )

    # 3. Select model class
    model_type = model_type.lower()
    ModelClass = {
        "sdm": pkg.SDMModel,
        "sar": pkg.SpatialLagModel,
        "sem": pkg.SpatialErrorModel,
        "slx": pkg.SLXModel,
    }[model_type]

    # 4. Fit
    fit_kw = {}
    if effects != "none":
        fit_kw["effects"]  = effects
        fit_kw["n_units"]  = n_units
    if method in ("bayes",):
        fit_kw["n_draws"] = n_draws
        fit_kw["n_burn"]  = n_burn

    if model_type in ("sdm", "sar"):
        m = ModelClass(W).fit(Y, X, method=method, **fit_kw)
    elif model_type in ("sem",):
        m = ModelClass(W).fit(Y, X, **fit_kw)
    else:  # slx
        m = ModelClass(W).fit(Y, X, **fit_kw)

    res = m.summary()

    # 5. Build coefficient vector and VCV for ereturn post
    #    Ordering: beta_X (k) | theta_WX (k, SDM only) | _cons | rho/lam
    beta_X   = np.asarray(res.get("beta_X",   np.zeros(k)))
    intercpt = float(res.get("intercept", 0.0))
    rho_val  = float(res.get("rho",  res.get("lam",  0.0)))
    rho_se   = float(res.get("rho_se", res.get("lam_se", np.nan)))

    se_bX    = np.asarray(res.get("beta_X_se", np.zeros(k)))
    se_ic    = np.nan  # intercept SE not always returned separately

    var_names = indepvars_str.split()

    if model_type == "sdm":
        theta   = np.asarray(res.get("theta_WX", np.zeros(k)))
        se_th   = np.asarray(res.get("theta_WX_se", np.zeros(k)))
        wx_names = ["W_" + v for v in var_names]
        b_vec   = np.concatenate([beta_X, theta, [intercpt, rho_val]])
        se_vec  = np.concatenate([se_bX,  se_th,  [np.nan,  rho_se]])
        col_names = var_names + wx_names + ["_cons", "rho_delta"]
    elif model_type == "sar":
        b_vec   = np.concatenate([beta_X, [intercpt, rho_val]])
        se_vec  = np.concatenate([se_bX,  [np.nan,  rho_se]])
        col_names = var_names + ["_cons", "rho"]
    elif model_type == "sem":
        lam_val  = float(res.get("lam", 0.0))
        lam_se   = float(res.get("lam_se", np.nan))
        b_vec    = np.concatenate([beta_X, [intercpt, lam_val]])
        se_vec   = np.concatenate([se_bX,  [np.nan,  lam_se]])
        col_names = var_names + ["_cons", "lambda"]
    else:  # slx
        theta   = np.asarray(res.get("theta_WX", np.zeros(k)))
        se_th   = np.asarray(res.get("theta_WX_se", np.zeros(k)))
        wx_names = ["W_" + v for v in var_names]
        b_vec   = np.concatenate([beta_X, theta, [intercpt]])
        se_vec  = np.concatenate([se_bX,  se_th,  [np.nan]])
        col_names = var_names + wx_names + ["_cons"]

    # Build b (1 × p) and V (p × p diagonal from se^2)
    p   = len(b_vec)
    b_m = np.where(np.isnan(b_vec), 0.0, b_vec).reshape(1, p)
    V_m = np.diag(np.where(np.isnan(se_vec**2), 1e-10, se_vec**2))

    # 6. Store as Stata matrices (r() — will be posted to e() in Stata)
    b_list = b_m.tolist()
    V_list = V_m.tolist()
    Matrix.store("b_mat_", b_list)
    Matrix.store("V_mat_", V_list)

    # Set column names on b
    from sfi import SFIToolkit as st
    cnames = " ".join(col_names)
    st.stata(f'matrix colnames b_mat_ = {cnames}')
    st.stata(f'matrix rownames b_mat_ = {depvar}')
    st.stata(f'matrix colnames V_mat_ = {cnames}')
    st.stata(f'matrix rownames V_mat_ = {cnames}')

    # 7. Store scalars
    Scalar.setValue("e_rho_",      rho_val)
    Scalar.setValue("e_rho_se_",   rho_se)
    Scalar.setValue("e_sigma2_",   float(res.get("sigma2", np.nan)))
    Scalar.setValue("e_N_",        float(N))
    Scalar.setValue("e_k_",        float(k))
    theta_re = res.get("theta_re")
    Scalar.setValue("e_theta_re_", float(theta_re) if theta_re is not None else float("nan"))

    # 8. Store effects decomposition as row-vectors
    d    = np.asarray(res.get("direct",      np.zeros(k))).reshape(1, k)
    ind  = np.asarray(res.get("indirect",    np.zeros(k))).reshape(1, k)
    tot  = np.asarray(res.get("total",       np.zeros(k))).reshape(1, k)
    dse  = np.asarray(res.get("direct_se",   np.zeros(k))).reshape(1, k)
    ise  = np.asarray(res.get("indirect_se", np.zeros(k))).reshape(1, k)
    tse  = np.asarray(res.get("total_se",    np.zeros(k))).reshape(1, k)

    Matrix.store("direct_mat_",     d.tolist())
    Matrix.store("indirect_mat_",   ind.tolist())
    Matrix.store("total_mat_",      tot.tolist())
    Matrix.store("direct_se_mat_",  dse.tolist())
    Matrix.store("indirect_se_mat_",ise.tolist())
    Matrix.store("total_se_mat_",   tse.tolist())

    for mname in ["direct_mat_","indirect_mat_","total_mat_",
                  "direct_se_mat_","indirect_se_mat_","total_se_mat_"]:
        st.stata(f"matrix colnames {mname} = {' '.join(var_names)}")

    # 9. touse marker (expose to Stata for ereturn post esample)
    Macro.setLocal("_touse_marker_", touse_var)

end
