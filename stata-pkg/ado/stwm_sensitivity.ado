*! stwm_sensitivity.ado  version 1.0.0  27Feb2026
*! Sensitivity analysis: vary TWM hyperparameters, check result stability
*!
*! Syntax:
*!   stwm_sensitivity depvar indepvars [if] [in], SWM(filename) N(#) T(#)
*!                    [WINSQlist(numlist) MAlist(numlist)]

program define stwm_sensitivity, rclass
    version 16.0

    syntax varlist(min=2 numeric) [if] [in] , ///
        SWM(string)                            ///  path to SWM file
        N(integer)                             ///  spatial units
        T(integer)                             ///  time periods
        [WINSQlist(numlist)                    ///  winsorize quantiles (default: .90 .95 .99)
         MAlist(numlist)]                      //   min_abs values (default: 1e-4 1e-3 1e-2)

    marksample touse

    tokenize `varlist'
    local depvar "`1'"; macro shift; local indepvars "`*'"

    if "`winsqlist'" == "" local winsqlist "0.90 0.95 0.99"
    if "`malist'"   == "" local malist    "0.0001 0.001 0.01"

    di as text "Sensitivity analysis: winsorize × min_abs grid ..."
    di as text "  winsorize quantiles : `winsqlist'"
    di as text "  min_abs values      : `malist'"

    python: _stwm_sensitivity(  ///
        "`depvar'",             ///
        "`indepvars'",          ///
        "`touse'",              ///
        "`swm'",                ///
        `n', `t',               ///
        "`winsqlist'",          ///
        "`malist'"              ///
    )

    // Results printed directly by Python; matrix stored as r(sens_table)
    di as text ""
    di as text "  r(sens_table) contains the full sensitivity grid."
    di as text "  r(cv_rho), r(cv_direct_1), r(cv_indirect_1) contain CVs."

    return matrix sens_table = r(sens_table_)
    return scalar cv_rho       = scalar(r_cv_rho_)

end


python:
import numpy as np

def _stwm_sensitivity(depvar, indepvars_str, touse_var, swm_path,
                      n, T, winsq_str, ma_str):
    from sfi import Scalar, Matrix, Macro
    import stwm as pkg

    vlist = [depvar] + indepvars_str.split()
    vdata, _ = _stata_vars_to_numpy(" ".join(vlist), touse_var)
    Y  = vdata[depvar]
    X  = np.column_stack([vdata[v] for v in indepvars_str.split()])
    W  = _load_swm(swm_path)
    var_names = indepvars_str.split()
    k  = X.shape[1]

    winsq_list = [float(q) for q in winsq_str.split()]
    ma_list    = [float(m) for m in ma_str.split()]

    rows  = []
    table = []   # for Stata matrix

    header = f"{'winsq':>8} {'min_abs':>10} {'rho':>8}"
    for i, vn in enumerate(var_names):
        header += f" {'D['+vn+']':>12} {'I['+vn+']':>12}"
    print(header)
    print("-" * len(header))

    for q in winsq_list:
        for ma in ma_list:
            TWM  = pkg.build_twm_morans(
                [pkg.compute_morans_i(Y[t*n:(t+1)*n], W) for t in range(T)],
                winsorize_quantile=q, min_abs=ma)
            STWM = pkg.build_stwm(TWM, W)
            try:
                m   = pkg.SDMModel(STWM).fit(Y, X)
                res = m.summary()
                rho = float(res.get("rho", float("nan")))
                d   = np.asarray(res.get("direct",   np.full(k, float("nan"))))
                ind = np.asarray(res.get("indirect", np.full(k, float("nan"))))
            except Exception as e:
                rho = float("nan")
                d   = np.full(k, float("nan"))
                ind = np.full(k, float("nan"))

            row_vals = [q, ma, rho] + list(d) + list(ind)
            table.append(row_vals)

            line = f"{q:>8.2f} {ma:>10.4f} {rho:>8.4f}"
            for i in range(k):
                line += f" {d[i]:>12.4f} {ind[i]:>12.4f}"
            print(line)

    # CV across grid
    arr = np.array(table)
    rho_col = arr[:, 2]
    cv_rho  = float(np.nanstd(rho_col) / abs(np.nanmean(rho_col))) \
              if abs(np.nanmean(rho_col)) > 1e-10 else float("nan")
    print(f"\n  CV(rho) = {cv_rho:.4f}  "
          f"({'stable' if cv_rho < 0.05 else 'sensitive'})")

    Scalar.setValue("r_cv_rho_", cv_rho)
    Matrix.store("r(sens_table_)", arr.tolist())

end
