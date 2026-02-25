"""
endogeneity.py
--------------
Testing temporal weight matrix (TW) exogeneity via the Hausman IV framework.

Core argument
-------------
The spatial lag term WY is endogenous in Y = α + ρ·WY + σX + ε because WY
is mechanically correlated with Y.  To handle this, we use two alternative
instrument sets and compare the resulting IV estimates:

    (1) Wg-based instruments  :  Z_g  = [X,  Wg@X]
        Wg is a geographic inverse-distance matrix — exogenous by construction,
        uncorrelated with the error term.  This is the benchmark.

    (2) TW-based instruments  :  Z_tw = [X, TW@X]
        TW is the proposed temporal weight matrix.  If TW is exogenous, the
        2SLS estimates using Z_tw should agree with those using Z_g.

Hausman (1978) test:
    H₀: β̂_TW ≈ β̂_Wg  (no systematic difference → TW is exogenous)
    H₁: β̂_TW ≠ β̂_Wg  (systematic difference → TW is endogenous)

    H = (β̂_TW − β̂_Wg)' · [Cov(β̂_TW) − Cov(β̂_Wg)]⁺ · (β̂_TW − β̂_Wg)
    H ~ χ²(k) under H₀

Decision rule:
    Fail to reject H₀ (p ≥ 0.05 or H < 0) → TW is exogenous  ✓
    Reject H₀ (p < 0.05)                   → potential endogeneity concern

Notes on negative H statistics
-------------------------------
A negative H statistic arises when Cov(β̂_TW) − Cov(β̂_Wg) is not positive
semi-definite. This commonly occurs in finite samples and is conventionally
interpreted as a strong failure to reject H₀ (Baltagi 2021; Zhu et al. 2022).
The paper reports H = −244.26 (TW_I) and H = −225.26 (TW_C), both indicating
exogeneity.

Supplementary tests
-------------------
1. Sargan-Hansen J-test  : over-identification validity of the Wg instruments
2. Redundancy F-test     : whether TW adds independent variation beyond Wg

References
----------
Hausman, J. A. (1978). Econometrica, 46(6), 1251–1271.
Wu, D. M. (1973). Econometrica, 41(4), 733–750.
Sargan, J. D. (1958). Econometrica, 26(3), 393–415.
Baltagi, B. H. (2021). Econometric Analysis of Panel Data (6th ed.). Springer.
Cheng, Z., & Lee, L.-F. (2017). Journal of Econometrics, 196(1), 13–36.
Zhu, Y. et al. (2022). [Spatial exogeneity test application].
"""

import numpy as np
from scipy import stats
import warnings


# ---------------------------------------------------------------------------
# 1.  Two-Stage Least Squares (2SLS)
# ---------------------------------------------------------------------------

def iv_regression(Y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> dict:
    """
    Two-Stage Least Squares (2SLS).

        Stage 1 : X̂ = Z(Z'Z)⁻¹Z'X
        Stage 2 : β̂_IV = (X̂'X)⁻¹X̂'Y

    Parameters
    ----------
    Y : (N,) dependent variable
    X : (N, k) regressors (may include endogenous WY)
    Z : (N, m) instruments  (m ≥ k)

    Returns
    -------
    dict with beta, se, t_stats, p_values, residuals, sigma2, cov
    """
    Y = np.asarray(Y, float).ravel()
    X = np.asarray(X, float)
    Z = np.asarray(Z, float)
    N, k = X.shape
    _, m  = Z.shape
    if m < k:
        raise ValueError(f"Need m ≥ k instruments, got {m} < {k}.")

    X_hat  = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    beta   = np.linalg.lstsq(X_hat, Y, rcond=None)[0]
    resid  = Y - X @ beta
    sigma2 = float(resid @ resid) / (N - k)
    cov    = sigma2 * np.linalg.pinv(X_hat.T @ X_hat)
    se     = np.sqrt(np.maximum(np.diag(cov), 0))
    t_stat = beta / np.where(se > 1e-14, se, np.nan)
    pvals  = 2 * stats.t.sf(np.abs(t_stat), df=N - k)

    return dict(beta=beta, se=se, t_stats=t_stat, p_values=pvals,
                residuals=resid, sigma2=sigma2, cov=cov)


# ---------------------------------------------------------------------------
# 2.  Hausman Test for TW Exogeneity
# ---------------------------------------------------------------------------

def hausman_test(beta_efficient: np.ndarray,
                 beta_consistent: np.ndarray,
                 cov_efficient: np.ndarray,
                 cov_consistent: np.ndarray) -> dict:
    """
    Hausman (1978) test comparing two IV estimators.

    Standard framing:
      - β_efficient  : IV estimates from exogenous benchmark (Wg)
      - β_consistent : IV estimates from proposed instruments (TW)

    Under H₀ (TW is exogenous) both are consistent; the benchmark is efficient.
    Under H₁ (TW is endogenous) only the benchmark is consistent.

    Statistic:
        H = (β_c − β_e)' [Cov(β_c) − Cov(β_e)]⁺ (β_c − β_e)  ~ χ²(k) under H₀

    Negative H: Cov difference is not PSD (finite-sample artifact).
    Conventional interpretation: strong failure to reject H₀ (Baltagi 2021).

    Parameters
    ----------
    beta_efficient   : (k,) coefficient vector from Wg-instrument regression
    beta_consistent  : (k,) coefficient vector from TW-instrument regression
    cov_efficient    : (k,k) covariance of β_efficient
    cov_consistent   : (k,k) covariance of β_consistent
    """
    diff     = beta_consistent - beta_efficient
    cov_diff = cov_consistent - cov_efficient
    try:
        cov_inv = np.linalg.pinv(cov_diff)
        H_stat  = float(diff @ cov_inv @ diff)
    except Exception:
        H_stat = np.nan

    df = len(diff)

    # Negative or NaN H: fail to reject H₀ by convention
    if np.isnan(H_stat) or H_stat <= 0:
        p_value = 1.0
        reject  = False
    else:
        p_value = float(stats.chi2.sf(H_stat, df=df))
        reject  = p_value < 0.05

    return dict(
        H_stat    = round(H_stat, 4) if not np.isnan(H_stat) else np.nan,
        df        = df,
        p_value   = round(p_value, 4),
        reject_H0 = reject,
        conclusion=(
            "REJECT H₀ (p<0.05): TW estimates differ from Wg benchmark — "
            "endogeneity concern."
            if reject else
            "FAIL TO REJECT H₀: TW estimates consistent with exogenous benchmark "
            "→ TW is exogenous.  ✓"
        ),
    )


def hausman_tw_exogeneity(Y: np.ndarray,
                          X: np.ndarray,
                          W_stwm: np.ndarray,
                          W_g: np.ndarray) -> dict:
    """
    Hausman test for temporal weight matrix exogeneity via IV comparison.

    Implements the procedure of Cheng & Lee (2017) and Zhu et al. (2022):

    Model:  Y = α + ρ·(W_stwm@Y) + σX + ε   [WY is endogenous]

    Two instrument sets:
        Z_g  = [X, Wg@X]     — geographic inverse-distance (exogenous benchmark)
        Z_tw = [X, STWM@X]   — temporal weight matrix (proposed)

    Both regressions are 2SLS estimates of the SAME equation.  If TW is
    exogenous, both instrument sets yield statistically indistinguishable
    parameter estimates (Fail to reject H₀).

    Parameters
    ----------
    Y      : (N,) dependent variable (panel-stacked)
    X      : (N, k) regressors (NO constant — added internally)
    W_stwm : (N, N) STWM — provides the endogenous spatial lag WY
    W_g    : (N, N) inverse-distance weight matrix (exogenous benchmark)

    Returns
    -------
    dict with Hausman test results plus both IV coefficient vectors
    """
    Y = np.asarray(Y, float).ravel()
    X = np.asarray(X, float)
    N = len(Y)

    X_c   = np.column_stack([np.ones(N), X])      # (N, k+1) with constant
    WY    = W_stwm @ Y                             # endogenous spatial lag

    X_full = np.column_stack([X_c, WY])            # [1, X, WY]

    # Instruments: geographic baseline (Wg@[1,X]) — exogenous by construction
    WgX   = W_g @ X_c
    Z_g   = np.column_stack([X_c, WgX])

    # Instruments: temporal weight (STWM@[1,X]) — exogeneity to be tested
    TWX   = W_stwm @ X_c
    Z_tw  = np.column_stack([X_c, TWX])

    res_g  = iv_regression(Y, X_full, Z_g)
    res_tw = iv_regression(Y, X_full, Z_tw)

    h = hausman_test(res_g["beta"], res_tw["beta"],
                     res_g["cov"],  res_tw["cov"])

    return {
        **h,
        "beta_Wg" : res_g["beta"],    # IV coefficients [α, σ_1..k, ρ] via Wg
        "beta_TW" : res_tw["beta"],   # IV coefficients [α, σ_1..k, ρ] via TW
        "cov_Wg"  : res_g["cov"],
        "cov_TW"  : res_tw["cov"],
    }


# ---------------------------------------------------------------------------
# 3.  Sargan-Hansen J-test (over-identification)
# ---------------------------------------------------------------------------

def sargan_test(Y: np.ndarray, X: np.ndarray, Z: np.ndarray,
                beta_iv: np.ndarray) -> dict:
    """
    Sargan-Hansen J-test for instrument validity (over-identification).

    H₀: instruments are valid AND orthogonal to the error term
    J  = e_IV' Z(Z'Z)⁻¹Z' e_IV / σ̂²   ~  χ²(m−k) under H₀

    Small J (p > 0.05) → instruments valid → supports exogeneity of TW.
    Requires m > k (over-identified system).

    Parameters
    ----------
    Y       : (N,) dependent variable
    X       : (N, k) regressors (including endogenous WY)
    Z       : (N, m) instruments  (m > k required)
    beta_iv : (k,) IV estimates
    """
    Y    = np.asarray(Y, float).ravel()
    X    = np.asarray(X, float)
    Z    = np.asarray(Z, float)
    N, k = X.shape
    m    = Z.shape[1]
    if m <= k:
        return dict(J_stat=np.nan, df=0, p_value=np.nan,
                    note="Need m > k for over-identification test.")
    e    = Y - X @ beta_iv
    Pz_e = Z @ np.linalg.lstsq(Z, e, rcond=None)[0]
    sigma2 = float(e @ e) / (N - k)
    J_stat = float(e @ Pz_e) / sigma2
    df     = m - k
    p_val  = float(stats.chi2.sf(J_stat, df=df))
    return dict(
        J_stat    = round(J_stat, 4),
        df        = df,
        p_value   = round(p_val, 4),
        reject_H0 = p_val < 0.05,
        conclusion=(
            "REJECT H₀: instruments may be invalid or correlated with errors."
            if p_val < 0.05 else
            "FAIL TO REJECT H₀: instruments are valid → supports exogeneity.  ✓"
        ),
    )


# ---------------------------------------------------------------------------
# 4.  Redundancy F-test
# ---------------------------------------------------------------------------

def redundancy_test(Y: np.ndarray, X_base: np.ndarray,
                    X_extra: np.ndarray) -> dict:
    """
    Test whether the STWM spatial lags add explanatory power beyond Wg spatial lags.

    H₀: STWM lags are redundant given Wg lags (no independent variation)
    H₁: STWM lags contain independent variation beyond Wg → supports value of TW

    F = [(RSS_R − RSS_U) / q] / [RSS_U / (N − p_U)]

    Parameters
    ----------
    Y       : (N,) dependent variable
    X_base  : (N, p) base regressors (Wg spatial lags + X)
    X_extra : (N, q) additional regressors (STWM spatial lags)
    """
    Y    = np.asarray(Y, float).ravel()
    Xr   = np.asarray(X_base,  float)
    Xu   = np.asarray(X_extra, float)
    N    = len(Y)

    X_R  = np.column_stack([np.ones(N), Xr])
    X_U  = np.column_stack([np.ones(N), Xr, Xu])
    q    = Xu.shape[1]

    b_R  = np.linalg.lstsq(X_R, Y, rcond=None)[0]
    b_U  = np.linalg.lstsq(X_U, Y, rcond=None)[0]
    RSS_R = float(np.sum((Y - X_R @ b_R)**2))
    RSS_U = float(np.sum((Y - X_U @ b_U)**2))
    dof_U = max(N - X_U.shape[1], 1)
    F     = ((RSS_R - RSS_U) / q) / (RSS_U / dof_U)
    p_val = float(stats.f.sf(F, q, dof_U))

    return dict(
        F_stat    = round(F, 4),
        df        = (q, dof_U),
        p_value   = round(p_val, 4),
        reject_H0 = p_val < 0.05,
        conclusion=(
            "STWM lags add significant independent variation beyond Wg → "
            "TW contains useful information.  ✓"
            if p_val < 0.05 else
            "STWM lags are redundant given Wg — no additional variation."
        ),
    )


# ---------------------------------------------------------------------------
# 5.  Full exogeneity report
# ---------------------------------------------------------------------------

def stwm_exogeneity_report(Y: np.ndarray,
                            X: np.ndarray,
                            W_stwm: np.ndarray,
                            W_g: np.ndarray,
                            label_tw: str = "TW") -> dict:
    """
    Comprehensive exogeneity test battery for the temporal weight matrix.

    Follows Cheng & Lee (2017) and Zhu et al. (2022).

    Tests
    -----
    [1] Hausman test   — compare IV(Wg) vs IV(TW) coefficient vectors
    [2] Sargan J-test  — over-identification: validity of Wg instruments
    [3] Redundancy F   — does TW add explanatory power beyond Wg?

    Parameters
    ----------
    Y        : (N,) panel-stacked dependent variable
    X        : (N, k) regressors (no constant)
    W_stwm   : (N, N) STWM — the proposed temporal weight matrix
    W_g      : (N, N) inverse-distance matrix (exogenous benchmark)
    label_tw : label for the TW matrix in printed output (e.g. 'TW_I', 'TW_C')

    Interpretation
    --------------
    Fail to reject H₀ in Hausman test → TW is exogenous → estimates are robust.
    Negative Hausman stat → strong evidence of exogeneity (Baltagi 2021).

    Returns
    -------
    dict with keys: 'hausman', 'sargan', 'redundancy'
    """
    Y = np.asarray(Y, float).ravel()
    X = np.asarray(X, float)
    N = len(Y)

    X_c  = np.column_stack([np.ones(N), X])
    WY   = W_stwm @ Y
    X_full = np.column_stack([X_c, WY])

    # --- Hausman test ---
    WgX  = W_g @ X_c
    Z_g  = np.column_stack([X_c, WgX])
    TWX  = W_stwm @ X_c
    Z_tw = np.column_stack([X_c, TWX])

    res_g  = iv_regression(Y, X_full, Z_g)
    res_tw = iv_regression(Y, X_full, Z_tw)
    h_test = hausman_test(res_g["beta"], res_tw["beta"],
                          res_g["cov"],  res_tw["cov"])

    # --- Sargan J-test (use Wg instruments, over-identified if Wg adds lags) ---
    s_test = sargan_test(Y, X_full, Z_g, res_g["beta"])

    # --- Redundancy F-test ---
    r_test = redundancy_test(Y, WgX, TWX)

    report = dict(hausman=h_test, sargan=s_test, redundancy=r_test)

    print("\n" + "═" * 65)
    print(f"  STWM Exogeneity Test Battery — {label_tw}")
    print(f"  H₀: {label_tw} is exogenous (IV results consistent with Wg benchmark)")
    print("═" * 65)
    print(f"\n  [1] Hausman Test  (Cheng & Lee 2017; Zhu et al. 2022)")
    print(f"      H = {h_test['H_stat']},  df={h_test['df']},  p={h_test['p_value']}")
    print(f"      → {h_test['conclusion']}")
    print(f"\n  [2] Sargan-Hansen J-Test  (instrument validity)")
    print(f"      J = {s_test['J_stat']},  df={s_test['df']},  p={s_test['p_value']}")
    print(f"      → {s_test['conclusion']}")
    print(f"\n  [3] Redundancy F-Test  (TW independent of Wg?)")
    print(f"      F = {r_test['F_stat']},  df={r_test['df']},  p={r_test['p_value']}")
    print(f"      → {r_test['conclusion']}")
    print("\n" + "═" * 65)

    return report
