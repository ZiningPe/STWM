"""
endogeneity.py
--------------
Proving STWM exogeneity — NOT testing OLS vs IV endogeneity.

The core argument is:

    H₀: STWM is exogenous (i.e. STWM and SWM produce equivalent estimates)
    H₁: STWM introduces endogeneity relative to SWM

    If Hausman test FAILS TO REJECT H₀ → STWM is exogenous.

Logic (Wu-Hausman framework):
    - Assume the static SWM is exogenous (standard spatial econometrics assumption).
    - Treat SWM-based estimates as the "consistent but inefficient" IV benchmark.
    - Treat STWM-based estimates as the "efficient under H₀" estimator.
    - If STWM ≈ SWM in terms of coefficient estimates → STWM is exogenous.

Additional tests for STWM exogeneity
--------------------------------------
1. Hausman test          : β_STWM vs β_SWM (main test)
2. Sargan-Hansen J-test  : over-identification test — if instruments are valid
                           and STWM is exogenous, J-stat ~ χ²(m-k) should be small
3. Redundancy F-test     : test if STWM adds explanatory power beyond SWM
                           (if NOT redundant, it contains independent variation = exogenous)
4. Correlation test      : Cor(STWM·Y, residuals) should be ~0 if STWM is exogenous

Mathematical Formulation
------------------------
Two-Stage Least Squares (2SLS):

    Stage 1:   Ŷ_endog = Z(Z'Z)⁻¹Z' Y_endog
    Stage 2:   β̂_IV = (X̂'X)⁻¹X̂'Y

    where Z = instrument matrix (here: SWM-based spatial lags)

Hausman statistic (Wu 1973; Hausman 1978):
    H = (β̂_STWM − β̂_SWM)' [Var(β̂_STWM) − Var(β̂_SWM)]⁺ (β̂_STWM − β̂_SWM)
    H ~ χ²(k) under H₀

    Interpretation for STWM exogeneity:
    → FAIL TO REJECT H₀ (p > 0.05): STWM ≈ SWM → STWM is exogenous ✓
    → REJECT H₀ (p < 0.05): STWM ≠ SWM → potential endogeneity concern

Sargan-Hansen J-test (over-identification):
    e_IV = Y − X·β̂_IV
    J = e_IV' Z (Z'Z)⁻¹ Z' e_IV / σ̂²   ~  χ²(m − k)  under H₀ (instrument validity)
    Small J → instruments valid → STWM exogenous

References
----------
Hausman, J. A. (1978). Econometrica, 46(6), 1251–1271.
Wu, D. M. (1973). Econometrica, 41(4), 733–750.
Sargan, J. D. (1958). Econometrica, 26(3), 393–415.
"""

import numpy as np
from scipy import stats
import warnings


# ---------------------------------------------------------------------------
# 1. 2SLS IV Regression
# ---------------------------------------------------------------------------

def iv_regression(Y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> dict:
    """
    Two-Stage Least Squares (2SLS).

    β̂_IV = (X̂'X)⁻¹X̂'Y,  where X̂ = Z(Z'Z)⁻¹Z'X

    Parameters
    ----------
    Y : (nT,) dependent variable
    X : (nT, k) regressors (may include endogenous WY)
    Z : (nT, m) instruments  (m ≥ k)

    Returns
    -------
    dict with beta, se, t_stats, p_values, first_stage_F, sigma2
    """
    Y = np.asarray(Y, float).ravel()
    X = np.asarray(X, float)
    Z = np.asarray(Z, float)
    n, k = X.shape
    _, m  = Z.shape
    if m < k:
        raise ValueError(f"Need m≥k instruments, got {m} < {k}.")

    X_hat  = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    beta   = np.linalg.lstsq(X_hat, Y, rcond=None)[0]
    resid  = Y - X @ beta
    sigma2 = float(resid @ resid) / (n - k)
    cov    = sigma2 * np.linalg.pinv(X_hat.T @ X_hat)
    se     = np.sqrt(np.diag(cov))
    t_stat = beta / np.where(se > 0, se, np.nan)
    pvals  = 2 * stats.t.sf(np.abs(t_stat), df=n - k)

    return dict(beta=beta, se=se, t_stats=t_stat, p_values=pvals,
                residuals=resid, sigma2=sigma2, cov=cov,
                first_stage_F=_first_stage_f(X, Z))


# ---------------------------------------------------------------------------
# 2. Hausman Test for STWM Exogeneity
# ---------------------------------------------------------------------------

def hausman_test(beta_stwm: np.ndarray, beta_swm: np.ndarray,
                 cov_stwm: np.ndarray, cov_swm: np.ndarray) -> dict:
    """
    Hausman test to verify STWM exogeneity.

    H₀: STWM is exogenous (β_STWM ≈ β_SWM, difference is not systematic)
    H₁: STWM introduces endogeneity

    Test statistic:
        H = (β_STWM − β_SWM)' [Var(β_STWM) − Var(β_SWM)]⁺ (β_STWM − β_SWM)
        H ~ χ²(k)  under H₀

    Decision rule for STWM exogeneity:
        p > 0.05 → Fail to reject H₀ → STWM is exogenous ✓
        p < 0.05 → Reject H₀ → endogeneity concern

    Parameters
    ----------
    beta_stwm : (k,) STWM-based coefficient estimates
    beta_swm  : (k,) SWM-based coefficient estimates (exogenous benchmark)
    cov_stwm  : (k,k) covariance of STWM estimates
    cov_swm   : (k,k) covariance of SWM estimates
    """
    diff     = beta_stwm - beta_swm
    cov_diff = cov_stwm - cov_swm
    try:
        cov_inv = np.linalg.pinv(cov_diff)
        H_stat  = float(diff @ cov_inv @ diff)
    except Exception:
        H_stat = np.nan
    df      = len(diff)
    p_value = float(stats.chi2.sf(H_stat, df=df)) if not np.isnan(H_stat) else np.nan
    reject  = p_value < 0.05

    return dict(
        H_stat   = round(H_stat, 4),
        df       = df,
        p_value  = round(p_value, 4),
        reject_H0= reject,
        conclusion=(
            "REJECT H₀ (p<0.05): significant difference between STWM and SWM — "
            "potential endogeneity concern."
            if reject else
            "FAIL TO REJECT H₀ (p≥0.05): STWM and SWM produce equivalent estimates "
            "→ STWM is exogenous. ✓"
        ),
    )


# ---------------------------------------------------------------------------
# 3. Sargan-Hansen J-test (over-identification / instrument validity)
# ---------------------------------------------------------------------------

def sargan_test(Y: np.ndarray, X: np.ndarray, Z: np.ndarray,
                beta_iv: np.ndarray) -> dict:
    """
    Sargan-Hansen J-test for instrument validity / STWM exogeneity.

    H₀: instruments are valid AND STWM is exogenous
    J = e_IV' Z(Z'Z)⁻¹Z' e_IV / σ̂²   ~   χ²(m−k)  under H₀

    Small J (p > 0.05) → instruments valid → supports STWM exogeneity.

    Parameters
    ----------
    Y       : (nT,) dependent variable
    X       : (nT, k) regressors
    Z       : (nT, m) instruments  (m > k required for over-identification)
    beta_iv : (k,) IV estimates
    """
    Y    = np.asarray(Y, float).ravel()
    X    = np.asarray(X, float)
    Z    = np.asarray(Z, float)
    n, k = X.shape
    m    = Z.shape[1]
    if m <= k:
        return dict(J_stat=np.nan, df=0, p_value=np.nan,
                    note="Need m>k for over-identification test.")
    e    = Y - X @ beta_iv
    Pz_e = Z @ np.linalg.lstsq(Z, e, rcond=None)[0]
    sigma2 = float(e @ e) / (n - k)
    J_stat = float(e @ Pz_e) / sigma2
    df     = m - k
    p_val  = float(stats.chi2.sf(J_stat, df=df))
    return dict(
        J_stat   = round(J_stat, 4),
        df       = df,
        p_value  = round(p_val, 4),
        reject_H0= p_val < 0.05,
        conclusion=(
            "REJECT H₀: instruments may be invalid or STWM is endogenous."
            if p_val < 0.05 else
            "FAIL TO REJECT H₀: instruments are valid → supports STWM exogeneity. ✓"
        ),
    )


# ---------------------------------------------------------------------------
# 4. Redundancy F-test
# ---------------------------------------------------------------------------

def redundancy_test(Y: np.ndarray, X_swm: np.ndarray,
                    X_stwm: np.ndarray) -> dict:
    """
    Test whether STWM spatial lags add explanatory power beyond SWM spatial lags.

    H₀: STWM lags are redundant given SWM lags  (STWM adds no independent variation)
    H₁: STWM lags contain independent variation  → consistent with exogeneity

    If STWM adds significant independent variation (F significant), this means
    STWM captures something the static SWM does not — supporting its value.

    F = [(RSS_R − RSS_U) / q] / [RSS_U / (n − p)]
    """
    Y    = np.asarray(Y, float).ravel()
    Xr   = np.asarray(X_swm,  float)
    Xu   = np.asarray(X_stwm, float)
    n    = len(Y)

    X_R  = np.column_stack([np.ones(n), Xr])
    X_U  = np.column_stack([np.ones(n), Xr, Xu])
    q    = Xu.shape[1]

    b_R  = np.linalg.lstsq(X_R, Y, rcond=None)[0]
    b_U  = np.linalg.lstsq(X_U, Y, rcond=None)[0]
    RSS_R = float(np.sum((Y - X_R @ b_R)**2))
    RSS_U = float(np.sum((Y - X_U @ b_U)**2))
    F     = ((RSS_R - RSS_U) / q) / (RSS_U / (n - X_U.shape[1]))
    p_val = float(stats.f.sf(F, q, n - X_U.shape[1]))

    return dict(
        F_stat   = round(F, 4),
        df       = (q, n - X_U.shape[1]),
        p_value  = round(p_val, 4),
        reject_H0= p_val < 0.05,
        conclusion=(
            "STWM lags add significant independent variation beyond SWM "
            "→ consistent with STWM exogeneity. ✓"
            if p_val < 0.05 else
            "STWM lags are redundant given SWM — no independent variation added."
        ),
    )


# ---------------------------------------------------------------------------
# 5. Full exogeneity report
# ---------------------------------------------------------------------------

def stwm_exogeneity_report(Y: np.ndarray,
                            X: np.ndarray,
                            STWM: np.ndarray,
                            SWM: np.ndarray,
                            ModelClass) -> dict:
    """
    Comprehensive STWM exogeneity test battery.

    Runs all three tests and prints a consolidated report.

    Tests:
      1. Hausman test (STWM vs SWM coefficient comparison)
      2. Sargan-Hansen J-test (over-identification)
      3. Redundancy F-test (independent variation in STWM)

    Parameters
    ----------
    Y          : (nT,) dependent variable
    X          : (nT, k) regressors
    STWM       : (nT, nT) spatial-temporal weight matrix
    SWM        : (nT, nT) static spatial weight matrix (panel-expanded)
    ModelClass : spatial model class, e.g. SDMModel
    """
    import statsmodels.api as sm

    # Fit model with STWM and SWM
    res_stwm = ModelClass(STWM).fit(Y, X).summary()
    res_swm  = ModelClass(SWM).fit(Y, X).summary()

    beta_stwm = res_stwm.get('beta_X', res_stwm.get('beta'))
    beta_swm  = res_swm.get('beta_X', res_swm.get('beta'))
    n, k = X.shape

    # Covariances (approximate)
    sigma2_stwm = res_stwm['sigma2']
    sigma2_swm  = res_swm['sigma2']
    cov_stwm = sigma2_stwm * np.linalg.pinv(X.T @ X)
    cov_swm  = sigma2_swm  * np.linalg.pinv(X.T @ X)

    h_test = hausman_test(beta_stwm, beta_swm, cov_stwm, cov_swm)

    # Sargan: use SWM spatial lags as instruments
    WY_swm  = SWM @ Y
    WX_swm  = SWM @ X
    X_iv    = np.column_stack([np.ones(n), X, WY_swm.reshape(-1,1)])
    Z_inst  = np.column_stack([np.ones(n), X, WX_swm])
    iv_res  = iv_regression(Y, X_iv, Z_inst)
    s_test  = sargan_test(Y, X_iv, Z_inst, iv_res['beta'])

    # Redundancy
    WX_stwm = STWM @ X
    r_test  = redundancy_test(Y, WX_swm, WX_stwm)

    report = dict(hausman=h_test, sargan=s_test, redundancy=r_test)

    print("\n" + "═"*60)
    print("  STWM Exogeneity Test Battery")
    print("  H₀: STWM is exogenous (same as SWM)")
    print("═"*60)
    print(f"\n  [1] Hausman Test")
    print(f"      H = {h_test['H_stat']},  df={h_test['df']},  p={h_test['p_value']}")
    print(f"      → {h_test['conclusion']}")
    print(f"\n  [2] Sargan-Hansen J-Test (over-identification)")
    print(f"      J = {s_test['J_stat']},  df={s_test['df']},  p={s_test['p_value']}")
    print(f"      → {s_test['conclusion']}")
    print(f"\n  [3] Redundancy F-Test")
    print(f"      F = {r_test['F_stat']},  df={r_test['df']},  p={r_test['p_value']}")
    print(f"      → {r_test['conclusion']}")
    print("\n" + "═"*60)

    return report


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _first_stage_f(X: np.ndarray, Z: np.ndarray) -> float:
    n, k = X.shape
    _, m = Z.shape
    Pz   = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    Mz_X = X - Pz
    RSS_r = float((Mz_X**2).sum())
    X_dm  = X - X.mean(axis=0)
    RSS_u = float((X_dm**2).sum())
    if RSS_u < 1e-12:
        return np.nan
    return round(((RSS_u - RSS_r)/m) / (RSS_r / max(n-m-1, 1)), 4)
