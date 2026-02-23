"""
endogeneity.py
--------------
Instrumental Variable (IV) regression and Hausman specification test.

Purpose
-------
When the spatial lag WY is included as a regressor, it is endogenous
(correlated with the error term) because Y and WY are jointly determined.
This module provides 2SLS estimation and the Hausman test to formally
assess whether endogeneity affects the results.

Mathematical Formulation
------------------------

Two-Stage Least Squares (2SLS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stage 1 — First stage (instrument relevance):
    X_endog = Z π + v

Stage 2 — Second stage (consistent estimation):
    Y = X̂ β + u

    where X̂ = P_Z X = Z (Z'Z)⁻¹ Z' X   (projection onto instrument space)

Estimator:
    β̂_IV = (X̂'X)⁻¹ X̂'Y

Covariance:
    Var(β̂_IV) = σ² (X̂'X̂)⁻¹,    σ² = (Y − Xβ̂_IV)'(Y − Xβ̂_IV) / (n−k)

Instrument validity requires:
  (a) Relevance:  Corr(Z, X_endog) ≠ 0   (tested via first-stage F > 10)
  (b) Exogeneity: Corr(Z, ε) = 0         (assumed / theory-based)

Hausman Specification Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compare OLS (efficient under H₀) with IV (consistent under H₁).

    H₀: OLS is consistent  (no endogeneity)
    H₁: OLS is inconsistent (endogeneity present; IV preferred)

    Test statistic:
        H = (β̂_IV − β̂_OLS)' [Var(β̂_IV) − Var(β̂_OLS)]⁺ (β̂_IV − β̂_OLS)
        H ~ χ²(k)   under H₀

    where [·]⁺ denotes the Moore-Penrose pseudo-inverse.

References
----------
Hausman, J. A. (1978). Specification Tests in Econometrics. Econometrica, 46(6), 1251-1271.
Davidson, R., & MacKinnon, J. G. (1993). Estimation and Inference in Econometrics. Oxford.
"""

import numpy as np
from scipy import stats
import warnings
from typing import Dict, List


# ---------------------------------------------------------------------------
# 1.  2SLS IV Regression
# ---------------------------------------------------------------------------

def iv_regression(Y: np.ndarray,
                  X: np.ndarray,
                  Z: np.ndarray) -> dict:
    """
    Two-Stage Least Squares (2SLS) IV regression.

    Parameters
    ----------
    Y : (nT,)   dependent variable
    X : (nT, k) regressors (may include endogenous spatial lag WY)
    Z : (nT, m) instrument matrix  (m ≥ k required)

    Returns
    -------
    dict with keys:
        beta          : (k,) IV coefficient estimates
        se            : (k,) standard errors
        t_stats       : (k,) t-statistics
        p_values      : (k,) two-sided p-values
        residuals     : (nT,) residuals
        first_stage_F : float, first-stage F-statistic (instrument strength)
        sigma2        : float, residual variance
    """
    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n, k = X.shape
    _, m = Z.shape

    if m < k:
        raise ValueError(f"Need m≥k instruments, got m={m} < k={k}.")

    # Stage 1: project X onto instrument space
    X_hat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]

    # Stage 2: OLS of Y on X_hat
    XhXh  = X_hat.T @ X_hat
    beta  = np.linalg.solve(XhXh, X_hat.T @ Y)
    resid = Y - X @ beta
    sigma2 = float(resid @ resid) / (n - k)
    cov   = sigma2 * np.linalg.pinv(XhXh)
    se    = np.sqrt(np.diag(cov))
    t_s   = beta / se
    pvals = 2 * stats.t.sf(np.abs(t_s), df=n - k)

    return {
        "beta"         : beta,
        "se"           : se,
        "t_stats"      : t_s,
        "p_values"     : pvals,
        "residuals"    : resid,
        "first_stage_F": _first_stage_f(X, Z),
        "sigma2"       : sigma2,
    }


# ---------------------------------------------------------------------------
# 2.  Hausman Test
# ---------------------------------------------------------------------------

def hausman_test(beta_ols: np.ndarray,
                 beta_iv: np.ndarray,
                 cov_ols: np.ndarray,
                 cov_iv: np.ndarray) -> dict:
    """
    Hausman specification test for endogeneity.

    H₀: OLS is consistent (no endogeneity)
    H₁: OLS is inconsistent; IV estimates preferred

    Parameters
    ----------
    beta_ols : (k,) OLS coefficient vector
    beta_iv  : (k,) IV  coefficient vector
    cov_ols  : (k,k) OLS covariance matrix
    cov_iv   : (k,k) IV  covariance matrix

    Returns
    -------
    dict with 'H_stat', 'df', 'p_value', 'reject_H0', 'conclusion'
    """
    diff     = beta_iv - beta_ols
    cov_diff = cov_iv - cov_ols
    cov_inv  = np.linalg.pinv(cov_diff)
    H_stat   = float(diff @ cov_inv @ diff)
    df       = len(diff)
    p_value  = float(stats.chi2.sf(H_stat, df=df))
    reject   = p_value < 0.05

    return {
        "H_stat"    : round(H_stat, 4),
        "df"        : df,
        "p_value"   : round(p_value, 4),
        "reject_H0" : reject,
        "conclusion": (
            "Reject H₀: endogeneity detected — IV estimates are preferred."
            if reject else
            "Fail to reject H₀: no significant endogeneity — OLS is consistent."
        ),
    }


# ---------------------------------------------------------------------------
# 3.  Multi-model endogeneity report
# ---------------------------------------------------------------------------

def endogeneity_report(Y: np.ndarray,
                       X_ols: Dict[str, np.ndarray],
                       X_iv: Dict[str, np.ndarray],
                       Z: np.ndarray,
                       model_names: List[str] = ("SAR", "SEM", "SDM")) -> dict:
    """
    Run IV + Hausman test across multiple spatial model specifications.

    Produces a consolidated table of endogeneity test results for all
    specified models, analogous to a robustness table in a paper.

    Parameters
    ----------
    Y          : (nT,) dependent variable
    X_ols      : {model_name: (nT, k) OLS design matrix}
    X_iv       : {model_name: (nT, k) IV  design matrix (includes WY)}
    Z          : (nT, m) instrument matrix
    model_names: list of model labels to test

    Returns
    -------
    dict of {model_name: {'hausman': hausman_result, 'iv': iv_result}}
    """
    report = {}
    for name in model_names:
        if name not in X_ols or name not in X_iv:
            continue

        ols_res = _ols_fit(Y, X_ols[name])
        iv_res  = iv_regression(Y, X_iv[name], Z)

        n, k   = X_ols[name].shape
        s2_ols = float(ols_res["residuals"] @ ols_res["residuals"]) / (n - k)
        cov_ols = s2_ols * np.linalg.pinv(X_ols[name].T @ X_ols[name])

        Z_arr  = np.asarray(Z, dtype=float)
        X_hat  = Z_arr @ np.linalg.lstsq(Z_arr, X_iv[name], rcond=None)[0]
        cov_iv = iv_res["sigma2"] * np.linalg.pinv(X_hat.T @ X_hat)

        report[name] = {
            "hausman": hausman_test(ols_res["beta"], iv_res["beta"], cov_ols, cov_iv),
            "iv"     : iv_res,
        }
    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ols_fit(Y, X):
    Y, X = np.asarray(Y, float).ravel(), np.asarray(X, float)
    n, k = X.shape
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ beta
    return {"beta": beta, "residuals": resid}


def _first_stage_f(X: np.ndarray, Z: np.ndarray) -> float:
    """Simplified first-stage F (joint instrument relevance)."""
    n, k = X.shape
    _, m = Z.shape
    Pz   = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    Mz_X = X - Pz
    RSS_r = float((Mz_X ** 2).sum())
    X_dm  = X - X.mean(axis=0)
    RSS_u = float((X_dm ** 2).sum())
    if RSS_u < 1e-12:
        return np.nan
    return round(((RSS_u - RSS_r) / m) / (RSS_r / max(n - m - 1, 1)), 4)
