"""
heterogeneity.py
----------------
Heterogeneity analysis and heteroskedasticity testing for spatial panel models.

Modules
-------
1. Heteroskedasticity tests (Breusch-Pagan, White)
2. Regional subgroup estimation — fit model separately per geographic group
3. Temporal subgroup estimation — fit model for different time sub-periods

Mathematical Formulation
------------------------

Breusch-Pagan Test
~~~~~~~~~~~~~~~~~~
Tests H₀: Var(ε_i) = σ² (homoskedasticity) against heteroskedasticity.

    Step 1: Obtain OLS residuals ê = Y − Xβ̂
    Step 2: Regress ê² on X:  ê² = X γ + v
    Step 3: Test statistic:   LM = n · R²  ~  χ²(k−1) under H₀

White Test
~~~~~~~~~~
Non-parametric version allowing for arbitrary heteroskedasticity.

    Regress ê² on X, X² (element-wise squares), and cross-products of X.
    Test statistic:  LM = n · R²  ~  χ²(p−1) under H₀
    where p = number of regressors in augmented White regression.

References
----------
Breusch, T. S., & Pagan, A. R. (1979). A Simple Test for Heteroscedasticity.
    Econometrica, 47(5), 1287-1294.
White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator.
    Econometrica, 48(4), 817-838.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 1.  Heteroskedasticity Tests
# ---------------------------------------------------------------------------

def heteroskedasticity_test(Y: np.ndarray,
                             X: np.ndarray,
                             test: str = "breusch_pagan") -> dict:
    """
    Test for heteroskedasticity in model residuals.

    H₀: Var(ε_i) = σ²  (homoskedasticity)
    H₁: Var(ε_i) = f(X_i)  (heteroskedasticity)

    Parameters
    ----------
    Y    : (nT,) dependent variable
    X    : (nT, k) regressors (include a constant column)
    test : 'breusch_pagan' — standard LM test
           'white'         — non-parametric White test

    Returns
    -------
    dict with 'test', 'statistic', 'df', 'p_value', 'reject_H0', 'conclusion'
    """
    from .models import _ols_fit
    Y    = np.asarray(Y, dtype=float).ravel()
    X    = np.asarray(X, dtype=float)
    res  = _ols_fit(Y, X)
    e2   = res["residuals"] ** 2
    n    = len(Y)

    if test == "breusch_pagan":
        res2 = _ols_fit(e2, X)
        R2   = res2["R2"]
        stat = n * R2
        df   = X.shape[1] - 1

    elif test == "white":
        X_w  = _white_regressors(X)
        res2 = _ols_fit(e2, X_w)
        R2   = res2["R2"]
        stat = n * R2
        df   = X_w.shape[1] - 1

    else:
        raise ValueError(f"Unknown test: '{test}'. Choose 'breusch_pagan' or 'white'.")

    pval = float(stats.chi2.sf(stat, df=df))
    return {
        "test"      : test,
        "statistic" : round(float(stat), 4),
        "df"        : df,
        "p_value"   : round(pval, 4),
        "reject_H0" : pval < 0.05,
        "conclusion": (
            "Heteroskedasticity detected — consider robust standard errors or GLS."
            if pval < 0.05 else
            "No significant heteroskedasticity detected at 5% level."
        ),
    }


def _white_regressors(X: np.ndarray) -> np.ndarray:
    """Construct White test augmented regressor matrix (X, X², cross-products)."""
    n, k  = X.shape
    parts = [X]
    for i in range(k):
        parts.append((X[:, i] ** 2).reshape(-1, 1))
    for i in range(k):
        for j in range(i + 1, k):
            parts.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.column_stack(parts)


# ---------------------------------------------------------------------------
# 2.  Regional Subgroup Estimation
# ---------------------------------------------------------------------------

def regional_subgroup(Y: np.ndarray,
                      X: np.ndarray,
                      W: np.ndarray,
                      region_labels: np.ndarray,
                      ModelClass,
                      T: int) -> Dict[str, dict]:
    """
    Estimate a spatial model separately for each region (geographic subgroup).

    Useful for testing whether the spatial spillover mechanism differs
    across regions — e.g. metropolitan vs. rural areas, or different
    country sub-regions.

    Parameters
    ----------
    Y             : (nT,) stacked dependent variable (time-major order)
    X             : (nT, k) stacked regressors
    W             : (n, n) spatial weight matrix
    region_labels : (n,) array assigning each spatial unit to a group
                    (integer or string labels)
    ModelClass    : model class, e.g. SDMModel
    T             : number of time periods

    Returns
    -------
    dict of {region_label: model.summary()}
    """
    n       = len(region_labels)
    regions = np.unique(region_labels)
    results = {}

    for reg in regions:
        idx = np.where(region_labels == reg)[0]
        panel_idx = np.sort(np.concatenate([idx + t * n for t in range(T)]))

        Y_sub = Y[panel_idx]
        X_sub = X[panel_idx]
        W_sub = W[np.ix_(idx, idx)]

        # Re-row-standardise sub-matrix
        rs    = W_sub.sum(axis=1, keepdims=True)
        rs    = np.where(rs == 0, 1.0, rs)
        W_sub = W_sub / rs

        try:
            model = ModelClass(W_sub).fit(Y_sub, X_sub)
            results[str(reg)] = model.summary()
        except Exception as e:
            results[str(reg)] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# 3.  Temporal Subgroup Estimation
# ---------------------------------------------------------------------------

def temporal_subgroup(Y: np.ndarray,
                      X: np.ndarray,
                      W: np.ndarray,
                      n: int,
                      T: int,
                      periods: List[Tuple[int, int]],
                      ModelClass) -> Dict[str, dict]:
    """
    Estimate a spatial model for specified time sub-periods.

    Use this to test whether spillover dynamics shift over different
    phases of the study period (e.g., pre/post policy change, economic cycles).

    Parameters
    ----------
    Y       : (nT,) stacked dependent variable
    X       : (nT, k) stacked regressors
    W       : (n, n) spatial weight matrix
    n       : number of spatial units
    T       : total number of time periods (unused but kept for clarity)
    periods : list of (t_start, t_end) tuples (0-indexed, inclusive)
              e.g. [(0, 4), (5, 9), (10, 13)]
    ModelClass : spatial model class

    Returns
    -------
    dict of {"t_start-t_end": model.summary()}
    """
    results = {}
    for (t0, t1) in periods:
        panel_idx = np.concatenate([np.arange(n) + t * n for t in range(t0, t1 + 1)])
        Y_sub = Y[panel_idx]
        X_sub = X[panel_idx]
        try:
            model = ModelClass(W).fit(Y_sub, X_sub)
            results[f"{t0}-{t1}"] = model.summary()
        except Exception as e:
            results[f"{t0}-{t1}"] = {"error": str(e)}
    return results
