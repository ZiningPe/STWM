"""
simulation.py
-------------
Monte Carlo validation and Granger causality testing.

Monte Carlo Validation
----------------------
Assesses finite-sample properties of spatial model estimators under
a known data-generating process (DGP).

DGP for SAR:
    Y = (I − ρ W)⁻¹ (X β + ε),    ε ~ N(0, σ² I)

Metrics reported for each parameter θ ∈ {ρ, β}:

    Bias(θ̂) = E[θ̂] − θ_true  ≈  (1/S) Σ_s (θ̂_s − θ_true)

    RMSE(θ̂) = √E[(θ̂ − θ_true)²]  ≈  √[(1/S) Σ_s (θ̂_s − θ_true)²]

    Coverage = P(|θ̂_s − θ_true| < 1.96 · std(θ̂))

where S = number of Monte Carlo replications.

Granger Causality Test
----------------------
Tests whether the autocorrelation-derived TWM is justified by testing
if changes in Moran's I (or Geary's C) Granger-cause changes in the
empirically estimated spillover parameters.

Procedure (standard Granger F-test):

    Restricted model (lag p):
        y_t = α + Σ_{l=1}^{p} b_l y_{t-l} + u_t

    Unrestricted model:
        y_t = α + Σ_{l=1}^{p} b_l y_{t-l} + Σ_{l=1}^{p} c_l x_{t-l} + v_t

    F-statistic:
        F = [(RSS_R − RSS_U)/p] / [RSS_U / (T − 2p − 1)]
        F ~ F(p, T − 2p − 1) under H₀

    H₀: c_1 = c_2 = ... = c_p = 0  (x does NOT Granger-cause y)
    H₁: at least one c_l ≠ 0

References
----------
Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models.
    Econometrica, 37(3), 424-438.
LeSage, J., & Pace, R. K. (2009). Introduction to Spatial Econometrics. CRC Press.
"""

import numpy as np
from scipy import stats
from typing import Optional


# ---------------------------------------------------------------------------
# 1.  Monte Carlo Validation
# ---------------------------------------------------------------------------

def monte_carlo_stwm(true_rho: float,
                     true_beta: np.ndarray,
                     W_spatial: np.ndarray,
                     TWM: np.ndarray,
                     n: int,
                     T: int,
                     n_simulations: int = 500,
                     ModelClass=None,
                     seed: int = 42) -> dict:
    """
    Monte Carlo validation of STWM-based spatial model estimation.

    Data-generating process:
        Y = (I − ρ · STWM)⁻¹ (X β + ε),    ε ~ N(0, I)

    Parameters
    ----------
    true_rho      : true spatial-temporal autocorrelation parameter
    true_beta     : (k,) true regressor coefficients
    W_spatial     : (n, n) spatial weight matrix
    TWM           : (T, T) time weight matrix
    n, T          : number of spatial units and time periods
    n_simulations : number of Monte Carlo replications
    ModelClass    : model class to use; defaults to SpatialLagModel
    seed          : random seed for reproducibility

    Returns
    -------
    dict with:
        bias_rho, rmse_rho, coverage_rho  — scalar performance metrics for ρ
        bias_beta, rmse_beta              — (k,) performance metrics for β
        rho_estimates                     — (n_valid,) array of ρ̂ values
        beta_estimates                    — (S, k) array of β̂ values
    """
    from .stwm_core import build_stwm
    from .models import SpatialLagModel

    if ModelClass is None:
        ModelClass = SpatialLagModel

    rng      = np.random.default_rng(seed)
    STWM     = build_stwm(TWM, W_spatial)
    nT       = n * T
    k        = len(true_beta)
    true_beta = np.asarray(true_beta, dtype=float)

    # Pre-compute (I − ρW)⁻¹ once
    I_inv = np.linalg.inv(np.eye(nT) - true_rho * STWM)

    rho_est  = []
    beta_est = []

    for _ in range(n_simulations):
        X   = rng.standard_normal((nT, k))
        eps = rng.standard_normal(nT)
        Y   = I_inv @ (X @ true_beta + eps)
        try:
            model = ModelClass(STWM).fit(Y, X)
            res   = model.summary()
            rho_est.append(res.get("rho", np.nan))
            beta_est.append(res.get("beta", np.full(k, np.nan)))
        except Exception:
            rho_est.append(np.nan)
            beta_est.append(np.full(k, np.nan))

    rho_arr  = np.array(rho_est)
    beta_arr = np.array(beta_est)
    valid    = rho_arr[~np.isnan(rho_arr)]

    std_rho  = np.std(valid) if len(valid) > 1 else np.nan
    coverage = float(np.mean(np.abs(valid - true_rho) < 1.96 * std_rho)) \
               if not np.isnan(std_rho) else np.nan

    return {
        "n_simulations": n_simulations,
        "true_rho"     : true_rho,
        "true_beta"    : true_beta,
        "bias_rho"     : float(np.nanmean(rho_arr) - true_rho),
        "rmse_rho"     : float(np.sqrt(np.nanmean((rho_arr - true_rho) ** 2))),
        "coverage_rho" : round(coverage, 4) if not np.isnan(coverage) else np.nan,
        "bias_beta"    : np.nanmean(beta_arr, axis=0) - true_beta,
        "rmse_beta"    : np.sqrt(np.nanmean((beta_arr - true_beta) ** 2, axis=0)),
        "rho_estimates": valid,
        "beta_estimates": beta_arr,
    }


# ---------------------------------------------------------------------------
# 2.  Granger Causality Test
# ---------------------------------------------------------------------------

def granger_spillover_test(morans_sequence: np.ndarray,
                            spillover_sequence: np.ndarray,
                            max_lag: int = 3) -> dict:
    """
    Test whether Moran's I (or Geary's C) Granger-causes estimated spillover
    parameters (e.g. annual indirect effects or rho estimates).

    If Moran's I Granger-causes spillovers, this supports the use of
    autocorrelation statistics as the basis for the time weight matrix.

    Parameters
    ----------
    morans_sequence    : (T,) annual Moran's I (or Geary's C) values
    spillover_sequence : (T,) annual estimated spillover parameter
                         (e.g. indirect effects from rolling estimation,
                          or time-varying rho estimates)
    max_lag            : maximum lag order to test (default 3)

    Returns
    -------
    dict of {f'lag_{p}': {'F_statistic', 'p_value', 'reject_H0', 'conclusion'}}
    """
    y = np.asarray(spillover_sequence, dtype=float)
    x = np.asarray(morans_sequence,    dtype=float)
    T = len(y)
    results = {}

    for p in range(1, max_lag + 1):
        if T - p < p + 5:
            results[f"lag_{p}"] = {"error": "Insufficient observations for this lag."}
            continue

        Y_dep  = y[p:]
        Y_lags = np.column_stack([y[p-l-1:T-l-1] for l in range(p)])
        X_lags = np.column_stack([x[p-l-1:T-l-1] for l in range(p)])

        X_R = np.column_stack([np.ones(len(Y_dep)), Y_lags])
        X_U = np.column_stack([np.ones(len(Y_dep)), Y_lags, X_lags])

        beta_R = np.linalg.lstsq(X_R, Y_dep, rcond=None)[0]
        beta_U = np.linalg.lstsq(X_U, Y_dep, rcond=None)[0]

        RSS_R = float(np.sum((Y_dep - X_R @ beta_R) ** 2))
        RSS_U = float(np.sum((Y_dep - X_U @ beta_U) ** 2))

        n_obs = len(Y_dep)
        k_U   = X_U.shape[1]
        F_stat = ((RSS_R - RSS_U) / p) / (RSS_U / max(n_obs - k_U, 1))
        p_val  = float(stats.f.sf(F_stat, p, n_obs - k_U))

        results[f"lag_{p}"] = {
            "F_statistic": round(F_stat, 4),
            "p_value"    : round(p_val, 4),
            "reject_H0"  : p_val < 0.05,
            "conclusion" : (
                f"Lag {p}: Moran's I Granger-causes spillover parameters "
                f"(F={F_stat:.3f}, p={p_val:.4f}). TWM construction is supported."
                if p_val < 0.05 else
                f"Lag {p}: No Granger causality from Moran's I "
                f"(F={F_stat:.3f}, p={p_val:.4f})."
            ),
        }

    return results
