"""
robustness.py
-------------
Robustness checks for STWM-based spatial models.

Three complementary strategies:

1. compare_weight_matrices
   Fit the same model with different weight matrices (STWM variants,
   decay-based alternatives, static W) and compare AIC / BIC / LogLik.
   This directly tests whether the proposed TWM outperforms alternatives
   on the same dataset.

2. rolling_window_estimation
   Re-estimate the model on sliding time windows, producing a sequence
   of parameter estimates. Useful for detecting structural breaks or
   time-varying spillover dynamics.

3. sensitivity_report
   Vary the TWM construction hyperparameters (winsorisation quantile,
   min_abs truncation) and check whether results remain stable.
   This guards against the risk that the proposed method is sensitive
   to arbitrary tuning choices.

Model Selection Criteria
------------------------
For a model with k parameters, n observations, log-likelihood L:

    AIC = −2L + 2k
    BIC = −2L + k · ln(n)

Lower AIC / BIC indicates better fit penalised by complexity.

References
----------
Akaike, H. (1974). A new look at the statistical model identification. IEEE TAC.
Schwarz, G. (1978). Estimating the dimension of a model. Annals of Statistics.
"""

import numpy as np
from typing import Dict, List, Callable
import warnings


# ---------------------------------------------------------------------------
# 1.  Compare weight matrices by model fit
# ---------------------------------------------------------------------------

def compare_weight_matrices(Y: np.ndarray,
                             X: np.ndarray,
                             weight_matrices: Dict[str, np.ndarray],
                             ModelClass,
                             metrics: List[str] = ("AIC", "BIC", "LogLik")) -> dict:
    """
    Fit the same spatial model with different weight matrices and compare fit.

    Parameters
    ----------
    Y               : (nT,) dependent variable
    X               : (nT, k) regressors
    weight_matrices : {label: W_matrix}   e.g.
                        {
                          'STWM_Morans'  : stwm_morans,
                          'STWM_GearyC'  : stwm_gearyc,
                          'Decay_Exp_0.3': stwm_decay_exp,
                          'Decay_Lin_0.2': stwm_decay_lin,
                          'Static_W'     : static_w,
                        }
    ModelClass      : spatial model class (e.g. SDMModel)
    metrics         : which fit metrics to include in the comparison table

    Returns
    -------
    dict with per-matrix full results and a 'comparison_table' sub-dict.
    """
    n       = len(Y)
    results = {}

    for label, W in weight_matrices.items():
        try:
            model    = ModelClass(W).fit(Y, X)
            res      = model.summary()
            sigma2   = res.get("sigma2", np.nan)
            k_params = X.shape[1] + 1   # +1 for intercept

            if not np.isnan(sigma2) and sigma2 > 0:
                log_lik = float(-n / 2 * (np.log(2 * np.pi * sigma2) + 1))
            else:
                log_lik = np.nan

            aic = (-2 * log_lik + 2 * k_params) if not np.isnan(log_lik) else np.nan
            bic = (-2 * log_lik + k_params * np.log(n)) if not np.isnan(log_lik) else np.nan

            results[label] = {
                "summary": res,
                "LogLik" : round(log_lik, 4) if not np.isnan(log_lik) else np.nan,
                "AIC"    : round(aic,     4) if not np.isnan(aic)     else np.nan,
                "BIC"    : round(bic,     4) if not np.isnan(bic)     else np.nan,
            }
        except Exception as e:
            results[label] = {"error": str(e)}

    # Build comparison table
    table = {m: {} for m in metrics}
    for label, res in results.items():
        for m in metrics:
            table[m][label] = "ERROR" if "error" in res else res.get(m, np.nan)

    results["comparison_table"] = table
    return results


# ---------------------------------------------------------------------------
# 2.  Rolling window estimation
# ---------------------------------------------------------------------------

def rolling_window_estimation(Y: np.ndarray,
                               X: np.ndarray,
                               W_spatial: np.ndarray,
                               morans_sequence: np.ndarray,
                               n: int,
                               T: int,
                               window: int,
                               ModelClass,
                               twm_builder: Callable) -> dict:
    """
    Re-estimate the model on a rolling time window.

    At each window position [t, t+window), the STWM is re-constructed
    from the Moran's I sub-sequence and the model is re-estimated.

    Parameters
    ----------
    Y               : (nT,) stacked dependent variable
    X               : (nT, k) stacked regressors
    W_spatial       : (n, n) spatial weight matrix
    morans_sequence : (T,) annual Moran's I series
    n, T            : number of spatial units and total time periods
    window          : rolling window size (number of years)
    ModelClass      : spatial model class
    twm_builder     : callable(morans_slice: ndarray) → TWM
                      e.g. lambda m: build_twm_morans(m)

    Returns
    -------
    dict of {'window_t0-t1': model.summary()}
    """
    from .stwm_core import build_stwm
    results = {}

    for t0 in range(T - window + 1):
        t1 = t0 + window
        panel_idx = np.sort(np.concatenate(
            [np.arange(n) + t * n for t in range(t0, t1)]
        ))
        Y_sub   = Y[panel_idx]
        X_sub   = X[panel_idx]
        m_slice = morans_sequence[t0:t1]
        TWM_w   = twm_builder(m_slice)
        STWM_w  = build_stwm(TWM_w, W_spatial)

        try:
            model = ModelClass(STWM_w).fit(Y_sub, X_sub)
            results[f"window_{t0}-{t1-1}"] = model.summary()
        except Exception as e:
            results[f"window_{t0}-{t1-1}"] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# 3.  Sensitivity to TWM construction parameters
# ---------------------------------------------------------------------------

def sensitivity_report(Y: np.ndarray,
                        X: np.ndarray,
                        W_spatial: np.ndarray,
                        morans_sequence: np.ndarray,
                        ModelClass,
                        winsorize_quantiles: List[float] = (0.90, 0.95, 0.99),
                        min_abs_values: List[float] = (1e-4, 1e-3, 1e-2)) -> dict:
    """
    Sensitivity analysis: vary TWM hyperparameters and check result stability.

    Tests whether key effect estimates (direct, indirect, total, rho) change
    materially when winsorisation threshold or the min_abs truncation value
    is varied. Results should be stable across a reasonable parameter range.

    Parameters
    ----------
    winsorize_quantiles : list of upper quantile caps to test
    min_abs_values      : list of minimum denominator thresholds to test

    Returns
    -------
    Nested dict: {f'q{q}': {f'ma{ma}': {'direct': ..., 'indirect': ..., ...}}}
    """
    from .temporal_weights import build_twm_morans
    from .stwm_core import build_stwm

    results = {}
    for q in winsorize_quantiles:
        results[f"q{q}"] = {}
        for ma in min_abs_values:
            TWM  = build_twm_morans(morans_sequence,
                                    winsorize_quantile=q,
                                    min_abs=ma)
            STWM = build_stwm(TWM, W_spatial)
            try:
                model = ModelClass(STWM).fit(Y, X)
                res   = model.summary()
                results[f"q{q}"][f"ma{ma}"] = {
                    "direct"  : res.get("direct"),
                    "indirect": res.get("indirect"),
                    "total"   : res.get("total"),
                    "rho"     : res.get("rho"),
                }
            except Exception as e:
                results[f"q{q}"][f"ma{ma}"] = {"error": str(e)}

    return results
