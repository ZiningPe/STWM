"""
dynamics.py
-----------
Dynamic analysis tools — the core value-add of spatial-TEMPORAL econometrics.

These functions answer the key question: HOW DO spillover effects vary across
space and time? This is what distinguishes STWM from a static spatial model.

Functions
---------
1. rolling_effects        : estimate effects for any sliding time window
2. regional_effects       : estimate effects for any user-defined set of regions
3. unit_effects           : effect of a shock happened in ONE spatial unit at a given time propagating to ALL other spatial units at t, t+1, t+2, etc.
4. spillover_heatmap      : N×T matrix of indirect effects (visualise evolution)
5. coefficient_stability  : track how β, direct, indirect change over time
"""

import numpy as np
from scipy import stats
import warnings
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# 1. Rolling window effects
# ---------------------------------------------------------------------------

def rolling_effects(Y: np.ndarray,
                    X: np.ndarray,
                    TWM_sequence: np.ndarray,
                    W_spatial: np.ndarray,
                    n: int,
                    T: int,
                    window: int,
                    ModelClass,
                    var_names: Optional[List[str]] = None) -> dict:
    """
    Estimate spatial model effects on a rolling time window.

    For each window [t, t+window), re-constructs the STWM from the
    sub-sequence of TWM and re-estimates the model. This reveals how
    direct/indirect/total effects evolve over time.

    Parameters
    ----------
    Y            : (nT,) stacked dependent variable
    X            : (nT, k) stacked regressors
    TWM_sequence : (T, T) full TWM (will be sub-setted per window)
    W_spatial    : (n, n) static spatial weight matrix
    n, T         : spatial units and total time periods
    window       : window size in years
    ModelClass   : model class, e.g. SDMModel

    Returns
    -------
    dict of {window_label: {rho, direct, indirect, total, direct_t, indirect_t, ...}}
    """
    from .stwm_core import build_stwm
    k = X.shape[1]
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(k)]

    results = {}
    windows = []

    for t0 in range(T - window + 1):
        t1 = t0 + window
        idx = np.concatenate([np.arange(n) + t * n for t in range(t0, t1)])
        Y_w = Y[idx]
        X_w = X[idx]

        # Sub-set TWM to this window
        TWM_w = TWM_sequence[t0:t1, t0:t1].copy()
        rs    = TWM_w.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        TWM_w /= rs

        STWM_w = build_stwm(TWM_w, W_spatial)

        label = f"t{t0}–t{t1-1}"
        try:
            model = ModelClass(STWM_w).fit(Y_w, X_w)
            res   = model.summary()
            results[label] = {k: res.get(k) for k in
                ['rho', 'direct', 'indirect', 'total',
                 'direct_se', 'indirect_se', 'total_se',
                 'direct_t', 'indirect_t', 'total_t',
                 'direct_p', 'indirect_p', 'total_p']}
            windows.append(label)
        except Exception as e:
            results[label] = {"error": str(e)}

    results["_windows"] = windows
    results["_var_names"] = var_names
    return results


def print_rolling_effects(results: dict, var_idx: int = 0):
    """
    Print how direct/indirect effects of one variable change across windows.

    Parameters
    ----------
    results : output from rolling_effects()
    var_idx : which variable to show (0-indexed)
    """
    windows   = results.get("_windows", [])
    var_names = results.get("_var_names", [])
    var_name  = var_names[var_idx] if var_idx < len(var_names) else f"x{var_idx+1}"

    print(f"\n{'─'*80}")
    print(f"  Rolling Effects: {var_name}")
    print(f"{'─'*80}")
    print(f"  {'Window':<12} {'Direct':>9} {'p':>7} {'Indirect':>10} {'p':>7} {'Total':>9} {'p':>7}")
    print(f"  {'─'*78}")
    for w in windows:
        res = results.get(w, {})
        if "error" in res:
            print(f"  {w:<12}  ERROR: {res['error']}")
            continue
        d   = res['direct'][var_idx]   if res.get('direct')   is not None else np.nan
        i   = res['indirect'][var_idx] if res.get('indirect') is not None else np.nan
        tt  = res['total'][var_idx]    if res.get('total')    is not None else np.nan
        dp  = res['direct_p'][var_idx]   if res.get('direct_p')   is not None else np.nan
        ip  = res['indirect_p'][var_idx] if res.get('indirect_p') is not None else np.nan
        tp  = res['total_p'][var_idx]    if res.get('total_p')    is not None else np.nan
        print(f"  {w:<12} {d:>9.4f} {dp:>7.4f} {i:>10.4f} {ip:>7.4f} {tt:>9.4f} {tp:>7.4f}")


# ---------------------------------------------------------------------------
# 2. Regional effects
# ---------------------------------------------------------------------------

def regional_effects(Y: np.ndarray,
                     X: np.ndarray,
                     STWM: np.ndarray,
                     n: int,
                     T: int,
                     region_groups: Dict[str, List[int]],
                     ModelClass,
                     var_names: Optional[List[str]] = None) -> dict:
    """
    Estimate effects separately for user-defined regional groups.

    Allows arbitrary region definitions — pass any list of unit indices.

    Parameters
    ----------
    Y             : (nT,) stacked dependent variable
    X             : (nT, k) stacked regressors
    STWM          : (nT, nT) full spatial-temporal weight matrix
    n, T          : spatial units and time periods
    region_groups : {region_name: [unit_index_0, unit_index_1, ...]}
                    e.g. {'East': [0,1,2,3], 'West': [4,5,6]}
                    Unit indices are 0-based (row index in annual cross-section)

    Returns
    -------
    dict of {region_name: model_summary}
    """
    k = X.shape[1]
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(k)]

    results = {}
    for reg_name, unit_idx in region_groups.items():
        unit_idx = np.array(unit_idx)
        # Build panel indices for these units across ALL time periods
        panel_idx = np.sort(np.concatenate(
            [unit_idx + t * n for t in range(T)]
        ))
        Y_r     = Y[panel_idx]
        X_r     = X[panel_idx]
        STWM_r  = STWM[np.ix_(panel_idx, panel_idx)]

        # Re-row-standardise sub-matrix
        rs = STWM_r.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        STWM_r /= rs

        try:
            model = ModelClass(STWM_r).fit(Y_r, X_r)
            res   = model.summary()
            res["_var_names"] = var_names
            res["_n_units"]   = len(unit_idx)
            res["_T"]         = T
            results[reg_name] = res
        except Exception as e:
            results[reg_name] = {"error": str(e)}

    return results


def print_regional_comparison(results: dict, var_idx: int = 0):
    """Compare direct/indirect/total effects across regions for one variable."""
    var_names = next(
        (v["_var_names"] for v in results.values() if "_var_names" in v), None
    )
    var_name = var_names[var_idx] if (var_names and var_idx < len(var_names)) \
               else f"x{var_idx+1}"

    print(f"\n{'─'*80}")
    print(f"  Regional Effect Comparison: {var_name}")
    print(f"{'─'*80}")
    print(f"  {'Region':<16} {'N units':>8} {'Direct':>9} {'p':>7} {'Indirect':>10} {'p':>7} {'Total':>9} {'p':>7}")
    print(f"  {'─'*78}")
    for reg, res in results.items():
        if "error" in res:
            print(f"  {reg:<16}  ERROR: {res['error']}")
            continue
        nu  = res.get("_n_units", "?")
        d   = res["direct"][var_idx]
        i   = res["indirect"][var_idx]
        tot = res.get("total", res["direct"])[var_idx]
        dp  = res.get("direct_p",   [np.nan]*20)[var_idx]
        ip  = res.get("indirect_p", [np.nan]*20)[var_idx]
        tp  = res.get("total_p",    [np.nan]*20)[var_idx]
        print(f"  {reg:<16} {nu:>8} {d:>9.4f} {dp:>7.4f} {i:>10.4f} {ip:>7.4f} {tot:>9.4f} {tp:>7.4f}")


# ---------------------------------------------------------------------------
# 3. Unit-level shock propagation
# ---------------------------------------------------------------------------

def unit_shock_propagation(STWM: np.ndarray,
                            n: int,
                            T: int,
                            rho: float,
                            beta_k: float,
                            source_units: List[int],
                            t_shock: int = 0) -> dict:
    """
    Compute the spatial propagation of a unit shock from specified units.

    Given a unit shock Δx_{i,k} = 1 at time t_shock in source units,
    compute ∂E(y_j)/∂x_{ik} = [(I−ρW)⁻¹]_{ji} · β_k  for all j.

    This directly uses Elhorst (2014) eq. 2.13:
        S_k(W) = (I−ρW)⁻¹ · β_k

    The (i,j) element gives: impact of a unit change in x_{jk} on y_i.

    Parameters
    ----------
    STWM         : (nT, nT) spatial-temporal weight matrix
    n, T         : spatial units and time periods
    rho          : estimated spatial autoregressive parameter
    beta_k       : coefficient for the variable of interest
    source_units : list of unit indices (0-based) where shock originates
    t_shock      : time period of shock (0-based)

    Returns
    -------
    dict with:
        impact_vector : (n,) impact on each unit in period t_shock
        S_matrix      : (nT, nT) full partial derivative matrix
    """
    nT     = n * T
    A      = np.eye(nT) - rho * STWM
    try:
        S  = np.linalg.inv(A) * beta_k
    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix in shock propagation.", stacklevel=2)
        return {"error": "Singular matrix"}

    # Extract the panel index for source units at t_shock
    source_panel = [u + t_shock * n for u in source_units]
    # Average impact across all source units and all recipient units
    impact = S[:, source_panel].mean(axis=1)  # (nT,)

    # Reshape to (T, n) to show time × space
    impact_matrix = impact.reshape(T, n)  # impact_matrix[t, i] = effect on unit i at time t

    return dict(
        source_units  = source_units,
        t_shock       = t_shock,
        impact_matrix = impact_matrix,   # (T, n)
        impact_t0     = impact_matrix[t_shock],  # (n,) effect in shock period
        total_direct  = float(np.trace(S) / nT),
        total_indirect= float((np.ones(nT) @ S @ np.ones(nT)) / nT
                              - np.trace(S) / nT),
    )


# ---------------------------------------------------------------------------
# 4. Coefficient stability across time (no rolling — single pass)
# ---------------------------------------------------------------------------

def coefficient_stability(rolling_results: dict,
                          var_names: Optional[List[str]] = None) -> dict:
    """
    Summarise stability of direct/indirect effects across rolling windows.

    Computes: mean, std, min, max, CV (coefficient of variation) for each
    variable's direct and indirect effect across all windows.

    High CV → effects vary substantially over time (interesting dynamics).
    Low CV  → stable effects (robust finding).

    Parameters
    ----------
    rolling_results : output from rolling_effects()

    Returns
    -------
    dict of {var_name: {direct_mean, direct_std, direct_cv, indirect_mean, ...}}
    """
    windows   = rolling_results.get("_windows", [])
    _vnames   = rolling_results.get("_var_names", var_names or [])
    k_det     = None

    d_all  = []
    i_all  = []
    t_all  = []
    for w in windows:
        res = rolling_results.get(w, {})
        if "error" in res or res.get("direct") is None:
            continue
        d_all.append(res["direct"])
        i_all.append(res["indirect"])
        t_all.append(res["total"])
        k_det = len(res["direct"])

    if not d_all:
        return {}

    d_arr = np.array(d_all)   # (W, k)
    i_arr = np.array(i_all)
    t_arr = np.array(t_all)

    stability = {}
    for j in range(k_det):
        vname = _vnames[j] if j < len(_vnames) else f"x{j+1}"
        d_col = d_arr[:, j]
        i_col = i_arr[:, j]
        stability[vname] = dict(
            direct_mean   = float(d_col.mean()),
            direct_std    = float(d_col.std()),
            direct_cv     = float(d_col.std() / abs(d_col.mean())) if d_col.mean() != 0 else np.nan,
            direct_min    = float(d_col.min()),
            direct_max    = float(d_col.max()),
            indirect_mean = float(i_col.mean()),
            indirect_std  = float(i_col.std()),
            indirect_cv   = float(i_col.std() / abs(i_col.mean())) if i_col.mean() != 0 else np.nan,
            indirect_min  = float(i_col.min()),
            indirect_max  = float(i_col.max()),
        )

    print(f"\n{'─'*70}")
    print(f"  Coefficient Stability across {len(d_all)} Windows")
    print(f"{'─'*70}")
    print(f"  {'Variable':<14} {'D.Mean':>8} {'D.Std':>8} {'D.CV':>7} | "
          f"{'I.Mean':>8} {'I.Std':>8} {'I.CV':>7}")
    print(f"  {'─'*68}")
    for vname, s in stability.items():
        print(f"  {vname:<14} {s['direct_mean']:>8.4f} {s['direct_std']:>8.4f} "
              f"{s['direct_cv']:>7.3f} | "
              f"{s['indirect_mean']:>8.4f} {s['indirect_std']:>8.4f} "
              f"{s['indirect_cv']:>7.3f}")
    print(f"\n  CV = std/|mean|. High CV → time-varying effects (rich dynamics).")

    return stability
