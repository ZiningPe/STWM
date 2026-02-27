"""
temporal_weights.py
-------------------
Time Weight Matrix (TWM) construction from multiple temporal statistics.

Five approaches to encode temporal dependence (choose based on theory):

┌─────────────────────┬──────────────────────────────────────────────────────┐
│ Method              │ Economic Interpretation                               │
├─────────────────────┼──────────────────────────────────────────────────────┤
│ 1. Moran's I ratio  │ Carry-forward of global spatial clustering strength  │
│ 2. Geary's C ratio  │ Carry-forward of local spatial dissimilarity         │
│ 3. Getis-Ord G*     │ Hot-spot concentration ratio (local clustering)      │
│ 4. Spatial Gini     │ Concentration/inequality of spatial distribution     │
│ 5. Decay-based      │ Agnostic benchmark: exp / linear / power decay       │
└─────────────────────┴──────────────────────────────────────────────────────┘

Mathematical Formulation (same for all ratio-based methods)
------------------------------------------------------------
Given a T-length sequence of statistics {a_1, ..., a_T}:

    Raw TWM entry:
        TWM[t, s] = a_t / a_s    for s < t  (lower triangle, causal)
        TWM[t, t] = 1            (diagonal)
        TWM[t, s] = 0            for s > t  (upper triangle = no future info)

    Processing pipeline:
        Step 1: Replace |denominator| < min_abs with sign * min_abs
        Step 2: Winsorise at quantile q: clip(., -Q_q, +Q_q)
        Step 3: Take absolute value (preserves dispersion magnitude)
        Step 4: Row-standardise: TWM[t,:] /= sum(TWM[t,:])

Admissibility Conditions (Elhorst 2014, Section 2.2)
-----------------------------------------------------
A valid weight matrix W must satisfy:
  (1) All entries ≥ 0
  (2) Each row sums to 1  (row-standardisation)
  (3) Spectral radius ρ(W) < 1  (required for (I−δW)⁻¹ to exist)
  (4) No NaN or Inf

References
----------
Moran, P. A. P. (1950). Biometrika, 37, 17–23.
Geary, R. C. (1954). The Incorporated Statistician, 5, 115–145.
Getis, A., & Ord, J. K. (1992). Geographical Analysis, 24(3), 189–206.
Elhorst, J. P. (2014). Spatial Econometrics. Springer.
"""

import numpy as np
import warnings
from typing import Literal


# ---------------------------------------------------------------------------
# Internal shared builder
# ---------------------------------------------------------------------------

def _ratio_matrix(stats: np.ndarray,
                  winsorize_quantile: float,
                  min_abs: float) -> np.ndarray:
    """Lower-triangular ratio TWM with winsorisation and row-standardisation."""
    T   = len(stats)
    mat = np.zeros((T, T))
    for t in range(T):
        for s in range(t):
            denom = stats[s]
            if abs(denom) < min_abs:
                denom = (np.sign(denom) * min_abs) if denom != 0 else min_abs
            mat[t, s] = stats[t] / denom
        mat[t, t] = 1.0
    off = mat[mat != 0]
    if len(off) > 0:
        cap = np.quantile(np.abs(off), winsorize_quantile)
        mat = np.clip(mat, -cap, cap)
    mat = np.abs(mat)   # preserve magnitude of dispersion (I < 0) rather than zeroing
    rs  = mat.sum(axis=1, keepdims=True)
    rs  = np.where(rs == 0, 1.0, rs)
    return mat / rs


def _warn_near_zero(stats: np.ndarray, name: str, threshold: float = 0.05):
    idx = np.where(np.abs(stats) < threshold)[0]
    if len(idx) > 0:
        warnings.warn(
            f"{name} near zero at t={idx.tolist()}. "
            "Ratio TWM may be unstable; winsorisation applied.",
            UserWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# 1. Moran's I
# ---------------------------------------------------------------------------

def compute_morans_i(y: np.ndarray, W: np.ndarray) -> float:
    """
    Global Moran's I (Moran 1950).

    Formula:
        I = (n / S₀) · (z'Wz) / (z'z)

    where z = y − ȳ, S₀ = Σᵢⱼ wᵢⱼ.

    Range ≈ [−1, 1]:  I > 0 → clustering,  I < 0 → dispersion.
    """
    n  = len(y)
    z  = y - y.mean()
    S0 = W.sum()
    return float(n * (z @ (W @ z)) / (S0 * (z @ z)))


def build_twm_morans(morans_sequence: np.ndarray,
                     winsorize_quantile: float = 0.95,
                     min_abs: float = 1e-3) -> np.ndarray:
    """
    TWM from Moran's I sequence.

    TWM[t, s] = I_t / I_s    (s < t)

    Economic interpretation: periods with higher spatial clustering relative
    to an earlier period receive stronger temporal weight. Captures how the
    evolution of global spatial agglomeration propagates forward in time.

    Parameters
    ----------
    morans_sequence    : (T,) annual Moran's I values
    winsorize_quantile : cap for extreme ratios (default 0.95)
    min_abs            : min |denominator| to avoid division explosion

    Returns
    -------
    (T, T) row-standardised lower-triangular TWM
    """
    stats = np.asarray(morans_sequence, float)
    _warn_near_zero(stats, "Moran's I")
    return _ratio_matrix(stats, winsorize_quantile, min_abs)


# ---------------------------------------------------------------------------
# 2. Geary's C
# ---------------------------------------------------------------------------

def compute_geary_c(y: np.ndarray, W: np.ndarray) -> float:
    """
    Global Geary's C (Geary 1954).

    Formula:
        C = [(n−1) · Σᵢⱼ wᵢⱼ(yᵢ−yⱼ)²] / [2S₀ · Σᵢ(yᵢ−ȳ)²]

    Range (0, 2):  C < 1 → clustering,  C = 1 → random,  C > 1 → dispersion.
    (Note: inverse direction from Moran's I)
    """
    n    = len(y)
    S0   = W.sum()
    diff = np.subtract.outer(y, y) ** 2
    num  = (n - 1) * float((W * diff).sum())
    den  = 2.0 * S0 * float(np.var(y) * n)
    return num / den


def build_twm_gearyc(gearyc_sequence: np.ndarray,
                     winsorize_quantile: float = 0.95,
                     min_abs: float = 1e-3) -> np.ndarray:
    """
    TWM from Geary's C sequence.

    Transformation: a_t = 2 − C_t  (re-scales so higher = more clustering)
    TWM[t, s] = (2 − C_t) / (2 − C_s)

    Economic interpretation: captures local-level spatial heterogeneity
    dynamics. Complements Moran's I (which is a global measure).
    Geary's C is more sensitive to local spatial differences.

    Returns
    -------
    (T, T) row-standardised lower-triangular TWM
    """
    stats = np.asarray(gearyc_sequence, float)
    transformed = 2.0 - stats
    _warn_near_zero(transformed, "Geary's C (2−C)")
    return _ratio_matrix(transformed, winsorize_quantile, min_abs)


# ---------------------------------------------------------------------------
# 3. Getis-Ord G* (hot-spot concentration)
# ---------------------------------------------------------------------------

def compute_getis_ord_g(y: np.ndarray, W: np.ndarray) -> float:
    """
    Global Getis-Ord G statistic (Getis & Ord 1992).

    Formula:
        G = Σᵢⱼ wᵢⱼ yᵢ yⱼ / Σᵢⱼ yᵢ yⱼ   (i ≠ j)

    Captures the concentration of high values near other high values
    (hot-spot clustering), distinct from Moran's I which measures
    clustering of all values relative to the mean.

    Requires y ≥ 0.
    """
    if np.any(y < 0):
        warnings.warn("Getis-Ord G requires y ≥ 0. Shifting y = y − min(y).",
                      UserWarning, stacklevel=2)
        y = y - y.min()
    n = len(y)
    # Remove diagonal (i ≠ j)
    W_nodiag = W * (1 - np.eye(n))
    num = float(W_nodiag @ y @ y)
    outer = np.outer(y, y)
    den = float((outer * (1 - np.eye(n))).sum())
    if den < 1e-12:
        return 0.0
    return num / den


def build_twm_getis_ord(g_sequence: np.ndarray,
                         winsorize_quantile: float = 0.95,
                         min_abs: float = 1e-3) -> np.ndarray:
    """
    TWM from Getis-Ord G sequence.

    TWM[t, s] = G_t / G_s

    Economic interpretation: tracks temporal evolution of hot-spot
    concentration. Particularly useful when the research question
    concerns spatial concentration of high-value activities
    (e.g. industrial clusters, patent hubs). Moran's I measures
    general clustering; G specifically measures co-location of HIGH values.

    Returns
    -------
    (T, T) row-standardised lower-triangular TWM
    """
    stats = np.asarray(g_sequence, float)
    _warn_near_zero(stats, "Getis-Ord G")
    return _ratio_matrix(stats, winsorize_quantile, min_abs)


# ---------------------------------------------------------------------------
# 4. Spatial Gini coefficient
# ---------------------------------------------------------------------------

def compute_spatial_gini(y: np.ndarray, W: np.ndarray) -> float:
    """
    Spatial Gini coefficient.

    Formula:
        SG = Σᵢⱼ wᵢⱼ |yᵢ − yⱼ| / (2n²·ȳ)

    Measures spatial inequality — how concentrated is the variable
    across space, weighted by spatial proximity.

    Range [0, 1]: 0 = perfect equality, 1 = extreme concentration.
    Unlike Moran's I and Geary's C, Gini focuses on distributional
    inequality rather than clustering direction.
    """
    n    = len(y)
    ybar = y.mean()
    if ybar == 0:
        return 0.0
    diff = np.abs(np.subtract.outer(y, y))
    return float((W * diff).sum()) / (2 * n**2 * ybar)


def build_twm_spatial_gini(gini_sequence: np.ndarray,
                            winsorize_quantile: float = 0.95,
                            min_abs: float = 1e-3) -> np.ndarray:
    """
    TWM from Spatial Gini sequence.

    TWM[t, s] = SG_t / SG_s

    Economic interpretation: tracks temporal evolution of spatial
    inequality. If your research question involves convergence/divergence
    of regions over time, the Gini-based TWM captures whether the
    current distribution is more or less unequal than past periods.

    Particularly suitable for green innovation studies where
    regional inequality in patent activity changes over policy periods.

    Returns
    -------
    (T, T) row-standardised lower-triangular TWM
    """
    stats = np.asarray(gini_sequence, float)
    _warn_near_zero(stats, "Spatial Gini")
    return _ratio_matrix(stats, winsorize_quantile, min_abs)


# ---------------------------------------------------------------------------
# 5. Decay-based TWM (benchmark)
# ---------------------------------------------------------------------------

def build_twm_decay(T: int,
                    decay_type: Literal["exponential", "linear", "power"] = "exponential",
                    param: float = 0.5) -> np.ndarray:
    """
    Benchmark decay-based TWM (no data-driven component).

    Formulas (before row-standardisation):
        Exponential : w[t,s] = exp(−λ(t−s))
        Linear      : w[t,s] = max(0, 1 − λ(t−s))
        Power       : w[t,s] = (t−s)^{−λ}  for t > s

    Use for:
      - Robustness comparison against data-driven TWMs
      - Sensitivity analysis to choice of λ
      - Situations where no annual statistics are available

    Parameters
    ----------
    T          : number of time periods
    decay_type : 'exponential' | 'linear' | 'power'
    param      : decay parameter λ

    Returns
    -------
    (T, T) row-standardised lower-triangular TWM
    """
    mat = np.zeros((T, T))
    for t in range(T):
        for s in range(t + 1):
            lag = t - s
            if lag == 0:
                mat[t, s] = 1.0
            elif decay_type == "exponential":
                mat[t, s] = np.exp(-param * lag)
            elif decay_type == "linear":
                mat[t, s] = max(0.0, 1.0 - param * lag)
            elif decay_type == "power":
                mat[t, s] = lag ** (-param)
            else:
                raise ValueError(f"Unknown decay_type: '{decay_type}'.")
    rs = mat.sum(axis=1, keepdims=True)
    rs = np.where(rs == 0, 1.0, rs)
    return mat / rs


# ---------------------------------------------------------------------------
# Admissibility check
# ---------------------------------------------------------------------------

def twm_stability_check(TWM: np.ndarray, rho_max: float = 0.99) -> dict:
    """
    Verify admissibility conditions for a TWM.

    Checks:
      (1) Non-negativity       : all entries ≥ 0
      (2) Row-standardisation  : each row sums to 1 (tol 1e-6)
      (3) No NaN/Inf

    Note on spectral radius:
      A row-normalised lower-triangular TWM always has spectral radius = 1.0
      (Perron-Frobenius: eigenvalues = diagonal = 1 for the first row).
      The relevant admissibility constraint for SAR/SDM is on the δ parameter
      (|δ| < 1/ρ(STWM)), NOT on ρ(TWM) itself.  We therefore report the
      spectral radius for information only and do NOT fail the check on it.

    Returns
    -------
    dict with keys: non_negative, row_standardised, spectral_radius,
                    has_nan_inf, passed
    """
    non_neg  = bool(np.all(TWM >= 0))
    row_std  = bool(np.allclose(TWM.sum(axis=1), 1.0, atol=1e-6))
    # Eigenvalues of lower-triangular matrix = diagonal entries
    spec_rad = float(np.max(np.abs(np.diag(TWM))))   # fast: O(T) not O(T³)
    has_nan  = bool(np.any(~np.isfinite(TWM)))
    # Spectral radius deliberately excluded from passed (always ≈ 1 by construction)
    passed   = non_neg and row_std and (not has_nan)
    result   = dict(non_negative=non_neg, row_standardised=row_std,
                    spectral_radius=round(spec_rad, 6),
                    has_nan_inf=has_nan, passed=passed)
    if not passed:
        warnings.warn(f"TWM admissibility FAILED: {result}", stacklevel=2)
    return result


# ---------------------------------------------------------------------------
# Convenience: compute all four statistics for a given year's data
# ---------------------------------------------------------------------------

def validate_stwm_ordering(W_full: np.ndarray,
                           SWM: np.ndarray,
                           TWM: np.ndarray,
                           tol: float = 1e-8) -> dict:
    """
    Validate that W_full is consistent with time-major Kronecker ordering.

    The STWM package convention is::

        W_full = np.kron(TWM, SWM)          # (nT × nT)

    which corresponds to *time-major* panel stacking:
    obs 0..n−1 → all units at t=0,  obs n..2n−1 → all units at t=1, …

    This function checks whether the supplied W_full equals np.kron(TWM, SWM)
    element-wise (up to tolerance) and reports block-level details if not.

    Parameters
    ----------
    W_full : (nT, nT) full spatial-temporal weight matrix
    SWM    : (n, n)  spatial weight matrix
    TWM    : (T, T)  temporal weight matrix
    tol    : absolute tolerance for element-wise comparison (default 1e-8)

    Returns
    -------
    dict with keys:
      consistent      : bool — True if W_full ≈ kron(TWM, SWM)
      max_abs_diff    : float — maximum element-wise absolute difference
      T               : int — number of time periods (inferred from TWM)
      n_units         : int — number of spatial units (inferred from SWM)
      stacking        : str — description of expected ordering convention
      message         : str — ✓ or ✗ with diff magnitude
      block_errors    : list of (t1, t2, diff) for blocks with diff > tol
                        (only populated when not consistent)

    Example
    -------
    >>> result = validate_stwm_ordering(STWM, SWM, TWM)
    >>> assert result['consistent'], result['message']
    """
    T = TWM.shape[0]
    n = SWM.shape[0]
    expected    = np.kron(TWM, SWM)
    diff_mat    = np.abs(W_full - expected)
    max_diff    = float(diff_mat.max())
    consistent  = bool(max_diff < tol)

    block_errors = []
    if not consistent:
        for t1 in range(T):
            for t2 in range(T):
                r0, r1 = t1 * n, (t1 + 1) * n
                c0, c1 = t2 * n, (t2 + 1) * n
                bd = float(diff_mat[r0:r1, c0:c1].max())
                if bd >= tol:
                    block_errors.append((t1, t2, round(bd, 6)))

    msg = ('✓ Consistent with kron(TWM, SWM)  [time-major ordering]'
           if consistent
           else f'✗ Inconsistent: max diff = {max_diff:.2e}; '
                f'{len(block_errors)} block(s) violate tolerance')

    return dict(
        consistent=consistent,
        max_abs_diff=round(max_diff, 10),
        T=T,
        n_units=n,
        stacking='time-major: W_full[t1*n:(t1+1)*n, t2*n:(t2+1)*n] = TWM[t1,t2]*SWM',
        message=msg,
        block_errors=block_errors,
    )


def compute_all_temporal_stats(y: np.ndarray, W: np.ndarray) -> dict:
    """
    Compute all four temporal statistics for one cross-section.

    Parameters
    ----------
    y : (n,) observed variable for one year
    W : (n,n) row-standardised spatial weight matrix

    Returns
    -------
    dict : {morans_i, geary_c, getis_g, spatial_gini}
    """
    return dict(
        morans_i    = compute_morans_i(y, W),
        geary_c     = compute_geary_c(y, W),
        getis_g     = compute_getis_ord_g(y, W),
        spatial_gini= compute_spatial_gini(y, W),
    )
