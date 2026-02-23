"""
temporal_weights.py
-------------------
Construct Time Weight Matrices (TWM) from spatial autocorrelation statistics.

Core Innovation
---------------
Rather than imposing arbitrary temporal decay functions, this module derives
temporal propagation "memory" from the year-to-year evolution of observed
spatial autocorrelation statistics (Moran's I or Geary's C).

The key idea: if spatial clustering in period s is a precursor to clustering
in period t, the ratio I_t / I_s encodes the relative strength of that
temporal link. Row-standardisation ensures each row sums to 1.

Mathematical Formulation
------------------------
Given a sequence of T annual statistics {a_1, a_2, ..., a_T}:

    Raw TWM entry:
        TWM[t, s] = a_t / a_s    for s < t   (lower triangle)
        TWM[t, t] = 1            (diagonal)
        TWM[t, s] = 0            for s > t   (upper triangle, causal)

    After winsorisation at quantile q and clipping negatives to 0:
        TWM[t, s] ← clip(TWM[t, s], 0, Q_q(|TWM|))

    Row-standardisation:
        TWM[t, :] ← TWM[t, :] / sum(TWM[t, :])

Admissibility Conditions
------------------------
A valid TWM must satisfy:
  (1) All entries ≥ 0                (non-negativity)
  (2) Each row sums to 1             (row-standardisation)
  (3) Spectral radius ρ(TWM) < 1    (stability for spatial lag models)
  (4) No NaN or Inf entries

The function `twm_stability_check` verifies all four conditions.

Decay-Based Alternative
-----------------------
`build_twm_decay` provides conventional decay-based TWMs for comparison:
  - Exponential:  w[t,s] = exp(-λ(t−s))
  - Linear:       w[t,s] = max(0, 1 − λ(t−s))
  - Power:        w[t,s] = (t−s)^{−λ}

Illustrative Example (T=3)
--------------------------
    stats = [0.40, 0.50, 0.30]

    Raw lower-triangle:
        TWM[1,0] = 0.50 / 0.40 = 1.25
        TWM[2,0] = 0.30 / 0.40 = 0.75
        TWM[2,1] = 0.30 / 0.50 = 0.60

    After row-standardisation:
        row 0: [1.00,  0,    0   ]
        row 1: [0.56,  0.44, 0   ]  (1.25 / 2.25, 1.00 / 2.25)
        row 2: [0.36,  0.29, 0.36]  ... (normalised)

    Economic interpretation: TWM[t,s] represents the relative weight of
    period s's spatial clustering pattern in predicting period t's pattern.
    Higher ratio → stronger temporal carry-forward from s to t.
"""

import numpy as np
import warnings
from typing import Literal


# ---------------------------------------------------------------------------
# 1.  Spatial autocorrelation statistics
# ---------------------------------------------------------------------------

def compute_morans_i(y: np.ndarray, W: np.ndarray) -> float:
    """
    Compute Global Moran's I.

    Formula
    -------
        I = (n / S_0) * (z' W z) / (z' z)

    where:
        z  = y − ȳ  (mean-centred variable)
        S_0 = sum of all weights in W
        n   = number of observations

    Range: approximately [−1, 1]
        I > 0 → positive spatial autocorrelation (clustering)
        I < 0 → negative spatial autocorrelation (dispersion)
        I ≈ E[I] = −1/(n−1) → spatial randomness

    Parameters
    ----------
    y : (n,) array   — observed variable
    W : (n,n) array  — row-standardised spatial weight matrix

    Returns
    -------
    float : Moran's I
    """
    n  = len(y)
    z  = y - y.mean()
    Wz = W @ z
    S0 = W.sum()
    return float(n * (z @ Wz) / (S0 * (z @ z)))


def compute_geary_c(y: np.ndarray, W: np.ndarray) -> float:
    """
    Compute Global Geary's C.

    Formula
    -------
        C = [(n−1) * Σ_ij w_ij (y_i − y_j)²] / [2 S_0 * Σ_i (y_i − ȳ)²]

    Range: (0, 2)
        C < 1 → positive spatial autocorrelation
        C ≈ 1 → spatial randomness
        C > 1 → negative spatial autocorrelation

    Parameters
    ----------
    y : (n,) array
    W : (n,n) array  — row-standardised

    Returns
    -------
    float : Geary's C
    """
    n    = len(y)
    S0   = W.sum()
    diff = np.subtract.outer(y, y) ** 2
    num  = (n - 1) * float((W * diff).sum())
    den  = 2.0 * S0 * float(np.var(y) * n)
    return num / den


# ---------------------------------------------------------------------------
# 2.  TWM from autocorrelation sequences
# ---------------------------------------------------------------------------

def _ratio_matrix(stats: np.ndarray,
                  winsorize_quantile: float,
                  min_abs: float) -> np.ndarray:
    """
    Internal: build a lower-triangular ratio matrix and row-standardise.

    Stability rules applied before row-standardisation:
      1. |denominator| < min_abs  →  replace with sign * min_abs
      2. All ratios winsorised to [0, Q_{winsorize_quantile}(|ratios|)]
      3. Negative entries clipped to 0
    """
    T   = len(stats)
    mat = np.zeros((T, T))

    for t in range(T):
        for s in range(t):
            denom = stats[s]
            if abs(denom) < min_abs:
                denom = np.sign(denom) * min_abs if denom != 0 else min_abs
            mat[t, s] = stats[t] / denom
        mat[t, t] = 1.0

    # Winsorise
    off = mat[mat != 0]
    if len(off) > 0:
        cap = np.quantile(np.abs(off), winsorize_quantile)
        mat = np.clip(mat, -cap, cap)

    # Clip negatives
    mat = np.where(mat < 0, 0.0, mat)

    # Row-standardise
    rs = mat.sum(axis=1, keepdims=True)
    rs = np.where(rs == 0, 1.0, rs)
    return mat / rs


def build_twm_morans(morans_sequence: np.ndarray,
                     winsorize_quantile: float = 0.95,
                     min_abs: float = 1e-3) -> np.ndarray:
    """
    Build a Time Weight Matrix (TWM) from a sequence of Moran's I values.

    Formula (element-wise, before row-standardisation):
        TWM[t, s] = I_t / I_s    for s < t
        TWM[t, t] = 1
        TWM[t, s] = 0            for s > t

    Parameters
    ----------
    morans_sequence     : (T,) array of annual Moran's I values
    winsorize_quantile  : upper quantile cap for extreme ratios (default 0.95)
    min_abs             : minimum absolute denominator, prevents ratio explosion

    Returns
    -------
    TWM : (T, T) row-standardised lower-triangular matrix
    """
    stats = np.asarray(morans_sequence, dtype=float)
    _warn_near_zero(stats, "Moran's I")
    return _ratio_matrix(stats, winsorize_quantile, min_abs)


def build_twm_gearyc(gearyc_sequence: np.ndarray,
                     winsorize_quantile: float = 0.95,
                     min_abs: float = 1e-3) -> np.ndarray:
    """
    Build a Time Weight Matrix (TWM) from a sequence of Geary's C values.

    Transformation applied before ratio construction:
        a_t = 2 − C_t

    This maps Geary's C (where lower values indicate stronger clustering)
    to a scale consistent with Moran's I (higher value = stronger clustering).

    Formula (element-wise, before row-standardisation):
        TWM[t, s] = (2 − C_t) / (2 − C_s)    for s < t

    Parameters
    ----------
    gearyc_sequence    : (T,) array of annual Geary's C values
    winsorize_quantile : upper quantile cap (default 0.95)
    min_abs            : minimum absolute denominator (default 1e-3)

    Returns
    -------
    TWM : (T, T) row-standardised lower-triangular matrix
    """
    stats = np.asarray(gearyc_sequence, dtype=float)
    transformed = 2.0 - stats
    _warn_near_zero(transformed, "Geary's C (transformed as 2−C)")
    return _ratio_matrix(transformed, winsorize_quantile, min_abs)


# ---------------------------------------------------------------------------
# 3.  Decay-based TWM (benchmark / robustness)
# ---------------------------------------------------------------------------

def build_twm_decay(T: int,
                    decay_type: Literal["exponential", "linear", "power"] = "exponential",
                    param: float = 0.5) -> np.ndarray:
    """
    Build a conventional decay-based Time Weight Matrix (benchmark).

    Use this to compare against the autocorrelation-derived TWM, or as
    a robustness check when Moran's I / Geary's C data are unavailable.

    Formulas (before row-standardisation)
    --------------------------------------
    Exponential:    w[t,s] = exp(−λ · (t−s))
    Linear:         w[t,s] = max(0,  1 − λ · (t−s))
    Power:          w[t,s] = (t−s)^{−λ}              for t > s
    Diagonal:       w[t,t] = 1

    Parameters
    ----------
    T          : number of time periods
    decay_type : 'exponential' | 'linear' | 'power'
    param      : decay parameter λ

    Returns
    -------
    TWM : (T, T) row-standardised lower-triangular matrix
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
                raise ValueError(f"Unknown decay_type: '{decay_type}'. "
                                 "Choose 'exponential', 'linear', or 'power'.")
    rs = mat.sum(axis=1, keepdims=True)
    rs = np.where(rs == 0, 1.0, rs)
    return mat / rs


# ---------------------------------------------------------------------------
# 4.  Admissibility check
# ---------------------------------------------------------------------------

def twm_stability_check(TWM: np.ndarray,
                        rho_max: float = 0.99) -> dict:
    """
    Verify admissibility conditions for a TWM.

    Checks
    ------
    (1) Non-negativity:       all entries ≥ 0
    (2) Row-standardisation:  each row sums to 1 (tolerance 1e-6)
    (3) Spectral radius:      max|eigenvalue| < rho_max
    (4) Finite entries:       no NaN or Inf

    Parameters
    ----------
    TWM     : (T, T) time weight matrix
    rho_max : maximum allowed spectral radius (default 0.99)

    Returns
    -------
    dict : {
        'non_negative'    : bool,
        'row_standardised': bool,
        'spectral_radius' : float,
        'stable'          : bool,
        'has_nan_inf'     : bool,
        'passed'          : bool   ← True only if all four checks pass
    }
    """
    non_neg  = bool(np.all(TWM >= 0))
    row_std  = bool(np.allclose(TWM.sum(axis=1), 1.0, atol=1e-6))
    eigvals  = np.linalg.eigvals(TWM)
    spec_rad = float(np.max(np.abs(eigvals)))
    stable   = spec_rad < rho_max
    has_nan  = bool(np.any(~np.isfinite(TWM)))
    passed   = non_neg and row_std and stable and (not has_nan)

    result = {
        "non_negative"    : non_neg,
        "row_standardised": row_std,
        "spectral_radius" : round(spec_rad, 6),
        "stable"          : stable,
        "has_nan_inf"     : has_nan,
        "passed"          : passed,
    }
    if not passed:
        warnings.warn(f"TWM admissibility check FAILED: {result}", stacklevel=2)
    return result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _warn_near_zero(stats: np.ndarray, name: str, threshold: float = 0.05):
    idx = np.where(np.abs(stats) < threshold)[0]
    if len(idx) > 0:
        warnings.warn(
            f"{name} is near zero at time indices {idx.tolist()}. "
            "Ratio-based TWM may be unstable; winsorisation will be applied.",
            UserWarning, stacklevel=3,
        )
