"""
stwm_core.py
------------
Assemble the Spatial-Temporal Weight Matrix (STWM) via Kronecker product.

Mathematical Formulation
------------------------
Given:
    SWM : (n × n) row-standardised Spatial Weight Matrix
    TWM : (T × T) row-standardised Time Weight Matrix

The STWM is defined as:

    STWM = TWM ⊗ SWM    (Kronecker product)

Resulting dimension:  (nT × nT)

Element interpretation:
    STWM[(t·n + i), (s·n + j)] = TWM[t, s] × SWM[i, j]

This encodes:
  - Spatial proximity of unit i to unit j (from SWM)
  - Temporal proximity of period t to period s (from TWM)
  simultaneously, without requiring separate spatial and temporal lags.

Row-standardisation is applied after the Kronecker product so that
each row of STWM sums to 1, preserving the interpretability of the
spatial lag as a weighted average.

Stacking Convention
-------------------
The panel vector Y of shape (nT,) is assumed to be stacked as:

    Y = [y_{1,1}, ..., y_{n,1},   ← period 1
         y_{1,2}, ..., y_{n,2},   ← period 2
         ...
         y_{1,T}, ..., y_{n,T}]   ← period T

i.e., spatial units are the inner dimension (time-major order).
"""

import numpy as np


def build_stwm(TWM: np.ndarray,
               SWM: np.ndarray,
               row_standardize: bool = True) -> np.ndarray:
    """
    Build STWM = TWM ⊗ SWM via Kronecker product.

    Parameters
    ----------
    TWM             : (T, T) time weight matrix
    SWM             : (n, n) spatial weight matrix
    row_standardize : whether to row-standardise the final STWM (default True)

    Returns
    -------
    STWM : (nT, nT) spatial-temporal weight matrix
    """
    STWM = np.kron(TWM, SWM)
    if row_standardize:
        rs   = STWM.sum(axis=1, keepdims=True)
        rs   = np.where(rs == 0, 1.0, rs)
        STWM = STWM / rs
    return STWM


def stwm_summary(STWM: np.ndarray, n: int, T: int) -> dict:
    """
    Summarise key properties of an assembled STWM.

    Parameters
    ----------
    STWM : (nT, nT) spatial-temporal weight matrix
    n    : number of spatial units
    T    : number of time periods

    Returns
    -------
    dict with:
        shape           : matrix dimensions
        n_spatial       : n
        T_temporal      : T
        sparsity        : fraction of zero entries
        spectral_radius : largest absolute eigenvalue
        weight_min/max/mean : statistics of non-zero off-diagonal entries
    """
    assert STWM.shape == (n * T, n * T), \
        f"Expected ({n*T}, {n*T}), got {STWM.shape}."

    nnz      = np.count_nonzero(STWM)
    sparsity = 1.0 - nnz / STWM.size
    spec_rad = float(np.max(np.abs(np.linalg.eigvals(STWM))))
    off      = STWM[STWM != 0]

    return {
        "shape"          : STWM.shape,
        "n_spatial"      : n,
        "T_temporal"     : T,
        "sparsity"       : round(sparsity, 4),
        "spectral_radius": round(spec_rad, 6),
        "weight_min"     : float(off.min())  if len(off) else 0.0,
        "weight_max"     : float(off.max())  if len(off) else 0.0,
        "weight_mean"    : float(off.mean()) if len(off) else 0.0,
    }
