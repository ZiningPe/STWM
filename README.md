# STWM: Spatial-Temporal Weight Matrix

[![PyPI version](https://img.shields.io/badge/pypi-0.1.0-blue)](https://github.com/ZiningPe/STWM)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://github.com/ZiningPe/STWM)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ZiningPe/STWM)

## Introduction

STWM is a Python package for constructing and applying **Spatial-Temporal Weight Matrices** in spatial econometrics. Traditional spatial models use a static cross-sectional weight matrix (SWM) that captures only geographic proximity. STWM extends this by incorporating a **Time Weight Matrix (TWM)** that encodes how the strength of spatial dependence evolves over time, producing a joint STWM via the Kronecker product: `STWM = TWM ⊗ SWM`.

The TWM can be built from any of four data-driven temporal statistics — Moran's I, Geary's C, Getis-Ord G, or Spatial Gini — each capturing a distinct aspect of how spatial clustering changes year over year. A decay-based benchmark is also provided for robustness comparison. This framework is particularly well-suited to panels where regional interdependence intensifies or attenuates over time, such as green innovation spillovers, FDI flows, or pollution diffusion across provinces or countries.

The package ships four fully-specified spatial models (SLX, SAR, SEM, SDM) with support for panel Fixed Effects and Random Effects, five estimation methods (ML, QML, IV, GMM, Bayes), analytic effect decomposition via the Delta Method, and a complete suite of diagnostics: exogeneity tests, heteroskedasticity tests, rolling-window dynamics, robustness comparisons, and Monte Carlo validation.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concept: STWM = TWM x SWM](#core-concept-stwm--twm--swm)
4. [Model Reference](#model-reference)
5. [Detailed Examples](#detailed-examples)
   - [TWM Construction and Validation](#1-twm-construction-and-validation)
   - [STWM Assembly and Ordering Check](#2-stwm-assembly-and-ordering-check)
   - [All Four Models: Basic Usage](#3-all-four-models-basic-usage)
   - [Estimation Methods Comparison](#4-estimation-methods-comparison-sdm)
   - [Panel FE and RE](#5-panel-fixed-and-random-effects)
   - [Effect Decomposition and Delta Method SE](#6-effect-decomposition-and-delta-method-se)
   - [Exogeneity Tests](#7-exogeneity-tests)
   - [Robustness Checks](#8-robustness-checks)
   - [Dynamics: Rolling Effects and Shock Propagation](#9-dynamics-rolling-effects-and-shock-propagation)
   - [Monte Carlo Validation and Granger Test](#10-monte-carlo-validation-and-granger-test)
6. [API Reference](#api-reference)
7. [Dependencies](#dependencies)
8. [Technical Notes](#technical-notes)
9. [Citation](#citation)

---

## Installation

Install directly from GitHub (no PyPI release yet):

```bash
pip install git+https://github.com/ZiningPe/STWM.git
```

Or clone and install in editable mode for development:

```bash
git clone https://github.com/ZiningPe/STWM.git
cd STWM
pip install -e .
```

**Requirements:** Python 3.9+, NumPy >= 1.24, SciPy >= 1.10.

---

## Quick Start

```python
import numpy as np
import stwm

# Simulated data: 30 spatial units, 10 time periods
rng = np.random.default_rng(42)
n, T = 30, 10
SWM = np.random.rand(n, n); np.fill_diagonal(SWM, 0)
SWM /= SWM.sum(axis=1, keepdims=True)          # row-standardise

annual_data = [rng.standard_normal(n) for _ in range(T)]
morans = [stwm.compute_morans_i(y, SWM) for y in annual_data]

TWM  = stwm.build_twm_morans(morans)            # (10, 10) time weight matrix
STWM = stwm.build_stwm(TWM, SWM)               # (300, 300) STWM

Y = rng.standard_normal(n * T)
X = rng.standard_normal((n * T, 2))

model = stwm.SDMModel(STWM).fit(Y, X)          # Spatial Durbin Model
model.summary()                                 # prints effects table
```

---

## Core Concept: STWM = TWM x SWM

### The Kronecker construction

A standard Spatial Weight Matrix (SWM) of size `(n, n)` captures proximity between `n` spatial units at a single point in time. When working with panel data of `n` units over `T` periods, you need a weight matrix of size `(nT, nT)`. STWM fills this role by combining spatial and temporal proximity multiplicatively:

```
STWM = TWM x SWM        # np.kron(TWM, SWM)
```

The block structure means:

```
STWM[(t1*n):(t1+1)*n, (t2*n):(t2+1)*n]  =  TWM[t1, t2] * SWM
```

So the weight between observation `(unit i, period t1)` and observation `(unit j, period t2)` is exactly `TWM[t1, t2] * SWM[i, j]`: the product of temporal closeness and spatial closeness.

### Time-major stacking

The panel vector `Y` of length `nT` is stacked **time-major** (all units within a period before moving to the next period):

```
Y = [y_{1,1}, y_{2,1}, ..., y_{n,1},   <- all units at t=1
     y_{1,2}, y_{2,2}, ..., y_{n,2},   <- all units at t=2
     ...
     y_{1,T}, y_{2,T}, ..., y_{n,T}]   <- all units at t=T
```

This is the natural ordering for `np.kron(TWM, SWM)`. Using a different stacking order (e.g. unit-major) will silently produce wrong results; use `validate_stwm_ordering` to verify consistency.

### The Time Weight Matrix (TWM)

The TWM is a `(T, T)` lower-triangular matrix encoding **causal temporal dependence** (no future information leakage):

```
TWM[t, s] = a_t / a_s    for s < t   (past influences present)
TWM[t, t] = 1            (diagonal)
TWM[t, s] = 0            for s > t   (upper triangle = zero)
```

where `a_t` is a spatial statistic (Moran's I, Geary's C, etc.) computed from year `t`'s cross-sectional data. The ratio `a_t / a_s` captures how much the spatial clustering pattern at time `t` resembles that at time `s`. Ratios are winsorised and the matrix is row-standardised so each row sums to 1.

### Five TWM construction methods

| Method | Statistic | Economic interpretation |
|--------|-----------|------------------------|
| `build_twm_morans` | Moran's I | Global spatial clustering strength carry-forward |
| `build_twm_gearyc` | Geary's C (as 2-C) | Local spatial dissimilarity dynamics |
| `build_twm_getis_ord` | Getis-Ord G | Hot-spot concentration ratio |
| `build_twm_spatial_gini` | Spatial Gini | Regional inequality / convergence dynamics |
| `build_twm_decay` | exp / linear / power | Agnostic benchmark (no data required) |

---

## Model Reference

### Supported models

| Class | Full name | Spatial lag? | Spatial error? | WX lag? |
|-------|-----------|:---:|:---:|:---:|
| `SLXModel` | Spatial Lag of X | No | No | Yes |
| `SpatialLagModel` | SAR (Spatial Autoregressive) | Yes | No | No |
| `SpatialErrorModel` | SEM (Spatial Error Model) | No | Yes | No |
| `SDMModel` | SDM (Spatial Durbin Model) | Yes | No | Yes |

### Panel effects (all four models)

| `effects=` | Description | Transformation |
|------------|-------------|---------------|
| `'none'` | No panel correction (default) | Identity; constant added |
| `'fe'` | Fixed Effects (within estimator) | Time-major within-demeaning; no intercept |
| `'re'` | Random Effects | Swamy-Arora quasi-demeaning; `theta_re` reported |

Pass `n_units=n` whenever using `effects='fe'` or `effects='re'`.

### Estimation methods

| `method=` | Models | Description |
|-----------|--------|-------------|
| `'ml'` | SAR, SEM, SDM | Maximum Likelihood (Ord 1975, Lee 2004) -- default |
| `'qml'` | SAR, SDM | Quasi-ML (same optimiser, robust label) |
| `'iv'` | SDM | 2SLS with `[X, WX, W2X]` instruments (Anselin 1988) |
| `'gmm'` | SDM | Kelejian-Prucha GMM (1998/1999) |
| `'bayes'` | SAR, SDM | Three-block Gibbs sampler (LeSage 1997) |

SLX is always estimated by OLS. SEM is always ML.

### Model constructor and fit signature

```python
# All models share the same interface:
model = ModelClass(W)                   # W is the (nT, nT) STWM
model.fit(Y, X,
          effects='none',              # 'none' | 'fe' | 're'
          n_units=None,               # required for fe/re
          method='ml',                # method-specific (see table above)
          rho_bounds=(-0.99, 0.99),   # SAR/SDM/SEM optimisation bounds
          n_draws=1000,               # Bayes: posterior draws to keep
          n_burn=200)                 # Bayes: burn-in draws
model.summary()                       # prints table and returns dict
```

---

## Detailed Examples

### Setup: shared synthetic panel data

All examples below use the following synthetic dataset unless stated otherwise.

```python
import numpy as np
import stwm

rng = np.random.default_rng(0)
n, T = 30, 12          # 30 units, 12 years

# --- Spatial weight matrix (row-standardised contiguity-style) ---
SWM_raw = rng.random((n, n))
np.fill_diagonal(SWM_raw, 0)
SWM = SWM_raw / SWM_raw.sum(axis=1, keepdims=True)

# --- Annual cross-sectional data (for computing temporal statistics) ---
annual_Y = [rng.standard_normal(n) for _ in range(T)]

# --- Panel data (time-major: all n units per year, T years) ---
nT = n * T
Y  = rng.standard_normal(nT)
X  = rng.standard_normal((nT, 2))
```

---

### 1. TWM Construction and Validation

#### Computing temporal statistics from annual data

```python
# Compute all four statistics for each year in one call
stats_by_year = [stwm.compute_all_temporal_stats(y, SWM) for y in annual_Y]

morans_seq = np.array([s['morans_i']     for s in stats_by_year])
gearyc_seq = np.array([s['geary_c']      for s in stats_by_year])
getis_seq  = np.array([s['getis_g']      for s in stats_by_year])
gini_seq   = np.array([s['spatial_gini'] for s in stats_by_year])

print("Moran's I sequence:", morans_seq.round(4))
```

Or compute statistics individually:

```python
I_t  = stwm.compute_morans_i(annual_Y[5], SWM)      # single year
C_t  = stwm.compute_geary_c(annual_Y[5], SWM)
G_t  = stwm.compute_getis_ord_g(annual_Y[5], SWM)
SG_t = stwm.compute_spatial_gini(annual_Y[5], SWM)
```

#### Building TWMs from different statistics

```python
# Data-driven TWMs
TWM_I  = stwm.build_twm_morans(morans_seq)
TWM_C  = stwm.build_twm_gearyc(gearyc_seq)
TWM_G  = stwm.build_twm_getis_ord(getis_seq)
TWM_SG = stwm.build_twm_spatial_gini(gini_seq)

# Decay-based benchmark (no data required -- just T)
TWM_exp = stwm.build_twm_decay(T, decay_type='exponential', param=0.5)
TWM_lin = stwm.build_twm_decay(T, decay_type='linear',      param=0.3)
TWM_pow = stwm.build_twm_decay(T, decay_type='power',       param=1.0)
```

Tuning winsorisation (default: 95th percentile cap on ratios):

```python
# More aggressive winsorisation for volatile statistics
TWM_I_robust = stwm.build_twm_morans(morans_seq,
                                      winsorize_quantile=0.90,
                                      min_abs=0.01)
```

#### Admissibility check

Every TWM should be validated before use. `twm_stability_check` verifies non-negativity, row-standardisation, and the absence of NaN/Inf entries.

```python
check = stwm.twm_stability_check(TWM_I)
print(check)
# {'non_negative': True, 'row_standardised': True,
#  'spectral_radius': 1.0, 'has_nan_inf': False, 'passed': True}

assert check['passed'], f"TWM failed admissibility: {check}"
```

Note: The spectral radius of a row-standardised lower-triangular TWM is always 1.0 (Perron-Frobenius). The relevant spectral constraint is on the SAR/SDM parameter `rho`, not on the TWM itself.

---

### 2. STWM Assembly and Ordering Check

#### Building the STWM

```python
STWM = stwm.build_stwm(TWM_I, SWM)   # (nT, nT) = (360, 360)
print(STWM.shape)                      # (360, 360)
```

#### Summarising the STWM

```python
info = stwm.stwm_summary(STWM, n=n, T=T)
print(info)
# {'shape': (360, 360), 'n_spatial': 30, 'T_temporal': 12,
#  'sparsity': 0.4722, 'spectral_radius': 0.9801,
#  'weight_min': 0.0001, 'weight_max': 0.1423, 'weight_mean': 0.0317}
```

#### Validating the ordering convention

If you construct the STWM outside this package (e.g. in R or a custom function), use `validate_stwm_ordering` to confirm it is consistent with the expected time-major Kronecker convention before fitting any models.

```python
result = stwm.validate_stwm_ordering(STWM, SWM, TWM_I)
print(result['consistent'])   # True
print(result['message'])
# Consistent with kron(TWM, SWM)  [time-major ordering]

# If there is an inconsistency, block-level details are reported:
result2 = stwm.validate_stwm_ordering(STWM.T, SWM, TWM_I)  # wrong matrix
if not result2['consistent']:
    print(f"Max diff: {result2['max_abs_diff']}")
    print(f"Problematic blocks: {result2['block_errors'][:3]}")
```

The `stacking` key in the result always reminds you of the expected convention:

```
'time-major: W_full[t1*n:(t1+1)*n, t2*n:(t2+1)*n] = TWM[t1,t2]*SWM'
```

---

### 3. All Four Models: Basic Usage

All four model classes share the same `.fit(Y, X)` and `.summary()` interface.

```python
# SLX: Spatial Lag of X (OLS with spatial lags of regressors)
slx = stwm.SLXModel(STWM).fit(Y, X)
slx.summary()
# Prints: variable names, beta, theta (WX coefficients), direct, indirect,
#         t-statistics, p-values

# SAR: Spatial Autoregressive Model
sar = stwm.SpatialLagModel(STWM).fit(Y, X)
sar.summary()
# Prints: rho (spatial autocorrelation), beta, direct/indirect/total effects

# SEM: Spatial Error Model
sem = stwm.SpatialErrorModel(STWM).fit(Y, X)
sem.summary()
# Prints: lambda (spatial error autocorrelation), beta, direct = beta (no spillover)

# SDM: Spatial Durbin Model (most general)
sdm = stwm.SDMModel(STWM).fit(Y, X)
sdm.summary()
# Prints: rho, beta, theta (WX), direct/indirect/total effects + Delta Method SE
```

Each `.summary()` call both prints a formatted table and returns a dictionary with all estimates:

```python
res = sdm.summary()
print("rho:     ", res['rho'])
print("direct:  ", res['direct'])    # (k,) average direct effects
print("indirect:", res['indirect'])  # (k,) average indirect/spillover effects
print("total:   ", res['total'])     # (k,) total effects
print("sigma2:  ", res['sigma2'])
```

---

### 4. Estimation Methods Comparison (SDM)

SDM supports five estimation methods. This example fits the same dataset with all methods and compares `rho` estimates.

```python
methods = ['ml', 'qml', 'iv', 'gmm', 'bayes']
results = {}

for m in methods:
    res = stwm.SDMModel(STWM).fit(Y, X, method=m).summary()
    results[m] = {'rho': res.get('rho'), 'direct': res.get('direct')}

print(f"{'Method':<8}  {'rho':>7}  {'direct[0]':>10}")
for m, r in results.items():
    rho = r['rho'] if r['rho'] is not None else float('nan')
    d0  = r['direct'][0] if r['direct'] is not None else float('nan')
    print(f"{m:<8}  {rho:>7.4f}  {d0:>10.4f}")
```

Bayesian estimation with custom MCMC settings:

```python
res_bayes = stwm.SDMModel(STWM).fit(Y, X,
                                     method='bayes',
                                     n_draws=2000,
                                     n_burn=500).summary()
print("Posterior mean rho:", res_bayes['rho'])
```

SAR also supports `'ml'`, `'qml'`, and `'bayes'`:

```python
sar_bayes = stwm.SpatialLagModel(STWM).fit(Y, X, method='bayes',
                                             n_draws=1000, n_burn=200)
sar_bayes.summary()
```

---

### 5. Panel Fixed and Random Effects

All four models accept `effects='fe'` or `effects='re'` together with `n_units=n`.

```python
# Fixed Effects (within estimator -- time-major demeaning, no intercept)
sdm_fe = stwm.SDMModel(STWM).fit(Y, X, effects='fe', n_units=n)
sdm_fe.summary()
# The printed table will show "[SDM, FE]" in the header

# Random Effects (Swamy-Arora quasi-demeaning)
sdm_re = stwm.SDMModel(STWM).fit(Y, X, effects='re', n_units=n)
sdm_re.summary()
# Reports theta_re (quasi-demeaning weight, 0 = OLS, 1 = within)
```

FE and RE are available for all models:

```python
sar_fe  = stwm.SpatialLagModel(STWM).fit(Y, X, effects='fe', n_units=n)
sar_re  = stwm.SpatialLagModel(STWM).fit(Y, X, effects='re', n_units=n)
sem_fe  = stwm.SpatialErrorModel(STWM).fit(Y, X, effects='fe', n_units=n)
slx_re  = stwm.SLXModel(STWM).fit(Y, X, effects='re', n_units=n)
```

Comparing pooled vs. FE vs. RE `rho` estimates:

```python
for eff in ['none', 'fe', 're']:
    res = stwm.SDMModel(STWM).fit(Y, X, effects=eff, n_units=n).summary()
    print(f"effects={eff:<4}:  rho={res['rho']:.4f}")
```

---

### 6. Effect Decomposition and Delta Method SE

For spatial models with a spatial lag (SAR, SDM), raw coefficients `beta` do not directly measure causal effects because of the spatial multiplier `(I - rho*W)^{-1}`. The correct decomposition (Elhorst 2014, Table 2.1) is:

| Effect | Formula | Interpretation |
|--------|---------|----------------|
| Direct | `(1/N) tr[S_k(W)]` | Average own-unit effect of a unit change in `x_k` |
| Indirect | `Total - Direct` | Average spillover to other units (feedback via W) |
| Total | `(1/N) 1' S_k(W) 1` | Average total impact including spatial feedback |

where `S_k(W) = (I - rho*W)^{-1} * (beta_k * I + theta_k * W)`.

Standard errors are computed analytically by the **Delta Method** using the full joint ML information-matrix covariance (Anselin 1988, eq. 6.5). This correctly propagates the `rho-beta` cross-terms that a block-diagonal covariance would miss.

```python
res = stwm.SDMModel(STWM).fit(Y, X).summary()

# Effect estimates and inference
print("Direct effects:  ", res['direct'])
print("Indirect effects:", res['indirect'])
print("Total effects:   ", res['total'])
print()
print("Direct SE:       ", res['direct_se'])
print("Indirect SE:     ", res['indirect_se'])
print("Direct t-stats:  ", res['direct_t'])
print("Indirect p-vals: ", res['indirect_p'])
```

For SLX, effects are exact (OLS, no multiplier needed):

```python
slx_res = stwm.SLXModel(STWM).fit(Y, X).summary()
# direct[k] = beta[k]   (own-unit coefficient)
# indirect[k] = theta[k] (WX coefficient = spillover)
```

For SEM, there are no spillovers: `direct = beta`, `indirect = 0`.

---

### 7. Exogeneity Tests

The STWM uses temporal statistics as instrument-like weights. The `stwm_exogeneity_report` function runs a three-test battery to verify that the proposed TWM is exogenous with respect to the error term:

1. **Hausman test** -- compares IV(geographic `W_g`) vs. IV(STWM) coefficient vectors. A negative statistic is a strong failure to reject H0 (Baltagi 2021).
2. **Sargan J-test** -- over-identification test for the geographic instruments.
3. **Redundancy F-test** -- tests whether STWM lags add independent variation beyond `W_g`.

```python
# Construct a geographic inverse-distance benchmark matrix
coords = rng.random((n, 2))
dist = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
np.fill_diagonal(dist, np.inf)
W_g_raw = 1.0 / dist
np.fill_diagonal(W_g_raw, 0)
W_g = W_g_raw / W_g_raw.sum(axis=1, keepdims=True)

# Expand W_g to panel size (block-diagonal / Kronecker with identity time)
I_T = np.eye(T)
W_g_panel = np.kron(I_T, W_g)

# Run the full exogeneity battery
report = stwm.stwm_exogeneity_report(Y, X, STWM, W_g_panel, label_tw='TW_Morans')
# Prints formatted test output with conclusions

# Individual tests are also accessible:
haus = stwm.hausman_tw_exogeneity(Y, X, STWM, W_g_panel)
print("Hausman H:", haus['H_stat'], "  p:", haus['p_value'])
print(haus['conclusion'])

# IV regression directly
WY = STWM @ Y
X_full = np.column_stack([np.ones(nT), X, WY])
Z_g    = np.column_stack([np.ones(nT), X,
                           W_g_panel @ np.column_stack([np.ones(nT), X])])
iv_res = stwm.iv_regression(Y, X_full, Z_g)
print("2SLS beta:", iv_res['beta'])
print("First-stage F:", iv_res['first_stage_F'])   # rule of thumb: F > 10

# Sargan over-identification
sargan = stwm.sargan_test(Y, X_full, Z_g, iv_res['beta'])
print("Sargan J:", sargan['J_stat'], "  p:", sargan['p_value'])

# Redundancy: does STWM add variation beyond W_g?
TWX    = STWM @ X
W_g_X  = W_g_panel @ X
redund = stwm.redundancy_test(Y, W_g_X, TWX)
print("Redundancy F:", redund['F_stat'], "  p:", redund['p_value'])
```

---

### 8. Robustness Checks

#### Compare multiple weight matrices by model fit (AIC/BIC)

```python
weight_matrices = {
    'STWM_Morans': stwm.build_stwm(TWM_I,   SWM),
    'STWM_GearyC': stwm.build_stwm(TWM_C,   SWM),
    'STWM_GetisG': stwm.build_stwm(TWM_G,   SWM),
    'STWM_Gini'  : stwm.build_stwm(TWM_SG,  SWM),
    'Decay_Exp'  : stwm.build_stwm(TWM_exp, SWM),
    'Decay_Lin'  : stwm.build_stwm(TWM_lin, SWM),
}

comparison = stwm.compare_weight_matrices(Y, X, weight_matrices, stwm.SDMModel)
table = comparison['comparison_table']

print(f"{'Matrix':<16}  {'AIC':>10}  {'BIC':>10}  {'LogLik':>10}")
for label in weight_matrices:
    aic = table['AIC'][label]
    bic = table['BIC'][label]
    ll  = table['LogLik'][label]
    print(f"{label:<16}  {aic!r:>10}  {bic!r:>10}  {ll!r:>10}")
```

#### Sensitivity to TWM hyperparameters

```python
sensitivity = stwm.sensitivity_report(
    Y, X, SWM, morans_seq, stwm.SDMModel,
    winsorize_quantiles=[0.90, 0.95, 0.99],
    min_abs_values=[1e-4, 1e-3, 1e-2]
)

# Check stability of indirect effects across hyperparameter grid
for q_key, inner in sensitivity.items():
    for ma_key, res in inner.items():
        if 'error' not in res:
            ind = res['indirect'][0]
            rho = res['rho']
            print(f"{q_key} / {ma_key}:  indirect[0]={ind:.4f}  rho={rho:.4f}")
```

#### Rolling window estimation

```python
roll = stwm.rolling_window_estimation(
    Y, X, SWM, morans_seq,
    n=n, T=T, window=6,
    ModelClass=stwm.SDMModel,
    twm_builder=stwm.build_twm_morans
)

for window_label, res in roll.items():
    if 'error' not in res:
        print(f"{window_label}: rho={res['rho']:.4f}")
```

#### Heteroskedasticity tests

```python
X_with_const = np.column_stack([np.ones(nT), X])

bp_test = stwm.heteroskedasticity_test(Y, X_with_const, test='breusch_pagan')
print(f"Breusch-Pagan: stat={bp_test['statistic']}, p={bp_test['p_value']}")
print(bp_test['conclusion'])

white_test = stwm.heteroskedasticity_test(Y, X_with_const, test='white')
print(f"White:         stat={white_test['statistic']}, p={white_test['p_value']}")
```

---

### 9. Dynamics: Rolling Effects and Shock Propagation

#### Rolling direct/indirect effects over time

```python
roll_dyn = stwm.rolling_effects(
    Y, X,
    TWM_sequence=TWM_I,
    W_spatial=SWM,
    n=n, T=T,
    window=5,
    ModelClass=stwm.SDMModel,
    var_names=['x1', 'x2']
)

# Print how effects of x1 change across windows
stwm.print_rolling_effects(roll_dyn, var_idx=0)

# Summarise stability (coefficient of variation across windows)
stability = stwm.coefficient_stability(roll_dyn)
print("x1 indirect CV:", stability['x1']['indirect_cv'])
```

#### Regional effect comparison

```python
region_groups = {
    'East'   : list(range(0,  10)),
    'Central': list(range(10, 20)),
    'West'   : list(range(20, 30)),
}

reg_res = stwm.regional_effects(
    Y, X, STWM, n=n, T=T,
    region_groups=region_groups,
    ModelClass=stwm.SDMModel,
    var_names=['x1', 'x2']
)

stwm.print_regional_comparison(reg_res, var_idx=0)
```

#### Unit shock propagation

```python
res = stwm.SDMModel(STWM).fit(Y, X).summary()

prop = stwm.unit_shock_propagation(
    STWM, n=n, T=T,
    rho=res['rho'],
    beta_k=res['direct'][0],
    source_units=[0, 1, 2],     # shock originates in units 0, 1, 2
    t_shock=0                   # at time period 0
)

print("Impact in shock period (all n units):")
print(prop['impact_t0'].round(4))
print("Average direct effect: ", prop['total_direct'])
print("Average indirect effect:", prop['total_indirect'])
```

#### Temporal and regional subgroups

```python
# Temporal subgroups (e.g. pre/post policy split)
temporal_res = stwm.temporal_subgroup(
    Y, X, SWM,
    n=n, T=T,
    periods=[(0, 5), (6, 11)],
    ModelClass=stwm.SDMModel
)
for period, res in temporal_res.items():
    print(f"Period {period}: rho={res['rho']:.4f}")

# Regional subgroups by integer label
region_labels = np.array([0]*10 + [1]*10 + [2]*10)
reg_sub = stwm.regional_subgroup(Y, X, SWM,
                                   region_labels=region_labels,
                                   ModelClass=stwm.SDMModel,
                                   T=T)
for reg, res in reg_sub.items():
    print(f"Region {reg}: rho={res['rho']:.4f}")
```

---

### 10. Monte Carlo Validation and Granger Test

#### Monte Carlo finite-sample properties

```python
mc = stwm.monte_carlo_stwm(
    true_rho=0.4,
    true_beta=np.array([1.0, -0.5]),
    W_spatial=SWM,
    TWM=TWM_I,
    n=n, T=T,
    n_simulations=300,
    ModelClass=stwm.SpatialLagModel,
    seed=42
)

print(f"rho:  bias={mc['bias_rho']:.4f}  RMSE={mc['rmse_rho']:.4f}  "
      f"coverage={mc['coverage_rho']:.4f}")
print(f"beta: bias={mc['bias_beta'].round(4)}  RMSE={mc['rmse_beta'].round(4)}")
```

#### Granger causality from Moran's I to spillovers

This test asks whether changes in Moran's I statistically precede changes in estimated spillover parameters. A significant result validates the use of autocorrelation statistics as the basis for the TWM.

```python
# First obtain a time series of rho estimates from rolling estimation
roll_annual = stwm.rolling_window_estimation(
    Y, X, SWM, morans_seq,
    n=n, T=T, window=3,
    ModelClass=stwm.SpatialLagModel,
    twm_builder=stwm.build_twm_morans
)
rho_series = np.array([
    roll_annual[k]['rho']
    for k in sorted(roll_annual)
    if isinstance(roll_annual.get(k), dict) and 'rho' in roll_annual[k]
])

# Granger test: does Moran's I Granger-cause rho?
granger = stwm.granger_spillover_test(
    morans_sequence=morans_seq[:len(rho_series)],
    spillover_sequence=rho_series,
    max_lag=3
)

for lag, res in granger.items():
    print(res['conclusion'])
```

---

## API Reference

### `stwm.temporal_weights`

| Function | Returns | Description |
|----------|---------|-------------|
| `compute_morans_i(y, W)` | `float` | Global Moran's I for one cross-section |
| `compute_geary_c(y, W)` | `float` | Global Geary's C |
| `compute_getis_ord_g(y, W)` | `float` | Global Getis-Ord G |
| `compute_spatial_gini(y, W)` | `float` | Spatial Gini coefficient |
| `compute_all_temporal_stats(y, W)` | `dict` | All four statistics in one call |
| `build_twm_morans(seq, winsorize_quantile, min_abs)` | `(T,T) ndarray` | TWM from Moran's I sequence |
| `build_twm_gearyc(seq, winsorize_quantile, min_abs)` | `(T,T) ndarray` | TWM from Geary's C sequence |
| `build_twm_getis_ord(seq, winsorize_quantile, min_abs)` | `(T,T) ndarray` | TWM from Getis-Ord G sequence |
| `build_twm_spatial_gini(seq, winsorize_quantile, min_abs)` | `(T,T) ndarray` | TWM from Spatial Gini sequence |
| `build_twm_decay(T, decay_type, param)` | `(T,T) ndarray` | Decay-based benchmark TWM |
| `twm_stability_check(TWM, rho_max)` | `dict` | Admissibility check: non-negativity, row-std, NaN |
| `validate_stwm_ordering(W_full, SWM, TWM, tol)` | `dict` | Verify time-major Kronecker convention |

### `stwm.stwm_core`

| Function | Returns | Description |
|----------|---------|-------------|
| `build_stwm(TWM, SWM, row_standardize)` | `(nT,nT) ndarray` | Assemble STWM via Kronecker product |
| `stwm_summary(STWM, n, T)` | `dict` | Shape, sparsity, spectral radius, weight stats |

### `stwm.models`

| Class / Function | Description |
|------------------|-------------|
| `SLXModel(W).fit(Y, X, effects, n_units)` | Spatial Lag of X (OLS) |
| `SpatialLagModel(W).fit(Y, X, method, effects, n_units, n_draws, n_burn)` | SAR: ML / QML / Bayes |
| `SpatialErrorModel(W).fit(Y, X, effects, n_units)` | SEM: ML |
| `SDMModel(W).fit(Y, X, method, effects, n_units, n_draws, n_burn)` | SDM: ML / QML / IV / GMM / Bayes |
| `print_effects_table(res)` | Print formatted direct / indirect / total table |

All `.summary()` methods return a `dict` with keys: `rho` (or `lam`), `beta`, `se`, `t_stats`, `p_values`, `direct`, `indirect`, `total`, `direct_se`, `indirect_se`, `total_se`, `direct_t`, `indirect_t`, `total_t`, `direct_p`, `indirect_p`, `total_p`, `sigma2`, `model`, `method`, `effects`, `theta_re`.

### `stwm.endogeneity`

| Function | Returns | Description |
|----------|---------|-------------|
| `iv_regression(Y, X, Z)` | `dict` | 2SLS with first-stage F statistic |
| `hausman_test(beta_e, beta_c, cov_e, cov_c)` | `dict` | Hausman chi-squared test |
| `hausman_tw_exogeneity(Y, X, W_stwm, W_g)` | `dict` | Hausman test via IV comparison |
| `sargan_test(Y, X, Z, beta_iv)` | `dict` | Sargan-Hansen J over-identification test |
| `redundancy_test(Y, X_base, X_extra)` | `dict` | F-test: does STWM add variation beyond W_g? |
| `stwm_exogeneity_report(Y, X, W_stwm, W_g, label_tw)` | `dict` | Full three-test battery with printed output |

### `stwm.heterogeneity`

| Function | Returns | Description |
|----------|---------|-------------|
| `heteroskedasticity_test(Y, X, test)` | `dict` | Breusch-Pagan or White test |
| `regional_subgroup(Y, X, W, region_labels, ModelClass, T)` | `dict` | Fit model per geographic group |
| `temporal_subgroup(Y, X, W, n, T, periods, ModelClass)` | `dict` | Fit model per time sub-period |

### `stwm.dynamics`

| Function | Returns | Description |
|----------|---------|-------------|
| `rolling_effects(Y, X, TWM_sequence, W_spatial, n, T, window, ModelClass, var_names)` | `dict` | Effects on sliding time windows |
| `print_rolling_effects(results, var_idx)` | None | Print rolling effects table for one variable |
| `regional_effects(Y, X, STWM, n, T, region_groups, ModelClass, var_names)` | `dict` | Effects per region group |
| `print_regional_comparison(results, var_idx)` | None | Print regional comparison table |
| `unit_shock_propagation(STWM, n, T, rho, beta_k, source_units, t_shock)` | `dict` | Spatial propagation of a unit shock |
| `coefficient_stability(rolling_results, var_names)` | `dict` | CV and range of effects across windows |

### `stwm.robustness`

| Function | Returns | Description |
|----------|---------|-------------|
| `compare_weight_matrices(Y, X, weight_matrices, ModelClass, metrics)` | `dict` | AIC / BIC / LogLik comparison across W matrices |
| `rolling_window_estimation(Y, X, W_spatial, morans_seq, n, T, window, ModelClass, twm_builder)` | `dict` | Rolling estimation with automatic STWM rebuilding |
| `sensitivity_report(Y, X, W_spatial, morans_seq, ModelClass, winsorize_quantiles, min_abs_values)` | `dict` | Sensitivity to TWM hyperparameters |

### `stwm.simulation`

| Function | Returns | Description |
|----------|---------|-------------|
| `monte_carlo_stwm(true_rho, true_beta, W_spatial, TWM, n, T, n_simulations, ModelClass, seed)` | `dict` | Monte Carlo validation: bias, RMSE, coverage |
| `granger_spillover_test(morans_sequence, spillover_sequence, max_lag)` | `dict` | Granger F-test: Moran's I -> spillover parameters |

---

## Dependencies

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| `numpy` | 1.24 | Array operations, Kronecker product, eigenvalue decomposition |
| `scipy` | 1.10 | Scalar optimisation (`scipy.optimize`), statistical distributions |

No other mandatory dependencies. Optional: `matplotlib` for custom visualisation of rolling effects or shock propagation maps.

---

## Technical Notes

### Eigenvalue computation for log-determinant

For non-symmetric weight matrices (which STWM typically is), eigenvalues may be complex. The log-determinant is computed via:

```
log|det(I - rho*W)| = Re( sum_i log(1 - rho * lambda_i) )
```

where `lambda_i` are the full complex eigenvalues of `W`. This is correct for non-symmetric matrices. Using only `Re(lambda_i)` before taking the log yields incorrect results for STWM.

### Negative Moran's I

When Moran's I is negative (indicating spatial dispersion), the ratio `I_t / I_s` is taken and then `abs()` is applied before row-standardisation. This preserves the magnitude of the dispersion signal rather than treating a sign flip as meaningless. The `min_abs` guard (default 1e-3) prevents explosion of the ratio when the denominator is near zero.

### Delta Method vs. Monte Carlo

Effect standard errors are computed analytically using the Delta Method with the full joint ML information-matrix covariance (Anselin 1988, eq. 6.5). This is:

- Exactly reproducible (no simulation variance)
- Approximately 17 times faster than D=1000 Monte Carlo draws
- Correct: the `rho-beta` cross-terms in the information matrix are included, which a block-diagonal approximation would miss

### Panel FE within-demeaning

Fixed Effects demeaning is applied in time-major order: for each spatial unit `i`, subtract the unit's time mean from all observations. This is equivalent to `M_FE @ arr` where `M_FE = I_{NT} - (J_T/T x I_n)`. The FE transformation absorbs the unit intercept, so no constant is added in the regression. Panel RE uses the Swamy-Arora estimator to compute the quasi-demeaning weight `theta_re`, which is reported in `.summary()`.

### Admissibility conditions (Elhorst 2014, Section 2.2)

A valid weight matrix must satisfy: (1) all entries >= 0, (2) each row sums to 1 (row-standardisation), (3) no NaN or Inf entries. The spectral radius of the STWM must be less than `1/|rho_max|` for the spatial multiplier `(I - rho*W)^{-1}` to exist. `stwm_summary` reports the spectral radius of the assembled STWM to allow manual verification.

---

## Citation

If you use STWM in published research, please cite:

```bibtex
@software{stwm2025,
  author  = {Zining, Pe},
  title   = {{STWM}: Spatial-Temporal Weight Matrix for Panel Econometrics},
  year    = {2025},
  url     = {https://github.com/ZiningPe/STWM},
  version = {0.1.0}
}
```

Please also cite the methodological references underlying the key components:

- **Moran's I**: Moran, P. A. P. (1950). Notes on Continuous Stochastic Phenomena. *Biometrika*, 37, 17-23.
- **Geary's C**: Geary, R. C. (1954). The Contiguity Ratio and Statistical Mapping. *The Incorporated Statistician*, 5(3), 115-145.
- **Getis-Ord G**: Getis, A., & Ord, J. K. (1992). The Analysis of Spatial Association by Use of Distance Statistics. *Geographical Analysis*, 24(3), 189-206.
- **Spatial models / effect decomposition**: Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
- **Effect inference**: LeSage, J., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- **ML covariance**: Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic Publishers.
- **Bayesian SAR**: LeSage, J. P. (1997). Bayesian Estimation of Spatial Autoregressive Models. *International Regional Science Review*, 20(1-2), 113-129.
- **Hausman test**: Hausman, J. A. (1978). Specification Tests in Econometrics. *Econometrica*, 46(6), 1251-1271.
- **IV exogeneity**: Cheng, Z., & Lee, L.-F. (2017). Uniform inference in panel autoregressive models. *Journal of Econometrics*, 196(1), 13-36.
- **Kelejian-Prucha GMM**: Kelejian, H. H., & Prucha, I. R. (1999). A Generalized Moments Estimator for the Autoregressive Parameter in a Spatial Model. *International Economic Review*, 40(2), 509-533.

---

*This package was developed as a research tool for spatial panel econometrics. For bug reports, feature requests, or questions, please open an issue at https://github.com/ZiningPe/STWM/issues.*
