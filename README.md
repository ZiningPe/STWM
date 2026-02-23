# STWM: Dynamic Spatial-Temporal Weight Matrices

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python package for constructing, validating, and applying **dynamic Spatial-Temporal Weight Matrices (STWMs)** in spatial econometric analysis.

## Core Idea

Rather than imposing arbitrary temporal decay functions, STWM derives temporal propagation "memory" directly from observed spatial autocorrelation statistics (Moran's I or Geary's C). The time weight matrix (TWM) and spatial weight matrix (SWM) are combined via Kronecker product:

```
STWM = TWM ⊗ SWM
```

This simultaneously encodes spatial proximity and temporal carry-forward structure in a single (nT × nT) matrix, enabling spatial panel models to capture the evolution of spatial spillovers over time.

## Installation

```bash
git clone https://github.com/ZiningPe/STWM.git
cd STWM
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from stwm import (
    geographic_swm, build_twm_morans, build_stwm,
    twm_stability_check, SDMModel,
    compare_weight_matrices, monte_carlo_stwm, granger_spillover_test,
)

# 1. Spatial weight matrix from coordinates (lon, lat)
coords = np.array([[lon1, lat1], [lon2, lat2], ...])
W = geographic_swm(coords, method="inverse_distance")

# 2. Time weight matrix from annual Moran's I sequence
morans_i = np.array([0.31, 0.34, 0.29, ...])     # one value per year
TWM = build_twm_morans(morans_i)
print(twm_stability_check(TWM))                    # verify admissibility

# 3. Assemble STWM
STWM = build_stwm(TWM, W)                         # shape: (nT, nT)

# 4. Estimate SDM with direct/indirect/total effect decomposition
model = SDMModel(STWM).fit(Y, X)
res   = model.summary()
print("Direct effects:",   res["direct"])
print("Indirect effects:", res["indirect"])
print("Total effects:",    res["total"])
```

## Feature Overview

| Module | Key functions | Description |
|---|---|---|
| `spatial_weights` | `geographic_swm`, `economic_swm`, `nested_swm` | Build SWM from coordinates, GDP, or a combination |
| `temporal_weights` | `build_twm_morans`, `build_twm_gearyc`, `build_twm_decay` | Build TWM from autocorrelation series or decay functions |
| `stwm_core` | `build_stwm`, `stwm_summary` | Kronecker product assembly; diagnostics |
| `models` | `SLXModel`, `SpatialLagModel`, `SpatialErrorModel`, `SDMModel` | SAR / SEM / SDM / SLX with effect decomposition |
| `endogeneity` | `iv_regression`, `hausman_test`, `endogeneity_report` | 2SLS and Hausman specification test |
| `heterogeneity` | `regional_subgroup`, `temporal_subgroup`, `heteroskedasticity_test` | Subgroup analysis; Breusch-Pagan / White tests |
| `robustness` | `compare_weight_matrices`, `rolling_window_estimation`, `sensitivity_report` | AIC/BIC comparison; rolling window; hyperparameter sensitivity |
| `simulation` | `monte_carlo_stwm`, `granger_spillover_test` | Monte Carlo validation; Granger causality |

## Detailed Examples

### Admissibility check with stability conditions

```python
from stwm import build_twm_morans, twm_stability_check

TWM   = build_twm_morans(morans_i, winsorize_quantile=0.95, min_abs=1e-3)
check = twm_stability_check(TWM, rho_max=0.99)
# Returns: spectral_radius, non_negative, row_standardised, stable, passed
```

### Compare STWM against decay-based alternatives

```python
from stwm import build_twm_decay, build_stwm, compare_weight_matrices, SDMModel

matrices = {
    "STWM_Morans"  : build_stwm(build_twm_morans(morans_i), W),
    "Decay_Exp_0.3": build_stwm(build_twm_decay(T, "exponential", 0.3), W),
    "Decay_Lin_0.2": build_stwm(build_twm_decay(T, "linear", 0.2), W),
    "Static_W"     : build_stwm(build_twm_decay(T, "linear", 0.0), W),
}
report = compare_weight_matrices(Y, X, matrices, SDMModel)
print(report["comparison_table"])   # AIC, BIC, LogLik for each
```

### Endogeneity test (IV + Hausman)

```python
from stwm import endogeneity_report

report = endogeneity_report(Y, X_ols_dict, X_iv_dict, Z_instruments,
                             model_names=["SAR", "SEM", "SDM"])
for name, res in report.items():
    print(name, "→", res["hausman"]["conclusion"])
```

### Granger causality: Moran's I → spillover parameters

```python
from stwm import granger_spillover_test

# spillover_sequence: annual indirect effects from rolling-window estimation
result = granger_spillover_test(morans_i, spillover_sequence, max_lag=3)
for lag, res in result.items():
    print(res["conclusion"])
```

### Monte Carlo validation

```python
from stwm import monte_carlo_stwm

mc = monte_carlo_stwm(
    true_rho=0.4, true_beta=np.array([0.5, -0.3, 0.2]),
    W_spatial=W, TWM=TWM, n=n, T=T, n_simulations=1000
)
print(f"Bias(ρ) = {mc['bias_rho']:.4f}")
print(f"RMSE(ρ) = {mc['rmse_rho']:.4f}")
print(f"Coverage = {mc['coverage_rho']:.2%}")
```

### Regional and temporal heterogeneity

```python
from stwm import regional_subgroup, temporal_subgroup, SDMModel

# Separate estimation per geographic region
region_results = regional_subgroup(Y, X, W, region_labels, SDMModel, T=T)

# Separate estimation per time sub-period
period_results = temporal_subgroup(Y, X, W, n=n, T=T,
                                   periods=[(0, 4), (5, 9), (10, 13)],
                                   ModelClass=SDMModel)
```

## Dependencies

- `numpy >= 1.24`
- `scipy >= 1.10`

Optional (for examples and plotting): `matplotlib`, `pandas`, `jupyter`

## Citation

If you use this package, please cite:

```
TBD. STWM: Dynamic Spatial-Temporal Weight Matrices (Python package).
GitHub: https://github.com/ZiningPe/STWM
```

## License

MIT
