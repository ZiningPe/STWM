# stwm — Stata Package

Stata interface for the **Spatial-Temporal Weight Matrix (STWM)** Python
package.  All heavy computation is delegated to the `stwm` Python library via
Stata 16+'s built-in Python integration.

---

## Requirements

| Requirement | Version |
|---|---|
| Stata | 16.0 or later |
| Python | 3.8 or later (configured in Stata via `set python_exec`) |
| Python package | `stwm` (install with `pip install stwm`) |
| Python package | `numpy`, `scipy`, `pandas` (installed as dependencies) |

Configure your Python path in Stata once:

```stata
set python_exec "/path/to/python3"   // e.g. /usr/local/bin/python3
python which stwm                    // verify the package is found
```

---

## Installation

### Option A — copy files manually

1. Download or clone this repository.
2. Copy the contents of `ado/` and `help/` to your personal ado-path.
   In Stata: `sysdir` shows your `PERSONAL` directory.

```stata
. sysdir
. adopath + "/path/to/stata-pkg/ado"
. adopath + "/path/to/stata-pkg/help"
```

### Option B — `net install` from GitHub

```stata
net install stwm, from("https://raw.githubusercontent.com/wangzining/stwm/main/stata-pkg")
```

---

## Quick Start

```stata
* 1. Sort data in time-major order (all units for year 1, then year 2, ...)
sort year citycode

* 2. Build the Spatial-Temporal Weight Matrix
stwm_build lnpatent, swm("swm.dta") n(276) t(14) decay(morans) saveas(STWM_I)

* 3. Fit the Spatial Durbin Model
stwm_sdm lnpatent lnrd lnfdi lnhuman lninfra lngdp lnpop, w(STWM_I)

* 4. View direct / indirect / total effects
estat effects
```

---

## Commands

### `stwm_build` — Build STWM

```stata
stwm_build varname [if] [in], swm(filename) n(#) t(#) [decay(string) param(#) saveas(name)]
```

Constructs STWM = TWM ⊗ SWM via Kronecker product.

| `decay()` option | Description |
|---|---|
| `morans` (default) | Moran's I ratio: TWM[t,s] = \|I_t / I_s\| |
| `gearyc` | Geary's C transform ratio |
| `getis` | Getis-Ord G ratio |
| `gini` | Spatial Gini coefficient ratio |
| `exponential` | Parametric: exp(−λ(t−s)) |
| `linear` | Parametric: max(0, 1 − λ(t−s)) |
| `power` | Parametric: (t−s)^(−λ) |

After `stwm_build`, the matrix is accessible to all subsequent commands via
the global macro `${STWM_I_path}`.

---

### `stwm_sdm` — Spatial Durbin Model

```stata
stwm_sdm depvar indepvars [if] [in], w(name) [method(ml|qml|iv|gmm|bayes) effects(none|fe|re) n(#)]
```

Fits **Y = δ·W·Y + X·β + W·X·θ + ε**.  Results stored in `e()`.

```stata
estat effects   // direct / indirect / total table
test lnrd       // Wald test
lincom lnrd + W_lnrd   // linear combination
```

---

### `stwm_sar` — Spatial Autoregressive Model

```stata
stwm_sar depvar indepvars [if] [in], w(name) [effects(none|fe|re) n(#)]
```

---

### `stwm_sem` — Spatial Error Model

```stata
stwm_sem depvar indepvars [if] [in], w(name) [effects(none|fe|re) n(#)]
```

---

### `stwm_slx` — Spatial Lag of X Model

```stata
stwm_slx depvar indepvars [if] [in], w(name) [effects(none|fe|re) n(#)]
```

---

### `stwm_hausman` — Exogeneity Test Battery

```stata
stwm_hausman depvar indepvars [if] [in], w(name) wg(name)
```

Runs Hausman + Sargan J + Redundancy F tests.
`wg(name)` is an exogenous benchmark STWM (e.g., static inverse-distance).

```stata
return list   // r(hausman_H), r(sargan_J), r(redund_F), ...
```

---

### `stwm_sensitivity` — Hyperparameter Sensitivity Grid

```stata
stwm_sensitivity depvar indepvars [if] [in], swm(filename) n(#) t(#) [winsqlist(numlist) malist(numlist)]
```

Rebuilds STWM and refits SDM for every combination of
`winsorize_quantile × min_abs`.  Reports CV(ρ); CV < 0.05 → stable.

---

### `stwm_mc` — Monte Carlo Validation

```stata
stwm_mc, w(name) n(#) t(#) [truerho(#) nsim(#) seed(#) ndemo(#)]
```

Simulates from the SAR DGP and reports Bias, RMSE, Coverage(95%) for ρ.

---

### `stwm_granger` — Granger Causality Test

```stata
stwm_granger depvar [if] [in], w(name) n(#) t(#) [maxlag(#) window(#)]
```

Tests whether annual Moran's I Granger-causes rolling spillover parameter
estimates, structurally justifying the Moran-I ratio TWM construction.

---

## Data Format

Data must be in **time-major (long) format**:

```
year  citycode  lnpatent  lnrd  ...
2001  1         ...
2001  2         ...
...
2001  276       ...
2002  1         ...
...
2014  276       ...
```

Always run `sort year citycode` before `stwm_build`.

---

## Typical Workflow

```stata
* 0. Prepare data
sort year citycode

* 1. Build multiple STWMs
stwm_build lnpatent, swm("swm.dta") n(276) t(14) decay(morans) saveas(STWM_I)
stwm_build lnpatent, swm("swm_geo.dta") n(276) t(14) decay(exponential) param(1) saveas(STWM_G)

* 2. Main estimation
stwm_sdm lnpatent lnrd lnfdi lnhuman lninfra lngdp lnpop, w(STWM_I)
estat effects

* 3. Robustness — panel FE
stwm_sdm lnpatent lnrd lnfdi lnhuman lninfra lngdp lnpop, w(STWM_I) effects(fe) n(276)
estat effects

* 4. Exogeneity battery
stwm_hausman lnpatent lnrd lnfdi lnhuman lninfra lngdp lnpop, w(STWM_I) wg(STWM_G)

* 5. Sensitivity analysis
stwm_sensitivity lnpatent lnrd lnfdi lnhuman, swm("swm.dta") n(276) t(14)

* 6. Monte Carlo validation
stwm_mc, w(STWM_I) n(276) t(14) truerho(0.3) nsim(200) ndemo(50)

* 7. Granger causality justification
stwm_granger lnpatent, w(STWM_I) n(276) t(14)
```

---

## Citation

If you use this package, please cite:

> Wang, Z. (2026). *Spatial-Temporal Weight Matrix (STWM): A Data-Driven
> Approach to Temporal Weighting in Spatial Panel Models.*
> [GitHub: wangzining/stwm]
