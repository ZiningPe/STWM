# STWM Package — Mathematical Formulas Reference

All formulas implemented in the STWM package, organised by module.

---

## 1. Spatial Autocorrelation Statistics (`temporal_weights.py`)

### Moran's I

$$I = \frac{n}{S_0} \cdot \frac{\mathbf{z}' W \mathbf{z}}{\mathbf{z}' \mathbf{z}}$$

where:
- $\mathbf{z} = \mathbf{y} - \bar{y}$ (mean-centred variable)
- $S_0 = \sum_{i} \sum_{j} w_{ij}$ (sum of all weights)
- $n$ = number of observations
- Range: approximately $[-1, 1]$; $I > 0$ indicates positive spatial autocorrelation

### Geary's C

$$C = \frac{(n-1) \sum_{i}\sum_{j} w_{ij}(y_i - y_j)^2}{2 S_0 \sum_{i}(y_i - \bar{y})^2}$$

- Range: $(0, 2)$; $C < 1$ indicates positive spatial autocorrelation
- Transformation used in TWM construction: $a_t = 2 - C_t$ (maps to same direction as Moran's I)

---

## 2. Time Weight Matrix (`temporal_weights.py`)

### TWM from Moran's I (core innovation)

Given a sequence $\{I_1, I_2, \ldots, I_T\}$:

**Raw ratios (lower triangle):**

$$\widetilde{TWM}_{t,s} = \frac{I_t}{I_s} \quad \text{for } s < t$$

$$\widetilde{TWM}_{t,t} = 1, \quad \widetilde{TWM}_{t,s} = 0 \text{ for } s > t$$

**Stability corrections before row-standardisation:**

1. If $|I_s| < \varepsilon_{\min}$: replace $I_s \leftarrow \text{sign}(I_s) \cdot \varepsilon_{\min}$
2. Winsorise: $\widetilde{TWM}_{t,s} \leftarrow \text{clip}\bigl(\widetilde{TWM}_{t,s},\ 0,\ Q_q(|\widetilde{TWM}|)\bigr)$
3. Clip negatives to zero: $\widetilde{TWM}_{t,s} \leftarrow \max(0,\ \widetilde{TWM}_{t,s})$

**Row-standardisation:**

$$TWM_{t,s} = \frac{\widetilde{TWM}_{t,s}}{\sum_{r} \widetilde{TWM}_{t,r}}$$

### TWM from Geary's C

Same as above with $a_t = 2 - C_t$ substituting $I_t$:

$$\widetilde{TWM}_{t,s} = \frac{2 - C_t}{2 - C_s} \quad \text{for } s < t$$

### Decay-based TWM (benchmark)

| Type | Formula ($l = t - s > 0$) |
|---|---|
| Exponential | $w_{t,s} = e^{-\lambda l}$ |
| Linear | $w_{t,s} = \max(0,\ 1 - \lambda l)$ |
| Power | $w_{t,s} = l^{-\lambda}$ |

All with $w_{t,t} = 1$ and row-standardised afterwards.

### Admissibility conditions

A TWM is valid if it satisfies **all four** conditions:

| Condition | Requirement |
|---|---|
| Non-negativity | $TWM_{t,s} \geq 0 \quad \forall\, t,s$ |
| Row-standardisation | $\sum_s TWM_{t,s} = 1 \quad \forall\, t$ |
| Spectral stability | $\rho(TWM) = \max_i |\lambda_i| < \rho_{\max}$ (default $\rho_{\max} = 0.99$) |
| Finiteness | No NaN or Inf entries |

---

## 3. STWM Assembly (`stwm_core.py`)

### Kronecker product

$$STWM = TWM \otimes SWM$$

**Dimension:** $(nT \times nT)$

**Element interpretation:**

$$STWM_{(tn+i),\ (sn+j)} = TWM_{t,s} \times SWM_{i,j}$$

This encodes simultaneously:
- Spatial proximity of unit $i$ to unit $j$ (from SWM)
- Temporal carry-forward from period $s$ to period $t$ (from TWM)

**Stacking convention (time-major):**

$$\mathbf{Y} = [y_{1,1},\ldots,y_{n,1},\ y_{1,2},\ldots,y_{n,2},\ \ldots,\ y_{1,T},\ldots,y_{n,T}]'$$

---

## 4. Spatial Econometric Models (`models.py`)

### SLX — Spatial Lag of X

$$\mathbf{Y} = \iota\alpha + X\boldsymbol{\beta} + WX\boldsymbol{\theta} + \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(0, \sigma^2 I)$$

Effect decomposition (exact, no simultaneity):

$$\text{Direct} = \boldsymbol{\beta}, \quad \text{Indirect} = \boldsymbol{\theta}, \quad \text{Total} = \boldsymbol{\beta} + \boldsymbol{\theta}$$

Estimation: **OLS** — $\hat{\boldsymbol{\beta}} = (X_{\text{aug}}'X_{\text{aug}})^{-1}X_{\text{aug}}'\mathbf{Y}$

---

### SAR — Spatial Autoregressive Model

$$\mathbf{Y} = \rho W\mathbf{Y} + X\boldsymbol{\beta} + \boldsymbol{\varepsilon}$$

Reduced form:

$$\mathbf{Y} = (I - \rho W)^{-1}X\boldsymbol{\beta} + (I - \rho W)^{-1}\boldsymbol{\varepsilon}$$

**Log-likelihood (concentrated):**

$$\ln L(\rho) = -\frac{n}{2}\ln(2\pi\hat{\sigma}^2) + \ln|I - \rho W| - \frac{n}{2}$$

where $\hat{\sigma}^2 = \frac{1}{n}(A\mathbf{Y} - X\hat{\boldsymbol{\beta}})'(A\mathbf{Y} - X\hat{\boldsymbol{\beta}})$, $A = I - \rho W$

**Effect decomposition** (LeSage & Pace 2009):

Let $S(W) = (I - \rho W)^{-1}$. For each variable $k$:

$$\text{Direct}_k = \frac{1}{n}\,\text{tr}[S(W)\,\beta_k]$$

$$\text{Total}_k = \frac{1}{n}\,\mathbf{1}'S(W)\mathbf{1}\,\beta_k$$

$$\text{Indirect}_k = \text{Total}_k - \text{Direct}_k$$

---

### SEM — Spatial Error Model

$$\mathbf{Y} = X\boldsymbol{\beta} + \mathbf{u}, \quad \mathbf{u} = \lambda W\mathbf{u} + \boldsymbol{\varepsilon}$$

GLS transformation: $(I - \lambda W)\mathbf{Y} = (I - \lambda W)X\boldsymbol{\beta} + \boldsymbol{\varepsilon}$

**Log-likelihood:**

$$\ln L(\lambda) = -\frac{n}{2}\ln(2\pi\hat{\sigma}^2) + \ln|I - \lambda W| - \frac{n}{2}$$

Effect decomposition: Direct $= \boldsymbol{\beta}$, Indirect $= \mathbf{0}$, Total $= \boldsymbol{\beta}$

---

### SDM — Spatial Durbin Model

$$\mathbf{Y} = \rho W\mathbf{Y} + X\boldsymbol{\beta} + WX\boldsymbol{\theta} + \boldsymbol{\varepsilon}$$

**Log-likelihood:** same as SAR with extended regressor matrix $[X\ WX]$.

**Effect decomposition** for variable $k$:

$$S_k(W) = (I - \rho W)^{-1}(I\beta_k + W\theta_k)$$

$$\text{Direct}_k = \frac{1}{n}\,\text{tr}[S_k(W)]$$

$$\text{Total}_k = \frac{1}{n}\,\mathbf{1}'S_k(W)\mathbf{1}$$

$$\text{Indirect}_k = \text{Total}_k - \text{Direct}_k$$

---

## 5. Endogeneity Testing (`endogeneity.py`)

### Two-Stage Least Squares (2SLS)

**Stage 1** — Project endogenous regressors onto instrument space:

$$\hat{X} = P_Z X = Z(Z'Z)^{-1}Z'X$$

**Stage 2** — IV estimator:

$$\hat{\boldsymbol{\beta}}_{IV} = (\hat{X}'X)^{-1}\hat{X}'\mathbf{Y}$$

**Covariance:**

$$\widehat{\text{Var}}(\hat{\boldsymbol{\beta}}_{IV}) = \hat{\sigma}^2_{IV}(\hat{X}'\hat{X})^{-1}, \quad \hat{\sigma}^2_{IV} = \frac{(\mathbf{Y} - X\hat{\boldsymbol{\beta}}_{IV})'(\mathbf{Y} - X\hat{\boldsymbol{\beta}}_{IV})}{n - k}$$

**First-stage F-statistic** (instrument relevance; rule of thumb: $F > 10$):

$$F_1 = \frac{(RSS_R - RSS_U)/m}{RSS_U/(n - m - 1)}$$

---

### Hausman Specification Test

$$H_0\text{: OLS is consistent (no endogeneity)} \quad H_1\text{: IV preferred}$$

**Test statistic:**

$$H = (\hat{\boldsymbol{\beta}}_{IV} - \hat{\boldsymbol{\beta}}_{OLS})'\bigl[\widehat{\text{Var}}(\hat{\boldsymbol{\beta}}_{IV}) - \widehat{\text{Var}}(\hat{\boldsymbol{\beta}}_{OLS})\bigr]^{+}(\hat{\boldsymbol{\beta}}_{IV} - \hat{\boldsymbol{\beta}}_{OLS})$$

$$H \sim \chi^2(k) \quad \text{under } H_0$$

where $[\cdot]^+$ denotes the Moore-Penrose pseudo-inverse.

---

## 6. Heteroskedasticity Tests (`heterogeneity.py`)

### Breusch-Pagan Test

1. Obtain OLS residuals: $\hat{\mathbf{e}} = \mathbf{Y} - X\hat{\boldsymbol{\beta}}$
2. Regress $\hat{e}_i^2$ on $X$; obtain $R^2$
3. Test statistic: $LM = n \cdot R^2 \sim \chi^2(k-1)$ under $H_0$

### White Test

Same procedure with an augmented regressor matrix including $X$, $X^2$ (element-wise), and all cross-products $x_{ij} \cdot x_{il}$.

$$LM = n \cdot R^2 \sim \chi^2(p-1) \quad \text{under } H_0$$

where $p$ = number of columns in the augmented White matrix.

---

## 7. Model Selection (`robustness.py`)

For a model with $k$ parameters, $n$ observations, maximised log-likelihood $\hat{L}$:

$$AIC = -2\hat{L} + 2k$$

$$BIC = -2\hat{L} + k\ln(n)$$

Lower AIC / BIC indicates better fit penalised by complexity.

---

## 8. Monte Carlo Simulation (`simulation.py`)

### Data-Generating Process

$$\mathbf{Y} = (I - \rho \cdot STWM)^{-1}(X\boldsymbol{\beta} + \boldsymbol{\varepsilon}), \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)$$

### Performance Metrics (over $S$ replications)

$$\text{Bias}(\hat{\theta}) = \frac{1}{S}\sum_{s=1}^{S}(\hat{\theta}_s - \theta_{\text{true}})$$

$$\text{RMSE}(\hat{\theta}) = \sqrt{\frac{1}{S}\sum_{s=1}^{S}(\hat{\theta}_s - \theta_{\text{true}})^2}$$

$$\text{Coverage} = \frac{1}{S}\sum_{s=1}^{S}\mathbf{1}\bigl[|\hat{\theta}_s - \theta_{\text{true}}| < 1.96 \cdot \text{std}(\hat{\theta})\bigr]$$

---

## 9. Granger Causality Test (`simulation.py`)

Tests whether $\{I_t\}$ Granger-causes $\{s_t\}$ (spillover parameter sequence).

**Restricted model (lag $p$):**

$$s_t = \alpha + \sum_{l=1}^{p} b_l s_{t-l} + u_t$$

**Unrestricted model:**

$$s_t = \alpha + \sum_{l=1}^{p} b_l s_{t-l} + \sum_{l=1}^{p} c_l I_{t-l} + v_t$$

**F-statistic:**

$$F = \frac{(RSS_R - RSS_U)/p}{RSS_U/(T - 2p - 1)} \sim F(p,\ T - 2p - 1) \quad \text{under } H_0: c_1 = \cdots = c_p = 0$$

$H_0$ rejected ($p < 0.05$) → Moran's I Granger-causes spillover parameters, supporting the TWM construction.

---

## References

- Anselin, L. (1988). *Spatial Econometrics*. Kluwer Academic Publishers.
- Breusch, T. S., & Pagan, A. R. (1979). Econometrica, 47(5), 1287-1294.
- Cliff, A. D., & Ord, J. K. (1981). *Spatial Processes*. Pion.
- Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Big Data*. Springer.
- Gibbons, S., & Overman, H. G. (2012). Journal of Regional Science, 52(2), 172-191.
- Granger, C. W. J. (1969). Econometrica, 37(3), 424-438.
- Halleck Vega, S., & Elhorst, J. P. (2015). Journal of Regional Science, 55(3), 339-363.
- Hausman, J. A. (1978). Econometrica, 46(6), 1251-1271.
- LeSage, J., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Schwarz, G. (1978). Annals of Statistics, 6(2), 461-464.
- White, H. (1980). Econometrica, 48(4), 817-838.
