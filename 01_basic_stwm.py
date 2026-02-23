"""
examples/01_basic_stwm.py
--------------------------
Complete walkthrough of the STWM package.

Sections:
    1. Build spatial weight matrices (geographic, economic, nested)
    2. Build time weight matrix from Moran's I
    3. Assemble STWM and check admissibility
    4. Simulate panel data and estimate SLX, SAR, SDM
    5. Heteroskedasticity test
    6. Robustness: compare alternative weight matrices
    7. Monte Carlo validation
    8. Granger causality test

Data: simulated for illustration — substitute your own panel dataset.
"""

import numpy as np
import sys
sys.path.insert(0, "..")

from stwm import (
    geographic_swm, economic_swm, nested_swm,
    build_twm_morans, build_twm_gearyc, build_twm_decay,
    compute_morans_i, compute_geary_c,
    twm_stability_check, build_stwm, stwm_summary,
    SLXModel, SpatialLagModel, SDMModel,
    heteroskedasticity_test,
    compare_weight_matrices,
    monte_carlo_stwm,
    granger_spillover_test,
)

np.random.seed(42)

# =============================================================================
# 1. Spatial weight matrix
# =============================================================================
# Replace with your own coordinates: (longitude, latitude) in decimal degrees
coords = np.array([
    [2.35,  48.85],   # Paris
    [2.30,  48.87],   # Saint-Denis
    [2.40,  48.80],   # Vincennes
    [2.20,  48.90],   # Argenteuil
    [2.55,  48.83],   # Noisy-le-Grand
    [2.45,  48.93],   # Bobigny
])
# Proxy GDP values (replace with real data)
gdp = np.array([100.0, 22.0, 18.0, 30.0, 25.0, 20.0])

W_geo    = geographic_swm(coords, method="inverse_distance")
W_eco    = economic_swm(gdp,    method="gdp_distance")
W_nested = nested_swm(W_geo, W_eco, alpha=0.5)

print("=== Spatial Weight Matrix ===")
print("W_nested shape:", W_nested.shape)
print("Row sums (should be ~1):", W_nested.sum(axis=1).round(4))

# =============================================================================
# 2. Time weight matrix
# =============================================================================
# Replace with your empirically computed annual Moran's I values
morans_i = np.array([0.31, 0.34, 0.29, 0.38, 0.41,
                     0.45, 0.43, 0.47, 0.51, 0.48,
                     0.52, 0.55, 0.58, 0.60])
T = len(morans_i)
n = len(coords)

TWM_morans  = build_twm_morans(morans_i, winsorize_quantile=0.95, min_abs=1e-3)
TWM_decay_e = build_twm_decay(T, decay_type="exponential", param=0.3)
TWM_decay_l = build_twm_decay(T, decay_type="linear",      param=0.15)

print("\n=== Time Weight Matrix ===")
print("Shape:", TWM_morans.shape)
check = twm_stability_check(TWM_morans)
print("Stability check:", check)

# =============================================================================
# 3. Assemble STWM
# =============================================================================
STWM = build_stwm(TWM_morans, W_nested)
summary = stwm_summary(STWM, n, T)
print("\n=== STWM Summary ===")
for k_name, v in summary.items():
    print(f"  {k_name}: {v}")

# =============================================================================
# 4. Simulate panel data (replace with real data)
# =============================================================================
nT   = n * T
X    = np.random.randn(nT, 3)
I_inv = np.linalg.inv(np.eye(nT) - 0.4 * STWM)
Y    = I_inv @ (X @ np.array([0.5, -0.3, 0.2]) + np.random.randn(nT))

# --- SLX (transparent OLS baseline) ---
slx = SLXModel(STWM).fit(Y, X)
print("\n=== SLX Model ===")
print("Direct   effects:", slx.summary()["direct"].round(4))
print("Indirect effects:", slx.summary()["indirect"].round(4))

# --- SAR ---
sar = SpatialLagModel(STWM).fit(Y, X)
print("\n=== SAR Model ===")
print("rho:", round(sar.summary()["rho"], 4))
print("Direct   effects:", sar.summary()["direct"].round(4))
print("Indirect effects:", sar.summary()["indirect"].round(4))

# --- SDM ---
sdm = SDMModel(STWM).fit(Y, X)
print("\n=== SDM Model ===")
print("rho:", round(sdm.summary()["rho"], 4))
print("Direct   effects:", sdm.summary()["direct"].round(4))
print("Indirect effects:", sdm.summary()["indirect"].round(4))
print("Total    effects:", sdm.summary()["total"].round(4))

# =============================================================================
# 5. Heteroskedasticity test
# =============================================================================
X_const = np.column_stack([np.ones(nT), X])
bp = heteroskedasticity_test(Y, X_const, test="breusch_pagan")
print("\n=== Breusch-Pagan Test ===")
print(bp["conclusion"], f"(stat={bp['statistic']}, p={bp['p_value']})")

# =============================================================================
# 6. Robustness: compare weight matrices
# =============================================================================
matrices = {
    "STWM_Morans"   : STWM,
    "Decay_Exp_0.3" : build_stwm(TWM_decay_e, W_nested),
    "Decay_Lin_0.15": build_stwm(TWM_decay_l, W_nested),
}
report = compare_weight_matrices(Y, X, matrices, SDMModel)
print("\n=== Weight Matrix Comparison (AIC) ===")
for label, aic_val in report["comparison_table"]["AIC"].items():
    print(f"  {label}: {aic_val}")

# =============================================================================
# 7. Monte Carlo validation
# =============================================================================
mc = monte_carlo_stwm(
    true_rho      = 0.4,
    true_beta     = np.array([0.5, -0.3, 0.2]),
    W_spatial     = W_nested,
    TWM           = TWM_morans,
    n             = n,
    T             = T,
    n_simulations = 200,    # use 1000+ for publication
)
print("\n=== Monte Carlo Validation (S=200) ===")
print(f"Bias(ρ)    = {mc['bias_rho']:.4f}")
print(f"RMSE(ρ)    = {mc['rmse_rho']:.4f}")
print(f"Coverage   = {mc['coverage_rho']:.2%}")
print(f"Bias(β)    = {mc['bias_beta'].round(4)}")

# =============================================================================
# 8. Granger causality test
# =============================================================================
# In practice, use rolling-window indirect effect estimates as spillover_sequence
annual_indirect = 0.3 + 0.5 * morans_i + np.random.randn(T) * 0.02

granger = granger_spillover_test(morans_i, annual_indirect, max_lag=3)
print("\n=== Granger Causality Test ===")
for lag_key, res in granger.items():
    if "error" not in res:
        print(f"  {res['conclusion']}")
