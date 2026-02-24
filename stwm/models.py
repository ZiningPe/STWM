"""
models.py
---------
Spatial econometric models following Elhorst (2014) exactly.

Effect Decomposition (Elhorst 2014, Table 2.1)
-----------------------------------------------
For a general model  Y = (I − δW)⁻¹(Xβ + WXθ) + R, the matrix of partial
derivatives of E(Y) w.r.t. the k-th variable x_k is (Elhorst 2014, eq. 2.13):

    ∂E(Y)/∂x_k' = (I − δW)⁻¹ · [I·β_k + W·θ_k]   ≡ S_k(W)

    where element (i,j) of S_k(W) = ∂E(y_i)/∂x_{jk}

Summary scalars (LeSage & Pace 2009, p.74; Elhorst 2014, Table 2.1):
    Direct effect   = (1/N) tr[S_k(W)]         ← average diagonal element
    Total effect    = (1/N) 1'S_k(W)1          ← average row sum
    Indirect effect = Total − Direct            ← average off-diagonal row sum

Model-specific simplifications (Elhorst 2014, Table 2.1):
┌─────────┬─────────────────────────────────┬──────────────────────────────────┐
│ Model   │ Direct effect                   │ Indirect effect                  │
├─────────┼─────────────────────────────────┼──────────────────────────────────┤
│ OLS/SEM │ β_k                             │ 0                                │
│ SAR/SAC │ diag[(I−δW)⁻¹] · β_k (avg)     │ off-diag[(I−δW)⁻¹] · β_k (avg)  │
│ SLX     │ β_k                             │ θ_k                              │
│ SDM     │ diag[(I−δW)⁻¹(Iβ_k+Wθ_k)](avg) │ off-diag[...](avg)               │
└─────────┴─────────────────────────────────┴──────────────────────────────────┘

Inference on Effects (Elhorst 2014, eq. 2.16–2.17)
----------------------------------------------------
Direct/indirect effects are non-linear functions of (ρ, β, θ). Standard errors
are obtained by simulation from the ML variance-covariance matrix:

    [α_d, β_d, θ_d, δ_d]' = P·ξ + [α̂, β̂, θ̂, δ̂]'    (eq. 2.16)

where P = upper-triangular Cholesky of Var(α̂,β̂,θ̂,δ̂) and ξ~N(0,I).
For D=1000 draws, the indirect effect mean and t-value are (eq. 2.17):

    μ_k = (1/D) Σ_d μ_{kd}
    t_k = μ_k / std(μ_{kd})

Log-likelihood (SAR/SDM, Elhorst 2014 eq. following 2.6)
---------------------------------------------------------
    ln L(δ) = −(N/2)ln(2πσ²) + ln|I−δW| − (1/2σ²)(AY−Xβ̂)'(AY−Xβ̂)

where A = I − δW, β̂ concentrated out by OLS of AY on X.

References
----------
Elhorst, J. P. (2014). Spatial Econometrics: From Cross-Sectional Data to Big Data.
    Springer. [Primary reference — all formulas follow this book exactly]
LeSage, J., & Pace, R. K. (2009). Introduction to Spatial Econometrics. CRC Press.
Halleck Vega, S., & Elhorst, J. P. (2015). Journal of Regional Science, 55(3), 339-363.
"""

import numpy as np
from scipy import stats, optimize
from typing import Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ols_fit(Y: np.ndarray, X: np.ndarray) -> dict:
    """OLS: β = (X'X)⁻¹X'Y with full inference."""
    Y    = np.asarray(Y, dtype=float).ravel()
    X    = np.asarray(X, dtype=float)
    n, k = X.shape
    beta  = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ beta
    sigma2 = float(resid @ resid) / (n - k)
    cov    = sigma2 * np.linalg.pinv(X.T @ X)
    se     = np.sqrt(np.diag(cov))
    t_stat = beta / np.where(se > 0, se, np.nan)
    pval   = 2 * stats.t.sf(np.abs(t_stat), df=n - k)
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    R2     = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return dict(beta=beta, se=se, t_stats=t_stat, p_values=pval,
                residuals=resid, sigma2=sigma2, cov=cov, R2=R2)


def _simulation_se(S_draws: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Elhorst (2014) eq. 2.17: mean, SE, t-value from D simulation draws.
    S_draws : (D, k) array of effect estimates across draws
    """
    mean  = S_draws.mean(axis=0)
    se    = S_draws.std(axis=0, ddof=1)
    tval  = mean / np.where(se > 0, se, np.nan)
    return mean, se, tval


def _effect_inference(W: np.ndarray, beta_X: np.ndarray, theta: np.ndarray,
                      rho: np.ndarray, cov_params: np.ndarray,
                      n: int, k: int, D: int = 1000,
                      seed: int = 0) -> dict:
    """
    Simulation-based inference for direct/indirect/total effects.
    Implements Elhorst (2014) eq. 2.16-2.17.

    Parameters
    ----------
    W          : (n,n) weight matrix
    beta_X     : (k,) point estimates of X coefficients
    theta      : (k,) point estimates of WX coefficients (0 for SAR)
    rho        : scalar spatial autoregressive parameter
    cov_params : (p,p) ML variance-covariance of [rho, beta_X, theta]
                 p = 1 + k + k (SDM) or 1 + k (SAR)
    D          : number of simulation draws (default 1000)
    """
    rng = np.random.default_rng(seed)
    try:
        P = np.linalg.cholesky(cov_params + 1e-10 * np.eye(len(cov_params)))
    except np.linalg.LinAlgError:
        P = np.diag(np.sqrt(np.maximum(np.diag(cov_params), 0)))

    p_dim  = cov_params.shape[0]
    has_theta = len(theta) == k

    direct_d   = np.zeros((D, k))
    indirect_d = np.zeros((D, k))
    total_d    = np.zeros((D, k))

    p_dim_check = cov_params.shape[0]
    if p_dim_check == 1 + len(beta_X) + len(theta):
        center = np.concatenate([[rho], beta_X, theta])
        has_theta = True
    else:
        center = np.concatenate([[rho], beta_X])
        has_theta = False

    for d in range(D):
        xi   = rng.standard_normal(p_dim)
        draw = P @ xi + center
        r_d  = float(np.clip(draw[0], -0.99, 0.99))
        b_d  = draw[1:k+1]
        t_d  = draw[k+1:] if has_theta else np.zeros(k)

        A_d = np.eye(n) - r_d * W
        try:
            S_inv = np.linalg.inv(A_d)
        except np.linalg.LinAlgError:
            continue

        for i in range(k):
            # S_k(W) = (I-δW)^{-1}(I·β_k + W·θ_k)  [Elhorst 2014, eq.2.13]
            M_k = S_inv @ (np.eye(n) * b_d[i] + W * t_d[i])
            direct_d[d, i]   = np.trace(M_k) / n
            total_d[d, i]    = np.ones(n) @ M_k @ np.ones(n) / n
            indirect_d[d, i] = total_d[d, i] - direct_d[d, i]

    dir_mean,  dir_se,  dir_t  = _simulation_se(direct_d)
    ind_mean,  ind_se,  ind_t  = _simulation_se(indirect_d)
    tot_mean,  tot_se,  tot_t  = _simulation_se(total_d)

    return dict(
        direct=dir_mean,   direct_se=dir_se,   direct_t=dir_t,
        indirect=ind_mean, indirect_se=ind_se, indirect_t=ind_t,
        total=tot_mean,    total_se=tot_se,    total_t=tot_t,
        direct_p  =2*stats.norm.sf(np.abs(dir_t)),
        indirect_p=2*stats.norm.sf(np.abs(ind_t)),
        total_p   =2*stats.norm.sf(np.abs(tot_t)),
    )


def _format_stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


def print_effects_table(res: dict, var_names: list):
    """Pretty-print direct/indirect/total effects with t-stats and p-values."""
    print(f"\n{'─'*75}")
    print(f"  Effect Decomposition  [Elhorst 2014, Table 2.1]")
    print(f"{'─'*75}")
    print(f"  {'Variable':<14} {'Effect':>9} {'SE':>9} {'t-stat':>9} {'p':>8}  Sig")
    print(f"{'─'*75}")
    for j, var in enumerate(var_names):
        for eff_name in ['direct', 'indirect', 'total']:
            v  = res[eff_name][j]
            se = res.get(f'{eff_name}_se', np.zeros_like(res[eff_name]))[j]
            t  = res.get(f'{eff_name}_t',  np.zeros_like(res[eff_name]))[j]
            p  = res.get(f'{eff_name}_p',  np.ones_like(res[eff_name]))[j]
            stars = _format_stars(p)
            label = f"{var}_{eff_name[0].upper()}" if eff_name != 'total' \
                    else f"{var}_Tot"
            print(f"  {label:<14} {v:>9.4f} {se:>9.4f} {t:>9.3f} {p:>8.4f}  {stars}")
        print()
    print(f"{'─'*75}")
    print("  Significance: *** p<0.01  ** p<0.05  * p<0.10")


# ---------------------------------------------------------------------------
# SLX
# ---------------------------------------------------------------------------

class SLXModel:
    """
    Spatial Lag of X (SLX).

    Model (Elhorst 2014):
        Y = ι·α + X·β + WX·θ + ε,    ε ~ N(0, σ²I)

    Effects (exact, no simulation needed — Elhorst 2014, Table 2.1):
        Direct   = β_k
        Indirect = θ_k
        Total    = β_k + θ_k

    SE of indirect = SE(θ_k) from OLS covariance.
    SE of total    = √[Var(β_k) + Var(θ_k) + 2·Cov(β_k,θ_k)]
    """

    def __init__(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)
        self.results_ = None

    def fit(self, Y: np.ndarray, X: np.ndarray) -> "SLXModel":
        Y  = np.asarray(Y, dtype=float).ravel()
        X  = np.asarray(X, dtype=float)
        n, k = X.shape
        WX    = self.W @ X
        X_aug = np.column_stack([np.ones(n), X, WX])
        res   = _ols_fit(Y, X_aug)

        beta  = res["beta"][1:k+1]
        theta = res["beta"][k+1:2*k+1]
        se_b  = res["se"][1:k+1]
        se_t  = res["se"][k+1:2*k+1]
        cov   = res["cov"]
        cov_bt = cov[1:k+1, k+1:2*k+1]

        se_total = np.sqrt(se_b**2 + se_t**2 + 2*np.diag(cov_bt))
        t_dir   = beta  / np.where(se_b      > 0, se_b,      np.nan)
        t_ind   = theta / np.where(se_t      > 0, se_t,      np.nan)
        t_tot   = (beta+theta) / np.where(se_total > 0, se_total, np.nan)
        p_dir   = 2*stats.t.sf(np.abs(t_dir), df=n-X_aug.shape[1])
        p_ind   = 2*stats.t.sf(np.abs(t_ind), df=n-X_aug.shape[1])
        p_tot   = 2*stats.t.sf(np.abs(t_tot), df=n-X_aug.shape[1])

        self.results_ = {
            **res, "model": "SLX",
            "beta_X": beta, "theta_WX": theta,
            "direct": beta,   "direct_se": se_b,   "direct_t": t_dir,   "direct_p": p_dir,
            "indirect": theta, "indirect_se": se_t, "indirect_t": t_ind, "indirect_p": p_ind,
            "total": beta+theta, "total_se": se_total, "total_t": t_tot, "total_p": p_tot,
        }
        return self

    def summary(self) -> dict:
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_

    def print_summary(self, var_names=None):
        res = self.summary()
        n_obs = len(res["residuals"])
        k = len(res["beta_X"])
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(k)]
        print(f"\n{'═'*55}")
        print(f"  SLX Model  (OLS, Elhorst 2014 Table 2.1)")
        print(f"{'═'*55}")
        print(f"  Observations: {n_obs}   R²: {res['R2']:.4f}   σ²: {res['sigma2']:.4f}")
        print_effects_table(res, var_names)


# ---------------------------------------------------------------------------
# SAR
# ---------------------------------------------------------------------------

class SpatialLagModel:
    """
    Spatial Autoregressive Model (SAR).

    Model (Elhorst 2014):
        Y = δ·WY + X·β + ε,    ε ~ N(0, σ²I)

    Reduced form:
        Y = (I−δW)⁻¹ X·β + (I−δW)⁻¹ ε

    Effects (Elhorst 2014, Table 2.1):
        S_k(W) = (I−δW)⁻¹ · I·β_k
        Direct   = (1/N) tr[S_k(W)]
        Total    = (1/N) 1'S_k(W)1
        Indirect = Total − Direct

    Inference: simulation from ML variance-covariance (Elhorst 2014, eq.2.16–2.17).

    Log-likelihood (concentrated):
        ln L(δ) = −(N/2)ln(2πσ̂²) + ln|I−δW| − N/2
        σ̂²(δ) = (1/N)(AY−Xβ̂)'(AY−Xβ̂),  A = I−δW
    """

    def __init__(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)
        self.results_ = None

    def fit(self, Y: np.ndarray, X: np.ndarray,
            rho_bounds: Tuple[float,float] = (-0.99, 0.99),
            n_draws: int = 1000) -> "SpatialLagModel":
        Y     = np.asarray(Y, dtype=float).ravel()
        X     = np.asarray(X, dtype=float)
        n, k  = X.shape
        X_aug = np.column_stack([np.ones(n), X])
        W     = self.W
        ev_W  = np.linalg.eigvals(W).real

        def neg_ll(rho):
            A      = np.eye(n) - rho * W
            AY     = A @ Y
            beta   = np.linalg.lstsq(X_aug, AY, rcond=None)[0]
            resid  = AY - X_aug @ beta
            sigma2 = float(resid @ resid) / n
            log_det = float(np.sum(np.log(np.abs(1 - rho * ev_W))))
            return 0.5 * n * np.log(sigma2) - log_det

        opt     = optimize.minimize_scalar(neg_ll, bounds=rho_bounds, method="bounded")
        rho_hat = float(opt.x)
        A       = np.eye(n) - rho_hat * W
        AY      = A @ Y
        beta    = np.linalg.lstsq(X_aug, AY, rcond=None)[0]
        resid   = AY - X_aug @ beta
        sigma2  = float(resid @ resid) / (n - X_aug.shape[1])

        # ML covariance via numerical Hessian of neg_ll for ρ;
        # OLS covariance for β conditional on ρ̂
        ols_cov = sigma2 * np.linalg.pinv(X_aug.T @ X_aug)
        # Approximate joint cov [rho, beta_X] — block diagonal (standard approximation)
        d2 = optimize.approx_fprime([rho_hat], neg_ll, 1e-5)
        se_rho = 1.0 / np.sqrt(max(float(d2[0]), 1e-12))
        cov_joint = np.zeros((1+k, 1+k))
        cov_joint[0, 0] = se_rho**2
        cov_joint[1:, 1:] = ols_cov[1:k+1, 1:k+1]

        eff = _effect_inference(W, beta[1:], np.zeros(k), rho_hat,
                                cov_joint, n, k, D=n_draws)

        # Standard errors for coefficient estimates
        se_beta = np.sqrt(np.diag(ols_cov))[1:]
        t_beta  = beta[1:] / np.where(se_beta > 0, se_beta, np.nan)
        p_beta  = 2 * stats.t.sf(np.abs(t_beta), df=n - X_aug.shape[1])
        t_rho   = rho_hat / se_rho
        p_rho   = 2 * stats.norm.sf(abs(t_rho))

        self.results_ = {
            "model": "SAR", "rho": rho_hat, "rho_se": se_rho,
            "rho_t": t_rho, "rho_p": p_rho,
            "beta": beta[1:], "beta_se": se_beta,
            "beta_t": t_beta, "beta_p": p_beta,
            "intercept": beta[0], "sigma2": sigma2,
            **eff,
        }
        return self

    def summary(self) -> dict:
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_

    def print_summary(self, var_names=None):
        res = self.summary()
        n_obs = len(res.get("beta", [])) + 1
        k = len(res["beta"])
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(k)]
        print(f"\n{'═'*55}")
        print(f"  SAR Model  (ML, Elhorst 2014)")
        print(f"{'═'*55}")
        print(f"  ρ = {res['rho']:.4f}  SE={res['rho_se']:.4f}  "
              f"t={res['rho_t']:.3f}  p={res['rho_p']:.4f}{_format_stars(res['rho_p'])}")
        print(f"  σ² = {res['sigma2']:.4f}")
        print(f"\n  Coefficient estimates:")
        print(f"  {'Variable':<14} {'β':>9} {'SE':>9} {'t':>9} {'p':>8}  Sig")
        print(f"  {'─'*55}")
        for j, v in enumerate(var_names):
            b = res['beta'][j]; se=res['beta_se'][j]
            t=res['beta_t'][j]; p=res['beta_p'][j]
            print(f"  {v:<14} {b:>9.4f} {se:>9.4f} {t:>9.3f} {p:>8.4f}  {_format_stars(p)}")
        print_effects_table(res, var_names)


# ---------------------------------------------------------------------------
# SEM
# ---------------------------------------------------------------------------

class SpatialErrorModel:
    """
    Spatial Error Model (SEM).

    Model (Elhorst 2014):
        Y = X·β + u,    u = λ·Wu + ε,    ε ~ N(0, σ²I)

    GLS transformation:
        (I−λW)Y = (I−λW)X·β + ε

    Effects (Elhorst 2014, Table 2.1):
        Direct   = β_k     (no spillover through mean equation)
        Indirect = 0
        Total    = β_k
    """

    def __init__(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)
        self.results_ = None

    def fit(self, Y: np.ndarray, X: np.ndarray,
            lam_bounds: Tuple[float,float] = (-0.99, 0.99)) -> "SpatialErrorModel":
        Y     = np.asarray(Y, dtype=float).ravel()
        X     = np.asarray(X, dtype=float)
        n, k  = X.shape
        X_aug = np.column_stack([np.ones(n), X])
        W     = self.W
        ev_W  = np.linalg.eigvals(W).real

        def neg_ll(lam):
            B      = np.eye(n) - lam * W
            BY, BX = B @ Y, B @ X_aug
            beta   = np.linalg.lstsq(BX, BY, rcond=None)[0]
            resid  = BY - BX @ beta
            sigma2 = float(resid @ resid) / n
            log_det = float(np.sum(np.log(np.abs(1 - lam * ev_W))))
            return 0.5 * n * np.log(sigma2) - log_det

        opt     = optimize.minimize_scalar(neg_ll, bounds=lam_bounds, method="bounded")
        lam_hat = float(opt.x)
        B       = np.eye(n) - lam_hat * W
        BY, BX  = B @ Y, B @ X_aug
        beta    = np.linalg.lstsq(BX, BY, rcond=None)[0]
        resid   = BY - BX @ beta
        sigma2  = float(resid @ resid) / (n - X_aug.shape[1])

        ols_cov = sigma2 * np.linalg.pinv(BX.T @ BX)
        se_beta = np.sqrt(np.diag(ols_cov))[1:]
        t_beta  = beta[1:] / np.where(se_beta > 0, se_beta, np.nan)
        p_beta  = 2 * stats.t.sf(np.abs(t_beta), df=n - X_aug.shape[1])

        d2 = optimize.approx_fprime([lam_hat], neg_ll, 1e-5)
        se_lam = 1.0 / np.sqrt(max(float(d2[0]), 1e-12))
        t_lam  = lam_hat / se_lam
        p_lam  = 2 * stats.norm.sf(abs(t_lam))

        self.results_ = {
            "model": "SEM", "lambda": lam_hat,
            "lambda_se": se_lam, "lambda_t": t_lam, "lambda_p": p_lam,
            "beta": beta[1:], "beta_se": se_beta,
            "beta_t": t_beta, "beta_p": p_beta,
            "intercept": beta[0], "sigma2": sigma2,
            "direct": beta[1:],        "direct_se": se_beta,
            "direct_t": t_beta,        "direct_p": p_beta,
            "indirect": np.zeros(k),   "indirect_se": np.zeros(k),
            "indirect_t": np.zeros(k), "indirect_p": np.ones(k),
            "total": beta[1:],         "total_se": se_beta,
            "total_t": t_beta,         "total_p": p_beta,
        }
        return self

    def summary(self) -> dict:
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_

    def print_summary(self, var_names=None):
        res = self.summary()
        k = len(res["beta"])
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(k)]
        print(f"\n{'═'*55}")
        print(f"  SEM Model  (ML, Elhorst 2014)")
        print(f"{'═'*55}")
        print(f"  λ = {res['lambda']:.4f}  SE={res['lambda_se']:.4f}  "
              f"t={res['lambda_t']:.3f}  p={res['lambda_p']:.4f}{_format_stars(res['lambda_p'])}")
        print(f"  σ² = {res['sigma2']:.4f}")
        print(f"\n  {'Variable':<14} {'β':>9} {'SE':>9} {'t':>9} {'p':>8}  Sig")
        print(f"  {'─'*55}")
        for j, v in enumerate(var_names):
            b=res['beta'][j]; se=res['beta_se'][j]
            t=res['beta_t'][j]; p=res['beta_p'][j]
            print(f"  {v:<14} {b:>9.4f} {se:>9.4f} {t:>9.3f} {p:>8.4f}  {_format_stars(p)}")


# ---------------------------------------------------------------------------
# SDM
# ---------------------------------------------------------------------------

class SDMModel:
    """
    Spatial Durbin Model (SDM).

    Model (Elhorst 2014):
        Y = δ·WY + X·β + WX·θ + ε,    ε ~ N(0, σ²I)

    Effects (Elhorst 2014, Table 2.1; eq. 2.13):
        S_k(W) = (I−δW)⁻¹ · (I·β_k + W·θ_k)
        Direct   = (1/N) tr[S_k(W)]
        Total    = (1/N) 1'S_k(W)1
        Indirect = Total − Direct

    Inference: Elhorst (2014) eq. 2.16–2.17 simulation from ML covariance.
    """

    def __init__(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)
        self.results_ = None

    def fit(self, Y: np.ndarray, X: np.ndarray,
            rho_bounds: Tuple[float,float] = (-0.99, 0.99),
            n_draws: int = 1000) -> "SDMModel":
        Y     = np.asarray(Y, dtype=float).ravel()
        X     = np.asarray(X, dtype=float)
        n, k  = X.shape
        WX    = self.W @ X
        X_aug = np.column_stack([np.ones(n), X, WX])
        W     = self.W
        ev_W  = np.linalg.eigvals(W).real

        def neg_ll(rho):
            A      = np.eye(n) - rho * W
            AY     = A @ Y
            beta   = np.linalg.lstsq(X_aug, AY, rcond=None)[0]
            resid  = AY - X_aug @ beta
            sigma2 = float(resid @ resid) / n
            log_det = float(np.sum(np.log(np.abs(1 - rho * ev_W))))
            return 0.5 * n * np.log(sigma2) - log_det

        opt     = optimize.minimize_scalar(neg_ll, bounds=rho_bounds, method="bounded")
        rho_hat = float(opt.x)
        A       = np.eye(n) - rho_hat * W
        AY      = A @ Y
        beta    = np.linalg.lstsq(X_aug, AY, rcond=None)[0]
        resid   = AY - X_aug @ beta
        sigma2  = float(resid @ resid) / (n - X_aug.shape[1])

        beta_X = beta[1:k+1]
        theta  = beta[k+1:]

        # Approximate joint covariance [rho, beta_X, theta]
        ols_cov = sigma2 * np.linalg.pinv(X_aug.T @ X_aug)
        d2      = optimize.approx_fprime([rho_hat], neg_ll, 1e-5)
        se_rho  = 1.0 / np.sqrt(max(float(d2[0]), 1e-12))
        cov_joint = np.zeros((1+2*k, 1+2*k))
        cov_joint[0, 0]  = se_rho**2
        cov_joint[1:, 1:] = ols_cov[1:2*k+1, 1:2*k+1]

        eff = _effect_inference(W, beta_X, theta, rho_hat,
                                cov_joint, n, k, D=n_draws)

        # Coefficient-level inference
        se_full = np.sqrt(np.diag(ols_cov))
        se_bX   = se_full[1:k+1]
        se_th   = se_full[k+1:2*k+1]
        t_bX    = beta_X / np.where(se_bX > 0, se_bX, np.nan)
        t_th    = theta  / np.where(se_th > 0, se_th, np.nan)
        p_bX    = 2*stats.t.sf(np.abs(t_bX), df=n-X_aug.shape[1])
        p_th    = 2*stats.t.sf(np.abs(t_th), df=n-X_aug.shape[1])
        t_rho   = rho_hat / se_rho
        p_rho   = 2*stats.norm.sf(abs(t_rho))

        self.results_ = {
            "model": "SDM", "rho": rho_hat, "rho_se": se_rho,
            "rho_t": t_rho, "rho_p": p_rho,
            "beta_X": beta_X, "beta_X_se": se_bX,
            "beta_X_t": t_bX, "beta_X_p": p_bX,
            "theta_WX": theta, "theta_WX_se": se_th,
            "theta_WX_t": t_th, "theta_WX_p": p_th,
            "intercept": beta[0], "sigma2": sigma2,
            **eff,
        }
        return self

    def summary(self) -> dict:
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_

    def print_summary(self, var_names=None):
        res = self.summary()
        k = len(res["beta_X"])
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(k)]
        print(f"\n{'═'*60}")
        print(f"  SDM Model  (ML, Elhorst 2014)")
        print(f"{'═'*60}")
        print(f"  ρ = {res['rho']:.4f}  SE={res['rho_se']:.4f}  "
              f"t={res['rho_t']:.3f}  p={res['rho_p']:.4f}{_format_stars(res['rho_p'])}")
        print(f"  σ² = {res['sigma2']:.4f}")
        print(f"\n  Coefficient estimates (β and θ):")
        print(f"  {'Variable':<14} {'β':>9} {'SE':>9} {'t':>9} {'p':>8}  {'θ(WX)':>9} {'SE':>9} {'t':>9} {'p':>8}")
        print(f"  {'─'*90}")
        for j, v in enumerate(var_names):
            b=res['beta_X'][j]; sb=res['beta_X_se'][j]
            tb=res['beta_X_t'][j]; pb=res['beta_X_p'][j]
            t=res['theta_WX'][j]; st=res['theta_WX_se'][j]
            tt=res['theta_WX_t'][j]; pt=res['theta_WX_p'][j]
            print(f"  {v:<14} {b:>9.4f} {sb:>9.4f} {tb:>9.3f} {pb:>8.4f}{_format_stars(pb)}  "
                  f"{t:>9.4f} {st:>9.4f} {tt:>9.3f} {pt:>8.4f}{_format_stars(pt)}")
        print_effects_table(res, var_names)
