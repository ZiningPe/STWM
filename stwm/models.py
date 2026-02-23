"""
models.py
---------
Spatial econometric models: SLX, SAR, SEM, SDM.

All models support direct / indirect / total effect decomposition following
LeSage & Pace (2009). Panel data is handled by stacking observations as
a (nT,) vector (time-major: spatial units are the inner loop).

Model Definitions
-----------------

SLX  (Spatial Lag of X):
    Y = X β + WX θ + ε
    ε ~ N(0, σ² I)

    Direct effect:    β
    Indirect effect:  θ
    Total effect:     β + θ

    Estimation: OLS (no simultaneity — most transparent baseline).

SAR  (Spatial Autoregressive / Spatial Lag):
    Y = ρ W Y + X β + ε
    ε ~ N(0, σ² I)

    Reduced form:  Y = (I − ρW)⁻¹ X β + (I − ρW)⁻¹ ε

    Effect decomposition (LeSage & Pace, 2009):
        S(W) = (I − ρW)⁻¹ β_k (scalar per variable k)
        Direct effect:    (1/n) tr[S(W)]
        Total effect:     (1/n) 1' S(W) 1
        Indirect effect:  Total − Direct

    Estimation: Maximum Likelihood (concentrated log-likelihood over ρ).

SEM  (Spatial Error):
    Y = X β + u
    u = λ W u + ε,    ε ~ N(0, σ² I)

    Equivalent to GLS with error covariance (I − λW)'(I − λW):
        (I − λW) Y = (I − λW) X β + ε

    Direct effect:    β   (no spatial spillover in mean)
    Indirect effect:  0
    Total effect:     β

    Estimation: Maximum Likelihood.

SDM  (Spatial Durbin):
    Y = ρ W Y + X β + WX θ + ε

    Effect decomposition:
        S_k(W) = (I − ρW)⁻¹ (I β_k + W θ_k)
        Direct:   (1/n) tr[S_k(W)]
        Total:    (1/n) 1' S_k(W) 1
        Indirect: Total − Direct

    Estimation: Maximum Likelihood.

Log-likelihood (SAR / SDM)
--------------------------
    ln L(ρ) = −(n/2) ln(2π σ²)
              + ln|I − ρW|
              − (1/(2σ²)) (AY − Xβ̂)' (AY − Xβ̂)

where A = I − ρW, and β̂ is concentrated out analytically.

References
----------
LeSage, J., & Pace, R. K. (2009). Introduction to Spatial Econometrics. CRC Press.
Elhorst, J. P. (2014). Spatial Econometrics: From Cross-Sectional Data to Big Data. Springer.
Gibbons, S., & Overman, H. G. (2012). Journal of Regional Science, 52(2), 172-191.
Halleck Vega, S., & Elhorst, J. P. (2015). Journal of Regional Science, 55(3), 339-363.
"""

import numpy as np
from scipy import stats, optimize
from typing import Tuple


# ---------------------------------------------------------------------------
# Internal OLS helper
# ---------------------------------------------------------------------------

def _ols_fit(Y: np.ndarray, X: np.ndarray) -> dict:
    """
    OLS: β = (X'X)⁻¹ X'Y
    """
    Y    = np.asarray(Y, dtype=float).ravel()
    X    = np.asarray(X, dtype=float)
    n, k = X.shape
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid   = Y - X @ beta
    sigma2  = float(resid @ resid) / (n - k)
    cov     = sigma2 * np.linalg.pinv(X.T @ X)
    se      = np.sqrt(np.diag(cov))
    t_stats = beta / se
    p_vals  = 2 * stats.t.sf(np.abs(t_stats), df=n - k)
    ss_res  = float(resid @ resid)
    ss_tot  = float(((Y - Y.mean()) ** 2).sum())
    R2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "beta": beta, "se": se, "t_stats": t_stats, "p_values": p_vals,
        "residuals": resid, "sigma2": sigma2, "cov": cov, "R2": R2,
    }


# ---------------------------------------------------------------------------
# SLX
# ---------------------------------------------------------------------------

class SLXModel:
    """
    Spatial Lag of X (SLX) Model.

    Recommended as a transparent benchmark (Gibbons & Overman 2012;
    Halleck Vega & Elhorst 2015). Estimated by OLS — no simultaneity.

    Model:  Y = ι α + X β + WX θ + ε
    """

    def __init__(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)
        self.results_ = None

    def fit(self, Y: np.ndarray, X: np.ndarray) -> "SLXModel":
        Y  = np.asarray(Y, dtype=float).ravel()
        X  = np.asarray(X, dtype=float)
        WX = self.W @ X
        X_aug = np.column_stack([np.ones(len(Y)), X, WX])
        res   = _ols_fit(Y, X_aug)
        k     = X.shape[1]
        beta  = res["beta"][1:k+1]
        theta = res["beta"][k+1:]
        self.results_ = {
            **res,
            "model": "SLX",
            "beta_X"   : beta,
            "theta_WX" : theta,
            "direct"   : beta,
            "indirect" : theta,
            "total"    : beta + theta,
        }
        return self

    def summary(self) -> dict:
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_


# ---------------------------------------------------------------------------
# SAR
# ---------------------------------------------------------------------------

class SpatialLagModel:
    """
    Spatial Autoregressive Model (SAR).

    Model:  Y = ρ W Y + X β + ε
    """

    def __init__(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)
        self.results_ = None

    def fit(self, Y: np.ndarray, X: np.ndarray,
            rho_bounds: Tuple[float, float] = (-0.99, 0.99)) -> "SpatialLagModel":
        Y     = np.asarray(Y, dtype=float).ravel()
        X     = np.asarray(X, dtype=float)
        n     = len(Y)
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
        rho_hat = opt.x
        A       = np.eye(n) - rho_hat * W
        AY      = A @ Y
        beta    = np.linalg.lstsq(X_aug, AY, rcond=None)[0]
        resid   = AY - X_aug @ beta
        sigma2  = float(resid @ resid) / (n - X_aug.shape[1])

        S      = np.linalg.inv(A)
        k      = X.shape[1]
        direct   = np.array([float(np.trace(S * beta[1+i])) / n for i in range(k)])
        total_v  = np.array([float(np.mean(S @ (beta[1+i] * np.ones(n)))) for i in range(k)])
        indirect = total_v - direct

        self.results_ = {
            "model": "SAR", "rho": rho_hat,
            "beta": beta[1:], "intercept": beta[0], "sigma2": sigma2,
            "direct": direct, "indirect": indirect, "total": total_v,
        }
        return self

    def summary(self) -> dict:
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_


# ---------------------------------------------------------------------------
# SEM
# ---------------------------------------------------------------------------

class SpatialErrorModel:
    """
    Spatial Error Model (SEM).

    Model:  Y = X β + u,   u = λ W u + ε
    """

    def __init__(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)
        self.results_ = None

    def fit(self, Y: np.ndarray, X: np.ndarray,
            lam_bounds: Tuple[float, float] = (-0.99, 0.99)) -> "SpatialErrorModel":
        Y     = np.asarray(Y, dtype=float).ravel()
        X     = np.asarray(X, dtype=float)
        n     = len(Y)
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
        lam_hat = opt.x
        B       = np.eye(n) - lam_hat * W
        BY, BX  = B @ Y, B @ X_aug
        beta    = np.linalg.lstsq(BX, BY, rcond=None)[0]
        resid   = BY - BX @ beta
        sigma2  = float(resid @ resid) / (n - X_aug.shape[1])

        self.results_ = {
            "model": "SEM", "lambda": lam_hat,
            "beta": beta[1:], "intercept": beta[0], "sigma2": sigma2,
            "direct": beta[1:], "indirect": np.zeros_like(beta[1:]),
            "total": beta[1:],
        }
        return self

    def summary(self) -> dict:
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_


# ---------------------------------------------------------------------------
# SDM
# ---------------------------------------------------------------------------

class SDMModel:
    """
    Spatial Durbin Model (SDM).

    Model:  Y = ρ W Y + X β + WX θ + ε
    """

    def __init__(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)
        self.results_ = None

    def fit(self, Y: np.ndarray, X: np.ndarray,
            rho_bounds: Tuple[float, float] = (-0.99, 0.99)) -> "SDMModel":
        Y     = np.asarray(Y, dtype=float).ravel()
        X     = np.asarray(X, dtype=float)
        n     = len(Y)
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
        rho_hat = opt.x
        A       = np.eye(n) - rho_hat * W
        AY      = A @ Y
        beta    = np.linalg.lstsq(X_aug, AY, rcond=None)[0]
        resid   = AY - X_aug @ beta
        sigma2  = float(resid @ resid) / (n - X_aug.shape[1])

        k      = X.shape[1]
        beta_X = beta[1:k+1]
        theta  = beta[k+1:]
        S      = np.linalg.inv(A)

        direct = np.array([
            float(np.trace(S @ (np.eye(n) * beta_X[i] + W * theta[i]))) / n
            for i in range(k)
        ])
        total_v = np.array([
            float(np.ones(n) @ S @ ((beta_X[i] + theta[i]) * np.ones(n))) / n
            for i in range(k)
        ])
        indirect = total_v - direct

        self.results_ = {
            "model": "SDM", "rho": rho_hat,
            "beta_X": beta_X, "theta_WX": theta,
            "intercept": beta[0], "sigma2": sigma2,
            "direct": direct, "indirect": indirect, "total": total_v,
        }
        return self

    def summary(self) -> dict:
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_
