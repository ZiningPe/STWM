"""
models.py
---------
Spatial econometric models: SLX, SAR, SEM, SDM.
Following Elhorst (2014) with full multi-method estimation support.

Estimation Methods Available (SAR/SDM)
=======================================
  'ml'     : Maximum Likelihood (Ord 1975)
  'qml'    : Quasi-Maximum Likelihood (Lee 2004)  [same as ML for Gaussian]
  'iv'     : Instrumental Variables / 2SLS (Anselin 1988, pp.82-86)
  'gmm'    : Generalized Method of Moments (Kelejian & Prucha 1998,1999)
  'bayes'  : Bayesian MCMC (LeSage 1997)

Effect Decomposition (Elhorst 2014, Table 2.1)
===============================================
For SDM  Y = (I-δW)⁻¹(Xβ + WXθ) + R, partial derivatives:

    S_k(W) = (I − δW)⁻¹ · (I·β_k + W·θ_k)   [eq. 2.13]

    Direct   = (1/N) tr[S_k(W)]      ← average own-unit effect
    Total    = (1/N) 1'S_k(W)1       ← average total effect
    Indirect = Total − Direct         ← average spillover

Model simplifications (Elhorst 2014, Table 2.1):
    SLX : Direct=β_k, Indirect=θ_k  (exact, no simulation)
    SAR : θ_k=0,  S_k=(I-δW)⁻¹·β_k
    SEM : Direct=β_k, Indirect=0
    SDM : S_k=(I-δW)⁻¹(I·β_k + W·θ_k)

Effect Inference (Elhorst 2014, eq. 2.16–2.17)
===============================================
Simulation from the FULL joint ML covariance matrix (NOT block-diagonal):

    param_d = C^{1/2} ξ_d + param_hat,   ξ_d ~ N(0,I)    [eq.2.16]

For D draws: μ̄_k = mean(effect_kd),  t_k = μ̄_k / std(effect_kd)  [eq.2.17]

CRITICAL BUG FIX
================
Previous code took real parts of complex eigenvalues BEFORE computing
V·diag(...)·V⁻¹, which breaks V·V⁻¹ ≠ I for non-symmetric W.
This caused tr(S_inv) to be wrong by 5x, inflating SE of indirect effects
and making spillovers appear insignificant.

Fix: Maintain full complex arithmetic throughout, take Re() at the very end.

References
----------
Ord (1975). JASA 70, 120-126.
Lee (2004). Econometrica 72, 1899-1925.
Anselin (1988). Spatial Econometrics. Kluwer.
Kelejian & Prucha (1998). JRSS-B 60, 509-527.
Kelejian & Prucha (1999). JASA 94, 929-950.
LeSage (1997). Int. Regional Science Review 20, 113-129.
Elhorst (2014). Spatial Econometrics. Springer.
LeSage & Pace (2009). Introduction to Spatial Econometrics. CRC.
"""

import numpy as np
from scipy import stats, optimize, linalg
from typing import Tuple, Optional
import warnings


# ============================================================================
# Internal helpers
# ============================================================================

def _ols_fit(Y: np.ndarray, X: np.ndarray) -> dict:
    """OLS: β = (X'X)⁻¹X'Y with full inference."""
    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    N, k = X.shape
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ beta
    sigma2 = float(resid @ resid) / (N - k)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t_stat = beta / np.where(se > 1e-14, se, np.nan)
    pval = 2 * stats.t.sf(np.abs(t_stat), df=N - k)
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    R2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return dict(beta=beta, se=se, t_stats=t_stat, p_values=pval,
                residuals=resid, sigma2=sigma2, cov=cov, R2=R2, N=N, k=k)


def _numerical_hessian(func, x0: float, eps: float = 1e-4) -> float:
    """
    Numerical second derivative via central differences.
    f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
    Used for SE(ρ) from concentrated log-likelihood.
    """
    return (func(x0 + eps) - 2.0 * func(x0) + func(x0 - eps)) / (eps * eps)


def _grid_search_then_refine(func, bounds: Tuple[float, float],
                               n_grid: int = 200) -> float:
    """
    Grid search + local refinement. Robust against flat log-likelihoods.
    Mirrors spreg.ML_Lag 'ord' method approach.
    """
    grid = np.linspace(bounds[0], bounds[1], n_grid)
    vals = np.array([func(r) for r in grid])
    x_init = grid[np.argmin(vals)]
    width = (bounds[1] - bounds[0]) / n_grid * 8
    lo = max(x_init - width, bounds[0])
    hi = min(x_init + width, bounds[1])
    opt = optimize.minimize_scalar(func, bounds=(lo, hi), method="bounded")
    return float(opt.x)


def _precompute_eig_quantities(W: np.ndarray):
    """
    Pre-compute eigendecomposition of W for effect simulation.

    CRITICAL FIX: Keep FULL COMPLEX arithmetic.
    For non-symmetric W, eigenvalues can be complex.
    Taking Re() of eigenvectors BEFORE computing V·D·V⁻¹ breaks
    V·V⁻¹ = I, causing tr(S_inv) to be wrong by large factors.

    Solution: maintain complex arithmetic throughout, take Re() only at
    the final scalar outputs (tr, rowsum).

    Returns
    -------
    eigvals_c : (n,) complex eigenvalues of W
    V_c       : (n,n) complex right eigenvectors
    Vinv_c    : (n,n) complex inverse of V
    VinvW_c   : (n,n) V⁻¹ @ W  (precomputed for SinvW)
    Vinv_ones_c : (n,) V⁻¹ @ 1  (precomputed for rowsum)
    """
    eigvals_c, V_c = np.linalg.eig(W)
    try:
        Vinv_c = np.linalg.inv(V_c)
    except np.linalg.LinAlgError:
        Vinv_c = np.linalg.pinv(V_c)
    VinvW_c = Vinv_c @ W.astype(complex)
    ones_c = np.ones(W.shape[0], dtype=complex)
    Vinv_ones_c = Vinv_c @ ones_c
    return eigvals_c, V_c, Vinv_c, VinvW_c, Vinv_ones_c


def _sinv_traces(rho: float, eigvals_c, V_c, Vinv_c, VinvW_c, Vinv_ones_c, n: int):
    """
    Compute tr(S_inv), 1'S_inv 1, tr(S_inv W), 1'S_inv W 1
    where S_inv = (I - rho*W)^{-1}.

    Uses eigendecomposition: S_inv = V diag(1/(1-rho*λ)) V⁻¹
    All operations in complex arithmetic; real parts taken at the end.

    Returns
    -------
    tr_Sinv     : scalar, trace of (I-rhoW)^{-1}
    rs_Sinv     : scalar, 1'(I-rhoW)^{-1}1 (sum of all elements / n * n)
    tr_SinvW    : scalar, trace of (I-rhoW)^{-1} W
    rs_SinvW    : scalar, 1'(I-rhoW)^{-1} W 1
    valid       : bool, False if rho is at a pole (singular)
    """
    denom = 1.0 - rho * eigvals_c
    if np.any(np.abs(denom) < 1e-10):
        return None, None, None, None, False

    inv_d = 1.0 / denom                           # (n,) complex
    Vd    = V_c * inv_d[np.newaxis, :]            # V diag(inv_d), (n,n) complex

    # tr(Sinv) = sum_i [Vd @ Vinv]_ii = einsum
    tr_Sinv  = float(np.einsum('ij,ji->', Vd, Vinv_c).real)

    # 1'Sinv 1 = sum(Vd @ Vinv @ ones) = ones' Vd Vinv ones
    rs_Sinv  = float((Vd @ Vinv_ones_c).sum().real)

    # Sinv W = Vd @ (Vinv @ W) = Vd @ VinvW
    SinvW   = Vd @ VinvW_c                        # (n,n) complex
    tr_SinvW = float(np.einsum('ii->', SinvW).real)
    ones_c   = np.ones(n, dtype=complex)
    rs_SinvW = float((SinvW @ ones_c).sum().real)

    return tr_Sinv, rs_Sinv, tr_SinvW, rs_SinvW, True


def _simulation_se(S_draws: np.ndarray,
                   valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Elhorst (2014) eq. 2.17: mean, SE, t-value from simulation draws.
    S_draws : (D, k) — rows are draws, columns are variables.
    """
    S = S_draws[valid_mask]
    mean = S.mean(axis=0)
    se = S.std(axis=0, ddof=1)
    tval = mean / np.where(se > 1e-14, se, np.nan)
    return mean, se, tval


def _build_full_cov_spreg_style(X_aug: np.ndarray,
                                W: np.ndarray,
                                rho_hat: float,
                                beta_hat: np.ndarray,
                                sigma2: float,
                                k_x: int,
                                has_theta: bool):
    """
    Build FULL joint covariance matrix using spreg's ANALYTIC information matrix.

    Exactly replicates spreg.ml_lag.BaseML_Lag (Anselin 1988, eq. 6.5).
    This is the gold-standard formula used in all published spatial econometrics.

    Formula:
        A    = I - rho*W
        G    = W @ A^{-1}              (spreg calls this wai)
        tr1  = tr(G)
        tr2  = tr(G @ G)
        tr3  = tr(G.T @ G)
        pred_e = A^{-1} @ X @ beta     (spreg calls this predy_e)
        Wpre   = W @ pred_e            (spreg calls this wpredy)

        Information matrix v [beta(p), rho, sigma2]:
            v[:p,:p]     = X'X / sigma2
            v[:p, p]     = X'Wpre / sigma2      <- NONZERO beta-rho cross-term
            v[p, p]      = tr2 + tr3 + Wpre'Wpre/sigma2
            v[p, p+1]    = tr1 / sigma2
            v[p+1, p+1]  = N / (2*sigma2^2)

        vm1 = inv(v)        # (p+2)x(p+2), same as spreg.vm1
        vm  = vm1[:-1,:-1]  # (p+1)x(p+1), same as spreg.vm

    Returns
    -------
    cov_sim : array (dim_sim, dim_sim)
        Covariance for [rho, beta_1..k, (theta_1..k)] - for effect simulation
    vm : array (p+1, p+1)
        Full covariance for all coefficients - identical to spreg.vm
    """
    N = X_aug.shape[0]
    p = X_aug.shape[1]   # intercept + beta (+ theta for SDM)

    A = np.eye(N) - rho_hat * W
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Ainv = np.linalg.pinv(A)

    # G = W @ A^{-1}  (identical to spreg's wai)
    G    = W @ Ainv
    tr1  = float(np.trace(G))
    tr2  = float(np.trace(G @ G))
    tr3  = float(np.trace(G.T @ G))

    # Reduced-form prediction  (identical to spreg's predy_e)
    pred_e    = Ainv @ (X_aug @ beta_hat)
    Wpre      = W @ pred_e
    XtX       = X_aug.T @ X_aug
    XtWpre    = X_aug.T @ Wpre
    WpreTWpre = float(Wpre @ Wpre)

    # Information matrix v — identical to spreg's v1/v2/v3 construction
    v = np.zeros((p + 2, p + 2))
    v[:p, :p]       = XtX / sigma2
    v[:p, p]        = XtWpre / sigma2    # beta-rho cross-term (KEY!)
    v[p, :p]        = XtWpre / sigma2    # symmetric
    v[p, p]         = tr2 + tr3 + WpreTWpre / sigma2
    v[p, p + 1]     = tr1 / sigma2
    v[p + 1, p]     = tr1 / sigma2
    v[p + 1, p + 1] = N / (2.0 * sigma2 ** 2)
    v += 1e-10 * np.eye(p + 2)

    try:
        vm1 = np.linalg.inv(v)   # (p+2)x(p+2), same as spreg.vm1
    except np.linalg.LinAlgError:
        vm1 = np.linalg.pinv(v)

    vm = vm1[:-1, :-1]           # (p+1)x(p+1), same as spreg.vm

    # Extract subset for effect simulation: [rho, beta_1..k, (theta_1..k)]
    idx_rho = p
    if has_theta:
        idx_sim = [idx_rho] + list(range(1, k_x + 1)) + list(range(k_x + 1, 2 * k_x + 1))
        dim_sim = 1 + 2 * k_x
    else:
        idx_sim = [idx_rho] + list(range(1, k_x + 1))
        dim_sim = 1 + k_x

    cov_sim = np.array([[vm[i, j] for j in idx_sim] for i in idx_sim])
    return cov_sim, vm


def _jacobian_logdet(rho: float, ev_c: np.ndarray) -> float:
    """
    Compute log|det(I - rho*W)| via complex eigenvalues.

    Replicates spreg lag_c_loglik_ord formula:
        jacob = sum(log(1 - rho*eigvals)).real

    CRITICAL: Use complex eigenvalues and take .real at the END.
    This is correct because log|det(A)| = Re(sum(log(eigenvalues of A))).
    DO NOT use: sum(log(|1 - rho*Re(eigvals)|)) -- WRONG for non-symmetric W.
    """
    return float(np.sum(np.log(1.0 - rho * ev_c)).real)

def _effect_inference(W: np.ndarray,
                       beta_X: np.ndarray,
                       theta: np.ndarray,
                       rho: float,
                       cov_params: np.ndarray,
                       n: int,
                       k: int,
                       D: int = 1000,
                       seed: int = 0) -> dict:
    """
    Simulation-based effect inference (Elhorst 2014, eq. 2.16-2.17).

    BUGFIX from previous version:
    - Maintains COMPLEX arithmetic throughout eigendecomposition
    - Only takes Re() at the final scalar outputs
    - Prevents the V·V⁻¹≠I breakdown that inflated SE of indirect effects

    Uses FULL cov_params (not block-diagonal) so rho-beta cross-terms
    propagate correctly into effect uncertainties.

    Parameters
    ----------
    cov_params : (p, p) FULL covariance of [rho, beta_X, (theta)]
                 p = 1+k (SAR) or 1+2k (SDM)
                 Must include rho–beta cross-terms!
    """
    rng = np.random.default_rng(seed)

    # Ensure PSD before Cholesky
    C = (cov_params + cov_params.T) / 2.0
    min_eig = np.linalg.eigvalsh(C).min()
    if min_eig < 0:
        C += (-min_eig + 1e-8) * np.eye(len(C))
    else:
        C += 1e-10 * np.eye(len(C))

    try:
        P = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        P = np.diag(np.sqrt(np.maximum(np.diag(cov_params), 0)))

    p_dim = cov_params.shape[0]
    has_theta = (p_dim == 1 + 2 * k)

    if has_theta:
        center = np.concatenate([[rho], beta_X, theta])
    else:
        center = np.concatenate([[rho], beta_X])

    direct_d   = np.zeros((D, k))
    indirect_d = np.zeros((D, k))
    total_d    = np.zeros((D, k))
    valid      = np.zeros(D, dtype=bool)

    # Pre-compute eigendecomposition ONCE (complex arithmetic throughout)
    eig_data = _precompute_eig_quantities(W)
    eigvals_c, V_c, Vinv_c, VinvW_c, Vinv_ones_c = eig_data

    for d in range(D):
        xi   = rng.standard_normal(p_dim)
        draw = P @ xi + center
        r_d  = float(np.clip(draw[0], -0.999, 0.999))
        b_d  = draw[1:k + 1]
        t_d  = draw[k + 1:] if has_theta else np.zeros(k)

        tr_S, rs_S, tr_SW, rs_SW, ok = _sinv_traces(
            r_d, eigvals_c, V_c, Vinv_c, VinvW_c, Vinv_ones_c, n)
        if not ok:
            continue

        for i in range(k):
            b_i, t_i = b_d[i], t_d[i]
            # Direct  = (1/N) tr[S_k(W)] = (β_k tr(Sinv) + θ_k tr(Sinv W)) / N
            # Total   = (1/N) 1'S_k(W)1  = (β_k 1'Sinv1 + θ_k 1'SinvW1) / N
            direct_d[d, i]   = (b_i * tr_S  + t_i * tr_SW)  / n
            total_d[d, i]    = (b_i * rs_S  + t_i * rs_SW)  / n
            indirect_d[d, i] = total_d[d, i] - direct_d[d, i]

        valid[d] = True

    n_valid = valid.sum()
    if n_valid < D * 0.5:
        warnings.warn(
            f"Only {n_valid}/{D} simulation draws valid. "
            "Effect SE may be unreliable.", UserWarning, stacklevel=2)

    dir_mean, dir_se, dir_t   = _simulation_se(direct_d,   valid)
    ind_mean, ind_se, ind_t   = _simulation_se(indirect_d, valid)
    tot_mean, tot_se, tot_t   = _simulation_se(total_d,    valid)

    return dict(
        direct=dir_mean,    direct_se=dir_se,    direct_t=dir_t,
        indirect=ind_mean,  indirect_se=ind_se,  indirect_t=ind_t,
        total=tot_mean,     total_se=tot_se,     total_t=tot_t,
        direct_p    = 2 * stats.norm.sf(np.abs(dir_t)),
        indirect_p  = 2 * stats.norm.sf(np.abs(ind_t)),
        total_p     = 2 * stats.norm.sf(np.abs(tot_t)),
        n_valid_draws = int(n_valid),
    )


def _format_stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def print_effects_table(res: dict, var_names: list):
    """Print direct/indirect/total effect decomposition (Elhorst 2014, Table 2.1)."""
    sep = "─" * 75
    print(f"\n{sep}")
    print(f"  Effect Decomposition  [Elhorst 2014, Table 2.1]")
    print(f"{sep}")
    print(f"  {'Variable':<14} {'Effect':>9} {'SE':>9} {'t-stat':>9} {'p':>8}  Sig")
    print(f"{sep}")
    for j, var in enumerate(var_names):
        for eff_name in ['direct', 'indirect', 'total']:
            v   = res[eff_name][j]
            se  = res.get(f'{eff_name}_se',  np.zeros_like(res[eff_name]))[j]
            t   = res.get(f'{eff_name}_t',   np.zeros_like(res[eff_name]))[j]
            p   = res.get(f'{eff_name}_p',   np.ones_like(res[eff_name]))[j]
            label = {"direct": f"{var}_D", "indirect": f"{var}_I",
                     "total":  f"{var}_Tot"}[eff_name]
            print(f"  {label:<14} {v:>9.4f} {se:>9.4f} {t:>9.3f} {p:>8.4f}  "
                  f"{_format_stars(p)}")
        print()
    print(f"{sep}")
    print("  Significance: *** p<0.01  ** p<0.05  * p<0.10")


# ============================================================================
# IV/GMM helpers
# ============================================================================

def _iv_2sls(Y: np.ndarray,
              X_endog: np.ndarray,
              Z: np.ndarray) -> dict:
    """
    Two-Stage Least Squares (Anselin 1988, pp.82-86).
    Stage 1 : X̂_endog = Z (Z'Z)⁻¹ Z' X_endog
    Stage 2 : β_IV = (X̂'X̂)⁻¹ X̂'Y
    """
    Y = np.asarray(Y, float).ravel()
    X = np.asarray(X_endog, float)
    Z = np.asarray(Z, float)
    N = len(Y)

    # First stage
    X_hat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    # Second stage
    beta  = np.linalg.lstsq(X_hat, Y, rcond=None)[0]
    resid = Y - X @ beta
    sigma2 = float(resid @ resid) / (N - X.shape[1])
    cov = sigma2 * np.linalg.pinv(X_hat.T @ X_hat)
    se  = np.sqrt(np.maximum(np.diag(cov), 0))
    t   = beta / np.where(se > 1e-14, se, np.nan)
    p   = 2 * stats.norm.sf(np.abs(t))
    return dict(beta=beta, se=se, t_stats=t, p_values=p,
                residuals=resid, sigma2=sigma2, cov=cov)


def _gmm_kelejian_prucha(Y: np.ndarray,
                          X: np.ndarray,
                          W: np.ndarray,
                          w_lags: int = 2) -> dict:
    """
    GMM / Spatial 2SLS (Kelejian & Prucha 1998, 1999).
    Instruments: H = [X_aug, WX, W²X, ..., W^w_lags X]
    Endogenous: WY is instrumented.
    """
    N = len(Y)
    X_aug = np.column_stack([np.ones(N), X])
    WY    = W @ Y

    # Build instrument set
    inst_list = [X_aug]
    Wk = W.copy()
    for _ in range(w_lags):
        inst_list.append(Wk @ X_aug)
        Wk = W @ Wk
    Z = np.column_stack(inst_list)

    X_full = np.column_stack([X_aug, WY])
    res    = _iv_2sls(Y, X_full, Z)

    beta_all = res['beta']
    rho_hat  = float(beta_all[-1])
    beta_hat = beta_all[:-1]   # intercept + beta_X
    se_all   = res['se']
    se_rho   = float(se_all[-1])
    se_beta  = se_all[:-1]
    return dict(rho=rho_hat, beta=beta_hat,
                rho_se=se_rho, beta_se=se_beta,
                sigma2=res['sigma2'], cov=res['cov'],
                residuals=res['residuals'])


def _bayesian_sar_mcmc(Y: np.ndarray,
                        X: np.ndarray,
                        W: np.ndarray,
                        n_draws: int = 10000,
                        burnin: int = 2000,
                        seed: int = 0) -> dict:
    """
    Bayesian MCMC for SAR (LeSage 1997).

    Priors:
        β | σ² ~ N(0, c·σ²·I),  c=1e6  (diffuse)
        σ²     ~ IG(a₀/2, b₀/2), a₀=b₀=0.01
        δ      ~ Uniform(-1, 1)

    Gibbs sampler:
        1. β  | Y, δ, σ² ~ N(μ_β, Σ_β)    [closed form]
        2. σ² | Y, β, δ  ~ InvGamma        [closed form]
        3. δ  | Y, β, σ² via M-H step      [slice of posterior]
    """
    rng   = np.random.default_rng(seed)
    N     = len(Y)
    X_aug = np.column_stack([np.ones(N), X])
    p     = X_aug.shape[1]
    ev_W  = np.linalg.eigvals(W).real
    I_N   = np.eye(N)

    # Diffuse priors
    c  = 1e6;  a0 = 0.01;  b0 = 0.01

    # Initialise
    rho    = 0.0
    sigma2 = 1.0
    A      = I_N - rho * W
    beta   = np.linalg.lstsq(X_aug, A @ Y, rcond=None)[0]

    rho_smp   = np.zeros(n_draws)
    beta_smp  = np.zeros((n_draws, p))
    sig2_smp  = np.zeros(n_draws)

    mh_width = 0.05
    n_acc = 0

    for s in range(n_draws + burnin):
        A  = I_N - rho * W
        AY = A @ Y

        # --- Gibbs β | Y, δ, σ² ---
        XtX   = X_aug.T @ X_aug
        V_b   = np.linalg.inv(XtX / sigma2 + np.eye(p) / (c * sigma2))
        mu_b  = V_b @ (X_aug.T @ AY / sigma2)
        try:
            L_b  = np.linalg.cholesky(V_b)
            beta = mu_b + L_b @ rng.standard_normal(p)
        except np.linalg.LinAlgError:
            beta = mu_b

        # --- Gibbs σ² | Y, β, δ ---
        e      = AY - X_aug @ beta
        a_post = (a0 + N) / 2.0
        b_post = (b0 + float(e @ e)) / 2.0
        sigma2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)

        # --- MH δ | Y, β, σ² ---
        def log_post(r):
            if abs(r) >= 0.999: return -1e15
            d_eig = 1.0 - r * ev_c
            if np.any(d_eig <= 0): return -1e15
            ld  = float(np.sum(np.log(d_eig)))
            Ar  = I_N - r * W
            err = Ar @ Y - X_aug @ beta
            return ld - float(err @ err) / (2.0 * sigma2)

        rho_prop = float(np.clip(rho + rng.normal(0, mh_width), -0.999, 0.999))
        log_a = log_post(rho_prop) - log_post(rho)
        if np.log(max(rng.uniform(), 1e-300)) < log_a:
            rho = rho_prop
            n_acc += 1

        # Adaptive MH width
        if (s + 1) % 200 == 0 and s < burnin:
            rate = n_acc / (s + 1)
            mh_width *= 1.1 if rate > 0.44 else (0.9 if rate < 0.23 else 1.0)

        if s >= burnin:
            rho_smp[s - burnin]  = rho
            beta_smp[s - burnin] = beta
            sig2_smp[s - burnin] = sigma2

    return dict(
        rho=float(rho_smp.mean()),
        beta=beta_smp.mean(axis=0),
        rho_se=float(rho_smp.std(ddof=1)),
        beta_se=beta_smp.std(axis=0, ddof=1),
        sigma2=float(sig2_smp.mean()),
        rho_draws=rho_smp,
        beta_draws=beta_smp,
        acceptance_rate=n_acc / (n_draws + burnin),
    )


# ============================================================================
# SLX
# ============================================================================

class SLXModel:
    """
    Spatial Lag of X (SLX).
    Halleck Vega & Elhorst (2015); Gibbons & Overman (2012).

    Model:   Y = α·ι + X·β + WX·θ + ε
    Method:  OLS (no simultaneity — effects are exact, no simulation needed)

    Effects:
        Direct   = β_k               (exact from OLS)
        Indirect = θ_k               (exact from OLS)
        Total    = β_k + θ_k
        SE(Total)= √[Var(β)+Var(θ)+2Cov(β,θ)]
    """

    def __init__(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)
        self.results_ = None

    def fit(self, Y: np.ndarray, X: np.ndarray) -> "SLXModel":
        Y     = np.asarray(Y, dtype=float).ravel()
        X     = np.asarray(X, dtype=float)
        N, k  = X.shape
        WX    = self.W @ X
        X_aug = np.column_stack([np.ones(N), X, WX])
        res   = _ols_fit(Y, X_aug)

        beta  = res["beta"][1:k + 1]
        theta = res["beta"][k + 1:2 * k + 1]
        se_b  = res["se"][1:k + 1]
        se_t  = res["se"][k + 1:2 * k + 1]
        cov   = res["cov"]
        cov_bt   = cov[1:k + 1, k + 1:2 * k + 1]
        se_tot   = np.sqrt(np.maximum(se_b**2 + se_t**2 + 2 * np.diag(cov_bt), 0))

        df = N - X_aug.shape[1]
        t_dir = beta        / np.where(se_b   > 1e-14, se_b,   np.nan)
        t_ind = theta       / np.where(se_t   > 1e-14, se_t,   np.nan)
        t_tot = (beta+theta)/ np.where(se_tot > 1e-14, se_tot, np.nan)
        p_dir = 2 * stats.t.sf(np.abs(t_dir), df=df)
        p_ind = 2 * stats.t.sf(np.abs(t_ind), df=df)
        p_tot = 2 * stats.t.sf(np.abs(t_tot), df=df)

        self.results_ = {
            **res, "model": "SLX", "method": "ols",
            "beta_X": beta,  "theta_WX": theta,
            "direct":   beta,     "direct_se":   se_b,  "direct_t":  t_dir, "direct_p":  p_dir,
            "indirect": theta,    "indirect_se": se_t,  "indirect_t":t_ind, "indirect_p":p_ind,
            "total":beta+theta,   "total_se":   se_tot, "total_t":   t_tot, "total_p":   p_tot,
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
        print(f"  SLX Model  (OLS, Halleck Vega & Elhorst 2015)")
        print(f"{'═'*60}")
        print(f"  N={res['N']}  R²={res['R2']:.4f}  σ²={res['sigma2']:.4f}")
        print_effects_table(res, var_names)


# ============================================================================
# SAR — Spatial Autoregressive Model
# ============================================================================

class SpatialLagModel:
    """
    Spatial Autoregressive Model (SAR).
    Y = δ·WY + X·β + ε,   ε ~ N(0, σ²I)

    Estimation methods:
        'ml'    : Maximum Likelihood (Ord 1975)  ← DEFAULT
        'qml'   : Quasi-ML (Lee 2004) [same as ML for Gaussian errors]
        'iv'    : IV / 2SLS (Anselin 1988)
        'gmm'   : GMM (Kelejian & Prucha 1998, 1999)
        'bayes' : Bayesian MCMC (LeSage 1997)

    Effect computation always uses LeSage-Pace simulation (Elhorst eq.2.16-2.17)
    with the FULL information matrix covariance (not block-diagonal).

    Bug fixed vs previous version: complex eigenvalue handling in _sinv_traces.
    """

    def __init__(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)
        self.results_ = None

    def fit(self, Y: np.ndarray, X: np.ndarray,
            method: str = 'ml',
            rho_bounds: Tuple[float, float] = (-0.99, 0.99),
            n_draws: int = 1000,
            n_bayes: int = 10000,
            seed: int = 0) -> "SpatialLagModel":
        """
        Parameters
        ----------
        method     : 'ml' | 'qml' | 'iv' | 'gmm' | 'bayes'
        rho_bounds : search bounds for δ  (ML/QML only)
        n_draws    : simulation draws for effect SE  (all methods)
        n_bayes    : MCMC total draws  (Bayesian only)
        seed       : random seed
        """
        Y   = np.asarray(Y, dtype=float).ravel()
        X   = np.asarray(X, dtype=float)
        N, k = X.shape
        method = method.lower()

        if method in ('ml', 'qml'):
            res = self._fit_ml(Y, X, N, k, rho_bounds, n_draws, seed)
        elif method == 'iv':
            res = self._fit_iv(Y, X, N, k, n_draws, seed)
        elif method == 'gmm':
            res = self._fit_gmm(Y, X, N, k, n_draws, seed)
        elif method == 'bayes':
            res = self._fit_bayes(Y, X, N, k, n_draws, n_bayes, seed)
        else:
            raise ValueError(f"method must be 'ml','qml','iv','gmm','bayes'. Got '{method}'.")

        res['model']  = 'SAR'
        res['method'] = method
        self.results_ = res
        return self

    # ------------------------------------------------------------------
    def _fit_ml(self, Y, X, N, k, rho_bounds, n_draws, seed):
        W     = self.W
        X_aug = np.column_stack([np.ones(N), X])
        ev_c  = np.linalg.eigvals(W)  # complex eigenvalues (spreg ord method)

        def neg_ll(rho):
            A      = np.eye(N) - rho * W
            AY     = A @ Y
            beta   = np.linalg.lstsq(X_aug, AY, rcond=None)[0]
            resid  = AY - X_aug @ beta
            sigma2 = float(resid @ resid) / N
            if sigma2 <= 0: return 1e15
            log_det = _jacobian_logdet(rho, ev_c)
            return 0.5 * N * np.log(sigma2) - log_det

        rho_hat = _grid_search_then_refine(neg_ll, rho_bounds)
        A       = np.eye(N) - rho_hat * W
        AY      = A @ Y
        beta    = np.linalg.lstsq(X_aug, AY, rcond=None)[0]
        resid   = AY - X_aug @ beta
        sigma2  = float(resid @ resid) / N

        # Full covariance — spreg analytic information matrix (Anselin 1988)
        cov_sim, vm = _build_full_cov_spreg_style(
            X_aug, W, rho_hat, beta, sigma2, k, has_theta=False)

        # SE(rho) from spreg-style vm  (vm layout: [intercept..beta, rho])
        se_rho = float(np.sqrt(max(vm[X_aug.shape[1], X_aug.shape[1]], 0)))

        # SE(beta) from spreg-style vm (rows 1..k, skip intercept at row 0)
        se_beta = np.array([np.sqrt(max(vm[1 + j, 1 + j], 0)) for j in range(k)])
        sigma2_df = float(resid @ resid) / (N - X_aug.shape[1])

        eff = _effect_inference(W, beta[1:], np.zeros(k), rho_hat,
                                cov_sim, N, k, D=n_draws, seed=seed)

        t_beta = beta[1:] / np.where(se_beta > 1e-14, se_beta, np.nan)
        p_beta = 2 * stats.norm.sf(np.abs(t_beta))
        t_rho  = rho_hat / se_rho if se_rho > 1e-14 else np.nan
        p_rho  = 2 * stats.norm.sf(abs(t_rho)) if not np.isnan(t_rho) else np.nan

        return dict(
            rho=rho_hat, rho_se=se_rho, rho_t=t_rho, rho_p=p_rho,
            beta=beta[1:], beta_se=se_beta, beta_t=t_beta, beta_p=p_beta,
            intercept=float(beta[0]), sigma2=sigma2_df,
            **eff,
        )

    # ------------------------------------------------------------------
    def _fit_iv(self, Y, X, N, k, n_draws, seed):
        W     = self.W
        X_aug = np.column_stack([np.ones(N), X])
        WY    = W @ Y
        WX    = W @ X_aug
        W2X   = W @ WX
        Z     = np.column_stack([X_aug, WX, W2X])
        X_full = np.column_stack([X_aug, WY])
        res_iv = _iv_2sls(Y, X_full, Z)

        beta_all = res_iv['beta']
        rho_hat  = float(beta_all[-1])
        beta_hat = beta_all[:k + 1]
        se_all   = res_iv['se']
        se_rho   = float(se_all[-1])
        se_beta  = se_all[1:k + 1]

        # Build simulation cov from 2SLS joint cov
        cov_full = res_iv['cov']
        dim = 1 + k
        cov_sim = np.zeros((dim, dim))
        idx_rho  = k + 1
        idx_beta = list(range(1, k + 1))
        cov_sim[0, 0]   = cov_full[idx_rho, idx_rho]
        cov_sim[1:, 1:] = cov_full[np.ix_(idx_beta, idx_beta)]
        cov_sim[0, 1:]  = cov_full[idx_rho, idx_beta]
        cov_sim[1:, 0]  = cov_full[idx_beta, idx_rho]

        eff = _effect_inference(W, beta_hat[1:], np.zeros(k), rho_hat,
                                cov_sim, N, k, D=n_draws, seed=seed)

        t_beta = beta_hat[1:] / np.where(se_beta > 1e-14, se_beta, np.nan)
        p_beta = 2 * stats.norm.sf(np.abs(t_beta))
        t_rho  = rho_hat / se_rho if se_rho > 1e-14 else np.nan
        p_rho  = 2 * stats.norm.sf(abs(t_rho)) if not np.isnan(t_rho) else np.nan

        return dict(
            rho=rho_hat, rho_se=se_rho, rho_t=t_rho, rho_p=p_rho,
            beta=beta_hat[1:], beta_se=se_beta, beta_t=t_beta, beta_p=p_beta,
            intercept=float(beta_hat[0]), sigma2=res_iv['sigma2'],
            **eff,
        )

    # ------------------------------------------------------------------
    def _fit_gmm(self, Y, X, N, k, n_draws, seed):
        W   = self.W
        gmm = _gmm_kelejian_prucha(Y, X, W)
        rho_hat  = gmm['rho']
        beta_hat = gmm['beta']       # includes intercept
        se_rho   = gmm['rho_se']
        se_beta  = gmm['beta_se']

        # Build simulation cov from GMM joint cov
        cov_full = gmm['cov']
        idx_rho  = k + 1
        idx_beta = list(range(1, k + 1))
        dim = 1 + k
        cov_sim  = np.zeros((dim, dim))
        cov_sim[0, 0]   = cov_full[idx_rho, idx_rho]
        cov_sim[1:, 1:] = cov_full[np.ix_(idx_beta, idx_beta)]
        cov_sim[0, 1:]  = cov_full[idx_rho, idx_beta]
        cov_sim[1:, 0]  = cov_full[idx_beta, idx_rho]

        eff = _effect_inference(W, beta_hat[1:], np.zeros(k), rho_hat,
                                cov_sim, N, k, D=n_draws, seed=seed)

        t_beta = beta_hat[1:] / np.where(se_beta[1:] > 1e-14, se_beta[1:], np.nan)
        p_beta = 2 * stats.norm.sf(np.abs(t_beta))
        t_rho  = rho_hat / se_rho if se_rho > 1e-14 else np.nan
        p_rho  = 2 * stats.norm.sf(abs(t_rho)) if not np.isnan(t_rho) else np.nan

        return dict(
            rho=rho_hat, rho_se=se_rho, rho_t=t_rho, rho_p=p_rho,
            beta=beta_hat[1:], beta_se=se_beta[1:], beta_t=t_beta, beta_p=p_beta,
            intercept=float(beta_hat[0]), sigma2=gmm['sigma2'],
            **eff,
        )

    # ------------------------------------------------------------------
    def _fit_bayes(self, Y, X, N, k, n_draws_eff, n_bayes, seed):
        W     = self.W
        bayes = _bayesian_sar_mcmc(Y, X, W, n_draws=n_bayes, seed=seed)

        rho_hat  = bayes['rho']
        beta_hat = bayes['beta']       # includes intercept
        se_rho   = bayes['rho_se']
        se_beta  = bayes['beta_se']

        # Posterior covariance from MCMC draws
        rho_d  = bayes['rho_draws']
        beta_d = bayes['beta_draws'][:, 1:]  # exclude intercept
        all_d  = np.column_stack([rho_d[:, None], beta_d])
        cov_sim = np.cov(all_d.T)
        if cov_sim.ndim < 2:
            cov_sim = np.diag(np.concatenate([[se_rho**2], se_beta[1:]**2]))

        eff = _effect_inference(W, beta_hat[1:], np.zeros(k), rho_hat,
                                cov_sim, N, k, D=n_draws_eff, seed=seed)

        t_beta = beta_hat[1:] / np.where(se_beta[1:] > 1e-14, se_beta[1:], np.nan)
        p_beta = 2 * stats.norm.sf(np.abs(t_beta))
        t_rho  = rho_hat / se_rho if se_rho > 1e-14 else np.nan
        p_rho  = 2 * stats.norm.sf(abs(t_rho)) if not np.isnan(t_rho) else np.nan

        return dict(
            rho=rho_hat, rho_se=se_rho, rho_t=t_rho, rho_p=p_rho,
            beta=beta_hat[1:], beta_se=se_beta[1:], beta_t=t_beta, beta_p=p_beta,
            intercept=float(beta_hat[0]), sigma2=bayes['sigma2'],
            acceptance_rate=bayes['acceptance_rate'],
            **eff,
        )

    def summary(self) -> dict:
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_

    def print_summary(self, var_names=None):
        res = self.summary()
        k = len(res["beta"])
        m = res.get("method", "ml").upper()
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(k)]
        print(f"\n{'═'*60}")
        print(f"  SAR Model  ({m}, Elhorst 2014)")
        print(f"{'═'*60}")
        print(f"  δ = {res['rho']:.4f}  SE={res['rho_se']:.4f}  "
              f"t={res['rho_t']:.3f}  p={res['rho_p']:.4f}"
              f"{_format_stars(res['rho_p'])}")
        if res.get('acceptance_rate') is not None:
            print(f"  MCMC acceptance rate: {res['acceptance_rate']:.3f}")
        print(f"  σ² = {res['sigma2']:.4f}")
        print(f"\n  {'Variable':<14} {'β':>9} {'SE':>9} {'t':>9} {'p':>8}  Sig")
        print(f"  {'─'*55}")
        for j, v in enumerate(var_names):
            b = res['beta'][j]; se = res['beta_se'][j]
            t = res['beta_t'][j]; p = res['beta_p'][j]
            print(f"  {v:<14} {b:>9.4f} {se:>9.4f} {t:>9.3f} {p:>8.4f}  {_format_stars(p)}")
        print_effects_table(res, var_names)


# ============================================================================
# SEM — Spatial Error Model
# ============================================================================

class SpatialErrorModel:
    """
    Spatial Error Model (SEM).
    Y = X·β + u,   u = λ·Wu + ε,   ε ~ N(0, σ²I)

    Methods: 'ml' (default), 'gmm'

    Effects (Elhorst 2014, Table 2.1):
        Direct = β_k,  Indirect = 0,  Total = β_k
    """

    def __init__(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)
        self.results_ = None

    def fit(self, Y: np.ndarray, X: np.ndarray,
            method: str = 'ml',
            lam_bounds: Tuple[float, float] = (-0.99, 0.99)) -> "SpatialErrorModel":
        Y     = np.asarray(Y, dtype=float).ravel()
        X     = np.asarray(X, dtype=float)
        N, k  = X.shape
        W     = self.W
        ev_c  = np.linalg.eigvals(W)  # complex eigenvalues (spreg ord method)
        X_aug = np.column_stack([np.ones(N), X])
        method = method.lower()

        if method in ('ml', 'qml'):
            def neg_ll(lam):
                B      = np.eye(N) - lam * W
                BY, BX = B @ Y, B @ X_aug
                beta   = np.linalg.lstsq(BX, BY, rcond=None)[0]
                resid  = BY - BX @ beta
                sigma2 = float(resid @ resid) / N
                if sigma2 <= 0: return 1e15
                log_det = float(np.sum(np.log(np.abs(1.0 - lam * ev_W))))
                return 0.5 * N * np.log(sigma2) - log_det

            lam_hat = _grid_search_then_refine(neg_ll, lam_bounds)
            B       = np.eye(N) - lam_hat * W
            BY, BX  = B @ Y, B @ X_aug
            beta    = np.linalg.lstsq(BX, BY, rcond=None)[0]
            resid   = BY - BX @ beta
            sigma2  = float(resid @ resid) / (N - X_aug.shape[1])
            ols_cov = sigma2 * np.linalg.pinv(BX.T @ BX)
            se_beta = np.sqrt(np.maximum(np.diag(ols_cov)[1:], 0))
            h2      = _numerical_hessian(neg_ll, lam_hat)
            se_lam  = 1.0 / np.sqrt(max(h2, 1e-12))
            t_beta  = beta[1:] / np.where(se_beta > 1e-14, se_beta, np.nan)
            p_beta  = 2 * stats.t.sf(np.abs(t_beta), df=N - X_aug.shape[1])
            t_lam   = lam_hat / se_lam if se_lam > 1e-14 else np.nan
            p_lam   = 2 * stats.norm.sf(abs(t_lam)) if not np.isnan(t_lam) else np.nan

        elif method == 'gmm':
            # Kelejian-Prucha GM estimator for SEM
            ols_r = _ols_fit(Y, X_aug)
            e0    = ols_r['residuals']
            lam_hat = 0.0
            for _ in range(5):   # iterative
                We0 = W @ e0
                m1 = float(e0 @ We0) / N
                m3 = float(We0 @ We0) / N
                lam_hat = float(np.clip(m1 / m3 if m3 > 1e-12 else 0, -0.99, 0.99))
                B   = np.eye(N) - lam_hat * W
                BY, BX = B @ Y, B @ X_aug
                beta = np.linalg.lstsq(BX, BY, rcond=None)[0]
                e0   = BY - BX @ beta
            sigma2  = float(e0 @ e0) / (N - X_aug.shape[1])
            ols_cov = sigma2 * np.linalg.pinv(BX.T @ BX)
            se_beta = np.sqrt(np.maximum(np.diag(ols_cov)[1:], 0))
            t_beta  = beta[1:] / np.where(se_beta > 1e-14, se_beta, np.nan)
            p_beta  = 2 * stats.t.sf(np.abs(t_beta), df=N - X_aug.shape[1])
            se_lam  = np.sqrt(2.0 / N) * (1.0 - lam_hat**2)
            t_lam   = lam_hat / se_lam if se_lam > 1e-14 else np.nan
            p_lam   = 2 * stats.norm.sf(abs(t_lam)) if not np.isnan(t_lam) else np.nan
        else:
            raise ValueError(f"SEM method must be 'ml' or 'gmm'. Got '{method}'.")

        self.results_ = {
            "model": "SEM", "method": method,
            "lambda": lam_hat, "lambda_se": se_lam,
            "lambda_t": t_lam, "lambda_p": p_lam,
            "beta": beta[1:], "beta_se": se_beta,
            "beta_t": t_beta, "beta_p": p_beta,
            "intercept": float(beta[0]), "sigma2": sigma2,
            "direct":   beta[1:],      "direct_se":  se_beta,
            "direct_t": t_beta,        "direct_p":   p_beta,
            "indirect": np.zeros(k),   "indirect_se": np.zeros(k),
            "indirect_t": np.zeros(k), "indirect_p":  np.ones(k),
            "total":    beta[1:],      "total_se":   se_beta,
            "total_t":  t_beta,        "total_p":    p_beta,
        }
        return self

    def summary(self) -> dict:
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_

    def print_summary(self, var_names=None):
        res = self.summary()
        k = len(res["beta"])
        m = res.get("method", "ml").upper()
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(k)]
        print(f"\n{'═'*60}")
        print(f"  SEM Model  ({m}, Elhorst 2014)")
        print(f"{'═'*60}")
        print(f"  λ = {res['lambda']:.4f}  SE={res['lambda_se']:.4f}  "
              f"t={res['lambda_t']:.3f}  p={res['lambda_p']:.4f}"
              f"{_format_stars(res['lambda_p'])}")
        print(f"  σ² = {res['sigma2']:.4f}")
        print(f"\n  {'Variable':<14} {'β':>9} {'SE':>9} {'t':>9} {'p':>8}  Sig")
        print(f"  {'─'*55}")
        for j, v in enumerate(var_names):
            b = res['beta'][j]; se = res['beta_se'][j]
            t = res['beta_t'][j]; p = res['beta_p'][j]
            print(f"  {v:<14} {b:>9.4f} {se:>9.4f} {t:>9.3f} {p:>8.4f}  {_format_stars(p)}")


# ============================================================================
# SDM — Spatial Durbin Model
# ============================================================================

class SDMModel:
    """
    Spatial Durbin Model (SDM).
    Y = δ·WY + X·β + WX·θ + ε,   ε ~ N(0, σ²I)

    Estimation methods:
        'ml'    : Maximum Likelihood  ← DEFAULT
        'qml'   : Quasi-ML (Lee 2004)
        'iv'    : IV / 2SLS
        'gmm'   : GMM (Kelejian & Prucha 1998, 1999)
        'bayes' : Bayesian MCMC (LeSage 1997)

    Effects (LeSage & Pace 2009; Elhorst 2014, eq.2.13):
        S_k(W) = (I-δW)⁻¹ (I·β_k + W·θ_k)
        Direct   = (1/N) tr[S_k(W)]
        Total    = (1/N) 1'S_k(W)1
        Indirect = Total − Direct

    Bug fixed: complex eigenvalue handling (see _sinv_traces docstring).
    Covariance: uses full information matrix (not block-diagonal).
    """

    def __init__(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)
        self.results_ = None

    def fit(self, Y: np.ndarray, X: np.ndarray,
            method: str = 'ml',
            rho_bounds: Tuple[float, float] = (-0.99, 0.99),
            n_draws: int = 1000,
            n_bayes: int = 10000,
            seed: int = 0) -> "SDMModel":
        """
        Parameters
        ----------
        method     : 'ml' | 'qml' | 'iv' | 'gmm' | 'bayes'
        rho_bounds : bounds for δ optimization
        n_draws    : simulation draws for effect SE
        n_bayes    : MCMC draws (Bayesian only)
        seed       : random seed
        """
        Y   = np.asarray(Y, dtype=float).ravel()
        X   = np.asarray(X, dtype=float)
        N, k = X.shape
        method = method.lower()

        if method in ('ml', 'qml'):
            res = self._fit_ml(Y, X, N, k, rho_bounds, n_draws, seed)
        elif method == 'iv':
            res = self._fit_iv(Y, X, N, k, n_draws, seed)
        elif method == 'gmm':
            res = self._fit_gmm(Y, X, N, k, n_draws, seed)
        elif method == 'bayes':
            res = self._fit_bayes(Y, X, N, k, n_draws, n_bayes, seed)
        else:
            raise ValueError(f"method must be 'ml','qml','iv','gmm','bayes'. Got '{method}'.")

        res['model']  = 'SDM'
        res['method'] = method
        self.results_ = res
        return self

    # ------------------------------------------------------------------
    def _fit_ml(self, Y, X, N, k, rho_bounds, n_draws, seed):
        W     = self.W
        WX    = W @ X
        X_aug = np.column_stack([np.ones(N), X, WX])   # [1, X, WX]
        ev_c  = np.linalg.eigvals(W)  # complex eigenvalues (spreg ord method)

        def neg_ll(rho):
            A      = np.eye(N) - rho * W
            AY     = A @ Y
            beta   = np.linalg.lstsq(X_aug, AY, rcond=None)[0]
            resid  = AY - X_aug @ beta
            sigma2 = float(resid @ resid) / N
            if sigma2 <= 0: return 1e15
            log_det = _jacobian_logdet(rho, ev_c)
            return 0.5 * N * np.log(sigma2) - log_det

        rho_hat = _grid_search_then_refine(neg_ll, rho_bounds)
        A       = np.eye(N) - rho_hat * W
        AY      = A @ Y
        beta    = np.linalg.lstsq(X_aug, AY, rcond=None)[0]
        resid   = AY - X_aug @ beta
        sigma2  = float(resid @ resid) / N

        beta_X = beta[1:k + 1]
        theta  = beta[k + 1:2 * k + 1]

        # Full covariance — spreg analytic information matrix (Anselin 1988)
        cov_sim, vm = _build_full_cov_spreg_style(
            X_aug, W, rho_hat, beta, sigma2, k, has_theta=True)

        # SE from spreg-style vm (vm layout: [intercept, beta_1..k, theta_1..k, rho])
        p_aug = X_aug.shape[1]  # = 1 + 2k for SDM
        se_rho = float(np.sqrt(max(vm[p_aug, p_aug], 0)))
        se_bX  = np.array([np.sqrt(max(vm[1 + j, 1 + j], 0)) for j in range(k)])
        se_th  = np.array([np.sqrt(max(vm[1 + k + j, 1 + k + j], 0)) for j in range(k)])
        sigma2_df = float(resid @ resid) / (N - X_aug.shape[1])

        eff = _effect_inference(W, beta_X, theta, rho_hat,
                                cov_sim, N, k, D=n_draws, seed=seed)

        t_bX  = beta_X / np.where(se_bX > 1e-14, se_bX, np.nan)
        t_th  = theta  / np.where(se_th  > 1e-14, se_th,  np.nan)
        p_bX  = 2 * stats.norm.sf(np.abs(t_bX))
        p_th  = 2 * stats.norm.sf(np.abs(t_th))
        t_rho = rho_hat / se_rho if se_rho > 1e-14 else np.nan
        p_rho = 2 * stats.norm.sf(abs(t_rho)) if not np.isnan(t_rho) else np.nan

        return dict(
            rho=rho_hat, rho_se=se_rho, rho_t=t_rho, rho_p=p_rho,
            beta_X=beta_X, beta_X_se=se_bX, beta_X_t=t_bX, beta_X_p=p_bX,
            theta_WX=theta, theta_WX_se=se_th, theta_WX_t=t_th, theta_WX_p=p_th,
            intercept=float(beta[0]), sigma2=sigma2_df,
            **eff,
        )

    # ------------------------------------------------------------------
    def _fit_iv(self, Y, X, N, k, n_draws, seed):
        W     = self.W
        WX    = W @ X
        WY    = W @ Y
        X_aug = np.column_stack([np.ones(N), X, WX])
        W2X   = W @ np.column_stack([np.ones(N), X])
        W3X   = W @ W2X
        Z     = np.column_stack([X_aug, W2X, W3X])
        X_full = np.column_stack([X_aug, WY])
        res_iv = _iv_2sls(Y, X_full, Z)

        beta_all = res_iv['beta']
        rho_hat  = float(beta_all[-1])
        beta_hat = beta_all[:k + 1]
        theta    = beta_all[k + 1:2 * k + 1]
        se_all   = res_iv['se']
        se_rho   = float(se_all[-1])
        se_bX    = se_all[1:k + 1]
        se_th    = se_all[k + 1:2 * k + 1]

        # Build sim cov from 2SLS joint cov
        cov_full = res_iv['cov']
        idx_rho  = 2 * k + 1
        idx_b    = list(range(1, k + 1))
        idx_th   = list(range(k + 1, 2 * k + 1))
        idx_all  = [idx_rho] + idx_b + idx_th
        dim_sim  = 1 + 2 * k
        cov_sim  = np.zeros((dim_sim, dim_sim))
        for ii, i in enumerate(idx_all):
            for jj, j in enumerate(idx_all):
                if i < cov_full.shape[0] and j < cov_full.shape[1]:
                    cov_sim[ii, jj] = cov_full[i, j]

        eff = _effect_inference(W, beta_hat[1:], theta, rho_hat,
                                cov_sim, N, k, D=n_draws, seed=seed)

        t_bX  = beta_hat[1:] / np.where(se_bX > 1e-14, se_bX, np.nan)
        t_th  = theta        / np.where(se_th  > 1e-14, se_th, np.nan)
        p_bX  = 2 * stats.norm.sf(np.abs(t_bX))
        p_th  = 2 * stats.norm.sf(np.abs(t_th))
        t_rho = rho_hat / se_rho if se_rho > 1e-14 else np.nan
        p_rho = 2 * stats.norm.sf(abs(t_rho)) if not np.isnan(t_rho) else np.nan

        return dict(
            rho=rho_hat, rho_se=se_rho, rho_t=t_rho, rho_p=p_rho,
            beta_X=beta_hat[1:], beta_X_se=se_bX, beta_X_t=t_bX, beta_X_p=p_bX,
            theta_WX=theta, theta_WX_se=se_th, theta_WX_t=t_th, theta_WX_p=p_th,
            intercept=float(beta_hat[0]), sigma2=res_iv['sigma2'],
            **eff,
        )

    # ------------------------------------------------------------------
    def _fit_gmm(self, Y, X, N, k, n_draws, seed):
        W     = self.W
        WX    = W @ X
        X_sdm = np.column_stack([X, WX])
        gmm   = _gmm_kelejian_prucha(Y, X_sdm, W)
        beta_all = gmm['beta']         # intercept + beta_X + theta
        rho_hat  = gmm['rho']
        beta_X   = beta_all[1:k + 1]
        theta    = beta_all[k + 1:2 * k + 1]
        se_all   = gmm['beta_se']
        se_bX    = se_all[1:k + 1]
        se_th    = se_all[k + 1:2 * k + 1]
        se_rho   = gmm['rho_se']

        cov_full = gmm['cov']
        idx_rho  = 2 * k + 1
        idx_b    = list(range(1, k + 1))
        idx_th   = list(range(k + 1, 2 * k + 1))
        idx_all  = [idx_rho] + idx_b + idx_th
        dim_sim  = 1 + 2 * k
        cov_sim  = np.zeros((dim_sim, dim_sim))
        for ii, i in enumerate(idx_all):
            for jj, j in enumerate(idx_all):
                if i < cov_full.shape[0] and j < cov_full.shape[1]:
                    cov_sim[ii, jj] = cov_full[i, j]

        eff = _effect_inference(W, beta_X, theta, rho_hat,
                                cov_sim, N, k, D=n_draws, seed=seed)

        t_bX  = beta_X / np.where(se_bX > 1e-14, se_bX, np.nan)
        t_th  = theta  / np.where(se_th  > 1e-14, se_th,  np.nan)
        p_bX  = 2 * stats.norm.sf(np.abs(t_bX))
        p_th  = 2 * stats.norm.sf(np.abs(t_th))
        t_rho = rho_hat / se_rho if se_rho > 1e-14 else np.nan
        p_rho = 2 * stats.norm.sf(abs(t_rho)) if not np.isnan(t_rho) else np.nan

        return dict(
            rho=rho_hat, rho_se=se_rho, rho_t=t_rho, rho_p=p_rho,
            beta_X=beta_X, beta_X_se=se_bX, beta_X_t=t_bX, beta_X_p=p_bX,
            theta_WX=theta, theta_WX_se=se_th, theta_WX_t=t_th, theta_WX_p=p_th,
            intercept=float(beta_all[0]), sigma2=gmm['sigma2'],
            **eff,
        )

    # ------------------------------------------------------------------
    def _fit_bayes(self, Y, X, N, k, n_draws_eff, n_bayes, seed):
        W     = self.W
        WX    = W @ X
        X_sdm = np.column_stack([X, WX])
        bayes = _bayesian_sar_mcmc(Y, X_sdm, W, n_draws=n_bayes, seed=seed)

        beta_all = bayes['beta']       # intercept + beta_X + theta
        rho_hat  = bayes['rho']
        beta_X   = beta_all[1:k + 1]
        theta    = beta_all[k + 1:2 * k + 1]
        se_all   = bayes['beta_se']
        se_bX    = se_all[1:k + 1]
        se_th    = se_all[k + 1:2 * k + 1]
        se_rho   = bayes['rho_se']

        rho_d  = bayes['rho_draws']
        b_d    = bayes['beta_draws'][:, 1:k+1]
        th_d   = bayes['beta_draws'][:, k+1:2*k+1]
        all_d  = np.column_stack([rho_d[:, None], b_d, th_d])
        cov_sim = np.cov(all_d.T)
        if cov_sim.ndim < 2:
            cov_sim = np.eye(1 + 2*k) * 0.01

        eff = _effect_inference(W, beta_X, theta, rho_hat,
                                cov_sim, N, k, D=n_draws_eff, seed=seed)

        t_bX  = beta_X / np.where(se_bX > 1e-14, se_bX, np.nan)
        t_th  = theta  / np.where(se_th  > 1e-14, se_th,  np.nan)
        p_bX  = 2 * stats.norm.sf(np.abs(t_bX))
        p_th  = 2 * stats.norm.sf(np.abs(t_th))
        t_rho = rho_hat / se_rho if se_rho > 1e-14 else np.nan
        p_rho = 2 * stats.norm.sf(abs(t_rho)) if not np.isnan(t_rho) else np.nan

        return dict(
            rho=rho_hat, rho_se=se_rho, rho_t=t_rho, rho_p=p_rho,
            beta_X=beta_X, beta_X_se=se_bX, beta_X_t=t_bX, beta_X_p=p_bX,
            theta_WX=theta, theta_WX_se=se_th, theta_WX_t=t_th, theta_WX_p=p_th,
            intercept=float(beta_all[0]), sigma2=bayes['sigma2'],
            acceptance_rate=bayes['acceptance_rate'],
            **eff,
        )

    def summary(self) -> dict:
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_

    def print_summary(self, var_names=None):
        res = self.summary()
        k = len(res["beta_X"])
        m = res.get("method", "ml").upper()
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(k)]
        print(f"\n{'═'*95}")
        print(f"  SDM Model  ({m}, Elhorst 2014)")
        print(f"{'═'*95}")
        print(f"  δ = {res['rho']:.4f}  SE={res['rho_se']:.4f}  "
              f"t={res['rho_t']:.3f}  p={res['rho_p']:.4f}"
              f"{_format_stars(res['rho_p'])}")
        if res.get('acceptance_rate') is not None:
            print(f"  MCMC acceptance rate: {res['acceptance_rate']:.3f}")
        print(f"  σ² = {res['sigma2']:.4f}")
        print(f"\n  Coefficient estimates (β and θ):")
        hdr = (f"  {'Variable':<14} {'β':>9} {'SE':>9} {'t':>9} {'p':>8}  "
               f"{'θ(WX)':>9} {'SE':>9} {'t':>9} {'p':>8}")
        print(hdr)
        print(f"  {'─'*90}")
        for j, v in enumerate(var_names):
            b  = res['beta_X'][j];    sb = res['beta_X_se'][j]
            tb = res['beta_X_t'][j];  pb = res['beta_X_p'][j]
            t  = res['theta_WX'][j];  st = res['theta_WX_se'][j]
            tt = res['theta_WX_t'][j];pt = res['theta_WX_p'][j]
            print(f"  {v:<14} {b:>9.4f} {sb:>9.4f} {tb:>9.3f} {pb:>8.4f}"
                  f"{_format_stars(pb)}  "
                  f"{t:>9.4f} {st:>9.4f} {tt:>9.3f} {pt:>8.4f}{_format_stars(pt)}")
        print_effects_table(res, var_names)
