*! stwm_mc.ado  version 1.0.0  27Feb2026
*! Monte Carlo simulation to validate STWM-based estimator
*!   DGP: Y = (I − ρ·STWM)⁻¹ (Xβ + ε),  ε ~ N(0,I)
*!
*! Syntax:
*!   stwm_mc, W(name) N(#) T(#)
*!            [TRUERHO(real) NSIM(integer) SEED(integer) NDemo(integer)]

program define stwm_mc, rclass
    version 16.0

    syntax , W(name)                  ///  STWM name (must be registered)
             N(integer)               ///  total spatial units
             T(integer)               ///  time periods
             [TRUERHO(real 0.3)       ///  true spatial parameter (default 0.3)
              NSIM(integer 200)       ///  Monte Carlo replications
              SEED(integer 42)        ///  random seed
              NDemo(integer 50)]      //   use first ndemo cities (keeps runtime short)

    if "${`w'_path}" == "" {
        di as error "STWM '`w'' not found. Run: stwm_build ..., saveas(`w')"
        exit 111
    }

    di as text "Monte Carlo Simulation"
    di as text "  STWM      : `w'"
    di as text "  n (demo)  : `ndemo' cities  (first `ndemo' of `n')"
    di as text "  T         : `t' years"
    di as text "  N_sim     : `nsim' replications"
    di as text "  True ρ    : `truerho'"

    python: _stwm_montecarlo("`w'", `n', `t', `truerho', `nsim', `seed', `ndemo')

    di as text ""
    di as text " ══════════════════════════════════════════════════════"
    di as text "  Monte Carlo Results  (DGP: SAR, true ρ = `truerho')"
    di as text " ══════════════════════════════════════════════════════"
    di as result "  Bias(ρ)     = " %8.4f scalar(r_mc_bias_rho_)
    di as result "  RMSE(ρ)     = " %8.4f scalar(r_mc_rmse_rho_)
    di as result "  Coverage(ρ) = " %8.4f scalar(r_mc_cov_rho_)
    di as text   "  (Coverage should be ≈ 0.95 for well-specified model)"
    di as text " ══════════════════════════════════════════════════════"

    return scalar mc_bias_rho  = scalar(r_mc_bias_rho_)
    return scalar mc_rmse_rho  = scalar(r_mc_rmse_rho_)
    return scalar mc_cov_rho   = scalar(r_mc_cov_rho_)
    return scalar mc_nsim      = `nsim'

end


python:
import numpy as np

def _stwm_montecarlo(w_name, n_full, T, true_rho, n_sim, seed, n_demo):
    from sfi import Scalar
    import stwm as pkg

    # Load sub-matrix for speed (first n_demo × n_demo block of SWM)
    W_full = np.load(_STWM_STORE.get(f"{w_name}_SWM",
                                      list(_STWM_STORE.values())[0]))
    W_sub  = W_full[:n_demo, :n_demo].copy()
    rs     = W_sub.sum(axis=1, keepdims=True); rs[rs==0]=1; W_sub /= rs

    TWM = np.load(_STWM_STORE.get(f"{w_name}_TWM",
                                   list(_STWM_STORE.values())[0]))

    # Run MC (n_demo cities keeps (I-ρW)^{-1} cheap: n_demo*T × n_demo*T)
    mc = pkg.monte_carlo_stwm(
        true_rho     = true_rho,
        true_beta    = np.array([0.1, 0.05, 0.08, -0.03, 0.06, 0.07, 0.04]),
        W_spatial    = W_sub,
        TWM          = TWM,
        n            = n_demo,
        T            = T,
        n_simulations= n_sim,
        ModelClass   = pkg.SpatialLagModel,
        seed         = seed,
    )

    Scalar.setValue("r_mc_bias_rho_", float(mc["bias_rho"]))
    Scalar.setValue("r_mc_rmse_rho_", float(mc["rmse_rho"]))
    cov = mc.get("coverage_rho", float("nan"))
    Scalar.setValue("r_mc_cov_rho_",
                    float(cov) if cov is not None and cov == cov else float("nan"))

    print(f"  Completed {mc['n_simulations']} valid replications.")
    print(f"  Bias(ρ) = {mc['bias_rho']:.4f}   RMSE(ρ) = {mc['rmse_rho']:.4f}")
    if mc.get("coverage_rho") is not None:
        print(f"  Coverage(ρ) = {mc['coverage_rho']:.4f}")

end
