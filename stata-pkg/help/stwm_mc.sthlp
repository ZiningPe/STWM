{smcl}
{* *! version 1.0.0  27Feb2026}{...}
{viewerjumpto "Syntax"        "stwm_mc##syntax"}{...}
{viewerjumpto "Description"   "stwm_mc##description"}{...}
{viewerjumpto "Options"       "stwm_mc##options"}{...}
{viewerjumpto "Stored results""stwm_mc##results"}{...}
{viewerjumpto "Example"       "stwm_mc##example"}{...}
{viewerjumpto "Also see"      "stwm_mc##also_see"}{...}

{title:Title}

{phang}
{bf:stwm_mc} {hline 2} Monte Carlo Validation of the STWM-Based Estimator

{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:stwm_mc}{cmd:,}
{cmd:w(}{it:name}{cmd:)}
{cmd:n(}{it:#}{cmd:)}
{cmd:t(}{it:#}{cmd:)}
[{it:options}]

{synoptset 24 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{opt w(name)}}registered STWM name (from {helpb stwm_build}){p_end}
{synopt:{opt n(#)}}total number of spatial units in the dataset{p_end}
{synopt:{opt t(#)}}number of time periods{p_end}
{syntab:Optional}
{synopt:{opt truerho(#)}}true ρ in the DGP; default {cmd:0.3}{p_end}
{synopt:{opt nsim(#)}}number of Monte Carlo replications; default {cmd:200}{p_end}
{synopt:{opt seed(#)}}random seed for reproducibility; default {cmd:42}{p_end}
{synopt:{opt ndemo(#)}}use first {it:#} cities (for speed); default {cmd:50}{p_end}
{synoptline}

{marker description}{...}
{title:Description}

{pstd}
{cmd:stwm_mc} validates the STWM-based SAR estimator via Monte Carlo
simulation.  The data-generating process (DGP) is:

{phang2}
{bf:Y = (I − ρ·W)⁻¹ (X·β + ε),   ε ~ N(0, I)}

{pstd}
where {bf:W} is constructed from the first {cmd:ndemo()} cities of the
registered STWM, and the true parameters (ρ, β) are set by the user.

{pstd}
For each of the {cmd:nsim()} replications, a new ε is drawn, Y is
generated from the DGP, and the SAR model is re-estimated via ML.
The command then reports:

{phang2}{bf:Bias(ρ)} = mean(ρ̂) − ρ_true

{phang2}{bf:RMSE(ρ)} = sqrt(mean((ρ̂ − ρ_true)²))

{phang2}{bf:Coverage(ρ)} = fraction of 95% confidence intervals containing ρ_true

{pstd}
{it:Coverage ≈ 0.95} indicates the estimator is well-calibrated.
Large bias or RMSE relative to |ρ_true| signals model mis-specification.

{pstd}
{bf:Note:} The {cmd:ndemo()} option subsets the SWM to the first
{it:#} cities to keep the matrix inversion (n·T × n·T) tractable.
For production validation, increase {cmd:ndemo()} and {cmd:nsim()}.

{marker options}{...}
{title:Options}

{phang}
{opt truerho(#)} sets the true spatial autoregressive parameter in the
DGP.  Default 0.3.  Must be within the admissible range of W's
spectral radius.

{phang}
{opt nsim(#)} number of Monte Carlo replications.  Default 200.

{phang}
{opt seed(#)} random seed passed to NumPy for reproducibility.
Default 42.

{phang}
{opt ndemo(#)} number of cities to use (first {it:#} cities of the
registered SWM).  Default 50.  The SWM sub-block is row-standardised
before use.

{marker results}{...}
{title:Stored results}

{pstd}
{cmd:stwm_mc} stores the following in {cmd:r()}:

{synoptset 22 tabbed}{...}
{synopt:{cmd:r(mc_bias_rho)}}bias of ρ̂{p_end}
{synopt:{cmd:r(mc_rmse_rho)}}root mean squared error of ρ̂{p_end}
{synopt:{cmd:r(mc_cov_rho)}}empirical coverage at 95% level{p_end}
{synopt:{cmd:r(mc_nsim)}}number of replications requested{p_end}
{synoptline}

{marker example}{...}
{title:Example}

{phang}{cmd:. sort year citycode}{p_end}
{phang}{cmd:. stwm_build lnpatent, swm("swm.dta") n(276) t(14) saveas(STWM_I)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Quick validation (50 cities, 100 replications)}{p_end}
{phang}{cmd:. stwm_mc, w(STWM_I) n(276) t(14) truerho(0.3) nsim(100) ndemo(50)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Access results}{p_end}
{phang}{cmd:. di "RMSE = " r(mc_rmse_rho) "   Coverage = " r(mc_cov_rho)}{p_end}

{marker also_see}{...}
{title:Also see}

{psee}
{helpb stwm_build} — Build STWM

{psee}
{helpb stwm_sensitivity} — Sensitivity analysis: hyperparameter grid

{psee}
{helpb stwm_sdm} — Spatial Durbin Model
