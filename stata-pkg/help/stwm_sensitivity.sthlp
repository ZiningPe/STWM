{smcl}
{* *! version 1.0.0  27Feb2026}{...}
{viewerjumpto "Syntax"        "stwm_sensitivity##syntax"}{...}
{viewerjumpto "Description"   "stwm_sensitivity##description"}{...}
{viewerjumpto "Options"       "stwm_sensitivity##options"}{...}
{viewerjumpto "Stored results""stwm_sensitivity##results"}{...}
{viewerjumpto "Example"       "stwm_sensitivity##example"}{...}
{viewerjumpto "Also see"      "stwm_sensitivity##also_see"}{...}

{title:Title}

{phang}
{bf:stwm_sensitivity} {hline 2} Sensitivity Analysis: TWM Hyperparameter Grid

{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:stwm_sensitivity} {varlist} {ifin}{cmd:,}
{cmd:swm(}{it:filename}{cmd:)}
{cmd:n(}{it:#}{cmd:)}
{cmd:t(}{it:#}{cmd:)}
[{cmd:winsqlist(}{it:numlist}{cmd:)}
 {cmd:malist(}{it:numlist}{cmd:)}]

{synoptset 24 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{opt swm(filename)}}path to spatial weight matrix file{p_end}
{synopt:{opt n(#)}}number of spatial units{p_end}
{synopt:{opt t(#)}}number of time periods{p_end}
{syntab:Optional}
{synopt:{opt winsqlist(numlist)}}winsorize quantiles to test; default {cmd:0.90 0.95 0.99}{p_end}
{synopt:{opt malist(numlist)}}min_abs values to test; default {cmd:0.0001 0.001 0.01}{p_end}
{synoptline}

{marker description}{...}
{title:Description}

{pstd}
{cmd:stwm_sensitivity} assesses whether estimated results are sensitive
to the two numerical hyperparameters of the Moran-I ratio TWM:

{phang2}
{bf:winsorize_quantile} — upper quantile at which the raw ratio
TWM[t,s] = |I_t/I_s| is winsorized, preventing extreme values from
dominating temporal weights.

{phang2}
{bf:min_abs} — minimum absolute denominator guard; prevents division
by near-zero Moran's I values.

{pstd}
For each combination in the user-specified grid, {cmd:stwm_sensitivity}
rebuilds the STWM from scratch, refits the SDM via ML, and records ρ
and the direct/indirect effects for each regressor.

{pstd}
At the end, the coefficient of variation (CV = std/|mean|) of ρ across
all grid configurations is reported. CV < 0.05 indicates results
are stable with respect to hyperparameter choice.

{marker options}{...}
{title:Options}

{phang}
{opt winsqlist(numlist)} specifies the winsorize quantile values to try.
All values must be in (0, 1).  Default: {cmd:0.90 0.95 0.99}.

{phang}
{opt malist(numlist)} specifies the minimum absolute values for the
denominator guard.  Default: {cmd:0.0001 0.001 0.01}.

{marker results}{...}
{title:Stored results}

{pstd}
{cmd:stwm_sensitivity} stores the following in {cmd:r()}:

{synoptset 22 tabbed}{...}
{synopt:{cmd:r(sens_table)}}full grid matrix (winsq, min_abs, rho, direct_*, indirect_*){p_end}
{synopt:{cmd:r(cv_rho)}}coefficient of variation of ρ across grid{p_end}
{synoptline}

{marker example}{...}
{title:Example}

{phang}{cmd:. sort year citycode}{p_end}
{phang}{cmd:. stwm_sensitivity lnpatent lnrd lnfdi lnhuman, ///}{p_end}
{phang}{cmd:      swm("swm.dta") n(276) t(14)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Custom grid}{p_end}
{phang}{cmd:. stwm_sensitivity lnpatent lnrd lnfdi lnhuman, ///}{p_end}
{phang}{cmd:      swm("swm.dta") n(276) t(14) ///}{p_end}
{phang}{cmd:      winsqlist(0.90 0.95) malist(0.001 0.01)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:. di "CV(rho) = " r(cv_rho)}{p_end}
{phang}{cmd:. matrix list r(sens_table)}{p_end}

{marker also_see}{...}
{title:Also see}

{psee}
{helpb stwm_build} — Build STWM

{psee}
{helpb stwm_sdm} — Spatial Durbin Model

{psee}
{helpb stwm_mc} — Monte Carlo validation
