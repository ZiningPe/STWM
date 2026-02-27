{smcl}
{* *! version 1.0.0  27Feb2026}{...}
{viewerjumpto "Syntax"        "stwm_slx##syntax"}{...}
{viewerjumpto "Description"   "stwm_slx##description"}{...}
{viewerjumpto "Stored results""stwm_slx##results"}{...}
{viewerjumpto "Example"       "stwm_slx##example"}{...}
{viewerjumpto "Also see"      "stwm_slx##also_see"}{...}

{title:Title}

{phang}
{bf:stwm_slx} {hline 2} Spatial Lag of X Model (SLX) via Spatial-Temporal Weight Matrix

{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:stwm_slx} {varlist} {ifin}{cmd:,}
{cmd:w(}{it:name}{cmd:)}
[{cmd:effects(}{it:string}{cmd:)}
 {cmd:n(}{it:#}{cmd:)}]

{synoptset 24 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{opt w(name)}}name of pre-built STWM (from {helpb stwm_build}){p_end}
{syntab:Optional}
{synopt:{opt eff:ects(string)}}panel effects: {cmd:none} (default), {cmd:fe}, {cmd:re}{p_end}
{synopt:{opt n(#)}}number of spatial units (required for fe/re){p_end}
{synoptline}

{marker description}{...}
{title:Description}

{pstd}
{cmd:stwm_slx} fits the Spatial Lag of X (SLX) model:

{phang2}
{bf:Y = X·β + W·X·θ + ε}

{pstd}
Unlike SAR/SDM, the SLX model contains no spatial lag of Y, so it can
be estimated by OLS.  Spillover effects arise purely through the
spatially-lagged regressors W·X.

{pstd}
The model is identified without an instrument for W·Y, making it a
useful robustness check against endogeneity in the spatial lag.

{marker results}{...}
{title:Stored results}

{pstd}
{cmd:stwm_slx} stores the following in {cmd:e()}:

{synoptset 22 tabbed}{...}
{syntab:Scalars}
{synopt:{cmd:e(N)}}number of observations{p_end}
{synopt:{cmd:e(sigma2)}}residual variance{p_end}
{syntab:Matrices}
{synopt:{cmd:e(b)}}coefficient row-vector: [β | θ_WX | _cons]{p_end}
{synopt:{cmd:e(V)}}variance-covariance matrix{p_end}
{synopt:{cmd:e(direct)}}direct effects = β (1 × k){p_end}
{synopt:{cmd:e(indirect)}}indirect effects = θ (1 × k){p_end}
{synopt:{cmd:e(total)}}total effects = β + θ (1 × k){p_end}
{synoptline}

{marker example}{...}
{title:Example}

{phang}{cmd:. stwm_build lnpatent, swm("swm.dta") n(276) t(14) saveas(STWM_I)}{p_end}
{phang}{cmd:. stwm_slx  lnpatent lnrd lnfdi lnhuman, w(STWM_I)}{p_end}
{phang}{cmd:. estat effects}{p_end}

{marker also_see}{...}
{title:Also see}

{psee}
{helpb stwm_sdm} — Spatial Durbin Model (adds spatial lag of Y)

{psee}
{helpb stwm_sar} — Spatial Autoregressive Model

{psee}
{helpb stwm_build} — Build STWM
