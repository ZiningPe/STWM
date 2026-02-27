{smcl}
{* *! version 1.0.0  27Feb2026}{...}
{viewerjumpto "Syntax"        "stwm_sar##syntax"}{...}
{viewerjumpto "Description"   "stwm_sar##description"}{...}
{viewerjumpto "Options"       "stwm_sar##options"}{...}
{viewerjumpto "Stored results""stwm_sar##results"}{...}
{viewerjumpto "Example"       "stwm_sar##example"}{...}
{viewerjumpto "Also see"      "stwm_sar##also_see"}{...}

{title:Title}

{phang}
{bf:stwm_sar} {hline 2} Spatial Autoregressive Model (SAR) via Spatial-Temporal Weight Matrix

{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:stwm_sar} {varlist} {ifin}{cmd:,}
{cmd:w(}{it:name}{cmd:)}
[{it:options}]

{synoptset 24 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{opt w(name)}}name of pre-built STWM (from {helpb stwm_build}){p_end}
{syntab:Optional}
{synopt:{opt eff:ects(string)}}panel effects: {cmd:none} (default), {cmd:fe}, {cmd:re}{p_end}
{synopt:{opt n(#)}}number of spatial units; required when {cmd:effects(fe/re)}{p_end}
{synoptline}

{marker description}{...}
{title:Description}

{pstd}
{cmd:stwm_sar} fits the Spatial Autoregressive (SAR) Model, also known as
the Spatial Lag Model:

{phang2}
{bf:Y = ρ·W·Y + X·β + ε}

{pstd}
where {bf:W} is the pre-built STWM, ρ is the spatial autoregressive
parameter, and ε ~ N(0, σ²I).

{pstd}
Results are posted as {cmd:e()}-class. Use {cmd:estat effects} for the
direct/indirect/total decomposition.

{marker options}{...}
{title:Options}

{phang}
{cmd:effects(none)} [default] Pooled estimator (no unit fixed effects).

{phang}
{cmd:effects(fe)} Within-group (demeaned) panel estimator.
Requires {cmd:n(#)}.

{phang}
{cmd:effects(re)} Random-effects panel estimator.
Requires {cmd:n(#)}.

{marker results}{...}
{title:Stored results}

{pstd}
{cmd:stwm_sar} stores the following in {cmd:e()}:

{synoptset 22 tabbed}{...}
{syntab:Scalars}
{synopt:{cmd:e(N)}}number of observations{p_end}
{synopt:{cmd:e(rho)}}estimated spatial parameter ρ{p_end}
{synopt:{cmd:e(rho_se)}}standard error of ρ{p_end}
{synopt:{cmd:e(sigma2)}}residual variance{p_end}
{syntab:Matrices}
{synopt:{cmd:e(b)}}coefficient row-vector: [β | _cons | ρ]{p_end}
{synopt:{cmd:e(V)}}variance-covariance matrix{p_end}
{synopt:{cmd:e(direct)}}direct effects (1 × k){p_end}
{synopt:{cmd:e(indirect)}}indirect effects (1 × k){p_end}
{synopt:{cmd:e(total)}}total effects (1 × k){p_end}
{synoptline}

{marker example}{...}
{title:Example}

{phang}{cmd:. sort year citycode}{p_end}
{phang}{cmd:. stwm_build lnpatent, swm("swm.dta") n(276) t(14) saveas(STWM_I)}{p_end}
{phang}{cmd:. stwm_sar lnpatent lnrd lnfdi lnhuman, w(STWM_I)}{p_end}
{phang}{cmd:. estat effects}{p_end}

{marker also_see}{...}
{title:Also see}

{psee}
{helpb stwm_build} — Build STWM

{psee}
{helpb stwm_sdm} — Spatial Durbin Model (richer spillover specification)

{psee}
{helpb stwm_sem} — Spatial Error Model
