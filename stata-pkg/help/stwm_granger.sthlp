{smcl}
{* *! version 1.0.0  27Feb2026}{...}
{viewerjumpto "Syntax"        "stwm_granger##syntax"}{...}
{viewerjumpto "Description"   "stwm_granger##description"}{...}
{viewerjumpto "Options"       "stwm_granger##options"}{...}
{viewerjumpto "Stored results""stwm_granger##results"}{...}
{viewerjumpto "Example"       "stwm_granger##example"}{...}
{viewerjumpto "Also see"      "stwm_granger##also_see"}{...}

{title:Title}

{phang}
{bf:stwm_granger} {hline 2} Granger Causality: Moran's I → Spillover Parameters

{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:stwm_granger} {varname} {ifin}{cmd:,}
{cmd:w(}{it:name}{cmd:)}
{cmd:n(}{it:#}{cmd:)}
{cmd:t(}{it:#}{cmd:)}
[{cmd:maxlag(}{it:#}{cmd:)}
 {cmd:window(}{it:#}{cmd:)}]

{synoptset 24 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{opt w(name)}}registered STWM name (from {helpb stwm_build}){p_end}
{synopt:{opt n(#)}}number of spatial units{p_end}
{synopt:{opt t(#)}}number of time periods{p_end}
{syntab:Optional}
{synopt:{opt maxlag(#)}}maximum Granger lag order to test; default {cmd:3}{p_end}
{synopt:{opt window(#)}}rolling-window size (years) for spillover sequence; default {cmd:4}{p_end}
{synoptline}

{marker description}{...}
{title:Description}

{pstd}
{cmd:stwm_granger} tests whether the annual Moran's I series
Granger-causes the estimated indirect (spillover) effect series,
thereby providing a structural justification for using Moran's I
as the basis for TWM construction.

{pstd}
{bf:Motivation:}  The STWM Moran-I ratio method sets
TWM[t,s] = |I_t / I_s|, so the temporal weights are derived from the
same outcome variable as the dependent variable of the regression.
A potential concern is circularity: the TWM may merely reflect the
structure of Y rather than capturing genuine temporal propagation.

{pstd}
The Granger test addresses this directly.  If Moran's I at time t−h
significantly predicts the magnitude of future spillover parameters,
the temporal dynamics in Moran's I carry forward-looking information
about spatial spillover.  This justifies the structural link from
Moran's I to TWM construction.

{pstd}
{bf:Procedure:}

{phang2}
1. Annual Moran's I sequence {I_1, ..., I_T} is computed from
   {varname} and the SWM embedded in the registered STWM.

{phang2}
2. Rolling-window SAR is estimated with window size {cmd:window()},
   producing a sequence of indirect (spillover) effect estimates.

{phang2}
3. The end-year Moran's I is aligned with each window's spillover estimate.

{phang2}
4. A standard Granger-causality VAR F-test is run for each lag
   order from 1 to {cmd:maxlag()}.

{pstd}
{bf:Interpretation:}  Reject H₀ (Moran's I does NOT Granger-cause
spillovers) → Moran's I dynamics predict spillover parameter changes
→ the TWM construction is structurally justified and not a circular
artifact.

{marker options}{...}
{title:Options}

{phang}
{opt maxlag(#)} maximum Granger lag order.  Results are reported for
each lag from 1 to {it:#}.  Default 3.

{phang}
{opt window(#)} number of years in each rolling-window regression
used to generate the spillover sequence.  Must be ≥ 3.  Default 4.
Larger windows yield smoother but fewer spillover estimates.

{marker results}{...}
{title:Stored results}

{pstd}
{cmd:stwm_granger} stores the following in {cmd:r()}:

{synoptset 22 tabbed}{...}
{synopt:{cmd:r(gr_F_lag1)}}F statistic at lag 1{p_end}
{synopt:{cmd:r(gr_p_lag1)}}p-value at lag 1{p_end}
{synopt:{cmd:r(gr_rej_lag1)}}Yes/No at lag 1{p_end}
{synoptline}

{pstd}
Results for lags 2 … {cmd:maxlag()} are printed to the screen and are
accessible via the displayed scalars {cmd:r_gr_F_{it:#}_} and
{cmd:r_gr_p_{it:#}_} immediately after the Python call.

{marker example}{...}
{title:Example}

{phang}{cmd:. sort year citycode}{p_end}
{phang}{cmd:. stwm_build lnpatent, swm("swm.dta") n(276) t(14) saveas(STWM_I)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Test whether Moran's I Granger-causes spillover dynamics}{p_end}
{phang}{cmd:. stwm_granger lnpatent, w(STWM_I) n(276) t(14) maxlag(3) window(4)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:. return list}{p_end}

{marker also_see}{...}
{title:Also see}

{psee}
{helpb stwm_build} — Build STWM

{psee}
{helpb stwm_hausman} — Exogeneity test battery (Hausman + Sargan + Redundancy)

{psee}
{helpb stwm_sensitivity} — Sensitivity analysis: hyperparameter grid
