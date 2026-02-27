{smcl}
{* *! version 1.0.0  27Feb2026}{...}
{viewerjumpto "Syntax"        "stwm_build##syntax"}{...}
{viewerjumpto "Description"   "stwm_build##description"}{...}
{viewerjumpto "Options"       "stwm_build##options"}{...}
{viewerjumpto "Stored results""stwm_build##results"}{...}
{viewerjumpto "Example"       "stwm_build##example"}{...}
{viewerjumpto "Also see"      "stwm_build##also_see"}{...}

{title:Title}

{phang}
{bf:stwm_build} {hline 2} Build a Spatial-Temporal Weight Matrix (STWM)

{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:stwm_build} {varname} {ifin}{cmd:,}
{cmd:swm(}{it:filename}{cmd:)}
{cmd:n(}{it:#}{cmd:)}
{cmd:t(}{it:#}{cmd:)}
[{it:options}]

{synoptset 22 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{opt swm(filename)}}path to spatial weight matrix file (.dta, .csv, or .npy){p_end}
{synopt:{opt n(#)}}number of spatial units (cities){p_end}
{synopt:{opt t(#)}}number of time periods (years){p_end}
{syntab:Optional}
{synopt:{opt dec:ay(string)}}TWM construction method; default {cmd:morans}{p_end}
{synopt:{opt par:am(#)}}decay parameter for parametric TWMs; default {cmd:0.5}{p_end}
{synopt:{opt saveas(name)}}register STWM under this name; default {cmd:STWM}{p_end}
{synopt:{opt wins:orize(#)}}winsorize quantile for ratio TWMs; default {cmd:0.95}{p_end}
{synopt:{opt minabs(#)}}minimum absolute denominator guard; default {cmd:0.001}{p_end}
{synoptline}

{marker description}{...}
{title:Description}

{pstd}
{cmd:stwm_build} constructs a Spatial-Temporal Weight Matrix (STWM) by:

{phang2}1. Computing one scalar statistic per year from {varname} and {bf:swm()}.{p_end}
{phang2}2. Building a T×T Temporal Weight Matrix (TWM) from the annual sequence.{p_end}
{phang2}3. Assembling STWM = TWM ⊗ SWM via Kronecker product.{p_end}

{pstd}
The resulting matrix captures both spatial spillovers (encoded in SWM)
and temporal propagation dynamics (encoded in TWM).

{pstd}
{bf:Data must be sorted in time-major order}: all units for period 1,
then all units for period 2, etc.  Use {cmd:sort year citycode} before running.

{marker options}{...}
{title:Options for decay()}

{phang}
{cmd:decay(morans)}   [default] Moran's I ratio method.
  TWM[t,s] = |I_t / I_s| for s < t.
  Captures temporal change in global spatial clustering.

{phang}
{cmd:decay(gearyc)}   Geary's C transform: a_t = 2 − C_t, then a_t/a_s.

{phang}
{cmd:decay(getis)}    Getis-Ord G ratio method.

{phang}
{cmd:decay(gini)}     Spatial Gini coefficient ratio method.

{phang}
{cmd:decay(exponential)}   Parametric: w[t,s] = exp(−λ(t−s)).

{phang}
{cmd:decay(linear)}   Parametric: w[t,s] = max(0, 1 − λ(t−s)).

{phang}
{cmd:decay(power)}    Parametric: w[t,s] = (t−s)^{−λ}.

{marker results}{...}
{title:Stored results}

{pstd}
{cmd:stwm_build} stores the following in {cmd:r()}:

{synoptset 22 tabbed}{...}
{synopt:{cmd:r(name)}}name under which STWM is registered{p_end}
{synopt:{cmd:r(path)}}file path of saved STWM (.npy){p_end}
{synopt:{cmd:r(spectral_radius)}}spectral radius of TWM (should be < 1){p_end}
{synopt:{cmd:r(twm_passed)}}1 if TWM admissibility checks passed, 0 otherwise{p_end}

{pstd}
The following globals are set (accessible throughout the session):

{phang2}{cmd:${STWM_I_path}}   — file path to STWM matrix{p_end}
{phang2}{cmd:${STWM_I_n}}      — number of spatial units{p_end}
{phang2}{cmd:${STWM_I_T}}      — number of time periods{p_end}

{marker example}{...}
{title:Example}

{phang}{cmd:* Sort data in time-major order}{p_end}
{phang}{cmd:. sort year citycode}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Build Moran's I STWM}{p_end}
{phang}{cmd:. stwm_build lnpatent, swm("swm.dta") n(276) t(14) decay(morans) saveas(STWM_I)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Build Geary's C STWM}{p_end}
{phang}{cmd:. stwm_build lnpatent, swm("swm.dta") n(276) t(14) decay(gearyc) saveas(STWM_C)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Build exponential-decay STWM (no data-driven component)}{p_end}
{phang}{cmd:. stwm_build lnpatent, swm("swm.dta") n(276) t(14) decay(exponential) param(0.5) saveas(STWM_D)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Check results}{p_end}
{phang}{cmd:. return list}{p_end}

{marker also_see}{...}
{title:Also see}

{psee}
{helpb stwm_sdm} — Spatial Durbin Model

{psee}
{helpb stwm_sar} — Spatial Autoregressive Model

{psee}
{helpb stwm_hausman} — Exogeneity test battery
