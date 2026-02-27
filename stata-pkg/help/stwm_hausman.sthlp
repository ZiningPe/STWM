{smcl}
{* *! version 1.0.0  27Feb2026}{...}
{viewerjumpto "Syntax"        "stwm_hausman##syntax"}{...}
{viewerjumpto "Description"   "stwm_hausman##description"}{...}
{viewerjumpto "Options"       "stwm_hausman##options"}{...}
{viewerjumpto "Stored results""stwm_hausman##results"}{...}
{viewerjumpto "Example"       "stwm_hausman##example"}{...}
{viewerjumpto "Also see"      "stwm_hausman##also_see"}{...}

{title:Title}

{phang}
{bf:stwm_hausman} {hline 2} STWM Exogeneity Test Battery

{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:stwm_hausman} {varlist} {ifin}{cmd:,}
{cmd:w(}{it:name}{cmd:)}
{cmd:wg(}{it:name}{cmd:)}

{synoptset 24 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{opt w(name)}}STWM under test (from {helpb stwm_build}){p_end}
{synopt:{opt wg(name)}}exogenous benchmark STWM (e.g., static inverse-distance){p_end}
{synoptline}

{marker description}{...}
{title:Description}

{pstd}
{cmd:stwm_hausman} runs three complementary exogeneity tests for the
Spatial-Temporal Weight Matrix:

{phang2}
{bf:[1] Hausman Test} — compares estimates under the STWM with estimates
under the exogenous benchmark W_G. H₀: the two estimators agree (STWM
is exogenous). A large, significant H statistic suggests the data-driven
TWM introduces endogeneity.

{phang2}
{bf:[2] Sargan J-Test} — over-identification test. Instruments are
W·X (spatial lags of regressors). H₀: instruments are orthogonal to the
error term. Rejection suggests invalid instruments or misspecification.

{phang2}
{bf:[3] Redundancy F-Test} — tests whether STWM·X lags add explanatory
power over and above the benchmark W_G·X lags. H₀: STWM lags are
redundant. Rejection confirms the data-driven component improves fit.

{pstd}
{bf:Interpretation:}
Fail to reject [1] AND reject [3] is the ideal outcome — the STWM
is exogenous yet informative relative to the static benchmark.

{marker options}{...}
{title:Options}

{phang}
{opt w(name)} specifies the STWM under scrutiny, built by {helpb stwm_build}.

{phang}
{opt wg(name)} specifies the exogenous benchmark, typically a static
inverse-distance or contiguity matrix also registered via {helpb stwm_build}.

{marker results}{...}
{title:Stored results}

{pstd}
{cmd:stwm_hausman} stores the following in {cmd:r()}:

{synoptset 22 tabbed}{...}
{synopt:{cmd:r(hausman_H)}}Hausman H statistic{p_end}
{synopt:{cmd:r(hausman_p)}}p-value of Hausman test{p_end}
{synopt:{cmd:r(hausman_reject)}}Yes/No{p_end}
{synopt:{cmd:r(sargan_J)}}Sargan J statistic{p_end}
{synopt:{cmd:r(sargan_p)}}p-value of Sargan test{p_end}
{synopt:{cmd:r(sargan_reject)}}Yes/No{p_end}
{synopt:{cmd:r(redund_F)}}Redundancy F statistic{p_end}
{synopt:{cmd:r(redund_p)}}p-value of redundancy test{p_end}
{synopt:{cmd:r(redund_reject)}}Yes/No{p_end}
{synoptline}

{marker example}{...}
{title:Example}

{phang}{cmd:* Build data-driven STWM}{p_end}
{phang}{cmd:. sort year citycode}{p_end}
{phang}{cmd:. stwm_build lnpatent, swm("swm.dta") n(276) t(14) saveas(STWM_I)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Build exogenous benchmark (inverse-distance, no time component)}{p_end}
{phang}{cmd:. stwm_build lnpatent, swm("swm_geo.dta") n(276) t(14) decay(exponential) param(1) saveas(STWM_G)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Run exogeneity battery}{p_end}
{phang}{cmd:. stwm_hausman lnpatent lnrd lnfdi lnhuman lninfra lngdp lnpop, w(STWM_I) wg(STWM_G)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Access stored results}{p_end}
{phang}{cmd:. return list}{p_end}

{marker also_see}{...}
{title:Also see}

{psee}
{helpb stwm_build} — Build STWM

{psee}
{helpb stwm_sdm} — Spatial Durbin Model

{psee}
{helpb stwm_granger} — Granger causality justification for TWM
