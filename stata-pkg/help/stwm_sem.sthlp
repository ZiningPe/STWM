{smcl}
{* *! version 1.0.0  27Feb2026}{...}
{viewerjumpto "Syntax"        "stwm_sem##syntax"}{...}
{viewerjumpto "Description"   "stwm_sem##description"}{...}
{viewerjumpto "Options"       "stwm_sem##options"}{...}
{viewerjumpto "Stored results""stwm_sem##results"}{...}
{viewerjumpto "Example"       "stwm_sem##example"}{...}
{viewerjumpto "Also see"      "stwm_sem##also_see"}{...}

{title:Title}

{phang}
{bf:stwm_sem} {hline 2} Spatial Error Model (SEM) via Spatial-Temporal Weight Matrix

{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:stwm_sem} {varlist} {ifin}{cmd:,}
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
{cmd:stwm_sem} fits the Spatial Error Model (SEM):

{phang2}
{bf:Y = X·β + u,     u = λ·W·u + ε}

{pstd}
where {bf:W} is the pre-built STWM and λ is the spatial error
autocorrelation parameter.  The SEM captures spatial clustering in
unobservables rather than in the outcome variable itself.

{marker options}{...}
{title:Options}

{phang}
{cmd:effects(none)} [default] Pooled SEM.

{phang}
{cmd:effects(fe)} Within-group panel SEM.  Requires {cmd:n(#)}.

{phang}
{cmd:effects(re)} Random-effects panel SEM.  Requires {cmd:n(#)}.

{marker results}{...}
{title:Stored results}

{pstd}
{cmd:stwm_sem} stores the following in {cmd:e()}:

{synoptset 22 tabbed}{...}
{syntab:Scalars}
{synopt:{cmd:e(N)}}number of observations{p_end}
{synopt:{cmd:e(rho)}}estimated error spatial parameter λ{p_end}
{synopt:{cmd:e(rho_se)}}standard error of λ{p_end}
{synopt:{cmd:e(sigma2)}}residual variance{p_end}
{syntab:Matrices}
{synopt:{cmd:e(b)}}coefficient row-vector: [β | _cons | λ]{p_end}
{synopt:{cmd:e(V)}}variance-covariance matrix{p_end}
{synoptline}

{pstd}
Note: {cmd:estat effects} is available but direct/indirect decomposition
is not standard for SEM; the table reports the OLS-type coefficient effects.

{marker example}{...}
{title:Example}

{phang}{cmd:. sort year citycode}{p_end}
{phang}{cmd:. stwm_build lnpatent, swm("swm.dta") n(276) t(14) saveas(STWM_I)}{p_end}
{phang}{cmd:. stwm_sem lnpatent lnrd lnfdi lnhuman, w(STWM_I)}{p_end}
{phang}{cmd:. di "Lambda = " e(rho)}{p_end}

{marker also_see}{...}
{title:Also see}

{psee}
{helpb stwm_build} — Build STWM

{psee}
{helpb stwm_sdm} — Spatial Durbin Model

{psee}
{helpb stwm_sar} — Spatial Autoregressive Model
