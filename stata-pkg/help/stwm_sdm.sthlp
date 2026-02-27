{smcl}
{* *! version 1.0.0  27Feb2026}{...}
{viewerjumpto "Syntax"        "stwm_sdm##syntax"}{...}
{viewerjumpto "Description"   "stwm_sdm##description"}{...}
{viewerjumpto "Options"       "stwm_sdm##options"}{...}
{viewerjumpto "Stored results""stwm_sdm##results"}{...}
{viewerjumpto "Post-estimation""stwm_sdm##postestimation"}{...}
{viewerjumpto "Example"       "stwm_sdm##example"}{...}
{viewerjumpto "Also see"      "stwm_sdm##also_see"}{...}

{title:Title}

{phang}
{bf:stwm_sdm} {hline 2} Spatial Durbin Model (SDM) via Spatial-Temporal Weight Matrix

{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:stwm_sdm} {varlist} {ifin}{cmd:,}
{cmd:w(}{it:name}{cmd:)}
[{it:options}]

{synoptset 24 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{opt w(name)}}name of pre-built STWM (from {helpb stwm_build}){p_end}
{syntab:Optional}
{synopt:{opt me:thod(string)}}estimation method: {cmd:ml} (default), {cmd:qml}, {cmd:iv}, {cmd:gmm}, {cmd:bayes}{p_end}
{synopt:{opt eff:ects(string)}}panel effects: {cmd:none} (default), {cmd:fe}, {cmd:re}{p_end}
{synopt:{opt n(#)}}number of spatial units; required when {cmd:effects(fe/re)}{p_end}
{synopt:{opt nocons:tant}}suppress intercept{p_end}
{synopt:{opt ndr:aws(#)}}MCMC draws for Bayesian estimation; default {cmd:2000}{p_end}
{synopt:{opt nbur:n(#)}}MCMC burn-in for Bayesian estimation; default {cmd:500}{p_end}
{synoptline}

{marker description}{...}
{title:Description}

{pstd}
{cmd:stwm_sdm} fits the Spatial Durbin Model (SDM):

{phang2}
{bf:Y = δ·W·Y + X·β + W·X·θ + ε}

{pstd}
where {bf:W} is the pre-built STWM (registered under name {it:name}),
δ is the spatial autoregressive parameter, β are direct-covariate
coefficients, and θ captures spillover effects from spatially-lagged X.

{pstd}
Results are posted as {cmd:e()}-class, fully compatible with {cmd:test},
{cmd:lincom}, {cmd:nlcom}, and other post-estimation commands.

{marker options}{...}
{title:Options}

{phang}
{cmd:method(ml)} [default] Maximum likelihood via sparse LU factorisation.

{phang}
{cmd:method(qml)} Quasi-ML (heteroskedasticity-robust).

{phang}
{cmd:method(iv)} Instrumental-variables estimator with WX instruments.

{phang}
{cmd:method(gmm)} Generalised Method of Moments.

{phang}
{cmd:method(bayes)} Bayesian MCMC (Gibbs sampler); use {cmd:ndraws()} and {cmd:nburn()}.

{phang}
{cmd:effects(fe)} Within-group (demeaned) panel estimator.
Requires {cmd:n(#)} specifying the number of spatial units.

{phang}
{cmd:effects(re)} Random-effects (Mundlak-Chamberlain) panel estimator.
Requires {cmd:n(#)}.

{marker results}{...}
{title:Stored results}

{pstd}
{cmd:stwm_sdm} stores the following in {cmd:e()}:

{synoptset 22 tabbed}{...}
{syntab:Scalars}
{synopt:{cmd:e(N)}}number of observations{p_end}
{synopt:{cmd:e(k)}}number of regressors (excl. spatial parameter){p_end}
{synopt:{cmd:e(rho)}}estimated spatial parameter δ{p_end}
{synopt:{cmd:e(rho_se)}}standard error of δ{p_end}
{synopt:{cmd:e(sigma2)}}residual variance{p_end}
{synopt:{cmd:e(theta_re)}}RE random-effects variance (if effects(re)){p_end}
{syntab:Matrices}
{synopt:{cmd:e(b)}}coefficient row-vector: [β | θ_WX | _cons | δ]{p_end}
{synopt:{cmd:e(V)}}variance-covariance matrix of {cmd:e(b)}{p_end}
{synopt:{cmd:e(direct)}}direct effects (1 × k){p_end}
{synopt:{cmd:e(indirect)}}indirect effects (1 × k){p_end}
{synopt:{cmd:e(total)}}total effects (1 × k){p_end}
{synopt:{cmd:e(direct_se)}}SE of direct effects (1 × k){p_end}
{synopt:{cmd:e(indirect_se)}}SE of indirect effects (1 × k){p_end}
{synopt:{cmd:e(total_se)}}SE of total effects (1 × k){p_end}
{syntab:Macros}
{synopt:{cmd:e(cmd)}}{cmd:stwm_sdm}{p_end}
{synopt:{cmd:e(model)}}{cmd:SDM}{p_end}
{synopt:{cmd:e(method)}}estimation method used{p_end}
{synopt:{cmd:e(effects)}}panel effects type{p_end}
{synopt:{cmd:e(W_name)}}STWM name{p_end}
{synopt:{cmd:e(depvar)}}name of dependent variable{p_end}
{synopt:{cmd:e(indepvars)}}names of independent variables{p_end}
{synoptline}

{marker postestimation}{...}
{title:Post-estimation}

{pstd}
After {cmd:stwm_sdm} you can use:

{phang2}{cmd:estat effects} — display direct / indirect / total effect table

{phang2}{cmd:test} {it:coeflist} — Wald test on any subset of coefficients

{phang2}{cmd:lincom} {it:exp} — linear combination of coefficients

{marker example}{...}
{title:Example}

{phang}{cmd:* Step 1 – build the STWM}{p_end}
{phang}{cmd:. sort year citycode}{p_end}
{phang}{cmd:. stwm_build lnpatent, swm("swm.dta") n(276) t(14) saveas(STWM_I)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Step 2 – fit SDM (ML)}{p_end}
{phang}{cmd:. stwm_sdm lnpatent lnrd lnfdi lnhuman lninfra lngdp lnpop, w(STWM_I)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Step 3 – view effect decomposition}{p_end}
{phang}{cmd:. estat effects}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Panel FE variant}{p_end}
{phang}{cmd:. stwm_sdm lnpatent lnrd lnfdi, w(STWM_I) effects(fe) n(276)}{p_end}
{phang}{cmd:}{p_end}
{phang}{cmd:* Bayesian MCMC variant}{p_end}
{phang}{cmd:. stwm_sdm lnpatent lnrd lnfdi, w(STWM_I) method(bayes) ndraws(3000) nburn(500)}{p_end}

{marker also_see}{...}
{title:Also see}

{psee}
{helpb stwm_build} — Build STWM

{psee}
{helpb stwm_sar} — Spatial Autoregressive Model

{psee}
{helpb stwm_sem} — Spatial Error Model

{psee}
{helpb stwm_slx} — Spatial Lag of X Model

{psee}
{helpb stwm_hausman} — Exogeneity test battery
