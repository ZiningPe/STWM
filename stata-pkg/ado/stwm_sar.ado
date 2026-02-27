*! stwm_sar.ado  version 1.0.0  27Feb2026
*! Spatial Autoregressive (SAR / Spatial Lag) Model via STWM
program define stwm_sar, eclass
    version 16.0
    syntax varlist(min=2 numeric) [if] [in] ,   ///
        W(name)                                  ///
        [METHOD(string) EFFects(string)          ///
         N(integer 0) NDRaws(integer 2000) NBUrn(integer 500)]
    marksample touse
    if "`method'"  == "" local method  "ml"
    if "`effects'" == "" local effects "none"
    tokenize `varlist'
    local depvar "`1'"; macro shift; local indepvars "`*'"
    if "${`w'_path}" == "" {
        di as error "STWM '`w'' not found. Run: stwm_build ..., saveas(`w')"
        exit 111
    }
    if "`effects'" != "none" & `n' == 0 {
        di as error "Option n(#) required when effects(fe/re) is specified."
        exit 198
    }
    di as text "Fitting SAR: method=`method', effects=`effects' ..."
    python: _stwm_fit("`depvar'","`indepvars'","`touse'","`w'","sar",  ///
                      "`method'","`effects'",`n',`ndraws',`nburn',0)
    _stwm_post_eresults "`depvar'" "`indepvars'" "`w'" "SAR" "`method'" "`effects'"
    di as result "SAR estimation complete."
    di as text   "  ρ = " %8.4f e(rho) "  SE = " %8.4f e(rho_se)
    di as text   "  Use estat effects  to display direct/indirect/total"
end
