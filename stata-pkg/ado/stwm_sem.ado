*! stwm_sem.ado  version 1.0.0  27Feb2026
*! Spatial Error Model (SEM) via STWM
program define stwm_sem, eclass
    version 16.0
    syntax varlist(min=2 numeric) [if] [in] ,   ///
        W(name)                                  ///
        [EFFects(string) N(integer 0)]
    marksample touse
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
    di as text "Fitting SEM: effects=`effects' ..."
    python: _stwm_fit("`depvar'","`indepvars'","`touse'","`w'","sem", ///
                      "ml","`effects'",`n',2000,500,0)
    _stwm_post_eresults "`depvar'" "`indepvars'" "`w'" "SEM" "ml" "`effects'"
    di as result "SEM estimation complete."
    di as text   "  λ = " %8.4f e(rho) "  SE = " %8.4f e(rho_se)
    di as text   "  Use estat effects  to display direct/indirect/total"
end
