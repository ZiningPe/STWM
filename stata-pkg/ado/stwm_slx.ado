*! stwm_slx.ado  version 1.0.0  27Feb2026
*! Spatial Lag of X (SLX) Model via STWM
program define stwm_slx, eclass
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
    di as text "Fitting SLX: effects=`effects' ..."
    python: _stwm_fit("`depvar'","`indepvars'","`touse'","`w'","slx", ///
                      "ml","`effects'",`n',2000,500,0)
    _stwm_post_eresults "`depvar'" "`indepvars'" "`w'" "SLX" "ml" "`effects'"
    di as result "SLX estimation complete."
    di as text   "  Use estat effects  to display direct/indirect/total"
end
