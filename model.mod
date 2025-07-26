//=========================================================================
// DEFINITIONS
//=========================================================================

@#define shocks = ["A", "U", "N", "I", "G", "T", "F", "m", "W", "P", "rp"]
@#define tax_rules = ["C", "K", "N"]
@#define regimes = ["_M", "_F"]

% 'shocks': identifiers for structural shocks and shock processes
% 'tax_rules': fiscal rules for C = consumption, K = capital, N = labor
% 'regimes': Monetary dominated (M) vs Fiscally dominated (F)
//=========================================================================
// DECLARATION OF ENDOGENOUS VARIABLES
//=========================================================================
var
c_Rstar_M    c_Rstar_F    c_Rstar
invest_M     invest_F     invest
lambda_M     lambda_F     lambda
rL_M         rL_F         rL
ups_M        ups_F        ups
pK_M         pK_F         pK
k_M          k_F          k
pL_M         pL_F         pL
varphi_R_M   varphi_R_F   varphi_R
z_M          z_F          z
pi_w_M       pi_w_F       pi_w
l_M          l_F          l
u_M          u_F          u
c_M          c_F          c
cstar_M      cstar_F      cstar
c_R_M        c_R_F        c_R
c_H_M        c_H_F        c_H
y_M          y_F          y
sVA_M        sVA_F        sVA
w_M          w_F          w
rK_M         rK_F         rK
n_M          n_F          n
s_M          s_F          s
bL_M         bL_F         bL
pi_p_M       pi_p_F       pi_p
rS_M         rS_F         rS
tau_M        tau_F        tau
g_M          g_F          g
tauK_M       tauK_F       tauK
tauC_M       tauC_F       tauC
tauN_M       tauN_F       tauN
sB_M         sB_F         sB

%==== Observables ====%
dy_obs         
dc_obs         
dinvest_obs    
dg_obs         
dtau_obs       
pi_w_obs       
pi_p_obs       
u_obs          
sB_obs         
rS_obs         

%==== Shocks ====%
@#for shock in shocks
    zeta_@{shock}
@#endfor
;

//=========================================================================
// DECLARATION OF EXOGENOUS VARIABLES
//=========================================================================
varexo
@#for shock in shocks
    eps_@{shock}
@#endfor
;

//=========================================================================
// DECLARATION OF PARAMETERS
//=========================================================================
parameters
%==== Structural parameters ====%
bbeta            (long_name='Discount factor')
varkappa         (long_name='Steady-state growth rate')
eta              (long_name='Habit parameter')
nu_I             (long_name='Investment adjustement cost')
vartheta         (long_name='Capital utilization cost')
delta            (long_name='Depreciation rate')
rho              (long_name='Rate of decay')
psi              (long_name='Consumption trend exponent')
nu               (long_name='Inverse frish elasticity')
theta            (long_name='Capital share')
omega            (long_name='Share of hand-to-mouth consumer')
alpha_G          (long_name='Sub. between gov. and private consumption')
alpha_w          (long_name='Wage Calvo parameter')
alpha_p          (long_name='Price Calvo parameter')
gamma_w          (long_name='Wage inflation indexation')
gamma_p          (long_name='Price inflation indexation')
thetaw           (long_name='Wage elasticity')
thetap           (long_name='Price elasticity')
sM               (long_name='Share of intermediate inputs')
chi              (long_name='Labor disutility scale factor')

%==== Calibrated Steady-states ====%
SG              (long_name='SS ratio: public consumption')
SB              (long_name='SS ratio: debt-to-GDP')
tauC_ss         (long_name='SS Consumption tax')
tauN_ss         (long_name='SS Labor tax')
tauK_ss         (long_name='SS Capital tax')
Pi              (long_name='SS level: inflation')

%==== Other Steady-states ====%
SC              (long_name='SS ratio: private consumption')
SCstar          (long_name='SS ratio: total consumption')
SBprime         (long_name='SS ratio: model implied debt-to-GDP ratio')
SI              (long_name='SS ratio: investment')
ST              (long_name='SS ratio: transfers')
SCh             (long_name='SS ratio: non-Ricardian private consumption')
SCr             (long_name='SS ratio: Ricardian private consumption')
U               (long_name='SS level: unemployment')

%===== Fiscal persistence parameters =====%
rho_R           (long_name='AR coeff. monetary rule')
rho_g           (long_name='AR coeff. gov. consumption rule')
rho_tau         (long_name='AR coeff. transfers rule')
rho_tauC        (long_name='AR coeff. tax on consumption rule')
rho_tauK        (long_name='AR coeff. tax on capital rule')
rho_tauN        (long_name='AR coeff. tax on labor rule')

%===== Fiscal response parameters =====%
phi_pi          (long_name='Monetary response to inflation')
phi_y           (long_name='Monetary response to unemployment')
gamma_tau       (long_name='Transfers response to debt')
phi_tau_y       (long_name='Transfers response to unemployment')
gamma_g         (long_name='Gov. consumption response to debt')
gamma_C         (long_name='Tax on consumption response to debt')
gamma_K         (long_name='Tax on capital response to debt')
gamma_N         (long_name='Tax on labor response to debt')

%===== Observable constant parameters ====%         
cst_dg_obs          
cst_dtau_obs  
cst_pi_p_obs
cst_rS_obs           

%===== Shock persistence & variance parameters =====%
@#for shock in shocks
    rho_@{shock}
    se_@{shock}
@#endfor
;

//=========================================================================
// PARAMETER VALUES
//=========================================================================

%==== Structural parameters values ====%
bbeta     = 0.99;
varkappa  = 0.005;
eta       = 0.5;
nu_I      = 6;      
vartheta  = 0.5;
delta     = 0.025;
rho       = 0.9593;
psi       = 0.5;
nu        = 2;   
theta     = 0.33;
omega     = 0.11;
alpha_G   = 0;
alpha_w   = 0.5;
alpha_p   = 0.5; 
gamma_w   = 0.5;  
gamma_p   = 0.5;
thetaw    = 7;
thetap    = 7;
sM        = 0.25;
chi       = 1;

%==== Steady-state calibration ====%
SB         = 0.7975;
SG         = 0.20;
tauC_ss    = 0.20;
tauN_ss    = 0.35;
tauK_ss    = 0.30;
Pi         = 1;

%===== Fiscal parameters values =====%
rho_R     = 0.5;    
rho_g     = 0.5;   
rho_tau   = 0.5;    
rho_tauC  = 0.0;  
rho_tauK  = 0.5; 
rho_tauN  = 0.5;

%===== Fiscal response parameters values =====%
phi_pi    = 2;
phi_y     = 0.25; 
gamma_tau = 0.25;  
phi_tau_y = 0.10; 
gamma_g   = 0.25;
gamma_C   = 0.0;
gamma_K   = 0.25;
gamma_N   = 0.25;

%===== Values ====%
cst_dg_obs      = 0.0029;
cst_dtau_obs    = 0.0035;
cst_pi_p_obs    = 0.0040;
cst_rS_obs      = 0.0042;

%===== Shock persistence & variance =====%
@#for shock in shocks
    rho_@{shock} = 0.5;
    se_@{shock} = 1;
@#endfor

//=========================================================================
// STEADY-STATE RESTRICTIONS
//=========================================================================

SI = (exp(varkappa) - (1 - delta)) * (bbeta * (1 - tauK_ss) * theta)
                                   / (exp(varkappa) - bbeta * (1 - delta));

SC = 1 - SI - SG;

SBprime = (1 + (exp(varkappa)*Pi)^(-1) + (exp(varkappa)*Pi)^(-2) + 
                                             (exp(varkappa)*Pi)^(-3)) * SB;

ST = tauN_ss * (1 - theta) + tauK_ss * theta + tauC_ss * SC
                                             - (1 - bbeta)/bbeta * SB - SG;

SCh = (1 / (1 + tauC_ss)) * ((1 - tauN_ss)*(1 - theta) + ST);
    
SCr = (1 / (1 - omega)) * (SC - omega * SCh);

SCstar = SC + alpha_G * SG;

U = 1 - ( 1 / (thetaw / ( thetaw - 1 )) )^(1/nu);

//=========================================================================
// MODEL EQUATIONS
//=========================================================================
model(linear);

%==========================================================================
%                              CORE SYSTEM
%==========================================================================
%----------------------------------(1)-------------------------------------
@#for FTPL in regimes
    [name = 'Consumption FOC (@{FTPL})']
    lambda@{FTPL} + (tauC_ss / (1 + tauC_ss)) * tauC@{FTPL} = zeta_U -
                     (1 / (1 - eta * exp(-varkappa))) * (c_Rstar@{FTPL} - 
                     eta * exp(-varkappa) * (c_Rstar@{FTPL}(-1) - zeta_A));
@#endfor

%----------------------------------(2)-------------------------------------
@#for FTPL in regimes
    [name = 'Investment FOC (@{FTPL})']
    pK@{FTPL} + zeta_I = exp(2*varkappa) * nu_I 
                 * (zeta_A + invest@{FTPL} - invest@{FTPL}(-1))
                       - bbeta * exp(2*varkappa) * nu_I * (zeta_A(+1) 
                                      + invest@{FTPL}(+1) - invest@{FTPL});
@#endfor

%----------------------------------(3)-------------------------------------
@#for FTPL in regimes
    [name = 'Short-term bond FOC (@{FTPL})']
    lambda@{FTPL} = lambda@{FTPL}(+1) - zeta_A(+1) 
                                   + rS@{FTPL} - pi_p@{FTPL}(+1) + zeta_rp;
@#endfor

%----------------------------------(4)-------------------------------------
@#for FTPL in regimes
    [name = 'Long-term bond FOC (@{FTPL}']
    lambda@{FTPL} = lambda@{FTPL}(+1) - zeta_A(+1) 
                                         + rL@{FTPL}(+1) - pi_p@{FTPL}(+1);
@#endfor

%----------------------------------(5)-------------------------------------
@#for FTPL in regimes
    [name = 'Capital utilization FOC (@{FTPL})']
    rK@{FTPL} - (tauK_ss / (1 - tauK_ss)) * tauK@{FTPL} = 
                                    (vartheta / (1-vartheta)) * ups@{FTPL};
@#endfor

%----------------------------------(6)-------------------------------------
@#for FTPL in regimes
    [name = 'Shadow price of capital (@{FTPL})']
    pK@{FTPL} = lambda@{FTPL}(+1) - lambda@{FTPL} - zeta_A(+1) + 
        (1 - (1 - delta)) * (rK@{FTPL} - 
                (tauK_ss / (1 - tauK_ss)) * tauK@{FTPL}(+1)) 
                    + bbeta * exp(-varkappa) * (1 - delta) * pK@{FTPL}(+1);
@#endfor

%----------------------------------(7)-------------------------------------
@#for FTPL in regimes
    [name = 'Law of motion for capital (@{FTPL})']
    k@{FTPL} = (1 - delta) * exp(-varkappa) * (k@{FTPL}(-1) - zeta_A) + 
            (1 - (1 - delta) * exp(-varkappa)) *  (invest@{FTPL} + zeta_I);
@#endfor

%----------------------------------(8)-------------------------------------
@#for FTPL in regimes
    [name = 'Long-term bond price (@{FTPL})']
    pL@{FTPL} = (exp(varkappa) * Pi / (rho * bbeta)) * 
                                               (rL@{FTPL} + pL@{FTPL}(-1));
@#endfor

%----------------------------------(9)-------------------------------------
@#for FTPL in regimes
    [name = 'Taste shifter (@{FTPL})']
    varphi_R@{FTPL} = z@{FTPL} - (1 / (1 - eta * exp(-varkappa))) 
                               * (c_Rstar@{FTPL} - eta * exp(-varkappa) 
                                          * (c_Rstar@{FTPL}(-1) - zeta_A));
@#endfor

%----------------------------------(10)------------------------------------
@#for FTPL in regimes
    [name = 'Consumption trend (@{FTPL})']
    z@{FTPL} = (1 - psi) * (z@{FTPL}(-1) - zeta_A) 
                               + (psi / (1 - eta * exp(-varkappa))) 
                                   * (cstar@{FTPL} - eta * exp(-varkappa)
                                            * (cstar@{FTPL}(-1) - zeta_A));
@#endfor

%----------------------------------(11)------------------------------------
@#for FTPL in regimes
    [name = 'Wage inflation (@{FTPL})']
    pi_w@{FTPL} = pi_p@{FTPL} + w@{FTPL} - w@{FTPL}(-1) + zeta_A;
@#endfor

%----------------------------------(12)------------------------------------    
@#for FTPL in regimes
    [name = 'Labor supply condition (@{FTPL})']
    nu * l@{FTPL} + z@{FTPL} + zeta_N - w@{FTPL}
                          + (tauN_ss / (1 - tauN_ss)) * tauN@{FTPL}
                             + (tauC_ss / (1 + tauC_ss)) * tauC@{FTPL} = 0;
@#endfor

%----------------------------------(13)------------------------------------
@#for FTPL in regimes
    [name = 'Unemployment (@{FTPL})']
    (U / (1 - U)) * u@{FTPL} = l@{FTPL} - n@{FTPL};
@#endfor

%----------------------------------(14)------------------------------------
@#for FTPL in regimes
    [name = 'Hand-to-mouth BC (@{FTPL})']
    c_H@{FTPL} + (tauC_ss / (1 + tauC_ss)) * tauC@{FTPL} = 
       ((1-tauN_ss)*(1-theta) / ((1-tauN_ss)*(1-theta) + ST))
        * ( w@{FTPL} + n@{FTPL} - (tauN_ss / (1 - tauN_ss)) * tauN@{FTPL}) 
                        + (ST / ((1-tauN_ss)*(1-theta) + ST)) * tau@{FTPL};
@#endfor

%----------------------------------(15)------------------------------------
@#for FTPL in regimes
  [name = 'Private consumption (@{FTPL})']
  c@{FTPL} = (omega * SCh) / (omega * SCh + (1-omega) * SCr)
      * c_H@{FTPL} + (omega * SCr / (omega * SCh + (1 - omega) * SCr)) 
                                                              * c_R@{FTPL};
@#endfor

%----------------------------------(16)------------------------------------
@#for FTPL in regimes
  [name = 'Total consumption (@{FTPL})']
  cstar@{FTPL} = (SC / SCstar) * c@{FTPL} + 
                                        (alpha_G * SG / SCstar) * g@{FTPL};
@#endfor

%----------------------------------(17)------------------------------------
@#for FTPL in regimes
  [name = 'Ricardian agents total consumption (@{FTPL})']
  c_Rstar@{FTPL} = (SCr / (SCr + alpha_G * SG)) * c_R@{FTPL}
                      + ((alpha_G * SG) / (SCr + alpha_G * SG)) * g@{FTPL};
@#endfor

%----------------------------------(18)------------------------------------
@#for FTPL in regimes
  [name = 'Resource constraint (@{FTPL})']
  y@{FTPL} = SC * c@{FTPL} + SI * invest@{FTPL} + SG * g@{FTPL} +
                                        (1 - tauK_ss) * theta * ups@{FTPL};
@#endfor

%----------------------------------(19)------------------------------------
@#for FTPL in regimes
  [name = 'Real marginal cost (@{FTPL})']
  s@{FTPL} = (1 - (thetap / ( thetap - 1 )) * sM) * sVA@{FTPL};
@#endfor

%----------------------------------(20)------------------------------------
@#for FTPL in regimes
  [name = 'Real wage (@{FTPL} FTPL)']
  w@{FTPL} = sVA@{FTPL} + theta *
                           (ups@{FTPL} + k@{FTPL}(-1) - n@{FTPL} - zeta_A);
@#endfor

%----------------------------------(21)------------------------------------
@#for FTPL in regimes
  [name = 'Real rental rate of capital (@{FTPL})']
  rK@{FTPL} = sVA@{FTPL} + (theta - 1) * 
                    (ups@{FTPL} + k@{FTPL}(-1) - n@{FTPL} - zeta_A);
@#endfor

%----------------------------------(22)------------------------------------
@#for FTPL in regimes
  [name = 'Production function (@{FTPL})']
  ((1 - (thetap / ( thetap - 1 )) * sM) / ((thetap / ( thetap - 1 ))
     * (1 - sM))) * y@{FTPL} = theta * (ups@{FTPL} + k@{FTPL}(-1) - zeta_A)
                                                  + (1 - theta) * n@{FTPL};
@#endfor

%----------------------------------(23)------------------------------------
@#for FTPL in regimes
  [name = 'Government BC (@{FTPL})']
  SBprime * (pL@{FTPL} + bL@{FTPL})
  + tauN_ss * (1 - theta) * (tauN@{FTPL} + w@{FTPL} + n@{FTPL})
  + tauK_ss * theta * (tauK@{FTPL} + rK@{FTPL} - zeta_A 
  + k@{FTPL}(-1) + ups@{FTPL})
  + tauC_ss * SC * (tauC@{FTPL} + c@{FTPL})
      = (1 / bbeta) * SBprime * (rL@{FTPL} + pL@{FTPL}(-1) + bL@{FTPL}(-1) 
                 - zeta_A - pi_p@{FTPL}) + SG * g@{FTPL} + ST * tau@{FTPL};
@#endfor

%----------------------------------(24)------------------------------------
@#for FTPL in regimes
  [name = 'Debt‐to‐output ratio (@{FTPL})']
  sB@{FTPL} = pL@{FTPL} + bL@{FTPL} - (1 / (1 + exp(-varkappa) * Pi^(-1) 
                                            + exp(-2*varkappa) * Pi^(-2) 
                                            + exp(-3*varkappa) * Pi^(-3)))
    * (y@{FTPL} + y@{FTPL}(-1) + y@{FTPL}(-2) + y@{FTPL}(-3)
            - (exp(-varkappa) * Pi^(-1) 
             + exp(-2*varkappa) * Pi^(-2) 
             + exp(-3*varkappa) * Pi^(-3)) 
             * (pi_p@{FTPL} + zeta_A)
               - (exp(-2*varkappa) * Pi^(-2) 
               + exp(-3*varkappa) * Pi^(-3)) 
               * (pi_p@{FTPL}(-1) + zeta_A(-1))
                 - (exp(-3*varkappa) * Pi^(-3)) 
                 * (pi_p@{FTPL}(-2) + zeta_A(-2)));
@#endfor

%----------------------------------(25)------------------------------------
@#for FTPL in regimes
  [name = 'Wage NKPC (@{FTPL})']
  pi_w@{FTPL} - gamma_w * pi_p@{FTPL}(-1)
      = ((1 - bbeta * alpha_w) * (1 - alpha_w)) / 
          (alpha_w * (1 + nu * thetaw))* nu * (n@{FTPL} - l@{FTPL})
              + bbeta * (pi_w@{FTPL}(+1) - gamma_w * pi_p@{FTPL}) + zeta_W;
@#endfor

%----------------------------------(26)------------------------------------
@#for FTPL in regimes
  [name = 'Price NKPC (@{FTPL})']
  pi_p@{FTPL} - gamma_p * pi_p@{FTPL}(-1)
      = ((1 - bbeta * alpha_p) * (1 - alpha_p) / alpha_p) * s@{FTPL}
              + bbeta * (pi_p@{FTPL}(+1) - gamma_p * pi_p@{FTPL}) + zeta_P;
@#endfor

%==========================================================================
%                              FISCAL BLOCK
%==========================================================================
%----------------------------------(27)------------------------------------
@#for FTPL in regimes

  @#if FTPL == "_F"
    [name = 'Monetary policy rule (F)']
    rS_F = rho_R * rS_F(-1) + (1 - rho_R) * (phi_y * y_F);
    
  @#else
    [name = 'Monetary policy rule (M)']
    rS_M = rho_R * rS_M(-1)
          + (1 - rho_R) * (phi_pi * (pi_p_M) + phi_y * y_M) + zeta_m;
  @#endif

@#endfor

%----------------------------------(28)------------------------------------
@#for FTPL in regimes

  @#if FTPL == "_F"
    [name = 'Transfer rule (F)']
    tau_F = rho_tau * tau_F(-1)
        - (1 - rho_tau) * (phi_tau_y * y_F) + zeta_F;

  @#else
    [name = 'Transfer rule (M)']
    tau_M = rho_tau * tau_M(-1)
        - (1 - rho_tau) * (gamma_tau * (sB_M(-1)) + phi_tau_y * y_M) 
        + zeta_T;
  @#endif

@#endfor

%----------------------------------(29)------------------------------------
@#for FTPL in regimes

  @#if FTPL == "_F"
    [name = 'Public spending rule (F)']
    g_F = rho_g * g_F(-1);

  @#else
    [name = 'Public spending rule (M)']
    g_M = rho_g * g_M(-1) - (1 - rho_g)*gamma_g * (sB_M(-1)) + zeta_G;
  @#endif

@#endfor

%----------------------------------(30)------------------------------------
@#for tax in tax_rules

  @#for FTPL in regimes
      @#if FTPL == "_F"
      [name = '@{tax} tax rule (F)']
      tau@{tax}_F = rho_tau@{tax} * tau@{tax}_F(-1);

      @#else
      [name = '@{tax} tax rule (M)']
      tau@{tax}_M = rho_tau@{tax} * tau@{tax}_M(-1)
                     + (1 - rho_tau@{tax})* gamma_@{tax} * (sB_M(-1));
      @#endif
  @#endfor

@#endfor

%==========================================================================
%                            STRUCTURAL SHOCKS
%==========================================================================

@#for shock in shocks
    zeta_@{shock} = rho_@{shock} * zeta_@{shock}(-1) 
                                        + (se_@{shock}/100) * eps_@{shock};
@#endfor

%==========================================================================
%                            AGGREGATION
%==========================================================================

c_Rstar  = c_Rstar_M + c_Rstar_F;
invest   = invest_M  + invest_F;
lambda   = lambda_M  + lambda_F;
rL       = rL_M      + rL_F;
ups      = ups_M     + ups_F;
pK       = pK_M      + pK_F;
k        = k_M       + k_F;
pL       = pL_M      + pL_F;
varphi_R = varphi_R_M + varphi_R_F;
z        = z_M       + z_F;
pi_w     = pi_w_M    + pi_w_F;
l        = l_M       + l_F;
u        = u_M       + u_F;
c        = c_M       + c_F;
cstar    = cstar_M   + cstar_F;
c_R      = c_R_M     + c_R_F;
c_H      = c_H_M     + c_H_F;
y        = y_M       + y_F;
sVA      = sVA_M     + sVA_F;
w        = w_M       + w_F;
rK       = rK_M      + rK_F;
n        = n_M       + n_F;
s        = s_M       + s_F;
bL       = bL_M      + bL_F;
pi_p     = pi_p_M    + pi_p_F;
rS       = rS_M      + rS_F;
tau      = tau_M     + tau_F;
g        = g_M       + g_F;
tauK     = tauK_M    + tauK_F;
tauC     = tauC_M    + tauC_F;
tauN     = tauN_M    + tauN_F;
sB       = sB_M      + sB_F;

%==========================================================================
%                       OBSERVABLE EQUATION
%==========================================================================

dy_obs      = varkappa      + y - y(-1) + zeta_A;
dc_obs      = varkappa      + c - c(-1) + zeta_A;
dinvest_obs = varkappa      + invest - invest(-1) + zeta_A;
dg_obs      = cst_dg_obs    + g - g(-1);
dtau_obs    = cst_dtau_obs  + tau - tau(-1);
pi_w_obs    = varkappa      + w - w(-1) + zeta_A;
pi_p_obs    = cst_pi_p_obs  + pi_p;
u_obs       = U             + u;
sB_obs      = SB            + sB;
rS_obs      = cst_rS_obs    + rS;

end;

//=========================================================================
// SHOCK VARIANCES
//=========================================================================
shocks;
    @#for shock in shocks
        var eps_@{shock}; stderr 1;
    @#endfor
end;

//=========================================================================
// OBSERVABLE VARIABLES
//=========================================================================
varobs dy_obs dc_obs dinvest_obs dg_obs dtau_obs pi_w_obs pi_p_obs
     u_obs sB_obs rS_obs;

//=========================================================================
// PRIORS
//=========================================================================
estimated_params;
    cst_dg_obs,       normal_pdf, 0.0029, 0.001;
    cst_dtau_obs,     normal_pdf, 0.0035, 0.001;
    cst_pi_p_obs,     normal_pdf, 0.0040, 0.001;
    cst_rS_obs,       normal_pdf, 0.0042, 0.001;
    U,                normal_pdf, 0.0949, 0.001;
    SB,               normal_pdf, 0.7974, 0.001;
    varkappa,         normal_pdf, 0.003, 0.001;  % 1.2% growth a year

    se_A,             inv_gamma_pdf, 1.0, 2.0;
    se_U,             inv_gamma_pdf, 1.0, 2.0;
    se_N,             inv_gamma_pdf, 1.0, 2.0;
    se_I,             inv_gamma_pdf, 1.0, 2.0;
    se_G,             inv_gamma_pdf, 1.0, 2.0;
    se_T,             inv_gamma_pdf, 1.0, 2.0;
    se_F,             inv_gamma_pdf, 1.0, 2.0;
    se_m,             inv_gamma_pdf, 1.0, 2.0;
    se_W,             inv_gamma_pdf, 1.0, 2.0;
    se_P,             inv_gamma_pdf, 1.0, 2.0;
    se_rp,            inv_gamma_pdf, 1.0, 2.0;

    rho_A,            beta_pdf, 0.50, 0.10;
    rho_U,            beta_pdf, 0.50, 0.10;
    rho_N,            beta_pdf, 0.50, 0.10;
    rho_I,            beta_pdf, 0.50, 0.10;
    rho_G,            beta_pdf, 0.50, 0.10;
    rho_T,            beta_pdf, 0.50, 0.10;
    rho_F,            beta_pdf, 0.50, 0.10;
    rho_m,            beta_pdf, 0.50, 0.10;
    rho_W,            beta_pdf, 0.50, 0.10;
    rho_P,            beta_pdf, 0.50, 0.10;
    rho_rp,           beta_pdf, 0.50, 0.10;

    nu,               normal_pdf, 2.000, 0.250;
    alpha_G,          normal_pdf, 0.000, 0.100;
    nu_I,             normal_pdf, 6.000, 0.500;
    vartheta,         beta_pdf,   0.500, 0.100;
    eta,              beta_pdf,   0.500, 0.200;  
    omega,            beta_pdf,   0.300, 0.100;  % Leeper (2017)
    psi,              beta_pdf,   0.500, 0.200;  % GSW (2012)
    gamma_p,          beta_pdf,   0.500, 0.200;
    gamma_w,          beta_pdf,   0.500, 0.200;
    alpha_p,          beta_pdf,   0.750, 0.050;  % Smets & Wouters (2003)
    alpha_w,          beta_pdf,   0.750, 0.050;  % Smets & Wouters (2003)
    phi_pi,           normal_pdf, 2.000, 0.100;
    phi_y,            normal_pdf, 0.125, 0.050;  % Smets & Wouters (2003)
    rho_R,            beta_pdf,   0.500, 0.100;  % Smets & Wouters (2003)
    phi_tau_y,        gamma_pdf,  0.100, 0.050;
    rho_tau,          beta_pdf,   0.500, 0.100;
    gamma_tau,        normal_pdf, 0.250, 0.100;
    gamma_g,          normal_pdf, 0.250, 0.100;
    rho_g,            beta_pdf,   0.500, 0.100;
    gamma_K,          normal_pdf, 0.250, 0.100;
    gamma_N,          normal_pdf, 0.250, 0.100;
    rho_tauK,         beta_pdf,   0.500, 0.100;
    rho_tauN,         beta_pdf,   0.500, 0.100;
end;

//=========================================================================
// INITIAL VALUES
//=========================================================================
estimated_params_init;
    cst_dg_obs          ,   0.00339128;
    cst_dtau_obs        ,   0.00408306;
    cst_pi_p_obs        ,   0.00408887;
    cst_rS_obs          ,   0.00452112;
    U                   ,   0.09582771;
    SB                  ,   0.79738977;
    varkappa            ,   0.00226592;
    se_A                ,   1.70756647;
    se_U                ,   2.61013172;
    se_N                ,   0.86796081;
    se_I                ,   4.17975099;
    se_G                ,   0.49675512;
    se_T                ,   0.30117051;
    se_F                ,   0.27681411;
    se_m                ,   0.13278783;
    se_W                ,   0.66039441;
    se_P                ,   0.14152651;
    se_rp               ,   0.17533780;
    rho_A               ,   0.15592951;
    rho_U               ,   0.72755484;
    rho_N               ,   0.88171470;
    rho_I               ,   0.56403615;
    rho_G               ,   0.35626214;
    rho_T               ,   0.54513086;
    rho_F               ,   0.45949901;
    rho_m               ,   0.49453586;
    rho_W               ,   0.33629965;
    rho_P               ,   0.19116446;
    rho_rp              ,   0.66101815;
    nu                  ,   2.41048708;
    alpha_G             ,   0.02936583;
    nu_I                ,   6.61946141;
    vartheta            ,   0.16247719;
    eta                 ,   0.84148840;
    omega               ,   0.38099680;
    psi                 ,   0.28017827;
    gamma_p             ,   0.59529230;
    gamma_w             ,   0.56360443;
    alpha_p             ,   0.82765648;
    alpha_w             ,   0.56134045;
    phi_pi              ,   1.94726921;
    phi_y               ,   0.03018203;
    rho_R               ,   0.70981246;
    phi_tau_y           ,   0.06794931;
    rho_tau             ,   0.76813400;
    gamma_tau           ,   0.26764555;
    gamma_g             ,   0.27498846;
    rho_g               ,   0.76824782;
    gamma_K             ,   0.08892395;
    gamma_N             ,   0.31590332;
    rho_tauK            ,   0.49856607;
    rho_tauN            ,   0.50146681;
end;

estimation(
    datafile=mydata,
    first_obs=1,
    nobs=79,
    order=1,
    mh_replic=250000,
    mh_nblocks=1,
    mh_jscale=0.25,
    mh_drop=0.25,
    lyapunov=doubling, 
    plot_priors=0); 

steady;
check;
model_diagnostics;
stoch_simul(irf=48) y pi_p rS sB;

shock_groups(name=group1);
'Supply Shocks' = eps_A, eps_P, eps_W, eps_I, eps_N;
'Demand Shocks' = eps_U, eps_rp;
'Policy Shocks' = eps_m, eps_G, eps_T;
'Unfunded Transfers Shocks' = eps_F;
end;

shock_decomposition(use_shock_groups=group1) y pi_p tau sB;

