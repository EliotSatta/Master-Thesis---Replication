addpath('/Applications/Dynare/6.2-arm64/matlab');
clear; close all; clc;

% 1) Load CSV and pull out each series, naming them exactly as in varobs
raw = readtable('estimation_df.csv');
dy_obs       = raw.dy_obs;
dc_obs       = raw.dc_obs;
dinvest_obs  = raw.dinvest_obs;
dg_obs       = raw.dg_obs;
dtau_obs     = raw.dtau_obs;
pi_w_obs     = raw.pi_w_obs;
pi_p_obs     = raw.pi_p_obs;
u_obs        = raw.u_obs;
sB_obs       = raw.sB_obs;
rS_obs       = raw.rS_obs;

% 2) Save each series into a .mat
save mydata dy_obs dc_obs dinvest_obs dg_obs dtau_obs pi_w_obs pi_p_obs u_obs sB_obs rS_obs;

% 3) Compute and display the mean of each series
fprintf('Observable constant parameters \n');
fprintf('cst_dy_obs      = %.4f;\n', mean(dy_obs));
fprintf('cst_dc_obs      = %.4f;\n', mean(dc_obs));
fprintf('cst_dinvest_obs = %.4f;\n', mean(dinvest_obs));
fprintf('cst_dg_obs      = %.4f;\n', mean(dg_obs));
fprintf('cst_dtau_obs    = %.4f;\n', mean(dtau_obs));
fprintf('cst_pi_w_obs    = %.4f;\n', mean(pi_w_obs));
fprintf('cst_pi_p_obs    = %.4f;\n', mean(pi_p_obs));
fprintf('cst_u_obs       = %.4f;\n', mean(u_obs));
fprintf('cst_sB_obs      = %.4f;\n', mean(sB_obs));
fprintf('cst_rS_obs      = %.4f;\n', mean(rS_obs));

% Run Dynare
mod_filename = 'Bayesian_FTPL_BFMxGSW';
dynare(mod_filename, 'noclearall');

%% Estimated_params_init block
param_names_clean = cellstr(M_.param_names);
estimated_param_indices = estim_params_.param_vals(:,1);
estimated_param_names = param_names_clean(estimated_param_indices);
pval = oo_.posterior_mode.parameters;

fprintf('estimated_params_init;\n');
for i = 1:length(estimated_param_names)
    pname = estimated_param_names{i};
    fprintf('    %-20s, %12.8f;\n', pname, pval.(pname));
end
fprintf('end;\n');
