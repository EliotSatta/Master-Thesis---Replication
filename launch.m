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

%mod_filename = 'BFMxGSW_M';
%mod_filename = 'BFMxGSW_F';
%mod_filename = 'BFMxGSW_FTPL';
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

%% Shock decomposition

% --- pull out some Dynare pieces -----------------
exo  = M_.exo_names;        % {'eps_A','eps_U',...,'eps_rp'}
exo_n = M_.exo_nbr;         % number of shocks (11 in your case)
T    = size(oo_.shock_decomposition,3);

% find the row indices of y and pi_p in the endogenous list
y_idx   = find(strcmp(M_.endo_names,'y'));
pi_idx  = find(strcmp(M_.endo_names,'pi_p'));

% build a proper quarterly time axis
startDate = datetime(2000,4,1);    % 2000Q2
dates     = startDate + calquarters(0:T-1);

% extract the contributions [T × exo_n] and scale to percent
allContribY  = squeeze(oo_.shock_decomposition(y_idx,   1:exo_n, :))' * 100;  
allContribPi = squeeze(oo_.shock_decomposition(pi_idx,  1:exo_n, :))' * 100;

% actual smoothed series is in column exo_n+2
actualY  = squeeze(oo_.shock_decomposition(y_idx,   exo_n+2, :)) * 100;
actualPi = squeeze(oo_.shock_decomposition(pi_idx,  exo_n+2, :)) * 100;


% --- define your four groups of shocks -------------
groups = { ...
  {'eps_A','eps_P','eps_W','eps_I','eps_N'}, ...   % supply
  {'eps_U','eps_rp'},        ...   % demand
  {'eps_m','eps_G','eps_T'},  ...   % policy
  {'eps_F'}                    ...  % unfunded transfers
};

% nice easily-distinguishable colors
groupColors = [...
  0.2 0.6 1.0;   % blue    supply  
  0.2 0.8 0.2;   % green   demand
  1.0 0.4 0.2;   % orange  policy
  0.1 0.1 0.4    % navy    unfunded
];

% helper to sum exactly the columns belonging to one group
sumGroup = @(C,shockNames) ...
   sum( C(:, ismember(exo(1:exo_n), shockNames)) , 2 );


% --- build the grouped contribution matrices [T×4] ----
Y_grp  = zeros(T,4);
Pi_grp = zeros(T,4);
for i = 1:4
  Y_grp(:,i)  = sumGroup(allContribY,  groups{i});
  Pi_grp(:,i) = sumGroup(allContribPi, groups{i});
end


% --- now plot -----------------------------------------
figure('Position',[100 100 1200 400]);

% (1) OUTPUT
subplot(1,2,1); hold on;
hb = bar(dates, Y_grp, 'stacked', 'EdgeColor','none');
for i=1:4, hb(i).FaceColor = groupColors(i,:); end
plot(dates, actualY, 'k-','LineWidth',1.5);
datetick('x','yyyy-QQ'); xlim([dates(1) dates(end)]);
title('Output Decomposition (y_t)','FontWeight','normal');
ylabel('% deviation from steady state');
grid on;
legend({'Supply','Demand','Policy','Unfunded','Actual'},'Location','best');

% (2) INFLATION
subplot(1,2,2); hold on;
hb = bar(dates, Pi_grp, 'stacked', 'EdgeColor','none');
for i=1:4, hb(i).FaceColor = groupColors(i,:); end
plot(dates, actualPi, 'k-','LineWidth',1.5);
datetick('x','yyyy-QQ'); xlim([dates(1) dates(end)]);
title('Inflation Decomposition (\pi^p_t)','FontWeight','normal');
ylabel('% deviation from steady state');
grid on;
legend({'Supply','Demand','Policy','Unfunded','Actual'},'Location','best');

%% Bayesian IRFs


%%
irf_horizon = 1:48;        % IRF horizon (match with stoch_simul irf=48)
shocks = {'P', 'F', 'T'};  % Corresponding to eps_P, eps_F, eps_T
shock_names = {'Price Shock', 'Unfunded Transfer Shock', 'Funded Transfer Shock'};
shock_colors = {'b-', 'r-', 'g-'};  % Blue, Red, Green

% Variables to plot
vars = {'y', 'pi_p', 'rS','sB'};  % Output, Inflation, Real interest rate, Debt-to-output ratio
titles = {'$y_t$ - Output', '$\pi_t$ - Inflation', '$r_{S,t}$ - Nominal Interest Rate', '$s_{B,t}$ - Debt-to-output ratio'};
ylabels = {'Output', 'Inflation', 'Nominal Interest Rate', 'Debt-to-output ratio'};

% Create figure
figure;
set(gcf, 'numbertitle', 'off', 'name', ...
    ['IRFs - ', mod_filename],...
    'Position', [100, 100, 1200, 400]);

for v = 1:length(vars)
    var = vars{v};
    subplot(1, 4, v);
    hold on;

    for s = 1:length(shocks)
        shock = shocks{s};
        irf_field = [var '_eps_' shock];
        if isfield(oo_.irfs, irf_field)
            plot(irf_horizon, oo_.irfs.(irf_field)(1:length(irf_horizon)), ...
                shock_colors{s}, 'LineWidth', 1.5);
        else
            warning(['Missing IRF: ', irf_field]);
        end
    end

    yline(0, 'k--', 'LineWidth', 1);
    title(titles{v}, 'Interpreter', 'latex');
    xlabel('Time');
    ylabel(ylabels{v});
    grid on;
    hold off;
    if v == 1
        legend(shock_names, 'Location', 'Best');
    end
end