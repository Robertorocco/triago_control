function build_thesis_figures(results_dir, opts)
%BUILD_THESIS_FIGURES Draw every objective results figure the thesis includes.
%   BUILD_THESIS_FIGURES() runs against the most recent results folder and writes
%   thesis/figures/results/*.pdf -- the panels of Figures "resmain" and "resextra"
%   plus the practice-and-order panel. The questionnaire panels of the same folder
%   come from scripts/analysis/subjective/run_subjective_report.py instead.
%   BUILD_THESIS_FIGURES(RESULTS_DIR) uses a specific results folder.
%   Options: out_dir ('' = the thesis folder of this checkout), export_dir.
%
%   Panels are written at their native print size and are never rescaled by
%   LaTeX, so a change of size belongs in fig_thesis_panel, not in the includes.

arguments
    results_dir (1,1) string = ""
    opts.out_dir (1,1) string = ""
    opts.export_dir (1,1) string = ""
end

here = fileparts(mfilename('fullpath'));
addpath(here);

if strlength(results_dir) == 0
    root = fullfile(study_export_dir(char(opts.export_dir)), 'analysis_results');
    latest = fullfile(root, 'latest.txt');
    if ~isfile(latest)
        error('build_thesis_figures:noResults', ...
              ['No analysis results yet.\nRun  run_study_analysis  first, then ' ...
               'this function again.\n(expected %s)'], latest);
    end
    results_dir = string(strtrim(fileread(latest)));
end

if strlength(opts.out_dir) == 0
    repo = fileparts(fileparts(fileparts(here)));   % .../scripts/analysis/matlab -> repo
    out_dir = fullfile(repo, 'thesis', 'figures', 'results');
else
    out_dir = char(opts.out_dir);
end
if ~isfolder(out_dir), mkdir(out_dir); end
fprintf('[build_thesis_figures] results: %s\n  -> %s\n', results_dir, out_dir);

L = load(fullfile(char(results_dir), 'results.mat'), 'S', 'items', 'trial');
S = L.S; items = L.items; trial = L.trial;

% metric, file stem, y-axis symbol and unit as the caption names it
M = {"duration_s","T","$T$ [s]"; ...
     "ee_path_len_m","L","$L$ [m]"; ...
     "ee_sparc","Phi","$\Phi$"; ...
     "ee_speed_mean_mps","vbar","$\bar{v}$ [m/s]"; ...
     "slack_mean","deltabar","$\bar{\delta}$"; ...
     "safety_min_dist_m","hmin","$h_{\min}$ [m]"; ...
     "cbf_active_s","Tlambda","$T_{\lambda}$ [s]"; ...
     "belief_mean_prob","bbar","$\bar{b}$"; ...
     "agreement_mean_cos","rho","$\rho$"; ...
     "alpha_autonomy_frac","phialpha","$\phi_{\alpha}$"; ...
     "intervention_mean_mps","Dmean","$\bar{\Delta}$ [m/s]"; ...
     "intervention_peak_mps","Dpeak","$\Delta_{\max}$ [m/s]"};
for i = 1:size(M, 1)
    fig_thesis_panel(S, items, M{i,1}, fullfile(out_dir, "res_" + M{i,2} + ".pdf"), ...
                     'ylab', M{i,3});
end
fprintf('  12 condition panels\n');

% Force and re-indexing are not comparable across modes, so each is drawn inside
% its own mode and carries that mode's colour ramp.
fig_thesis_panel(S, items, "force_mean_N", fullfile(out_dir, 'res_fbarC.pdf'), ...
                 'source', "Q2_withinC", 'cell_colors', "C", 'ylab', "$\bar{f}$ [N]");
fig_thesis_panel(S, items, "force_mean_N", fullfile(out_dir, 'res_fbarJ.pdf'), ...
                 'source', "Q2_withinJ", 'cell_colors', "J", 'ylab', "$\bar{f}$ [N]");
fig_thesis_panel(S, items, "clutch_presses", fullfile(out_dir, 'res_Nc.pdf'), ...
                 'source', "Q2_withinC", 'cell_colors', "C", 'ylab', "$N_c$");
fprintf('  force + re-indexing panels\n');

fig_thesis_learning(S, items, trial, out_dir);
fprintf('  practice panel\nDONE -> %s\n', out_dir);
end
