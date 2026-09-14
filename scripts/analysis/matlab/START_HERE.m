%% START_HERE  Open the study results in this MATLAB session.
%   Run this file (press Run, or type START_HERE) and it will:
%     * put this folder on the path,
%     * load the most recent analysis results into the Workspace panel,
%     * open the paper-style figures as real windows on screen,
%     * open the readable PDF report if one has been built.
%
%   Nothing else has to be run first, and nothing has to be edited. If no
%   analysis has been run yet it says so and tells you the one command to type.

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(here);
fprintf('folder on path: %s\n', here);

%% ---- locate the data and the latest analysis -------------------------------
try
    export_dir = study_export_dir('');
catch err
    fprintf(2, '\nNo manifest.mat found.\n%s\n', err.message);
    return
end
fprintf('data          : %s\n', export_dir);

latest = fullfile(export_dir, 'analysis_results', 'latest.txt');
if ~isfile(latest)
    fprintf(2, ['\nNo analysis has been run yet.\n' ...
                'Type this in the Command Window and press Enter:\n\n' ...
                '    run_study_analysis\n\n' ...
                'then run START_HERE again.\n']);
    return
end
results_dir = strtrim(fileread(latest));
fprintf('results       : %s\n', results_dir);

%% ---- load everything into this workspace -----------------------------------
% These land in the Workspace panel, so they can be clicked and inspected:
%   trial  one row per trial   R  every test result   S  the full test structs
%   items  metric catalogue    meta  who is included  settings  alpha, n_boot
load(fullfile(results_dir, 'results.mat'), ...
     'trial', 'spec', 'families', 'fam', 'meta', 'S', 'R', 'items', 'settings');

fprintf('\nLoaded into the workspace:\n');
fprintf('  trial    %d trials x %d columns   (double-click to browse)\n', height(trial), width(trial));
fprintf('  R        %d test results\n', height(R));
fprintf('  meta     %d complete participants\n', meta.n_complete);

%% ---- draw the figures on screen --------------------------------------------
METRICS = ["duration_s", "ee_path_len_m", "safety_min_dist_m", ...
           "cbf_active_frac", "ee_sparc", "qdot_cmd_rms"];
SUMMARY_METRICS = ["composite", "fam_time_effectiveness", "fam_safety", "fam_motion_quality"];

fprintf('\nDrawing figures ...\n');
fig_paper_panels(S, items, METRICS, 'cols', 2, 'name', 'Headline results');
fig_paper_panels(S, items, SUMMARY_METRICS, 'cols', 2, 'name', 'Overall scores');

%% ---- offer the written report ----------------------------------------------
pdf = fullfile(results_dir, 'study_paper_figures.pdf');
if isfile(pdf)
    fprintf('Opening the short report: %s\n', pdf);
    try, open(pdf); catch, fprintf('  (open it by hand: %s)\n', pdf); end
else
    fprintf(['No PDF report yet. To build the readable one, type:\n' ...
             '    build_paper_figures\n']);
end

fprintf(['\nWhat to do next\n' ...
         '  build_paper_figures   short visual report  (PDF + Live Script)\n' ...
         '  build_report          full statistical report\n' ...
         '  analyze_participant   one participant at a time\n' ...
         '  run_study_analysis    re-run the statistics after new data\n']);
