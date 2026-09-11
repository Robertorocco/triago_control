function trial = load_trial_row(row, export_dir)
%LOAD_TRIAL_ROW Load a trial's full struct from one row of LOAD_MANIFEST's table.
%   TRIAL = LOAD_TRIAL_ROW(ROW) takes a single-row table (e.g. T(k,:) from
%   LOAD_MANIFEST) and loads its mat_file via LOAD_TRIAL. Saves re-typing
%   participant/world/cell when iterating over a filtered manifest.
%
%   Example:
%       T = load_manifest();
%       hard = T(T.right_safety_nearmiss_frac > 0.1, :);
%       for k = 1:height(hard)
%           trial = load_trial_row(hard(k, :));
%           fprintf('%s %s: peak lambda_cbf = %.2f\n', hard.participant{k}, ...
%                   hard.cell{k}, trial.metrics_right.cbf_lambda_peak);
%       end

if nargin < 2
    export_dir = [];
end
if height(row) ~= 1
    error('load_trial_row:badInput', 'pass exactly one row, e.g. T(k, :)');
end
trial = load_trial(row.participant{1}, row.world{1}, row.cell{1}, export_dir);
end
