function T = load_manifest(export_dir)
%LOAD_MANIFEST Load the cross-trial manifest written by export_to_matlab.py.
%   T = LOAD_MANIFEST() finds manifest.mat next to these scripts, in the current
%   folder, or in the default export location (see STUDY_EXPORT_DIR), so a copied
%   bundle works on any machine.
%   T = LOAD_MANIFEST(EXPORT_DIR) reads <export_dir>/manifest.mat into a table,
%   one row per trial that passed check_study_data.py's checks -- meta fields
%   (participant, world, cell, condition, control_mode, assist_feedback,
%   assist_blending, success, notes, duration_s) plus both arms' whole
%   study_metrics.compute_metrics() output, flattened with right_/left_
%   prefixes. EXPORT_DIR defaults to ~/exchange/triago_study_data/matlab_export.
%
%   Text fields (participant, world, cell, condition, success, notes,
%   right_belief_winner, ...) come back as a cellstr column -- use strcmp/
%   contains/ismember on them. Missing text is '' and missing numeric is NaN,
%   already MATLAB's own "missing" sentinels, so no further cleanup is needed.
%
%   Example:
%       T = load_manifest();
%       T = T(strcmp(T.success, 'yes'), :);              % drop failed trials
%       grpstats(T, {'control_mode', 'assist_feedback', 'assist_blending'}, ...
%               'mean', 'DataVars', 'right_ee_path_efficiency')

if nargin < 1, export_dir = ''; end
export_dir = study_export_dir(export_dir);
manifest_path = fullfile(export_dir, 'manifest.mat');
if ~isfile(manifest_path)
    error('load_manifest:notFound', 'manifest.mat not found at %s -- run export_to_matlab.py first', ...
        manifest_path);
end
S = load(manifest_path);
T = struct2table(S.trials);
end
