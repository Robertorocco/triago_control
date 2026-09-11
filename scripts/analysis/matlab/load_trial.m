function trial = load_trial(participant, world, cell_code, export_dir)
%LOAD_TRIAL Load one trial's full struct exported by export_to_matlab.py.
%   TRIAL = LOAD_TRIAL(PARTICIPANT, WORLD, CELL_CODE) loads
%   <export_dir>/mat/<PARTICIPANT>_<WORLD>_<CELL_CODE>.mat, e.g.
%   LOAD_TRIAL('P07', 'rack', 'CFB'). EXPORT_DIR defaults to
%   ~/exchange/triago_study_data/matlab_export.
%
%   TRIAL.meta            -- metadata.json (provenance, cfg_snapshot nested struct)
%   TRIAL.metrics_right/_left -- study_metrics.compute_metrics() per arm
%   TRIAL.series.<topic>  -- t (s, column vector) + every column
%                            study_metrics.load_bag() decoded for that topic
%                            (sanitized field names, e.g. /qp_debug/qdot_cmd
%                            -> series.qp_debug_qdot_cmd.d0..d13)
%
%   Example:
%       trial = load_trial('P07', 'rack', 'CFB');
%       ee = trial.series.qp_debug_ee_real;
%       plot(ee.t, vecnorm([ee.d3 ee.d4 ee.d5], 2, 2));   % right EE speed
%       trial.metrics_right.ee_path_efficiency

if nargin < 4, export_dir = ''; end
export_dir = study_export_dir(export_dir);
mat_path = fullfile(export_dir, 'mat', sprintf('%s_%s_%s.mat', participant, world, cell_code));
if ~isfile(mat_path)
    if ~isfolder(fullfile(export_dir, 'mat'))
        error('load_trial:noMatFolder', ...
             ['No mat/ folder in %s.\n\nPer-trial files are the heavy part of the export ' ...
              '(~29 MB each).\nThe manifest-based analysis does not need them, but raw ' ...
              'time series do --\ncopy the mat/ folder over, or analyse on the machine that ran the export.'], ...
              export_dir);
    end
    error('load_trial:notFound', 'no export for %s/%s_%s at %s', ...
        participant, world, cell_code, mat_path);
end
trial = load(mat_path);
end
