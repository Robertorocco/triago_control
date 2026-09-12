function build_report(results_dir, opts)
%BUILD_REPORT Turn study_report.m into an executed Live Script, HTML and PDF.
%   BUILD_REPORT() converts study_report.m (next to this file) into
%   study_report.mlx, runs it against the most recent results folder
%   (analysis_results/latest.txt), saves the Live Script with its outputs and
%   exports study_report.html and study_report.pdf into that results folder.
%   BUILD_REPORT(RESULTS_DIR) uses a specific results folder.
%   Options: pdf (true), html (true), export_dir ('' = auto).
%
%   The .m source stays the editable master: edit the text there, re-run
%   this function. The .mlx is regenerated every time.

arguments
    results_dir (1,1) string = ""
    opts.pdf (1,1) logical = true
    opts.html (1,1) logical = true
    opts.export_dir (1,1) string = ""
end

here = fileparts(mfilename('fullpath'));
src = fullfile(here, 'study_report.m');
mlx = fullfile(here, 'study_report.mlx');

if strlength(results_dir) == 0
    root = fullfile(study_export_dir(char(opts.export_dir)), 'analysis_results');
    results_dir = string(strtrim(fileread(fullfile(root, 'latest.txt'))));
end
fprintf('[build_report] results: %s\n', results_dir);

% The report's own settings block is patched so that the executed copy is
% bound to this results folder (the .m master keeps the auto-detect default).
txt = fileread(src);
txt = strrep(txt, "RESULTS_DIR = '';", "RESULTS_DIR = '" + results_dir + "';");
tmp = fullfile(tempdir, 'study_report.m');
fid = fopen(tmp, 'w'); fwrite(fid, txt); fclose(fid);

fprintf('[build_report] converting to Live Script ...\n');
if isfile(mlx), delete(mlx); end
matlab.internal.liveeditor.openAndSave(char(tmp), char(mlx));
delete(tmp);

% Figures must stay visible: the Live Editor only captures visible figures
% into the script output (they are embedded in the .mlx, not opened as windows).
fprintf('[build_report] executing the Live Script (this draws every figure) ...\n');
matlab.internal.liveeditor.executeAndSave(char(mlx));

if opts.html
    out = fullfile(results_dir, 'study_report.html');
    export(mlx, out, 'Format', 'html');
    fprintf('[build_report] html -> %s\n', out);
end
if opts.pdf
    out = fullfile(results_dir, 'study_report.pdf');
    export(mlx, out, 'Format', 'pdf');
    fprintf('[build_report] pdf  -> %s\n', out);
end
copyfile(mlx, fullfile(results_dir, 'study_report.mlx'));
fprintf('[build_report] live script -> %s  (open it in MATLAB to read the report)\n', mlx);
end
