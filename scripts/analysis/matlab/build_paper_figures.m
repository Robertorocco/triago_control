function build_paper_figures(results_dir, opts)
%BUILD_PAPER_FIGURES Turn study_paper_figures.m into an executed Live Script + PDF/HTML.
%   BUILD_PAPER_FIGURES() converts study_paper_figures.m (next to this file)
%   into study_paper_figures.mlx, runs it against the most recent results
%   folder, and exports the PDF and HTML into that folder. Open the PDF (or the
%   .mlx) and read it -- nothing else has to be run.
%   BUILD_PAPER_FIGURES(RESULTS_DIR) uses a specific results folder.
%   Options: pdf (true), html (true), export_dir ('' = auto).
%
%   The .m source stays the editable master: edit the text or the METRICS list
%   there, re-run this function. The .mlx is regenerated every time, which is
%   why it is not kept in git.

arguments
    results_dir (1,1) string = ""
    opts.pdf (1,1) logical = true
    opts.html (1,1) logical = true
    opts.export_dir (1,1) string = ""
end

here = fileparts(mfilename('fullpath'));
src = fullfile(here, 'study_paper_figures.m');
mlx = fullfile(here, 'study_paper_figures.mlx');

if strlength(results_dir) == 0
    root = fullfile(study_export_dir(char(opts.export_dir)), 'analysis_results');
    latest = fullfile(root, 'latest.txt');
    if ~isfile(latest)
        error('build_paper_figures:noResults', ...
              ['No analysis results yet.\nRun  run_study_analysis  first, then ' ...
               'this function again.\n(expected %s)'], latest);
    end
    results_dir = string(strtrim(fileread(latest)));
end
fprintf('[build_paper_figures] results: %s\n', results_dir);

% Bind the executed copy to this results folder; the .m master keeps auto-detect.
txt = fileread(src);
txt = strrep(txt, "RESULTS_DIR = '';", "RESULTS_DIR = '" + results_dir + "';");
tmp = fullfile(tempdir, 'study_paper_figures.m');
fid = fopen(tmp, 'w'); fwrite(fid, txt); fclose(fid);

fprintf('[build_paper_figures] converting to Live Script ...\n');
if isfile(mlx), delete(mlx); end
matlab.internal.liveeditor.openAndSave(char(tmp), char(mlx));
delete(tmp);

% Figures must stay visible: the Live Editor only embeds figures it controls.
fprintf('[build_paper_figures] executing (drawing every figure) ...\n');
matlab.internal.liveeditor.executeAndSave(char(mlx));

if opts.html
    out = fullfile(results_dir, 'study_paper_figures.html');
    export(mlx, out, 'Format', 'html', 'HideCode', true);
    fprintf('[build_paper_figures] html -> %s\n', out);
end
if opts.pdf
    out = fullfile(results_dir, 'study_paper_figures.pdf');
    export(mlx, out, 'Format', 'pdf', 'HideCode', true);
    fprintf('[build_paper_figures] pdf  -> %s   <-- open this one\n', out);
end
copyfile(mlx, fullfile(results_dir, 'study_paper_figures.mlx'));
end
