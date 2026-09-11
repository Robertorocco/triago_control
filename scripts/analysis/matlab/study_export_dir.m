function d = study_export_dir(override)
%STUDY_EXPORT_DIR  Locate the exported study data, wherever it was copied to.
%   D = STUDY_EXPORT_DIR() searches, in order:
%     1. the folder these .m files live in  (a copied, self-contained bundle)
%     2. the current folder
%     3. ~/exchange/triago_study_data/matlab_export  (the machine that exported)
%   D = STUDY_EXPORT_DIR(OVERRIDE) uses OVERRIDE when it is non-empty.
%
%   A folder counts as the export only if it holds manifest.mat, so copying
%   manifest.mat next to these scripts is enough to analyse on any machine --
%   no path editing, Windows or Linux.

if nargin >= 1 && ~isempty(override)
    d = char(override);
    return
end

here = fileparts(mfilename('fullpath'));
candidates = {here, pwd, fullfile(homeDir(), 'exchange', 'triago_study_data', 'matlab_export')};
for k = 1:numel(candidates)
    if isfile(fullfile(candidates{k}, 'manifest.mat'))
        d = candidates{k};
        return
    end
end
error('study_export_dir:notFound', ...
      ['No manifest.mat found. Looked in:\n  %s\n\n' ...
       'Copy manifest.mat next to these .m files (that is all the per-participant\n' ...
       'and comparison analysis needs), or pass the folder explicitly.'], ...
      strjoin(candidates, sprintf('\n  ')));
end

function h = homeDir()
% HOME on Linux/Mac, USERPROFILE on Windows, current folder as a last resort.
    h = getenv('HOME');
    if isempty(h), h = getenv('USERPROFILE'); end
    if isempty(h), h = pwd; end
end
