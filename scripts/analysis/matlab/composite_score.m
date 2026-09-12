function [trial, fam] = composite_score(trial, spec)
%COMPOSITE_SCORE Pooled, oriented z-scores per metric, family scores, composite.
%   [TRIAL, FAM] = COMPOSITE_SCORE(TRIAL, SPEC) adds to TRIAL:
%     z_<metric>       metric standardised over ALL trials of all complete
%                      participants (pooled mean/SD), multiplied by the metric
%                      direction so that higher always means better
%     fam_<family>     mean of the oriented z-scores of that family's metrics
%     composite        mean of the family scores
%   and returns FAM, a table listing per family which metrics entered the
%   score and which were dropped (constant over the whole data set, no signal).
%
%   Only metrics with cell_scope "all" and mode_comparable == true enter a
%   family score, so a family score is always comparable across every cell;
%   the one exception is assistance_quality, whose metrics are defined only in
%   blending cells, so that family score is NaN elsewhere. The composite is a
%   descriptive index for overview figures: every rigorous claim in the report
%   rests on the per-metric and per-family tests, not on the composite.

families = study_families();
fam = table();
fam.key = families.key;
fam.members = strings(height(families), 1);
fam.dropped = strings(height(families), 1);
fam.n_metrics = zeros(height(families), 1);

for f = 1:height(families)
    key = families.key(f);
    rows = spec.family == key & spec.dir ~= 0 & spec.mode_comparable ...
         & (spec.cell_scope == "all" | (key == "assistance_quality" & spec.cell_scope == "blend_only"));
    names = spec.name(rows);
    dirs = spec.dir(rows);
    used = strings(0, 1); dropped = strings(0, 1);
    Zf = [];
    for k = 1:numel(names)
        x = trial.(names(k));
        z = zscore_safe(x);
        if isempty(z)                       % constant metric: carries no signal
            dropped(end+1, 1) = names(k); %#ok<AGROW>
            continue
        end
        z = dirs(k) * z;
        trial.("z_" + names(k)) = z;
        Zf = [Zf, z]; %#ok<AGROW>
        used(end+1, 1) = names(k); %#ok<AGROW>
    end
    if isempty(Zf)
        trial.("fam_" + key) = nan(height(trial), 1);
    else
        trial.("fam_" + key) = mean(Zf, 2, 'omitnan');
    end
    fam.members(f) = strjoin(used, ", ");
    fam.dropped(f) = strjoin(dropped, ", ");
    fam.n_metrics(f) = numel(used);
end

famCols = "fam_" + families.key(fam.n_metrics > 0);
trial.composite = mean(trial{:, famCols}, 2, 'omitnan');
end

function z = zscore_safe(x)
% Pooled z-score; returns [] when the metric never varies (SD ~ 0 or all NaN).
m = mean(x, 'omitnan'); s = std(x, 'omitnan');
if ~isfinite(s) || s < 1e-12
    z = []; return
end
z = (x - m) ./ s;
end
