function [E, P] = paper_stats_tables(S, spec, metrics, alpha)
%PAPER_STATS_TABLES Effects and significant-pair tables behind the paper panels.
%   [E, P] = PAPER_STATS_TABLES(S, SPEC, METRICS, ALPHA) collects, for every
%   metric of METRICS that has a six-cell result:
%     E  one row per (metric, factor): the two-way repeated-measures ANOVA
%        main effects of control Mode and Assistance and their interaction,
%        with Greenhouse-Geisser corrected p and partial eta squared;
%     P  one row per pair of cells that survives the Holm correction, with
%        the paired effect size.
%   Reporting convention of a results table in a paper: every factor is listed
%   so the reader sees what was tested, but only the significant pairs are
%   spelled out.

if nargin < 4 || isempty(alpha), alpha = 0.05; end
metrics = metrics(arrayfun(@(m) isfield(S.Q3, m), metrics));

erows = {}; prows = {};
for i = 1:numel(metrics)
    nm = metrics(i);
    R = S.Q3.(nm);
    srow = spec(spec.name == nm, :);
    if isempty(srow), label = nm; else, label = srow.label(1); end

    if ~isempty(R.effects)
        for j = 1:height(R.effects)
            e = R.effects(j, :);
            erows(end + 1, :) = {shortLabel(label), prettyFactor(e.effect), ...
                sprintf("F(%g,%g)=%.1f", e.df1, e.df2, e.F), fmt_p(e.p_gg), ...
                round(e.eta2p, 3), verdict(e.p_gg, alpha)}; %#ok<AGROW>
        end
    end

    if ~isempty(R.pairs)
        sig = R.pairs(R.pairs.significant, :);
        for j = 1:height(sig)
            prows(end + 1, :) = {shortLabel(label), sig.a(j) + " vs " + sig.b(j), ...
                round(sig.mean_diff(j), 3), fmt_p(sig.p_holm(j)), ...
                round(sig.r_rb(j), 2)}; %#ok<AGROW>
        end
    end
end

E = table();
if ~isempty(erows)
    E = cell2table(erows, 'VariableNames', ...
        {'Metric', 'Factor', 'F_test', 'p_GG', 'partial_eta2', 'Real'});
end
P = table();
if ~isempty(prows)
    P = cell2table(prows, 'VariableNames', ...
        {'Metric', 'Pair', 'Mean_difference', 'p_Holm', 'rank_biserial'});
end
end

function s = shortLabel(label)
% Table columns are narrow; the full name is in the figure next to it.
s = string(label);
if strlength(s) > 18, s = extractBefore(s, 17) + ".."; end
end

function s = prettyFactor(f)
switch string(f)
    case "Mode",        s = "Control mode";
    case "Assist",      s = "Assistance";
    case "Mode:Assist", s = "Interaction";
    otherwise,          s = string(f);
end
end

function s = verdict(p, alpha)
if isnan(p),        s = "?";
elseif p < alpha,   s = "YES";
else,               s = "no";
end
end
