function fig = fig_metric_dashboard(trial, item, S)
%FIG_METRIC_DASHBOARD One figure that answers Q1, Q2, Q3 and Q6 for a single metric.
%   FIG = FIG_METRIC_DASHBOARD(TRIAL, ITEM, S) draws four panels:
%     top-left     Clutch vs Joystick, one grey line per participant (Q1)
%     top-right    F vs B vs FB, one grey line per participant (Q2)
%     bottom-left  2 x 3 grid of the six cells, mean +/- 95% CI (Q3)
%     bottom-right performance along the six experiment slots (Q6)
%   ITEM is one row of the analysis item list (name, label, unit, dir,
%   cell_scope, mode_comparable); S holds the test results computed by
%   RUN_STUDY_ANALYSIS. Every title states the test outcome so the figure can
%   be read on its own. Metrics that are not comparable across control modes
%   (haptic force, clutch use) get within-mode panels instead.

name = item.name;
ttl = sprintf("%s [%s]  -  %s", item.label, item.unit, studyplot.dirtext(item.dir));
fig = studyplot.newfig("Metric: " + item.label, 1250, 820);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, ttl, 'FontWeight', 'bold', 'FontSize', 13);
subtitle(tl, sprintf("%d complete participants; grey = one participant, coloured = mean with 95%% CI", ...
    numel(unique(trial.participant))));

% ---------------- Q1: mode ----------------
ax = nexttile(tl);
if isfield(S.Q1, name)
    R = S.Q1.(name);
    M = participant_means(trial, name, "mode", ["C" "J"]);
    studyplot.paired(ax, M, ["C" "J"], 'ylabel', item.unit);
    title(ax, sprintf("Q1  Clutch vs Joystick:  %s %s (%s), %s = %.2f (%s)", ...
        fmt_p(R.p), studyplot.stars(R.p), studyplot.testname(R.recommended), R.effect_name, R.effect, effect_band(bandkind(R.effect_name), R.effect)));
    subtitle(ax, verdict_line(R.better, R.significant, sprintf("Clutch - Joystick = %.3g", R.mean_diff)));
else
    M = participant_means(trial, name, "mode", ["C" "J"]);
    studyplot.paired(ax, M, ["C" "J"], 'ylabel', item.unit);
    title(ax, "Q1  Clutch vs Joystick: NOT compared");
    subtitle(ax, "this quantity has a different physical origin in the two modes (context only)");
end

% ---------------- Q2: assistance ----------------
ax = nexttile(tl);
if isfield(S.Q2, name)
    R = S.Q2.(name);
    if R.test_family == "paired2"
        M = participant_means(trial, name, "assist", ["B" "FB"]);
        studyplot.paired(ax, M, ["B" "FB"], 'ylabel', item.unit);
        title(ax, sprintf("Q2  B vs FB (blending cells only):  %s %s (%s), %s = %.2f", ...
            fmt_p(R.p), studyplot.stars(R.p), studyplot.testname(R.recommended), R.effect_name, R.effect));
        subtitle(ax, verdict_line(R.better, R.significant, sprintf("B - FB = %.3g", R.mean_diff)));
    else
        M = participant_means(trial, name, "assist", ["F" "B" "FB"]);
        studyplot.paired(ax, M, ["F" "B" "FB"], 'ylabel', item.unit);
        add_pair_brackets(ax, M, R);
        title(ax, sprintf("Q2  Assistance F / B / FB:  %s %s (%s), %s = %.2f", ...
            fmt_p(R.p), studyplot.stars(R.p), studyplot.testname(R.recommended), R.effect_name, R.effect));
        subtitle(ax, verdict_line(R.best, R.significant, "ranking: " + strjoin(R.ranking, " > ")));
    end
elseif isfield(S.Q2_withinC, name)
    R = S.Q2_withinC.(name);
    M = participant_means(trial, name, "assist", ["F" "B" "FB"], trial.mode == "C");
    studyplot.paired(ax, M, ["F" "B" "FB"], 'ylabel', item.unit);
    add_pair_brackets(ax, M, R);
    title(ax, sprintf("Q2  Assistance within CLUTCH only:  %s %s, %s = %.2f", ...
        fmt_p(R.p), studyplot.stars(R.p), R.effect_name, R.effect));
    subtitle(ax, verdict_line(R.best, R.significant, "ranking: " + strjoin(R.ranking, " > ")));
else
    axis(ax, 'off'); text(ax, 0.5, 0.5, "not defined", 'HorizontalAlignment', 'center');
end

% ---------------- Q3: cells ----------------
ax = nexttile(tl);
if isfield(S.Q3, name)
    R = S.Q3.(name);
    cells = R.levels; modes = unique(R.modes, 'stable'); assists = unique(R.assists, 'stable');
    means = nan(numel(modes), numel(assists)); cis = means;
    M = participant_means(trial, name, "cell", cells);
    for j = 1:numel(cells)
        r = find(modes == R.modes(j)); c = find(assists == R.assists(j));
        v = M(:, j); v = v(~isnan(v));
        means(r, c) = mean(v);
        if numel(v) > 1, cis(r, c) = tinv(0.975, numel(v) - 1) * std(v) / sqrt(numel(v)); end
    end
    studyplot.heat23(ax, means, cis, modes, assists, 'dir', item.dir, 'best', R.best);
    title(ax, sprintf("Q3  Mode x Assistance cells:  Friedman %s %s, Kendall W = %.2f", ...
        fmt_p(R.p), studyplot.stars(R.p), R.W));
    sub = "ranking: " + strjoin(R.ranking, " > ");
    if ~isempty(R.effects)
        ia = R.effects.effect == "Mode:Assist";
        if any(ia), sub = sub + sprintf("   |   interaction %s", fmt_p(R.effects.p_gg(ia))); end
    end
    subtitle(ax, sub);
elseif isfield(S.Q2_withinJ, name)
    R = S.Q2_withinJ.(name);
    M = participant_means(trial, name, "assist", ["F" "B" "FB"], trial.mode == "J");
    studyplot.paired(ax, M, ["F" "B" "FB"], 'ylabel', item.unit);
    add_pair_brackets(ax, M, R);
    title(ax, sprintf("Q2  Assistance within JOYSTICK only:  %s %s, %s = %.2f", ...
        fmt_p(R.p), studyplot.stars(R.p), R.effect_name, R.effect));
    subtitle(ax, verdict_line(R.best, R.significant, "ranking: " + strjoin(R.ranking, " > ")));
else
    axis(ax, 'off'); text(ax, 0.5, 0.5, "Q3 not applicable (metric defined in one mode only)", 'HorizontalAlignment', 'center');
end

% ---------------- Q6: learning ----------------
ax = nexttile(tl);
if isfield(S.Q6, name)
    R = S.Q6.(name);
    studyplot.learning(ax, R.slot_matrix, 'ylabel', item.unit);
    title(ax, sprintf("Q6  Trend over the session:  slope %.3g/slot, Wilcoxon %s %s", ...
        R.slope_mean, fmt_p(R.p_slope_wilcoxon), studyplot.stars(R.p_slope_wilcoxon)));
    if isnan(R.p_order_ranksum)
        subtitle(ax, "slots 1-3 = first control mode met, 4-6 = second");
    else
        subtitle(ax, sprintf("slots 1-3 = first mode, 4-6 = second; mode-order bias check: rank-sum %s", fmt_p(R.p_order_ranksum)));
    end
else
    axis(ax, 'off'); text(ax, 0.5, 0.5, "Q6 not applicable", 'HorizontalAlignment', 'center');
end
end

% ------------------------------------------------------------------ helpers
function add_pair_brackets(ax, M, R)
if ~isfield(R, 'pairs') || isempty(R.pairs), return, end
sig = R.pairs(R.pairs.significant, :);
if isempty(sig), return, end
yl = ylim(ax); span = yl(2) - yl(1);
lv = R.levels;
y = yl(2) + 0.05 * span;
for i = 1:height(sig)
    x1 = find(lv == sig.a(i)); x2 = find(lv == sig.b(i));
    studyplot.sigbracket(ax, x1, x2, y, sprintf("Holm p = %.3f", sig.p_holm(i)));
    y = y + 0.09 * span;
end
ylim(ax, [yl(1), y + 0.02 * span]);
end

function s = verdict_line(best, significant, extra)
if strlength(best) > 0 && significant
    s = "-> " + studyplot.levelname(best) + " is better";
elseif strlength(best) > 0
    s = "trend favours " + studyplot.levelname(best) + " but not significant";
else
    s = "no direction defined";
end
if nargin > 2 && strlength(extra) > 0, s = s + "   |   " + extra; end
end

function k = bandkind(effect_name)
switch effect_name
    case "Cohen dz", k = "dz";
    case "rank-biserial r", k = "r";
    case "partial eta2", k = "eta2p";
    case "Kendall W", k = "W";
    otherwise, k = "dz";
end
end
