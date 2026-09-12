function fig = fig_metric_dashboard(trial, item, S, opts)
%FIG_METRIC_DASHBOARD One figure that answers Q1, Q2, Q3 and Q6 for a single metric.
%   FIG = FIG_METRIC_DASHBOARD(TRIAL, ITEM, S) draws four panels:
%     Clutch vs Joystick, one grey line per participant (Q1)
%     F vs B vs FB, one grey line per participant (Q2)
%     2 x 3 grid of the six cells, mean +/- 95% CI (Q3)
%     performance along the six experiment slots (Q6)
%   ITEM is one row of the analysis item list (name, label, unit, dir,
%   cell_scope, mode_comparable); S holds the test results computed by
%   RUN_STUDY_ANALYSIS. Every title states the test outcome so the figure can
%   be read on its own. Metrics that are not comparable across control modes
%   (haptic force, clutch use) get within-mode panels instead.
%   FIG_METRIC_DASHBOARD(..., 'compact', true) stacks the four panels in one
%   column with small fonts, the layout used inside the Live Script report.

arguments
    trial table
    item struct
    S struct
    opts.compact (1,1) logical = false
end
c = opts.compact;
name = item.name;
n = numel(unique(trial.participant));
ttl = sprintf("%s [%s]  -  %s", item.label, item.unit, studyplot.dirtext(item.dir));
if c
    fig = studyplot.newfig("Metric: " + item.label, 480, 660, true);
    tl = tiledlayout(fig, 4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    W = 72;
else
    fig = studyplot.newfig("Metric: " + item.label, 1250, 820);
    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    W = 200;
end
title(tl, ttl, 'FontWeight', 'bold', 'FontSize', 13 - 4 * c);
subtitle(tl, sprintf("%d participants; grey = one participant, coloured = mean with 95%% CI", n), 'FontSize', 10 - 3 * c);

% ---------------- Q1: mode ----------------
ax = nexttile(tl);
M = participant_means(trial, name, "mode", ["C" "J"]);
studyplot.paired(ax, M, ["C" "J"], 'ylabel', item.unit, 'horizontal', c);
if isfield(S.Q1, name)
    R = S.Q1.(name);
    set_titles(ax, sprintf("Q1  Clutch vs Joystick:  %s %s (%s), %s = %.2f (%s)", ...
        fmt_p(R.p), studyplot.stars(R.p), studyplot.testname(R.recommended), R.effect_name, R.effect, ...
        effect_band(bandkind(R.effect_name), R.effect)), ...
        verdict_line(R.better, R.significant, sprintf("Clutch - Joystick = %.3g", R.mean_diff)), W);
else
    set_titles(ax, "Q1  Clutch vs Joystick: NOT compared", ...
        "different physical origin in the two modes (context only)", W);
end

% ---------------- Q2: assistance ----------------
ax = nexttile(tl);
if isfield(S.Q2, name)
    R = S.Q2.(name);
    if R.test_family == "paired2"
        M = participant_means(trial, name, "assist", ["B" "FB"]);
        studyplot.paired(ax, M, ["B" "FB"], 'ylabel', item.unit, 'horizontal', c);
        set_titles(ax, sprintf("Q2  B vs FB (blending cells only):  %s %s (%s), %s = %.2f", ...
            fmt_p(R.p), studyplot.stars(R.p), studyplot.testname(R.recommended), R.effect_name, R.effect), ...
            verdict_line(R.better, R.significant, sprintf("B - FB = %.3g", R.mean_diff)), W);
    else
        M = participant_means(trial, name, "assist", ["F" "B" "FB"]);
        studyplot.paired(ax, M, ["F" "B" "FB"], 'ylabel', item.unit, 'horizontal', c);
        add_pair_brackets(ax, R, c);
        set_titles(ax, sprintf("Q2  Assistance F / B / FB:  %s %s (%s), %s = %.2f", ...
            fmt_p(R.p), studyplot.stars(R.p), studyplot.testname(R.recommended), R.effect_name, R.effect), ...
            verdict_line(R.best, R.significant, "ranking: " + strjoin(R.ranking, " > ") + sigpairs(R, c)), W);
    end
elseif isfield(S.Q2_withinC, name)
    R = S.Q2_withinC.(name);
    M = participant_means(trial, name, "assist", ["F" "B" "FB"], trial.mode == "C");
    studyplot.paired(ax, M, ["F" "B" "FB"], 'ylabel', item.unit, 'horizontal', c);
    add_pair_brackets(ax, R, c);
    set_titles(ax, sprintf("Q2  Assistance within CLUTCH only:  %s %s, %s = %.2f", ...
        fmt_p(R.p), studyplot.stars(R.p), R.effect_name, R.effect), ...
        verdict_line(R.best, R.significant, "ranking: " + strjoin(R.ranking, " > ") + sigpairs(R, c)), W);
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
        r = find(modes == R.modes(j)); k = find(assists == R.assists(j));
        v = M(:, j); v = v(~isnan(v));
        means(r, k) = mean(v);
        if numel(v) > 1, cis(r, k) = tinv(0.975, numel(v) - 1) * std(v) / sqrt(numel(v)); end
    end
    if c, mark = " *"; else, mark = "  (best)"; end
    studyplot.heat23(ax, means, cis, modes, assists, 'dir', item.dir, 'best', R.best, 'bestmark', mark);
    sub = "ranking: " + strjoin(R.ranking, " > ");
    if c, sub = sub + " (* best)"; end
    if ~isempty(R.effects)
        ia = R.effects.effect == "Mode:Assist";
        if any(ia), sub = sub + sprintf("  |  interaction %s", fmt_p(R.effects.p_gg(ia))); end
    end
    set_titles(ax, sprintf("Q3  Mode x Assistance cells:  Friedman %s %s, Kendall W = %.2f", ...
        fmt_p(R.p), studyplot.stars(R.p), R.W), sub, W);
elseif isfield(S.Q2_withinJ, name)
    R = S.Q2_withinJ.(name);
    M = participant_means(trial, name, "assist", ["F" "B" "FB"], trial.mode == "J");
    studyplot.paired(ax, M, ["F" "B" "FB"], 'ylabel', item.unit, 'horizontal', c);
    add_pair_brackets(ax, R, c);
    set_titles(ax, sprintf("Q2  Assistance within JOYSTICK only:  %s %s, %s = %.2f", ...
        fmt_p(R.p), studyplot.stars(R.p), R.effect_name, R.effect), ...
        verdict_line(R.best, R.significant, "ranking: " + strjoin(R.ranking, " > ") + sigpairs(R, c)), W);
else
    axis(ax, 'off'); text(ax, 0.5, 0.5, "Q3 not applicable (metric defined in one mode only)", 'HorizontalAlignment', 'center');
end

% ---------------- Q6: learning ----------------
ax = nexttile(tl);
if isfield(S.Q6, name)
    R = S.Q6.(name);
    studyplot.learning(ax, R.slot_matrix, 'ylabel', item.unit);
    if isnan(R.p_order_ranksum)
        sub = "slots 1-3 = first control mode met, 4-6 = second";
    else
        sub = sprintf("slots 1-3 = first mode, 4-6 = second; mode-order bias check: rank-sum %s", fmt_p(R.p_order_ranksum));
    end
    set_titles(ax, sprintf("Q6  Trend over the session:  slope %.3g/slot, Wilcoxon %s %s", ...
        R.slope_mean, fmt_p(R.p_slope_wilcoxon), studyplot.stars(R.p_slope_wilcoxon)), sub, W);
else
    axis(ax, 'off'); text(ax, 0.5, 0.5, "Q6 not applicable", 'HorizontalAlignment', 'center');
end
end

% ------------------------------------------------------------------ helpers
function set_titles(ax, main, sub, W)
title(ax, studyplot.wrap(main, W));
subtitle(ax, studyplot.wrap(sub, W));
end

function s = sigpairs(R, compact)
% Compact panels have no room for brackets: list the Holm-significant pairs.
s = "";
if ~compact || ~isfield(R, 'pairs') || isempty(R.pairs), return, end
sig = R.pairs(R.pairs.significant, :);
if isempty(sig), s = "  |  no pair significant after Holm"; return, end
pp = strings(height(sig), 1);
for i = 1:height(sig), pp(i) = sprintf("%s vs %s (p=%.3f)", sig.a(i), sig.b(i), sig.p_holm(i)); end
s = "  |  Holm-significant: " + strjoin(pp, ", ");
end

function add_pair_brackets(ax, R, compact)
if compact, return, end
if ~isfield(R, 'pairs') || isempty(R.pairs), return, end
sig = R.pairs(R.pairs.significant, :);
if isempty(sig), return, end
lv = R.levels;
yl = ylim(ax); span = yl(2) - yl(1); y = yl(2) + 0.05 * span;
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
