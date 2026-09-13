function fig = fig_summary_scorecard(S, items, families, meta, settings)
%FIG_SUMMARY_SCORECARD One-look results matrix: families x questions, winner + evidence per tile.
%   FIG = FIG_SUMMARY_SCORECARD(S, ITEMS, FAMILIES, META, SETTINGS) draws a
%   grid with one row per family score (composite first) and one column per
%   study question. Every tile names the winning condition and the test
%   evidence (p-value, effect size); the tile colour encodes the strength of
%   the evidence (green ramp = significant, grey = not demonstrated). The
%   practice column is coloured amber/red when a trend or an order bias was
%   detected, because there "significant" is a warning, not a win.

rows = items(items.is_family, :);
rows = [rows(rows.name == "composite", :); rows(rows.name ~= "composite", :)];
nr = height(rows);
cols = ["Clutch vs Joystick" "F vs B vs FB" "Best of the 6 cells" "More consistent mode" "More consistent assistance" "Practice / order"];
nc = numel(cols);

fig = studyplot.profig("Scorecard", 1500, 110 + 68 * nr);
ax = axes(fig, 'Position', [0.13 0.10 0.85 0.74]);
hold(ax, 'on');
axis(ax, [0 nc 0 nr]); ax.YDir = 'reverse'; axis(ax, 'off');

for i = 1:nr
    nm = rows.name(i);
    lab = studyplot.famlabel(families, rows.family(i));
    if nm == "composite", lab = "COMPOSITE (all families)"; end
    text(ax, -0.05, i - 0.5, lab, 'HorizontalAlignment', 'right', 'FontWeight', 'bold', 'FontSize', 10.5);
    for j = 1:nc
        [line1, line2, col] = tile_content(S, nm, j, settings.alpha);
        patch(ax, [j-1 j j j-1] + [0.04 -0.04 -0.04 0.04], [i-1 i-1 i i] + [0.06 0.06 -0.06 -0.06], col, 'EdgeColor', 'none');
        tc = studyplot.textcolor_on(col);
        text(ax, j - 0.5, i - 0.58, line1, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11, 'Color', tc);
        text(ax, j - 0.5, i - 0.28, line2, 'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', tc);
    end
end
for j = 1:nc
    text(ax, j - 0.5, -0.22, cols(j), 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10.5);
end
hold(ax, 'off');

title(ax, sprintf("TRIAGo user study -- results at a glance   (n = %d complete participants)", meta.n_complete), ...
      'FontSize', 14, 'Units', 'normalized', 'Position', [0.5 1.14 0]);

% legend for the colour scale
lx = 0.13; ly = 0.035;
sw = {[0.13 0.47 0.29] "p < 0.001"; [0.33 0.65 0.42] "p < 0.01"; [0.66 0.84 0.62] "p < 0.05"; [0.92 0.92 0.92] "not significant"; ...
      [0.96 0.80 0.45] "practice trend"; [0.85 0.45 0.40] "order bias"};
for k = 1:size(sw, 1)
    annotation(fig, 'rectangle', [lx + (k-1) * 0.115, ly, 0.018, 0.022], 'FaceColor', sw{k, 1}, 'EdgeColor', 'none');
    annotation(fig, 'textbox', [lx + (k-1) * 0.115 + 0.022, ly - 0.006, 0.09, 0.034], 'String', sw{k, 2}, ...
               'EdgeColor', 'none', 'FontSize', 9, 'VerticalAlignment', 'middle', 'Interpreter', 'none');
end
annotation(fig, 'textbox', [0.13 0.075 0.85 0.03], 'EdgeColor', 'none', 'FontSize', 8.5, 'Color', [0.35 0.35 0.35], 'Interpreter', 'none', ...
    'String', "Family scores are z-units (higher = better). Winner = better condition in that family; effect: dz = Cohen's dz, r = rank-biserial, η² = partial eta squared, W = Kendall's W. Consistency = smaller spread across participants (Pitman-Morgan).");
end

function [l1, l2, col] = tile_content(S, nm, j, alpha)
l1 = "--"; l2 = ""; col = [1 1 1];
switch j
    case 1   % mode
        if ~isfield(S.Q1, nm), return, end
        R = S.Q1.(nm); col = studyplot.sigcolor(R.p);
        if R.significant, l1 = studyplot.levelname(R.better); else, l1 = "no difference"; end
        l2 = sprintf("%s, %s = %.2f", fmt_p(R.p), effshort(R.effect_name), abs(R.effect));
    case 2   % assistance
        if ~isfield(S.Q2, nm), return, end
        R = S.Q2.(nm); col = studyplot.sigcolor(R.p);
        if R.test_family == "paired2"
            if R.significant, l1 = R.better + " (of B, FB)"; else, l1 = "no difference (B, FB)"; end
            l2 = sprintf("%s, %s = %.2f", fmt_p(R.p), effshort(R.effect_name), abs(R.effect));
        else
            if R.significant, l1 = R.best; else, l1 = "no difference"; end
            l2 = sprintf("%s, %s = %.2f  |  %s", fmt_p(R.p), effshort(R.effect_name), R.effect, strjoin(R.ranking, ">"));
        end
    case 3   % cells
        if ~isfield(S.Q3, nm), return, end
        R = S.Q3.(nm); col = studyplot.sigcolor(R.p);
        if R.significant, l1 = R.best; else, l1 = "no difference"; end
        l2 = sprintf("%s, W = %.2f  |  top: %s", fmt_p(R.p), R.W, strjoin(R.ranking(1:min(2, end)), ">"));
    case 4   % consistency mode
        if ~isfield(S.Q4, nm), return, end
        R = S.Q4.(nm); col = studyplot.sigcolor(R.p);
        if R.significant, l1 = studyplot.levelname(R.most_consistent); else, l1 = "no difference"; end
        L = R.levels_table;
        l2 = sprintf("%s  |  SD %s", fmt_p(R.p), strjoin(compose("%s %.2f", L.level, L.sd), ", "));
    case 5   % consistency assistance
        if ~isfield(S.Q5, nm), return, end
        R = S.Q5.(nm); col = studyplot.sigcolor(R.p);
        if R.significant, l1 = R.most_consistent; else, l1 = "no difference"; end
        L = R.levels_table;
        l2 = sprintf("%s  |  SD %s", fmt_p(R.p), strjoin(compose("%s %.2f", L.level, L.sd), ", "));
    case 6   % practice / order
        if ~isfield(S.Q6, nm), return, end
        R = S.Q6.(nm);
        col = [0.92 0.92 0.92]; l1 = "no trend";
        if R.significant
            col = [0.96 0.80 0.45];
            if R.slope_mean > 0, l1 = "improves with practice"; else, l1 = "worsens with practice"; end
        end
        l2 = sprintf("slope %s  |  order %s", fmt_p(R.p_slope_wilcoxon), fmt_p(R.p_order_ranksum));
        if ~isnan(R.p_order_ranksum) && R.p_order_ranksum < alpha
            col = [0.85 0.45 0.40]; l1 = l1 + " + ORDER BIAS";
        end
end
end

function s = effshort(name)
switch string(name)
    case "Cohen dz",        s = "dz";
    case "rank-biserial r", s = "r";
    case "partial eta2",    s = "η²";
    case "Kendall W",       s = "W";
    otherwise,              s = "effect";
end
end
