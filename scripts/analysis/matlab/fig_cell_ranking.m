function fig = fig_cell_ranking(trial, S, opts)
%FIG_CELL_RANKING Composite score of the six cells, ranked, with per-participant dots (Q3 headline).
%   FIG = FIG_CELL_RANKING(TRIAL, S) draws the composite score (mean of the
%   family scores, z-units, higher = better) of every cell as a bar with its
%   95% CI, ordered best to worst, and the cell grid as a heatmap. The
%   composite is a descriptive index: the rigorous statements are the
%   per-family tests, but this is the single picture that summarises them.
%   FIG_CELL_RANKING(..., 'compact', true) stacks the two panels (Live Script).

arguments
    trial table
    S struct
    opts.compact (1,1) logical = false
end
c = opts.compact;
nm = "composite";
R = S.Q3.(nm);
cells = R.levels;
M = participant_means(trial, nm, "cell", cells);
n = size(M, 1);
means = mean(M, 1, 'omitnan');
cis = tinv(0.975, n - 1) * std(M, 0, 1, 'omitnan') / sqrt(n);
[~, order] = sort(means, 'descend');

if c
    fig = studyplot.newfig("Cell ranking", 480, 620, true);
    tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
else
    fig = studyplot.newfig("Cell ranking", 1150, 460);
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
end
W = 70 + 130 * ~c;
title(tl, studyplot.wrap(sprintf("Which mode + assistance combination is best overall?  (composite score, z-units, higher = better, n = %d)", n), W), ...
      'FontWeight', 'bold', 'FontSize', 13 - 4 * c);

ax = nexttile(tl);
hold(ax, 'on');
rng(5, 'twister');
for j = 1:numel(cells)
    k = order(j);
    bar(ax, j, means(k), 0.6, 'FaceColor', studyplot.color(cells(k)), 'EdgeColor', 'none');
    errorbar(ax, j, means(k), cis(k), 'k', 'LineWidth', 1.2, 'CapSize', 8);
    plot(ax, j + (rand(n, 1) - 0.5) * 0.3, M(:, k), 'o', 'MarkerSize', 4, 'MarkerFaceColor', [0.75 0.75 0.75], 'MarkerEdgeColor', [0.4 0.4 0.4]);
end
yline(ax, 0, ':', 'Color', [0.4 0.4 0.4]);
hold(ax, 'off');
xticks(ax, 1:numel(cells)); xticklabels(ax, cells(order)); xlim(ax, [0.4 numel(cells) + 0.6]);
ylabel(ax, "composite score (z)"); grid(ax, 'on'); box(ax, 'on');
title(ax, studyplot.wrap(sprintf("Ranking best -> worst:  Friedman %s %s, Kendall W = %.2f", fmt_p(R.p), studyplot.stars(R.p), R.W), W));
sig = R.pairs(R.pairs.significant, :);
if ~isempty(sig)
    pp = strings(height(sig), 1);
    for q = 1:height(sig), pp(q) = sig.a(q) + " vs " + sig.b(q); end
    subtitle(ax, studyplot.wrap("Holm-significant pairs: " + strjoin(pp, ", "), W));
else
    subtitle(ax, studyplot.wrap("no pair of cells differs significantly after Holm correction", W));
end

ax = nexttile(tl);
modes = ["C" "J"]; assists = ["F" "B" "FB"];
G = nan(2, 3); Gc = G;
for j = 1:numel(cells)
    r = find(modes == R.modes(j)); a = find(assists == R.assists(j));
    G(r, a) = means(j); Gc(r, a) = cis(j);
end
if c, mark = " *"; else, mark = "  (best)"; end
studyplot.heat23(ax, G, Gc, modes, assists, 'dir', 1, 'best', R.best, 'fmt', "%.2f", 'bestmark', mark);
title(ax, "Cell grid (mean composite +/- 95% CI)");
if ~isempty(R.effects)
    E = R.effects;
    subtitle(ax, studyplot.wrap(sprintf("RM-ANOVA: mode %s, assistance %s, interaction %s", ...
        fmt_p(E.p_gg(E.effect == "Mode")), fmt_p(E.p_gg(E.effect == "Assist")), fmt_p(E.p_gg(E.effect == "Mode:Assist"))), W));
end
end
