function fig = fig_cell_ranking(trial, S)
%FIG_CELL_RANKING Composite score of the six cells, ranked, with per-participant dots (Q3 headline).
%   FIG = FIG_CELL_RANKING(TRIAL, S) draws the composite score (mean of the
%   family scores, z-units, higher = better) of every cell as a bar with its
%   95% CI, ordered best to worst, and the cell grid as a heatmap. The
%   composite is a descriptive index: the rigorous statements are the
%   per-family tests, but this is the single picture that summarises them.

nm = "composite";
R = S.Q3.(nm);
cells = R.levels;
M = participant_means(trial, nm, "cell", cells);
n = size(M, 1);
means = mean(M, 1, 'omitnan');
cis = tinv(0.975, n - 1) * std(M, 0, 1, 'omitnan') / sqrt(n);
[~, order] = sort(means, 'descend');

fig = studyplot.newfig("Cell ranking", 1150, 460);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, sprintf("Which mode + assistance combination is best overall?  (composite score, z-units, higher = better, n = %d)", n), ...
      'FontWeight', 'bold', 'FontSize', 13);

ax = nexttile(tl);
hold(ax, 'on');
rng(5, 'twister');
for j = 1:numel(cells)
    c = order(j);
    bar(ax, j, means(c), 0.6, 'FaceColor', studyplot.color(cells(c)), 'EdgeColor', 'none');
    errorbar(ax, j, means(c), cis(c), 'k', 'LineWidth', 1.2, 'CapSize', 8);
    plot(ax, j + (rand(n, 1) - 0.5) * 0.3, M(:, c), 'o', 'MarkerSize', 4, 'MarkerFaceColor', [0.75 0.75 0.75], 'MarkerEdgeColor', [0.4 0.4 0.4]);
end
yline(ax, 0, ':', 'Color', [0.4 0.4 0.4]);
hold(ax, 'off');
xticks(ax, 1:numel(cells)); xticklabels(ax, cells(order)); xlim(ax, [0.4 numel(cells) + 0.6]);
ylabel(ax, "composite score (z)"); grid(ax, 'on'); box(ax, 'on');
title(ax, sprintf("Ranking best -> worst:  Friedman %s %s, Kendall W = %.2f", fmt_p(R.p), studyplot.stars(R.p), R.W));
sig = R.pairs(R.pairs.significant, :);
if ~isempty(sig)
    pp = strings(height(sig), 1);
    for q = 1:height(sig), pp(q) = sig.a(q) + " vs " + sig.b(q); end
    subtitle(ax, "Holm-significant pairs: " + strjoin(pp, ", "));
else
    subtitle(ax, "no pair of cells differs significantly after Holm correction");
end

ax = nexttile(tl);
modes = ["C" "J"]; assists = ["F" "B" "FB"];
G = nan(2, 3); Gc = G;
for j = 1:numel(cells)
    r = find(modes == R.modes(j)); c = find(assists == R.assists(j));
    G(r, c) = means(j); Gc(r, c) = cis(j);
end
studyplot.heat23(ax, G, Gc, modes, assists, 'dir', 1, 'best', R.best, 'fmt', "%.2f");
title(ax, "Cell grid (mean composite +/- 95% CI)");
if ~isempty(R.effects)
    E = R.effects;
    subtitle(ax, sprintf("RM-ANOVA: mode %s, assistance %s, interaction %s", ...
        fmt_p(E.p_gg(E.effect == "Mode")), fmt_p(E.p_gg(E.effect == "Assist")), fmt_p(E.p_gg(E.effect == "Mode:Assist"))));
end
end
