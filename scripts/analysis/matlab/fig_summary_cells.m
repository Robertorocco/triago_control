function fig = fig_summary_cells(trial, S, items, families)
%FIG_SUMMARY_CELLS Families x cells heat matrix and the composite ranking of the six cells.
%   FIG = FIG_SUMMARY_CELLS(TRIAL, S, ITEMS, FAMILIES). Left: mean family
%   score (z-units, higher = better) in every mode x assistance cell, red =
%   better, blue = worse, star = best cell of that row, Friedman test on the
%   right margin. Right: composite score per cell ranked best to worst, bar
%   = mean with 95% CI, dots = participants.

CELLS = ["CF" "CB" "CFB" "JF" "JB" "JFB"];
it = items(items.is_family, :);
it = [it(it.name ~= "composite", :); it(it.name == "composite", :)];
nr = height(it);
n = numel(unique(trial.participant));

V = nan(nr, 6); best = strings(nr, 1); ptxt = strings(nr, 1);
for i = 1:nr
    M = participant_means(trial, it.name(i), "cell", CELLS);
    V(i, :) = mean(M, 1, 'omitnan');
    if isfield(S.Q3, it.name(i))
        R = S.Q3.(it.name(i)); best(i) = R.best;
        ptxt(i) = sprintf("%s %s   W = %.2f", fmt_p(R.p), studyplot.stars(R.p), R.W);
    end
end
labs = arrayfun(@(k) studyplot.famlabel(families, k), it.family);
labs(it.name == "composite") = "COMPOSITE";

fig = studyplot.profig("Cells summary", 1450, 700);
tl = tiledlayout(fig, 1, 5, 'TileSpacing', 'loose', 'Padding', 'loose');
title(tl, sprintf("Which mode + assistance combination is best?   (n = %d participants; scores in z-units, higher = better)", n), ...
      'FontWeight', 'bold', 'FontSize', 13);

% ---------------- heat matrix ----------------
ax = nexttile(tl, [1 3]);
c = max(abs(V(:)), [], 'omitnan'); if isempty(c) || c == 0, c = 1; end
Vp = V; Vp(isnan(Vp)) = 0;
imagesc(ax, Vp); colormap(ax, studyplot.cool2warm(128)); clim(ax, [-c c]);
hold(ax, 'on');
for i = 1:nr
    for j = 1:6
        if isnan(V(i, j))
            patch(ax, j + [-0.5 0.5 0.5 -0.5], i + [-0.5 -0.5 0.5 0.5], [0.93 0.93 0.93], 'EdgeColor', 'none');
            text(ax, j, i, "n/a", 'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', [0.5 0.5 0.5]);
            continue
        end
        bg = interp1(linspace(-c, c, 128), studyplot.cool2warm(128), V(i, j));
        txt = sprintf("%+.2f", V(i, j)); if CELLS(j) == best(i), txt = txt + " *"; end
        text(ax, j, i, txt, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10, 'Color', studyplot.textcolor_on(bg));
    end
    text(ax, 6.62, i, ptxt(i), 'FontSize', 8.5, 'Color', [0.25 0.25 0.25], 'VerticalAlignment', 'middle');
end
% white grid between cells, thicker separators between the modes and before the composite row
for j = 0.5:1:6.5, xline(ax, j, 'Color', 'w', 'LineWidth', 1.5 + 2.5 * (j == 3.5)); end
for i = 0.5:1:nr + 0.5, yline(ax, i, 'Color', 'w', 'LineWidth', 1.5 + 2.5 * (i == nr - 0.5)); end
hold(ax, 'off');
xticks(ax, 1:6); xticklabels(ax, CELLS); yticks(ax, 1:nr); yticklabels(ax, labs);
ax.YDir = 'reverse'; ax.TickLength = [0 0];
xlim(ax, [0.5 8.9]); ylim(ax, [-0.1 nr + 0.5]);
% mode group labels under the cell codes, Friedman header in the band above row 1
text(ax, 2, nr + 1.05, "CLUTCH", 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10, 'Color', studyplot.color("C"));
text(ax, 5, nr + 1.05, "JOYSTICK", 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10, 'Color', studyplot.color("J"));
text(ax, 6.62, 0.2, "Friedman over the 6 cells", 'FontSize', 8.5, 'FontWeight', 'bold', 'Color', [0.25 0.25 0.25]);
title(ax, "Mean family score per cell");
subtitle(ax, "red = better, blue = worse, * = best cell of the row", 'FontSize', 9, 'Color', [0.35 0.35 0.35]);

% ---------------- composite ranking ----------------
ax = nexttile(tl, [1 2]);
M = participant_means(trial, "composite", "cell", CELLS);
mu = mean(M, 1, 'omitnan'); ci = tinv(0.975, n - 1) * std(M, 0, 1, 'omitnan') / sqrt(n);
[~, ord] = sort(mu, 'descend');
hold(ax, 'on');
rng(5, 'twister');
for r = 1:6
    k = ord(r);
    barh(ax, r, mu(k), 0.62, 'FaceColor', studyplot.color(CELLS(k)), 'EdgeColor', 'none');
    errorbar(ax, mu(k), r, ci(k), 'horizontal', 'k', 'LineStyle', 'none', 'LineWidth', 1.1, 'CapSize', 6);
    plot(ax, M(:, k), r + (rand(n, 1) - 0.5) * 0.35, 'o', 'MarkerSize', 4, 'MarkerFaceColor', [0.8 0.8 0.8], 'MarkerEdgeColor', [0.45 0.45 0.45]);
end
xline(ax, 0, ':', 'Color', [0.4 0.4 0.4]);
hold(ax, 'off');
ylim(ax, [0.4 6.6]); yticks(ax, 1:6); yticklabels(ax, CELLS(ord)); ax.YDir = 'reverse';
xlabel(ax, "composite score (z)"); grid(ax, 'on'); ax.YGrid = 'off';
R = S.Q3.("composite");
title(ax, "Composite ranking, best first");
sig = R.pairs(R.pairs.significant, :);
if ~isempty(sig)
    pp = strings(height(sig), 1);
    for q = 1:height(sig)
        if sig.mean_diff(q) > 0, pp(q) = sig.a(q) + " > " + sig.b(q); else, pp(q) = sig.b(q) + " > " + sig.a(q); end
    end
    sub = sprintf("Friedman %s, W = %.2f | Holm-sig.: %s", fmt_p(R.p), R.W, strjoin(pp, ", "));
else
    sub = sprintf("Friedman %s, W = %.2f | no pair significant after Holm", fmt_p(R.p), R.W);
end
subtitle(ax, studyplot.wrap(sub, 48), 'FontSize', 8.5, 'Color', [0.35 0.35 0.35]);
end
