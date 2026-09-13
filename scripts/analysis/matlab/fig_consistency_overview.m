function fig = fig_consistency_overview(S, items, families, opts)
%FIG_CONSISTENCY_OVERVIEW Between-participant spread of every family score per condition (Q4, Q5).
%   FIG = FIG_CONSISTENCY_OVERVIEW(S, ITEMS, FAMILIES) draws, per family, the
%   standard deviation across participants of the family score for each
%   control mode (left) and each assistance level (right), with bootstrap
%   95% CIs. A shorter bar = participants obtained more similar results
%   under that condition. The title carries the Pitman-Morgan test of equal
%   spread and Kendall's W (agreement on the ranking of the conditions).
%   FIG_CONSISTENCY_OVERVIEW(..., 'compact', true): two stacked panels with
%   the families side by side (Live Script layout); a star above a family
%   marks a significant difference in spread.

arguments
    S struct
    items table
    families table
    opts.compact (1,1) logical = false
end
famItems = items(items.is_family & items.name ~= "composite", :);
nf = height(famItems);

if opts.compact
    % Horizontal grouped bars: family names as plain y-axis text (no rotated,
    % overlapping tick labels), one legend per panel placed clear of the
    % bars, and a single small marker (not floating text) on the winning bar
    % of a family whose spread difference is significant -- the full
    % statistics are already in the show_results table right below this
    % figure in the report, so the plot itself only needs to be readable.
    fig = studyplot.newfig("Consistency overview", 480, 230 * 2 + 110, true);
    tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'loose', 'Padding', 'loose');
    title(tl, "Most similar results across participants", 'FontWeight', 'bold', 'FontSize', 10);
    subtitle(tl, "shorter bar = more consistent; * = significantly more consistent (Pitman-Morgan)", 'FontSize', 7.5);
    labels = arrayfun(@(k) families.label(families.key == k), famItems.family);
    for q = ["Q4" "Q5"]
        ax = nexttile(tl);
        if q == "Q4", levels = ["C" "J"]; panelTitle = "By control mode"; else, levels = ["F" "B" "FB"]; panelTitle = "By assistance level"; end
        k = numel(levels);
        V = nan(nf, k); Lo = V; Hi = V; sigf = false(nf, 1); best = strings(nf, 1);
        for i = 1:nf
            nm = famItems.name(i);
            if ~isfield(S.(q), nm), continue; end
            R = S.(q).(nm); L = R.levels_table;
            for j = 1:k
                r = find(L.level == levels(j), 1);
                if isempty(r), continue; end
                V(i, j) = L.sd(r); Lo(i, j) = L.sd_ci_lo(r); Hi(i, j) = L.sd_ci_hi(r);
            end
            sigf(i) = R.significant; best(i) = R.most_consistent;
        end
        hold(ax, 'on');
        wdt = 0.75 / k;
        ypos = nan(nf, k);
        for j = 1:k
            y = (1:nf)' - 0.375 + wdt * (j - 0.5);
            ypos(:, j) = y;
            barh(ax, y, V(:, j), wdt * 0.9, 'FaceColor', studyplot.color(levels(j)), 'EdgeColor', 'none', 'DisplayName', levels(j));
            errorbar(ax, V(:, j), y, Lo(:, j) - V(:, j), Hi(:, j) - V(:, j), 'horizontal', 'k', ...
                     'LineStyle', 'none', 'LineWidth', 0.8, 'CapSize', 2, 'HandleVisibility', 'off');
        end
        for i = 1:nf
            if sigf(i)
                jbest = find(levels == best(i), 1);
                plot(ax, Hi(i, jbest) + 0.02 * max(Hi(:), [], 'omitnan'), ypos(i, jbest), 'k*', ...
                     'MarkerSize', 5, 'HandleVisibility', 'off');
            end
        end
        hold(ax, 'off');
        ylim(ax, [0.5 nf + 0.5]); yticks(ax, 1:nf); yticklabels(ax, labels); ax.YDir = 'reverse';
        xlim(ax, [0, max(Hi(:), [], 'omitnan') * 1.22]);
        xlabel(ax, "SD across participants"); grid(ax, 'on'); box(ax, 'on');
        legend(ax, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off', 'FontSize', 7);
        title(ax, panelTitle, 'FontSize', 9);
    end
    return
end

fig = studyplot.newfig("Consistency overview", 1100, 230 * nf + 80);
tl = tiledlayout(fig, nf, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, "Which condition gives the most SIMILAR results across participants?  (shorter bar = more consistent)", ...
      'FontWeight', 'bold', 'FontSize', 13);
subtitle(tl, "bars: SD across participants of the family score, whiskers: bootstrap 95% CI");

for i = 1:nf
    nm = famItems.name(i);
    lab = families.label(families.key == famItems.family(i));
    for q = ["Q4" "Q5"]
        ax = nexttile(tl);
        if isfield(S.(q), nm)
            R = S.(q).(nm); L = R.levels_table;
            studyplot.bars(ax, L.sd', L.sd_ci_lo', L.sd_ci_hi', R.levels, 'ylabel', "SD of family score");
            if q == "Q4", what = "mode"; else, what = "assistance"; end
            title(ax, sprintf("%s  -  by %s: most consistent = %s, Pitman-Morgan %s %s", ...
                lab, what, studyplot.levelname(R.most_consistent), fmt_p(R.p), studyplot.stars(R.p)));
            subtitle(ax, sprintf("agreement on the ranking: Kendall W = %.2f (%s)", R.W, effect_band("W", R.W)));
        else
            axis(ax, 'off');
        end
    end
end
end
