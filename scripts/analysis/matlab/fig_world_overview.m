function fig = fig_world_overview(trial, S, items, families, opts)
%FIG_WORLD_OVERVIEW Rack vs shield world for every family score (supplementary).
%   FIG = FIG_WORLD_OVERVIEW(TRIAL, S, ITEMS, FAMILIES): one paired plot per
%   family (grey = participants). The world is not a design question, but
%   its effect is large and every participant did both worlds in every cell,
%   so it is shown to make clear that averaging over worlds in Q1-Q3 adds
%   variance but no bias.
%   FIG_WORLD_OVERVIEW(..., 'compact', true): one panel with grouped bars
%   (mean +/- 95% CI per family and world) for the Live Script.

arguments
    trial table
    S struct
    items table
    families table
    opts.compact (1,1) logical = false
end
famItems = items(items.is_family, :);
nf = height(famItems);
labs = strings(nf, 1);
for i = 1:nf
    if famItems.name(i) == "composite", labs(i) = "Composite"; else, labs(i) = families.label(families.key == famItems.family(i)); end
end

if opts.compact
    fig = studyplot.newfig("World overview", 480, 380, true);
    ax = axes(fig);
    worlds = ["rack" "shield"];
    V = nan(nf, 2); C = V; p = nan(nf, 1);
    for i = 1:nf
        nm = famItems.name(i);
        M = participant_means(trial, nm, "world", worlds);
        for j = 1:2
            v = M(:, j); v = v(~isnan(v)); V(i, j) = mean(v);
            if numel(v) > 1, C(i, j) = tinv(0.975, numel(v) - 1) * std(v) / sqrt(numel(v)); end
        end
        if isfield(S.QW, nm), p(i) = S.QW.(nm).p; end
    end
    hold(ax, 'on');
    for j = 1:2
        x = (1:nf) - 0.2 + 0.4 * (j - 1);
        bar(ax, x, V(:, j), 0.36, 'FaceColor', studyplot.color(worlds(j)), 'EdgeColor', 'none', 'DisplayName', worlds(j));
        errorbar(ax, x, V(:, j), C(:, j), 'k', 'LineStyle', 'none', 'LineWidth', 0.8, 'CapSize', 3, 'HandleVisibility', 'off');
    end
    for i = 1:nf
        text(ax, i, max(V(i, :) + C(i, :)) + 0.05, studyplot.stars(p(i)), 'HorizontalAlignment', 'center', 'FontSize', 7);
    end
    yline(ax, 0, ':', 'Color', [0.4 0.4 0.4]);
    hold(ax, 'off');
    xticks(ax, 1:nf); xticklabels(ax, labs); ax.XTickLabelRotation = 25;
    ylabel(ax, "mean score (z)"); grid(ax, 'on'); box(ax, 'on');
    legend(ax, 'Location', 'northeast', 'Orientation', 'horizontal', 'Box', 'off');
    title(ax, studyplot.wrap("Supplementary: rack vs shield per family score (higher = better; stars = paired test)", 62));
    return
end

ncol = 4; nrow = ceil(nf / ncol);
fig = studyplot.newfig("World overview", 320 * ncol, 300 * nrow + 60);
tl = tiledlayout(fig, nrow, ncol, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, "Supplementary: rack vs shield world, per family score (z-units, higher = better)", 'FontWeight', 'bold', 'FontSize', 13);
for i = 1:nf
    nm = famItems.name(i);
    ax = nexttile(tl);
    if ~isfield(S.QW, nm), axis(ax, 'off'); continue; end
    R = S.QW.(nm);
    M = participant_means(trial, nm, "world", ["rack" "shield"]);
    studyplot.paired(ax, M, ["rack" "shield"], 'ylabel', "z");
    title(ax, sprintf("%s: %s %s, %s = %.2f", labs(i), fmt_p(R.p), studyplot.stars(R.p), R.effect_name, R.effect));
    if R.significant, subtitle(ax, "-> " + R.better + " scores higher"); else, subtitle(ax, "no significant difference"); end
end
end
