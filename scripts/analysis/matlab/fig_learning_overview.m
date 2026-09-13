function fig = fig_learning_overview(S, items, families, opts)
%FIG_LEARNING_OVERVIEW Family scores along the six experiment slots (Q6) and the mode-order check.
%   FIG = FIG_LEARNING_OVERVIEW(S, ITEMS, FAMILIES) draws, per family, the
%   score versus slot (left: grey = participants, black = mean +/- SD) with
%   the slope test in the title, and (right) the Joystick-minus-Clutch
%   difference of each participant split by which mode they met first: if
%   the two groups differ, the mode comparison is contaminated by practice.
%   FIG_LEARNING_OVERVIEW(..., 'compact', true): two stacked panels, the mean
%   of every family score against the slot (one line per family) and the
%   mode-order check of the composite (Live Script layout).

arguments
    S struct
    items table
    families table
    opts.compact (1,1) logical = false
end
famItems = items(items.is_family, :);
famItems = famItems(arrayfun(@(nm) isfield(S.Q6, nm), famItems.name), :);
nf = height(famItems);

if opts.compact
    % Kept deliberately short: the legend only names each line (the slope
    % and p-value for every family are already in the show_results table
    % right below this figure in the report, so repeating them here would
    % just crowd the plot without adding anything new to read).
    fig = studyplot.newfig("Learning overview", 480, 210 * 2 + 110, true);
    tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'loose', 'Padding', 'loose');
    title(tl, "Learning and order effects", 'FontWeight', 'bold', 'FontSize', 10);
    subtitle(tl, "slot 1 = first condition met; slots 1-3 one control mode, 4-6 the other", 'FontSize', 7.5);
    ax = nexttile(tl); hold(ax, 'on');
    cmap = lines(nf);
    for i = 1:nf
        nm = famItems.name(i); R = S.Q6.(nm);
        if nm == "composite", lab = "Composite"; lw = 2.5; col = [0 0 0]; else, lab = families.label(families.key == famItems.family(i)); lw = 1.2; col = cmap(i, :); end
        plot(ax, 1:6, R.slot_mean, '-o', 'Color', col, 'LineWidth', lw, 'MarkerFaceColor', col, 'MarkerSize', 3, ...
             'DisplayName', lab);
    end
    hold(ax, 'off'); xlim(ax, [0.8 6.2]); xticks(ax, 1:6); grid(ax, 'on'); box(ax, 'on');
    xlabel(ax, "experiment slot"); ylabel(ax, "mean score (z)");
    legend(ax, 'Location', 'eastoutside', 'Box', 'off', 'FontSize', 6.5);
    title(ax, "Score per slot", 'FontSize', 9);

    ax = nexttile(tl);
    R = S.Q6.("composite");
    order_panel(ax, R, "Composite");
    return
end

fig = studyplot.newfig("Learning overview", 1100, 230 * nf + 80);
tl = tiledlayout(fig, nf, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, "Learning and order effects  (slot 1 = first condition met; slots 1-3 one control mode, 4-6 the other)", ...
      'FontWeight', 'bold', 'FontSize', 13);

for i = 1:nf
    nm = famItems.name(i);
    if nm == "composite", lab = "Composite"; else, lab = families.label(families.key == famItems.family(i)); end
    R = S.Q6.(nm);
    ax = nexttile(tl);
    studyplot.learning(ax, R.slot_matrix, 'ylabel', "score (z)");
    title(ax, sprintf("%s  -  slope %.3f z/slot (95%% CI %.3f to %.3f), Wilcoxon %s %s", ...
        lab, R.slope_mean, R.slope_ci(1), R.slope_ci(2), fmt_p(R.p_slope_wilcoxon), studyplot.stars(R.p_slope_wilcoxon)));
    subtitle(ax, sprintf("block effect (slots 4-6 vs 1-3): %s  |  Friedman across slots %s", ...
        fmt_p(R.block.p), fmt_p(R.p_friedman_slots)));
    ax = nexttile(tl);
    order_panel(ax, R, lab);
end
end

function order_panel(ax, R, lab)
gC = R.mode_diff_C_first; gJ = R.mode_diff_J_first;
hold(ax, 'on');
rng(3, 'twister');
plot(ax, 1 + (rand(numel(gC), 1) - 0.5) * 0.3, gC, 'o', 'MarkerFaceColor', studyplot.color("C"), 'MarkerEdgeColor', 'none', 'MarkerSize', 6);
plot(ax, 2 + (rand(numel(gJ), 1) - 0.5) * 0.3, gJ, 'o', 'MarkerFaceColor', studyplot.color("J"), 'MarkerEdgeColor', 'none', 'MarkerSize', 6);
plot(ax, [0.7 1.3], mean(gC) * [1 1], 'k-', 'LineWidth', 2);
plot(ax, [1.7 2.3], mean(gJ) * [1 1], 'k-', 'LineWidth', 2);
yline(ax, 0, ':', 'Color', [0.4 0.4 0.4]);
hold(ax, 'off');
xlim(ax, [0.5 2.5]); xticks(ax, 1:2); xticklabels(ax, {'Clutch first', 'Joystick first'});
ylabel(ax, "Joystick - Clutch (z)"); grid(ax, 'on'); box(ax, 'on');
title(ax, sprintf("%s: order check  %s %s", lab, fmt_p(R.p_order_ranksum), studyplot.stars(R.p_order_ranksum)), 'FontSize', 9);
if ~isnan(R.p_order_ranksum) && R.p_order_ranksum < R.alpha
    subtitle(ax, studyplot.wrap(sprintf("n=%d/%d -- WARNING: depends on order, partly a practice effect", R.mode_order_n(1), R.mode_order_n(2)), 55), 'FontSize', 7.5);
else
    subtitle(ax, studyplot.wrap(sprintf("n=%d/%d -- does not depend on order, no bias detected", R.mode_order_n(1), R.mode_order_n(2)), 55), 'FontSize', 7.5);
end
end
