function fig = fig_learning_overview(S, items, families)
%FIG_LEARNING_OVERVIEW Family scores along the six experiment slots (Q6) and the mode-order check.
%   FIG = FIG_LEARNING_OVERVIEW(S, ITEMS, FAMILIES) draws, per family, the
%   score versus slot (left: grey = participants, black = mean +/- SD) with
%   the slope test in the title, and (right) the Joystick-minus-Clutch
%   difference of each participant split by which mode they met first: if
%   the two groups differ, the mode comparison is contaminated by practice.

famItems = items(items.is_family, :);
famItems = famItems(arrayfun(@(nm) isfield(S.Q6, nm), famItems.name), :);
nf = height(famItems);
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
    gC = R.mode_diff_C_first; gJ = R.mode_diff_J_first;
    hold(ax, 'on');
    rng(3, 'twister');
    plot(ax, 1 + (rand(numel(gC), 1) - 0.5) * 0.3, gC, 'o', 'MarkerFaceColor', studyplot.color("C"), 'MarkerEdgeColor', 'none', 'MarkerSize', 7);
    plot(ax, 2 + (rand(numel(gJ), 1) - 0.5) * 0.3, gJ, 'o', 'MarkerFaceColor', studyplot.color("J"), 'MarkerEdgeColor', 'none', 'MarkerSize', 7);
    plot(ax, [0.7 1.3], mean(gC) * [1 1], 'k-', 'LineWidth', 2);
    plot(ax, [1.7 2.3], mean(gJ) * [1 1], 'k-', 'LineWidth', 2);
    yline(ax, 0, ':', 'Color', [0.4 0.4 0.4]);
    hold(ax, 'off');
    xlim(ax, [0.5 2.5]); xticks(ax, 1:2); xticklabels(ax, {'Clutch first', 'Joystick first'});
    ylabel(ax, "Joystick - Clutch (z)"); grid(ax, 'on'); box(ax, 'on');
    title(ax, sprintf("Mode-order check: rank-sum %s %s  (n = %d / %d)", ...
        fmt_p(R.p_order_ranksum), studyplot.stars(R.p_order_ranksum), R.mode_order_n(1), R.mode_order_n(2)));
    if ~isnan(R.p_order_ranksum) && R.p_order_ranksum < R.alpha
        subtitle(ax, "WARNING: the mode difference depends on the order -> partly a practice effect");
    else
        subtitle(ax, "the mode difference does not depend on the order -> no order bias detected");
    end
end
end
