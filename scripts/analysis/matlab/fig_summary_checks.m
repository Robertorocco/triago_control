function fig = fig_summary_checks(trial, S)
%FIG_SUMMARY_CHECKS Validity checks on the composite: practice, order bias, world.
%   FIG = FIG_SUMMARY_CHECKS(TRIAL, S) draws three panels: the composite
%   score along the six experiment slots (practice), the Joystick-minus-Clutch
%   gap by mode order (order bias) and rack versus shield (world difficulty).
%   These are the things that could make the headline results misleading; the
%   titles state whether they do.

n = numel(unique(trial.participant));
fig = studyplot.profig("Checks", 1500, 470);
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'loose', 'Padding', 'loose');
title(tl, sprintf("Sanity checks on the composite score   (n = %d participants)", n), 'FontWeight', 'bold', 'FontSize', 13);

% ---------------- practice ----------------
ax = nexttile(tl);
R = S.Q6.("composite");
Sm = R.slot_matrix;
mu = mean(Sm, 1, 'omitnan'); sd = std(Sm, 0, 1, 'omitnan'); nn = sum(~isnan(Sm), 1);
ci = tinv(0.975, max(nn - 1, 1)) .* sd ./ sqrt(nn);
hold(ax, 'on');
fill(ax, [1:6, 6:-1:1], [mu + ci, fliplr(mu - ci)], [0.85 0.45 0.20], 'FaceAlpha', 0.18, 'EdgeColor', 'none');
for i = 1:size(Sm, 1), plot(ax, 1:6, Sm(i, :), '-', 'Color', [0.82 0.82 0.82], 'LineWidth', 0.8); end
plot(ax, 1:6, mu, '-o', 'Color', [0.75 0.35 0.10], 'LineWidth', 2.2, 'MarkerFaceColor', [0.75 0.35 0.10], 'MarkerSize', 6);
hold(ax, 'off');
xlim(ax, [0.7 6.3]); xticks(ax, 1:6); grid(ax, 'on');
xlabel(ax, "experiment slot (1 = first condition met)"); ylabel(ax, "composite score (z)");
title(ax, "Practice: does performance drift along the session?");
if R.significant, v = "yes, a trend exists"; else, v = "no significant trend"; end
subtitle(ax, sprintf("%s  --  slope %.3f/slot, %s; block 4-6 vs 1-3: %s", v, R.slope_mean, fmt_p(R.p_slope_wilcoxon), fmt_p(R.block.p)), ...
         'FontSize', 9, 'Color', [0.35 0.35 0.35]);

% ---------------- order bias ----------------
ax = nexttile(tl);
gC = R.mode_diff_C_first; gJ = R.mode_diff_J_first;
hold(ax, 'on');
rng(3, 'twister');
plot(ax, 1 + (rand(numel(gC), 1) - 0.5) * 0.3, gC, 'o', 'MarkerFaceColor', studyplot.color("C"), 'MarkerEdgeColor', 'none', 'MarkerSize', 8);
plot(ax, 2 + (rand(numel(gJ), 1) - 0.5) * 0.3, gJ, 'o', 'MarkerFaceColor', studyplot.color("J"), 'MarkerEdgeColor', 'none', 'MarkerSize', 8);
plot(ax, [0.7 1.3], mean(gC) * [1 1], 'k-', 'LineWidth', 2.2);
plot(ax, [1.7 2.3], mean(gJ) * [1 1], 'k-', 'LineWidth', 2.2);
yline(ax, 0, ':', 'Color', [0.4 0.4 0.4]);
hold(ax, 'off');
xlim(ax, [0.5 2.5]); xticks(ax, 1:2); xticklabels(ax, {'met Clutch first', 'met Joystick first'});
ylabel(ax, "Joystick - Clutch  (composite, z)"); grid(ax, 'on'); ax.XGrid = 'off';
title(ax, "Order bias: does the mode effect depend on which mode came first?");
if ~isnan(R.p_order_ranksum) && R.p_order_ranksum < R.alpha, v = "YES -- read the mode comparison with care"; else, v = "no -- the mode comparison is not an order artefact"; end
subtitle(ax, sprintf("%s  (rank-sum %s, n = %d / %d)", v, fmt_p(R.p_order_ranksum), R.mode_order_n(1), R.mode_order_n(2)), 'FontSize', 9, 'Color', [0.35 0.35 0.35]);

% ---------------- world ----------------
ax = nexttile(tl);
M = participant_means(trial, "composite", "world", ["rack" "shield"]);
studyplot.paired(ax, M, ["rack" "shield"], 'ylabel', "composite score (z)");
Rw = S.QW.("composite");
title(ax, "World: rack versus shield (same participants, paired)");
if Rw.significant, v = Rw.better + " is the easier scene"; else, v = "no significant difference between the scenes"; end
subtitle(ax, sprintf("%s  (%s, %s = %.2f)", v, fmt_p(Rw.p), Rw.effect_name, abs(Rw.effect)), 'FontSize', 9, 'Color', [0.35 0.35 0.35]);
end
