function fig = fig_world_overview(trial, S, items, families)
%FIG_WORLD_OVERVIEW Rack vs shield world for every family score (supplementary).
%   FIG = FIG_WORLD_OVERVIEW(TRIAL, S, ITEMS, FAMILIES): one paired plot per
%   family (grey = participants). The world is not a design question, but
%   its effect is large and every participant did both worlds in every cell,
%   so it is shown to make clear that averaging over worlds in Q1-Q3 adds
%   variance but no bias.

famItems = items(items.is_family, :);
nf = height(famItems);
ncol = 4; nrow = ceil(nf / ncol);
fig = studyplot.newfig("World overview", 320 * ncol, 300 * nrow + 60);
tl = tiledlayout(fig, nrow, ncol, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, "Supplementary: rack vs shield world, per family score (z-units, higher = better)", 'FontWeight', 'bold', 'FontSize', 13);
for i = 1:nf
    nm = famItems.name(i);
    if nm == "composite", lab = "Composite"; else, lab = families.label(families.key == famItems.family(i)); end
    ax = nexttile(tl);
    if ~isfield(S.QW, nm), axis(ax, 'off'); continue; end
    R = S.QW.(nm);
    M = participant_means(trial, nm, "world", ["rack" "shield"]);
    studyplot.paired(ax, M, ["rack" "shield"], 'ylabel', "z");
    title(ax, sprintf("%s: %s %s, %s = %.2f", lab, fmt_p(R.p), studyplot.stars(R.p), R.effect_name, R.effect));
    if R.significant, subtitle(ax, "-> " + R.better + " scores higher"); else, subtitle(ax, "no significant difference"); end
end
end
