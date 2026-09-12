function fig = fig_consistency_overview(S, items, families)
%FIG_CONSISTENCY_OVERVIEW Between-participant spread of every family score per condition (Q4, Q5).
%   FIG = FIG_CONSISTENCY_OVERVIEW(S, ITEMS, FAMILIES) draws, per family, the
%   standard deviation across participants of the family score for each
%   control mode (left) and each assistance level (right), with bootstrap
%   95% CIs. A shorter bar = participants obtained more similar results
%   under that condition. The title carries the Pitman-Morgan test of equal
%   spread and Kendall's W (agreement on the ranking of the conditions).

famItems = items(items.is_family & items.name ~= "composite", :);
nf = height(famItems);
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
