function fig = fig_family_compact(trial, S, item, families)
%FIG_FAMILY_COMPACT One family score: cell grid (Q3), mode (Q1), assistance (Q2), stacked.
%   FIG = FIG_FAMILY_COMPACT(TRIAL, S, ITEM, FAMILIES) is the Live-Script
%   sized version of one row of FIG_FAMILY_OVERVIEW: three panels in one
%   column (480 x 660 px), each title carrying the test result.

nm = item.name;
if nm == "composite", lab = "Composite score"; else, lab = families.label(families.key == item.family) + " score"; end
n = numel(unique(trial.participant));
fig = studyplot.newfig(lab, 480, 660, true);
tl = tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, sprintf("%s  (z-units, higher = better, n = %d)", lab, n), 'FontWeight', 'bold', 'FontSize', 9);
W = 72;

ax = nexttile(tl);
if isfield(S.Q3, nm)
    R = S.Q3.(nm);
    modes = unique(R.modes, 'stable'); assists = unique(R.assists, 'stable');
    M = participant_means(trial, nm, "cell", R.levels);
    means = nan(numel(modes), numel(assists)); cis = means;
    for j = 1:numel(R.levels)
        r = find(modes == R.modes(j)); c = find(assists == R.assists(j));
        v = M(:, j); v = v(~isnan(v)); means(r, c) = mean(v);
        if numel(v) > 1, cis(r, c) = tinv(0.975, numel(v) - 1) * std(v) / sqrt(numel(v)); end
    end
    studyplot.heat23(ax, means, cis, modes, assists, 'dir', 1, 'best', R.best, 'fmt', "%.2f", 'bestmark', " *");
    title(ax, studyplot.wrap(sprintf("Q3 cells: Friedman %s %s, Kendall W = %.2f", fmt_p(R.p), studyplot.stars(R.p), R.W), W));
    sub = "ranking: " + strjoin(R.ranking, " > ") + " (* best)";
    if ~isempty(R.effects)
        ia = R.effects.effect == "Mode:Assist";
        if any(ia), sub = sub + sprintf("  |  interaction %s", fmt_p(R.effects.p_gg(ia))); end
    end
    subtitle(ax, studyplot.wrap(sub, W));
else
    axis(ax, 'off');
end

ax = nexttile(tl);
if isfield(S.Q1, nm)
    R = S.Q1.(nm);
    M = participant_means(trial, nm, "mode", ["C" "J"]);
    studyplot.paired(ax, M, ["C" "J"], 'ylabel', "z", 'horizontal', true);
    title(ax, studyplot.wrap(sprintf("Q1 mode: %s %s (%s), %s = %.2f", fmt_p(R.p), studyplot.stars(R.p), ...
        studyplot.testname(R.recommended), R.effect_name, R.effect), W));
    if R.significant, subtitle(ax, "-> " + studyplot.levelname(R.better) + " is better"); else, subtitle(ax, "no significant difference"); end
else
    axis(ax, 'off');
end

ax = nexttile(tl);
if isfield(S.Q2, nm)
    R = S.Q2.(nm);
    if R.test_family == "paired2"
        M = participant_means(trial, nm, "assist", ["B" "FB"]);
        studyplot.paired(ax, M, ["B" "FB"], 'ylabel', "z", 'horizontal', true);
        title(ax, studyplot.wrap(sprintf("Q2 assistance (B vs FB): %s %s, %s = %.2f", fmt_p(R.p), studyplot.stars(R.p), R.effect_name, R.effect), W));
        if R.significant, subtitle(ax, "-> " + studyplot.levelname(R.better) + " is better"); else, subtitle(ax, "no significant difference"); end
    else
        M = participant_means(trial, nm, "assist", ["F" "B" "FB"]);
        studyplot.paired(ax, M, ["F" "B" "FB"], 'ylabel', "z", 'horizontal', true);
        title(ax, studyplot.wrap(sprintf("Q2 assistance: %s %s (%s), %s = %.2f", fmt_p(R.p), studyplot.stars(R.p), ...
            studyplot.testname(R.recommended), R.effect_name, R.effect), W));
        sig = R.pairs(R.pairs.significant, :);
        if ~isempty(sig)
            pp = strings(height(sig), 1);
            for q = 1:height(sig), pp(q) = sig.a(q) + " vs " + sig.b(q); end
            subtitle(ax, studyplot.wrap("ranking: " + strjoin(R.ranking, " > ") + "  |  Holm-significant: " + strjoin(pp, ", "), W));
        else
            subtitle(ax, studyplot.wrap("ranking: " + strjoin(R.ranking, " > ") + "  |  no pair significant after Holm", W));
        end
    end
else
    axis(ax, 'off');
end
end
