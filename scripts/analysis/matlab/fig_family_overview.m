function fig = fig_family_overview(trial, S, items, families)
%FIG_FAMILY_OVERVIEW One row per metric family: cell grid, mode effect, assistance effect.
%   FIG = FIG_FAMILY_OVERVIEW(TRIAL, S, ITEMS, FAMILIES) draws, for every
%   family score (z-units, higher = better), the 2 x 3 grid of cell means
%   (Q3), the Clutch-vs-Joystick paired plot (Q1) and the F/B/FB paired plot
%   (Q2), each with its test result in the title. This is the "at a glance"
%   answer to which mode / assistance / cell is best per family.

famItems = items(items.is_family & items.name ~= "composite", :);
nf = height(famItems);
fig = studyplot.newfig("Family overview", 1400, 270 * nf + 80);
tl = tiledlayout(fig, nf, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, sprintf("Which condition is best, per metric family  (family scores in z-units, higher = better, n = %d participants)", ...
    numel(unique(trial.participant))), 'FontWeight', 'bold', 'FontSize', 13);

for i = 1:nf
    it = table2struct(famItems(i, :));
    nm = it.name;
    lab = families.label(families.key == it.family);

    % --- cells ---
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
        studyplot.heat23(ax, means, cis, modes, assists, 'dir', 1, 'best', R.best, 'fmt', "%.2f");
        title(ax, sprintf("%s  -  cells: Friedman %s %s, W = %.2f", lab, fmt_p(R.p), studyplot.stars(R.p), R.W));
        subtitle(ax, "ranking: " + strjoin(R.ranking, " > "));
    else
        axis(ax, 'off');
    end

    % --- mode ---
    ax = nexttile(tl);
    if isfield(S.Q1, nm)
        R = S.Q1.(nm);
        M = participant_means(trial, nm, "mode", ["C" "J"]);
        studyplot.paired(ax, M, ["C" "J"], 'ylabel', "z");
        title(ax, sprintf("Q1 mode: %s %s, %s = %.2f", fmt_p(R.p), studyplot.stars(R.p), R.effect_name, R.effect));
        if R.significant, subtitle(ax, "-> " + studyplot.levelname(R.better) + " is better"); else, subtitle(ax, "no significant difference"); end
    else
        axis(ax, 'off');
    end

    % --- assistance ---
    ax = nexttile(tl);
    if isfield(S.Q2, nm)
        R = S.Q2.(nm);
        if R.test_family == "paired2"
            M = participant_means(trial, nm, "assist", ["B" "FB"]);
            studyplot.paired(ax, M, ["B" "FB"], 'ylabel', "z");
            title(ax, sprintf("Q2 assistance (B vs FB): %s %s, %s = %.2f", fmt_p(R.p), studyplot.stars(R.p), R.effect_name, R.effect));
            if R.significant, subtitle(ax, "-> " + studyplot.levelname(R.better) + " is better"); else, subtitle(ax, "no significant difference"); end
        else
            M = participant_means(trial, nm, "assist", ["F" "B" "FB"]);
            studyplot.paired(ax, M, ["F" "B" "FB"], 'ylabel', "z");
            title(ax, sprintf("Q2 assistance: %s %s, %s = %.2f", fmt_p(R.p), studyplot.stars(R.p), R.effect_name, R.effect));
            sig = R.pairs(R.pairs.significant, :);
            if ~isempty(sig)
                pp = strings(height(sig), 1);
                for q = 1:height(sig), pp(q) = sig.a(q) + " vs " + sig.b(q); end
                subtitle(ax, "ranking: " + strjoin(R.ranking, " > ") + "  |  Holm-significant: " + strjoin(pp, ", "));
            else
                subtitle(ax, "ranking: " + strjoin(R.ranking, " > ") + "  |  no pair significant after Holm");
            end
        end
    else
        axis(ax, 'off');
    end
end
end
