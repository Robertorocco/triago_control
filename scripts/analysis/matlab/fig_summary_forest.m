function fig = fig_summary_forest(trial, S, items, families, question, level, Rtab)
%FIG_SUMMARY_FOREST Forest plot of the condition effects: point estimate + 95% CI per row.
%   FIG = FIG_SUMMARY_FOREST(TRIAL, S, ITEMS, FAMILIES, QUESTION, LEVEL)
%     QUESTION "Q1": Joystick minus Clutch (positive = Joystick better)
%     QUESTION "Q2": B minus F, FB minus F, FB minus B (positive = more
%                    assistance better), one colour per contrast
%     LEVEL "family": rows = family scores (z-units, mean difference with a
%                    95% t confidence interval)
%     LEVEL "metric": rows = every directional metric, grouped by family,
%                    effect expressed as Cohen's dz (approximate 95% CI) so
%                    that metrics with different units share one axis;
%                    metrics that only compare within one control mode get
%                    one row per mode
%   Filled markers = significant, hollow = not; the annotation on the right
%   gives the p-values. At metric level the Q1 p-values are Holm-corrected
%   within the family (taken from the master table RTAB when supplied) and
%   the Q2 pairwise p-values are Holm-corrected within the metric.

if nargin < 7, Rtab = table(); end
if question == "Q1"
    A = "C"; B = "J"; clab = "Joystick - Clutch"; group = "mode";
    xpos = "Joystick better  ->"; xneg = "<-  Clutch better";
    ccol = [0.85 0.45 0.20];
else
    A = ["F" "F" "B"]; B = ["B" "FB" "FB"]; clab = ["B - F" "FB - F" "FB - B"]; group = "assist";
    xpos = "more assistance better  ->"; xneg = "<-  less assistance better";
    ccol = [0.55 0.70 0.35; 0.20 0.62 0.55; 0.10 0.30 0.45];
end
nk = numel(A);

% ---------------- rows: name, label, family, dir, source struct ----------------
rows = struct('name', {}, 'label', {}, 'family', {}, 'dir', {}, 'src', {});
if level == "family"
    it = items(items.is_family, :);
    it = [it(it.name == "composite", :); it(it.name ~= "composite", :)];
    for i = 1:height(it)
        lab = studyplot.famlabel(families, it.family(i)); if it.name(i) == "composite", lab = "COMPOSITE"; end
        rows(end + 1) = struct('name', it.name(i), 'label', lab, 'family', it.family(i), 'dir', 1, 'src', question); %#ok<AGROW>
    end
    unit = "difference in family score (z-units)";
else
    it = items(~items.is_family & items.dir ~= 0, :);
    [~, ord] = ismember(it.family, families.key); [~, o2] = sort(ord); it = it(o2, :);
    lastfam = "";
    for i = 1:height(it)
        nm = it.name(i);
        cand = struct('name', {}, 'label', {}, 'family', {}, 'dir', {}, 'src', {});
        if question == "Q1"
            if isfield(S.Q1, nm), cand(end + 1) = struct('name', nm, 'label', it.label(i), 'family', it.family(i), 'dir', it.dir(i), 'src', "Q1"); end %#ok<AGROW>
        else
            if isfield(S.Q2, nm), cand(end + 1) = struct('name', nm, 'label', it.label(i), 'family', it.family(i), 'dir', it.dir(i), 'src', "Q2"); end %#ok<AGROW>
            if isfield(S.Q2_withinC, nm), cand(end + 1) = struct('name', nm, 'label', it.label(i) + "  (Clutch)", 'family', it.family(i), 'dir', it.dir(i), 'src', "Q2_withinC"); end %#ok<AGROW>
            if isfield(S.Q2_withinJ, nm), cand(end + 1) = struct('name', nm, 'label', it.label(i) + "  (Joystick)", 'family', it.family(i), 'dir', it.dir(i), 'src', "Q2_withinJ"); end %#ok<AGROW>
        end
        if isempty(cand), continue, end
        % one header row (no data) per family, shown in the label column
        if it.family(i) ~= lastfam
            rows(end + 1) = struct('name', "", 'label', upper(studyplot.famlabel(families, it.family(i))), 'family', it.family(i), 'dir', 0, 'src', ""); %#ok<AGROW>
            lastfam = it.family(i);
        end
        rows = [rows, cand]; %#ok<AGROW>
    end
    unit = "Cohen's dz (standardised difference, sign = direction of 'better')";
end
nr = numel(rows);

% ---------------- estimates ----------------
est = nan(nr, nk); lo = est; hi = est; pv = est; sig = false(nr, nk);
for i = 1:nr
    r = rows(i);
    if strlength(r.name) == 0, continue, end        % family header row
    for k = 1:nk
        if level == "family"
            mask = true(height(trial), 1);
            M = participant_means(trial, r.name, group, [A(k) B(k)], mask);
            dd = M(:, 2) - M(:, 1); dd = dd(~isnan(dd)); n = numel(dd);
            if n < 3, continue, end
            est(i, k) = mean(dd); se = std(dd) / sqrt(n);
            lo(i, k) = est(i, k) - tinv(0.975, n - 1) * se; hi(i, k) = est(i, k) + tinv(0.975, n - 1) * se;
            [pv(i, k), sig(i, k)] = lookup(S.(r.src), r.name, A(k), B(k));
        else
            [p, s, dz, n] = lookup(S.(r.src), r.name, A(k), B(k));
            if isnan(dz), continue, end
            if question == "Q1" && ~isempty(Rtab)
                rr = find(Rtab.question == "Q1_mode" & Rtab.metric == r.name, 1);
                if ~isempty(rr), p = Rtab.p_holm(rr); s = Rtab.significant_holm(rr); end
            end
            dz = -r.dir * dz;                       % positive = second level better
            se = sqrt(1 / n + dz^2 / (2 * n));
            est(i, k) = dz; lo(i, k) = dz - 1.96 * se; hi(i, k) = dz + 1.96 * se; pv(i, k) = p; sig(i, k) = s;
        end
    end
end
keep = any(~isnan(est), 2) | (strlength(string({rows.name})) == 0)';
rows = rows(keep); est = est(keep, :); lo = lo(keep, :); hi = hi(keep, :); pv = pv(keep, :); sig = sig(keep, :);
nr = numel(rows);
rowlab = string({rows.label}); famkey = string({rows.family});

% ---------------- draw ----------------
rowh = 24 + 12 * (level == "family");
fig = studyplot.profig("Forest " + question + " " + level, 1200, 170 + rowh * nr);
ax = axes(fig, 'Position', [0.27 0.10 0.46 0.78]);
hold(ax, 'on');
xl = max(abs([lo(:); hi(:)]), [], 'omitnan') * 1.08;
if level == "metric"
    fams = unique(famkey, 'stable');
    for f = 1:numel(fams)
        rr = find(famkey == fams(f)); y0 = min(rr) - 0.5; y1 = max(rr) + 0.5;
        if mod(f, 2) == 0, patch(ax, [-xl xl xl -xl], [y0 y0 y1 y1], [0.955 0.955 0.955], 'EdgeColor', 'none'); end
    end
end
xline(ax, 0, '-', 'Color', [0.3 0.3 0.3], 'LineWidth', 1);
offs = linspace(-0.26, 0.26, nk); if nk == 1, offs = 0; end
hL = gobjects(nk, 1);
for k = 1:nk
    for i = 1:nr
        if isnan(est(i, k)), continue, end
        y = i + offs(k);
        plot(ax, [lo(i, k) hi(i, k)], [y y], '-', 'Color', ccol(k, :), 'LineWidth', 1.4);
        if sig(i, k), mf = ccol(k, :); else, mf = [1 1 1]; end
        h = plot(ax, est(i, k), y, 'o', 'MarkerSize', 6.5, 'MarkerFaceColor', mf, 'MarkerEdgeColor', ccol(k, :), 'LineWidth', 1.4);
        if ~isgraphics(hL(k)), hL(k) = h; end
    end
end
hold(ax, 'off');
ylim(ax, [-0.15 nr + 0.6]); ax.YDir = 'reverse';
yticks(ax, 1:nr); yticklabels(ax, rowlab);
xlim(ax, [-xl xl]); grid(ax, 'on'); ax.YGrid = 'off';
xlabel(ax, unit);
% direction cues in the empty band above the first row
text(ax, -xl * 0.97, 0.15, xneg, 'HorizontalAlignment', 'left', 'FontSize', 9, 'FontAngle', 'italic', 'Color', [0.35 0.35 0.35], 'VerticalAlignment', 'middle');
text(ax,  xl * 0.97, 0.15, xpos, 'HorizontalAlignment', 'right', 'FontSize', 9, 'FontAngle', 'italic', 'Color', [0.35 0.35 0.35], 'VerticalAlignment', 'middle');

for i = 1:nr
    parts = strings(1, 0);
    for k = 1:nk
        if isnan(pv(i, k)), continue, end
        s = fmt_p(pv(i, k)); if sig(i, k), s = s + " " + studyplot.stars(pv(i, k)); end
        if nk > 1, s = clab(k) + ": " + s; end
        parts(end + 1) = s; %#ok<AGROW>
    end
    text(ax, xl * 1.03, i, strjoin(parts, "    "), 'FontSize', 8.5 - 0.7 * (nk > 1), 'Color', [0.2 0.2 0.2], 'VerticalAlignment', 'middle');
end
if nk > 1
    legend(ax, hL(isgraphics(hL)), clab(isgraphics(hL)), 'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 9);
end
if question == "Q1", ttl = "Control mode: Joystick versus Clutch"; else, ttl = "Assistance: pairwise contrasts between F, B and FB"; end
if level == "family", ttl = ttl + "  --  family scores"; else, ttl = ttl + "  --  every metric"; end
title(ax, sprintf("%s   (n = %d participants)", ttl, numel(unique(trial.participant))), 'Units', 'normalized', 'Position', [0.5 1.05 0]);
if nk > 1, corr = " (Holm-corrected pairs)"; elseif level == "metric", corr = " (Holm-corrected within family)"; else, corr = ""; end
subtitle(ax, "point = estimate, bar = 95% CI, filled marker = significant" + corr, 'FontSize', 9, 'Color', [0.35 0.35 0.35]);
end

% ------------------------------------------------------------- helpers
function [p, s, dz, n] = lookup(Sq, nm, a, b)
% p, significance, raw dz (a - b) and n for one contrast from a result struct.
p = NaN; s = false; dz = NaN; n = NaN;
if ~isfield(Sq, nm), return, end
R = Sq.(nm);
n = R.n;
switch R.test_family
    case "paired2"
        if (R.levels(1) == a && R.levels(2) == b)
            p = R.p; s = R.significant; dz = R.dz;
        end
    otherwise
        if ~isfield(R, 'pairs') || isempty(R.pairs), return, end
        r = find(R.pairs.a == a & R.pairs.b == b, 1);
        if ~isempty(r), p = R.pairs.p_holm(r); s = R.pairs.significant(r); dz = R.pairs.dz(r); end
end
end
