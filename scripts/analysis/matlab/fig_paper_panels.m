function fig = fig_paper_panels(S, spec, metrics, opts)
%FIG_PAPER_PANELS Publication-style results figure: one metric per panel, six cells.
%   FIG = FIG_PAPER_PANELS(S, SPEC, METRICS) draws one bar panel per metric of
%   METRICS. Each panel shows the six mode x assistance cells in the fixed
%   order CF CB CFB JF JB JFB, the mean across participants with its standard
%   error, and a bracket with the p-value over every pair that survives the
%   Holm correction (Q3 pairwise Wilcoxon, from RUN_STUDY_ANALYSIS).
%
%   Layout convention of a robotics results figure: one quantity per panel,
%   the same condition order in every panel, error bars = SEM, and only the
%   significant comparisons annotated so the reader is not asked to scan a
%   full pairwise matrix.
%
%   Bars start at zero unless that would flatten the whole panel into one
%   line (a narrow band of values far from zero, e.g. SPARC around -12): the
%   axis is then clipped to the data and the panel says so, because a clipped
%   bar chart exaggerates differences and must never be read as if it were
%   drawn from zero.
%
%   Options: cols (2), panel_w, panel_h, letters (true), max_brackets (5).

arguments
    S struct
    spec table
    metrics (1,:) string
    opts.cols (1,1) double = 2
    opts.panel_w (1,1) double = 520
    opts.panel_h (1,1) double = 330
    opts.letters (1,1) logical = true
    opts.max_brackets (1,1) double = 4
    opts.name (1,1) string = "paper panels"
    opts.compact (1,1) logical = false   % size and fonts for the Live Editor canvas
end

metrics = metrics(arrayfun(@(m) isfield(S.Q3, m), metrics));
k = numel(metrics);
if k == 0
    error('fig_paper_panels:noMetrics', 'None of the requested metrics has a Q3 (six-cell) result.');
end
cols = min(opts.cols, k);
rows = ceil(k / cols);

[w, h] = studyplot.size_for(opts.compact, cols * opts.panel_w, rows * opts.panel_h);
fig = studyplot.profig(opts.name, w, h);
if opts.compact
    set(fig, 'DefaultAxesFontSize', 7, 'DefaultTextFontSize', 7, ...
             'DefaultAxesTitleFontSizeMultiplier', 1.15);
end
tl = tiledlayout(fig, rows, cols, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:k
    nm = metrics(i);
    R = S.Q3.(nm);
    srow = spec(spec.name == nm, :);
    if isempty(srow)
        label = nm; unit = ""; d = 0;
    else
        label = srow.label(1); unit = srow.unit(1); d = srow.dir(1);
    end
    ax = nexttile(tl);
    fs = get(fig, 'DefaultAxesFontSize');
    panel(ax, R, label, unit, d, opts.max_brackets, fs);
    if opts.letters
        title(ax, sprintf('(%c) %s', char('a' + i - 1), label), 'FontSize', fs);
    else
        title(ax, label, 'FontSize', fs);
    end
end
end

% ---------------------------------------------------------------- one panel
function panel(ax, R, label, unit, d, max_brackets, fs)
cells = string(R.levels);
vals = R.means(:)';
n = R.n;
sem = R.sds(:)' / sqrt(max(n, 1));

lo = min(vals - sem, [], 'omitnan');
hi = max(vals + sem, [], 'omitnan');
span = max(hi - lo, eps);
% Clip only when zero is outside the data and the band is narrow next to its
% own distance from zero -- otherwise every bar would look identical.
clipped = ~(lo <= 0 && hi >= 0) && span < 0.25 * max(abs([lo hi]));
if clipped, base = lo - 0.55 * span; else, base = 0; end

hold(ax, 'on');
for j = 1:numel(vals)
    bar(ax, j, vals(j), 0.62, 'FaceColor', studyplot.color(cells(j)), ...
        'EdgeColor', 'none', 'BaseValue', base);
    errorbar(ax, j, vals(j), sem(j), 'k', 'LineWidth', 1.0, 'CapSize', 6);
end

% Significance brackets. They must clear the drawing, which for bars grown
% downward from zero (an all-negative metric) is the zero line, not the bars.
ytop = max([hi, base]);
top = ytop;
sig = R.pairs(R.pairs.significant, :);
if ~isempty(sig)
    ia = arrayfun(@(s) find(cells == s, 1), sig.a);
    ib = arrayfun(@(s) find(cells == s, 1), sig.b);
    x1 = min(ia, ib); x2 = max(ia, ib);
    % Strongest evidence first, then narrowest span: what gets dropped when
    % there are more significant pairs than room is the weakest, widest one.
    [~, ord] = sortrows([sig.p_holm, x2 - x1]);
    sig = sig(ord, :); x1 = x1(ord); x2 = x2(ord);
    nb = min(height(sig), max_brackets);
    step = 0.16 * span;
    used = [];   % rightmost x already occupied on each stacked level
    for j = 1:nb
        % Clearance is set by the centred p-label, which is wider than a short
        % bracket, not by the bracket span itself.
        lvl = find(x1(j) > used + 1.3, 1);
        if isempty(lvl), used(end + 1) = x2(j); lvl = numel(used); %#ok<AGROW>
        else,            used(lvl) = x2(j);
        end
        y = ytop + step * lvl;
        plot(ax, [x1(j) x1(j) x2(j) x2(j)], [y - 0.22 * step, y, y, y - 0.22 * step], ...
             'k-', 'LineWidth', 0.8);
        text(ax, (x1(j) + x2(j)) / 2, y + 0.02 * step, fmt_p(sig.p_holm(j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fs - 2);
        top = max(top, y + 0.9 * step);
    end
end
hold(ax, 'off');

% Axis extent must hold the bar baseline, the whiskers and the brackets, for
% data of either sign (a signed margin metric is negative in every cell).
ylo = min([base, lo]);
yhi = max([base, top]);
rngy = max(yhi - ylo, eps);
ylo = ylo - 0.02 * rngy; yhi = yhi + 0.06 * rngy;

xline(ax, 3.5, '-', 'Color', [0.75 0.75 0.75], 'LineWidth', 0.8, 'HandleVisibility', 'off');
ylim(ax, [ylo, yhi]);
xlim(ax, [0.4, numel(vals) + 0.6]);
xticks(ax, 1:numel(vals)); xticklabels(ax, cellstr(cells));
text(ax, 2, ylo, 'CLUTCH', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
     'FontSize', fs - 2, 'Color', studyplot.color("C"), 'FontWeight', 'bold');
text(ax, 5, ylo, 'JOYSTICK', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
     'FontSize', fs - 2, 'Color', studyplot.color("J"), 'FontWeight', 'bold');
ax.XRuler.TickLabelGapOffset = 12;

yl = label;
if strlength(unit) > 0 && unit ~= "-", yl = label + " [" + unit + "]"; end
ylabel(ax, yl, 'FontSize', fs - 1);
grid(ax, 'on'); ax.YGrid = 'on'; ax.XGrid = 'off';

sub = studyplot.dirtext(d) + sprintf(" | n=%d", n);
if clipped, sub = sub + " | clipped axis"; end
subtitle(ax, sub, 'FontSize', fs - 2, 'Color', [0.35 0.35 0.35]);
end
