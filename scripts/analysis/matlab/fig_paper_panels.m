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
    % Which result set the bars come from. "Q3" is the six-cell comparison; the
    % within-mode sets are how a metric that is NOT comparable across modes (a
    % clutch tether force and a joystick centring spring are different physical
    % quantities) can still be shown honestly -- three bars inside one mode.
    opts.source (1,1) string = "Q3"
end

if ~isfield(S, opts.source)
    error('fig_paper_panels:noSource', 'Result set "%s" is not in S.', opts.source);
end
metrics = metrics(arrayfun(@(m) isfield(S.(opts.source), m), metrics));
k = numel(metrics);
if k == 0
    error('fig_paper_panels:noMetrics', 'None of the requested metrics has a %s result.', opts.source);
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

fullspec = table();
for i = 1:k
    nm = metrics(i);
    R = S.(opts.source).(nm);
    srow = spec(spec.name == nm, :);
    if isempty(srow)
        % ITEMS drops the family-less derived metrics, so that a derived
        % second-count is never double-counted inside its own family score --
        % but its label, unit and direction still live in the catalogue.
        if isempty(fullspec), fullspec = study_metric_spec(); end
        srow = fullspec(fullspec.name == nm, :);
    end
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
rawspan = hi - lo;
% A metric that is identical in every condition (e.g. an outcome that always
% happened) has no spread to scale by; give it a nominal one and draw from zero
% so the equal bars read as equal instead of as a degenerate axis.
flat = ~(rawspan > 0);
if flat, span = max(0.1 * max(abs([lo hi])), 0.1); else, span = rawspan; end
% Clip when zero is outside the data and either the whole metric is negative
% (bars from zero would hang downward and read as inverted against the other
% panels) or the band is narrow next to its own distance from zero (bars from
% zero would look identical). Clipping is always labelled under the title.
clipped = ~flat && ~(lo <= 0 && hi >= 0) && (hi < 0 || rawspan < 0.25 * max(abs([lo hi])));
if clipped, base = lo - 0.55 * span; else, base = 0; end

hold(ax, 'on');
for j = 1:numel(vals)
    face = studyplot.color(cells(j));
    % A pale fill needs an outline to read as a bar on white paper. The bar's
    % own baseline is off: the axis floor below is already that line, and two
    % of them a few pixels apart read as a drawing error.
    bar(ax, j, vals(j), 0.62, 'FaceColor', face, 'EdgeColor', face * 0.72, ...
        'LineWidth', 0.5, 'BaseValue', base, 'ShowBaseLine', 'off');
    errorbar(ax, j, vals(j), sem(j), 'k', 'LineWidth', 1.0, 'CapSize', 6);
end

% Significance brackets. They must clear the drawing, which for bars grown
% downward from zero (an all-negative metric) is the zero line, not the bars.
ytop = max([hi, base]);
% Bars grow from BASE, so the axis floor must BE the baseline; padding under it
% would hang every bar off a second line. Only data that crosses the base needs
% room below, and then the base is drawn explicitly as the sign reference.
if base <= lo
    ylo = base;
else
    ylo = lo - 0.04 * span;
    yline(ax, base, '-', 'Color', [0.55 0.55 0.55], 'LineWidth', 0.6, 'HandleVisibility', 'off');
end

% A metric the omnibus test skipped (constant, or too few participants) comes
% back with an empty pairs table that has no columns at all.
if ~isempty(R.pairs) && ismember('significant', R.pairs.Properties.VariableNames)
    sig = R.pairs(R.pairs.significant, :);
else
    sig = table();
end
nb = 0; lvls = []; x1 = []; x2 = []; nlvl = 0;
if ~isempty(sig)
    ia = arrayfun(@(s) find(cells == s, 1), sig.a);
    ib = arrayfun(@(s) find(cells == s, 1), sig.b);
    x1 = min(ia, ib); x2 = max(ia, ib);
    % Strongest evidence first, then narrowest span: what gets dropped when
    % there are more significant pairs than room is the weakest, widest one.
    [~, ord] = sortrows([sig.p_holm, x2 - x1]);
    sig = sig(ord, :); x1 = x1(ord); x2 = x2(ord);
    nb = min(height(sig), max_brackets);
    used = [];   % rightmost x already occupied on each stacked level
    lvls = zeros(1, nb);
    for j = 1:nb
        % Clearance is set by the centred p-label, which is wider than a short
        % bracket, not by the bracket span itself.
        lvl = find(x1(j) > used + 1.7, 1);
        if isempty(lvl), used(end + 1) = x2(j); lvl = numel(used); %#ok<AGROW>
        else,            used(lvl) = x2(j);
        end
        lvls(j) = lvl;
    end
    nlvl = numel(used);
end

% Levels are packed before anything is drawn because the spacing depends on how
% many there are: see bracket_step.
step = bracket_step(ax, ytop - ylo, nlvl, span, fs);
for j = 1:nb
    y = ytop + step * lvls(j);
    plot(ax, [x1(j) x1(j) x2(j) x2(j)], [y - 0.18 * step, y, y, y - 0.18 * step], ...
         'k-', 'LineWidth', 0.8);
    text(ax, (x1(j) + x2(j)) / 2, y + 0.06 * step, fmt_p(sig.p_holm(j)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fs - 2, ...
         'BackgroundColor', 'w', 'Margin', 0.5);   % stays legible where brackets stack
end
hold(ax, 'off');

if nlvl > 0, yhi = ytop + (nlvl + 0.92) * step; else, yhi = ytop + 0.06 * max(span, eps); end

% Mode split is read from the result, so a 4-cell (blending-only) metric is
% grouped and divided in the same way as the full six.
if isfield(R, 'modes'), modes = string(R.modes); else, modes = extractBefore(cells, 2); end
isJ = modes(:)' == "J";
bnd = find(diff(isJ) ~= 0, 1);
if ~isempty(bnd)
    xline(ax, bnd + 0.5, '-', 'Color', [0.75 0.75 0.75], 'LineWidth', 0.8, 'HandleVisibility', 'off');
end
ylim(ax, [ylo, yhi]);
xlim(ax, [0.4, numel(vals) + 0.6]);
xticks(ax, 1:numel(vals)); xticklabels(ax, cellstr(cells));
ax.XRuler.TickLabelGapOffset = 2;

yl = label;
if strlength(unit) > 0 && unit ~= "-", yl = label + " [" + unit + "]"; end
ylabel(ax, yl, 'FontSize', fs - 1);
grid(ax, 'on'); ax.YGrid = 'on'; ax.XGrid = 'off';

sub = studyplot.dirtext(d) + sprintf(" | n=%d", n);
if clipped, sub = sub + " | clipped axis"; end
if flat, sub = sub + " | identical in every condition"; end
subtitle(ax, sub, 'FontSize', fs - 2, 'Color', [0.35 0.35 0.35]);
end

% ------------------------------------------------------- bracket spacing
function step = bracket_step(ax, D, nlvl, span, fs)
%BRACKET_STEP Data-unit gap between stacked significance brackets.
%   The room one bracket and its p-label need is a FONT height, so a fixed
%   fraction of the data span under-spaces a tall panel and over-spaces a short
%   one -- labels then touch the bracket above. Solve instead for the step that
%   renders as S_PX pixels AFTER the stack has grown the axis itself:
%   step = (S_PX/H)*D / (1 - (S_PX/H)*(nlvl+0.92)), with D the drawing height.
step = 0.20 * span;                       % fallback when the pixel height is unknown
if nlvl < 1 || ~(D > 0), return, end
s_px = 1.6 * max(fs - 2, 5) + 6;          % label height + clearance
H = 0;
try
    drawnow;
    p = getpixelposition(ax);
    H = p(4);
catch
    H = 0;
end
r = s_px / max(H, eps);
if H > 0 && r * (nlvl + 0.92) < 0.75      % leave the data at least a quarter of the axis
    step = (r * D) / (1 - r * (nlvl + 0.92));
end
end
