function fig = fig_thesis_panel(S, spec, metric, outfile, opts)
%FIG_THESIS_PANEL One metric, drawn for inclusion in the thesis.
%   FIG_THESIS_PANEL(S, SPEC, METRIC, OUTFILE) draws the condition means of
%   METRIC with their standard error and the Holm-significant pairs, and writes
%   OUTFILE as vector PDF.
%
%   Differences from FIG_PAPER_PANELS, all required by the thesis style: no
%   title, no subtitle and no y-axis label (the letter, the quantity and its
%   unit live in the LaTeX caption instead), a figure sized in centimetres so
%   that it is included at its natural size and never rescaled, and every
%   string typeset by the LaTeX interpreter so the panel face matches the
%   Latin Modern body text of the document.
%
%   Options: source ("Q3"), width_cm, height_cm, fontsize, max_brackets,
%   cell_colors (label the bars by level but colour them by cell, which is what
%   a within-one-mode panel needs).

arguments
    S struct
    spec table
    metric (1,1) string
    outfile (1,1) string
    opts.source (1,1) string = "Q3"
    opts.width_cm (1,1) double = 7.4
    opts.height_cm (1,1) double = 4.8
    opts.fontsize (1,1) double = 7.5
    opts.max_brackets (1,1) double = 4
    opts.cell_colors (1,1) string = ""     % "C" or "J": colour F/B/FB as CF/CB/CFB
    opts.ylab (1,1) string = ""            % LaTeX symbol and unit, e.g. "$T$ [s]"
end

R = S.(opts.source).(metric);
cells = string(R.levels);
if strlength(opts.cell_colors) > 0
    cells = opts.cell_colors + cells;      % F -> CF, so colour and label agree
end
vals = R.means(:)';
n = R.n;
sem = R.sds(:)' / sqrt(max(n, 1));

fs = opts.fontsize;
fig = figure('Color', 'w', 'Units', 'centimeters', ...
             'Position', [2 2 opts.width_cm opts.height_cm], ...
             'DefaultTextInterpreter', 'latex', 'DefaultAxesTickLabelInterpreter', 'latex', ...
             'DefaultLegendInterpreter', 'latex', ...
             'DefaultAxesFontSize', fs, 'DefaultTextFontSize', fs);
ax = axes(fig, 'Units', 'normalized', 'Position', [0.165 0.145 0.815 0.825]);
set(ax, 'TickDir', 'out', 'TickLength', [0.015 0.015], 'Box', 'off', ...
        'LineWidth', 0.6, 'XColor', [0.25 0.25 0.25], 'YColor', [0.25 0.25 0.25], ...
        'FontSize', fs);

lo = min(vals - sem, [], 'omitnan');
hi = max(vals + sem, [], 'omitnan');
rawspan = hi - lo;
flat = ~(rawspan > 0);
if flat, span = max(0.1 * max(abs([lo hi])), 0.1); else, span = rawspan; end
clipped = ~flat && ~(lo <= 0 && hi >= 0) && (hi < 0 || rawspan < 0.25 * max(abs([lo hi])));
if clipped, base = lo - 0.55 * span; else, base = 0; end

hold(ax, 'on');
for j = 1:numel(vals)
    face = studyplot.color(cells(j));
    bar(ax, j, vals(j), 0.70, 'FaceColor', face, 'EdgeColor', face * 0.72, ...
        'LineWidth', 0.5, 'BaseValue', base, 'ShowBaseLine', 'off');
    errorbar(ax, j, vals(j), sem(j), 'k', 'LineWidth', 0.7, 'CapSize', 4);
end

ytop = max([hi, base]);
if base <= lo
    ylo = base;
else
    ylo = lo - 0.04 * span;
    yline(ax, base, '-', 'Color', [0.55 0.55 0.55], 'LineWidth', 0.5, 'HandleVisibility', 'off');
end

if ~isempty(R.pairs) && ismember('significant', R.pairs.Properties.VariableNames)
    sig = R.pairs(R.pairs.significant, :);
else
    sig = table();
end
nb = 0; lvls = []; x1 = []; x2 = []; nlvl = 0;
if ~isempty(sig)
    raw = string(R.levels);
    ia = arrayfun(@(s) find(raw == s, 1), sig.a);
    ib = arrayfun(@(s) find(raw == s, 1), sig.b);
    x1 = min(ia, ib); x2 = max(ia, ib);
    [~, ord] = sortrows([sig.p_holm, x2 - x1]);
    sig = sig(ord, :); x1 = x1(ord); x2 = x2(ord);
    nb = min(height(sig), opts.max_brackets);
    used = []; lvls = zeros(1, nb);
    for j = 1:nb
        lvl = find(x1(j) > used + 1.7, 1);
        if isempty(lvl), used(end + 1) = x2(j); lvl = numel(used); %#ok<AGROW>
        else,            used(lvl) = x2(j);
        end
        lvls(j) = lvl;
    end
    nlvl = numel(used);
end

% Same font-relative spacing rule as the paper panels, in this figure's own
% pixel height: a p-label needs a font height, not a fraction of the data.
step = 0.20 * span;
if nlvl >= 1 && ytop - ylo > 0
    drawnow;
    try
        p = getpixelposition(ax); H = p(4);
    catch
        H = 0;
    end
    s_px = 1.25 * fs + 4.5;   % label height plus clearance; below this the labels touch
    r = s_px / max(H, eps);
    if H > 0 && r * (nlvl + 0.92) < 0.75
        step = (r * (ytop - ylo)) / (1 - r * (nlvl + 0.92));
    end
end
for j = 1:nb
    y = ytop + step * lvls(j);
    plot(ax, [x1(j) x1(j) x2(j) x2(j)], [y - 0.18 * step, y, y, y - 0.18 * step], ...
         'k-', 'LineWidth', 0.6);
    text(ax, (x1(j) + x2(j)) / 2, y + 0.06 * step, "$" + fmt_p(sig.p_holm(j)) + "$", ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', fs - 1.5, 'BackgroundColor', 'w', 'Margin', 0.5);
end
hold(ax, 'off');

if nlvl > 0, yhi = ytop + (nlvl + 0.92) * step; else, yhi = ytop + 0.06 * max(span, eps); end

if isfield(R, 'modes'), modes = string(R.modes); else, modes = extractBefore(cells, 2); end
isJ = modes(:)' == "J";
bnd = find(diff(isJ) ~= 0, 1);
if ~isempty(bnd)
    xline(ax, bnd + 0.5, '-', 'Color', [0.75 0.75 0.75], 'LineWidth', 0.6, 'HandleVisibility', 'off');
end
ylim(ax, [ylo, yhi]);
xlim(ax, [0.4, numel(vals) + 0.6]);
xticks(ax, 1:numel(vals)); xticklabels(ax, cellstr(cells));
ax.XRuler.TickLabelGapOffset = 1;
grid(ax, 'off');
% Symbol and unit only: the quantity is named in the running text, not here.
if strlength(opts.ylab) > 0
    ylabel(ax, opts.ylab, 'Interpreter', 'latex', 'FontSize', fs + 1);
end

exportgraphics(fig, outfile, 'ContentType', 'vector', 'BackgroundColor', 'white');
if nargout == 0, close(fig); clear fig; end
end
