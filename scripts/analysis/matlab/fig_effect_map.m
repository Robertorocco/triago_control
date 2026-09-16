function fig = fig_effect_map(S, spec, metrics, alpha, opts)
%FIG_EFFECT_MAP  Which factor moves which metric, as bars instead of a table.
%   FIG = FIG_EFFECT_MAP(S, SPEC, METRICS, ALPHA) draws one row per metric and
%   three columns -- control Mode, Assistance, their Interaction -- with a bar
%   whose length is the partial eta squared of that factor in the two-way
%   repeated-measures ANOVA (the share of the metric's variation that factor
%   explains, 0 to 1). A filled bar is significant at ALPHA; a hollow bar is
%   not. The number is printed at the bar end so nothing has to be read off
%   the axis.
%
%   Options: name ("effect map"), row_h (34 px), compact (false).

arguments
    S struct
    spec table
    metrics (1,:) string
    alpha (1,1) double = 0.05
    opts.name (1,1) string = "effect map"
    opts.row_h (1,1) double = 34
    opts.compact (1,1) logical = false
end

metrics = metrics(arrayfun(@(m) isfield(S.Q3, m) && ~isempty(S.Q3.(m).effects), metrics));
n = numel(metrics);
if n == 0
    error('fig_effect_map:noMetrics', 'None of the requested metrics has ANOVA effects.');
end
factors = ["Mode" "Assist" "Mode:Assist"];
heads = ["Control mode" "Assistance" "Interaction"];

[w, h] = studyplot.size_for(opts.compact, 700, 130 + opts.row_h * n);
fig = studyplot.profig(opts.name, w, h);
% Outer-position constraint so long metric labels and the title always fit,
% also when the Live Editor rescales the figure to its column width.
ax = axes(fig, 'OuterPosition', [0 0 1 1], 'PositionConstraint', 'outerposition');
hold(ax, 'on');

colw = 1.0; gap = 0.08; barmax = colw - gap - 0.22;   % room for the number
sig_c = [0.20 0.45 0.75]; ns_c = [0.86 0.86 0.86];
labels = strings(n, 1);
for i = 1:n
    nm = metrics(i);
    srow = spec(spec.name == nm, :);
    labels(i) = nm; if ~isempty(srow), labels(i) = srow.label(1); end
    E = S.Q3.(nm).effects;
    y = n - i + 1;
    for j = 1:3
        e = E(string(E.effect) == factors(j), :);
        if isempty(e), continue; end
        eta = max(min(e.eta2p, 1), 0);
        p = e.p_gg;
        x0 = (j - 1) * colw + gap / 2;
        len = barmax * eta;
        if p < alpha
            rectangle(ax, 'Position', [x0, y - 0.34, max(len, 0.005), 0.68], ...
                      'FaceColor', sig_c, 'EdgeColor', 'none');
        else
            rectangle(ax, 'Position', [x0, y - 0.34, max(len, 0.005), 0.68], ...
                      'FaceColor', 'none', 'EdgeColor', [0.6 0.6 0.6], 'LineWidth', 0.8);
        end
        text(ax, x0 + len + 0.02, y, sprintf('%.2f%s', eta, stars(p)), ...
             'VerticalAlignment', 'middle', 'FontSize', 9, ...
             'Color', ternary(p < alpha, [0.1 0.1 0.1], [0.5 0.5 0.5]));
    end
end
% faint column separators and the effect-size landmarks
for j = 1:2
    xline(ax, j * colw, '-', 'Color', [0.85 0.85 0.85]);
end
for j = 1:3
    x0 = (j - 1) * colw + gap / 2;
    for lm = [0.06 0.14]     % medium / large by the usual convention
        plot(ax, [x0 x0] + barmax * lm, [0.45 n + 0.55], ':', 'Color', [0.75 0.75 0.75], 'LineWidth', 0.6);
    end
    text(ax, x0 + barmax / 2, n + 0.9, heads(j), 'HorizontalAlignment', 'center', ...
         'FontWeight', 'bold', 'FontSize', 10);
end
hold(ax, 'off');
xlim(ax, [0 3 * colw]); ylim(ax, [0.4 n + 1.3]);
yticks(ax, 1:n); yticklabels(ax, cellstr(flipud(labels)));
xticks(ax, []); ax.XAxis.Visible = 'off'; ax.YAxis.TickLength = [0 0];
ax.YColor = [0.15 0.15 0.15]; box(ax, 'off'); grid(ax, 'off');
title(ax, 'How much of each metric each factor explains', 'FontSize', 11);
subtitle(ax, {sprintf('bar = partial eta squared (0 to 1); filled = significant at p < %.2f, hollow = not', alpha), ...
              'dotted marks = medium (0.06) and large (0.14) effect; * p < 0.05, ** p < 0.01, *** p < 0.001'}, ...
         'FontSize', 8, 'Color', [0.35 0.35 0.35]);
end

function s = stars(p)
if isnan(p),        s = "";
elseif p < 0.001,   s = " ***";
elseif p < 0.01,    s = " **";
elseif p < 0.05,    s = " *";
else,               s = "";
end
end

function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end
