classdef studyplot
%STUDYPLOT Shared drawing primitives for the study figures (static methods).
%   Every figure of the group analysis and of the report is drawn with these
%   helpers so that colours, annotations and reading conventions are the same
%   everywhere: participants are thin grey lines/dots, condition means are
%   thick markers with a 95% CI, and the title states what is plotted, n, the
%   direction ("lower is better") and the test result.

methods (Static)

    function c = color(name)
        % Palette: control modes, assistance levels, worlds, and the 6 cells.
        switch string(name)
            case "C",      c = [0.20 0.42 0.68];
            case "J",      c = [0.85 0.45 0.20];
            case "F",      c = [0.55 0.70 0.35];
            case "B",      c = [0.25 0.60 0.55];
            case "FB",     c = [0.15 0.40 0.50];
            case "rack",   c = [0.20 0.42 0.68];
            case "shield", c = [0.85 0.45 0.20];
            case "CF",     c = [0.55 0.70 0.90];
            case "CB",     c = [0.35 0.55 0.80];
            case "CFB",    c = [0.15 0.35 0.65];
            case "JF",     c = [0.95 0.72 0.50];
            case "JB",     c = [0.90 0.55 0.30];
            case "JFB",    c = [0.75 0.35 0.10];
            case "grey",   c = [0.55 0.55 0.55];
            otherwise,     c = [0.40 0.40 0.40];
        end
    end

    function s = dirtext(d)
        if d > 0,     s = "higher is better";
        elseif d < 0, s = "lower is better";
        else,         s = "no better/worse direction";
        end
    end

    function s = levelname(code)
        switch string(code)
            case "C",  s = "Clutch";
            case "J",  s = "Joystick";
            case "F",  s = "Feedback (F)";
            case "B",  s = "Blending (B)";
            case "FB", s = "Feedback+Blending (FB)";
            otherwise, s = string(code);
        end
    end

    function paired(ax, M, levels, opts)
        % Participants as grey dots joined by lines across the levels, level
        % means as thick markers with 95% CI (t-based).
        arguments
            ax
            M (:,:) double
            levels (1,:) string
            opts.ylabel (1,1) string = ""
            opts.colors = []
            opts.horizontal (1,1) logical = false   % levels on the y-axis (compact layout)
        end
        k = numel(levels); n = size(M, 1);
        hold(ax, 'on');
        rng(7, 'twister'); jit = (rand(n, 1) - 0.5) * 0.12;
        h = opts.horizontal;
        for i = 1:n
            xy = {(1:k) + jit(i), M(i, :)}; if h, xy = xy([2 1]); end
            plot(ax, xy{:}, '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
            plot(ax, xy{:}, 'o', 'MarkerSize', 4 - h, ...
                 'MarkerFaceColor', [0.75 0.75 0.75], 'MarkerEdgeColor', [0.5 0.5 0.5]);
        end
        for j = 1:k
            v = M(:, j); v = v(~isnan(v)); m = mean(v);
            ci = 0;
            if numel(v) > 1, ci = tinv(0.975, numel(v) - 1) * std(v) / sqrt(numel(v)); end
            if isempty(opts.colors), col = studyplot.color(levels(j)); else, col = opts.colors(j, :); end
            if h
                errorbar(ax, m, j + 0.28, ci, 'horizontal', 'o', 'Color', col, 'MarkerFaceColor', col, ...
                         'MarkerSize', 6, 'LineWidth', 1.5, 'CapSize', 6);
            else
                errorbar(ax, j + 0.28, m, ci, 'o', 'Color', col, 'MarkerFaceColor', col, ...
                         'MarkerSize', 8, 'LineWidth', 1.8, 'CapSize', 8);
            end
        end
        hold(ax, 'off');
        names = arrayfun(@studyplot.levelname, levels);
        if h
            ylim(ax, [0.5 k + 0.6]); yticks(ax, 1:k); yticklabels(ax, names);
            xlabel(ax, opts.ylabel); ax.YDir = 'reverse';
        else
            xlim(ax, [0.5 k + 0.6]); xticks(ax, 1:k); xticklabels(ax, names);
            ylabel(ax, opts.ylabel);
        end
        grid(ax, 'on'); box(ax, 'on');
    end

    function heat23(ax, means, cis, rows, cols, opts)
        % 2 x 3 (or 2 x 2) grid of condition means, colour = mean, text = mean +/- CI.
        arguments
            ax
            means (:,:) double
            cis (:,:) double
            rows (1,:) string
            cols (1,:) string
            opts.dir (1,1) double = 0
            opts.fmt (1,1) string = "%.3g"
            opts.best (1,1) string = ""
            opts.sig = []      % logical matrix, cells significantly different from another
            opts.bestmark (1,1) string = "  (best)"
        end
        oriented = means * (opts.dir + (opts.dir == 0));
        imagesc(ax, oriented);
        colormap(ax, studyplot.cool2warm(64));
        c = max(abs(oriented(:) - mean(oriented(:), 'omitnan')), [], 'omitnan');
        if isempty(c) || c == 0 || isnan(c), c = 1; end
        clim(ax, mean(oriented(:), 'omitnan') + [-c c]);
        for r = 1:numel(rows)
            for k = 1:numel(cols)
                txt = sprintf(opts.fmt, means(r, k));
                if ~isnan(cis(r, k)), txt = txt + sprintf(" ± " + opts.fmt, cis(r, k)); end
                lab = rows(r) + cols(k);
                if lab == opts.best, txt = txt + opts.bestmark; end
                text(ax, k, r, txt, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
            end
        end
        xticks(ax, 1:numel(cols)); xticklabels(ax, cols);
        yticks(ax, 1:numel(rows)); yticklabels(ax, arrayfun(@studyplot.levelname, rows));
        ax.YDir = 'normal';
        cb = colorbar(ax); cb.Ticks = [];
        cb.Label.String = ternary(opts.dir ~= 0, "worse -> better", "low -> high");
    end

    function learning(ax, S, opts)
        % Per-participant thin lines over slots + mean +/- SD.
        arguments
            ax
            S (:,:) double
            opts.ylabel (1,1) string = ""
            opts.xlabel (1,1) string = "experiment slot (1 = first condition met)"
        end
        [n, k] = size(S);
        hold(ax, 'on');
        for i = 1:n
            plot(ax, 1:k, S(i, :), '-', 'Color', [0.75 0.75 0.75], 'LineWidth', 0.7);
        end
        m = mean(S, 1, 'omitnan'); sd = std(S, 0, 1, 'omitnan');
        errorbar(ax, 1:k, m, sd, '-o', 'Color', [0.15 0.15 0.15], 'LineWidth', 2, ...
                 'MarkerFaceColor', [0.15 0.15 0.15], 'MarkerSize', 6, 'CapSize', 6);
        hold(ax, 'off');
        xlim(ax, [0.5 k + 0.5]); xticks(ax, 1:k);
        xlabel(ax, opts.xlabel); ylabel(ax, opts.ylabel); grid(ax, 'on'); box(ax, 'on');
    end

    function bars(ax, vals, lo, hi, levels, opts)
        % Bars with asymmetric CI whiskers, one colour per level.
        arguments
            ax
            vals (1,:) double
            lo (1,:) double
            hi (1,:) double
            levels (1,:) string
            opts.ylabel (1,1) string = ""
            opts.labels (1,:) string = strings(1, 0)
        end
        k = numel(vals);
        hold(ax, 'on');
        for j = 1:k
            bar(ax, j, vals(j), 0.6, 'FaceColor', studyplot.color(levels(j)), 'EdgeColor', 'none');
            if ~isnan(lo(j)), errorbar(ax, j, vals(j), vals(j) - lo(j), hi(j) - vals(j), 'k', 'LineWidth', 1.2, 'CapSize', 8); end
        end
        hold(ax, 'off');
        xlim(ax, [0.4 k + 0.6]); xticks(ax, 1:k);
        if isempty(opts.labels), xticklabels(ax, arrayfun(@studyplot.levelname, levels)); else, xticklabels(ax, opts.labels); end
        ylabel(ax, opts.ylabel); grid(ax, 'on'); box(ax, 'on');
    end

    function sigbracket(ax, x1, x2, y, txt)
        hold(ax, 'on');
        plot(ax, [x1 x1 x2 x2], [y - 0.01 y y y - 0.01] .* [1 1 1 1], 'k-', 'LineWidth', 1);
        text(ax, (x1 + x2) / 2, y, txt, 'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom', 'FontSize', 8);
        hold(ax, 'off');
    end

    function s = stars(p)
        if isnan(p), s = "";
        elseif p < 0.001, s = "***";
        elseif p < 0.01, s = "**";
        elseif p < 0.05, s = "*";
        else, s = "n.s.";
        end
    end

    function cm = cool2warm(n)
        % Blue (low) - white - red (high) diverging colormap.
        lo = [0.23 0.30 0.75]; mid = [0.97 0.97 0.97]; hi = [0.75 0.15 0.20];
        h = floor(n / 2);
        cm = [interp1([0 1], [lo; mid], linspace(0, 1, h)'); interp1([0 1], [mid; hi], linspace(0, 1, n - h)')];
    end

    function fig = newfig(name, w, h, compact)
        % Plain-text interpreter everywhere: metric names contain underscores.
        % Visibility is deliberately NOT set: the Live Editor only captures
        % figures whose visibility it controls itself (an explicit 'Visible'
        % argument, even 'on', makes the figure disappear from the report).
        % COMPACT figures are designed for the Live Editor canvas (~480 x 660
        % px): stacked panels and small fonts.
        if nargin < 4, compact = false; end
        fig = figure('Name', name, 'Color', 'w', 'Position', [60 60 w h], ...
                     'DefaultTextInterpreter', 'none', 'DefaultAxesTickLabelInterpreter', 'none');
        if compact
            set(fig, 'DefaultAxesFontSize', 7, 'DefaultTextFontSize', 7, ...
                     'DefaultAxesTitleFontSizeMultiplier', 1.1, 'DefaultLegendFontSize', 7);
        end
    end

    function [w, h] = size_for(compact, w, h)
        % Compact figures are always 480 px wide; height is capped at 660.
        if compact, w = 480; h = min(h, 660); end
    end

    function s = wrap(str, width)
        % Break a long title into lines of at most WIDTH characters.
        words = split(string(str), " ");
        lines = strings(0, 1); cur = "";
        for w = words'
            if strlength(cur) == 0, cur = w;
            elseif strlength(cur) + 1 + strlength(w) > width, lines(end + 1, 1) = cur; cur = w; %#ok<AGROW>
            else, cur = cur + " " + w;
            end
        end
        lines(end + 1, 1) = cur;
        s = strjoin(lines, newline);
    end

    function s = testname(code)
        % Internal test identifiers -> names used in figure titles.
        switch string(code)
            case "ttest",         s = "paired t-test";
            case "wilcoxon",      s = "Wilcoxon signed-rank";
            case "rm_anova",      s = "RM-ANOVA";
            case "friedman",      s = "Friedman";
            case "pitman_morgan", s = "Pitman-Morgan";
            otherwise,            s = string(code);
        end
    end

    function save(fig, out_dir, name)
        if strlength(string(out_dir)) == 0, return, end
        if ~isfolder(out_dir), mkdir(out_dir); end
        exportgraphics(fig, fullfile(out_dir, name + ".png"), 'Resolution', 140);
    end

    % ------------------------------------------------ publication style
    function fig = profig(name, w, h)
        % Publication-style figure: white, thin dark-grey axes, outward ticks,
        % light grid, no box, Helvetica, plain-text interpreters.
        fig = figure('Name', name, 'Color', 'w', 'Position', [60 60 w h], ...
            'DefaultTextInterpreter', 'none', 'DefaultAxesTickLabelInterpreter', 'none', ...
            'DefaultAxesFontName', 'Helvetica', 'DefaultTextFontName', 'Helvetica', ...
            'DefaultAxesFontSize', 10, 'DefaultTextFontSize', 10, ...
            'DefaultAxesTickDir', 'out', 'DefaultAxesTickLength', [0.006 0.006], ...
            'DefaultAxesBox', 'off', 'DefaultAxesLineWidth', 0.8, ...
            'DefaultAxesXColor', [0.25 0.25 0.25], 'DefaultAxesYColor', [0.25 0.25 0.25], ...
            'DefaultAxesGridColor', [0.88 0.88 0.88], 'DefaultAxesGridAlpha', 1, ...
            'DefaultAxesTitleFontWeight', 'bold', 'DefaultAxesTitleFontSizeMultiplier', 1.15, ...
            'DefaultLegendBox', 'off', 'DefaultLegendFontSize', 9);
    end

    function export(fig, base, opts)
        % PNG (200 dpi) + vector PDF next to each other; optional append to a
        % multi-page PDF (one document with every summary figure).
        arguments
            fig
            base (1,1) string
            opts.append_to (1,1) string = ""
        end
        exportgraphics(fig, base + ".png", 'Resolution', 200, 'BackgroundColor', 'white');
        exportgraphics(fig, base + ".pdf", 'ContentType', 'vector', 'BackgroundColor', 'white');
        if strlength(opts.append_to) > 0
            exportgraphics(fig, opts.append_to, 'ContentType', 'vector', 'BackgroundColor', 'white', 'Append', true);
        end
    end

    function c = sigcolor(p)
        % Evidence strength as a green ramp; grey = not significant; white = untested.
        if isnan(p),        c = [1 1 1];
        elseif p < 0.001,   c = [0.13 0.47 0.29];
        elseif p < 0.01,    c = [0.33 0.65 0.42];
        elseif p < 0.05,    c = [0.66 0.84 0.62];
        else,               c = [0.92 0.92 0.92];
        end
    end

    function c = textcolor_on(bg)
        % Black or white text depending on the background luminance.
        if 0.299 * bg(1) + 0.587 * bg(2) + 0.114 * bg(3) < 0.5, c = [1 1 1]; else, c = [0.1 0.1 0.1]; end
    end

    function s = famlabel(families, key)
        % Family label with the composite handled as a pseudo-family.
        if key == "composite", s = "Composite"; else, s = families.label(families.key == key); end
    end
end
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
