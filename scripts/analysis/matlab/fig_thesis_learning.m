function fig_thesis_learning(S, items, trial, out_dir)
%FIG_THESIS_LEARNING The two practice/order panels, drawn for the thesis.
%   Writes res_learn_slot.pdf and res_learn_order.pdf into OUT_DIR.
%
%   Only three series are drawn on the slot panel -- the composite and the two
%   families that actually drift -- because a line per family is unreadable at
%   print size and buries the one comparison that matters. Series are separated
%   by marker and line style as well as colour, so the panel survives monochrome
%   printing.

fs = 7.5;
W = 7.4; H = 4.8;

    function [fig, ax] = newpanel()
        fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 W H], ...
                     'DefaultTextInterpreter', 'none', 'DefaultAxesTickLabelInterpreter', 'none', ...
                     'DefaultAxesFontName', 'Helvetica', 'DefaultTextFontName', 'Helvetica', ...
                     'DefaultAxesFontSize', fs, 'DefaultTextFontSize', fs);
        ax = axes(fig, 'Units', 'normalized', 'Position', [0.165 0.175 0.815 0.795]);
        set(ax, 'TickDir', 'out', 'TickLength', [0.012 0.012], 'Box', 'off', ...
                'LineWidth', 0.6, 'XColor', [0.25 0.25 0.25], 'YColor', [0.25 0.25 0.25], ...
                'FontSize', fs);
    end

% ---------------------------------------------------------------- slot panel
series = ["composite" "fam_time_effectiveness" "fam_human_effort"];
labels = ["Composite" "Time & effectiveness" "Human effort"];
styles = {'-', '--', ':'};
marks  = {'o', 's', '^'};
cols   = [0 0 0; studyplot.color("C"); studyplot.color("J")];

[fig, ax] = newpanel();
hold(ax, 'on');
for i = 1:numel(series)
    if ~isfield(S.Q6, series(i)), continue; end
    R = S.Q6.(series(i));
    Sm = R.slot_matrix;
    mu = mean(Sm, 1, 'omitnan');
    se = std(Sm, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(Sm), 1));
    errorbar(ax, 1:numel(mu), mu, se, [styles{i} marks{i}], 'Color', cols(i, :), ...
             'MarkerFaceColor', cols(i, :), 'MarkerSize', 3.2, 'LineWidth', 1.1, ...
             'CapSize', 3, 'DisplayName', labels(i));
end
yline(ax, 0, '-', 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5, 'HandleVisibility', 'off');
hold(ax, 'off');
xlim(ax, [0.7 6.3]); xticks(ax, 1:6);
% The one axis in this set whose ticks are not self-explanatory.
xlabel(ax, 'experiment slot', 'FontSize', fs);
ylabel(ax, '$z$', 'Interpreter', 'latex', 'FontSize', fs + 1);
grid(ax, 'off');
lg = legend(ax, 'Location', 'southwest', 'Box', 'off', 'FontSize', fs - 1.5);
lg.ItemTokenSize = [12 8];
exportgraphics(fig, fullfile(out_dir, 'res_learn_slot.pdf'), 'ContentType', 'vector', ...
               'BackgroundColor', 'white');
close(fig);

end
