function files = fig_thesis_hw(mat_path, out_dir, opts)
%FIG_THESIS_HW Per-panel thesis figures for one offline trial bag (hardware or sim).
%   FILES = FIG_THESIS_HW(MAT_PATH, OUT_DIR) draws every telemetry panel the
%   .mat written by scripts/analysis/export_offline_bag.py can support and
%   writes each as OUT_DIR/hw_<key>.pdf (vector), one file per panel so LaTeX
%   can lay them out as lettered subfigures. Panels whose topic is absent from
%   the bag are skipped, so the same call serves a teleoperation bag.
%
%   Thesis style, as fig_thesis_panel: no title, no in-panel text, figure sized
%   in centimetres, Helvetica at print size, LaTeX symbol + unit as y label.
%   Right arm is red, left arm blue, throughout, but that R/L key is drawn
%   ONCE by the caller (shared legend ahead of the whole subfigure grid) --
%   individual panels carry no R/L legend of their own, only a panel-specific
%   one where something else needs explaining (e.g. dashed=raw in the
%   governor panels, forced top-right). The dashed vertical line is the end
%   of the open-loop reference (meta.t_off_s).
%
%   Options: panels (string array, default: all available), width_cm (7.4),
%   height_cm (3.9), fontsize (7.5), prefix ("hw_"), xlabel_on (string array of
%   keys that get the time-axis label; default: all), t_max ([] = bag length).
%
%   Panel keys: lambda_cbf, lambda_jl, margin, dmin, dsafe, wslack, slack,
%   gov_v, gov_w, gov_ep, gov_theta, q_R, q_L, qdot_R, qdot_L, qdotcmd_R,
%   qdotcmd_L, loop_freq.

arguments
    mat_path (1,1) string
    out_dir (1,1) string
    opts.panels string = string.empty
    opts.width_cm (1,1) double = 7.4
    opts.height_cm (1,1) double = 3.9
    opts.fontsize (1,1) double = 7.5
    opts.prefix (1,1) string = "hw_"
    opts.xlabel_on string = string.empty
    opts.t_max double = []
end

D = load(mat_path);
S = D.series;
if isfield(D, 'derived'), DV = D.derived; else, DV = struct(); end
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

t_off = D.meta.t_off_s;
if isempty(opts.t_max), t_max = D.meta.duration_s; else, t_max = opts.t_max; end

RED  = [0.753 0.224 0.169];
BLUE = [0.133 0.333 0.643];
TEAL = [0.090 0.541 0.478];
% Per-joint colours: Okabe-Ito (colour-blind safe, no yellow -- jet's J5 is
% unreadable on white in print), one colour per joint J1..J7.
JOINT = [0.000 0.447 0.698;   % J1 blue
         0.835 0.369 0.000;   % J2 vermillion
         0.000 0.620 0.451;   % J3 bluish green
         0.800 0.475 0.655;   % J4 reddish purple
         0.902 0.624 0.000;   % J5 orange
         0.337 0.706 0.914;   % J6 sky blue
         0.000 0.000 0.000];  % J7 black
fs = opts.fontsize;

% --- panel registry: key -> {needs, drawer}. A drawer takes (ax) and returns
% the legend handles+labels it wants (or empty). Adding a panel is one entry.
reg = containers.Map('KeyType', 'char', 'ValueType', 'any');
% Governor ceilings for the "where the cut is done" reference lines: gov_ep's
% is read off its own data (the governed trace visibly plateaus there once
% clamping engages -- robust, no config file needed); gov_theta's never
% clamps in this trial, so there is no plateau to read, and GOV_E_MAX_ORI =
% 1.0 rad is taken from the real-hw branch config instead (cross-checked
% against gov_ep's own empirical ceiling matching GOV_E_MAX_POS = 0.30 m on
% that same branch to 3 significant figures, i.e. those constants were the
% ones actually active for this capture).
if isfield(DV, 'gov')
    gov_ep_ceiling = max(DV.gov.pos_err_gov_r);
else
    gov_ep_ceiling = [];
end
gov_theta_ceiling = 1.0;

reg('lambda_cbf') = {"qp_debug_lambda_cbf",    @(ax) draw_rl(ax, S.qp_debug_lambda_cbf,    "$\lambda_{\mathrm{cbf}}$")};
reg('lambda_jl')  = {"qp_debug_lambda_joints", @(ax) draw_rl(ax, S.qp_debug_lambda_joints, "$\lambda_{\mathrm{jl}}$")};
reg('margin')     = {"qp_debug_safety_margin", @(ax) draw_rl(ax, S.qp_debug_safety_margin, "$h_{\mathrm{soft}}-d_{\mathrm{safe}}$ [m]", 'refline', 0)};
reg('dmin')       = {"qp_debug_min_distance",  @(ax) draw_one(ax, S.qp_debug_min_distance, "$d_{\min}$ [m]", TEAL, 'refline', 0)};
reg('dsafe')      = {"qp_debug_d_safe_dynamic", @(ax) draw_rl(ax, S.qp_debug_d_safe_dynamic, "$d_{\mathrm{safe}}$ [m]", 'refline', min(S.qp_debug_d_safe_dynamic.data(:)))};
reg('wslack')     = {"qp_debug_dynamic_weights", @(ax) draw_rl(ax, S.qp_debug_dynamic_weights, "$w_{\delta}$")};
reg('slack')      = {"qp_debug_slacks",        @(ax) draw_rl(ax, S.qp_debug_slacks,        "$\delta$")};
reg('loop_freq')  = {"qp_debug_loop_freq",     @(ax) draw_one(ax, S.qp_debug_loop_freq, "$f$ [Hz]", [0.3 0.3 0.3])};
reg('gov_v')      = {"derived.gov", @(ax) draw_gov(ax, DV, "lin_vel", "$\|v\|$ [m/s]")};
reg('gov_w')      = {"derived.gov", @(ax) draw_gov(ax, DV, "ang_vel", "$\|\omega\|$ [rad/s]")};
reg('gov_ep')     = {"derived.gov", @(ax) draw_gov(ax, DV, "pos_err", "$\|e_p\|$ [m]", 'refline', gov_ep_ceiling)};
reg('gov_theta')  = {"derived.gov", @(ax) draw_gov(ax, DV, "ori_err", "$\theta$ [rad]", 'refline', gov_theta_ceiling)};
reg('q_R')        = {"joint_states", @(ax) draw_joints(ax, S.joint_states.t, joint_cols(S.joint_states, "right", "position"), "$q$ [rad]")};
reg('q_L')        = {"joint_states", @(ax) draw_joints(ax, S.joint_states.t, joint_cols(S.joint_states, "left",  "position"), "$q$ [rad]")};
reg('qdot_R')     = {"qp_debug_qdot_measured", @(ax) draw_joints(ax, S.qp_debug_qdot_measured.t, S.qp_debug_qdot_measured.data(:, 1:7),  "$\dot{q}$ [rad/s]")};
reg('qdot_L')     = {"qp_debug_qdot_measured", @(ax) draw_joints(ax, S.qp_debug_qdot_measured.t, S.qp_debug_qdot_measured.data(:, 8:14), "$\dot{q}$ [rad/s]")};
reg('qdotcmd_R')  = {"qp_debug_qdot_cmd", @(ax) draw_joints(ax, S.qp_debug_qdot_cmd.t, S.qp_debug_qdot_cmd.data(:, 1:7),  "$\dot{q}^{\star}$ [rad/s]")};
reg('qdotcmd_L')  = {"qp_debug_qdot_cmd", @(ax) draw_joints(ax, S.qp_debug_qdot_cmd.t, S.qp_debug_qdot_cmd.data(:, 8:14), "$\dot{q}^{\star}$ [rad/s]")};

keys = string(reg.keys);
if ~isempty(opts.panels), keys = opts.panels; end
if isempty(opts.xlabel_on), xlab_keys = keys; else, xlab_keys = opts.xlabel_on; end

files = strings(0, 1);
for k = keys
    entry = reg(char(k));
    if ~available(entry{1}, S, DV)
        fprintf('[fig_thesis_hw] %s: source %s missing, skipped\n', k, entry{1});
        continue;
    end
    fig = figure('Color', 'w', 'Units', 'centimeters', 'Visible', 'off', ...
                 'Position', [2 2 opts.width_cm opts.height_cm], ...
                 'DefaultTextInterpreter', 'none', 'DefaultAxesTickLabelInterpreter', 'none', ...
                 'DefaultAxesFontName', 'Helvetica', 'DefaultTextFontName', 'Helvetica', ...
                 'DefaultAxesFontSize', fs, 'DefaultTextFontSize', fs);

    if k == "lambda_cbf"
        % A single ~10x taller sample dwarfs the rest of the trial on a linear
        % axis (557 vs a 150 ceiling everywhere else) -- the standard fix in
        % print is a broken axis, not clipping the peak away: two stacked axes
        % sharing x, a small gap between them, and the top one's own ticks
        % jumping straight to the next natural gridline (600) instead of
        % continuing linearly. Diagonal marks at the gap mark the break itself.
        draw_broken_lambda_cbf(fig, S.qp_debug_lambda_cbf, t_max, t_off, ismember(k, xlab_keys));
    else
        ax = axes(fig, 'Units', 'normalized', 'Position', [0.19 0.20 0.79 0.77]);
        set(ax, 'TickDir', 'out', 'TickLength', [0.015 0.015], 'Box', 'off', ...
                'LineWidth', 0.6, 'XColor', [0.25 0.25 0.25], 'YColor', [0.25 0.25 0.25], ...
                'FontSize', fs, 'NextPlot', 'add');
        grid(ax, 'on'); ax.GridAlpha = 0.18; ax.GridLineStyle = '-';

        [h, lbl, ncol] = entry{2}(ax);

        xlim(ax, [0, t_max]);
        if ~isnan(t_off)
            xline(ax, t_off, '--', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.7, 'HandleVisibility', 'off');
        end
        if ismember(k, xlab_keys)
            xlabel(ax, "$t$ [s]", 'Interpreter', 'latex', 'FontSize', fs + 1);
        end
        if ~isempty(h)
            if ncol >= 7,        loc = 'northoutside';
            elseif ncol == -1,   loc = 'northeast';   % forced top-right (e.g. the "raw" style key)
            else,                loc = 'best';
            end
            lg = legend(ax, h, lbl, 'Location', loc, 'Box', 'off', 'FontSize', fs - 0.5, ...
                        'NumColumns', max(ncol, 1), 'Interpreter', 'none');
            lg.ItemTokenSize = [12 8];
        end
        ax.XRuler.TickLabelGapOffset = 1;
    end

    outfile = fullfile(out_dir, opts.prefix + k + ".pdf");
    exportgraphics(fig, outfile, 'ContentType', 'vector', 'BackgroundColor', 'white');
    close(fig);
    files(end + 1, 1) = string(outfile); %#ok<AGROW>
    fprintf('[fig_thesis_hw] wrote %s\n', outfile);
end

% ----------------------------------------------------------------- drawers
    function [h, lbl, ncol] = draw_rl(ax, ser, ylab, varargin)
        % R/L colour is explained once by the shared legend ahead of the whole
        % figure (fig_thesis_hw draws no per-panel R/L legend) -- so h/lbl are
        % empty here unless 'yclip' asks for the off-scale-peak arrow below.
        p = inputParser;
        p.addParameter('refline', []);
        p.addParameter('yclip', []);    % clip the axis to this value...
        p.addParameter('peak', []);     % ...and annotate the true peak at the clip point
        p.parse(varargin{:});
        plot(ax, ser.t, ser.data(:, 1), '-', 'Color', RED,  'LineWidth', 1.1);
        plot(ax, ser.t, ser.data(:, 2), '-', 'Color', BLUE, 'LineWidth', 1.1);
        h = []; lbl = {}; ncol = 1;
        ref_and_label(ax, p.Results.refline, ylab);
        if ~isempty(p.Results.yclip)
            yl = p.Results.yclip;
            ylim(ax, [ax.YLim(1), yl]);
            [pk, i1] = max(ser.data(:, 1)); [pk2, i2] = max(ser.data(:, 2));
            if pk2 > pk, pk = pk2; i1 = i2; end
            tpk = ser.t(i1);
            % Off-scale-peak marker: a short upward arrow at the clipped top,
            % annotated with the true value -- the break is stated, not hidden.
            plot(ax, [tpk tpk], [yl * 0.90, yl], '-', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0, 'HandleVisibility', 'off');
            plot(ax, tpk, yl, '^', 'MarkerSize', 4, 'MarkerFaceColor', [0.35 0.35 0.35], ...
                 'MarkerEdgeColor', 'none', 'HandleVisibility', 'off');
            text(ax, tpk, yl, sprintf(' %.0f', p.Results.peak), 'FontSize', fs - 1, ...
                 'Color', [0.35 0.35 0.35], 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');
        end
    end

    function [h, lbl, ncol] = draw_one(ax, ser, ylab, col, varargin)
        p = inputParser; p.addParameter('refline', []); p.parse(varargin{:});
        plot(ax, ser.t, ser.data(:, 1), '-', 'Color', col, 'LineWidth', 1.1);
        h = []; lbl = {}; ncol = 1;
        ref_and_label(ax, p.Results.refline, ylab);
    end

    function [h, lbl, ncol] = draw_gov(ax, DVs, stem, ylab, varargin)
        % R/L colour is shared (see draw_rl); the only thing this panel needs
        % to explain locally is dashed=raw vs solid=governed, forced top-right.
        % 'refline', when given, is the governor's ceiling for this quantity --
        % where the raw command is cut down to the governed one.
        p = inputParser; p.addParameter('refline', []); p.parse(varargin{:});
        G = DVs.gov;
        plot(ax, G.t, G.(stem + "_raw_r"), '--', 'Color', RED,  'LineWidth', 0.9);
        plot(ax, G.t, G.(stem + "_raw_l"), '--', 'Color', BLUE, 'LineWidth', 0.9);
        plot(ax, G.t, G.(stem + "_gov_r"), '-', 'Color', RED,  'LineWidth', 1.2);
        plot(ax, G.t, G.(stem + "_gov_l"), '-', 'Color', BLUE, 'LineWidth', 1.2);
        h = plot(ax, nan, nan, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 0.9);
        lbl = {'raw'}; ncol = -1;   % -1 = forced top-right, see the legend-placement block above
        ref_and_label(ax, p.Results.refline, ylab);
    end

    function [h, lbl, ncol] = draw_joints(ax, t, Q, ylab)
        % J1..J7 colour is explained once by the shared legend ahead of the
        % whole joints figure (see \hwjointlegend in the .tex), not repeated here.
        for j = 1:7
            plot(ax, t, Q(:, j), '-', 'Color', JOINT(j, :), 'LineWidth', 0.9);
        end
        h = []; lbl = {}; ncol = 1;
        ref_and_label(ax, [], ylab);
    end

    function draw_broken_lambda_cbf(fig, ser, t_max, t_off, show_xlabel)
        % Two stacked axes sharing x: bottom covers the trial's normal range,
        % top is a short strip that jumps straight from the bottom's ceiling to
        % the next round gridline above the true peak (600), with the peak's
        % single line segment inside it. Diagonal marks at the gap are the
        % conventional break notation.
        bottom_top_val = 180;               % bottom axis ceiling (covers everything but the one spike)
        top_bot_val = 500; top_top_val = 600; % top strip: just enough to hold the 557 peak, next gridline at 600
        axb = axes(fig, 'Units', 'normalized', 'Position', [0.19 0.20 0.79 0.58]);
        axt = axes(fig, 'Units', 'normalized', 'Position', [0.19 0.83 0.79 0.12]);
        for a = [axb axt]
            set(a, 'TickDir', 'out', 'TickLength', [0.015 0.015], 'Box', 'off', ...
                   'LineWidth', 0.6, 'XColor', [0.25 0.25 0.25], 'YColor', [0.25 0.25 0.25], ...
                   'FontSize', fs, 'NextPlot', 'add', 'XLim', [0, t_max]);
            grid(a, 'on'); a.GridAlpha = 0.18; a.GridLineStyle = '-';
            plot(a, ser.t, ser.data(:, 1), '-', 'Color', RED,  'LineWidth', 1.1);
            plot(a, ser.t, ser.data(:, 2), '-', 'Color', BLUE, 'LineWidth', 1.1);
            if ~isnan(t_off)
                xline(a, t_off, '--', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.7, 'HandleVisibility', 'off');
            end
        end
        axb.YLim = [0, bottom_top_val];
        axt.YLim = [top_bot_val, top_top_val];
        axt.YTick = top_top_val;                 % only the next natural gridline is labelled
        axt.XTick = [];                            % top strip carries no x-ticks of its own
        ylabel(axb, "$\lambda_{\mathrm{cbf}}$", 'Interpreter', 'latex', 'FontSize', fs + 1);
        if show_xlabel
            xlabel(axb, "$t$ [s]", 'Interpreter', 'latex', 'FontSize', fs + 1);
        end
        axb.XRuler.TickLabelGapOffset = 1;

        % Break marks: short diagonal strokes at the gap, in figure units so
        % they sit cleanly between the two axes regardless of their data range.
        d = 0.012;
        for xc = [0.19, 0.19 + 0.79]
            annotation(fig, 'line', [xc - d, xc + d], [0.79, 0.81], 'Color', [0.25 0.25 0.25], 'LineWidth', 0.8);
            annotation(fig, 'line', [xc - d, xc + d], [0.815, 0.835], 'Color', [0.25 0.25 0.25], 'LineWidth', 0.8);
        end
    end

    function ref_and_label(ax, refline, ylab)
        % Small grey dashed line: the fixed reference/ceiling value for this
        % panel's quantity (zero, a static floor, or a governor ceiling) --
        % one consistent style for "where the cut/limit is" everywhere it appears.
        if ~isempty(refline)
            yline(ax, refline, '--', 'Color', [0.55 0.55 0.55], 'LineWidth', 0.8, 'HandleVisibility', 'off');
            % Headroom above the line itself when it sits above the data (e.g.
            % gov_theta's never-reached ceiling) -- otherwise it renders flush
            % on the axis border and reads as a frame, not a reference value.
            if refline >= ax.YLim(2)
                ylim(ax, [ax.YLim(1), refline * 1.08]);
            end
        end
        ylabel(ax, ylab, 'Interpreter', 'latex', 'FontSize', fs + 1);
    end
end

% ----------------------------------------------------------------- helpers
function ok = available(src, S, DV)
if startsWith(src, "derived.")
    ok = isfield(DV, extractAfter(src, "derived."));
else
    ok = isfield(S, src);
end
end

function Q = joint_cols(js, side, attr)
% 7 arm joints of one side, in joint order, from a JointState series.
names = string(js.names(:))';
Q = nan(numel(js.t), 7);
for j = 1:7
    idx = find(names == sprintf("arm_%s_%d_joint", side, j), 1);
    if ~isempty(idx), Q(:, j) = js.(attr)(:, idx); end
end
end
