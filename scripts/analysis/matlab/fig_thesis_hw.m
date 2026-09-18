function files = fig_thesis_hw(mat_path, out_dir, opts)
%FIG_THESIS_HW Per-panel thesis figures for one offline trial bag (hardware or sim).
%   FILES = FIG_THESIS_HW(MAT_PATH, OUT_DIR) draws every telemetry panel the
%   .mat written by scripts/analysis/export_offline_bag.py can support and
%   writes each as OUT_DIR/hw_<key>.pdf (vector), one file per panel so LaTeX
%   can lay them out as lettered subfigures. Panels whose topic is absent from
%   the bag are skipped, so the same call serves a teleoperation bag.
%
%   Thesis style, as fig_thesis_panel: no title, no in-panel text, figure sized
%   in centimetres, Helvetica at print size, LaTeX symbol + unit as y label,
%   legend inside the axes. Right arm is red, left arm blue, throughout; the
%   dashed vertical line is the end of the open-loop reference (meta.t_off_s).
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
reg('gov_ep')     = {"derived.gov", @(ax) draw_gov(ax, DV, "pos_err", "$\|e_p\|$ [m]")};
reg('gov_theta')  = {"derived.gov", @(ax) draw_gov(ax, DV, "ori_err", "$\theta$ [rad]")};
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
        if ncol >= 7, loc = 'northoutside'; else, loc = 'best'; end
        lg = legend(ax, h, lbl, 'Location', loc, 'Box', 'off', 'FontSize', fs - 0.5, ...
                    'NumColumns', ncol, 'Interpreter', 'none');
        lg.ItemTokenSize = [12 8];
    end
    ax.XRuler.TickLabelGapOffset = 1;

    outfile = fullfile(out_dir, opts.prefix + k + ".pdf");
    exportgraphics(fig, outfile, 'ContentType', 'vector', 'BackgroundColor', 'white');
    close(fig);
    files(end + 1, 1) = string(outfile); %#ok<AGROW>
    fprintf('[fig_thesis_hw] wrote %s\n', outfile);
end

% ----------------------------------------------------------------- drawers
    function [h, lbl, ncol] = draw_rl(ax, ser, ylab, varargin)
        p = inputParser; p.addParameter('refline', []); p.parse(varargin{:});
        h(1) = plot(ax, ser.t, ser.data(:, 1), '-', 'Color', RED,  'LineWidth', 1.1);
        h(2) = plot(ax, ser.t, ser.data(:, 2), '-', 'Color', BLUE, 'LineWidth', 1.1);
        lbl = {'R', 'L'}; ncol = 1;
        ref_and_label(ax, p.Results.refline, ylab);
    end

    function [h, lbl, ncol] = draw_one(ax, ser, ylab, col, varargin)
        p = inputParser; p.addParameter('refline', []); p.parse(varargin{:});
        plot(ax, ser.t, ser.data(:, 1), '-', 'Color', col, 'LineWidth', 1.1);
        h = []; lbl = {}; ncol = 1;
        ref_and_label(ax, p.Results.refline, ylab);
    end

    function [h, lbl, ncol] = draw_gov(ax, DVs, stem, ylab)
        G = DVs.gov;
        plot(ax, G.t, G.(stem + "_raw_r"), '--', 'Color', RED,  'LineWidth', 0.9);
        plot(ax, G.t, G.(stem + "_raw_l"), '--', 'Color', BLUE, 'LineWidth', 0.9);
        h(1) = plot(ax, G.t, G.(stem + "_gov_r"), '-', 'Color', RED,  'LineWidth', 1.2);
        h(2) = plot(ax, G.t, G.(stem + "_gov_l"), '-', 'Color', BLUE, 'LineWidth', 1.2);
        h(3) = plot(ax, nan, nan, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 0.9);
        lbl = {'R', 'L', 'raw'}; ncol = 1;
        ref_and_label(ax, [], ylab);
    end

    function [h, lbl, ncol] = draw_joints(ax, t, Q, ylab)
        h = gobjects(1, 7); lbl = cell(1, 7);
        for j = 1:7
            h(j) = plot(ax, t, Q(:, j), '-', 'Color', JOINT(j, :), 'LineWidth', 0.9);
            lbl{j} = sprintf('J%d', j);
        end
        ncol = 7;
        ref_and_label(ax, [], ylab);
    end

    function ref_and_label(ax, refline, ylab)
        if ~isempty(refline)
            yline(ax, refline, ':', 'Color', [0.2 0.2 0.2], 'LineWidth', 0.7, 'HandleVisibility', 'off');
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
