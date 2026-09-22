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
    opts.broken_panels string = "lambda_cbf"
    opts.broken_ranges double = [180 500 600]
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
% Governor ceilings for the "where the cut is done" reference lines: fixed
% real-hw branch config constants (GOV_E_MAX_POS = 0.30 m, GOV_E_MAX_ORI =
% 1.0 rad), not read per-trial from the data -- the home-to-crossed trial
% happens to plateau exactly at these values (empirical cross-check that
% they were the ones active for that capture), but a trial whose governor
% barely saturates (e.g. a teleoperated one) would otherwise show a
% misleading "ceiling" sitting at whatever its own data happened to peak at.
gov_ep_ceiling = 0.30;
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

% Haption handle telemetry (teleoperated bags only). x/y/z components share
% one colour key, drawn once by the caller (\hwxyzlegend); the operator-in-
% control windows (deadman held AND clutch released, as offline_plotter's
% own _control_intervals) are shaded behind every hap_* panel.
XYZ = JOINT(1:3, :);
reg('hap_pos')    = {"virtuose_pose",     @(ax) draw_xyz(ax, S.virtuose_pose.t, S.virtuose_pose.data(:, 1:3), "$p_h$ [m]")};
reg('hap_rpy')    = {"virtuose_pose",     @(ax) draw_xyz(ax, S.virtuose_pose.t, quat_xyzw_to_rpy(S.virtuose_pose.data(:, 4:7)), "$\phi_h,\theta_h,\psi_h$ [rad]")};
reg('hap_vlin')   = {"virtuose_velocity", @(ax) draw_one(ax, norm_of(S.virtuose_velocity, 1:3), "$\|v_h\|$ [m/s]", TEAL)};
reg('hap_vang')   = {"virtuose_velocity", @(ax) draw_one(ax, norm_of(S.virtuose_velocity, 4:6), "$\|\omega_h\|$ [rad/s]", TEAL)};
% No +-10N/+-1Nm clip reflines here: the trial never gets close to them, so
% forcing the axis to include them would flatten the real signal down to a
% sliver -- the clip values are already stated numerically in the text.
reg('hap_force')  = {"virtuose_force_cmd", @(ax) draw_xyz(ax, S.virtuose_force_cmd.t, S.virtuose_force_cmd.data(:, 1:3), "$F_h$ [N]")};
reg('hap_torque') = {"virtuose_force_cmd", @(ax) draw_xyz(ax, S.virtuose_force_cmd.t, S.virtuose_force_cmd.data(:, 4:6), "$\tau_h$ [N\,m]")};

% Shared-autonomy telemetry: blend_debug = [alpha, v_user(6), v_policy(6),
% v_blend(6)]; the blended-action share is (1-a)|v_user| vs a|v_policy| as a
% fraction of their sum -- the two are complementary, so only the policy's
% is drawn. Belief colours: red family = right object, blue = left, dark =
% top grasp, light = side; platform grey.
reg('sa_alpha')   = {"shared_autonomy_blend_debug", @(ax) draw_one(ax, cols_of(S.shared_autonomy_blend_debug, 1), "$\alpha$", TEAL, 'refline', 0.5)};
reg('sa_share')   = {"shared_autonomy_blend_debug", @(ax) draw_one(ax, policy_share(S.shared_autonomy_blend_debug), "policy share [\%]", TEAL)};
reg('sa_belief')  = {"shared_autonomy_goal_probabilities", @(ax) draw_belief(ax, S.shared_autonomy_goal_probabilities, S.shared_autonomy_goal_names)};
reg('sa_state')   = {"shared_autonomy_active_arm", @(ax) draw_state(ax, S.shared_autonomy_active_arm, S.shared_autonomy_grasp_active)};

if isfield(S, 'virtuose_deadman') && isfield(S, 'virtuose_button_right')
    control_iv = control_intervals(S.virtuose_deadman, S.virtuose_button_right, t_max);
else
    control_iv = zeros(0, 2);
end
SHADE = [0.996 0.976 0.878];   % pale yellow: operator in control (same tone as \hwshade)

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
                 'DefaultTextInterpreter', 'latex', 'DefaultAxesTickLabelInterpreter', 'latex', ...
                 'DefaultLegendInterpreter', 'latex', ...
                 'DefaultAxesFontSize', fs, 'DefaultTextFontSize', fs);

    broken_idx = find(opts.broken_panels == k, 1);
    if ~isempty(broken_idx)
        % A single ~10x taller sample dwarfs the rest of the trial on a linear
        % axis -- the standard fix in print is a broken axis, not clipping the
        % peak away: two stacked axes sharing x, a small gap between them, and
        % the top one's own ticks jumping straight to the next natural
        % gridline instead of continuing linearly. Diagonal marks at the gap
        % mark the break itself. Which panels get this treatment, and where
        % the break sits, is per-trial (see caller): a trial whose barrier
        % never produces an outlier sample needs no break at all.
        broken_series = struct('lambda_cbf', {{S, "qp_debug_lambda_cbf", "$\lambda_{\mathrm{cbf}}$"}}, ...
                                'lambda_jl',  {{S, "qp_debug_lambda_joints", "$\lambda_{\mathrm{jl}}$"}});
        bs = broken_series.(char(k));
        ser = bs{1}.(bs{2});
        draw_broken(fig, ser, bs{3}, t_max, t_off, ismember(k, xlab_keys), opts.broken_ranges(broken_idx, :));
    else
        % Axes reclaim the strip normally reserved for the "$t$ [s]" label
        % when this panel doesn't carry one (only the bottom row of a
        % multi-row figure does -- see xlabel_on) -- tick numbers stay either
        % way, only the label text + its margin are dropped.
        has_xlabel = ismember(k, xlab_keys);
        if has_xlabel, ax_pos = [0.19 0.20 0.79 0.77]; else, ax_pos = [0.19 0.10 0.79 0.87]; end
        ax = axes(fig, 'Units', 'normalized', 'Position', ax_pos);
        set(ax, 'TickDir', 'out', 'TickLength', [0.015 0.015], 'Box', 'off', ...
                'LineWidth', 0.6, 'XColor', [0.25 0.25 0.25], 'YColor', [0.25 0.25 0.25], ...
                'FontSize', fs, 'NextPlot', 'add');
        grid(ax, 'on'); ax.GridAlpha = 0.18; ax.GridLineStyle = '-';

        [h, lbl, ncol] = entry{2}(ax);

        xlim(ax, [0, t_max]);
        if startsWith(k, "hap_") && ~isempty(control_iv)
            shade_intervals(ax, control_iv, SHADE);
        end
        if ~isnan(t_off)
            xline(ax, t_off, '--', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.7, 'HandleVisibility', 'off');
        end
        if has_xlabel
            xlabel(ax, "$t$ [s]", 'Interpreter', 'latex', 'FontSize', fs + 1);
        end
        if ~isempty(h)
            if ncol >= 7,        loc = 'northoutside';
            elseif ncol == -1,   loc = 'northeast';     % forced top-right (e.g. the "raw" style key)
            elseif ncol <= -2,   loc = 'northoutside';  % above the axes, |ncol| columns (many entries)
            else,                loc = 'best';
            end
            lg = legend(ax, h, lbl, 'Location', loc, 'Box', 'off', 'FontSize', fs - 0.5, ...
                        'NumColumns', max(abs(ncol), 1), 'Interpreter', 'latex');
            % A short swatch shows too few dash segments to read as dashed
            % (looks solid) -- the "raw" key needs a longer one to actually
            % show the gap; R/L(-less) legends have no dash to worry about.
            if ncol == -1, lg.ItemTokenSize = [26 8]; else, lg.ItemTokenSize = [12 8]; end
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

    function draw_broken(fig, ser, ylab, t_max, t_off, show_xlabel, brk)
        % Two stacked axes sharing x: bottom covers the trial's normal range,
        % top is a short strip that jumps straight from the bottom's ceiling to
        % the next round gridline above the true peak, with the peak's single
        % line segment inside it. Diagonal marks at the gap are the
        % conventional break notation. brk = [bottom_ceiling, top_floor, top_ceiling],
        % chosen per-trial from where the data actually has a gap (see caller).
        bottom_top_val = brk(1);
        top_bot_val = brk(2); top_top_val = brk(3);
        % Same xlabel-driven reclaim as the single-axes path: the bottom axis
        % keeps its top edge fixed (where the break sits) and grows downward
        % into the label's freed strip when this panel isn't in the bottom row.
        if show_xlabel, axb_bottom = 0.20; else, axb_bottom = 0.10; end
        axb = axes(fig, 'Units', 'normalized', 'Position', [0.19 axb_bottom 0.79 0.78 - axb_bottom]);
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
        ylabel(axb, ylab, 'Interpreter', 'latex', 'FontSize', fs + 1);
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

    function [h, lbl, ncol] = draw_xyz(ax, t, X, ylab, varargin)
        % Three components of one vector quantity, x/y/z in the shared
        % \hwxyzlegend colours; optional symmetric 'reflines' [lo hi] for a
        % device clip (e.g. the +-10 N / +-1 Nm force-manager limits).
        p = inputParser; p.addParameter('reflines', []); p.parse(varargin{:});
        for j = 1:3
            plot(ax, t, X(:, j), '-', 'Color', XYZ(j, :), 'LineWidth', 0.9);
        end
        h = []; lbl = {}; ncol = 1;
        ref_and_label(ax, [], ylab);
        for v = p.Results.reflines
            yline(ax, v, '--', 'Color', [0.55 0.55 0.55], 'LineWidth', 0.8, 'HandleVisibility', 'off');
        end
        if ~isempty(p.Results.reflines)
            r = p.Results.reflines;
            ylim(ax, [min(ax.YLim(1), r(1) * 1.08), max(ax.YLim(2), r(2) * 1.08)]);
        end
    end

    function [h, lbl, ncol] = draw_belief(ax, ser, names_ser)
        % One line per goal, coloured by object (red = right, blue = left)
        % and grasp type (dark = top, light = side); the platform grey.
        nm = string(names_ser.data); names = split(nm(1), ",");
        col = containers.Map();
        col("Red_Top")        = RED;
        col("Red_Side")       = [0.93 0.60 0.52];
        col("Blue_Top")       = BLUE;
        col("Blue_Side")      = [0.58 0.72 0.90];
        col("Platform_Place") = [0.50 0.50 0.50];
        h = gobjects(1, numel(names)); lbl = cell(1, numel(names));
        for j = 1:numel(names)
            if isKey(col, names(j)), c = col(names(j)); else, c = JOINT(mod(j - 1, 7) + 1, :); end
            h(j) = plot(ax, ser.t, ser.data(:, j), '-', 'Color', c, 'LineWidth', 0.9);
            lbl{j} = char(strrep(strrep(names(j), "_", " "), "Platform Place", "Platform"));
        end
        ncol = -3;   % legend above the axes, 3 columns
        ylim(ax, [-0.03, 1.03]);
        ref_and_label(ax, [], "$P(\mathrm{goal})$");
    end

    function [h, lbl, ncol] = draw_state(ax, arm_ser, grasp_ser)
        % Which arm is active as a red/blue band across the whole panel, the
        % autonomous-grasp flag as a 0/1 step on top of it.
        arm_t = arm_ser.t;
        s = string(arm_ser.data); s = s(:);
        is_right = double(s == "right");
        edges = [arm_t(1); arm_t(find(diff(is_right) ~= 0) + 1); t_max];
        for i = 1:numel(edges) - 1
            c = RED; if is_right(find(arm_t >= edges(i), 1)) == 0, c = BLUE; end
            patch(ax, [edges(i) edges(i + 1) edges(i + 1) edges(i)], [-0.1 -0.1 1.1 1.1], c, ...
                  'FaceAlpha', 0.16, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        end
        g = grasp_ser.data(:, 1);
        hg = stairs(ax, grasp_ser.t, g, '-', 'Color', TEAL, 'LineWidth', 1.1);
        hr = patch(ax, nan, nan, RED,  'FaceAlpha', 0.16, 'EdgeColor', 'none');
        hl = patch(ax, nan, nan, BLUE, 'FaceAlpha', 0.16, 'EdgeColor', 'none');
        h = [hr hl hg]; lbl = {'right active', 'left active', 'autonomous grasp'}; ncol = -3;
        ylim(ax, [-0.1, 1.1]); yticks(ax, [0 1]);
        ref_and_label(ax, [], "grasp");
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

function iv = control_intervals(deadman, clutch, t_max)
% [start end] windows where the deadman is held AND the clutch is released --
% offline_plotter's _control_intervals: state re-evaluated at every edge of
% either signal, last value carried forward.
bp = unique([deadman.t(:); clutch.t(:); 0; t_max]);
bp = bp(bp <= t_max);
iv = zeros(0, 2); run_start = NaN;
for i = 1:numel(bp)
    t = bp(i);
    on = last_at(deadman, t) > 0.5 && last_at(clutch, t) < 0.5;
    if on && isnan(run_start)
        run_start = t;
    elseif ~on && ~isnan(run_start)
        iv(end + 1, :) = [run_start t]; run_start = NaN; %#ok<AGROW>
    end
end
if ~isnan(run_start), iv(end + 1, :) = [run_start t_max]; end
end

function v = last_at(ser, t)
% Value of a 0/1 series at time t (last sample at or before t; 0 before the first).
i = find(ser.t <= t, 1, 'last');
if isempty(i), v = 0; else, v = ser.data(i, 1); end
end

function shade_intervals(ax, iv, c)
% Translucent bands behind the data, spanning the axes' current y range.
yl = ax.YLim;
for i = 1:size(iv, 1)
    p = patch(ax, [iv(i, 1) iv(i, 2) iv(i, 2) iv(i, 1)], [yl(1) yl(1) yl(2) yl(2)], c, ...
              'FaceAlpha', 0.8, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    uistack(p, 'bottom');
end
ylim(ax, yl);
end

function rpy = quat_xyzw_to_rpy(q)
% Body-fixed ZYX (yaw-pitch-roll) angles from [x y z w] quaternions.
x = q(:, 1); y = q(:, 2); z = q(:, 3); w = q(:, 4);
roll  = atan2(2 * (w .* x + y .* z), 1 - 2 * (x.^2 + y.^2));
pitch = asin(max(-1, min(1, 2 * (w .* y - z .* x))));
yaw   = atan2(2 * (w .* z + x .* y), 1 - 2 * (y.^2 + z.^2));
rpy = unwrap([roll pitch yaw]);   % no +-pi jumps mid-trace
end

function out = norm_of(ser, cols)
out = struct('t', ser.t, 'data', vecnorm(ser.data(:, cols), 2, 2));
end

function out = cols_of(ser, cols)
out = struct('t', ser.t, 'data', ser.data(:, cols));
end

function out = policy_share(blend)
% a|v_policy| / ((1-a)|v_user| + a|v_policy|) in percent; NaN where both are zero.
a = blend.data(:, 1);
u = (1 - a) .* vecnorm(blend.data(:, 2:7), 2, 2);
p = a .* vecnorm(blend.data(:, 8:13), 2, 2);
tot = u + p;
share = nan(size(a));
ok = tot > 1e-9;
share(ok) = 100 * p(ok) ./ tot(ok);
out = struct('t', blend.t, 'data', share);
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
