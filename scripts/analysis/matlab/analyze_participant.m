%% ANALYZE_PARTICIPANT  Per-participant performance profile for the TRIAGo study.
%
%  WHAT IT DOES
%    Loads every exported trial of ONE participant and answers three questions
%    about that person, from their objective performance (not questionnaires):
%       * did they do better in a particular WORLD?          (rack vs shield)
%       * did they do better with a particular TELEOPERATION? (clutch vs joystick)
%       * did they do better with a particular ASSISTANCE?    (F vs B vs FB)
%    plus an overall profile and the full 2x3 condition grid.
%
%  HOW TO USE
%    1. Set PARTICIPANT below (or leave '' to get a pick-list dialog).
%    2. Press Run.
%    3. Read the console report, look at the figure, and inspect the variable
%       P in the workspace (double-click P.summary for a spreadsheet view).
%
%  HOW THE INDICES WORK
%    Every metric is z-scored WITHIN this participant's own trials and flipped
%    so that higher is always better. Metrics are grouped into four families
%    (efficiency / smoothness / safety / success); the composite is the mean of
%    the four family scores, so a family with many metrics cannot dominate.
%    A score of 0 = this participant's own average. It is a DESCRIPTIVE profile
%    of one person, not a statistical test -- group comparisons come later.
%
%    Mode-specific metrics (haptic force, clutch presses) are deliberately NOT
%    in the composite: force feedback is rendered differently per control mode
%    and clutch presses only exist in CLUTCH, so including them would make
%    "clutch vs joystick" meaningless. They are reported separately instead.
%
%  OUTPUT (all left in the workspace)
%    P.participant  P.trials  P.metrics  P.z  P.summary
%    P.byWorld  P.byMode  P.byAssist  P.byCell  P.overall  P.preference  P.spec

%% ============================ USER SETTINGS ============================
PARTICIPANT = '';      % '' = ask with a dialog, or set e.g. 'P07'
EXPORT_DIR  = '';      % '' = default ~/exchange/triago_study_data/matlab_export
SHOW_FIGURE = true;    % set false for console-only
%% =======================================================================

T = load_manifest(EXPORT_DIR);
allP = unique(string(T.participant));

if isempty(PARTICIPANT)
    [sel, ok] = listdlg('PromptString', 'Pick a participant:', ...
                        'SelectionMode', 'single', 'ListSize', [220 260], ...
                        'ListString', cellstr(allP));
    if ~ok, fprintf('cancelled\n'); return; end
    PARTICIPANT = char(allP(sel));
end
if ~ismember(string(PARTICIPANT), allP)
    error('analyze_participant:noSuchParticipant', ...
          '%s is not in the export. Available: %s', PARTICIPANT, strjoin(cellstr(allP), ', '));
end

rows = string(T.participant) == string(PARTICIPANT);
Traw = T(rows, :);

%% ---- metric spec: name, label, direction (+1 better high), family ----
% family '' = reported only, kept OUT of the composite (mode-specific or not
% comparable across control modes).
spec = { ...
  'duration_s',            'task time',                's',    -1, 'efficiency'
  'ee_path_len_m',         'total hand path',          'm',    -1, 'efficiency'
  'ee_path_efficiency',    'path efficiency',          '',     +1, 'efficiency'
  'ee_sparc',              'smoothness (SPARC)',       '',     +1, 'smoothness'
  'qdot_cmd_rms',          'joint-rate effort',        'rad/s',-1, 'smoothness'
  'ee_speed_mean_mps',     'mean hand speed',          'm/s',   0, 'smoothness'
  'safety_min_dist_m',     'min clearance',            'm',    +1, 'safety'
  'safety_nearmiss_frac',  'near-miss time',           'frac', -1, 'safety'
  'safety_nearmiss_episodes','near-miss episodes',     '',     -1, 'safety'
  'cbf_active_frac',       'CBF active',               'frac', -1, 'safety'
  'success',               'success',                  '',     +1, 'success'
  'force_impulse_Ns',      'force impulse',            'N.s',  -1, ''
  'force_mean_N',          'mean force',               'N',    -1, ''
  'clutch_presses',        'clutch presses',           '',     -1, ''
  'clutch_duty_frac',      'clutch duty',              'frac', -1, ''
  'alpha_mean',            'mean authority alpha',     '',      0, ''
  'agreement_mean_cos',    'user-policy agreement',    'cos',  +1, ''
  'autonomy_grasp_time_s', 'autonomous grasp time',    's',     0, ''
  'belief_time_to_conf_s', 'time to confident intent', 's',    -1, ''
  'loop_freq_mean_hz',     'control loop rate',        'Hz',    0, ''
};
spec = cell2table(spec, 'VariableNames', {'name','label','unit','dir','family'});
spec.name = string(spec.name);   spec.label  = string(spec.label);
spec.unit = string(spec.unit);   spec.family = string(spec.family);

%% ---- per-trial metrics, both arms combined ----
M = table();
M.world  = gettext(Traw, 'world');
M.cell   = gettext(Traw, 'cell');
M.mode   = gettext(Traw, 'control_mode');
M.assist = assistOf(M.cell);
M.label  = M.world + "_" + M.cell;

% Active-arm fractions are complementary and weight every per-arm metric.
wR = getnum(Traw, 'right_this_arm_active_frac');
wL = getnum(Traw, 'left_this_arm_active_frac');
M.act_right = wR;

% Shared (identical in both arms' metric blocks -- verified against the export).
shared = {'duration_s','safety_min_dist_m','safety_mean_dist_m', ...
          'safety_nearmiss_frac','safety_nearmiss_episodes', ...
          'safety_min_dist_graspincl_m','force_mean_N','force_peak_N', ...
          'force_impulse_Ns','clutch_presses','clutch_duty_frac', ...
          'autonomy_grasp_time_s','autonomy_grasp_frac','alpha_mean', ...
          'alpha_autonomy_frac','agreement_mean_cos','user_active_frac', ...
          'belief_max_prob','belief_time_to_conf_s','loop_freq_mean_hz'};
for k = 1:numel(shared)
    M.(shared{k}) = getnum(Traw, ['right_' shared{k}]);
end

% Extensive per-arm quantities add; the task uses both hands.
pathR = getnum(Traw,'right_ee_path_len_m');      pathL = getnum(Traw,'left_ee_path_len_m');
strR  = getnum(Traw,'right_ee_straight_len_m');  strL  = getnum(Traw,'left_ee_straight_len_m');
M.ee_path_len_m      = pathR + pathL;
M.ee_straight_len_m  = strR + strL;
% Ratio of the sums, not a mean of ratios -- an idle arm cannot inflate it.
M.ee_path_efficiency = M.ee_straight_len_m ./ M.ee_path_len_m;

% Intensive per-arm quantities: weight by how long that arm was the active one.
wpairs = {'ee_speed_mean_mps','ee_sparc','qdot_cmd_rms','qdot_meas_rms', ...
          'slack_mean','cbf_lambda_mean','cbf_active_frac'};
for k = 1:numel(wpairs)
    M.(wpairs{k}) = wmean2(getnum(Traw,['right_' wpairs{k}]), ...
                           getnum(Traw,['left_'  wpairs{k}]), wR, wL);
end
% Worst-case per-arm quantities take the max over both arms.
mpairs = {'ee_speed_max_mps','qdot_cmd_max','qdot_meas_max','slack_peak','cbf_lambda_peak'};
for k = 1:numel(mpairs)
    M.(mpairs{k}) = max(getnum(Traw,['right_' mpairs{k}]), ...
                        getnum(Traw,['left_'  mpairs{k}]), 'omitnan');
end

M.success = double(gettext(Traw,'success') == "yes");

%% ---- oriented z-scores, family scores, composite ----
inComposite = spec.family ~= "";
Z = table();
for k = 1:height(spec)
    nm = char(spec.name(k));
    if ismember(nm, M.Properties.VariableNames)
        Z.(nm) = spec.dir(k) * zscoreSafe(M.(nm));
    else
        Z.(nm) = nan(height(M),1);
    end
end

families = unique(spec.family(inComposite), 'stable');
F = table();
for k = 1:numel(families)
    cols = spec.name(spec.family == families(k));
    F.(char(families(k))) = mean(Z{:, cols}, 2, 'omitnan');
end

% A family that never varies (e.g. every trial succeeded) carries no signal;
% averaging its constant 0 into the composite would shrink every score toward
% zero, so drop it from the composite but keep the column visible.
keepFam = false(1, numel(families));
for k = 1:numel(families)
    v = F.(char(families(k)));
    keepFam(k) = any(isfinite(v)) && std(v, 'omitnan') > eps;
end
droppedFamilies = families(~keepFam);
F.composite = mean(F{:, cellstr(families(keepFam))}, 2, 'omitnan');
scoreCols = F.Properties.VariableNames;

%% ---- aggregate by every grouping the study cares about ----
byWorld  = aggregate(M.world,  F, scoreCols);
byMode   = aggregate(M.mode,   F, scoreCols);
byAssist = aggregate(M.assist, F, scoreCols, ["F" "B" "FB"]);
byCell   = aggregate(M.cell,   F, scoreCols);

summary = [tagGroup(byWorld,'world'); tagGroup(byMode,'mode'); ...
           tagGroup(byAssist,'assist'); tagGroup(byCell,'cell')];

%% ---- pack the output struct ----
P = struct();
P.participant = PARTICIPANT;
P.trials      = Traw;
P.metrics     = M;
P.z           = Z;
P.scores      = F;
P.summary     = summary;
P.byWorld     = levelStruct(byWorld);
P.byMode      = levelStruct(byMode);
P.byAssist    = levelStruct(byAssist);
P.byCell      = levelStruct(byCell);
P.spec        = spec;
P.overall     = overallStruct(M, spec);
P.preference  = struct('world',  preference(byWorld), ...
                       'mode',   preference(byMode), ...
                       'assist', preference(byAssist));
P.built       = datetime('now');

%% ============================ CONSOLE REPORT ============================
nExpected = 12;
missing = missingCells(M);
line = repmat('=', 1, 78);

fprintf('\n%s\n PARTICIPANT %s  --  per-condition performance profile\n%s\n', ...
        line, PARTICIPANT, line);
fprintf(' trials      : %d of %d   |   success: %d yes, %d no\n', ...
        height(M), nExpected, sum(M.success==1), sum(M.success==0));
if ~isempty(missing)
    fprintf(' MISSING     : %s  (indices below use only what exists)\n', strjoin(missing, ', '));
end
fprintf(' task time   : %.0f s mean  (%.0f - %.0f)\n', ...
        mean(M.duration_s,'omitnan'), min(M.duration_s), max(M.duration_s));
fprintf(['\n scores are z-units within THIS participant''s own trials, higher = better.\n' ...
         ' 0 = their own average. Descriptive profile of one person, not a test.\n']);
if ~isempty(droppedFamilies)
    fprintf(' note        : %s constant across all trials -> shown but excluded from composite\n', ...
            strjoin(cellstr(droppedFamilies), ', '));
end

printGroup(' BY WORLD           (does this person do better in one scene?)', byWorld, scoreCols);
printGroup(' BY TELEOPERATION   (clutch vs joystick)',                       byMode, scoreCols);
printGroup(' BY ASSISTANCE      (guidance F / blending B / both FB)',        byAssist, scoreCols);

fprintf('\n 2x3 CONDITION GRID  (composite)\n');
printGrid(byCell);

fprintf('\n MODE-SPECIFIC METRICS  (never cross-mode comparable -- within-mode only)\n');
printModeSpecific(M);

fprintf('\n VERDICT\n');
printPreference('world ', P.preference.world);
printPreference('mode  ', P.preference.mode);
printPreference('assist', P.preference.assist);

fprintf(['\n query it:  P.byWorld.rack.composite    P.byAssist.FB.safety\n' ...
         '            double-click  P.summary  in the workspace for a table view\n%s\n\n'], line);

%% ============================== FIGURE =================================
if SHOW_FIGURE
    fig = figure('Name', sprintf('Participant %s', PARTICIPANT), ...
                 'Color', 'w', 'Position', [80 80 1180 720]);
    tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    barPanel(nexttile(tl), byWorld,  'By world',         [0.20 0.42 0.68]);
    barPanel(nexttile(tl), byMode,   'By teleoperation', [0.85 0.45 0.20]);
    barPanel(nexttile(tl), byAssist, 'By assistance',    [0.30 0.62 0.35]);

    gridPanel(nexttile(tl), byCell);
    familyPanel(nexttile(tl), byCell, scoreCols);
    trialPanel(nexttile(tl), M, F);

    title(tl, sprintf('%s  --  performance profile (z within participant, higher = better)', ...
          PARTICIPANT), 'FontWeight', 'bold');
end

%% ============================ LOCAL FUNCTIONS ==========================
function v = getnum(T, name)
% Numeric column as double, tolerating a cell column or missing variable.
    if ~ismember(name, T.Properties.VariableNames)
        v = nan(height(T), 1); return
    end
    c = T.(name);
    if isnumeric(c)
        v = double(c(:));
    elseif iscell(c)
        v = nan(numel(c), 1);
        for i = 1:numel(c)
            x = c{i};
            if isnumeric(x) && isscalar(x), v(i) = double(x);
            elseif ischar(x) || isstring(x), v(i) = str2double(x);
            end
        end
    else
        v = double(c(:));
    end
end

function s = gettext(T, name)
% Text column as a string array, tolerating a missing variable.
    if ~ismember(name, T.Properties.VariableNames)
        s = strings(height(T), 1); return
    end
    s = string(T.(name));
    s = s(:);
end

function a = assistOf(cellCode)
% 'CF'->"F", 'CFB'->"FB", 'C'->"none": the assistance half of the cell code.
    a = extractAfter(string(cellCode), 1);
    a(a == "") = "none";
end

function v = wmean2(a, b, wa, wb)
% Combine the two arms by how long each was the active one; NaN-tolerant.
    w = wa + wb;
    wa = wa ./ w;  wb = wb ./ w;
    v = wa .* a + wb .* b;
    onlyA = isnan(b) & ~isnan(a);   v(onlyA) = a(onlyA);
    onlyB = isnan(a) & ~isnan(b);   v(onlyB) = b(onlyB);
end

function z = zscoreSafe(x)
% z-score that degrades to 0 instead of Inf when a metric never varies.
    m = mean(x, 'omitnan');
    s = std(x, 'omitnan');
    if ~isfinite(s) || s < eps
        z = zeros(size(x));
        z(isnan(x)) = NaN;
    else
        z = (x - m) ./ s;
    end
end

function A = aggregate(key, F, scoreCols, order)
% Mean of every score column per level of `key`, as a table.
    key = string(key);
    if nargin < 4 || isempty(order)
        levels = unique(key, 'stable');
        levels = sort(levels);
    else
        levels = order(ismember(order, unique(key)));
    end
    A = table();
    A.level = levels(:);
    A.n = zeros(numel(levels), 1);
    for c = 1:numel(scoreCols)
        A.(scoreCols{c}) = nan(numel(levels), 1);
    end
    for i = 1:numel(levels)
        sel = key == levels(i);
        A.n(i) = sum(sel);
        for c = 1:numel(scoreCols)
            A.(scoreCols{c})(i) = mean(F.(scoreCols{c})(sel), 'omitnan');
        end
    end
end

function A = tagGroup(A, name)
% Prefix an aggregate table with the grouping it came from, for P.summary.
    A = addvars(A, repmat(string(name), height(A), 1), 'Before', 1, ...
                'NewVariableNames', 'group');
end

function S = levelStruct(A)
% Aggregate table -> struct addressable as S.<level>.<score>.
    S = struct();
    for i = 1:height(A)
        fn = matlab.lang.makeValidName(A.level(i));
        for c = 2:width(A)
            S.(fn).(A.Properties.VariableNames{c}) = A{i, c};
        end
    end
end

function S = overallStruct(M, spec)
% Raw (un-z-scored) participant means, so absolute values stay inspectable.
    S = struct();
    for k = 1:height(spec)
        nm = char(spec.name(k));
        if ismember(nm, M.Properties.VariableNames)
            S.(nm) = mean(M.(nm), 'omitnan');
        end
    end
    S.n_trials = height(M);
    S.success_rate = mean(M.success, 'omitnan');
end

function pref = preference(A)
% Winning level plus the margin over the runner-up, with a confidence word.
    [~, idx] = sort(A.composite, 'descend', 'MissingPlacement', 'last');
    pref = struct('best', A.level(idx(1)), 'score', A.composite(idx(1)));
    if numel(idx) > 1
        pref.runnerUp = A.level(idx(2));
        pref.margin   = A.composite(idx(1)) - A.composite(idx(2));
    else
        pref.runnerUp = "";
        pref.margin   = NaN;
    end
    if isnan(pref.margin) || pref.margin < 0.20
        pref.verdict = "no clear preference";
    elseif pref.margin < 0.50
        pref.verdict = "slight preference";
    else
        pref.verdict = "clear preference";
    end
end

function m = missingCells(M)
% Which of the 12 expected world x cell trials are absent for this person.
    worlds = ["rack" "shield"];
    cells  = ["CF" "CB" "CFB" "JF" "JB" "JFB"];
    have = string(M.world) + "_" + string(M.cell);
    m = {};
    for w = worlds
        for c = cells
            want = w + "_" + c;
            if ~any(have == want), m{end+1} = char(want); end %#ok<AGROW>
        end
    end
end

function printGroup(header, A, scoreCols)
% One aggregate table as an aligned console block.
    fprintf('\n%s\n', header);
    fprintf('   %-10s %4s', 'level', 'n');
    for c = 1:numel(scoreCols), fprintf(' %11s', scoreCols{c}); end
    fprintf('\n');
    for i = 1:height(A)
        fprintf('   %-10s %4d', A.level(i), A.n(i));
        for c = 1:numel(scoreCols)
            fprintf(' %+11.2f', A.(scoreCols{c})(i));
        end
        fprintf('\n');
    end
end

function printGrid(byCell)
% The 2x3 study grid as a small composite table.
    modes  = ["C" "J"];  modeName = ["CLUTCH" "JOYSTICK"];
    assist = ["F" "B" "FB"];
    fprintf('   %-10s %8s %8s %8s\n', '', assist(1), assist(2), assist(3));
    for m = 1:2
        fprintf('   %-10s', modeName(m));
        for a = 1:3
            code = modes(m) + assist(a);
            idx = find(byCell.level == code, 1);
            if isempty(idx), fprintf(' %8s', '--');
            else,            fprintf(' %+8.2f', byCell.composite(idx)); end
        end
        fprintf('\n');
    end
end

function printModeSpecific(M)
% Force / clutch summaries, split by mode and never compared across them.
    for mode = ["CLUTCH" "JOYSTICK"]
        sel = string(M.mode) == mode;
        if ~any(sel), continue; end
        fprintf('   %-9s n=%d  force impulse %7.1f N.s | mean force %5.2f N | clutch presses %5.1f (duty %.2f)\n', ...
            mode, sum(sel), mean(M.force_impulse_Ns(sel),'omitnan'), ...
            mean(M.force_mean_N(sel),'omitnan'), mean(M.clutch_presses(sel),'omitnan'), ...
            mean(M.clutch_duty_frac(sel),'omitnan'));
    end
end

function printPreference(name, pref)
    if pref.runnerUp == ""
        fprintf('   %s : %-8s (only level present)\n', name, pref.best);
    else
        fprintf('   %s : %-8s  %+.2f   (over %s by %.2f -- %s)\n', ...
            name, pref.best, pref.score, pref.runnerUp, pref.margin, pref.verdict);
    end
end

function barPanel(ax, A, ttl, colour)
% Composite bar chart for one grouping.
    b = bar(ax, A.composite, 'FaceColor', colour, 'EdgeColor', 'none');
    b.FaceAlpha = 0.9;
    set(ax, 'XTick', 1:height(A), 'XTickLabel', cellstr(A.level));
    ylabel(ax, 'composite (z)');
    title(ax, ttl);
    yline(ax, 0, 'k-');
    grid(ax, 'on');
    ax.YGrid = 'on'; ax.XGrid = 'off';
    lim = max(0.6, max(abs(A.composite)) * 1.35);
    ylim(ax, [-lim lim]);
    for i = 1:height(A)
        text(ax, i, A.composite(i), sprintf('%+.2f', A.composite(i)), ...
            'HorizontalAlignment','center', 'FontSize', 8, ...
            'VerticalAlignment', ternary(A.composite(i) >= 0, 'bottom', 'top'));
    end
end

function gridPanel(ax, byCell)
% 2x3 grid of composites; grey = that cell was not recorded.
    modes  = ["C" "J"];  assist = ["F" "B" "FB"];
    G = nan(2, 3);
    for m = 1:2
        for a = 1:3
            idx = find(byCell.level == modes(m) + assist(a), 1);
            if ~isempty(idx), G(m, a) = byCell.composite(idx); end
        end
    end
    imagesc(ax, G, 'AlphaData', ~isnan(G));
    set(ax, 'Color', [0.9 0.9 0.9], 'XTick', 1:3, 'XTickLabel', cellstr(assist), ...
        'YTick', 1:2, 'YTickLabel', {'CLUTCH','JOYSTICK'});
    lim = max(0.6, max(abs(G(:)), [], 'omitnan') * 1.1);
    clim(ax, [-lim lim]);
    colormap(ax, cool2warm());
    colorbar(ax);
    title(ax, 'Condition grid (composite)');
    for m = 1:2
        for a = 1:3
            if isnan(G(m, a)), txt = '--'; else, txt = sprintf('%+.2f', G(m, a)); end
            text(ax, a, m, txt, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        end
    end
end

function familyPanel(ax, byCell, scoreCols)
% Which family drives each cell's score.
    fams = scoreCols(~strcmp(scoreCols, 'composite'));
    V = zeros(height(byCell), numel(fams));
    for c = 1:numel(fams), V(:, c) = byCell.(fams{c}); end
    b = bar(ax, V, 'grouped', 'EdgeColor', 'none');
    set(ax, 'XTick', 1:height(byCell), 'XTickLabel', cellstr(byCell.level));
    legend(ax, b, fams, 'Location', 'southoutside', 'Orientation', 'horizontal', ...
           'Box', 'off', 'AutoUpdate', 'off');
    title(ax, 'Score breakdown by family');
    ylabel(ax, 'z'); yline(ax, 0, 'k-'); ax.YGrid = 'on';
end

function trialPanel(ax, M, F)
% Every individual trial, so a single outlier stays visible.
    [~, ord] = sort(F.composite, 'descend', 'MissingPlacement', 'last');
    vals = F.composite(ord);
    cols = zeros(numel(ord), 3);
    isRack = string(M.world(ord)) == "rack";
    cols(isRack, :)  = repmat([0.20 0.42 0.68], sum(isRack), 1);
    cols(~isRack, :) = repmat([0.85 0.45 0.20], sum(~isRack), 1);
    b = bar(ax, vals, 'FaceColor', 'flat', 'EdgeColor', 'none');
    b.CData = cols;
    set(ax, 'XTick', 1:numel(ord), 'XTickLabel', cellstr(M.label(ord)), ...
        'XTickLabelRotation', 60, 'FontSize', 8);
    title(ax, 'Per-trial composite  (blue = rack, orange = shield)');
    ylabel(ax, 'z'); yline(ax, 0, 'k-'); ax.YGrid = 'on';
end

function c = cool2warm()
% Diverging blue-white-red map, so 0 reads as neutral in the grid panel.
    n = 128;
    t = linspace(0, 1, n)';
    lo = [0.23 0.30 0.75] .* (1 - t) + [1 1 1] .* t;
    hi = [1 1 1] .* (1 - t) + [0.71 0.02 0.15] .* t;
    c = [lo; hi];
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
