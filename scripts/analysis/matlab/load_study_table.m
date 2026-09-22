function [trial, meta] = load_study_table(export_dir, opts)
%LOAD_STUDY_TABLE Tidy per-trial table of COMPLETE participants, with trial order.
%   [TRIAL, META] = LOAD_STUDY_TABLE() loads manifest.mat (see LOAD_MANIFEST),
%   folds the right_/left_ arm columns into one value per metric according to
%   STUDY_METRIC_SPEC, keeps only participants whose data set is complete, and
%   joins participant_schedule.csv to recover the order in which each
%   participant met the conditions.
%
%   Completeness rule (no participant id is ever hard-coded): a participant is
%   complete when (a) it appears in participant_schedule.csv and (b) the
%   manifest holds exactly the 12 (world, cell) trials of the design. Every
%   other participant is listed in META.excluded with the reason. Re-running
%   after new participants are exported therefore needs no code change.
%
%   TRIAL columns: participant, world, cell, mode ("C"/"J"), assist ("F"/"B"/
%   "FB"), control_mode, success, notes, act_right_frac, one column
%   per metric in the spec, and (from the schedule) slot (1-6),
%   slot_in_mode (1-3), mode_order ("C_first"/"J_first"), trial_index (1-12).
%   Metrics outside their cell_scope are NaN, so every 'omitnan' aggregation
%   downstream automatically uses only the cells where the metric is defined.
%
%   Options: include_incomplete (default false), schedule_path (default:
%   auto-search next to these scripts, next to the manifest, then
%   ../participant_schedule.csv relative to this file).

arguments
    export_dir (1,1) string = ""
    opts.include_incomplete (1,1) logical = false
    opts.schedule_path (1,1) string = ""
end

WORLDS = ["rack" "shield"];
CELLS  = ["CF" "CB" "CFB" "JF" "JB" "JFB"];

% char(): study_export_dir treats a char '' (not a string "") as "auto-locate".
T = load_manifest(char(export_dir));
export_dir = string(study_export_dir(char(export_dir)));
spec = study_metric_spec();

%% ---- schedule --------------------------------------------------------------
sched = read_schedule(opts.schedule_path, export_dir);

%% ---- completeness per participant -----------------------------------------
allP = unique(string(T.participant), 'stable');
allP = sort(allP);
complete = false(numel(allP), 1);
reason = strings(numel(allP), 1);
for i = 1:numel(allP)
    rows = string(T.participant) == allP(i);
    have = unique(string(T.world(rows)) + "_" + string(T.cell(rows)));
    want = reshape(WORLDS' + "_" + CELLS, [], 1);
    missing = setdiff(want, have);
    extra = setdiff(have, want);
    inSched = ~isempty(sched) && any(sched.participant == allP(i));
    if ~inSched
        reason(i) = "not in participant_schedule.csv (pilot?)";
    elseif ~isempty(missing)
        reason(i) = sprintf("%d/12 trials, missing: %s", numel(have), strjoin(missing, ", "));
    elseif ~isempty(extra)
        reason(i) = "unexpected folders: " + strjoin(extra, ", ");
    else
        complete(i) = true;
    end
end
meta = struct();
meta.export_dir = export_dir;
meta.schedule_path = "";
if ~isempty(sched), meta.schedule_path = sched.Properties.UserData; end
meta.complete_list = allP(complete);
meta.n_complete = nnz(complete);
meta.excluded = table(allP(~complete), reason(~complete), 'VariableNames', {'participant','reason'});
meta.built = datetime('now');

keep = ismember(string(T.participant), allP(complete));
if opts.include_incomplete, keep(:) = true; end
Traw = T(keep, :);

%% ---- design columns --------------------------------------------------------
trial = table();
trial.participant = string(Traw.participant);
trial.world  = string(Traw.world);
trial.cell   = string(Traw.cell);
trial.mode   = extractBefore(trial.cell + " ", 2);          % "C" / "J"
trial.assist = extractAfter(trial.cell, 1);                  % "F" / "B" / "FB"
trial.control_mode = string(Traw.control_mode);
trial.success  = gettext(Traw, 'success') == "yes";
trial.notes    = gettext(Traw, 'notes');

wR = getnum(Traw, 'right_this_arm_active_frac');
wL = getnum(Traw, 'left_this_arm_active_frac');
trial.act_right_frac = wR;

%% ---- metrics: fold right_/left_ into one value per trial -------------------
for k = 1:height(spec)
    nm = spec.name(k);
    switch spec.combine(k)
        case "shared"
            v = getnum(Traw, "right_" + nm);
        case "sum"
            v = getnum(Traw, "right_" + nm) + getnum(Traw, "left_" + nm);
        case "wmean"
            v = wmean2(getnum(Traw, "right_" + nm), getnum(Traw, "left_" + nm), wR, wL);
        case "max"
            v = max(getnum(Traw, "right_" + nm), getnum(Traw, "left_" + nm), 'omitnan');
        case "min"
            v = min(getnum(Traw, "right_" + nm), getnum(Traw, "left_" + nm), 'omitnan');
        case "derived"
            v = derived_metric(nm, trial, Traw);
        otherwise
            error('load_study_table:combine', 'unknown combine rule "%s" for %s', spec.combine(k), nm);
    end
    v = double(v(:));
    % Out-of-scope cells hold no physical meaning for this metric -> NaN.
    switch spec.cell_scope(k)
        case "clutch_only", v(trial.mode ~= "C") = NaN;
        case "blend_only",  v(~contains(trial.assist, "B")) = NaN;
    end
    trial.(nm) = v;
end

%% ---- trial order from the schedule -----------------------------------------
n = height(trial);
trial.slot = nan(n, 1);
trial.slot_in_mode = nan(n, 1);
trial.mode_order = strings(n, 1);
trial.trial_index = nan(n, 1);
if ~isempty(sched)
    for i = 1:n
        r = find(sched.participant == trial.participant(i), 1);
        if isempty(r), continue; end
        order = sched.slots(r, :);
        s = find(order == trial.cell(i), 1);
        if isempty(s), continue; end
        trial.slot(i) = s;
        trial.slot_in_mode(i) = mod(s - 1, 3) + 1;
        trial.mode_order(i) = extractBefore(order(1) + " ", 2) + "_first";
        % Within a slot the worlds were always run shield first, then rack.
        trial.trial_index(i) = 2 * (s - 1) + (1 + double(trial.world(i) == "rack"));
    end
end
trial = sortrows(trial, {'participant', 'trial_index'});
end

%% ============================ local functions ===============================
function v = derived_metric(nm, trial, Traw)
switch nm
    case "teleop_time_s"
        v = getnum(Traw, 'right_duration_s') - getnum(Traw, 'right_autonomy_grasp_time_s');
    case "ee_path_efficiency"
        % Ratio of the arm sums, not a mean of two ratios: an idle arm with a
        % tiny path cannot inflate the efficiency.
        s = getnum(Traw, 'right_ee_straight_len_m') + getnum(Traw, 'left_ee_straight_len_m');
        p = getnum(Traw, 'right_ee_path_len_m') + getnum(Traw, 'left_ee_path_len_m');
        v = s ./ p; v(p <= 1e-6) = NaN;
    case "belief_confident_ever"
        v = double(~isnan(getnum(Traw, 'right_belief_time_to_conf_s')));
    case "cbf_active_s"
        % Absolute seconds, so a fraction that only rose because the trial got
        % shorter can be told apart from more time actually spent at the barrier.
        % The fraction is already over T_h, so the base is the time under human control.
        v = trial.cbf_active_frac .* trial.teleop_time_s;
    case "safety_nearmiss_s"
        v = trial.safety_nearmiss_frac .* trial.teleop_time_s;
    otherwise
        error('load_study_table:derived', 'no rule for derived metric %s', nm);
end
end

function sched = read_schedule(schedule_path, export_dir)
% Returns [] if not found. Otherwise a table with participant (string) and
% slots (n x 6 string) and the path used in Properties.UserData.
here = fileparts(mfilename('fullpath'));
cands = [schedule_path, ...
         fullfile(here, "participant_schedule.csv"), ...
         fullfile(export_dir, "participant_schedule.csv"), ...
         fullfile(here, "..", "participant_schedule.csv")];
cands = cands(strlength(cands) > 0);
sched = [];
for c = cands
    if isfile(c)
        raw = readtable(c, 'TextType', 'string', 'VariableNamingRule', 'preserve');
        sched = table();
        sched.participant = strtrim(string(raw{:, 1}));
        sched.slots = strtrim(string(raw{:, 2:7}));
        sched.Properties.UserData = string(c);
        return
    end
end
warning('load_study_table:noSchedule', ...
    ['participant_schedule.csv not found -- every participant is treated as ' ...
     'unscheduled and trial order (learning effect) is unavailable. Looked in: %s'], ...
    strjoin(cands, ' ; '));
end

function v = getnum(T, name)
% Numeric column as double, tolerating a cell column or a missing variable.
name = char(name);
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
name = char(name);
if ~ismember(name, T.Properties.VariableNames)
    s = strings(height(T), 1); return
end
s = string(T.(name)); s = s(:);
s(ismissing(s)) = "";
end

function v = wmean2(a, b, wa, wb)
% Combine the two arms weighted by how long each was the active one; NaN-tolerant.
w = wa + wb;
wa = wa ./ w;  wb = wb ./ w;
v = wa .* a + wb .* b;
onlyA = isnan(b) & ~isnan(a);   v(onlyA) = a(onlyA);
onlyB = isnan(a) & ~isnan(b);   v(onlyB) = b(onlyB);
end
