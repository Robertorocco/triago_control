%% RUN_STUDY_ANALYSIS  Group-level statistics of the TRIAGo teleoperation study.
%
%  WHAT IT DOES
%    Loads every COMPLETE participant (see load_study_table), computes family
%    and composite scores, and answers for every metric and every family:
%      Q1  Clutch vs Joystick               (paired t-test + Wilcoxon)
%      Q2  Feedback vs Blending vs both     (RM-ANOVA + Friedman + Holm pairs)
%      Q3  the six mode x assistance cells  (2-way RM-ANOVA + Friedman + Holm)
%      Q4  which mode is most consistent across participants
%      Q5  which assistance is most consistent across participants
%      Q6  learning / order effects along the experiment
%      QW  rack vs shield world (supplementary)
%    Everything is written to a time-stamped folder so that results can be
%    kept and compared when more participants are added. Re-run this script:
%    nothing has to be edited when new participants are exported.
%
%  HOW TO USE
%    Open the folder that holds manifest.mat (or the bundle folder) as the
%    current folder, or set EXPORT_DIR, and press Run. Then run
%    build_report to regenerate the Live Script report from the results.
%
%  OUTPUT  (<OUT_ROOT>/study_results_<date>_n<N>/)
%    results.mat        trial table, spec, S (full test structs), R (master
%                       table), meta, settings
%    results_all.csv    the master table, one row per (question, metric)
%    trial_table.csv    the tidy per-trial table used for every test
%    figures/*.png      one dashboard per metric + overview figures
%    also: <OUT_ROOT>/latest.txt with the path of this run

%% ============================ USER SETTINGS ============================
EXPORT_DIR   = '';        % '' = auto-locate manifest.mat (see study_export_dir)
OUT_ROOT     = '';        % '' = <export dir>/analysis_results
ALPHA        = 0.05;      % significance level
N_BOOT       = 5000;      % bootstrap resamples for confidence intervals
MAKE_FIGURES = true;      % write PNG figures
%% =======================================================================

%% ---- load ----
[trial, meta] = load_study_table(EXPORT_DIR);
spec = study_metric_spec();
families = study_families();
[trial, fam] = composite_score(trial, spec);
if isempty(OUT_ROOT), OUT_ROOT = fullfile(meta.export_dir, 'analysis_results'); end
out_dir = fullfile(OUT_ROOT, sprintf('study_results_%s_n%d', ...
    string(datetime('now'), 'yyyyMMdd_HHmmss'), meta.n_complete));
mkdir(out_dir);
settings = struct('alpha', ALPHA, 'n_boot', N_BOOT, 'export_dir', meta.export_dir, ...
                  'out_dir', out_dir, 'run_time', datetime('now'));

line = repmat('=', 1, 78);
fprintf('\n%s\n TRIAGo study -- group analysis\n%s\n', line, line);
fprintf(' complete participants : %d  (%s)\n', meta.n_complete, strjoin(meta.complete_list, ', '));
for i = 1:height(meta.excluded)
    fprintf(' excluded              : %s  -- %s\n', meta.excluded.participant(i), meta.excluded.reason(i));
end
fprintf(' trials analysed       : %d  |  success %d  |  incidents noted %d\n', ...
    height(trial), nnz(trial.success), nnz(trial.incident));
fprintf(' output                : %s\n', out_dir);

%% ---- analysis items: every family metric + family scores + composite ----
items = spec(spec.family ~= "", {'name', 'label', 'unit', 'dir', 'family', 'cell_scope', 'mode_comparable', 'cv_safe'});
items.is_family = false(height(items), 1);
for f = 1:height(families)
    key = families.key(f);
    if fam.n_metrics(fam.key == key) == 0, continue; end
    scope = "all"; if key == "assistance_quality", scope = "blend_only"; end
    items(end + 1, :) = {"fam_" + key, families.label(f) + " score", "z", 1, key, scope, true, false, true}; %#ok<SAGROW>
end
items(end + 1, :) = {"composite", "Composite score", "z", 1, "composite", "all", true, false, true};

%% ---- run every test ----
CELLS6 = ["CF" "CB" "CFB" "JF" "JB" "JFB"]; MODES6 = ["C" "C" "C" "J" "J" "J"]; ASSIST6 = ["F" "B" "FB" "F" "B" "FB"];
CELLS4 = ["CB" "CFB" "JB" "JFB"];          MODES4 = ["C" "C" "J" "J"];          ASSIST4 = ["B" "FB" "B" "FB"];
S = struct('Q1', struct(), 'Q2', struct(), 'Q2_withinC', struct(), 'Q2_withinJ', struct(), ...
           'Q3', struct(), 'Q4', struct(), 'Q5', struct(), 'Q5_withinC', struct(), 'Q5_withinJ', struct(), ...
           'Q6', struct(), 'QW', struct());
rows = {};
for k = 1:height(items)
    it = items(k, :);
    nm = it.name; d = it.dir;
    args = {'alpha', ALPHA, 'n_boot', N_BOOT, 'dir', d};
    scopeNote = "";
    if it.cell_scope == "blend_only", scopeNote = "blending cells only"; end
    if it.cell_scope == "clutch_only", scopeNote = "CLUTCH cells only"; end

    % Q1 -- control mode
    if it.mode_comparable && it.cell_scope ~= "clutch_only"
        M = participant_means(trial, nm, "mode", ["C" "J"]);
        R = stat_paired2(M(:, 1), M(:, 2), "C", "J", args{:});
        S.Q1.(nm) = R; rows(end + 1, :) = row_of("Q1_mode", it, R, "C vs J", scopeNote); %#ok<SAGROW>
    end

    % Q2 -- assistance
    if it.cell_scope == "blend_only"
        M = participant_means(trial, nm, "assist", ["B" "FB"]);
        R = stat_paired2(M(:, 1), M(:, 2), "B", "FB", args{:});
        S.Q2.(nm) = R; rows(end + 1, :) = row_of("Q2_assist", it, R, "B vs FB", scopeNote); %#ok<SAGROW>
    elseif it.mode_comparable && it.cell_scope == "all"
        M = participant_means(trial, nm, "assist", ["F" "B" "FB"]);
        R = stat_rm_oneway(M, ["F" "B" "FB"], 'alpha', ALPHA, 'dir', d);
        S.Q2.(nm) = R; rows(end + 1, :) = row_of("Q2_assist", it, R, "F/B/FB", scopeNote); %#ok<SAGROW>
    else
        M = participant_means(trial, nm, "assist", ["F" "B" "FB"], trial.mode == "C");
        R = stat_rm_oneway(M, ["F" "B" "FB"], 'alpha', ALPHA, 'dir', d);
        S.Q2_withinC.(nm) = R; rows(end + 1, :) = row_of("Q2_assist", it, R, "F/B/FB within C", "CLUTCH cells only"); %#ok<SAGROW>
        if it.cell_scope == "all"
            M = participant_means(trial, nm, "assist", ["F" "B" "FB"], trial.mode == "J");
            R = stat_rm_oneway(M, ["F" "B" "FB"], 'alpha', ALPHA, 'dir', d);
            S.Q2_withinJ.(nm) = R; rows(end + 1, :) = row_of("Q2_assist", it, R, "F/B/FB within J", "JOYSTICK cells only"); %#ok<SAGROW>
        end
    end

    % Q3 -- cells
    if it.mode_comparable && it.cell_scope == "all"
        M = participant_means(trial, nm, "cell", CELLS6);
        R = stat_rm_twoway(M, CELLS6, MODES6, ASSIST6, 'alpha', ALPHA, 'dir', d);
        S.Q3.(nm) = R; rows(end + 1, :) = row_of("Q3_cell", it, R, "6 cells", scopeNote); %#ok<SAGROW>
    elseif it.cell_scope == "blend_only"
        M = participant_means(trial, nm, "cell", CELLS4);
        R = stat_rm_twoway(M, CELLS4, MODES4, ASSIST4, 'alpha', ALPHA, 'dir', d);
        S.Q3.(nm) = R; rows(end + 1, :) = row_of("Q3_cell", it, R, "4 blending cells", scopeNote); %#ok<SAGROW>
    end

    % Q4 -- consistency across modes
    if it.mode_comparable && it.cell_scope ~= "clutch_only"
        M = participant_means(trial, nm, "mode", ["C" "J"]);
        R = stat_consistency(M, ["C" "J"], 'alpha', ALPHA, 'n_boot', N_BOOT, 'cv_safe', it.cv_safe);
        S.Q4.(nm) = R; rows(end + 1, :) = row_of("Q4_consistency_mode", it, R, "C vs J", scopeNote); %#ok<SAGROW>
    end

    % Q5 -- consistency across assistance levels
    if it.cell_scope == "blend_only"
        M = participant_means(trial, nm, "assist", ["B" "FB"]);
        R = stat_consistency(M, ["B" "FB"], 'alpha', ALPHA, 'n_boot', N_BOOT, 'cv_safe', it.cv_safe);
        S.Q5.(nm) = R; rows(end + 1, :) = row_of("Q5_consistency_assist", it, R, "B vs FB", scopeNote); %#ok<SAGROW>
    elseif it.mode_comparable && it.cell_scope == "all"
        M = participant_means(trial, nm, "assist", ["F" "B" "FB"]);
        R = stat_consistency(M, ["F" "B" "FB"], 'alpha', ALPHA, 'n_boot', N_BOOT, 'cv_safe', it.cv_safe);
        S.Q5.(nm) = R; rows(end + 1, :) = row_of("Q5_consistency_assist", it, R, "F/B/FB", scopeNote); %#ok<SAGROW>
    else
        M = participant_means(trial, nm, "assist", ["F" "B" "FB"], trial.mode == "C");
        R = stat_consistency(M, ["F" "B" "FB"], 'alpha', ALPHA, 'n_boot', N_BOOT, 'cv_safe', it.cv_safe);
        S.Q5_withinC.(nm) = R; rows(end + 1, :) = row_of("Q5_consistency_assist", it, R, "F/B/FB within C", "CLUTCH cells only"); %#ok<SAGROW>
        if it.cell_scope == "all"
            M = participant_means(trial, nm, "assist", ["F" "B" "FB"], trial.mode == "J");
            R = stat_consistency(M, ["F" "B" "FB"], 'alpha', ALPHA, 'n_boot', N_BOOT, 'cv_safe', it.cv_safe);
            S.Q5_withinJ.(nm) = R; rows(end + 1, :) = row_of("Q5_consistency_assist", it, R, "F/B/FB within J", "JOYSTICK cells only"); %#ok<SAGROW>
        end
    end

    % Q6 -- learning / order (needs the same quantity in both modes)
    if it.mode_comparable && it.cell_scope ~= "clutch_only" && any(~isnan(trial.slot))
        R = stat_learning(trial, nm, args{:});
        S.Q6.(nm) = R; rows(end + 1, :) = row_of("Q6_learning", it, R, "slope over slots", scopeNote); %#ok<SAGROW>
    end

    % QW -- world
    M = participant_means(trial, nm, "world", ["rack" "shield"]);
    R = stat_paired2(M(:, 1), M(:, 2), "rack", "shield", args{:});
    S.QW.(nm) = R; rows(end + 1, :) = row_of("QW_world", it, R, "rack vs shield", scopeNote); %#ok<SAGROW>
end

R = cell2table(rows, 'VariableNames', {'question', 'family', 'metric', 'label', 'is_family_score', ...
    'levels', 'scope_note', 'test', 'n', 'statistic', 'df', 'p_raw', 'effect', 'effect_name', ...
    'ci_lo', 'ci_hi', 'best', 'interpretation'});
for c = ["question" "family" "metric" "label" "levels" "scope_note" "test" "effect_name" "best" "interpretation"]
    R.(c) = string(R.(c));
end
% Holm correction across the metrics of one family within one question
% (the family scores are the headline answers and are not part of any
% correction family).
R.p_holm = nan(height(R), 1);
for q = unique(R.question)'
    for f = unique(R.family)'
        sel = R.question == q & R.family == f & ~R.is_family_score;
        R.p_holm(sel) = holm_adjust(R.p_raw(sel));
    end
end
R.p_holm(R.is_family_score) = R.p_raw(R.is_family_score);
R.significant_raw = R.p_raw < ALPHA;
R.significant_holm = R.p_holm < ALPHA;
R = movevars(R, {'p_holm', 'significant_raw', 'significant_holm'}, 'After', 'p_raw');

%% ---- console verdict ----
fprintf('\n%s\n VERDICT PER FAMILY (family scores; z-units, higher = better)\n%s\n', line, line);
for f = 1:height(families)
    nm = "fam_" + families.key(f);
    if ~any(items.name == nm), continue; end
    fprintf('\n [%s]  %s\n', families.label(f), families.meaning(f));
    for q = ["Q1" "Q2" "Q3" "Q4" "Q5" "Q6" "QW"]
        if isfield(S.(q), nm), fprintf('   %-3s %s\n', q, S.(q).(nm).interpretation); end
    end
end
fprintf('\n%s\n SIGNIFICANT PER-METRIC RESULTS (Holm-corrected within family)\n%s\n', line, line);
sig = R(R.significant_holm & ~R.is_family_score & ~ismember(R.question, ["Q6_learning" "QW_world"]), :);
sig = sortrows(sig, {'question', 'family', 'p_holm'});
for i = 1:height(sig)
    fprintf(' %-22s %-22s %-32s Holm p = %.3f  %s\n', sig.question(i), sig.family(i), sig.label(i), sig.p_holm(i), sig.best(i));
end
if isempty(sig), fprintf(' (none)\n'); end

%% ---- save ----
save(fullfile(out_dir, 'results.mat'), 'trial', 'spec', 'families', 'fam', 'meta', 'S', 'R', 'items', 'settings');
writetable(R, fullfile(out_dir, 'results_all.csv'));
writetable(trial, fullfile(out_dir, 'trial_table.csv'));
writetable(meta.excluded, fullfile(out_dir, 'excluded_participants.csv'));
fid = fopen(fullfile(OUT_ROOT, 'latest.txt'), 'w'); fprintf(fid, '%s\n', out_dir); fclose(fid);
fprintf('\n saved -> %s\n', out_dir);

%% ---- figures ----
if MAKE_FIGURES
    fig_dir = fullfile(out_dir, 'figures');
    prev = get(0, 'DefaultFigureVisible'); set(0, 'DefaultFigureVisible', 'off');
    try
        for k = 1:height(items)
            it = table2struct(items(k, :));
            f = fig_metric_dashboard(trial, it, S);
            studyplot.save(f, fig_dir, "metric_" + it.name); close(f);
        end
        f = fig_family_overview(trial, S, items, families);    studyplot.save(f, fig_dir, "overview_families"); close(f);
        f = fig_consistency_overview(S, items, families);      studyplot.save(f, fig_dir, "overview_consistency"); close(f);
        f = fig_learning_overview(S, items, families);         studyplot.save(f, fig_dir, "overview_learning"); close(f);
        f = fig_success_incidents(trial);                      studyplot.save(f, fig_dir, "overview_success_incidents"); close(f);
        f = fig_world_overview(trial, S, items, families);     studyplot.save(f, fig_dir, "overview_world"); close(f);
        f = fig_cell_ranking(trial, S);                        studyplot.save(f, fig_dir, "overview_cell_ranking"); close(f);
    catch err
        set(0, 'DefaultFigureVisible', prev);
        rethrow(err);
    end
    set(0, 'DefaultFigureVisible', prev);
    fprintf(' figures -> %s\n', fig_dir);
end
fprintf('%s\n', line);

%% ============================ local functions ==========================
function r = row_of(question, it, R, levels, scopeNote)
% One master-table row from a result struct (fields differ per test family).
stat = NaN; df = NaN; ci = [NaN NaN]; best = "";
switch R.test_family
    case "paired2"
        stat = R.t; df = R.df; ci = R.ci_boot; best = R.better;
    case "rm_oneway"
        if R.recommended == "rm_anova", stat = R.F; df = R.df(1); else, stat = R.chi2; df = R.k - 1; end
        best = R.best;
    case "rm_twoway"
        stat = R.chi2; df = R.k - 1; best = R.best;
    case "consistency"
        best = R.most_consistent;
    case "learning"
        stat = R.slope_mean; ci = R.slope_ci; best = "";
end
if isfield(R, 'recommended'), test = R.recommended; else, test = R.test_family; end
r = {question, it.family, it.name, it.label, it.is_family, levels, scopeNote, test, R.n, stat, df, ...
     R.p, R.effect, R.effect_name, ci(1), ci(2), best, R.interpretation};
end
