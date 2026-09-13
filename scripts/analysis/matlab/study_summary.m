function out = study_summary(results_dir, opts)
%STUDY_SUMMARY Compact, publication-style summary of one analysis run.
%   OUT = STUDY_SUMMARY() summarises the most recent results folder;
%   STUDY_SUMMARY(RESULTS_DIR) a specific one. It writes <RESULTS_DIR>/summary/:
%     summary.pdf              one document, every figure below, vector
%     01_scorecard.png/.pdf    families x questions: winner + evidence
%     02_mode_families         forest plot, Joystick vs Clutch, family scores
%     03_assist_families       forest plot, F/B/FB contrasts, family scores
%     04_cells                 families x cells heat matrix + composite ranking
%     05_checks                practice, order bias, world, success
%     06_mode_metrics          forest plot, Joystick vs Clutch, every metric
%     07_assist_metrics        forest plot, F/B/FB contrasts, every metric
%     summary_table.csv        one row per family: winners, p-values, effects
%     summary.txt              plain-language verdict
%   Everything is generated from results.mat, so the summary stays valid for
%   any number of participants: re-run run_study_analysis, then this.
%   Options: show (true = leave the figures open), export_dir ('' = auto).

arguments
    results_dir (1,1) string = ""
    opts.show (1,1) logical = true
    opts.export_dir (1,1) string = ""
end

if strlength(results_dir) == 0
    root = fullfile(study_export_dir(char(opts.export_dir)), 'analysis_results');
    results_dir = string(strtrim(fileread(fullfile(root, 'latest.txt'))));
end
L = load(fullfile(results_dir, 'results.mat'), 'trial', 'families', 'meta', 'S', 'R', 'items', 'settings');
trial = L.trial; families = L.families; meta = L.meta; S = L.S; items = L.items; settings = L.settings;

sdir = fullfile(results_dir, 'summary');
if ~isfolder(sdir), mkdir(sdir); end
pdf = fullfile(sdir, 'summary.pdf');
if isfile(pdf), delete(pdf); end

prev = get(0, 'DefaultFigureVisible');
if ~opts.show, set(0, 'DefaultFigureVisible', 'off'); end
figs = gobjects(0, 1);
try
    figs(end + 1) = fig_summary_scorecard(S, items, families, meta, settings);         studyplot.export(figs(end), fullfile(sdir, "01_scorecard"), 'append_to', pdf);
    figs(end + 1) = fig_summary_forest(trial, S, items, families, "Q1", "family");     studyplot.export(figs(end), fullfile(sdir, "02_mode_families"), 'append_to', pdf);
    figs(end + 1) = fig_summary_forest(trial, S, items, families, "Q2", "family");     studyplot.export(figs(end), fullfile(sdir, "03_assist_families"), 'append_to', pdf);
    figs(end + 1) = fig_summary_cells(trial, S, items, families);                       studyplot.export(figs(end), fullfile(sdir, "04_cells"), 'append_to', pdf);
    figs(end + 1) = fig_summary_checks(trial, S);                                       studyplot.export(figs(end), fullfile(sdir, "05_checks"), 'append_to', pdf);
    figs(end + 1) = fig_summary_forest(trial, S, items, families, "Q1", "metric", L.R); studyplot.export(figs(end), fullfile(sdir, "06_mode_metrics"), 'append_to', pdf);
    figs(end + 1) = fig_summary_forest(trial, S, items, families, "Q2", "metric", L.R); studyplot.export(figs(end), fullfile(sdir, "07_assist_metrics"), 'append_to', pdf);
catch err
    set(0, 'DefaultFigureVisible', prev);
    rethrow(err);
end
set(0, 'DefaultFigureVisible', prev);
if ~opts.show, close(figs); end

T = summary_table(S, items, families, settings.alpha);
writetable(T, fullfile(sdir, 'summary_table.csv'));
txt = verdict_text(S, items, families, meta, settings);
fid = fopen(fullfile(sdir, 'summary.txt'), 'w'); fprintf(fid, '%s', txt); fclose(fid);
fprintf('%s\n', txt);
fprintf('summary -> %s\n', sdir);

out = struct('dir', sdir, 'pdf', pdf, 'table', T, 'text', txt);
end

% ======================================================================
function T = summary_table(S, items, families, alpha)
it = items(items.is_family, :);
it = [it(it.name == "composite", :); it(it.name ~= "composite", :)];
nr = height(it);
T = table();
T.family = arrayfun(@(k) studyplot.famlabel(families, k), it.family);
[T.mode_winner, T.assist_best, T.assist_ranking, T.best_cell, T.cell_ranking, T.consistent_mode, T.consistent_assist, T.practice, T.world_easier] = deal(strings(nr, 1));
[T.mode_p, T.mode_effect, T.assist_p, T.assist_effect, T.cell_p, T.cell_W, T.consistent_mode_p, T.consistent_assist_p, T.practice_slope, T.practice_p, T.order_bias_p, T.world_p] = deal(nan(nr, 1));
for i = 1:nr
    nm = it.name(i);
    if isfield(S.Q1, nm), R = S.Q1.(nm); T.mode_p(i) = R.p; T.mode_effect(i) = R.effect; T.mode_winner(i) = winner(R.better, R.significant); end
    if isfield(S.Q2, nm)
        R = S.Q2.(nm); T.assist_p(i) = R.p; T.assist_effect(i) = R.effect;
        if R.test_family == "paired2", T.assist_best(i) = winner(R.better, R.significant); T.assist_ranking(i) = "B vs FB only";
        else, T.assist_best(i) = winner(R.best, R.significant); T.assist_ranking(i) = strjoin(R.ranking, " > "); end
    end
    if isfield(S.Q3, nm), R = S.Q3.(nm); T.cell_p(i) = R.p; T.cell_W(i) = R.W; T.best_cell(i) = winner(R.best, R.significant); T.cell_ranking(i) = strjoin(R.ranking, " > "); end
    if isfield(S.Q4, nm), R = S.Q4.(nm); T.consistent_mode_p(i) = R.p; T.consistent_mode(i) = winner(R.most_consistent, R.significant); end
    if isfield(S.Q5, nm), R = S.Q5.(nm); T.consistent_assist_p(i) = R.p; T.consistent_assist(i) = winner(R.most_consistent, R.significant); end
    if isfield(S.Q6, nm)
        R = S.Q6.(nm); T.practice_slope(i) = R.slope_mean; T.practice_p(i) = R.p_slope_wilcoxon; T.order_bias_p(i) = R.p_order_ranksum;
        if R.significant, if R.slope_mean > 0, T.practice(i) = "improves"; else, T.practice(i) = "worsens"; end, else, T.practice(i) = "no trend"; end
        if ~isnan(R.p_order_ranksum) && R.p_order_ranksum < alpha, T.practice(i) = T.practice(i) + " + ORDER BIAS"; end
    end
    if isfield(S.QW, nm), R = S.QW.(nm); T.world_p(i) = R.p; T.world_easier(i) = winner(R.better, R.significant); end
end
end

function s = winner(name, significant)
if significant && strlength(name) > 0, s = string(name); else, s = "n.s."; end
end

% ======================================================================
function txt = verdict_text(S, items, families, meta, settings)
it = items(items.is_family, :);
it = [it(it.name == "composite", :); it(it.name ~= "composite", :)];
L = strings(0, 1);
L(end + 1) = "TRIAGo shared-autonomy user study -- summary of the group analysis";
L(end + 1) = sprintf("run: %s   participants: %d complete (%s)", string(settings.run_time), meta.n_complete, strjoin(meta.complete_list, ", "));
for i = 1:height(meta.excluded), L(end + 1) = sprintf("excluded: %s -- %s", meta.excluded.participant(i), meta.excluded.reason(i)); end
L(end + 1) = "";
L(end + 1) = "WHAT IS ESTABLISHED (significant at alpha = " + string(settings.alpha) + ", family scores)";
L(end + 1) = "----------------------------------------------------------------------";
open = strings(0, 1);
for i = 1:height(it)
    nm = it.name(i); lab = studyplot.famlabel(families, it.family(i)); if nm == "composite", lab = "COMPOSITE"; end
    if isfield(S.Q1, nm)
        R = S.Q1.(nm);
        if R.significant, L(end + 1) = sprintf("* %s: %s is better than %s  (%s, |%s| = %.2f, %s effect)", lab, studyplot.levelname(R.better), studyplot.levelname(other(R.better)), fmt_p(R.p), R.effect_name, abs(R.effect), effect_band(bandkind(R.effect_name), R.effect));
        else, open(end + 1) = sprintf("* %s: Clutch vs Joystick not separated yet  (%s, |%s| = %.2f)", lab, fmt_p(R.p), R.effect_name, abs(R.effect)); end
    end
    if isfield(S.Q2, nm)
        R = S.Q2.(nm);
        if R.test_family == "paired2"
            if R.significant, L(end + 1) = sprintf("* %s: %s is better than %s  (%s, |%s| = %.2f)", lab, R.better, other2(R.better), fmt_p(R.p), R.effect_name, abs(R.effect));
            else, open(end + 1) = sprintf("* %s: B vs FB not separated yet  (%s)", lab, fmt_p(R.p)); end
        else
            if R.significant
                sig = R.pairs(R.pairs.significant, :); pp = strings(height(sig), 1);
                for q = 1:height(sig), if sig.mean_diff(q) * 1 > 0, pp(q) = sig.a(q) + " > " + sig.b(q); else, pp(q) = sig.b(q) + " > " + sig.a(q); end, end
                L(end + 1) = sprintf("* %s: assistance matters, ranking %s  (%s, %s = %.2f); confirmed pairs: %s", lab, strjoin(R.ranking, " > "), fmt_p(R.p), R.effect_name, R.effect, ternary(isempty(pp), "none after Holm", strjoin(pp, ", ")));
            else, open(end + 1) = sprintf("* %s: F / B / FB not separated yet  (%s)", lab, fmt_p(R.p)); end
        end
    end
    if isfield(S.Q3, nm)
        R = S.Q3.(nm);
        if R.significant, L(end + 1) = sprintf("* %s: best cell %s, ranking %s  (Friedman %s, W = %.2f)", lab, R.best, strjoin(R.ranking, " > "), fmt_p(R.p), R.W); end
    end
    if isfield(S.Q4, nm) && S.Q4.(nm).significant, L(end + 1) = sprintf("* %s: %s gives more consistent results across participants  (Pitman-Morgan %s)", lab, studyplot.levelname(S.Q4.(nm).most_consistent), fmt_p(S.Q4.(nm).p)); end
    if isfield(S.Q5, nm) && S.Q5.(nm).significant, L(end + 1) = sprintf("* %s: %s gives more consistent results across participants  (Pitman-Morgan %s)", lab, S.Q5.(nm).most_consistent, fmt_p(S.Q5.(nm).p)); end
end
L(end + 1) = "";
L(end + 1) = "NOT DEMONSTRATED YET (more participants may or may not change this)";
L(end + 1) = "----------------------------------------------------------------------";
L = [L(:); open(:)];
L(end + 1) = "";
L(end + 1) = "VALIDITY CHECKS";
L(end + 1) = "----------------------------------------------------------------------";
for i = 1:height(it)
    nm = it.name(i); lab = studyplot.famlabel(families, it.family(i)); if nm == "composite", lab = "COMPOSITE"; end
    if ~isfield(S.Q6, nm), continue, end
    R = S.Q6.(nm);
    flag = "";
    if R.significant, flag = flag + sprintf("practice trend (slope %.3g/slot, %s); ", R.slope_mean, fmt_p(R.p_slope_wilcoxon)); end
    if ~isnan(R.p_order_ranksum) && R.p_order_ranksum < settings.alpha, flag = flag + sprintf("ORDER BIAS on the mode comparison (rank-sum %s); ", fmt_p(R.p_order_ranksum)); end
    if strlength(flag) == 0, flag = "no practice trend, no order bias"; end
    L(end + 1) = sprintf("* %s: %s", lab, flag);
end
if isfield(S.QW, "composite")
    R = S.QW.("composite");
    L(end + 1) = sprintf("* World: %s  (paired within participant, so it does not bias the comparisons above)", ternary(R.significant, R.better + " is the easier scene, " + fmt_p(R.p), "no significant difference between rack and shield"));
end
txt = strjoin(L(:), newline) + newline;
end

function o = other(m), if m == "C", o = "J"; else, o = "C"; end, end
function o = other2(m), if m == "B", o = "FB"; else, o = "B"; end, end
function k = bandkind(effect_name)
switch effect_name
    case "Cohen dz", k = "dz"; case "rank-biserial r", k = "r"; case "partial eta2", k = "eta2p"; case "Kendall W", k = "W"; otherwise, k = "dz";
end
end
function out = ternary(c, a, b), if c, out = a; else, out = b; end, end
