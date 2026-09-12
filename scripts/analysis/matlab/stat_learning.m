function R = stat_learning(trial, metric, opts)
%STAT_LEARNING Learning / order effects of one metric along the experiment.
%   R = STAT_LEARNING(TRIAL, METRIC, ...) uses the schedule columns of TRIAL
%   (slot 1-6, trial_index 1-12, mode_order) added by LOAD_STUDY_TABLE.
%
%   1. Trend over slots. For every participant the metric is averaged over
%      the two worlds of each slot (6 points) and a straight line is fitted:
%      slope = change of the metric per slot. The n slopes are tested against
%      zero with a paired-style one-sample t-test and a Wilcoxon signed-rank
%      test, with a bootstrap CI of the mean slope. A Friedman test across the
%      6 slot positions is the trend-free omnibus check.
%   2. Block effect. Mean of slots 1-3 (first control mode met) versus mean
%      of slots 4-6 (second mode), paired across participants.
%   3. Mode-order check. Because control mode is blocked, the "mode effect"
%      of a participant (J - C) could be contaminated by practice. The J - C
%      difference of the C-first participants is compared with that of the
%      J-first participants (rank-sum test): if it does not differ, order did
%      not bias the mode comparison.
%   The slot is averaged over the two worlds because within a slot the
%   worlds were always run in the same order (shield, then rack), so the raw
%   trial index would confound practice with world difficulty.
%
%   Options: alpha (0.05), n_boot (5000), dir (+1/-1/0).

arguments
    trial table
    metric (1,1) string
    opts.alpha (1,1) double = 0.05
    opts.n_boot (1,1) double = 5000
    opts.dir (1,1) double = 0
end

P = unique(trial.participant, 'stable');
n = numel(P);
S = nan(n, 6);          % participant x slot (mean over worlds)
Tr = nan(n, 12);        % participant x trial index
modeOrder = strings(n, 1);
modeDiff = nan(n, 1);   % J - C per participant
for i = 1:n
    rows = trial.participant == P(i);
    t = trial(rows, :);
    for s = 1:6
        v = t.(metric)(t.slot == s);
        if ~isempty(v), S(i, s) = mean(v, 'omitnan'); end
    end
    for j = 1:12
        v = t.(metric)(t.trial_index == j);
        if ~isempty(v), Tr(i, j) = mean(v, 'omitnan'); end
    end
    mo = t.mode_order(strlength(t.mode_order) > 0);
    if ~isempty(mo), modeOrder(i) = mo(1); end
    mJ = mean(t.(metric)(t.mode == "J"), 'omitnan');
    mC = mean(t.(metric)(t.mode == "C"), 'omitnan');
    modeDiff(i) = mJ - mC;
end

R = struct('test_family', "learning", 'metric', metric, 'n', n, 'participants', P);
R.slot_matrix = S; R.trial_matrix = Tr;
R.slot_mean = mean(S, 1, 'omitnan'); R.slot_sd = std(S, 0, 1, 'omitnan');
R.trial_mean = mean(Tr, 1, 'omitnan'); R.trial_sd = std(Tr, 0, 1, 'omitnan');

% ---- 1. slopes ----
slopes = nan(n, 1);
for i = 1:n
    y = S(i, :); okk = ~isnan(y);
    if nnz(okk) >= 4
        c = polyfit(find(okk), y(okk), 1); slopes(i) = c(1);
    end
end
R.slopes = slopes;
sl = slopes(~isnan(slopes));
R.n_slopes = numel(sl);
R.slope_mean = mean(sl); R.slope_sd = std(sl);
R.p_slope_t = NaN; R.p_slope_wilcoxon = NaN; R.slope_ci = [NaN NaN]; R.slope_dz = NaN;
if numel(sl) >= 3 && std(sl) > 0
    [~, R.p_slope_t] = ttest(sl);
    R.p_slope_wilcoxon = signrank(sl);
    R.slope_dz = mean(sl) / std(sl);
    rng(12345, 'twister');
    b = bootstrp(opts.n_boot, @mean, sl);
    R.slope_ci = prctile(b, 100 * [opts.alpha / 2, 1 - opts.alpha / 2]);
end
R.p_friedman_slots = NaN; R.W_slots = NaN;
Sc = S(all(~isnan(S), 2), :);
if size(Sc, 1) >= 2 && any(std(Sc, 0, 1) > 0)
    [p_fr, tbl] = friedman(Sc, 1, 'off');
    R.p_friedman_slots = p_fr; R.W_slots = kendalls_w(tbl{2, 5}, size(Sc, 1), 6);
end

% ---- 2. block effect ----
b1 = mean(S(:, 1:3), 2, 'omitnan'); b2 = mean(S(:, 4:6), 2, 'omitnan');
R.block = stat_paired2(b1, b2, "slots 1-3", "slots 4-6", 'alpha', opts.alpha, 'n_boot', opts.n_boot, 'dir', opts.dir);

% ---- 3. mode-order check ----
gC = modeDiff(modeOrder == "C_first"); gJ = modeDiff(modeOrder == "J_first");
gC = gC(~isnan(gC)); gJ = gJ(~isnan(gJ));
R.mode_order_n = [numel(gC) numel(gJ)];
R.mode_diff_C_first = gC; R.mode_diff_J_first = gJ;
R.p_order_ranksum = NaN; R.p_order_t = NaN;
if numel(gC) >= 2 && numel(gJ) >= 2
    R.p_order_ranksum = ranksum(gC, gJ);
    [~, R.p_order_t] = ttest2(gC, gJ);
end
R.mode_diff_mean_C_first = mean(gC); R.mode_diff_mean_J_first = mean(gJ);

% ---- slopes by mode order ----
sC = slopes(modeOrder == "C_first"); sJ = slopes(modeOrder == "J_first");
sC = sC(~isnan(sC)); sJ = sJ(~isnan(sJ));
R.p_slope_by_order = NaN;
if numel(sC) >= 2 && numel(sJ) >= 2, R.p_slope_by_order = ranksum(sC, sJ); end

R.alpha = opts.alpha;
R.p = R.p_slope_wilcoxon; R.effect = R.slope_dz; R.effect_name = "slope dz";
R.significant = R.p < opts.alpha;
R.interpretation = interpret(R, opts);
end

function s = interpret(R, opts)
if isnan(R.p_slope_wilcoxon)
    s = "Not enough scheduled trials to estimate a trend.";
    return
end
if opts.dir == 0
    trend = "changes";
elseif sign(R.slope_mean) * opts.dir > 0
    trend = "improves";
else
    trend = "worsens";
end
if R.significant
    s = sprintf("The metric %s over the session: mean slope %.3g per slot (95%% CI %.3g to %.3g), Wilcoxon %s, t-test %s, dz = %.2f. Friedman across slots %s.", ...
        trend, R.slope_mean, R.slope_ci(1), R.slope_ci(2), fmt_p(R.p_slope_wilcoxon), fmt_p(R.p_slope_t), R.slope_dz, fmt_p(R.p_friedman_slots));
else
    s = sprintf("No significant trend over the session: mean slope %.3g per slot (95%% CI %.3g to %.3g), Wilcoxon %s. Friedman across slots %s.", ...
        R.slope_mean, R.slope_ci(1), R.slope_ci(2), fmt_p(R.p_slope_wilcoxon), fmt_p(R.p_friedman_slots));
end
if ~isnan(R.p_order_ranksum)
    if R.p_order_ranksum < opts.alpha
        s = s + sprintf(" WARNING: the Joystick-minus-Clutch difference depends on which mode came first (C-first %.3g vs J-first %.3g, rank-sum %s): the mode comparison is partly an order effect.", ...
            R.mode_diff_mean_C_first, R.mode_diff_mean_J_first, fmt_p(R.p_order_ranksum));
    else
        s = s + sprintf(" The mode difference does not depend on mode order (C-first %.3g vs J-first %.3g, rank-sum %s): no order bias detected.", ...
            R.mode_diff_mean_C_first, R.mode_diff_mean_J_first, fmt_p(R.p_order_ranksum));
    end
end
end
