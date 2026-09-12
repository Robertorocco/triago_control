function R = stat_consistency(wide, levels, opts)
%STAT_CONSISTENCY Which condition gives the most similar results across participants?
%   R = STAT_CONSISTENCY(WIDE, LEVELS, ...) with WIDE n x k (participants x
%   levels, one value per participant per level) quantifies, per level, the
%   between-participant dispersion of that level's values:
%     sd     standard deviation across participants (raw units)
%     iqr    inter-quartile range
%     cv     coefficient of variation sd/|mean| (only when cv_safe)
%   with percentile-bootstrap confidence intervals, and compares the
%   dispersions of every pair of levels with the Pitman-Morgan test for equal
%   variances of paired samples (the correlation between X+Y and X-Y is zero
%   if and only if Var(X) = Var(Y)), Holm-corrected. Kendall's W from the
%   Friedman test is added as the agreement of participants on the RANKING of
%   the levels (a different notion of consistency: do people agree which
%   level is best?).
%
%   The most consistent level is the one with the smallest SD; the
%   comparison is significant only if its Pitman-Morgan test survives Holm.
%
%   Options: alpha (0.05), n_boot (5000), cv_safe (true).

arguments
    wide (:,:) double
    levels (1,:) string
    opts.alpha (1,1) double = 0.05
    opts.n_boot (1,1) double = 5000
    opts.cv_safe (1,1) logical = true
end

k = numel(levels);
ok = all(~isnan(wide), 2);
X = wide(ok, :);
n = size(X, 1);
R = struct('test_family', "consistency", 'levels', levels, 'n', n, 'k', k, 'cv_safe', opts.cv_safe);

L = table();
L.level = levels(:);
L.mean = mean(X, 1)'; L.sd = std(X, 0, 1)'; L.iqr = iqr(X, 1)';
L.cv = L.sd ./ abs(L.mean);
if ~opts.cv_safe, L.cv(:) = NaN; end
L.sd_ci_lo = nan(k, 1); L.sd_ci_hi = nan(k, 1); L.cv_ci_lo = nan(k, 1); L.cv_ci_hi = nan(k, 1);
if n >= 4
    rng(12345, 'twister');
    for j = 1:k
        b = bootstrp(opts.n_boot, @std, X(:, j));
        L.sd_ci_lo(j) = prctile(b, 100 * opts.alpha / 2); L.sd_ci_hi(j) = prctile(b, 100 * (1 - opts.alpha / 2));
        if opts.cv_safe
            bc = bootstrp(opts.n_boot, @(v) std(v) / abs(mean(v)), X(:, j));
            L.cv_ci_lo(j) = prctile(bc, 100 * opts.alpha / 2); L.cv_ci_hi(j) = prctile(bc, 100 * (1 - opts.alpha / 2));
        end
    end
end
R.levels_table = L;

% ---- pairwise Pitman-Morgan ----
pairs = nchoosek(1:k, 2);
np = size(pairs, 1);
P = table();
P.a = levels(pairs(:, 1))'; P.b = levels(pairs(:, 2))';
P.sd_a = L.sd(pairs(:, 1)); P.sd_b = L.sd(pairs(:, 2));
P.r_pm = nan(np, 1); P.p_raw = nan(np, 1);
for i = 1:np
    x = X(:, pairs(i, 1)); y = X(:, pairs(i, 2));
    if n >= 3 && std(x + y) > 0 && std(x - y) > 0
        [r, p] = corr(x + y, x - y);
        P.r_pm(i) = r; P.p_raw(i) = p;
    end
end
P.p_holm = holm_adjust(P.p_raw);
P.significant = P.p_holm < opts.alpha;
P.more_consistent = strings(np, 1);
for i = 1:np
    if P.sd_a(i) < P.sd_b(i), P.more_consistent(i) = P.a(i); else, P.more_consistent(i) = P.b(i); end
end
R.pairs = P;

% ---- ranking agreement ----
R.W = NaN; R.p_friedman = NaN;
if n >= 2 && k >= 2 && any(std(X, 0, 1) > 0)
    [p_fr, tbl] = friedman(X, 1, 'off');
    R.p_friedman = p_fr; R.W = kendalls_w(tbl{2, 5}, n, k);
end

[~, imin] = min(L.sd);
R.most_consistent = levels(imin);
sigWith = P.significant & (P.more_consistent == R.most_consistent);
R.significant = any(sigWith);
R.p = min(P.p_holm(P.more_consistent == R.most_consistent), [], 'omitnan');
if isempty(R.p), R.p = NaN; end
R.alpha = opts.alpha;
% Effect size reported for this test: ratio of the largest to the smallest SD.
R.effect = max(L.sd) / max(min(L.sd), eps); R.effect_name = "SD ratio (max/min)";
R.recommended = "pitman_morgan";
R.interpretation = interpret(R, L);
end

function s = interpret(R, L)
parts = strings(height(L), 1);
for j = 1:height(L)
    if isnan(L.cv(j))
        parts(j) = sprintf("%s SD = %.3g", L.level(j), L.sd(j));
    else
        parts(j) = sprintf("%s SD = %.3g (CV %.0f%%)", L.level(j), L.sd(j), 100 * L.cv(j));
    end
end
s = "Between-participant spread: " + strjoin(parts, ", ") + ". ";
if R.significant
    s = s + sprintf("%s is the most consistent condition and its spread is significantly smaller (Pitman-Morgan, Holm %s). ", R.most_consistent, fmt_p(R.p));
else
    s = s + sprintf("%s has the smallest spread, but the difference in spread is not significant (Pitman-Morgan, Holm %s). ", R.most_consistent, fmt_p(R.p));
end
if ~isnan(R.W)
    s = s + sprintf("Agreement of participants on the ranking of the conditions: Kendall W = %.2f (%s).", R.W, effect_band("W", R.W));
end
end
