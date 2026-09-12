function R = stat_paired2(x, y, nameX, nameY, opts)
%STAT_PAIRED2 Within-subject comparison of two conditions (one value per participant each).
%   R = STAT_PAIRED2(X, Y, NAMEX, NAMEY, ...) compares paired vectors X and Y
%   (participant i contributes X(i) and Y(i)). Both a parametric and a
%   non-parametric test are always computed and reported; the assumption
%   check on the differences decides which one is flagged as recommended.
%
%   Parametric:     paired t-test on d = X - Y; effect size Cohen's dz =
%                   mean(d)/std(d); 95% t-based CI of mean(d).
%   Non-parametric: Wilcoxon signed-rank test (exact for n <= 15); effect
%                   size matched-pairs rank-biserial r; sign test and the
%                   share of participants with X > Y as the plainest summary.
%   Robust CI:      percentile bootstrap CI of mean(d) (N_BOOT resamples).
%   Assumption:     Lilliefors normality test on d (p < 0.05 -> recommend
%                   Wilcoxon).
%
%   Options: alpha (0.05), n_boot (5000), dir (+1 higher better, -1 lower
%   better, 0 none) used to say which condition is "better".

arguments
    x (:,1) double
    y (:,1) double
    nameX (1,1) string
    nameY (1,1) string
    opts.alpha (1,1) double = 0.05
    opts.n_boot (1,1) double = 5000
    opts.dir (1,1) double = 0
end

ok = ~isnan(x) & ~isnan(y);
x = x(ok); y = y(ok);
d = x - y;
n = numel(d);
R = struct('test_family', "paired2", 'levels', [nameX nameY], 'n', n, ...
           'mean_x', mean(x), 'mean_y', mean(y), 'sd_x', std(x), 'sd_y', std(y), ...
           'median_x', median(x), 'median_y', median(y), 'mean_diff', mean(d), 'sd_diff', std(d));

if n < 3
    R.p_t = NaN; R.t = NaN; R.df = NaN; R.ci_t = [NaN NaN]; R.dz = NaN;
    R.p_wilcoxon = NaN; R.signedrank = NaN; R.r_rb = NaN; R.p_sign = NaN;
    R.pct_x_greater = NaN; R.ci_boot = [NaN NaN]; R.p_normality = NaN;
    R.recommended = "none"; R.p = NaN; R.effect = NaN; R.effect_name = "";
    R.better = ""; R.significant = false;
    R.interpretation = sprintf("Only %d complete pairs: no test possible.", n);
    return
end

% --- parametric ---
[~, p_t, ci_t, st] = ttest(x, y, 'Alpha', opts.alpha);
R.p_t = p_t; R.t = st.tstat; R.df = st.df; R.ci_t = ci_t(:)';
R.dz = mean(d) / std(d);

% --- non-parametric ---
if all(d == 0)
    R.p_wilcoxon = 1; R.signedrank = 0;
else
    [p_w, ~, sw] = signrank(x, y, 'Alpha', opts.alpha);
    R.p_wilcoxon = p_w; R.signedrank = sw.signedrank;
end
R.r_rb = rank_biserial(x, y);
if all(d == 0), R.p_sign = 1; else, R.p_sign = signtest(x, y); end
R.pct_x_greater = 100 * mean(d > 0);
R.pct_y_greater = 100 * mean(d < 0);

% --- bootstrap CI of the mean difference ---
if n >= 4 && std(d) > 0
    rng(12345, 'twister');
    b = bootstrp(opts.n_boot, @mean, d);
    R.ci_boot = prctile(b, 100 * [opts.alpha/2, 1 - opts.alpha/2]);
else
    R.ci_boot = [NaN NaN];
end

% --- assumption check ---
R.p_normality = lillie_p(d);
if ~isnan(R.p_normality) && R.p_normality < 0.05
    R.recommended = "wilcoxon"; R.p = R.p_wilcoxon; R.effect = R.r_rb; R.effect_name = "rank-biserial r";
else
    R.recommended = "ttest"; R.p = R.p_t; R.effect = R.dz; R.effect_name = "Cohen dz";
end
R.significant = R.p < opts.alpha;

% --- which level is better ---
R.better = "";
if opts.dir ~= 0 && R.mean_diff ~= 0
    if sign(R.mean_diff) * opts.dir > 0, R.better = nameX; else, R.better = nameY; end
end
R.alpha = opts.alpha;
R.interpretation = interpret(R, nameX, nameY, opts.dir);
end

function s = interpret(R, nameX, nameY, dir)
if R.effect_name == "Cohen dz", band = effect_band("dz", R.effect); else, band = effect_band("r", R.effect); end
if R.mean_diff > 0, hi = nameX; lo = nameY; else, hi = nameY; lo = nameX; end
verdict = "";
if dir ~= 0 && strlength(R.better) > 0
    verdict = sprintf(" -> %s is better", R.better);
end
if R.sd_diff == 0
    s = sprintf("%s and %s are identical in every participant: nothing to test.", nameX, nameY);
elseif R.significant
    s = sprintf("%s is higher than %s (mean diff %.3g, %s, %s effect |%s| = %.2f, %d/%d participants)%s.", ...
        hi, lo, abs(R.mean_diff), fmt_p(R.p), band, R.effect_name, abs(R.effect), ...
        round(max(R.pct_x_greater, R.pct_y_greater) / 100 * R.n), R.n, verdict);
else
    s = sprintf("No significant difference between %s and %s (%s - %s = %.3g, %s, %s effect %s = %.2f).", ...
        nameX, nameY, nameX, nameY, R.mean_diff, fmt_p(R.p), band, R.effect_name, R.effect);
end
end
