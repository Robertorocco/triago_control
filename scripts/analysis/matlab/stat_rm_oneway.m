function R = stat_rm_oneway(wide, levels, opts)
%STAT_RM_ONEWAY One within-subject factor with k >= 3 levels (one value per participant per level).
%   R = STAT_RM_ONEWAY(WIDE, LEVELS, ...) where WIDE is an n x k matrix
%   (participants x levels) and LEVELS the k level names.
%
%   Parametric track:   repeated-measures ANOVA (fitrm/ranova) with the
%                       Greenhouse-Geisser sphericity correction, Mauchly's
%                       sphericity test, partial eta squared, Bonferroni
%                       post-hoc (multcompare).
%   Non-parametric:     Friedman test, Kendall's W (agreement of the
%                       participants' rankings), all pairwise Wilcoxon
%                       signed-rank tests with Holm correction, rank-biserial
%                       effect sizes.
%   The Holm-corrected pairwise Wilcoxon table is the primary post-hoc
%   result; the ANOVA multcompare table is kept as a cross-check.
%
%   Options: alpha (0.05), dir (+1/-1/0) to name the best level.

arguments
    wide (:,:) double
    levels (1,:) string
    opts.alpha (1,1) double = 0.05
    opts.dir (1,1) double = 0
end

k = numel(levels);
ok = all(~isnan(wide), 2);
X = wide(ok, :);
n = size(X, 1);
R = struct('test_family', "rm_oneway", 'levels', levels, 'n', n, 'k', k, ...
           'means', mean(X, 1), 'sds', std(X, 0, 1), 'medians', median(X, 1));

if n < 3
    R = empty_result(R, levels);
    R.interpretation = sprintf("Only %d complete participants: no test possible.", n);
    return
end
if all(abs(X(:) - X(1)) < 1e-12)
    R = empty_result(R, levels);
    R.interpretation = "Identical value in every participant and condition: nothing to test.";
    return
end

% ---------------- parametric: RM-ANOVA ----------------
R.p_anova = NaN; R.p_anova_gg = NaN; R.F = NaN; R.df = [NaN NaN]; R.eta2p = NaN;
R.p_mauchly = NaN; R.eps_gg = NaN; R.posthoc_anova = table();
try
    vn = matlab.lang.makeValidName(cellstr(levels));
    t = array2table(X, 'VariableNames', vn);
    within = table(categorical(levels(:)), 'VariableNames', {'Level'});
    rm = fitrm(t, sprintf('%s-%s ~ 1', vn{1}, vn{end}), 'WithinDesign', within);
    ra = ranova(rm, 'WithinModel', 'Level');
    ie = find(contains(string(ra.Properties.RowNames), "Level") & ~startsWith(string(ra.Properties.RowNames), "Error"), 1);
    ir = find(startsWith(string(ra.Properties.RowNames), "Error(Level)"), 1);
    R.F = ra.F(ie); R.df = [ra.DF(ie) ra.DF(ir)];
    R.p_anova = ra.pValue(ie); R.p_anova_gg = ra.pValueGG(ie);
    R.eta2p = ra.SumSq(ie) / (ra.SumSq(ie) + ra.SumSq(ir));
    try, mt = mauchly(rm); R.p_mauchly = mt.pValue(1); catch, end %#ok<CTCH>
    try, et = epsilon(rm); R.eps_gg = et.GreenhouseGeisser(1); catch, end %#ok<CTCH>
    try
        mc = multcompare(rm, 'Level', 'ComparisonType', 'bonferroni');
        R.posthoc_anova = mc;
    catch, end %#ok<CTCH>
catch err
    R.anova_error = string(err.message);
end

% ---------------- non-parametric: Friedman + pairwise Wilcoxon ----------------
R.p_friedman = NaN; R.chi2 = NaN; R.W = NaN;
if n >= 2 && any(std(X, 0, 1) > 0)
    [p_fr, tbl] = friedman(X, 1, 'off');
    R.p_friedman = p_fr; R.chi2 = tbl{2, 5}; R.W = kendalls_w(R.chi2, n, k);
end

pairs = nchoosek(1:k, 2);
np = size(pairs, 1);
P = table();
P.a = levels(pairs(:, 1))'; P.b = levels(pairs(:, 2))';
P.mean_diff = nan(np, 1); P.p_raw = nan(np, 1); P.r_rb = nan(np, 1); P.dz = nan(np, 1);
P.pct_a_greater = nan(np, 1);
for i = 1:np
    a = X(:, pairs(i, 1)); b = X(:, pairs(i, 2)); d = a - b;
    P.mean_diff(i) = mean(d);
    if all(d == 0), P.p_raw(i) = 1; else, P.p_raw(i) = signrank(a, b); end
    P.r_rb(i) = rank_biserial(a, b);
    P.dz(i) = mean(d) / max(std(d), eps);
    P.pct_a_greater(i) = 100 * mean(d > 0);
end
P.p_holm = holm_adjust(P.p_raw);
P.significant = P.p_holm < opts.alpha;
R.pairs = P;

% ---------------- which track to trust ----------------
nonnormal = false;
for i = 1:np
    pn = lillie_p(X(:, pairs(i, 1)) - X(:, pairs(i, 2)));
    nonnormal = nonnormal || (~isnan(pn) && pn < 0.05);
end
sphericity_violated = ~isnan(R.p_mauchly) && R.p_mauchly < 0.05;
if nonnormal || sphericity_violated || isnan(R.p_anova_gg)
    R.recommended = "friedman"; R.p = R.p_friedman; R.effect = R.W; R.effect_name = "Kendall W";
else
    R.recommended = "rm_anova"; R.p = R.p_anova_gg; R.effect = R.eta2p; R.effect_name = "partial eta2";
end
R.significant = R.p < opts.alpha;
R.alpha = opts.alpha;

% ---------------- ranking ----------------
oriented = R.means * (opts.dir + (opts.dir == 0));
[~, order] = sort(oriented, 'descend');
R.ranking = levels(order);
R.best = ""; if opts.dir ~= 0, R.best = levels(order(1)); end
R.interpretation = interpret(R, levels, opts);
end

function R = empty_result(R, levels)
R.p_anova = NaN; R.p_anova_gg = NaN; R.F = NaN; R.df = [NaN NaN]; R.eta2p = NaN;
R.p_mauchly = NaN; R.eps_gg = NaN; R.posthoc_anova = table();
R.p_friedman = NaN; R.chi2 = NaN; R.W = NaN; R.pairs = table();
R.recommended = "none"; R.p = NaN; R.effect = NaN; R.effect_name = "";
R.significant = false; R.ranking = levels; R.best = "";
end

function s = interpret(R, levels, opts)
if R.effect_name == "Kendall W", band = effect_band("W", R.effect); else, band = effect_band("eta2p", R.effect); end
sig = R.pairs(R.pairs.significant, :);
if R.significant
    s = sprintf("The %d conditions differ (%s, %s, %s = %.2f, %s effect). Ranking (best first): %s.", ...
        R.k, R.recommended, fmt_p(R.p), R.effect_name, R.effect, band, strjoin(R.ranking, " > "));
    if ~isempty(sig)
        parts = strings(height(sig), 1);
        for i = 1:height(sig)
            parts(i) = sprintf("%s vs %s (Holm p = %.3f)", sig.a(i), sig.b(i), sig.p_holm(i));
        end
        s = s + " Significant pairs: " + strjoin(parts, "; ") + ".";
    else
        s = s + " No single pair survives the Holm correction: the difference is spread over the levels.";
    end
    if opts.dir ~= 0, s = s + sprintf(" -> %s is best.", R.best); end
else
    s = sprintf("No significant difference among %s (%s, %s, %s = %.2f, %s effect).", ...
        strjoin(levels, "/"), R.recommended, fmt_p(R.p), R.effect_name, R.effect, band);
end
end
