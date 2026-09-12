function R = stat_rm_twoway(wide, cells, modes, assists, opts)
%STAT_RM_TWOWAY Two within-subject factors (mode x assistance) on the cell grid.
%   R = STAT_RM_TWOWAY(WIDE, CELLS, MODES, ASSISTS, ...) where WIDE is n x k
%   (participants x cells), CELLS the k cell codes, MODES / ASSISTS the mode
%   and assistance level of each column (e.g. "C","J" and "F","B","FB").
%
%   Parametric track:  two-way repeated-measures ANOVA (fitrm/ranova) with
%                      main effects of Mode and Assist and their interaction,
%                      Greenhouse-Geisser corrected p-values, partial eta
%                      squared; simple effects (multcompare 'By') when the
%                      interaction is significant.
%   Non-parametric:    Friedman over the k cells (do the cells differ at
%                      all?), Kendall's W, all pairwise Wilcoxon signed-rank
%                      tests with Holm correction, and a ranking of the cells.
%
%   Options: alpha (0.05), dir (+1/-1/0).

arguments
    wide (:,:) double
    cells (1,:) string
    modes (1,:) string
    assists (1,:) string
    opts.alpha (1,1) double = 0.05
    opts.dir (1,1) double = 0
end

k = numel(cells);
ok = all(~isnan(wide), 2);
X = wide(ok, :);
n = size(X, 1);
R = struct('test_family', "rm_twoway", 'levels', cells, 'modes', modes, 'assists', assists, ...
           'n', n, 'k', k, 'means', mean(X, 1), 'sds', std(X, 0, 1), 'medians', median(X, 1));
R.effects = table();
R.simple_assist_by_mode = table(); R.simple_mode_by_assist = table();
R.p_friedman = NaN; R.chi2 = NaN; R.W = NaN; R.pairs = table();
R.p = NaN; R.effect = NaN; R.effect_name = ""; R.significant = false; R.recommended = "none";
R.ranking = cells; R.best = "";

if n < 3
    R.interpretation = sprintf("Only %d complete participants: no test possible.", n);
    return
end
if all(abs(X(:) - X(1)) < 1e-12)
    R.interpretation = "Identical value in every participant and cell: nothing to test.";
    return
end

% ---------------- parametric: 2-way RM-ANOVA ----------------
try
    vn = matlab.lang.makeValidName(cellstr(cells));
    t = array2table(X, 'VariableNames', vn);
    within = table(categorical(modes(:)), categorical(assists(:)), 'VariableNames', {'Mode', 'Assist'});
    rm = fitrm(t, sprintf('%s-%s ~ 1', vn{1}, vn{end}), 'WithinDesign', within);
    ra = ranova(rm, 'WithinModel', 'Mode*Assist');
    rn = string(ra.Properties.RowNames);
    names = ["Mode" "Assist" "Mode:Assist"];
    ne = numel(names);
    [Fv, df1, df2, pv, pgg, eta] = deal(nan(ne, 1));
    for i = 1:ne
        ie = find(rn == "(Intercept):" + names(i), 1);
        ir = find(rn == "Error(" + names(i) + ")", 1);
        if isempty(ie) || isempty(ir), continue; end
        Fv(i) = ra.F(ie); df1(i) = ra.DF(ie); df2(i) = ra.DF(ir);
        pv(i) = ra.pValue(ie); pgg(i) = ra.pValueGG(ie);
        eta(i) = ra.SumSq(ie) / (ra.SumSq(ie) + ra.SumSq(ir));
    end
    E = table(names(:), Fv, df1, df2, pv, pgg, eta, ...
              'VariableNames', {'effect', 'F', 'df1', 'df2', 'p', 'p_gg', 'eta2p'});
    R.effects = E;
    ia = find(E.effect == "Mode:Assist", 1);
    if ~isempty(ia) && E.p_gg(ia) < opts.alpha
        try, R.simple_assist_by_mode = multcompare(rm, 'Assist', 'By', 'Mode', 'ComparisonType', 'bonferroni'); catch, end %#ok<CTCH>
        try, R.simple_mode_by_assist = multcompare(rm, 'Mode', 'By', 'Assist', 'ComparisonType', 'bonferroni'); catch, end %#ok<CTCH>
    end
catch err
    R.anova_error = string(err.message);
end

% ---------------- non-parametric: Friedman on the cells + pairwise ----------------
if any(std(X, 0, 1) > 0)
    [p_fr, tbl] = friedman(X, 1, 'off');
    R.p_friedman = p_fr; R.chi2 = tbl{2, 5}; R.W = kendalls_w(R.chi2, n, k);
end
pairs = nchoosek(1:k, 2);
np = size(pairs, 1);
P = table();
P.a = cells(pairs(:, 1))'; P.b = cells(pairs(:, 2))';
P.mean_diff = nan(np, 1); P.p_raw = nan(np, 1); P.r_rb = nan(np, 1); P.dz = nan(np, 1);
for i = 1:np
    a = X(:, pairs(i, 1)); b = X(:, pairs(i, 2)); d = a - b;
    P.mean_diff(i) = mean(d);
    if all(d == 0), P.p_raw(i) = 1; else, P.p_raw(i) = signrank(a, b); end
    P.r_rb(i) = rank_biserial(a, b);
    P.dz(i) = mean(d) / max(std(d), eps);
end
P.p_holm = holm_adjust(P.p_raw);
P.significant = P.p_holm < opts.alpha;
R.pairs = P;

% Omnibus verdict: the Friedman test is the primary "do the cells differ"
% answer (robust at small n); the ANOVA effects table gives the factor-level
% decomposition.
R.recommended = "friedman";
R.p = R.p_friedman; R.effect = R.W; R.effect_name = "Kendall W";
R.significant = R.p < opts.alpha;
R.alpha = opts.alpha;

oriented = R.means * (opts.dir + (opts.dir == 0));
[~, order] = sort(oriented, 'descend');
R.ranking = cells(order);
if opts.dir ~= 0, R.best = cells(order(1)); end
R.interpretation = interpret(R, opts);
end

function s = interpret(R, opts)
s = "";
E = R.effects;
if ~isempty(E)
    parts = strings(height(E), 1);
    for i = 1:height(E)
        parts(i) = sprintf("%s %s (F(%g,%g) = %.2f, partial eta2 = %.2f, %s)", ...
            E.effect(i), ternary(E.p_gg(i) < opts.alpha, "significant", "n.s."), ...
            E.df1(i), E.df2(i), E.F(i), E.eta2p(i), fmt_p(E.p_gg(i)));
    end
    s = "RM-ANOVA: " + strjoin(parts, "; ") + ". ";
end
if R.significant
    s = s + sprintf("Friedman over the %d cells: %s, Kendall W = %.2f (%s agreement). Ranking (best first): %s.", ...
        R.k, fmt_p(R.p), R.W, effect_band("W", R.W), strjoin(R.ranking, " > "));
    sig = R.pairs(R.pairs.significant, :);
    if ~isempty(sig)
        pp = strings(height(sig), 1);
        for i = 1:height(sig), pp(i) = sprintf("%s vs %s (Holm p = %.3f)", sig.a(i), sig.b(i), sig.p_holm(i)); end
        s = s + " Significant pairs: " + strjoin(pp, "; ") + ".";
    else
        s = s + " No single pair survives the Holm correction.";
    end
    if opts.dir ~= 0, s = s + sprintf(" -> %s is best.", R.best); end
else
    s = s + sprintf("Friedman over the %d cells: %s, Kendall W = %.2f: the cells do not differ significantly.", ...
        R.k, fmt_p(R.p), R.W);
end
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
