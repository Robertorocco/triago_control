function W = kendalls_w(chi2, n, k)
%KENDALLS_W Kendall's coefficient of concordance from a Friedman statistic.
%   W = KENDALLS_W(CHI2, N, K) with N participants (blocks) ranking K
%   conditions: W = CHI2 / (N * (K - 1)). W = 1 means every participant ranks
%   the conditions identically, W = 0 means no agreement at all. Rough bands:
%   < 0.1 negligible, 0.1-0.3 weak, 0.3-0.5 moderate, > 0.5 strong agreement.

W = chi2 / (n * (k - 1));
end
