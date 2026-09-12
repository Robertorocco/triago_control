function r = rank_biserial(x, y)
%RANK_BISERIAL Matched-pairs rank-biserial correlation (effect size for signrank).
%   R = RANK_BISERIAL(X, Y) ranks |X - Y| over the non-zero, non-NaN pairs and
%   returns (sum of ranks where X > Y - sum of ranks where X < Y) / total rank
%   sum. R = +1 when every pair has X > Y, -1 when every pair has X < Y, 0
%   when the two directions balance. Works for any N (MATLAB's signrank only
%   exposes a z statistic for N > 15, so it is computed directly here).

d = x(:) - y(:);
d = d(~isnan(d) & d ~= 0);
if isempty(d), r = NaN; return, end
rk = tiedrank(abs(d));
r = (sum(rk(d > 0)) - sum(rk(d < 0))) / sum(rk);
end
