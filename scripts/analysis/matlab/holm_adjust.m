function p_adj = holm_adjust(p)
%HOLM_ADJUST Holm-Bonferroni step-down adjusted p-values.
%   P_ADJ = HOLM_ADJUST(P) returns adjusted p-values for the family of m tests
%   in P (any shape; NaN entries are ignored and returned as NaN). Sort the
%   p-values ascending, multiply the i-th by (m - i + 1), enforce monotonicity
%   with a running maximum, cap at 1. Controls the family-wise error rate
%   without assuming independence and is uniformly more powerful than plain
%   Bonferroni.

p_adj = nan(size(p));
ok = ~isnan(p);
q = p(ok); q = q(:);
m = numel(q);
if m == 0, return, end
[qs, idx] = sort(q, 'ascend');
adj = qs .* (m - (1:m)' + 1);
adj = cummax(adj);
adj = min(adj, 1);
out = zeros(m, 1); out(idx) = adj;
p_adj(ok) = out;
end
