function p = lillie_p(d)
%LILLIE_P Lilliefors normality p-value, NaN when not computable, warnings silenced.
%   The tabulated p-value saturates at 0.5 (clearly normal) and 0.001 (clearly
%   not); both are fine for the only use here, deciding between a t-test and
%   a rank test, so the out-of-range warnings are suppressed.
d = d(~isnan(d));
p = NaN;
if numel(d) < 4 || std(d) == 0, return, end
ws = warning('off', 'stats:lillietest:OutOfRangePHigh');
ws2 = warning('off', 'stats:lillietest:OutOfRangePLow');
try, [~, p] = lillietest(d); catch, end %#ok<CTCH>
warning(ws); warning(ws2);
end
