function band = effect_band(kind, value)
%EFFECT_BAND Plain-language magnitude of an effect size.
%   BAND = EFFECT_BAND(KIND, VALUE) maps |VALUE| to "negligible" / "small" /
%   "medium" / "large" using the conventional thresholds for KIND:
%     "dz"    Cohen's dz (paired):           0.2 / 0.5 / 0.8
%     "r"     rank-biserial correlation:     0.1 / 0.3 / 0.5
%     "eta2p" partial eta squared:           0.01 / 0.06 / 0.14
%     "W"     Kendall's W (agreement):       0.1 / 0.3 / 0.5
%   These bands are conventions (Cohen 1988), not laws; the report explains
%   that the practical meaning depends on the metric.

switch kind
    case "dz",    th = [0.2 0.5 0.8];
    case "r",     th = [0.1 0.3 0.5];
    case "eta2p", th = [0.01 0.06 0.14];
    case "W",     th = [0.1 0.3 0.5];
    otherwise,    th = [0.2 0.5 0.8];
end
v = abs(value);
if isnan(v),        band = "n/a";
elseif v < th(1),   band = "negligible";
elseif v < th(2),   band = "small";
elseif v < th(3),   band = "medium";
else,               band = "large";
end
end
