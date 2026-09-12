function s = fmt_p(p)
%FMT_P Format a p-value the way it is reported in text ("p = 0.031", "p < 0.001").
if isnan(p),        s = "p = n/a";
elseif p < 0.001,   s = "p < 0.001";
else,               s = sprintf("p = %.3f", p);
end
end
