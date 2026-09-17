"""Non-parametric tests, effect sizes and the repeated-measures correlation."""
import numpy as np
from scipy import stats


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def wilcoxon_pair(x, y):
    """Signed-rank test on paired vectors with the matched-pairs rank-biserial size."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    d = x - y
    nz = d[d != 0]
    if nz.size == 0:
        return dict(n=int(ok.sum()), W=np.nan, p=1.0, r=0.0, med=0.0)
    try:
        W, p = stats.wilcoxon(x, y, zero_method='wilcox')
    except ValueError:
        return dict(n=int(ok.sum()), W=np.nan, p=1.0, r=0.0, med=float(np.median(d)))
    r = np.abs(stats.rankdata(np.abs(nz)))
    Rp = r[nz > 0].sum()
    Rm = r[nz < 0].sum()
    rb = (Rp - Rm) / (Rp + Rm)
    return dict(n=int(ok.sum()), W=float(W), p=float(p), r=float(rb),
                med=float(np.median(d)))


def friedman_w(M):
    """Friedman test over the columns of M (rows = participants) plus Kendall's W."""
    M = np.asarray(M, float)
    M = M[~np.isnan(M).any(axis=1)]
    n, k = M.shape
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    W = chi2 / (n * (k - 1))
    return dict(n=n, k=k, chi2=float(chi2), df=k - 1, p=float(p), W=float(W))


def shapiro_safe(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size < 3 or np.allclose(x, x[0]):
        return np.nan, np.nan
    W, p = stats.shapiro(x)
    return float(W), float(p)


def spearman_safe(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum())
    xs, ys = x[ok], y[ok]
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return np.nan, np.nan, int(ok.sum())
    rho, p = stats.spearmanr(xs, ys)
    return float(rho), float(p), int(ok.sum())


def mannwhitney(a, b):
    """Rank-sum test with the common-language rank-biserial effect size."""
    a = np.asarray(a, float)[~np.isnan(a)]
    b = np.asarray(b, float)[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return dict(n1=a.size, n2=b.size, U=np.nan, p=np.nan, r=np.nan)
    U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    r = 2 * U / (a.size * b.size) - 1
    return dict(n1=int(a.size), n2=int(b.size), U=float(U), p=float(p), r=float(r))


def cronbach_alpha(X):
    """Alpha over the columns of X (rows = observations), listwise."""
    X = np.asarray(X, float)
    X = X[~np.isnan(X).any(axis=1)]
    n, k = X.shape
    vi = X.var(axis=0, ddof=1).sum()
    vt = X.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - vi / vt))


def rm_corr(P, X, Y):
    """Repeated-measures correlation: pooled within-participant association.

    Each participant contributes their own deviations from their own mean, so a
    common slope is estimated free of between-participant differences.
    """
    P, X, Y = np.asarray(P), np.asarray(X, float), np.asarray(Y, float)
    ok = ~(np.isnan(X) | np.isnan(Y))
    P, X, Y = P[ok], X[ok], Y[ok]
    xs, ys = np.empty_like(X), np.empty_like(Y)
    subs = np.unique(P)
    for s in subs:
        m = P == s
        xs[m] = X[m] - X[m].mean()
        ys[m] = Y[m] - Y[m].mean()
    df = X.size - len(subs) - 1
    if df < 1:
        return dict(r=np.nan, p=np.nan, df=df, n=X.size, k=len(subs))
    sx, sy = np.sqrt((xs ** 2).sum()), np.sqrt((ys ** 2).sum())
    if sx == 0 or sy == 0:
        return dict(r=np.nan, p=np.nan, df=df, n=X.size, k=len(subs))
    r = float((xs * ys).sum() / (sx * sy))
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt(df / (1 - r ** 2))
    p = float(2 * stats.t.sf(abs(t), df))
    return dict(r=r, p=p, df=int(df), n=int(X.size), k=int(len(subs)))


def fisher_z(r):
    r = np.clip(np.asarray(r, float), -0.999999, 0.999999)
    return np.arctanh(r)


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 0.001:
        return "$p < 0.001$"
    return "$p = %.3f$" % p


def fmt_p_bare(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 0.001:
        return "< 0.001"
    return "%.3f" % p
