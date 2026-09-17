"""Compute every number the subjective report quotes, once, into one dict."""
import numpy as np
from scipy import stats
import subjdata as sd
import stats_helpers as sh

C_IDX, J_IDX = [0, 1, 2], [3, 4, 5]
ASSIST_PAIRS = [(0, 1), (0, 2), (1, 2)]
ASSIST_NAME = ["F", "B", "FB"]


def binom_p(k, n):
    if hasattr(stats, 'binomtest'):
        return float(stats.binomtest(k, n, 0.5).pvalue)
    return float(stats.binom_test(k, n, 0.5))


def run():
    D = sd.load_subjective()
    pids, R = D['pids'], D['R']
    O = sd.load_objective(pids)
    Q = sd.subjective_index(R)
    n = len(pids)
    best, rank = D['best'], D['rank']
    P = np.repeat(np.arange(n), 6)

    out = dict(n=n, pids=pids, R=R, O=O, Q=Q, best=best, rank=rank,
               notes=D['notes'], final_comment=D['final_comment'],
               demo=D['demo'], gender=D['gender'], hand=D['hand'])

    keys = sd.ITEMS + ["index"]

    def mat(k):
        return Q if k == "index" else R[k]

    # ---- descriptives -----------------------------------------------------
    out['cell_mean'] = {k: np.nanmean(mat(k), 0) for k in keys}
    out['cell_sd'] = {k: np.nanstd(mat(k), 0, ddof=1) for k in keys}
    out['cell_med'] = {k: np.nanmedian(mat(k), 0) for k in keys}

    # ---- mode effect ------------------------------------------------------
    out['mode'] = {}
    for k in keys:
        X = mat(k)
        c, j = np.nanmean(X[:, C_IDX], 1), np.nanmean(X[:, J_IDX], 1)
        w = sh.wilcoxon_pair(j, c)
        W, ps = sh.shapiro_safe(j - c)
        out['mode'][k] = dict(C=float(c.mean()), J=float(j.mean()),
                              d=float(j.mean() - c.mean()), shapiro_W=W, shapiro_p=ps,
                              nfav=int((j > c).sum()), **w)

    # ---- assistance effect ------------------------------------------------
    out['assist'] = {}
    for k in keys:
        X = mat(k)
        A = np.column_stack([np.nanmean(X[:, [0, 3]], 1), np.nanmean(X[:, [1, 4]], 1),
                             np.nanmean(X[:, [2, 5]], 1)])
        f = sh.friedman_w(A)
        pw = [sh.wilcoxon_pair(A[:, a], A[:, b]) for a, b in ASSIST_PAIRS]
        adj = sh.holm([q['p'] for q in pw])
        out['assist'][k] = dict(means=[float(A[:, i].mean()) for i in range(3)],
                                fried=f, pairs=[dict(a=ASSIST_NAME[a], b=ASSIST_NAME[b],
                                                     p_holm=float(adj[i]), r=pw[i]['r'])
                                                for i, (a, b) in enumerate(ASSIST_PAIRS)],
                                A=A)

    # ---- six-cell omnibus + normality -------------------------------------
    out['omnibus'] = {k: sh.friedman_w(mat(k)) for k in keys}
    out['shapiro_cell'] = {k: [sh.shapiro_safe(mat(k)[:, j]) for j in range(6)] for k in keys}

    # ---- reliability ------------------------------------------------------
    stack = np.vstack([np.column_stack([(R[it][:, j] if sd.ITEM_DIR[it] > 0
                                         else 8 - R[it][:, j]) for it in sd.ITEMS])
                       for j in range(6)])
    out['alpha6'] = sh.cronbach_alpha(stack)
    pos = [it for it in sd.ITEMS if sd.ITEM_DIR[it] > 0]
    out['alpha4'] = sh.cronbach_alpha(
        np.vstack([np.column_stack([R[it][:, j] for it in pos]) for j in range(6)]))
    flat = {it: R[it].T.reshape(-1) for it in sd.ITEMS}
    out['interitem'] = np.array([[sh.spearman_safe(flat[a], flat[b])[0]
                                  for b in sd.ITEMS] for a in sd.ITEMS])

    # ---- preference -------------------------------------------------------
    nJ, nC = int((best == 'J').sum()), int((best == 'C').sum())
    out['pref'] = dict(nJ=nJ, nC=nC, p=binom_p(nJ, nJ + nC))
    out['rankstats'] = {}
    for name, idx in [("C", [0, 1, 2]), ("J", [3, 4, 5])]:
        Rk = rank[:, idx]
        f = sh.friedman_w(Rk)
        pw = [sh.wilcoxon_pair(Rk[:, a], Rk[:, b]) for a, b in ASSIST_PAIRS]
        adj = sh.holm([q['p'] for q in pw])
        out['rankstats'][name] = dict(
            mean=[float(Rk[:, t].mean()) for t in range(3)],
            first=[int((Rk[:, t] == 1).sum()) for t in range(3)],
            dist=[[int((Rk[:, t] == r).sum()) for r in (1, 2, 3)] for t in range(3)],
            fried=f, pairs=[dict(a=ASSIST_NAME[a], b=ASSIST_NAME[b], p_holm=float(adj[i]))
                            for i, (a, b) in enumerate(ASSIST_PAIRS)])

    # ---- objective mode effect (the claim the subjective data is tested against)
    out['obj_mode'] = {}
    for k in ["composite", "duration_s", "ee_path_len_m", "ee_sparc",
              "fam_time_effectiveness", "safety_min_dist_m", "cbf_active_s"]:
        X = O[k]
        c, j = np.nanmean(X[:, C_IDX], 1), np.nanmean(X[:, J_IDX], 1)
        w = sh.wilcoxon_pair(j, c)
        out['obj_mode'][k] = dict(C=float(c.mean()), J=float(j.mean()),
                                  d=float(j.mean() - c.mean()),
                                  nfav=int((j > c).sum()), **w)

    # ---- preference vs advantage -----------------------------------------
    dcomp = np.nanmean(O['composite'][:, J_IDX], 1) - np.nanmean(O['composite'][:, C_IDX], 1)
    dtime = np.nanmean(O['duration_s'][:, J_IDX], 1) - np.nanmean(O['duration_s'][:, C_IDX], 1)
    dsubj = np.nanmean(Q[:, J_IDX], 1) - np.nanmean(Q[:, C_IDX], 1)
    gJ, gC = best == 'J', best == 'C'
    out['delta'] = dict(comp=dcomp, time=dtime, subj=dsubj, gJ=gJ, gC=gC)
    out['choice'] = {}
    for nm, d in [("comp", dcomp), ("time", dtime), ("subj", dsubj)]:
        m = sh.mannwhitney(d[gJ], d[gC])
        out['choice'][nm] = dict(mJ=float(np.nanmean(d[gJ])), mC=float(np.nanmean(d[gC])), **m)
    out['agree_obj'] = int((((dcomp > 0) == gJ)).sum())
    out['agree_subj'] = int((((dsubj > 0) == gJ)).sum())
    out['agree_time'] = int((((dtime < 0) == gJ)).sum())

    # ---- self-insight -----------------------------------------------------
    suc = R['success']
    out['insight_within'] = {}
    for objname, sign in [("composite", +1), ("duration_s", -1), ("fam_time_effectiveness", +1)]:
        rs = np.array([sh.spearman_safe(suc[i, :], sign * O[objname][i, :])[0]
                       for i in range(n)], float)
        good = ~np.isnan(rs)
        z = sh.fisher_z(rs[good])
        _, pw = stats.wilcoxon(z)
        tt = stats.ttest_1samp(z, 0)
        out['insight_within'][objname] = dict(
            rho=rs, median=float(np.nanmedian(rs)), mean=float(np.nanmean(rs)),
            npos=int((rs[good] > 0).sum()), nval=int(good.sum()),
            p_wilcoxon=float(pw), p_t=float(tt.pvalue))
    out['insight_rm'] = {}
    for objname, sign, lab in [("composite", +1, "composite score"),
                               ("duration_s", -1, "shorter task time"),
                               ("ee_path_len_m", -1, "shorter hand path"),
                               ("fam_time_effectiveness", +1, "time and effectiveness")]:
        out['insight_rm'][lab] = sh.rm_corr(P, suc.reshape(-1), sign * O[objname].reshape(-1))
    out['insight_between'] = {}
    for objname, sign, lab in [("composite", +1, "composite score"),
                               ("duration_s", -1, "shorter task time"),
                               ("ee_path_len_m", -1, "shorter hand path")]:
        rho, p, nn = sh.spearman_safe(np.nanmean(suc, 1), sign * np.nanmean(O[objname], 1))
        out['insight_between'][lab] = dict(rho=rho, p=p, n=nn)

    # ---- other subjective-objective links ---------------------------------
    pairs = [("mental", "duration_s", +1, "task time"),
             ("mental", "ee_path_len_m", +1, "hand path length"),
             ("physical", "force_mean_N", +1, "rendered force"),
             ("physical", "qdot_cmd_rms", +1, "commanded joint rate"),
             ("physical", "clutch_presses", +1, "re-indexing operations"),
             ("control", "alpha_autonomy_frac", -1, "less autonomy-led time"),
             ("control", "agreement_mean_cos", +1, "operator-policy agreement"),
             ("trust", "cbf_active_s", -1, "less filter activity"),
             ("trust", "safety_min_dist_m", +1, "minimum clearance"),
             ("trust", "belief_mean_prob", +1, "intent confidence"),
             ("smooth", "ee_sparc", +1, "movement smoothness"),
             ("smooth", "slack_mean", -1, "less tracking slack")]
    out['links'] = []
    for it, ob, sign, lab in pairs:
        rc = sh.rm_corr(P, R[it].reshape(-1), sign * O[ob].reshape(-1))
        out['links'].append(dict(item=it, obj=ob, lab=lab, sign=sign, **rc))
    padj = sh.holm([L['p'] for L in out['links']])
    for L, a in zip(out['links'], padj):
        L['p_holm'] = float(a)

    # ---- prior experience -------------------------------------------------
    ex = D['demo']['exper']
    out['exper'] = {}
    for lab, y in [("composite score", np.nanmean(O['composite'], 1)),
                   ("task time", np.nanmean(O['duration_s'], 1)),
                   ("perceived success", np.nanmean(suc, 1)),
                   ("subjective index", np.nanmean(Q, 1)),
                   ("mental demand", np.nanmean(R['mental'], 1))]:
        rho, p, nn = sh.spearman_safe(ex, y)
        out['exper'][lab] = dict(rho=rho, p=p, n=nn)
    out['exper_choice'] = sh.mannwhitney(ex[gJ], ex[gC])
    return out


if __name__ == "__main__":
    r = run()
    print("computed; n =", r['n'], "alpha =", round(r['alpha6'], 3))
