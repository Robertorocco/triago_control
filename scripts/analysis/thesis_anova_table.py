#!/usr/bin/env python3
"""Reproduce thesis Table 4.2 (assumption checks + RM-ANOVA) from trial_table.csv.

Recomputes, by hand with numpy/scipy, the two-way and one-way repeated-measures
ANOVAs that MATLAB's study pipeline reports in results_all.csv, plus
Greenhouse-Geisser epsilon, Mauchly's test, Shapiro-Wilk on model residuals,
and an aligned-rank-transform (ART) nonparametric check. Validates against
results_all.csv and prints PASS/FAIL before writing the thesis table.
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

MODES = ["C", "J"]
ASSIST3 = ["F", "B", "FB"]
ASSIST2 = ["B", "FB"]

# (metric column, thesis symbol) for the 6-cell (Mode x 3-level Assist) design.
TWO_WAY_6 = [
    ("duration_s", "T"),
    ("ee_path_len_m", "L"),
    ("ee_sparc", "\\bar{\\Phi}"),
    ("ee_speed_mean_mps", "\\bar{v}"),
    ("slack_mean", "\\bar{\\delta}"),
    ("safety_min_dist_m", "h_{\\min}"),
    ("cbf_active_s", "T_{\\lambda}"),
    ("belief_mean_prob", "\\bar{b}"),
]
# (metric column, thesis symbol) for the 4-cell (Mode x 2-level Assist=B/FB) design.
TWO_WAY_4 = [
    ("agreement_mean_cos", "\\rho"),
    ("alpha_autonomy_frac", "\\phi_\\alpha"),
    ("intervention_mean_mps", "\\bar{\\Delta}"),
    ("intervention_peak_mps", "\\Delta_{\\max}"),
]
# (metric column, fixed mode, thesis symbol) for the one-way (3-level Assist) design.
ONE_WAY = [
    ("force_mean_N", "C", "\\bar{f}"),
    ("force_mean_N", "J", "\\bar{f}"),
    ("clutch_presses", "C", "N_c"),
]

# Hand-typed reference values from the current thesis table (assist-factor only).
# Only valid for metrics whose definition has not changed since that table was written.
REF_MAUCHLY = {
    "duration_s": (0.180, 0.87),
    "ee_path_len_m": (0.0, 0.67),          # "<0.001"
    "ee_speed_mean_mps": (0.389, 0.92),
    "slack_mean": (0.006, 0.73),
    "safety_min_dist_m": (0.0, 0.68),      # "<0.001"
    "belief_mean_prob": (0.152, 0.86),
    "force_mean_N|C": (0.0, 0.66),         # "<0.001"
    "force_mean_N|J": (0.16, 0.87),
    "clutch_presses|C": (0.592, 0.96),
}
REF_SHAPIRO = {
    "duration_s": 0.42,
    "ee_path_len_m": 0.0,   # "<0.001"
    "ee_speed_mean_mps": 0.22,
    "slack_mean": 0.0,      # "<0.001"
    "safety_min_dist_m": 0.0,  # "<0.001"
    "belief_mean_prob": 0.13,
    "force_mean_N|C": 0.44,
    "force_mean_N|J": 0.44,
    "clutch_presses|C": 0.92,
}
REF_LT001 = {"ee_path_len_m", "safety_min_dist_m", "force_mean_N|C"}  # entries hand-typed as "<0.001"


def default_results_dir():
    latest = Path("/home/roberto/exchange/matlab_export/analysis_results/latest.txt")
    if latest.exists():
        return Path(latest.read_text().strip())
    return None


def read_trial_table(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames


def to_float(s):
    if s is None or s == "" or s.strip().lower() == "nan":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def participant_cell_means(rows, metric, cells, header):
    """Average the two worlds per (participant, cell) with nanmean; return {participant: {cell: value}}."""
    if metric not in header:
        return None
    acc = {}
    for row in rows:
        p = row["participant"]
        c = row["cell"]
        if c not in cells:
            continue
        v = to_float(row.get(metric))
        acc.setdefault(p, {}).setdefault(c, []).append(v)
    out = {}
    for p, cell_map in acc.items():
        out[p] = {}
        for c in cells:
            vals = cell_map.get(c, [])
            if not vals:
                out[p][c] = float("nan")
                continue
            arr = np.array(vals, dtype=float)
            if np.all(np.isnan(arr)):
                out[p][c] = float("nan")
            else:
                out[p][c] = float(np.nanmean(arr))
    return out


def build_matrix(pc_means, cells_layout):
    """cells_layout: list of lists of cell names, shape rows x cols (e.g. MODES x ASSIST).
    Drops participants with any NaN across the flattened cells. Returns (Y, participants)."""
    participants = sorted(pc_means.keys())
    flat_cells = [c for row in cells_layout for c in row]
    kept = []
    for p in participants:
        vals = [pc_means[p].get(c, float("nan")) for c in flat_cells]
        if any(math.isnan(v) for v in vals):
            continue
        kept.append(p)
    n_dropped = len(participants) - len(kept)
    a = len(cells_layout)
    b = len(cells_layout[0])
    Y = np.zeros((len(kept), a, b))
    for pi, p in enumerate(kept):
        for i, row in enumerate(cells_layout):
            for j, c in enumerate(row):
                Y[pi, i, j] = pc_means[p][c]
    return Y, kept, n_dropped


def build_vector_matrix(pc_means, cells):
    """cells: list of k cell names. Drops participants with any NaN. Returns (Y[n,k], participants, n_dropped)."""
    participants = sorted(pc_means.keys())
    kept = []
    for p in participants:
        vals = [pc_means[p].get(c, float("nan")) for c in cells]
        if any(math.isnan(v) for v in vals):
            continue
        kept.append(p)
    n_dropped = len(participants) - len(kept)
    Y = np.zeros((len(kept), len(cells)))
    for pi, p in enumerate(kept):
        for j, c in enumerate(cells):
            Y[pi, j] = pc_means[p][c]
    return Y, kept, n_dropped


def helmert_contrasts(k):
    """k x (k-1) orthonormal matrix whose columns are orthogonal to the ones vector."""
    C = np.zeros((k, k - 1))
    for i in range(1, k):
        C[:i, i - 1] = -1.0 / math.sqrt(i * (i + 1))
        C[i, i - 1] = i / math.sqrt(i * (i + 1))
    return C


def gg_epsilon(scores):
    """scores: n x k matrix of per-participant scores for one effect. Returns (eps, S) or (1.0, None) if k<=2."""
    n, k = scores.shape
    if k <= 2:
        return 1.0, None
    Sigma = np.cov(scores, rowvar=False, ddof=1)
    C = helmert_contrasts(k)
    S = C.T @ Sigma @ C
    eps = float(np.trace(S) ** 2 / ((k - 1) * np.trace(S @ S)))
    return eps, S


def mauchly_test(S, n, k):
    """S: (k-1)x(k-1) transformed covariance. Returns (W, chi2, df, p) or (nan,nan,nan,nan) if k<=2."""
    if S is None or k <= 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    m = k - 1
    trS = np.trace(S)
    detS = np.linalg.det(S)
    if detS <= 0 or trS <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    W = float(detS / (trS / m) ** m)
    lnW = math.log(W)
    corr = 1 - (2 * m ** 2 + m + 2) / (6 * m * (n - 1))
    chi2 = -(n - 1) * corr * lnW
    df = k * (k - 1) / 2 - 1
    p = float(stats.chi2.sf(chi2, df))
    return W, chi2, df, p


def rm_anova_2way(Y):
    """Y: n x a x b. Returns dict with SS/MS/F/df/eta2 for A (rows), B (cols), AB."""
    n, a, b = Y.shape
    grand = Y.mean()
    p_mean = Y.mean(axis=(1, 2))
    a_mean = Y.mean(axis=(0, 2))
    b_mean = Y.mean(axis=(0, 1))
    ab_mean = Y.mean(axis=0)

    SS_A = n * b * np.sum((a_mean - grand) ** 2)
    SS_B = n * a * np.sum((b_mean - grand) ** 2)
    SS_AB = n * np.sum((ab_mean - a_mean[:, None] - b_mean[None, :] + grand) ** 2)
    SS_S = a * b * np.sum((p_mean - grand) ** 2)
    SS_total = np.sum((Y - grand) ** 2)

    y_pi = Y.mean(axis=2)  # n x a
    SS_PA = b * np.sum((y_pi - p_mean[:, None] - a_mean[None, :] + grand) ** 2)
    y_pj = Y.mean(axis=1)  # n x b
    SS_PB = a * np.sum((y_pj - p_mean[:, None] - b_mean[None, :] + grand) ** 2)
    SS_PAB = SS_total - SS_A - SS_B - SS_AB - SS_S - SS_PA - SS_PB

    df_A, df_PA = a - 1, (a - 1) * (n - 1)
    df_B, df_PB = b - 1, (b - 1) * (n - 1)
    df_AB, df_PAB = (a - 1) * (b - 1), (a - 1) * (b - 1) * (n - 1)

    MS_A, MS_PA = SS_A / df_A, SS_PA / df_PA
    MS_B, MS_PB = SS_B / df_B, SS_PB / df_PB
    MS_AB, MS_PAB = SS_AB / df_AB, SS_PAB / df_PAB

    F_A = MS_A / MS_PA
    F_B = MS_B / MS_PB
    F_AB = MS_AB / MS_PAB

    eta2_A = SS_A / (SS_A + SS_PA)
    eta2_B = SS_B / (SS_B + SS_PB)
    eta2_AB = SS_AB / (SS_AB + SS_PAB)

    return {
        "grand": grand, "p_mean": p_mean, "a_mean": a_mean, "b_mean": b_mean, "ab_mean": ab_mean,
        "A": dict(F=F_A, df1=df_A, df2=df_PA, eta2=eta2_A),
        "B": dict(F=F_B, df1=df_B, df2=df_PB, eta2=eta2_B),
        "AB": dict(F=F_AB, df1=df_AB, df2=df_PAB, eta2=eta2_AB),
    }


def rm_anova_1way(Y):
    """Y: n x k. Returns dict with F/df/eta2 for the single within factor."""
    n, k = Y.shape
    grand = Y.mean()
    p_mean = Y.mean(axis=1)
    k_mean = Y.mean(axis=0)
    SS_between = n * np.sum((k_mean - grand) ** 2)
    SS_subj = k * np.sum((p_mean - grand) ** 2)
    SS_total = np.sum((Y - grand) ** 2)
    SS_error = SS_total - SS_between - SS_subj
    df1, df2 = k - 1, (k - 1) * (n - 1)
    F = (SS_between / df1) / (SS_error / df2)
    eta2 = SS_between / (SS_between + SS_error)
    return dict(F=F, df1=df1, df2=df2, eta2=eta2, grand=grand, p_mean=p_mean, k_mean=k_mean)


def p_from_F(F, df1, df2):
    return float(stats.f.sf(F, df1, df2))


def art_effect_p(residual_full, effect_estimate, cells_layout_shape):
    """Aligned rank transform: rank(residual + effect estimate) then re-run the 2-way ANOVA,
    reporting the uncorrected p for that same effect (ARTool convention)."""
    aligned = residual_full + effect_estimate
    n, a, b = aligned.shape
    ranks = stats.rankdata(aligned.reshape(-1), method="average").reshape(n, a, b)
    return rm_anova_2way(ranks)


def fmt_p(p, small_is_lt=True):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "---"
    if small_is_lt and p < 0.001:
        return "$p<0.001$"
    return f"$p={p:.3f}$"


def fmt_F(F):
    if F is None or (isinstance(F, float) and math.isnan(F)):
        return "---"
    return f"{F:.1f}" if F >= 100 else f"{F:.2f}"


def fmt_effect_cell(F, df1, df2, eps, mauchly_p, k):
    """$F(d1,d2)=v$, $p=q$ with integer df unless Mauchly p<0.05 and k>2, then eps-adjusted df (2dp)."""
    use_gg = (k is not None and k > 2) and (mauchly_p is not None and not math.isnan(mauchly_p) and mauchly_p < 0.05)
    if use_gg:
        d1, d2 = df1 * eps, df2 * eps
        d1s, d2s = f"{d1:.2f}", f"{d2:.2f}"
        p = p_from_F(F, d1, d2)
    else:
        d1s, d2s = f"{int(round(df1))}", f"{int(round(df2))}"
        p = p_from_F(F, df1, df2)
    return f"${{F}}({d1s},{d2s})={fmt_F(F)}$, {fmt_p(p)}"


# ---------------------------------------------------------------------------
# Per-metric analysis
# ---------------------------------------------------------------------------

def analyze_two_way(rows, header, metric, symbol, assist_levels, out_rows, cache):
    cells_layout = [[m + a for a in assist_levels] for m in MODES]
    pc_means = participant_cell_means(rows, metric, [c for row in cells_layout for c in row], header)
    if pc_means is None:
        print(f"[skip] column '{metric}' not found in trial_table.csv")
        return None
    Y, participants, n_dropped = build_matrix(pc_means, cells_layout)
    n, a, b = Y.shape
    if n_dropped:
        print(f"[{metric}] dropped {n_dropped} participant(s) with NaN in required cells (n={n})")

    res = rm_anova_2way(Y)
    grand, p_mean, a_mean, b_mean, ab_mean = res["grand"], res["p_mean"], res["a_mean"], res["b_mean"], res["ab_mean"]

    # GG epsilon + Mauchly for Assist (avg over Mode) and Interaction (Mode-diff across Assist)
    scores_assist = Y.mean(axis=1)  # n x b
    eps_assist, S_assist = gg_epsilon(scores_assist)
    W_a, chi2_a, dfm_a, p_a = mauchly_test(S_assist, n, b)

    diff_int = Y[:, 1, :] - Y[:, 0, :]  # n x b, mode J minus mode C per assist level
    eps_int, S_int = gg_epsilon(diff_int)
    W_i, chi2_i, dfm_i, p_i = mauchly_test(S_int, n, b)

    eps_mode, S_mode = 1.0, None  # a == 2 always here
    W_m, chi2_m, dfm_m, p_m = float("nan"), float("nan"), float("nan"), float("nan")

    # Shapiro on residuals of the additive/full two-way model
    residual_full = Y - p_mean[:, None, None] - ab_mean[None, :, :] + grand
    sw_stat, sw_p = stats.shapiro(residual_full.reshape(-1))

    # ART per effect
    eff_mode = (a_mean - grand)[:, None] * np.ones((1, b))
    eff_assist = (b_mean - grand)[None, :] * np.ones((a, 1))
    eff_int = ab_mean - a_mean[:, None] - b_mean[None, :] + grand
    art_mode = art_effect_p(residual_full, eff_mode, Y.shape)
    art_assist = art_effect_p(residual_full, eff_assist, Y.shape)
    art_int = art_effect_p(residual_full, eff_int, Y.shape)
    art_p_mode = p_from_F(art_mode["A"]["F"], art_mode["A"]["df1"], art_mode["A"]["df2"])
    art_p_assist = p_from_F(art_assist["B"]["F"], art_assist["B"]["df1"], art_assist["B"]["df2"])
    art_p_int = p_from_F(art_int["AB"]["F"], art_int["AB"]["df1"], art_int["AB"]["df2"])

    def make_row(effect_key, effname, eff, eps, W, chi2, dfm, pmau, art_p, k):
        F, df1, df2, eta2 = eff["F"], eff["df1"], eff["df2"], eff["eta2"]
        p_raw = p_from_F(F, df1, df2)
        p_gg = p_from_F(F, df1 * eps, df2 * eps)
        return dict(metric=metric, symbol=symbol, effect=effname, F=F, df1=df1, df2=df2, eps=eps,
                    df1_gg=df1 * eps, df2_gg=df2 * eps, p=p_raw, p_gg=p_gg, eta2p=eta2,
                    mauchly_W=W, mauchly_chi2=chi2, mauchly_p=pmau, shapiro_p=sw_p, art_p=art_p, n=n)

    row_mode = make_row("A", "Mode", res["A"], eps_mode, W_m, chi2_m, dfm_m, p_m, art_p_mode, a)
    row_assist = make_row("B", "Assist", res["B"], eps_assist, W_a, chi2_a, dfm_a, p_a, art_p_assist, b)
    row_int = make_row("AB", "Mode:Assist", res["AB"], eps_int, W_i, chi2_i, dfm_i, p_i, art_p_int, b)
    out_rows += [row_mode, row_assist, row_int]

    cache[metric] = dict(n=n, k=b, b=b,
                          mode=row_mode, assist=row_assist, interaction=row_int,
                          shapiro_p=sw_p, eps_assist=eps_assist, mauchly_p_assist=p_a)
    return cache[metric]


def analyze_one_way(rows, header, metric, mode, symbol, assist_levels, out_rows, cache, key):
    cells = [mode + a for a in assist_levels]
    pc_means = participant_cell_means(rows, metric, cells, header)
    if pc_means is None:
        print(f"[skip] column '{metric}' not found in trial_table.csv")
        return None
    Y, participants, n_dropped = build_vector_matrix(pc_means, cells)
    n, k = Y.shape
    if n_dropped:
        print(f"[{metric}|{mode}] dropped {n_dropped} participant(s) with NaN in required cells (n={n})")

    res = rm_anova_1way(Y)
    grand, p_mean, k_mean = res["grand"], res["p_mean"], res["k_mean"]

    eps, S = gg_epsilon(Y)
    W, chi2, dfm, pmau = mauchly_test(S, n, k)

    residual = Y - p_mean[:, None] - k_mean[None, :] + grand
    sw_stat, sw_p = stats.shapiro(residual.reshape(-1))

    fr_stat, fr_p = stats.friedmanchisquare(*[Y[:, j] for j in range(k)])

    F, df1, df2, eta2 = res["F"], res["df1"], res["df2"], res["eta2"]
    p_raw = p_from_F(F, df1, df2)
    p_gg = p_from_F(F, df1 * eps, df2 * eps)
    row = dict(metric=f"{metric}|{mode}", symbol=symbol, effect="Assist", F=F, df1=df1, df2=df2, eps=eps,
               df1_gg=df1 * eps, df2_gg=df2 * eps, p=p_raw, p_gg=p_gg, eta2p=eta2,
               mauchly_W=W, mauchly_chi2=chi2, mauchly_p=pmau, shapiro_p=sw_p, art_p=fr_p, n=n)
    out_rows.append(row)
    cache[key] = dict(n=n, k=k, row=row, shapiro_p=sw_p, eps_assist=eps, mauchly_p_assist=pmau,
                       friedman_stat=fr_stat, friedman_p=fr_p)
    return cache[key]


# ---------------------------------------------------------------------------
# results_all.csv parsing / validation
# ---------------------------------------------------------------------------

EFFECT_RE = {
    "Mode": re.compile(r"Mode (?:n\.s\.|significant) \(F\(([\d.]+),\s*([\d.]+)\) = ([\d.\-]+), partial eta2 = ([\d.\-]+), p ([<=]) ([\d.]+)\)"),
    "Assist": re.compile(r"(?<!:)Assist (?:n\.s\.|significant) \(F\(([\d.]+),\s*([\d.]+)\) = ([\d.\-]+), partial eta2 = ([\d.\-]+), p ([<=]) ([\d.]+)\)"),
    "Mode:Assist": re.compile(r"Mode:Assist (?:n\.s\.|significant) \(F\(([\d.]+),\s*([\d.]+)\) = ([\d.\-]+), partial eta2 = ([\d.\-]+), p ([<=]) ([\d.]+)\)"),
}


def parse_results_all(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    q3 = {}
    q2_rm = {}
    q2_friedman = {}
    for row in rows:
        if row["question"] == "Q3_cell":
            q3[row["metric"]] = row["interpretation"]
        if row["question"] == "Q2_assist":
            m = row["metric"]
            levels = row["levels"]
            if row["test"] == "rm_anova":
                q2_rm[(m, levels)] = (to_float(row["statistic"]), to_float(row["p_raw"]))
            elif row["test"] == "friedman":
                q2_friedman[(m, levels)] = (to_float(row["statistic"]), to_float(row["p_raw"]))
    return q3, q2_rm, q2_friedman


def parse_effect(interp, effect):
    mobj = EFFECT_RE[effect].search(interp)
    if not mobj:
        return None
    df1, df2, F, eta2, op, pval = mobj.groups()
    p = 0.0 if op == "<" else float(pval)
    return dict(df1=float(df1), df2=float(df2), F=float(F), eta2=float(eta2), p=float(pval), p_lt=(op == "<"))


def check(label, ok, detail):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {label}: {detail}")
    return ok


def validate(cache_two, cache_one, q3, q2_rm, q2_friedman):
    print("\n=== Validation against results_all.csv (F within 0.02, p_GG within 0.002) ===")
    all_ok = True
    for metric, c in cache_two.items():
        interp = q3.get(metric)
        if interp is None:
            print(f"[N/A]  {metric}: no Q3_cell row in results_all.csv (metric not present there) -- computed anyway")
            continue
        for effname, key in (("Mode", "mode"), ("Assist", "assist"), ("Mode:Assist", "interaction")):
            ref = parse_effect(interp, effname)
            if ref is None:
                print(f"[N/A]  {metric} {effname}: could not parse reference from interpretation string")
                continue
            row = c[key]
            f_ok = abs(row["F"] - ref["F"]) <= 0.02
            if ref["p_lt"]:
                p_ok = row["p_gg"] < 0.001
                p_detail = f"mine p_gg={row['p_gg']:.4g} vs ref p<0.001"
            else:
                p_ok = abs(row["p_gg"] - ref["p"]) <= 0.002
                p_detail = f"mine p_gg={row['p_gg']:.4f} vs ref p={ref['p']:.3f}"
            ok = f_ok and p_ok
            all_ok &= ok
            check(f"{metric} {effname}", ok,
                  f"F mine={row['F']:.3f} ref={ref['F']:.2f} (df {row['df1']:.0f},{row['df2']:.0f} vs ref {ref['df1']:.0f},{ref['df2']:.0f}); {p_detail}")

    print("\n=== Validation of one-way metrics (F, p_raw) ===")
    for key, c in cache_one.items():
        metric, mode = key.split("|")
        levels = f"F/B/FB within {mode}"
        row = c["row"]
        ref = q2_rm.get((metric, levels))
        if ref is None:
            fref = q2_friedman.get((metric, levels))
            if fref is not None:
                print(f"[N/A]  {metric} within {mode}: MATLAB used Friedman instead of rm_anova "
                      f"(stat={fref[0]:.3f}, p={fref[1]:.3g}); mine F={row['F']:.3f}, p_raw={row['p']:.3g}, "
                      f"friedman here p={c['friedman_p']:.3g}")
            else:
                print(f"[N/A]  {metric} within {mode}: no reference row found in results_all.csv")
            continue
        refF, refp = ref
        # MATLAB's "p_raw" column for the rm_anova test is itself GG-corrected (matches our p_gg exactly).
        f_ok = abs(row["F"] - refF) <= 0.02
        p_ok = (row["p_gg"] < 0.001 and refp < 0.001) or abs(row["p_gg"] - refp) <= 0.002
        ok = f_ok and p_ok
        all_ok &= ok
        check(f"{metric} within {mode}", ok, f"F mine={row['F']:.3f} ref={refF:.3f}; p_gg mine={row['p_gg']:.4g} ref={refp:.4g}")

    print("\n=== Cross-check vs hand-typed thesis table (assist-factor Mauchly/eps, Shapiro) ===")
    def get_ref_key(metric_key):
        return metric_key
    all_cache = dict(cache_two)
    for key, c in cache_one.items():
        all_cache[key.replace("|", "|")] = c
    for refkey in list(REF_MAUCHLY.keys()):
        c = cache_two.get(refkey) or cache_one.get(refkey)
        if c is None:
            continue
        ref_mau_p, ref_eps = REF_MAUCHLY[refkey]
        mine_mau_p = c["mauchly_p_assist"]
        mine_eps = c["eps_assist"]
        lt = refkey in REF_LT001
        mau_str = "<0.001" if lt else f"{ref_mau_p:.3f}"
        print(f"  {refkey}: Mauchly p mine={mine_mau_p:.3f} ref={mau_str} | eps mine={mine_eps:.2f} ref={ref_eps:.2f}")
    for refkey, ref_sw in REF_SHAPIRO.items():
        c = cache_two.get(refkey) or cache_one.get(refkey)
        if c is None:
            continue
        lt = refkey == "ee_path_len_m" or refkey == "safety_min_dist_m" or refkey == "slack_mean"
        sw_str = "<0.001" if lt else f"{ref_sw:.2f}"
        print(f"  {refkey}: Shapiro p mine={c['shapiro_p']:.3f} ref={sw_str}")

    print(f"\n=== Overall validation: {'PASS' if all_ok else 'FAIL'} ===")
    return all_ok


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

CSV_FIELDS = ["metric", "symbol", "effect", "F", "df1", "df2", "eps", "df1_gg", "df2_gg",
              "p", "p_gg", "eta2p", "mauchly_W", "mauchly_chi2", "mauchly_p", "shapiro_p", "art_p", "n"]


def write_csv(out_dir, out_rows):
    path = out_dir / "thesis_anova.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"Wrote {path}")


def _num(p, lt=True):
    """A bare p value for a numeric column: '<0.001' or three decimals."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "---"
    return "$<0.001$" if (lt and p < 0.001) else f"${p:.3f}$"


def _effect_line(label, row, k, last_p, first_cell=""):
    """One table line: measure | effect | SW | Mauchly | eps | F(d1,d2) | p | eta2p | last."""
    use_gg = k > 2 and not math.isnan(row["mauchly_p"]) and row["mauchly_p"] < 0.05
    if use_gg:
        d1, d2 = row["df1"] * row["eps"], row["df2"] * row["eps"]
        d1s, d2s, p = f"{d1:.2f}", f"{d2:.2f}", row["p_gg"]
    else:
        d1s, d2s, p = f"{int(round(row['df1']))}", f"{int(round(row['df2']))}", row["p"]
    mau = "---" if k <= 2 else _num(row["mauchly_p"])
    eps = "---" if k <= 2 else f"${row['eps']:.2f}$"
    return (f"{first_cell} & {label} & {mau} & {eps} & $F({d1s},{d2s})={fmt_F(row['F'])}$ & "
            f"{_num(p)} & ${row['eta2p']:.2f}$ & {_num(last_p)} \\\\")


def write_tex(out_dir, order, cache_two, cache_one):
    """Body rows for the thesis table: one line per tested effect, grouped per measure.

    The Shapiro-Wilk p sits on the measure's first line; Mauchly's p and the
    epsilon are those of the effect on that line, since the interaction has its
    own; the degrees of freedom are epsilon-adjusted only where Mauchly rejects.
    """
    lines = []
    for entry in order:
        symbol = entry["symbol"]
        if entry["kind"] == "two":
            c = cache_two.get(entry["metric"])
            if c is None:
                continue
            head = f"${symbol}$ ({_num(c['shapiro_p'])})"
            lines.append(_effect_line("Mode", c["mode"], 2, c["mode"]["art_p"], head))
            lines.append(_effect_line("Assist.", c["assist"], c["b"], c["assist"]["art_p"]))
            lines.append(_effect_line("Mode$\\times$Assist.", c["interaction"], c["b"], c["interaction"]["art_p"]))
        else:
            c = cache_one.get(f"{entry['metric']}|{entry['mode']}")
            if c is None:
                continue
            tag = {"C": " (clutch)", "J": " (joystick)"}[entry["mode"]] if entry["metric"] == "force_mean_N" else ""
            head = f"${symbol}${tag} ({_num(c['shapiro_p'])})"
            lines.append(_effect_line("Assist.", c["row"], c["k"], c["friedman_p"], head))
        lines.append("\\addlinespace[2pt]")
    path = out_dir / "thesis_anova_rows.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def print_summary(order, cache_two, cache_one):
    print("\n=== Thesis Table 4.2 summary ===")
    header = f"{'symbol':>18} | {'Shapiro p':>9} | {'Mauchly p':>9} | {'eps':>5} | Mode | Assist | Interaction"
    print(header)
    for entry in order:
        kind = entry["kind"]
        symbol = entry["symbol"]
        if kind == "two":
            c = cache_two.get(entry["metric"])
            if c is None:
                continue
            mode_row, assist_row, int_row = c["mode"], c["assist"], c["interaction"]
            print(f"{symbol:>18} | {c['shapiro_p']:.3f}     | "
                  f"{c['mauchly_p_assist']:.3f}     | {c['eps_assist']:.2f}  | "
                  f"F={mode_row['F']:.2f},p={mode_row['p_gg']:.3g} | "
                  f"F={assist_row['F']:.2f},p={assist_row['p_gg']:.3g} | "
                  f"F={int_row['F']:.2f},p={int_row['p_gg']:.3g}")
        else:
            key = f"{entry['metric']}|{entry['mode']}"
            c = cache_one.get(key)
            if c is None:
                continue
            row = c["row"]
            print(f"{symbol:>18} | {c['shapiro_p']:.3f}     | "
                  f"{c['mauchly_p_assist']:.3f}     | {c['eps_assist']:.2f}  | "
                  f"---  | F={row['F']:.2f},p={row['p_gg']:.3g} | ---")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=None, help="results folder (default: latest.txt)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <results>/thesis_table/)")
    args = ap.parse_args()

    results_dir = args.results or default_results_dir()
    if results_dir is None:
        print("No --results given and latest.txt not found.", file=sys.stderr)
        sys.exit(1)
    results_dir = Path(results_dir)
    out_dir = args.out or (results_dir / "thesis_table")
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_path = results_dir / "trial_table.csv"
    results_all_path = results_dir / "results_all.csv"
    rows, header = read_trial_table(trial_path)
    print(f"Loaded {len(rows)} trial rows from {trial_path}")

    out_rows = []
    cache_two = {}
    cache_one = {}
    order = []

    for metric, symbol in TWO_WAY_6:
        r = analyze_two_way(rows, header, metric, symbol, ASSIST3, out_rows, cache_two)
        order.append(dict(kind="two", metric=metric, symbol=symbol))
        if r is None:
            order.pop()
    for metric, symbol in TWO_WAY_4:
        r = analyze_two_way(rows, header, metric, symbol, ASSIST2, out_rows, cache_two)
        order.append(dict(kind="two", metric=metric, symbol=symbol))
        if r is None:
            order.pop()
    for metric, mode, symbol in ONE_WAY:
        key = f"{metric}|{mode}"
        r = analyze_one_way(rows, header, metric, mode, symbol, ASSIST3, out_rows, cache_one, key)
        order.append(dict(kind="one", metric=metric, mode=mode, symbol=symbol))
        if r is None:
            order.pop()

    if results_all_path.exists():
        q3, q2_rm, q2_friedman = parse_results_all(results_all_path)
        validate(cache_two, cache_one, q3, q2_rm, q2_friedman)
    else:
        print(f"[warn] {results_all_path} not found; skipping validation")

    write_csv(out_dir, out_rows)
    write_tex(out_dir, order, cache_two, cache_one)
    print_summary(order, cache_two, cache_one)


if __name__ == "__main__":
    main()
