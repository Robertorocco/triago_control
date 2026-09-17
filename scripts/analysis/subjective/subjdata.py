"""Load the questionnaire export and the objective trial table into aligned arrays."""
import csv
import os
import numpy as np

# Both inputs live outside git; the runner overrides these from the command line.
CSV_Q = os.environ.get('TRIAGO_QUESTIONNAIRE_CSV',
                       '/home/roberto/exchange/User Study Questionnaire.csv')
RES = os.environ.get(
    'TRIAGO_RESULTS_DIR',
    '/home/roberto/exchange/matlab_export/analysis_results/study_results_20260917_112426_n24')

CELLS = ["CF", "CB", "CFB", "JF", "JB", "JFB"]
# Google Forms writes one seven-column block per condition, in this order.
BLOCK_COND = ["CLUTCH FEEDBACK", "CLUTCH BLENDING", "CLUTCH FEEDBACK+BLENDING",
              "JOYSTICK FEEDBACK", "JOYSTICK BLENDING", "JOYSTICK FEEDBACK+BLENDING"]
COND2CELL = dict(zip(BLOCK_COND, CELLS))

ITEMS = ["mental", "physical", "success", "control", "trust", "smooth"]
ITEM_LABEL = {"mental": "Mental demand", "physical": "Physical demand",
              "success": "Perceived success", "control": "Sense of control",
              "trust": "Trust", "smooth": "Motion comfort"}
# +1 = a high score is a good outcome; -1 = a high score is a cost.
ITEM_DIR = {"mental": -1, "physical": -1, "success": +1,
            "control": +1, "trust": +1, "smooth": +1}

PAL = {"C": (0.169, 0.424, 0.690), "J": (0.910, 0.451, 0.059),
       "F": (0.553, 0.827, 0.373), "B": (0.098, 0.620, 0.541), "FB": (0.059, 0.373, 0.471),
       "CF": (0.455, 0.682, 0.910), "CB": (0.239, 0.486, 0.745), "CFB": (0.090, 0.290, 0.490),
       "JF": (0.976, 0.753, 0.290), "JB": (0.937, 0.486, 0.122), "JFB": (0.800, 0.200, 0.067)}

RANKMAP = {"1st (most preferred)": 1, "2nd": 2, "3rd": 3}


def load_subjective(path=None):
    rows = list(csv.reader(open(path or CSV_Q)))
    data = rows[1:]
    pids = sorted({r[1] for r in data})
    pidx = {p: i for i, p in enumerate(pids)}
    n = len(pids)

    R = {k: np.full((n, 6), np.nan) for k in ITEMS}       # participant x cell
    notes = []                                            # (pid, cell, text)
    blocks = [(8, 15), (15, 22), (22, 29), (29, 36), (36, 43), (43, 50)]

    for r in data:
        cond = r[7]
        if not cond:
            continue
        bi = BLOCK_COND.index(cond)
        a, _ = blocks[bi]
        i, j = pidx[r[1]], bi
        for k, it in enumerate(ITEMS):
            v = r[a + k].strip()
            if v:
                R[it][i, j] = float(v)
        if r[a + 6].strip():
            notes.append((r[1], CELLS[bi], r[a + 6].strip()))

    # Final-submission items: overall winner, within-mode rankings, free comment.
    best = np.full(n, '', dtype=object)
    rank = np.full((n, 6), np.nan)
    final_comment = []
    demo = {k: np.full(n, np.nan) for k in ("age", "exper")}
    gender = np.full(n, '', dtype=object)
    hand = np.full(n, '', dtype=object)

    for r in data:
        i = pidx[r[1]]
        if r[3].strip():
            demo["age"][i] = float(r[3])
        if r[6].strip():
            demo["exper"][i] = float(r[6])
        if r[4].strip():
            gender[i] = r[4].strip()
        if r[5].strip():
            hand[i] = r[5].strip()
        if r[50].strip():
            best[i] = 'J' if 'Joystick' in r[50] else 'C'
        for k, c in enumerate(range(51, 57)):
            if r[c].strip() in RANKMAP:
                rank[i, k] = RANKMAP[r[c].strip()]
        if r[57].strip():
            final_comment.append((r[1], r[57].strip()))

    return dict(pids=pids, R=R, notes=notes, best=best, rank=rank,
                final_comment=final_comment, demo=demo, gender=gender, hand=hand)


def load_objective(pids, res_dir=None):
    """Per participant x cell means of the objective metrics, averaged over the two worlds."""
    rows = list(csv.reader(open(os.path.join(res_dir or RES, 'trial_table.csv'))))
    h = {c: i for i, c in enumerate(rows[0])}
    pidx = {p: i for i, p in enumerate(pids)}
    cidx = {c: i for i, c in enumerate(CELLS)}

    wanted = ["composite", "duration_s", "ee_path_len_m", "ee_sparc", "ee_speed_mean_mps",
              "safety_min_dist_m", "cbf_active_s", "belief_mean_prob", "agreement_mean_cos",
              "alpha_autonomy_frac", "force_mean_N", "clutch_presses", "qdot_cmd_rms",
              "slack_mean", "fam_time_effectiveness", "fam_human_effort", "fam_safety"]
    acc = {k: [[[] for _ in range(6)] for _ in range(len(pids))] for k in wanted}

    for r in rows[1:]:
        p, c = r[h['participant']], r[h['cell']]
        if p not in pidx or c not in cidx:
            continue
        for k in wanted:
            v = r[h[k]].strip()
            if v and v.lower() not in ('nan', 'na'):
                acc[k][pidx[p]][cidx[c]].append(float(v))

    O = {}
    for k in wanted:
        M = np.full((len(pids), 6), np.nan)
        for i in range(len(pids)):
            for j in range(6):
                if acc[k][i][j]:
                    M[i, j] = np.mean(acc[k][i][j])
        O[k] = M
    return O


def subjective_index(R):
    """One 1-7 quality score per cell: the four positive items plus the reversed demands."""
    parts = []
    for it in ITEMS:
        X = R[it]
        parts.append(X if ITEM_DIR[it] > 0 else 8.0 - X)
    return np.nanmean(np.dstack(parts), axis=2)
