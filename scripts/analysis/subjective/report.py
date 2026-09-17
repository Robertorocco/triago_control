"""Render the subjective analysis as a standalone LaTeX report."""
import os
import re
import numpy as np
import subjdata as sd
import stats_helpers as sh
from stats_helpers import fmt_p_bare as pb

CELLS = sd.CELLS


def esc(s):
    rep = [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
           ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
           ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}')]
    for a, b in rep:
        s = s.replace(a, b)
    return s


def sig(p):
    return r'\textbf{%s}' % pb(p) if (p is not None and not np.isnan(p) and p < 0.05) else pb(p)


def build(r, out_dir):
    L = []
    A = L.append
    n = r['n']

    A(r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{caption}
\usepackage[section]{placeins}
\usepackage[colorlinks=true,linkcolor=black!70,urlcolor=blue!60]{hyperref}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{0.35em}
\setlength{\parindent}{0pt}
\title{\vspace{-1.5cm}Subjective results of the TRIAGo shared-autonomy user study}
\author{Analysis of the post-condition questionnaire, $n=24$}
\date{\today}
\begin{document}
\maketitle
\vspace{-0.7cm}""")

    # ------------------------------------------------------------------ intro
    age = r['demo']['age']
    ex = r['demo']['exper']
    gcount = {k: int((r['gender'] == k).sum()) for k in np.unique(r['gender']) if k}
    A(r"\section{What was collected}")
    A(r"""Every participant filled one short form immediately after each of the six
conditions, and one closing form at the end of the session. The per-condition form
carries six items on a 1--7 scale: mental demand, physical demand and perceived
success (the three NASA-TLX dimensions that were retained), then agreement with
``I felt in control of the robot's motion'', ``I trusted the robot to do what I
intended'' and ``The robot's motion felt smooth and comfortable''. The closing form
asks which of the two teleoperation strategies was best overall, and asks for a
preference ranking of the three assistance settings within each strategy.""")
    A(r"""All %d participants returned all six condition forms and the closing form:
%d condition reports with no missing rating, and %d complete ranking triples.
Ages span %d--%d years (mean %.1f, SD %.1f, $n=%d$; %s). Self-rated prior experience
with teleoperation devices averages %.1f on a 1--5 scale ($n=%d$), with %d
participants reporting the lowest level. Demographics are missing for one
participant who did not submit the opening form, and one further participant left
age blank."""
      % (n, n * 6, 2 * n, int(np.nanmin(age)), int(np.nanmax(age)), np.nanmean(age),
         np.nanstd(age, ddof=1), int(np.sum(~np.isnan(age))),
         ", ".join("%d %s" % (v, k.lower()) for k, v in sorted(gcount.items(),
                                                              key=lambda t: -t[1])),
         np.nanmean(ex), int(np.sum(~np.isnan(ex))), int(np.nansum(ex == 1))))

    A(r"\paragraph{Scoring.} Mental and physical demand are costs, the other four "
      r"items are benefits. Wherever a single number per condition is useful the two "
      r"demand items are reversed ($8-x$) and all six are averaged into a "
      r"\emph{subjective quality index} on the same 1--7 scale, so that higher always "
      r"means a better experience. The six items hang together well enough to justify "
      r"this: Cronbach's $\alpha = %.3f$ across the %d condition reports "
      r"($\alpha = %.3f$ for the four positive items alone). The two demand items "
      r"correlate at $\rho = %.2f$ with each other, and control, trust and comfort form "
      r"a tight cluster ($\rho = %.2f$ to $%.2f$), so the index is dominated by a "
      r"general ``this felt good'' factor rather than by any single item."
      % (r['alpha6'], n * 6, r['alpha4'], r['interitem'][0, 1],
         min(r['interitem'][3, 4], r['interitem'][3, 5], r['interitem'][4, 5]),
         max(r['interitem'][3, 4], r['interitem'][3, 5], r['interitem'][4, 5])))

    A(r"\paragraph{Tests.} The ratings are ordinal and several are visibly skewed, so "
      r"every claim below rests on rank statistics: Friedman tests for the omnibus, "
      r"Wilcoxon signed-rank for paired contrasts with Holm correction inside each "
      r"family, Mann--Whitney for the one between-participant comparison, and Spearman "
      r"correlations. Effect sizes are the matched-pairs rank-biserial correlation "
      r"$r_{\mathrm{rb}}$ and Kendall's $W$. Means are quoted for readability only.")

    # -------------------------------------------------------------- flat data
    A(r"\section{Flat results}")
    A(r"""The descriptive picture is uniform enough to state before any test. Ratings
sit in the upper half of the scale throughout --- the task was feasible in every
condition --- and the spread between conditions is driven entirely by the assistance
setting. Within each strategy the feedback-only condition is the worst on all six
items, and the two blended conditions sit close together well above it. The two
strategies, compared like with like, are almost superimposed.""")
    A(r"""\begin{table}[htbp]\centering\small
\caption{Mean $\pm$ standard deviation of every item in every condition, and the
row collapsed by strategy. Higher is better except for the two demand items.}
\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{3}{c}{Clutch} & \multicolumn{3}{c}{Joystick} & \multicolumn{2}{c}{collapsed}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}
Item & CF & CB & CFB & JF & JB & JFB & C & J\\
\midrule""")
    for it in sd.ITEMS + ["index"]:
        X = r['Q'] if it == "index" else r['R'][it]
        mu, sdv = np.nanmean(X, 0), np.nanstd(X, 0, ddof=1)
        lab = "Quality index" if it == "index" else sd.ITEM_LABEL[it]
        if it == "index":
            A(r"\midrule")
        cells = " & ".join("%.2f\\,$\\pm$\\,%.2f" % (mu[j], sdv[j]) for j in range(6))
        A(r"%s & %s & %.2f & %.2f\\" % (lab, cells, np.nanmean(X[:, :3]),
                                        np.nanmean(X[:, 3:])))
    A(r"\bottomrule\end{tabular}\end{table}")

    A(r"""\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{sub_items.pdf}
\caption{Every item, every condition. Bars are means with the standard error;
brackets give Holm-corrected Wilcoxon $p$ within each strategy. The pattern is the
same in all six items and in both strategies: the feedback-only condition is the
worst, and adding blending moves every rating in the favourable direction.}
\end{figure}""")

    A(r"""\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{sub_index.pdf}
\caption{The composite index. Left, per condition. Centre and right, one grey line
per participant: the strategy contrast is flat and disordered, the assistance
contrast rises for almost everyone.}
\end{figure}""")

    # ------------------------------------------------------------ inferential
    A(r"\section{What moved the ratings}")
    A(r"\subsection{Assistance: a large effect on everything}")
    A(r"""\begin{table}[htbp]\centering\small
\caption{Assistance setting collapsed over strategy. Friedman omnibus over the three
settings, then Holm-corrected Wilcoxon contrasts. Significant $p$ in bold.}
\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{3}{c}{mean rating} & \multicolumn{3}{c}{Friedman} & \multicolumn{2}{c}{Holm $p$}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}
Item & F & B & FB & $\chi^2(2)$ & $p$ & $W$ & F--B & F--FB\\
\midrule""")
    for it in sd.ITEMS + ["index"]:
        d = r['assist'][it]
        lab = "Quality index" if it == "index" else sd.ITEM_LABEL[it]
        if it == "index":
            A(r"\midrule")
        ph = {(q['a'], q['b']): q['p_holm'] for q in d['pairs']}
        A(r"%s & %.2f & %.2f & %.2f & %.2f & %s & %.2f & %s & %s\\"
          % (lab, d['means'][0], d['means'][1], d['means'][2], d['fried']['chi2'],
             sig(d['fried']['p']), d['fried']['W'], sig(ph[('F', 'B')]),
             sig(ph[('F', 'FB')])))
    A(r"\bottomrule\end{tabular}\end{table}")

    idx = r['assist']['index']
    bvfb = [q for q in idx['pairs'] if q['a'] == 'B'][0]
    A(r"""Assistance is the one manipulation the operators felt. Every item separates
the feedback-only setting from both blended settings, with Kendall's $W$ between
%.2f and %.2f, which for a within-subject design of this size is a large effect. The
composite index rises from %.2f under feedback alone to %.2f and %.2f under blending
and blending with feedback. What the operators did \emph{not} distinguish is
blending from blending-with-feedback: that contrast is null on every single item
(index $p = %s$), so the haptic layer adds nothing perceptible once the motion is
already being blended."""
      % (min(r['assist'][k]['fried']['W'] for k in sd.ITEMS),
         max(r['assist'][k]['fried']['W'] for k in sd.ITEMS),
         idx['means'][0], idx['means'][1], idx['means'][2], pb(bvfb['p_holm'])))

    A(r"""The direction deserves a sentence because it is not the obvious one. Blending
takes authority away from the operator, yet the item it improves most is the
\emph{sense of control}, which climbs from %.2f to %.2f. Trust climbs by a similar
amount. Operators did not experience the arbitration as a loss of control; they
experienced it as the robot finally doing what they meant."""
      % (r['assist']['control']['means'][0], r['assist']['control']['means'][1]))

    A(r"\subsection{Strategy: no subjective difference at all}")
    A(r"""\begin{table}[htbp]\centering\small
\caption{Clutch against joystick on the per-participant means, Wilcoxon signed-rank.
``J $>$ C'' counts participants who rated the joystick higher, which for the two
demand items means worse, not better. The last column is the Shapiro--Wilk $p$ on the
paired differences, reported for completeness; no contrast is significant under
either a rank or a normal-theory test.}
\begin{tabular}{lccccccc}
\toprule
Item & C & J & J$-$C & $p$ & $r_{\mathrm{rb}}$ & J $>$ C & Shapiro $p$\\
\midrule""")
    for it in sd.ITEMS + ["index"]:
        d = r['mode'][it]
        lab = "Quality index" if it == "index" else sd.ITEM_LABEL[it]
        if it == "index":
            A(r"\midrule")
        A(r"%s & %.2f & %.2f & %+.2f & %s & %+.2f & %d/%d & %.3f\\"
          % (lab, d['C'], d['J'], d['d'], sig(d['p']), d['r'], d['nfav'], n,
             d['shapiro_p']))
    A(r"\bottomrule\end{tabular}\end{table}")

    mc = r['mode']['smooth']
    A(r"""Nothing separates the two strategies. The largest of the seven contrasts is
motion comfort, and it runs %+.2f points in favour of the \emph{clutch}
($p = %s$) --- that is, in the opposite direction to the objective advantage
established in the results chapter. The composite index differs by %+.2f points
($p = %s$, $r_{\mathrm{rb}} = %+.2f$), with %d of %d participants rating the joystick
higher. This is a null result on a well-powered within-subject contrast, not a
marginal one."""
      % (-mc['d'], pb(mc['p']), r['mode']['index']['d'], pb(r['mode']['index']['p']),
         r['mode']['index']['r'], r['mode']['index']['nfav'], n))

    # ------------------------------------------------------------- preference
    A(r"\section{Stated preference}")
    A(r"""\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{sub_pref.pdf}
\caption{Left, the forced choice of the better strategy. Right, the distribution of
within-strategy preference ranks; each row sums to the %d participants.}
\end{figure}""" % n)
    pf = r['pref']
    A(r"""Asked which strategy was best overall, %d participants named the clutch and
%d named the joystick --- an even split (exact binomial $p = %s$). The forced choice
offered no ``no difference'' option, so this number cannot distinguish a genuinely
divided population from a uniformly indifferent one; the flat index contrast above
suggests the latter is at least as likely."""
      % (pf['nC'], pf['nJ'], pb(pf['p'])))
    rc, rj = r['rankstats']['C'], r['rankstats']['J']
    A(r"""Within each strategy the ranking is decisive and reproduces the rating data.
For the clutch the mean ranks are %.2f, %.2f and %.2f for feedback, blending and
blending with feedback (Friedman $\chi^2(2) = %.2f$, $p = %s$, $W = %.2f$); for the
joystick they are %.2f, %.2f and %.2f ($\chi^2(2) = %.2f$, $p = %s$, $W = %.2f$).
Feedback-only was ranked last by %d of %d participants under the clutch and by %d of
%d under the joystick, where it collected no first-place vote at all. Blending and
blending-with-feedback are again indistinguishable from each other
($p = %s$ and $p = %s$)."""
      % (rc['mean'][0], rc['mean'][1], rc['mean'][2], rc['fried']['chi2'],
         pb(rc['fried']['p']), rc['fried']['W'],
         rj['mean'][0], rj['mean'][1], rj['mean'][2], rj['fried']['chi2'],
         pb(rj['fried']['p']), rj['fried']['W'],
         rc['dist'][0][2], n, rj['dist'][0][2], n,
         pb([q for q in rc['pairs'] if q['a'] == 'B'][0]['p_holm']),
         pb([q for q in rj['pairs'] if q['a'] == 'B'][0]['p_holm'])))

    # ------------------------------------------------- cross-analysis: choice
    A(r"\section{Cross-analysis I: they performed better with the joystick, "
      r"but did they prefer it?}")
    A(r"""\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{sub_dissoc.pdf}
\caption{Left, each participant's objective advantage of the joystick against their
subjective advantage of the joystick, coloured by the strategy they named as best.
Right, the same strategy contrast measured four ways; the sign is normalised so that
a positive bar always means the joystick came out ahead.}
\end{figure}""")
    om = r['obj_mode']
    A(r"""The objective advantage is real but narrow. The composite score favours the
joystick by %.2f standard-deviation units, and %d of %d participants score higher
with it ($p = %s$, $r_{\mathrm{rb}} = %+.2f$); the hand path is %.2f\,m shorter
($p = %s$). Task time, however, does \emph{not} differ (%.0f\,s against %.0f\,s,
$p = %s$). The joystick bought path economy, not speed."""
      % (om['composite']['d'], om['composite']['nfav'], n, pb(om['composite']['p']),
         om['composite']['r'], -om['ee_path_len_m']['d'], pb(om['ee_path_len_m']['p']),
         om['duration_s']['C'], om['duration_s']['J'], pb(om['duration_s']['p'])))

    ch = r['choice']
    A(r"""Against that, the subjective record is silent, and the silence goes further
than a null group difference. The participants who named the joystick as best had an
objective joystick advantage of %+.2f, and those who named the clutch had %+.2f ---
statistically indistinguishable ($U = %.0f$, $p = %s$). Whether a participant's own
data favoured the joystick predicted their choice in %d of %d cases, which is chance.
Their own \emph{subjective} advantage did only slightly better, %d of %d
($p = %s$). In other words, the stated winner is not explained by how well the
person actually did, nor by how they rated the two strategies twenty minutes
earlier."""
      % (ch['comp']['mJ'], ch['comp']['mC'], ch['comp']['U'], pb(ch['comp']['p']),
         r['agree_obj'], n, r['agree_subj'], n, pb(ch['subj']['p'])))

    A(r"""The free-text answers say why, and one participant states the dissociation
outright: \emph{``The joystick mode is more useful and effective to complete tasks,
but it makes me nervous as it goes in an opposite direction of the movement (it seems
I need to fight against the robot) \dots{} Anyway, I chose joystick because of the
task achievement.''} The measure that improved --- path economy --- is not one the
operator can feel, while the thing they could feel, elapsed time, did not improve.
The honest reading is that the joystick's benefit is real, modest, and invisible from
inside the task.""")

    # ---------------------------------------------- cross-analysis: calibration
    A(r"\section{Cross-analysis II: did people judge their own performance well?}")
    A(r"""\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{sub_calib.pdf}
\caption{Left, perceived success against the composite score with each participant's
own mean removed, so only within-person variation remains. Centre, the same two
quantities averaged per participant. Right, the per-participant Spearman
correlations; the orange line is the median.}
\end{figure}""")
    iw = r['insight_within']['composite']
    iwt = r['insight_within']['duration_s']
    rm = r['insight_rm']
    A(r"""Yes, but only relatively. Within a participant, perceived success tracks
measured performance well: pooling the %d reports with each person's own mean removed
gives $r = %+.2f$ ($p = %s$) against the composite score and $r = %+.2f$
($p = %s$) against task time. Taken one participant at a time, the median Spearman
correlation between perceived success and the composite across their six conditions
is $%+.2f$, positive for %d of the %d participants for whom it is defined
($p = %s$). Against task time the median rises to $%+.2f$, positive for %d of %d.
Operators reliably knew which of their own conditions had gone well."""
      % (n * 6, rm['composite score']['r'], pb(rm['composite score']['p']),
         rm['shorter task time']['r'], pb(rm['shorter task time']['p']),
         iw['median'], iw['npos'], iw['nval'], pb(iw['p_wilcoxon']),
         iwt['median'], iwt['npos'], iwt['nval']))

    ib = r['insight_between']
    A(r"""Between participants that insight disappears. A participant's average
perceived success is unrelated to their average measured performance
($\rho = %+.2f$, $p = %s$ against the composite; $\rho = %+.2f$, $p = %s$ against
task time). The slowest operators rated their own success as highly as the fastest.
This is the standard relative-but-not-absolute calibration pattern, and it has a
practical consequence for the thesis: self-report is usable as a
\emph{within-participant} comparison between conditions, which is exactly how the
study uses it, and is not usable as an estimate of an operator's competence."""
      % (ib['composite score']['rho'], pb(ib['composite score']['p']),
         ib['shorter task time']['rho'], pb(ib['shorter task time']['p'])))

    # ------------------------------------------------- cross-analysis: links
    A(r"\section{Cross-analysis III: what the ratings actually track}")
    A(r"""\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{sub_links.pdf}
\caption{Repeated-measures correlation between each subjective item and the objective
quantity it ought to reflect, computed within participants and Holm-corrected across
the twelve pairs. Grey bars are not significant.}
\end{figure}""")

    def link(item, lab):
        for d in r['links']:
            if d['item'] == item and d['lab'] == lab:
                return d
        raise KeyError(lab)

    A(r"""\begin{table}[htbp]\centering\small
\caption{The same twelve pairs in numbers. $r$ is the pooled within-participant
correlation, $p_{\mathrm{Holm}}$ is corrected across the whole family.}
\begin{tabular}{llcccc}
\toprule
Subjective item & Objective quantity & $r$ & df & $p$ & $p_{\mathrm{Holm}}$\\
\midrule""")
    for d in r['links']:
        A(r"%s & %s & %+.3f & %d & %s & %s\\"
          % (sd.ITEM_LABEL[d['item']], d['lab'], d['r'], d['df'], pb(d['p']),
             sig(d['p_holm'])))
    A(r"\bottomrule\end{tabular}\end{table}")

    mt = link('mental', 'task time')
    tf = link('trust', 'less filter activity')
    tc = link('trust', 'minimum clearance')
    ti = link('trust', 'intent confidence')
    ca = link('control', 'less autonomy-led time')
    cg = link('control', 'operator-policy agreement')
    pf_ = link('physical', 'rendered force')
    pq = link('physical', 'commanded joint rate')
    pc = link('physical', 're-indexing operations')
    ms = link('smooth', 'movement smoothness')
    sl = link('smooth', 'less tracking slack')

    A(r"""Four of these links are worth carrying into the discussion.""")
    A(r"""\emph{Mental demand is a clock.} Its strongest correlate is elapsed time
($r = %+.2f$, $p = %s$) followed by path length ($r = %+.2f$). Perceived mental
effort in this task is very largely perceived duration."""
      % (mt['r'], pb(mt['p_holm']), link('mental', 'hand path length')['r']))
    A(r"""\emph{Trust tracks the intervention, not the danger.} Trust rises when the
safety filter is quiet ($r = %+.2f$, $p = %s$) and when the intent estimate is
confident ($r = %+.2f$, $p = %s$), but it is flat against the quantity that actually
measures risk, the minimum clearance ($r = %+.2f$, $p = %s$). Operators cannot
perceive how close they came to a collision; they perceive being corrected. A filter
that intervenes early and often will be distrusted even when it is the reason nothing
went wrong."""
      % (tf['r'], pb(tf['p_holm']), ti['r'], pb(ti['p_holm']), tc['r'],
         pb(tc['p_holm'])))
    A(r"""\emph{The sense of control is blind to the arbitration.} It is uncorrelated
with the fraction of time the policy led ($r = %+.2f$, $p = %s$) and with
operator--policy agreement ($r = %+.2f$, $p = %s$), even though it is the item that
blending raised the most. Felt control responds to whether the task is going well,
not to how much authority the arbitration actually held."""
      % (ca['r'], pb(ca['p_holm']), cg['r'], pb(cg['p_holm'])))
    A(r"""\emph{Physical demand is about re-indexing, not about force.} It correlates
with the number of clutch re-indexing operations ($r = %+.2f$, $p = %s$) but not with
the rendered haptic force ($r = %+.2f$, $p = %s$) and not at all with the commanded
joint rate ($r = %+.2f$, $p = %s$). This corroborates, from the operator's side, the
caveat already raised in the results chapter: the joint-rate statistic used for the
human-effort family is a property of the robot, not a measure of what the person
did."""
      % (pc['r'], pb(pc['p_holm']), pf_['r'], pb(pf_['p_holm']), pq['r'],
         pb(pq['p_holm'])))
    A(r"""Motion comfort behaves sensibly but weakly: it follows the tracking slack
($r = %+.2f$, $p = %s$) more closely than the spectral-arc-length smoothness measure
($r = %+.2f$, $p = %s$), which suggests that what operators call ``smooth'' is the
robot keeping up with the command rather than the geometric quality of the path."""
      % (sl['r'], pb(sl['p_holm']), ms['r'], pb(ms['p_holm'])))

    # ------------------------------------------------------------- experience
    A(r"\section{Prior experience}")
    exs = r['exper']
    A(r"""Self-rated prior experience explains nothing. It is uncorrelated with the
composite score ($\rho = %+.2f$, $p = %s$), with task time ($\rho = %+.2f$,
$p = %s$), with perceived success ($\rho = %+.2f$, $p = %s$) and with the subjective
index ($\rho = %+.2f$, $p = %s$), and it does not differ between the participants who
chose the joystick and those who chose the clutch ($p = %s$). With %d of %d
respondents at the lowest level the scale has little room to discriminate, so this is
weak evidence of no effect rather than evidence of no effect."""
      % (exs['composite score']['rho'], pb(exs['composite score']['p']),
         exs['task time']['rho'], pb(exs['task time']['p']),
         exs['perceived success']['rho'], pb(exs['perceived success']['p']),
         exs['subjective index']['rho'], pb(exs['subjective index']['p']),
         pb(r['exper_choice']['p']),
         int(np.nansum(r['demo']['exper'] == 1)),
         int(np.sum(~np.isnan(r['demo']['exper'])))))

    # ------------------------------------------------------------- free text
    A(r"\section{What the free text says}")
    A(r"""%d of the %d condition forms carried a note, and %d participants left a
closing comment. Read together they fall into five recurring themes, listed with the
participants who raised them."""
      % (len(r['notes']), n * 6, len(r['final_comment'])))
    themes = [
        ("The haptic feedback is intrusive or confusing",
         "P05, P06, P07, P10, P14, P16, P17, P19, P22",
         "the single most common complaint, and it is raised about both strategies. "
         "Wording ranges from ``not always intuitive'' and ``confusing'' to ``really "
         "annoying'' and ``too much force applied''. It is the direct qualitative "
         "counterpart of the F-versus-B result."),
        ("Blending is what people liked",
         "P05, P06, P14, P19, P04",
         "described as giving more control, as helping most near the goal and in "
         "orientation, and in one case as the best of the joystick conditions."),
        ("The assistance sometimes guided towards the wrong goal",
         "P04, P06, P08, P21",
         "the guidance is reported as suggesting wider lateral motion than intended, "
         "as leading away from the easiest route, and as misleading when a competing "
         "goal sits near the current position. This is the belief estimator picking "
         "the wrong target, and it is visible to the operator."),
        ("Feedback on top of blending overwrites intention",
         "P16, P19",
         "both state that with feedback added they no longer felt in control, which "
         "is the one qualitative signal running against the flat B-versus-FB result."),
        ("The joystick feels like fighting the robot",
         "P10, P14, P17",
         "reported as fatiguing over time, as too fast, and as moving opposite to the "
         "operator's motion. This is the most plausible explanation for why a "
         "measurable objective advantage produced no preference."),
    ]
    A(r"\begin{description}")
    for t, who, txt in themes:
        A(r"\item[%s] (%s) --- %s" % (esc(t), who, txt))
    A(r"\end{description}")
    A(r"""Three notes are about session conduct rather than about a condition and are
flagged here as data-quality items: one participant reports repeated grasp attempts
caused by their own misplacement, one reports pressing the wrong button, and one
reports that instructions given during the trial lengthened their time. One further
note (P22, clutch feedback) is a facetious answer whose only codeable content is a
strong negative reaction to the clutch feedback forces; it is counted in the first
theme and is reproduced in the appendix unedited.""")

    # ----------------------------------------------------------------- caveats
    A(r"\section{Caveats to weigh before using any of this}")
    caveats = [
        ("Only three NASA-TLX dimensions were asked",
         "temporal demand, effort and frustration were not collected, so no raw or "
         "weighted TLX workload score can be computed and none is reported. The three "
         "retained items must be read individually, not as a workload scale."),
        ("Every construct is a single item",
         "trust, control and comfort each rest on one 1--7 question, so item-level "
         "reliability cannot be estimated and measurement error is absorbed entirely "
         "into the residual. The composite index is the only quantity here with a "
         "reliability figure attached."),
        ("The index mixes two constructs",
         "averaging reversed workload items with quality items produces a usable "
         "summary ($\\alpha = %.3f$) but is not a validated instrument; every "
         "conclusion drawn from the index is also visible item by item, which is why "
         "both are reported." % r['alpha6']),
        ("The preference question is forced and binary",
         "with no neutral option, the %d--%d split cannot separate a divided "
         "population from an indifferent one." % (r['pref']['nC'], r['pref']['nJ'])),
        ("Rankings are within-strategy only",
         "participants never ranked the six conditions against each other, so no "
         "overall preference order exists and the clutch and joystick ranks cannot be "
         "pooled."),
        ("The choice-group comparison is between participants",
         "%d against %d, which gives it far less power than everything else in this "
         "report. It can exclude only a large association between individual "
         "performance and individual choice."
         % (r['pref']['nJ'], r['pref']['nC'])),
        ("Ratings inherit the order confound",
         "each form was filled immediately after its condition, so practice and "
         "fatigue affect the ratings exactly as they affect the objective measures; "
         "the counterbalancing addresses this at the group level only."),
        ("Prior experience is one self-rated item",
         "a 1--5 self-report with %d of %d respondents at the floor is a weak "
         "instrument, and the null results involving it should be read accordingly."
         % (int(np.nansum(r['demo']['exper'] == 1)),
            int(np.sum(~np.isnan(r['demo']['exper']))))),
        ("Two participants have incomplete demographics",
         "one missed the opening form entirely and one left age blank; ratings are "
         "complete for all %d." % n),
        ("The twelve subjective--objective pairs were chosen in advance but not "
         "pre-registered",
         "they are Holm-corrected as a family, which is the right correction for a "
         "planned set, but they remain exploratory."),
    ]
    A(r"\begin{description}")
    for t, txt in caveats:
        A(r"\item[%s] %s" % (esc(t), txt))
    A(r"\end{description}")

    # ---------------------------------------------------------------- summary
    A(r"\section{Summary}")
    A(r"""Four findings survive every check. Assistance dominates the subjective
record: blending improves all six items with large effects, and adding haptic
feedback on top of blending adds nothing that operators can detect. The two
teleoperation strategies are subjectively identical, on every item and in the forced
choice, even though the joystick is measurably better on the composite score and on
path length --- and an individual's own performance gap does not predict which one
they name. Operators judge their own conditions accurately in relative terms and not
at all in absolute terms. And the ratings that look like safety and authority
judgements are nothing of the kind: trust follows how often the filter intervened
rather than how close the robot came to a collision, and the sense of control is
blind to how much authority the arbitration actually took.""")

    # --------------------------------------------------------------- appendix
    A(r"\appendix")
    A(r"\section{Verbatim free-text responses}")
    A(r"\subsection{Per-condition notes}")
    A(r"\begin{small}\begin{description}")
    for p, c, t in r['notes']:
        A(r"\item[%s, %s] %s" % (p, c, esc(t)))
    A(r"\end{description}\end{small}")
    A(r"\subsection{Closing comments}")
    A(r"\begin{small}\begin{description}")
    for p, t in r['final_comment']:
        A(r"\item[%s] %s" % (p, esc(t)))
    A(r"\end{description}\end{small}")

    A(r"\end{document}")

    tex = "\n\n".join(L)
    # Prose writes "$p = {value}$" while a floored p-value already carries its own
    # relational operator; drop the duplicated "=" rather than branch at every call.
    tex = re.sub(r'=\s*<\s*0\.001', '< 0.001', tex)
    path = os.path.join(out_dir, 'subjective_report.tex')
    with open(path, 'w') as f:
        f.write(tex)
    return path
