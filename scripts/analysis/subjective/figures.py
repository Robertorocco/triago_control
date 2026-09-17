"""Figures for the subjective report, in the palette the objective chapter uses."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import subjdata as sd
import stats_helpers as sh

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': '0.25',
    'xtick.color': '0.25', 'ytick.color': '0.25', 'axes.labelcolor': '0.15',
    'text.color': '0.15', 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'figure.dpi': 150,
    'pdf.fonttype': 42,
})
CELLS = sd.CELLS
COL = [sd.PAL[c] for c in CELLS]


def _stars(p):
    if np.isnan(p):
        return "n.s."
    if p < 0.001:
        return "p<.001"
    if p < 0.01:
        return "p=%.3f" % p
    if p < 0.05:
        return "p=%.3f" % p
    return "n.s."


def _brackets(ax, spans, top, span_h, fs=6.0):
    """Draw stacked significance brackets above the bars."""
    step = 0.13 * span_h
    used = []
    for (x1, x2, p) in spans:
        lvl = None
        for li, u in enumerate(used):
            if x1 > u + 0.35:
                used[li] = x2
                lvl = li
                break
        if lvl is None:
            used.append(x2)
            lvl = len(used) - 1
        y = top + step * (lvl + 1)
        ax.plot([x1, x1, x2, x2], [y - 0.22 * step, y, y, y - 0.22 * step],
                color='0.2', lw=0.6, clip_on=False)
        ax.text((x1 + x2) / 2, y + 0.04 * step, _stars(p), ha='center', va='bottom',
                fontsize=fs, color='0.2', clip_on=False)
    return top + step * (len(used) + 1.15)


def _cellbars(ax, X, ylab, brackets=True, ylim_lo=1.0):
    mu = np.nanmean(X, 0)
    se = np.nanstd(X, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(X), 0))
    xs = np.arange(6)
    for i in xs:
        ax.bar(i, mu[i] - ylim_lo, bottom=ylim_lo, width=0.72, color=COL[i],
               edgecolor=np.array(COL[i]) * 0.72, linewidth=0.5)
    ax.errorbar(xs, mu, se, fmt='none', ecolor='k', elinewidth=0.7, capsize=2.5)
    ax.axvline(2.5, color='0.8', lw=0.6)
    top = np.nanmax(mu + se)
    if brackets:
        spans = []
        for off in (0, 3):
            pw = [sh.wilcoxon_pair(X[:, off + a], X[:, off + b]) for a, b in
                  [(0, 1), (0, 2), (1, 2)]]
            adj = sh.holm([q['p'] for q in pw])
            for (a, b), p in zip([(0, 1), (0, 2), (1, 2)], adj):
                if p < 0.05:
                    spans.append((off + a, off + b, p))
        top = _brackets(ax, spans, top, 7.0 - ylim_lo)
    ax.set_xticks(xs)
    ax.set_xticklabels(CELLS, fontsize=7)
    ax.set_xlim(-0.65, 5.65)
    ax.set_ylim(ylim_lo, max(7.05, top))
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_ylabel(ylab, fontsize=7.5)
    ax.tick_params(length=2.5, labelsize=7)


def fig_items(r, path):
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.5))
    for ax, it in zip(axes.ravel(), sd.ITEMS):
        _cellbars(ax, r['R'][it], '')
        arrow = "lower is better" if sd.ITEM_DIR[it] < 0 else "higher is better"
        ax.set_title(f"{sd.ITEM_LABEL[it]}  ({arrow})", fontsize=8, pad=4)
    fig.supylabel('rating (1--7)', fontsize=8)
    fig.tight_layout(rect=(0.015, 0, 1, 1))
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_index(r, path):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7),
                             gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    _cellbars(axes[0], r['Q'], 'subjective quality index')
    axes[0].set_title('Composite subjective index', fontsize=8, pad=4)

    Q = r['Q']
    c = np.nanmean(Q[:, [0, 1, 2]], 1)
    j = np.nanmean(Q[:, [3, 4, 5]], 1)
    ax = axes[1]
    for i in range(len(c)):
        ax.plot([0, 1], [c[i], j[i]], color='0.75', lw=0.6, marker='o', ms=2.2,
                mfc='0.55', mec='none', zorder=1)
    ax.plot([0, 1], [c.mean(), j.mean()], color='k', lw=1.8, marker='o', ms=5, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['clutch', 'joystick'], fontsize=7.5)
    ax.set_xlim(-0.35, 1.35)
    ax.set_title('Mode: %s' % _stars(r['mode']['index']['p']), fontsize=8, pad=4)
    ax.tick_params(length=2.5, labelsize=7)

    A = r['assist']['index']['A']
    ax = axes[2]
    for i in range(A.shape[0]):
        ax.plot([0, 1, 2], A[i], color='0.75', lw=0.6, marker='o', ms=2.2,
                mfc='0.55', mec='none', zorder=1)
    ax.plot([0, 1, 2], A.mean(0), color='k', lw=1.8, marker='o', ms=5, zorder=3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['F', 'B', 'FB'], fontsize=7.5)
    ax.set_xlim(-0.35, 2.35)
    ax.set_title('Assistance: %s' % _stars(r['assist']['index']['fried']['p']),
                 fontsize=8, pad=4)
    ax.tick_params(length=2.5, labelsize=7)
    for ax in axes[1:]:
        ax.set_ylim(1, 7.2)
        ax.set_yticks([1, 3, 5, 7])
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_pref(r, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6),
                             gridspec_kw=dict(width_ratios=[1, 2]))
    ax = axes[0]
    ax.bar([0, 1], [r['pref']['nC'], r['pref']['nJ']], width=0.6,
           color=[sd.PAL['C'], sd.PAL['J']],
           edgecolor=[np.array(sd.PAL['C']) * 0.72, np.array(sd.PAL['J']) * 0.72], lw=0.5)
    for x, v in zip([0, 1], [r['pref']['nC'], r['pref']['nJ']]):
        ax.text(x, v + 0.3, str(v), ha='center', fontsize=8)
    ax.axhline(r['n'] / 2, color='0.4', ls='--', lw=0.7)
    ax.text(1.42, r['n'] / 2, 'chance', fontsize=6.5, color='0.4', va='center')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['clutch', 'joystick'], fontsize=7.5)
    ax.set_ylabel('participants choosing as best', fontsize=7.5)
    ax.set_ylim(0, r['n'] * 0.72)
    ax.set_yticks(range(0, 17, 4))
    ax.set_title('Overall winner (%s)' % _stars(r['pref']['p']), fontsize=8, pad=4)
    ax.tick_params(length=2.5, labelsize=7)

    ax = axes[1]
    shades = ['0.20', '0.55', '0.82']
    order = [('C', 0), ('C', 1), ('C', 2), ('J', 0), ('J', 1), ('J', 2)]
    for k, (m, t) in enumerate(order):
        left = 0
        for ri in range(3):
            v = r['rankstats'][m]['dist'][t][ri]
            ax.barh(k, v, left=left, color=shades[ri], edgecolor='w', lw=0.6, height=0.68)
            if v >= 2:
                ax.text(left + v / 2, k, str(v), ha='center', va='center', fontsize=6.5,
                        color='w' if ri < 2 else '0.2')
            left += v
    ax.set_yticks(range(6))
    ax.set_yticklabels(CELLS, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel('participants', fontsize=7.5)
    ax.set_xlim(0, r['n'])
    ax.set_title('Within-mode preference ranking', fontsize=8, pad=4)
    ax.tick_params(length=2.5, labelsize=7)
    ax.legend(handles=[Patch(facecolor=shades[i], edgecolor='w',
                             label=['1st', '2nd', '3rd'][i]) for i in range(3)],
              fontsize=6.5, frameon=False, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_dissoc(r, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    d = r['delta']
    ax = axes[0]
    ax.axhline(0, color='0.75', lw=0.7)
    ax.axvline(0, color='0.75', lw=0.7)
    for g, col, lab in [(d['gC'], sd.PAL['C'], 'chose clutch'),
                        (d['gJ'], sd.PAL['J'], 'chose joystick')]:
        ax.scatter(d['comp'][g], d['subj'][g], s=26, color=col, edgecolor='w',
                   linewidth=0.5, label=lab, zorder=3)
    ax.set_xlabel('objective advantage of joystick\n(composite, J $-$ C)', fontsize=7.5)
    ax.set_ylabel('subjective advantage of joystick\n(index, J $-$ C)', fontsize=7.5)
    ax.legend(fontsize=6.5, frameon=False, loc='upper left')
    ax.set_title('Per participant: performance vs preference', fontsize=8, pad=4)
    ax.tick_params(length=2.5, labelsize=7)

    ax = axes[1]
    names = ['composite\nscore', 'hand path\nlength', 'task\ntime', 'subjective\nindex']
    vals = [r['obj_mode']['composite']['r'], r['obj_mode']['ee_path_len_m']['r'],
            r['obj_mode']['duration_s']['r'], r['mode']['index']['r']]
    ps = [r['obj_mode']['composite']['p'], r['obj_mode']['ee_path_len_m']['p'],
          r['obj_mode']['duration_s']['p'], r['mode']['index']['p']]
    # path length and time are costs: flip so that positive always means "joystick better"
    vals = [vals[0], -vals[1], -vals[2], vals[3]]
    cols = ['0.35', '0.35', '0.35', sd.PAL['J']]
    ax.axvline(0, color='0.75', lw=0.7)
    y = np.arange(4)
    ax.barh(y, vals, color=cols, height=0.6, edgecolor='none')
    for i, (v, p) in enumerate(zip(vals, ps)):
        ax.text(v + (0.05 if v >= 0 else -0.05), i, _stars(p), va='center',
                ha='left' if v >= 0 else 'right', fontsize=6.5, color='0.25')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(-1.15, 1.35)
    ax.set_xlabel('rank-biserial effect size\n(positive = joystick better)', fontsize=7.5)
    ax.set_title('Objective gain, subjective silence', fontsize=8, pad=4)
    ax.tick_params(length=2.5, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_calib(r, path):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    suc, comp = r['R']['success'], r['O']['composite']
    n = r['n']

    ax = axes[0]
    for i in range(n):
        x = comp[i] - np.nanmean(comp[i])
        y = suc[i] - np.nanmean(suc[i])
        ax.plot(x, y, 'o', ms=2.6, color='0.6', mec='none', alpha=0.75, zorder=2)
    X = (comp - np.nanmean(comp, 1, keepdims=True)).reshape(-1)
    Y = (suc - np.nanmean(suc, 1, keepdims=True)).reshape(-1)
    ok = ~(np.isnan(X) | np.isnan(Y))
    b, a = np.polyfit(X[ok], Y[ok], 1)
    xs = np.linspace(np.nanmin(X), np.nanmax(X), 20)
    ax.plot(xs, b * xs + a, color=sd.PAL['CB'], lw=1.6, zorder=3)
    rc = r['insight_rm']['composite score']
    ax.set_title('Within participant\n$r$ = %+.2f, %s' % (rc['r'], _stars(rc['p'])),
                 fontsize=8, pad=4)
    ax.set_xlabel('composite score (centred)', fontsize=7.5)
    ax.set_ylabel('perceived success (centred)', fontsize=7.5)
    ax.tick_params(length=2.5, labelsize=7)

    ax = axes[1]
    x, y = np.nanmean(comp, 1), np.nanmean(suc, 1)
    ax.scatter(x, y, s=26, color=sd.PAL['CB'], edgecolor='w', linewidth=0.5, zorder=3)
    b2, a2 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 20)
    ax.plot(xs, b2 * xs + a2, color='0.45', lw=1.2, ls='--', zorder=2)
    ib = r['insight_between']['composite score']
    ax.set_title('Between participants\n$\\rho$ = %+.2f, %s' % (ib['rho'], _stars(ib['p'])),
                 fontsize=8, pad=4)
    ax.set_xlabel('mean composite score', fontsize=7.5)
    ax.set_ylabel('mean perceived success', fontsize=7.5)
    ax.tick_params(length=2.5, labelsize=7)

    ax = axes[2]
    rs = r['insight_within']['composite']['rho']
    rs = rs[~np.isnan(rs)]
    ax.hist(rs, bins=np.arange(-1.0, 1.01, 0.2), color=sd.PAL['CB'],
            edgecolor='w', linewidth=0.6)
    ax.axvline(0, color='0.4', lw=0.8, ls='--')
    ax.axvline(np.median(rs), color=sd.PAL['J'], lw=1.6)
    ax.set_xlabel(r'per-participant Spearman $\rho$', fontsize=7.5)
    ax.set_ylabel('participants', fontsize=7.5)
    ax.set_title('Individual insight\nmedian %+.2f, %d/%d positive'
                 % (np.median(rs), r['insight_within']['composite']['npos'],
                    r['insight_within']['composite']['nval']), fontsize=8, pad=4)
    ax.tick_params(length=2.5, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_links(r, path):
    L = r['links']
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    y = np.arange(len(L))[::-1]
    for k, (yy, d) in enumerate(zip(y, L)):
        sig = d['p_holm'] < 0.05
        col = sd.PAL['CB'] if d['r'] >= 0 else sd.PAL['JB']
        ax.barh(yy, d['r'], height=0.62, color=col if sig else '0.82',
                edgecolor='none', zorder=2)
        ax.text(d['r'] + (0.02 if d['r'] >= 0 else -0.02), yy,
                ('%+.2f, ' % d['r']) + _stars(d['p_holm']),
                va='center', ha='left' if d['r'] >= 0 else 'right',
                fontsize=6.3, color='0.25')
    ax.axvline(0, color='0.6', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(['%s  vs  %s' % (sd.ITEM_LABEL[d['item']], d['lab']) for d in L],
                       fontsize=7)
    ax.set_xlim(-0.3, 0.78)
    ax.set_xlabel('repeated-measures correlation $r$ (Holm corrected; '
                  'grey = not significant)', fontsize=7.5)
    ax.tick_params(length=2.5, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def make_all(r, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    fig_items(r, os.path.join(out_dir, 'sub_items.pdf'))
    fig_index(r, os.path.join(out_dir, 'sub_index.pdf'))
    fig_pref(r, os.path.join(out_dir, 'sub_pref.pdf'))
    fig_dissoc(r, os.path.join(out_dir, 'sub_dissoc.pdf'))
    fig_calib(r, os.path.join(out_dir, 'sub_calib.pdf'))
    fig_links(r, os.path.join(out_dir, 'sub_links.pdf'))
