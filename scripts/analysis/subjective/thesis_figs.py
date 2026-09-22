"""Thesis-styled panels for the questionnaire results.

Drawn to match the MATLAB panels of the objective chapter: no title, every string
typeset by the real LaTeX engine in the document's own Latin Modern face, no grid,
sized in centimetres so LaTeX includes them at native size, vector PDF.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import subjdata as sd
import stats_helpers as sh

CM = 1 / 2.54
FS = 7.0            # tick face, as printed
FS_P = 5.6          # p-value labels
W_ITEM, H_ITEM = 4.95 * CM, 4.0 * CM
W_RANK, H_RANK = 10.5 * CM, 4.3 * CM

plt.rcParams.update({
    # Same engine and same font package as the document, so ticks and labels are
    # set in the body face rather than merely resembling it. The three family
    # lists are named away from matplotlib's own Computer Modern aliases, which
    # would pull in type1ec/cm-super; lmodern already supplies scalable T1 faces.
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{lmodern}',
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman'],
    'font.sans-serif': ['Latin Modern Sans'],
    'font.monospace': ['Courier'],
    'font.size': FS, 'axes.linewidth': 0.6, 'axes.edgecolor': '0.25',
    'xtick.color': '0.25', 'ytick.color': '0.25', 'axes.labelcolor': '0.15',
    'text.color': '0.15', 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'pdf.fonttype': 42, 'savefig.pad_inches': 0.01,
})

# A rank is an ordinal magnitude, so the three steps are one hue ordered by
# lightness; the hue is deliberately neither of the two that carry mode identity.
RANK_COL = [(0.247, 0.157, 0.416), (0.494, 0.400, 0.690), (0.796, 0.761, 0.886)]
RANK_TXT = ['w', 'w', '0.15']

ITEM_SYMBOL = {
    'mental':   r'$S_{\mathrm{md}}$',
    'physical': r'$S_{\mathrm{pd}}$',
    'success':  r'$S_{\mathrm{pf}}$',
    'control':  r'$S_{\mathrm{ct}}$',
    'trust':    r'$S_{\mathrm{tr}}$',
    'smooth':   r'$S_{\mathrm{sm}}$',
}
ITEM_FILE = {'mental': 'md', 'physical': 'pd', 'success': 'pf',
             'control': 'ct', 'trust': 'tr', 'smooth': 'sm'}
BASE = 1.0          # the floor of the response scale, not zero


def _fmt_p(p):
    return r'$p<0.001$' if p < 0.001 else r'$p=%.3f$' % p


def _sig_pairs(X):
    """The two contrasts against feedback-only, inside each mode, where significant.

    Holm still corrects over all three assistance pairs, but only the two drawn here
    are shown, so every panel of the matrix stacks to the same two levels and the
    blending-against-both contrast is left to the table.
    """
    out = []
    for off in (0, 3):
        pw = [sh.wilcoxon_pair(X[:, off + a], X[:, off + b])
              for a, b in [(0, 1), (0, 2), (1, 2)]]
        adj = sh.holm([q['p'] for q in pw])
        for (a, b), p in zip([(0, 1), (0, 2)], adj[:2]):
            if p < 0.05:
                out.append((off + a, off + b, float(p)))
    return out


def _pack(spans):
    """Skyline packing: returns the level of each bracket and the level count."""
    used, lvls = [], []
    for x1, x2, _ in spans:
        lvl = None
        for li, u in enumerate(used):
            if x1 > u + 0.35:
                used[li] = x2
                lvl = li
                break
        if lvl is None:
            used.append(x2)
            lvl = len(used) - 1
        lvls.append(lvl)
    return lvls, len(used)


def _levels(X):
    spans = _sig_pairs(X)
    return _pack(spans)[1] if spans else 0


def _item_panel(X, symbol, ylo, yhi, base, step, path):
    fig = plt.figure(figsize=(W_ITEM, H_ITEM))
    ax = fig.add_axes([0.215, 0.135, 0.755, 0.845])
    ax.set_facecolor('none')
    fig.patch.set_facecolor('w')

    mu = np.nanmean(X, 0)
    se = np.nanstd(X, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(X), 0))
    for i in range(6):
        c = sd.PAL[sd.CELLS[i]]
        ax.bar(i, mu[i] - BASE, bottom=BASE, width=0.70, color=c,
               edgecolor=np.array(c) * 0.72, linewidth=0.5, zorder=2)
    ax.errorbar(np.arange(6), mu, se, fmt='none', ecolor='k', elinewidth=0.7,
                capsize=2.2, capthick=0.7, zorder=3)
    ax.axvline(2.5, color=(0.75, 0.75, 0.75), lw=0.6, zorder=1)

    spans = _sig_pairs(X)
    lvls, _ = _pack(spans)
    for (x1, x2, p), lvl in zip(spans, lvls):
        y = base + step * lvl
        ax.plot([x1, x1, x2, x2], [y - 0.20 * step, y, y, y - 0.20 * step],
                color='k', lw=0.6, zorder=4)
        ax.text((x1 + x2) / 2, y + 0.05 * step, _fmt_p(p), ha='center', va='bottom',
                fontsize=FS_P, color='0.15', zorder=4)

    ax.set_xticks(range(6))
    ax.set_xticklabels(sd.CELLS, fontsize=6.4)
    ax.set_xlim(-0.62, 5.62)
    ax.set_ylim(ylo, yhi)
    ax.set_yticks([1, 3, 5, 7])
    # The axis stops at the ceiling of the response scale; the brackets sit above it.
    ax.spines['left'].set_bounds(ylo, 7)
    ax.set_ylabel(symbol, fontsize=FS + 1.5, labelpad=2)
    ax.tick_params(length=2.2, width=0.6, pad=1.5, labelsize=FS - 0.6)
    ax.grid(False)
    fig.savefig(path, format='pdf', transparent=False,
                metadata={'CreationDate': None})
    plt.close(fig)


def item_panels(r, out_dir):
    """The six items on one common vertical scale so the matrix reads as a unit."""
    # step clears a whole p-label: LaTeX math sets them taller than the sans
    # text it replaced, and at 0.60 the lower level touched the bracket above.
    base, step = 7.30, 0.82
    nlvl = max(_levels(r['R'][it]) for it in sd.ITEMS)
    ylo, yhi = BASE, base + step * max(nlvl - 1, 0) + 0.62
    for it in sd.ITEMS:
        _item_panel(r['R'][it], ITEM_SYMBOL[it], ylo, yhi, base, step,
                    os.path.join(out_dir, 'res_subj_%s.pdf' % ITEM_FILE[it]))
    return ylo, yhi


def rank_panel(r, path):
    fig = plt.figure(figsize=(W_RANK, H_RANK))
    ax = fig.add_axes([0.105, 0.245, 0.875, 0.720])
    fig.patch.set_facecolor('w')
    n = r['n']

    rows = [('C', 0), ('C', 1), ('C', 2), ('J', 0), ('J', 1), ('J', 2)]
    for k, (m, t) in enumerate(rows):
        left = 0
        for ri in range(3):
            v = r['rankstats'][m]['dist'][t][ri]
            if v == 0:
                continue
            ax.barh(k, v, left=left, color=RANK_COL[ri], edgecolor='w',
                    linewidth=0.7, height=0.70, zorder=2)
            ax.text(left + v / 2, k, str(v), ha='center', va='center',
                    fontsize=FS - 0.6, color=RANK_TXT[ri], zorder=3)
            left += v
    ax.axhline(2.5, color=(0.75, 0.75, 0.75), lw=0.6, zorder=1)

    ax.set_yticks(range(6))
    ax.set_yticklabels(sd.CELLS, fontsize=FS)
    ax.invert_yaxis()
    ax.set_xlim(0, n)
    ax.set_xticks(range(0, n + 1, 4))
    # No axis title: the caption names the count, as on the objective panels.
    ax.tick_params(length=2.2, width=0.6, pad=1.5, labelsize=FS - 0.4)
    ax.grid(False)
    ax.legend(handles=[Patch(facecolor=RANK_COL[i], edgecolor='w',
                             label=['first', 'second', 'third'][i]) for i in range(3)],
              fontsize=FS - 0.6, frameon=False, ncol=3, loc='lower center',
              bbox_to_anchor=(0.5, -0.27), handlelength=1.1, handleheight=0.9,
              columnspacing=1.4, borderpad=0.0)
    fig.savefig(path, format='pdf', transparent=False,
                metadata={'CreationDate': None})
    plt.close(fig)


def make_all(r, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ylo, yhi = item_panels(r, out_dir)
    rank_panel(r, os.path.join(out_dir, 'res_subj_ranks.pdf'))
    return ylo, yhi
