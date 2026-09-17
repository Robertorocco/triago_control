#!/usr/bin/env python3
"""Analyse the post-condition questionnaire and typeset the subjective report.

Reads the Google Forms export and the objective trial table produced by the MATLAB
pipeline, cross-references the two, and writes subjective_report.pdf.

    python3 run_subjective_report.py --csv "<export>.csv" --results <study_results_dir>
"""
import argparse
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subjdata  # noqa: E402

# .../scripts/analysis/subjective -> repo root -> the figures the thesis includes
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_THESIS_DIR = os.path.join(_REPO, 'thesis', 'figures', 'results')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', default=subjdata.CSV_Q, help='questionnaire CSV export')
    ap.add_argument('--results', default=subjdata.RES,
                    help='study_results_* directory holding trial_table.csv')
    ap.add_argument('--out', default=None,
                    help='output directory (default: <results>/subjective)')
    ap.add_argument('--thesis-dir', default=DEFAULT_THESIS_DIR,
                    help='where the thesis-styled panels go ("" to skip)')
    ap.add_argument('--no-open', action='store_true', help='do not open the PDF')
    a = ap.parse_args()

    subjdata.CSV_Q, subjdata.RES = a.csv, a.results
    out_dir = a.out or os.path.join(a.results, 'subjective')
    os.makedirs(out_dir, exist_ok=True)

    import analysis
    import figures
    import report

    r = analysis.run()
    print("analysis: n = %d, %d condition reports" % (r['n'], r['n'] * 6))
    figures.make_all(r, out_dir)
    print("figures:  6 written")

    if a.thesis_dir:
        import thesis_figs
        thesis_figs.make_all(r, a.thesis_dir)
        print("thesis:   7 panels -> %s" % a.thesis_dir)

    tex = report.build(r, out_dir)

    for _ in range(2):
        p = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                            os.path.basename(tex)], cwd=out_dir,
                           capture_output=True, text=True)
    if p.returncode != 0:
        bad = [l for l in p.stdout.splitlines() if l.startswith('!')]
        print("pdflatex failed:\n" + "\n".join(bad[-20:] or p.stdout.splitlines()[-30:]))
        return 1

    pdf = os.path.join(out_dir, 'subjective_report.pdf')
    print("report:   %s" % pdf)
    if not a.no_open:
        subprocess.Popen(['xdg-open', pdf], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
