TRIAGo user study -- MATLAB analysis bundle
===========================================

WHAT THIS IS
    Everything needed to analyse the study on any PC with MATLAB and the
    Statistics and Machine Learning Toolbox. No ROS, no Python, no path editing.

HOW TO USE  (group analysis -- the study results)
    1. Copy this whole folder anywhere on the other PC.
    2. In MATLAB, use the "Browse for folder" button to open this folder
       (so it becomes the Current Folder).
    3. Run  run_study_analysis  (type it in the Command Window, or open the
       file and press Run). It prints the verdict per question and writes a
       time-stamped folder analysis_results/study_results_<date>_n<N>/ with
       results.mat, results_all.csv, trial_table.csv and figures/*.png.
       It also writes summary/ inside that folder: summary.pdf (7 pages, one
       figure each: scorecard, forest plots, cell matrix, checks),
       summary.txt (plain-language verdict) and summary_table.csv. This is
       the short, publication-style output; study_summary regenerates it.
    4. Run  build_paper_figures  for the SHORT read: paper-style bar charts
       (one quantity per panel, six conditions, error bars, p-value brackets)
       with every figure explained in plain words. Writes
       study_paper_figures.pdf / .html / .mlx into the results folder. Open
       the PDF -- that is the whole document, nothing else has to be run.
    5. Run  build_report  for the LONG read: study_report.mlx (executed, with
       every figure) plus .html and .pdf in that results folder. It explains
       every metric, every test and every figure of the full analysis.

HOW TO USE  (one participant)
    Open analyze_participant.m and press Run; leave PARTICIPANT = '' and a
    pick-list dialog appears.

WHAT IS IN HERE
    run_study_analysis.m     group statistics for Q1-Q6 (main script)
    study_summary.m          short publication-style output (summary/ folder)
    fig_summary_*.m          scorecard, forest plots, cell matrix, checks
    study_paper_figures.m    the short visual summary (source); build_paper_figures makes the .mlx
    build_paper_figures.m    .m -> .mlx -> html/pdf  (short read)
    fig_paper_panels.m       paper-style panels: six cells, SEM, p-value brackets
    paper_stats_tables.m     the effects / significant-pairs tables behind them
    study_report.m           the report source (text + code); build_report makes the .mlx
    build_report.m           .m -> .mlx -> html/pdf  (long read)
    load_study_table.m       tidy per-trial table, completeness rule, trial order
    study_metric_spec.m      every metric: label, unit, direction, family, scope
    study_families.m         the metric families
    composite_score.m        z-scores, family scores, composite
    stat_paired2.m           two conditions (t-test + Wilcoxon + bootstrap CI)
    stat_rm_oneway.m         three conditions (RM-ANOVA + Friedman + Holm pairs)
    stat_rm_twoway.m         mode x assistance cells (2-way RM-ANOVA + Friedman)
    stat_consistency.m       between-participant spread (Pitman-Morgan, Kendall W)
    stat_learning.m          learning / order effects along the schedule
    participant_means.m, holm_adjust.m, rank_biserial.m, kendalls_w.m,
    effect_band.m, fmt_p.m, lillie_p.m          small helpers
    studyplot.m, fig_*.m     figures
    analyze_participant.m    per-participant performance profile
    load_manifest.m          loads manifest.mat as a table of every trial
    load_trial.m             loads one trial's full raw time series (needs mat/, see below)
    load_trial_row.m         same, addressed from a manifest table row
    study_export_dir.m       finds the data; leave it alone
    manifest.mat             every trial, every summary metric  <- the analysis reads this
    manifest.csv             the same table as plain text (Excel, pandas, R)
    participant_schedule.csv the order in which every participant met the conditions

WHAT IS *NOT* IN HERE
    The per-trial raw time series (mat/ folder, ~29 MB per trial, ~4.5 GB total).
    None of the per-participant or cross-participant analysis needs them -- they
    are only for digging into one trial's signals (EE traces, forces, CBF, ...).
    If you want that too, copy the mat/ folder from the export machine into this
    folder; load_trial will then find it automatically.

KEEPING IT UP TO DATE
    After new participants are recorded, the export regenerates manifest.mat.
    Copy the new manifest.mat over the old one here (and the schedule if it
    changed), then re-run run_study_analysis and build_report. Only complete
    participants (all 12 trials, present in the schedule) are analysed; the
    others are listed with the reason. Every run keeps its own results folder.
