TRIAGo user study -- MATLAB analysis bundle
===========================================

WHAT THIS IS
    Everything needed to analyse the study on any PC with MATLAB.
    Self-contained: no ROS, no Python, no path editing.

HOW TO USE
    1. Copy this whole folder anywhere on the other PC.
    2. In MATLAB, use the "Browse for folder" button to open this folder
       (so it becomes the Current Folder).
    3. Double-click  analyze_participant.m  and press Run.
       Leave PARTICIPANT = '' at the top and a pick-list dialog appears.

WHAT IS IN HERE
    analyze_participant.m   per-participant performance profile (the main script)
    load_manifest.m         loads manifest.mat as a table of every trial
    load_trial.m            loads one trial's full raw time series (needs mat/, see below)
    load_trial_row.m        same, addressed from a manifest table row
    study_export_dir.m      finds the data; leave it alone
    manifest.mat            every trial, every summary metric  <- the analysis reads this
    manifest.csv            the same table as plain text (Excel, pandas, R)

WHAT IS *NOT* IN HERE
    The per-trial raw time series (mat/ folder, ~29 MB per trial, ~4.5 GB total).
    None of the per-participant or cross-participant analysis needs them -- they
    are only for digging into one trial's signals (EE traces, forces, CBF, ...).
    If you want that too, copy the mat/ folder from the export machine into this
    folder; load_trial will then find it automatically.

KEEPING IT UP TO DATE
    After new participants are recorded, the export regenerates manifest.mat.
    Copy the new manifest.mat over the old one here -- that is the only file
    that changes. Everything else stays as is.
