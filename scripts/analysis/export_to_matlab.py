#!/usr/bin/env python3
"""Export study trials that PASS check_study_data.py's checks into MATLAB .mat files.

Only trials with status OK under check_study_data's pass/fail ladder are
exported -- anything that script would put on its RE-RUN LIST (missing, too
short, not finalised, empty, no robot data) is skipped. Reuses
``check_study_data.inspect_trial`` directly so the inclusion rule can never
drift between the two scripts.

Per included trial (``<out_dir>/mat/<participant>_<world>_<cell>.mat``):
  meta                          -- metadata.json fields (provenance, cfg snapshot)
  metrics_right / metrics_left  -- study_metrics.compute_metrics() per arm
  series.<topic>                -- t (s, column vector) + every column
                                    study_metrics.load_bag() decoded for that topic

Across all included trials (``<out_dir>/manifest.{csv,mat}``): one row per trial,
meta + both arms' metrics flattened -- the entry point for cross-trial statistics
in MATLAB (``S = load('manifest.mat'); T = struct2table(S.trials);``).

NOT included -- a ``study_metrics.load_bag`` limitation, not this script's: it
only decodes Float64/Float64MultiArray/Bool/String/Pose/Twist/Wrench topics, so
/joint_states, /tf, /tf_static never appear in ``series`` (same scope
analyze_trial.py's own dashboards already work from). Raw joint angles / full TF
are not decoded anywhere in this repo yet.

Requires a sourced ROS 2 environment (study_metrics.load_bag needs rosbag2_py +
rclpy) and scipy (already a package dependency, see package.xml).

Run:
  ros2 run triago_control export_to_matlab.py
  python3 scripts/analysis/export_to_matlab.py --dry-run
  python3 scripts/analysis/export_to_matlab.py --participants P07,P08 --skip-existing
  python3 scripts/analysis/export_to_matlab.py --recompute-metrics   # metric change, bags untouched
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as sc  # noqa: E402
import study_metrics as sm  # noqa: E402
from check_study_data import (  # noqa: E402
    DEFAULT_CELLS, DEFAULT_MIN_DURATION_S, DEFAULT_WORLDS, discover_participants,
    inspect_trial,
)

try:
    import scipy.io as sio
    _SCIPY_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    sio = None
    _SCIPY_IMPORT_ERROR = exc

MANIFEST_NAME = "manifest"   # + .csv / .mat


# ------------------------------------------------------------- MATLAB encoding
def matlab_field(name: str, used: set) -> str:
    """Sanitize a topic/column/key name into a unique, valid MATLAB struct field.

    MATLAB field names must start with a letter and hold only [A-Za-z0-9_];
    scipy raises past 31 chars unless long_field_names=True is passed to
    savemat (always is, here) -- 63 is the hard cap even then (namelengthmax).
    """
    s = re.sub(r"^/+", "", name)
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    if not s or not s[0].isalpha():
        s = "f_" + s
    s = s[:63]
    base, i = s, 1
    while s in used:
        suffix = f"_{i}"
        s = base[:63 - len(suffix)] + suffix
        i += 1
    used.add(s)
    return s


def _pyval(v):
    """Coerce one Python value from metadata/metrics JSON into a MATLAB-safe type."""
    if v is None:
        return ""                                     # -> MATLAB '' (empty char)
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return _dict_to_struct(v)
    if isinstance(v, (list, tuple)):
        if all(isinstance(x, str) for x in v):
            return np.array(v, dtype=object)          # -> MATLAB cell array
        if all(isinstance(x, (int, float, bool)) or x is None for x in v):
            return np.array([np.nan if x is None else float(x) for x in v],
                            dtype=float).reshape(-1, 1)
        return np.array([str(x) for x in v], dtype=object)
    return str(v)


def _dict_to_struct(d: dict) -> dict:
    used: set = set()
    return {matlab_field(k, used): _pyval(v) for k, v in d.items()}


def series_to_struct(series: sm.Series) -> dict:
    """One topic's Series -> a MATLAB struct: t (column vector) + every column."""
    used = {"t"}
    out = {"t": np.asarray(series.t, dtype=float).reshape(-1, 1)}
    for name, vals in series.cols.items():
        field = matlab_field(name, used)
        if isinstance(vals, list):                    # text column (String msgs)
            out[field] = np.array(vals, dtype=object)
        else:
            out[field] = np.asarray(vals, dtype=float).reshape(-1, 1)
    return out


def build_trial_mat(bag_dir: str, metadata: dict):
    """Read one bag and assemble its {meta, metrics_right/left, series} struct."""
    series = sm.load_bag(bag_dir)
    used: set = set()
    series_struct = {matlab_field(topic, used): series_to_struct(s)
                     for topic, s in series.items()}
    metrics_right = sm.compute_metrics(series, metadata, "right")
    metrics_left = sm.compute_metrics(series, metadata, "left")
    mat_dict = {
        "meta": _dict_to_struct(metadata),
        "metrics_right": _dict_to_struct(metrics_right),
        "metrics_left": _dict_to_struct(metrics_left),
        "series": series_struct,
    }
    return mat_dict, metrics_right, metrics_left


def _matval(v):
    """A loaded .mat value back to the plain Python type _pyval expects."""
    if hasattr(v, "_fieldnames"):
        return _struct_to_dict(v)
    if isinstance(v, np.ndarray):
        return [_matval(x) for x in v.ravel().tolist()] if v.size != 1 else _matval(v.ravel()[0])
    if isinstance(v, np.generic):
        return v.item()
    return v


def _struct_to_dict(mstruct) -> dict:
    """Flatten a loaded scipy mat_struct (struct_as_record=False) back to a dict."""
    return {name: _matval(getattr(mstruct, name)) for name in mstruct._fieldnames}


def _series_from_mat(series_struct) -> dict:
    """Rebuild {topic: Series} from a saved .mat, inverting the field sanitiser."""
    used: set = set()
    field_to_topic = {matlab_field(t, used): t for t in sc.BAG_TOPICS}
    out = {}
    for field in series_struct._fieldnames:
        topic = field_to_topic.get(field)
        if topic is None:
            continue
        st = getattr(series_struct, field)
        cols = {}
        for name in st._fieldnames:
            if name == "t":
                continue
            v = getattr(st, name)
            arr = np.atleast_1d(v)
            cols[name] = [str(x) for x in arr] if arr.dtype.kind in "OUS" else arr.astype(float).ravel()
        out[topic] = sm.Series(np.atleast_1d(np.asarray(st.t, dtype=float)).ravel(), cols)
    return out


def recompute_trial_mat(mat_path: str, metadata: dict):
    """Re-run compute_metrics on a .mat's stored series; the bag is not read."""
    cached = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    series = _series_from_mat(cached["series"])
    metrics_right = sm.compute_metrics(series, metadata, "right")
    metrics_left = sm.compute_metrics(series, metadata, "left")
    mat_dict = {
        "meta": _dict_to_struct(metadata),
        "metrics_right": _dict_to_struct(metrics_right),
        "metrics_left": _dict_to_struct(metrics_left),
        "series": cached["series"],
    }
    return mat_dict, metrics_right, metrics_left


def manifest_row(r, metadata: dict, metrics_right: dict, metrics_left: dict,
                 mat_relpath: str) -> dict:
    row = {
        "participant": r.participant, "world": r.world, "cell": r.cell,
        "folder": r.folder, "mat_file": mat_relpath,
        "condition": metadata.get("condition"),
        "control_mode": metadata.get("control_mode"),
        "assist_feedback": bool(metadata.get("assist_feedback")),
        "assist_blending": bool(metadata.get("assist_blending")),
        "success": metadata.get("success"),
        "notes": metadata.get("notes"),
        "duration_s": metadata.get("duration_s"),
    }
    for arm, metrics in (("right", metrics_right), ("left", metrics_left)):
        for k, v in metrics.items():
            if k != "arm":
                row[f"{arm}_{k}"] = v
    return row


def _rows_to_struct_array(rows: list[dict]) -> np.ndarray:
    """Build a genuine MATLAB struct ARRAY for savemat.

    A plain Python list of dicts round-trips fine through scipy's OWN loadmat
    (which is how this looked "verified" at first), but scipy actually writes
    it as a cell array of scalar structs, not a struct array -- MATLAB's own
    struct2table then rejects it ("must be a scalar structure, or a structure
    array with one column or one row"), confirmed against real MATLAB. A
    structured (record) ndarray with object-dtype fields is what scipy's mio5
    writer serializes as one real struct array instead.
    """
    converted = [_dict_to_struct(r) for r in rows]
    fields = list(converted[0].keys())
    for c in converted[1:]:                   # struct array needs identical fields
        for f in fields:
            c.setdefault(f, "")
    arr = np.zeros((len(converted),), dtype=[(f, "O") for f in fields])
    for i, c in enumerate(converted):
        for f in fields:
            arr[f][i] = c[f]
    return arr


def write_manifest(out_dir: str, rows: list[dict]) -> None:
    if not rows:
        print("nothing exported -- no manifest written")
        return
    fieldnames = list(rows[0].keys())
    for r in rows[1:]:                                 # union, first-seen order
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    csv_path = os.path.join(out_dir, f"{MANIFEST_NAME}.csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"manifest CSV -> {csv_path}  ({len(rows)} trials)")

    if sio is None:
        print(f"manifest.mat skipped -- scipy unavailable ({_SCIPY_IMPORT_ERROR})")
        return
    mat_path = os.path.join(out_dir, f"{MANIFEST_NAME}.mat")
    sio.savemat(mat_path, {"trials": _rows_to_struct_array(rows)},
               long_field_names=True, oned_as="column", do_compression=True)
    print(f"manifest MAT -> {mat_path}  "
          f"(MATLAB: S = load('{MANIFEST_NAME}.mat'); T = struct2table(S.trials);)")


# ------------------------------------------------------------------------ main
def _check_scipy() -> bool:
    if sio is not None:
        return True
    print(f"scipy.io is not importable: {_SCIPY_IMPORT_ERROR}", file=sys.stderr)
    print("If this is a NumPy 1.x/2.x ABI mismatch (a user-site pip numpy "
          "shadowing the apt one scipy was built against -- check with "
          "`python3 -c \"import numpy; print(numpy.__file__)\"`), either run "
          "this script with `python3 -s ...` to skip user site-packages, or "
          "`pip uninstall numpy` from the user site: ROS's own numpy already "
          "satisfies every dependency here.", file=sys.stderr)
    return False


class _MatTrial:
    """The (participant, world, cell) identity of a trial, parsed from its .mat name."""
    def __init__(self, mat_name: str):
        m = re.match(r"(P\d+)_([a-z]+)_([A-Z]+)\.mat$", mat_name)
        if not m:
            raise ValueError(f"not a trial .mat name: {mat_name}")
        self.participant, self.world, self.cell = m.groups()
        self.folder = f"{self.world}_{self.cell}"


def recompute_from_mats(out_dir: str) -> int:
    """Re-run every metric from the .mat files alone, for a machine without the bags.

    metadata comes back out of each .mat's own `meta` struct; the manifest is
    rewritten from the result exactly as the bag-driven path writes it.
    """
    mat_dir = os.path.join(out_dir, "mat")
    names = sorted(n for n in os.listdir(mat_dir) if n.endswith(".mat"))
    rows = []
    for i, name in enumerate(names, 1):
        r = _MatTrial(name)
        mat_path = os.path.join(mat_dir, name)
        print(f"[{i}/{len(names)}] {r.participant}/{r.folder} -- recomputing metrics...", flush=True)
        cached = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        metadata = {**_struct_to_dict(cached["meta"]), "cell": r.cell}
        mat_dict, metrics_right, metrics_left = recompute_trial_mat(mat_path, metadata)
        sio.savemat(mat_path, mat_dict, long_field_names=True, oned_as="column", do_compression=True)
        rows.append(manifest_row(r, metadata, metrics_right, metrics_left, os.path.join("mat", name)))
    write_manifest(out_dir, rows)
    print(f"\n{len(rows)} trial(s) recomputed in {mat_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=sc.DATA_ROOT)
    ap.add_argument("--out-dir", default=None, help="default: <data-root>/matlab_export")
    ap.add_argument("--participants", help="comma list, default: every Pxx folder")
    ap.add_argument("--worlds", default=",".join(DEFAULT_WORLDS))
    ap.add_argument("--cells", default=",".join(DEFAULT_CELLS))
    ap.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_S,
                    help="must match the check_study_data.py run this is gated on")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse a .mat already newer than its bag's metadata.yaml "
                         "(manifest row is read back from it, bag is not re-read)")
    ap.add_argument("--recompute-metrics", action="store_true",
                    help="after a study_metrics.py change: recompute every metric from the "
                         "series already stored in each .mat (minutes, not an hour of bag decoding)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only print what would be included/excluded, export nothing")
    ap.add_argument("--from-mats", action="store_true",
                    help="recompute metrics from an existing matlab_export/ alone (no bags, "
                         "no data root): every mat/*.mat is re-scored and the manifest rewritten")
    args = ap.parse_args()

    if args.from_mats:
        out_dir = args.out_dir or os.path.join(args.data_root, "matlab_export")
        if not os.path.isdir(os.path.join(out_dir, "mat")):
            print(f"no mat/ folder under {out_dir}", file=sys.stderr)
            return 2
        if not _check_scipy():
            return 2
        return recompute_from_mats(out_dir)

    if not os.path.isdir(args.data_root):
        print(f"data root not found: {args.data_root}", file=sys.stderr)
        return 2
    out_dir = args.out_dir or os.path.join(args.data_root, "matlab_export")
    mat_dir = os.path.join(out_dir, "mat")

    worlds = [w.strip() for w in args.worlds.split(",") if w.strip()]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    expected = [(w, c) for w in worlds for c in cells]

    participants = ([p.strip() for p in args.participants.split(",") if p.strip()]
                    if args.participants else discover_participants(args.data_root))
    if not participants:
        print(f"no participant (P<number>) folders under {args.data_root}", file=sys.stderr)
        return 2

    if not args.dry_run and not _check_scipy():
        return 2

    included, excluded = [], []
    for p in participants:
        for w, c in expected:
            r = inspect_trial(args.data_root, p, w, c, args.min_duration)
            (included if r.ok else excluded).append(r)

    print(f"{len(included)} trial(s) pass check_study_data's criteria, "
          f"{len(excluded)} excluded (RE-RUN LIST):")
    for r in excluded:
        print(f"  EXCLUDE  {r.participant}/{r.folder:<12} {r.status:<14} {r.reason}")
    print()

    if args.dry_run:
        print(f"[dry-run] would export {len(included)} trial(s) to {mat_dir}")
        return 0

    os.makedirs(mat_dir, exist_ok=True)
    rows = []
    for i, r in enumerate(included, 1):
        mat_name = f"{r.participant}_{r.folder}.mat"
        mat_path = os.path.join(mat_dir, mat_name)
        bag_meta_mtime = os.path.getmtime(os.path.join(r.path, "metadata.yaml"))
        fresh = os.path.isfile(mat_path) and os.path.getmtime(mat_path) >= bag_meta_mtime
        recompute = args.recompute_metrics and fresh
        reuse = args.skip_existing and fresh and not recompute
        how = '(reusing existing .mat)' if reuse else '-- recomputing metrics...' if recompute else '-- reading bag...'
        print(f"[{i}/{len(included)}] {r.participant}/{r.folder} {how}", flush=True)
        # The folder name is the ground truth for the cell; metadata.json is free-typed.
        metadata = {**sm.load_metadata(r.path), "cell": r.cell}
        if reuse:
            cached = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
            metrics_right = _struct_to_dict(cached["metrics_right"])
            metrics_left = _struct_to_dict(cached["metrics_left"])
        else:
            try:
                if recompute:
                    mat_dict, metrics_right, metrics_left = recompute_trial_mat(mat_path, metadata)
                else:
                    mat_dict, metrics_right, metrics_left = build_trial_mat(r.path, metadata)
            except Exception as exc:                   # noqa: BLE001
                print(f"  FAILED to export {r.participant}/{r.folder}: {exc}", file=sys.stderr)
                continue
            sio.savemat(mat_path, mat_dict, long_field_names=True,
                       oned_as="column", do_compression=True)
        rows.append(manifest_row(r, metadata, metrics_right, metrics_left,
                                 os.path.join("mat", mat_name)))

    write_manifest(out_dir, rows)
    print(f"\n{len(rows)}/{len(included)} trial(s) exported to {mat_dir}")
    return 0 if len(rows) == len(included) else 1


if __name__ == "__main__":
    sys.exit(main())
