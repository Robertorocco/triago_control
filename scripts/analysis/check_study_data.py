#!/usr/bin/env python3
"""Fast integrity check over the user-study data tree (see .kiro/context.md §15).

Answers one question per participant: is every required (world, condition) trial
present AND long enough to hold a completed task? Prints what is missing and why,
so an incomplete session gets re-run before any analysis starts.

A trial folder is ``DATA_ROOT/<participant>/<world>_<cell>/``. It is GOOD when:
  * the rosbag exists              -- ``<name>_0.db3`` / ``.mcap``
  * it finalised cleanly           -- ``metadata.yaml`` (rosbag2 writes it only
                                      on a clean SIGINT stop)
  * it holds messages              -- total ``message_count`` > 0
  * the sim was actually running   -- ``/joint_states`` and ``/qp_debug/ee_real``
                                      carry messages
  * it is long enough              -- bag duration >= ``--min-duration``
                                      (default 1.45 min = 87 s, the task minimum)

Our ``metadata.json`` sidecar missing / mislabelled / marking ``success=no`` is a
WARNING, not a failure: the bag is still usable, but success, notes and the cfg
snapshot are lost or suspect.

No ROS dependency: reads only the rosbag2 ``metadata.yaml`` and our
``metadata.json``. Works for any number of participants and any expected grid.

Exit code: 0 = every participant complete, 1 = at least one incomplete,
2 = usage / environment error.

Run:
  python3 scripts/analysis/check_study_data.py
  python3 scripts/analysis/check_study_data.py --min-duration 105 --csv report.csv
  python3 scripts/analysis/check_study_data.py --participants P07,P08 --quiet
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

# Sibling helper import works both in-source and installed flat.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as sc  # noqa: E402

try:
    import yaml  # not a ROS dep; used only to read the rosbag2 metadata.yaml
except Exception:  # pragma: no cover - fall back to regex parsing
    yaml = None


DEFAULT_WORLDS = ["rack", "shield"]
DEFAULT_CELLS = ["CF", "CB", "CFB", "JF", "JB", "JFB"]      # C / J tutorials excluded
DEFAULT_MIN_DURATION_S = 1.45 * 60.0                        # task-completion minimum
DURATION_MISMATCH_TOL_S = 20.0                              # GUI timer vs bag length
REQUIRED_TOPICS = ("/joint_states", "/qp_debug/ee_real")    # proof the sim ran

OK = "OK"
FAIL_MISSING = "MISSING"
FAIL_NO_BAG = "NO_BAG"
FAIL_NOT_FINALIZED = "NOT_FINALIZED"
FAIL_EMPTY_BAG = "EMPTY_BAG"
FAIL_NO_ROBOT_DATA = "NO_ROBOT_DATA"
FAIL_TOO_SHORT = "TOO_SHORT"

REASON_TEXT = {
    FAIL_MISSING: "no trial folder -- this condition was never recorded",
    FAIL_NO_BAG: "folder exists but no *.db3/*.mcap -- `ros2 bag record` never started",
    FAIL_NOT_FINALIZED: "no metadata.yaml -- recording killed before clean finalise (bag truncated)",
    FAIL_EMPTY_BAG: "bag holds zero messages -- nothing was captured",
    FAIL_NO_ROBOT_DATA: "no /joint_states or /qp_debug/ee_real messages -- sim/controller not running during capture",
    FAIL_TOO_SHORT: "bag shorter than the task minimum -- trial aborted or stopped early",
}

_SHORT_TAG = {
    OK: "ok",
    FAIL_MISSING: "--",
    FAIL_NO_BAG: "NOBAG",
    FAIL_NOT_FINALIZED: "NOFIN",
    FAIL_EMPTY_BAG: "EMPTY",
    FAIL_NO_ROBOT_DATA: "NODATA",
    FAIL_TOO_SHORT: "SHORT",
}


class TrialResult:
    """Outcome of inspecting one expected (participant, world, cell) trial."""

    def __init__(self, participant: str, world: str, cell: str):
        self.participant = participant
        self.world = world
        self.cell = cell
        self.folder = f"{world}_{cell}"
        self.path: str | None = None
        self.status = OK
        self.warnings: list[str] = []
        self.bag_duration_s: float | None = None
        self.json_duration_s: float | None = None
        self.message_count: int | None = None
        self.success: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == OK

    @property
    def reason(self) -> str:
        return REASON_TEXT.get(self.status, "")

    def to_row(self) -> dict:
        return {
            "participant": self.participant,
            "world": self.world,
            "cell": self.cell,
            "folder": self.folder,
            "status": self.status,
            "ok": int(self.ok),
            "bag_duration_s": round(self.bag_duration_s, 1) if self.bag_duration_s is not None else "",
            "json_duration_s": round(self.json_duration_s, 1) if self.json_duration_s is not None else "",
            "message_count": self.message_count if self.message_count is not None else "",
            "success": self.success or "",
            "reason": self.reason,
            "warnings": "; ".join(self.warnings),
            "path": self.path or "",
        }


def parse_bag_metadata(path: str) -> dict | None:
    """Return {duration_s, message_count, topic_counts} from a rosbag2 metadata.yaml."""
    if yaml is not None:
        try:
            info = yaml.safe_load(open(path))["rosbag2_bagfile_information"]
            counts = {t["topic_metadata"]["name"]: int(t["message_count"])
                      for t in info.get("topics_with_message_count", [])}
            return {"duration_s": int(info["duration"]["nanoseconds"]) / 1e9,
                    "message_count": int(info["message_count"]),
                    "topic_counts": counts}
        except Exception:
            pass
    # Fallback: pull only the two top-level scalars; per-topic counts unavailable.
    try:
        txt = open(path).read()
    except OSError:
        return None
    m_dur = re.search(r"duration:\s*\n\s*nanoseconds:\s*(\d+)", txt)
    m_cnt = re.search(r"^  message_count:\s*(\d+)\s*$", txt, re.M)
    if not (m_dur and m_cnt):
        return None
    return {"duration_s": int(m_dur.group(1)) / 1e9,
            "message_count": int(m_cnt.group(1)),
            "topic_counts": {}}


def discover_participants(data_root: str) -> list[str]:
    """Every ``P<number>`` folder directly under data_root, sorted numerically."""
    return sorted(
        (n for n in os.listdir(data_root)
         if re.match(r"^P\d+$", n) and os.path.isdir(os.path.join(data_root, n))),
        key=lambda s: int(s[1:]))


def inspect_trial(data_root: str, participant: str, world: str, cell: str,
                  min_duration: float) -> TrialResult:
    """Run the check ladder for one expected trial, hardest failure wins."""
    r = TrialResult(participant, world, cell)
    d = os.path.join(data_root, participant, r.folder)
    r.path = d

    if not os.path.isdir(d):
        r.status = FAIL_MISSING
        return r

    bags = sorted(glob.glob(os.path.join(d, "*.db3")) + glob.glob(os.path.join(d, "*.mcap")))
    if not bags:
        r.status = FAIL_NO_BAG
        return r

    yml = os.path.join(d, "metadata.yaml")
    if not os.path.isfile(yml):
        r.status = FAIL_NOT_FINALIZED
        return r

    info = parse_bag_metadata(yml)
    if info is None:
        r.status = FAIL_NOT_FINALIZED
        r.warnings.append("metadata.yaml present but unparseable")
        return r

    r.bag_duration_s = info["duration_s"]
    r.message_count = info["message_count"]
    topic_counts = info["topic_counts"]

    if not r.message_count:
        r.status = FAIL_EMPTY_BAG
        return r
    if topic_counts and any(topic_counts.get(t, 0) == 0 for t in REQUIRED_TOPICS):
        r.status = FAIL_NO_ROBOT_DATA
        return r

    _attach_provenance(d, r)

    if r.bag_duration_s < min_duration:
        r.status = FAIL_TOO_SHORT
    return r


def _attach_provenance(trial_dir: str, r: TrialResult) -> None:
    """Read metadata.json for success/notes and cross-check its labels (soft)."""
    mj = os.path.join(trial_dir, sc.METADATA_NAME)
    if not os.path.isfile(mj):
        r.warnings.append("no metadata.json -- success/notes/cfg snapshot lost (SAVE not pressed?)")
        return
    try:
        meta = json.load(open(mj))
    except Exception:
        r.warnings.append("metadata.json unreadable")
        return

    r.success = meta.get("success")
    r.json_duration_s = meta.get("duration_s")

    mp, mw, mc = meta.get("participant"), meta.get("world_shortcut"), meta.get("cell_code")
    if (mp and mp != r.participant) or (mw and mw != r.world) or (mc and mc != r.cell):
        r.warnings.append(
            f"MISLABELLED: metadata.json says {mp}/{mw}/{mc} but folder is "
            f"{r.participant}/{r.world}/{r.cell}")
    if (r.json_duration_s and r.bag_duration_s
            and abs(r.json_duration_s - r.bag_duration_s) > DURATION_MISMATCH_TOL_S):
        r.warnings.append(
            f"GUI timer {r.json_duration_s:.0f}s vs bag {r.bag_duration_s:.0f}s disagree")
    if r.success and str(r.success).lower() not in ("yes", "true", "1"):
        r.warnings.append(f"experimenter marked success={r.success}")


# --------------------------------------------------------------------- reporting
def _cell_tag(r: TrialResult) -> str:
    if r.status == FAIL_TOO_SHORT and r.bag_duration_s is not None:
        tag = f"{r.bag_duration_s:.0f}s"
    else:
        tag = _SHORT_TAG[r.status]
    if r.ok and r.warnings:
        tag += "*"
    return tag


def print_matrix(participants, expected, results) -> None:
    w = 8
    header = " " * 6 + "".join(f"{wl[0].upper()}·{cl}".ljust(w) for wl, cl in expected)
    print("trial grid  (row = participant, col = world·cell)")
    print(header)
    for p in participants:
        print(f"{p:<6}" + "".join(_cell_tag(r).ljust(w) for r in results[p]))
    print("legend: ok good | -- missing | <n>s too short | NOFIN not finalised | "
          "NOBAG no bag | EMPTY no msgs | NODATA sim off | * warning\n")


def print_details(participants, results, quiet: bool) -> int:
    """Print a block per participant; return the count of incomplete participants."""
    n_incomplete = 0
    for p in participants:
        rs = results[p]
        bad = [r for r in rs if not r.ok]
        warn_only = [r for r in rs if r.ok and r.warnings]
        good = len(rs) - len(bad)
        if bad:
            n_incomplete += 1
        if quiet and not bad and not warn_only:
            continue
        verdict = "COMPLETE  " if not bad else "INCOMPLETE"
        summary = f"{good}/{len(rs)} good"
        if bad:
            kinds = {}
            for r in bad:
                kinds[r.status] = kinds.get(r.status, 0) + 1
            summary += ", " + ", ".join(f"{n} {k.lower()}" for k, n in sorted(kinds.items()))
        print(f"{p}  {verdict}  {summary}")
        for r in bad:
            print(f"     {r.folder:<12} {r.status:<14} {r.reason}")
            for wmsg in r.warnings:
                print(f"     {'':<12} {'':<14} WARN: {wmsg}")
        for r in warn_only:
            for wmsg in r.warnings:
                print(f"     {r.folder:<12} {'OK':<14} WARN: {wmsg}")
    print()
    return n_incomplete


def print_summary(participants, expected, results, min_duration) -> None:
    complete, incomplete = [], []
    n_expected = n_good = 0
    tally = {}
    rerun = {}
    for p in participants:
        rs = results[p]
        n_expected += len(rs)
        bad = [r for r in rs if not r.ok]
        n_good += len(rs) - len(bad)
        (incomplete if bad else complete).append(p)
        for r in bad:
            tally[r.status] = tally.get(r.status, 0) + 1
            rerun.setdefault(p, []).append(
                r.folder if r.status != FAIL_TOO_SHORT
                else f"{r.folder}({r.bag_duration_s:.0f}s)")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"participants checked : {len(participants)}")
    print(f"complete             : {len(complete)}  {complete}")
    print(f"incomplete           : {len(incomplete)}  {incomplete}")
    print(f"trials expected      : {n_expected}  ({len(participants)} x {len(expected)})")
    print(f"trials good          : {n_good}")
    for k in (FAIL_MISSING, FAIL_TOO_SHORT, FAIL_NOT_FINALIZED, FAIL_NO_BAG,
              FAIL_EMPTY_BAG, FAIL_NO_ROBOT_DATA):
        if tally.get(k):
            print(f"trials {k.lower():<14}: {tally[k]}")
    print(f"min-duration used    : {min_duration:.0f} s")
    if rerun:
        print("\nRE-RUN LIST:")
        for p in sorted(rerun):
            print(f"  {p}: {' '.join(sorted(rerun[p]))}")


def write_csv(path: str, participants, results) -> None:
    rows = [r.to_row() for p in participants for r in results[p]]
    with open(path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nper-trial CSV -> {path}")


def write_json(path: str, data_root, participants, expected, results, min_duration) -> None:
    payload = {
        "data_root": data_root,
        "min_duration_s": min_duration,
        "expected_grid": [f"{w}_{c}" for w, c in expected],
        "participants": {
            p: {"complete": all(r.ok for r in results[p]),
                "trials": [r.to_row() for r in results[p]]}
            for p in participants},
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"per-trial JSON -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=sc.DATA_ROOT,
                    help=f"study data root (default: {sc.DATA_ROOT})")
    ap.add_argument("--participants",
                    help="comma list e.g. P00,P07 (default: every P<number> folder)")
    ap.add_argument("--worlds", default=",".join(DEFAULT_WORLDS),
                    help=f"expected worlds (default: {','.join(DEFAULT_WORLDS)})")
    ap.add_argument("--cells", default=",".join(DEFAULT_CELLS),
                    help=f"expected study cells (default: {','.join(DEFAULT_CELLS)})")
    ap.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_S,
                    help="minimum acceptable bag duration in seconds "
                         f"(default: {DEFAULT_MIN_DURATION_S:.0f} = 1.45 min)")
    ap.add_argument("--csv", metavar="PATH", help="write a per-trial CSV report")
    ap.add_argument("--json", metavar="PATH", help="write a per-trial JSON report")
    ap.add_argument("--quiet", action="store_true",
                    help="hide participants with no problems")
    ap.add_argument("--no-matrix", action="store_true", help="skip the grid view")
    args = ap.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"data root not found: {args.data_root}", file=sys.stderr)
        return 2

    worlds = [w.strip() for w in args.worlds.split(",") if w.strip()]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    expected = [(w, c) for w in worlds for c in cells]

    if args.participants:
        participants = [p.strip() for p in args.participants.split(",") if p.strip()]
    else:
        participants = discover_participants(args.data_root)
    if not participants:
        print(f"no participant (P<number>) folders under {args.data_root}", file=sys.stderr)
        return 2

    results = {p: [inspect_trial(args.data_root, p, w, c, args.min_duration)
                   for w, c in expected]
               for p in participants}

    print(f"\nchecking {len(participants)} participant(s) against "
          f"{len(expected)} expected trials each, min-duration "
          f"{args.min_duration:.0f}s\n")
    if not args.no_matrix:
        print_matrix(participants, expected, results)
    n_incomplete = print_details(participants, results, args.quiet)
    print_summary(participants, expected, results, args.min_duration)

    if args.csv:
        write_csv(args.csv, participants, results)
    if args.json:
        write_json(args.json, args.data_root, participants, expected, results, args.min_duration)

    return 1 if n_incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
