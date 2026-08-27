#!/usr/bin/env python3
"""Nested-counterbalanced participant scheduling for the user study. Writes
ONE master CSV (participant_schedule.csv) -- the single reference document
for the whole study. No live progress-tracking here: progress is tracked
manually in that same file's "Completed" column.

Design: a participant completes every condition of ONE control-mode block
(Clutch or Joystick) before switching to the other; never interleaved.
Three factors are balanced across participants, not just one flat order:
  (a) which block goes first             -- crossed with (b)/(c), not just
                                             alternated on its own
  (b) order within the Clutch block      -- all 3! = 6 orderings
  (c) order within the Joystick block    -- all 3! = 6 orderings
For a 3-item block, the 6 permutations ARE the complete Williams (1949)
odd-n design (2n = 3! = 6 for n=3) -- full first-order carryover balance
within each block. Position and ordering are independent factors: each of
the 6 orderings appears exactly once with its block leading and once with
its block trailing per 12-participant cycle, so no ordering is ever locked
to only one position. Full cycle repeats every 12 participants.

Participant IDs are generated as P01, P02, ... (1-indexed, ID == row number)
-- not typed in, so there's no way for an ID to drift from its row.

No ROS import (same principle as study_config.py): usable with plain python3,
no workspace sourcing required.

Usage:
  python3 study_schedule.py 24      # writes participant_schedule.csv for P01..P24
"""

from __future__ import annotations

import csv
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as sc  # noqa: E402

# The 6 non-tutorial conditions, in VALID_CELLS' own declared order (C/J
# excluded -- tutorial only). Single-sourced from study_config, not
# re-hardcoded, so this can never silently drift from the real cell list.
CONDITIONS = [c for c in sc.VALID_CELLS.values() if c not in ("C", "J")]
_CLUTCH_ITEMS = [c for c in CONDITIONS if c.startswith("C")]
_JOYSTICK_ITEMS = [c for c in CONDITIONS if c.startswith("J")]
_CLUTCH_PERMS = list(itertools.permutations(_CLUTCH_ITEMS))      # 6 orderings
_JOYSTICK_PERMS = list(itertools.permutations(_JOYSTICK_ITEMS))  # 6 orderings


def build_schedule(n_participants: int) -> list[tuple[str, list[str]]]:
    """[(participant_id, [6 conditions in assigned order]), ...], P01-first --
    row 1 of the table is P01, no P00. (The counterbalancing math below still
    indexes from 0 internally; only the displayed ID label is 1-based.)"""
    rows = []
    for i in range(n_participants):
        participant_id = f"P{i+1:02d}"
        r = i % 12  # 12-row block: each ordering appears once leading, once trailing
        perm_idx = r % 6
        clutch_first = r < 6
        c_block = list(_CLUTCH_PERMS[perm_idx])
        j_block = list(_JOYSTICK_PERMS[perm_idx])
        order = (c_block + j_block) if clutch_first else (j_block + c_block)
        rows.append((participant_id, order))
    return rows


def print_schedule(rows: list[tuple[str, list[str]]]):
    print(f"{'Participant':<14}" + "".join(f"Slot {i+1:<9}" for i in range(len(CONDITIONS))))
    for participant_id, order in rows:
        print(f"{participant_id:<14}" + "".join(f"{cond:<14}" for cond in order))


def write_schedule_csv(rows: list[tuple[str, list[str]]], path: str):
    """The persistent master document: one row per participant, their full
    6-slot order, plus a blank Completed column to tick by hand as each
    participant finishes -- this file IS the progress tracker now, no
    separate script re-derives status from recorded bags."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Participant"] + [f"Slot {i+1}" for i in range(len(CONDITIONS))]
                        + ["Completed"])
        for participant_id, order in rows:
            writer.writerow([participant_id] + order + [""])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or not argv[0].isdigit():
        print(__doc__)
        return
    n = int(argv[0])
    rows = build_schedule(n)
    print_schedule(rows)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "participant_schedule.csv")
    write_schedule_csv(rows, csv_path)
    print(f"\nSaved master schedule -> {csv_path}")


if __name__ == "__main__":
    main()
