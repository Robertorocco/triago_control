#!/usr/bin/env python3
"""User-study trial recorder -- a tiny Tkinter GUI wrapper around `ros2 bag record`.

Design (see triago_control/.kiro/context.md §15):
  * Launched FRESH per trial (Gazebo + all nodes are restarted between trials),
    so the node is stateless -- it holds no cross-trial state in memory.
  * It CAPTURES ONLY: it spawns `ros2 bag record` for the curated topic
    allowlist (study_config.BAG_TOPICS) and does nothing else during the trial.
    All metrics / tidy time-series are derived OFFLINE from the bag afterwards.
  * Storage layout (study_config):
        DATA_ROOT/<participant>/<world_shortcut>_<condition_short>/
    The innermost folder is the rosbag output dir; our metadata.json is written
    inside it. Re-recording the same (participant, world, condition) OVERWRITES.

GUI workflow:
  1. Fill Participant (prefilled) and World shortcut. The study cell
     (C/CF/CFB/J/JB/JFB) is AUTO-DETECTED from the controller config
     (CONTROL_MODE / ASSIST_FEEDBACK / ASSIST_BLENDING) and shown read-only.
  2. Press START  -> `ros2 bag record` begins; a timer runs.
  3. Press STOP   -> the bag is closed cleanly (SIGINT).
  4. Mark Success (Yes/No) + optional Notes, press SAVE -> metadata.json written,
     the form resets for the next trial.

This script intentionally does NOT use rclpy: pure bag capture needs only a
subprocess. It is still launched via `ros2 run` so the ROS environment (and
therefore `ros2 bag record`) is sourced for the child process.

Run:  ros2 run triago_control study_recorder.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import signal
import socket
import subprocess
import sys

# --- Make the sibling helper modules importable both in-source (scripts/
# analysis/) and when installed flat into lib/triago_control/. ---------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as sc  # noqa: E402

# cfg is only needed for the (optional) BLENDING sanity check + provenance
# snapshot. Guarded so the GUI still runs if the controller package is not
# importable (e.g. run straight from source without sourcing the workspace).
try:
    import triago_control.qp_controller.config as cfg  # noqa: E402
except Exception:  # pragma: no cover - environment dependent
    cfg = None

# Tkinter is only needed to actually show the GUI. Guard the import so the pure
# helper functions in this module stay importable/testable on a headless box
# without python3-tk. (With `from __future__ import annotations`, the `tk.Tk`
# type hints below are never evaluated at definition time.)
try:
    import tkinter as tk  # noqa: E402
    from tkinter import messagebox, ttk  # noqa: E402
    _TK_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - environment dependent
    tk = ttk = messagebox = None  # type: ignore
    _TK_IMPORT_ERROR = _exc


# How long to wait for `ros2 bag record` to finalise after SIGINT before forcing.
_STOP_GRACE_S = 15.0

# Lowest scheduling priority for the post-save analysis. The sim + QP controller
# are usually STILL RUNNING when SAVE is pressed, and reading a multi-hundred-MB
# bag through matplotlib is heavy: at normal priority it competes with the 150 Hz
# control loop, whose executor starvation is a documented failure mode (bunched
# ticks, lambda_cbf spike -- see .kiro/context.md Sec.10.6). Niced this far down,
# the control loop always wins the CPU.
_ANALYZE_NICE = 19
_ANALYZE_LOG = "analyze.log"


def build_analyze_command(bag_dir: str) -> list[str]:
    """Assemble the niced `analyze_trial.py` argv for one finished trial."""
    return ["nice", "-n", str(_ANALYZE_NICE),
            "ros2", "run", "triago_control", "analyze_trial.py", bag_dir]


def sanitize_token(text: str) -> str:
    """Reduce free text to a filesystem-safe token (alnum, dash, underscore)."""
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in text.strip()]
    token = "".join(keep).strip("_")
    return token


def build_record_command(bag_dir: str) -> list[str]:
    """Assemble the `ros2 bag record` argv for the curated topic allowlist."""
    return [
        "ros2", "bag", "record",
        "-s", sc.BAG_STORAGE_ID,
        "-o", bag_dir,
        *sc.BAG_TOPICS,
    ]


def snapshot_cfg() -> dict:
    """Read the whitelisted cfg values for provenance (missing keys skipped)."""
    if cfg is None:
        return {}
    snap = {}
    for key in sc.CFG_SNAPSHOT_KEYS:
        if hasattr(cfg, key):
            val = getattr(cfg, key)
            try:
                json.dumps(val)          # keep only JSON-serialisable scalars
                snap[key] = val
            except TypeError:
                snap[key] = str(val)
    return snap


class RecorderApp:
    """Tiny state machine GUI: IDLE -> RECORDING -> AWAIT_SAVE -> IDLE."""

    IDLE, RECORDING, AWAIT_SAVE = "idle", "recording", "await_save"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.state = self.IDLE
        self.proc: subprocess.Popen | None = None
        self.bag_dir: str | None = None
        self.t_start: _dt.datetime | None = None
        self.t_stop: _dt.datetime | None = None
        # Post-save analysis runs detached, so the GUI stays live and the next
        # trial can be set up while the previous one is still being plotted.
        self.analyze_proc: subprocess.Popen | None = None
        self.analyze_bag_dir: str | None = None

        root.title("TRIAGo Study Recorder")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.resizable(False, False)

        # Study cell is AUTO-DETECTED from the controller config -- never picked.
        self.cell = sc.derive_cell(cfg)

        self.var_participant = tk.StringVar(value=sc.resolve_participant_id())
        self.var_world = tk.StringVar(value="")
        self.var_success = tk.StringVar(value="")
        self.var_notes = tk.StringVar(value="")
        self.var_status = tk.StringVar(value="Idle - fill the fields and press START.")
        self.var_cell = tk.StringVar(value="")
        self.var_cellnote = tk.StringVar(value="")
        self.var_analyze = tk.BooleanVar(value=True)
        self.var_analyze_status = tk.StringVar(value="")

        self._build_widgets()
        self._refresh_cell_display()
        self._apply_state()

    # ------------------------------------------------------------------ UI
    def _build_widgets(self):
        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self.root, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="Participant ID").grid(row=0, column=0, sticky="w", **pad)
        self.e_participant = ttk.Entry(frm, textvariable=self.var_participant, width=24)
        self.e_participant.grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(frm, text="World shortcut").grid(row=1, column=0, sticky="w", **pad)
        self.e_world = ttk.Entry(frm, textvariable=self.var_world, width=24)
        self.e_world.grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(frm, text="e.g. shield, movement, noobstacle",
                  foreground="#666").grid(row=1, column=2, sticky="w", **pad)

        ttk.Label(frm, text="Study cell (auto)").grid(row=2, column=0, sticky="nw", **pad)
        self.lbl_cell = ttk.Label(frm, textvariable=self.var_cell,
                                  font=("TkDefaultFont", 11, "bold"))
        self.lbl_cell.grid(row=2, column=1, columnspan=2, sticky="w", **pad)

        self.lbl_cfgcheck = ttk.Label(frm, textvariable=self.var_cellnote)
        self.lbl_cfgcheck.grid(row=3, column=0, columnspan=3, sticky="w", **pad)

        ttk.Separator(frm, orient="horizontal").grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=8)

        self.btn_start = ttk.Button(frm, text="\u25CF  START", command=self.on_start)
        self.btn_start.grid(row=5, column=0, **pad)
        self.btn_stop = ttk.Button(frm, text="\u25A0  STOP", command=self.on_stop)
        self.btn_stop.grid(row=5, column=1, **pad)

        self.lbl_status = ttk.Label(frm, textvariable=self.var_status,
                                    font=("TkDefaultFont", 10, "bold"))
        self.lbl_status.grid(row=6, column=0, columnspan=3, sticky="w", **pad)

        ttk.Separator(frm, orient="horizontal").grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=8)

        ttk.Label(frm, text="Success").grid(row=8, column=0, sticky="w", **pad)
        succ_frame = ttk.Frame(frm)
        succ_frame.grid(row=8, column=1, columnspan=2, sticky="w", **pad)
        self.rb_succ_yes = ttk.Radiobutton(succ_frame, text="Yes", value="yes",
                                            variable=self.var_success)
        self.rb_succ_no = ttk.Radiobutton(succ_frame, text="No", value="no",
                                           variable=self.var_success)
        self.rb_succ_yes.pack(side="left")
        self.rb_succ_no.pack(side="left", padx=(12, 0))

        ttk.Label(frm, text="Notes").grid(row=9, column=0, sticky="w", **pad)
        self.e_notes = ttk.Entry(frm, textvariable=self.var_notes, width=40)
        self.e_notes.grid(row=9, column=1, columnspan=2, sticky="w", **pad)

        self.chk_analyze = ttk.Checkbutton(
            frm, text="Analyze after SAVE (background, lowest CPU priority)",
            variable=self.var_analyze)
        self.chk_analyze.grid(row=10, column=0, columnspan=3, sticky="w", **pad)

        self.btn_save = ttk.Button(frm, text="SAVE TRIAL", command=self.on_save)
        self.btn_save.grid(row=11, column=1, sticky="w", **pad)

        self.lbl_analyze = ttk.Label(frm, textvariable=self.var_analyze_status,
                                     foreground="#666")
        self.lbl_analyze.grid(row=12, column=0, columnspan=3, sticky="w", **pad)

    def _refresh_cell_display(self):
        """Show the auto-detected study cell (read-only) + its config flags."""
        c = self.cell
        self.var_cell.set(f"{c.get('code', 'unknown')}   {c.get('label', '')}")
        detail = (f"CONTROL_MODE={c.get('control_mode')}, "
                  f"ASSIST_FEEDBACK={c.get('assist_feedback')}, "
                  f"ASSIST_BLENDING={c.get('assist_blending')}")
        if c.get("valid"):
            self.var_cellnote.set(f"{detail}   \u2713 valid 2x3 cell")
            colour = "#227722"
        else:
            self.var_cellnote.set(f"{detail}   \u26A0 {c.get('warning') or 'non-standard cell'}")
            colour = "#cc0000"
        self.lbl_cell.configure(foreground=colour)
        self.lbl_cfgcheck.configure(foreground=colour)

    def _apply_state(self):
        rec = self.state == self.RECORDING
        awaiting = self.state == self.AWAIT_SAVE
        idle = self.state == self.IDLE
        inputs = "normal" if idle else "disabled"
        for w in (self.e_participant, self.e_world, self.btn_start):
            w.configure(state=inputs)
        self.btn_stop.configure(state="normal" if rec else "disabled")
        save_state = "normal" if awaiting else "disabled"
        for w in (self.rb_succ_yes, self.rb_succ_no, self.e_notes, self.btn_save):
            w.configure(state=save_state)

    # -------------------------------------------------------------- actions
    def on_start(self):
        participant = sanitize_token(self.var_participant.get())
        world = sanitize_token(self.var_world.get())
        cell_code = self.cell.get("code", "unknown")
        if not participant:
            messagebox.showerror("Missing field", "Participant ID is required.")
            return
        if not world:
            messagebox.showerror("Missing field", "World shortcut is required.")
            return

        bag_dir = sc.bag_path(participant, world, cell_code)
        os.makedirs(sc.participant_dir(participant), exist_ok=True)

        # Never delete a folder the background analysis is still reading.
        if (self.analyze_proc is not None and self.analyze_proc.poll() is None
                and self.analyze_bag_dir == bag_dir):
            messagebox.showerror(
                "Analysis in progress",
                f"analyze_trial.py is still reading:\n\n{bag_dir}\n\n"
                "Wait for it to finish before overwriting this trial.")
            return

        if os.path.exists(bag_dir):
            if not messagebox.askyesno(
                    "Overwrite?",
                    f"A recording already exists:\n\n{bag_dir}\n\nOverwrite it?"):
                return
            shutil.rmtree(bag_dir, ignore_errors=True)

        cmd = build_record_command(bag_dir)
        try:
            # New session so we can SIGINT the whole process group cleanly.
            self.proc = subprocess.Popen(cmd, start_new_session=True)
        except FileNotFoundError:
            messagebox.showerror(
                "ros2 not found",
                "`ros2` is not on PATH. Launch this via `ros2 run` in a sourced "
                "ROS 2 environment.")
            return

        self.bag_dir = bag_dir
        self.t_start = _dt.datetime.now()
        self.t_stop = None
        self.var_success.set("")
        self.var_notes.set("")
        self.state = self.RECORDING
        self._apply_state()
        print(f"[study_recorder] recording -> {bag_dir}\n  cmd: {' '.join(cmd)}")
        self._tick()

    def _tick(self):
        if self.state != self.RECORDING:
            return
        # If the recorder died on its own, surface it.
        if self.proc is not None and self.proc.poll() is not None:
            self.var_status.set("ros2 bag record exited unexpectedly - check console.")
            self._finish_recording()
            return
        elapsed = (_dt.datetime.now() - self.t_start).total_seconds()
        mm, ss = divmod(int(elapsed), 60)
        name = os.path.basename(self.bag_dir or "")
        self.var_status.set(f"\u25CF REC  {mm:02d}:{ss:02d}   \u2192  {name}")
        self.root.after(500, self._tick)

    def on_stop(self):
        if self.state != self.RECORDING or self.proc is None:
            return
        self.var_status.set("Stopping - finalising bag ...")
        self.root.update_idletasks()
        self._stop_process()
        self._finish_recording()

    def _stop_process(self):
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=_STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            print("[study_recorder] bag did not stop in time; terminating.")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _finish_recording(self):
        self.t_stop = _dt.datetime.now()
        self.state = self.AWAIT_SAVE
        dur = (self.t_stop - self.t_start).total_seconds() if self.t_start else 0.0
        self.var_status.set(
            f"Stopped ({dur:0.1f}s). Mark Success + Notes, then SAVE TRIAL.")
        self._apply_state()

    def on_save(self):
        if self.state != self.AWAIT_SAVE or not self.bag_dir:
            return
        success = self.var_success.get() or "unknown"
        meta = {
            "participant": sanitize_token(self.var_participant.get()),
            "world_shortcut": sanitize_token(self.var_world.get()),
            # Auto-detected study cell (from the controller config, not declared).
            "cell_code": self.cell.get("code"),
            "condition": self.cell.get("label"),        # human-readable (display)
            "condition_short": self.cell.get("code"),   # compact tag (folder suffix)
            "control_mode": self.cell.get("control_mode"),
            "assist_feedback": self.cell.get("assist_feedback"),
            "assist_blending": self.cell.get("assist_blending"),
            "cell_valid": self.cell.get("valid"),
            "cell_warning": self.cell.get("warning"),
            "bag_name": os.path.basename(self.bag_dir),
            "bag_path": os.path.abspath(self.bag_dir),
            "success": success,
            "notes": self.var_notes.get().strip(),
            "start_time": self.t_start.isoformat() if self.t_start else None,
            "stop_time": self.t_stop.isoformat() if self.t_stop else None,
            "duration_s": round((self.t_stop - self.t_start).total_seconds(), 3)
            if (self.t_start and self.t_stop) else None,
            "storage_id": sc.BAG_STORAGE_ID,
            "topics": list(sc.BAG_TOPICS),
            "cfg_blending": bool(cfg.BLENDING) if (cfg and hasattr(cfg, "BLENDING")) else None,
            "cfg_snapshot": snapshot_cfg(),
            "hostname": socket.gethostname(),
            "recorder_schema": 1,
        }
        meta_path = os.path.join(self.bag_dir, sc.METADATA_NAME)
        try:
            with open(meta_path, "w") as fh:
                json.dump(meta, fh, indent=2)
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not write metadata:\n{exc}")
            return
        print(f"[study_recorder] saved metadata -> {meta_path}")
        self.var_status.set(
            f"Saved '{meta['bag_name']}' (success={success}). Ready for next trial.")
        # Metadata is on disk BEFORE analysis is attempted: a broken/slow analysis
        # must never cost the trial's provenance.
        if self.var_analyze.get():
            self._start_analysis(self.bag_dir)
        # Reset for the next trial (keep participant/world sticky; the cell is
        # auto-detected from config, so nothing to preserve there).
        self.proc = None
        self.bag_dir = None
        self.t_start = self.t_stop = None
        self.var_success.set("")
        self.var_notes.set("")
        self.state = self.IDLE
        self._apply_state()

    # ------------------------------------------------------- post-save analysis
    def _start_analysis(self, bag_dir: str):
        """Spawn analyze_trial.py on the finished bag, detached and niced."""
        if self.analyze_proc is not None and self.analyze_proc.poll() is None:
            self.var_analyze_status.set(
                "Analysis still running on the previous trial - skipped this one. "
                "Run analyze_trial.py by hand later.")
            return
        # rosbag2 writes metadata.yaml only on a clean finalise; without it the bag
        # is truncated and analyze_trial would fail on a half-written file.
        if not os.path.isfile(os.path.join(bag_dir, "metadata.yaml")):
            self.var_analyze_status.set(
                "Analysis skipped: bag has no metadata.yaml (did not finalise cleanly).")
            return
        cmd = build_analyze_command(bag_dir)
        try:
            log = open(os.path.join(bag_dir, _ANALYZE_LOG), "w")
            self.analyze_proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except (OSError, FileNotFoundError) as exc:
            self.var_analyze_status.set(f"Analysis could not start: {exc}")
            return
        self.analyze_bag_dir = bag_dir
        self.var_analyze_status.set(
            f"Analyzing '{os.path.basename(bag_dir)}' in the background ...")
        print(f"[study_recorder] analyzing -> {' '.join(cmd)}")
        self._poll_analysis()

    def _poll_analysis(self):
        if self.analyze_proc is None:
            return
        rc = self.analyze_proc.poll()
        if rc is None:
            self.root.after(1000, self._poll_analysis)
            return
        name = os.path.basename(self.analyze_bag_dir or "")
        if rc == 0:
            self.var_analyze_status.set(
                f"Analysis of '{name}' done - plots + metrics.json written.")
        else:
            self.var_analyze_status.set(
                f"Analysis of '{name}' FAILED (exit {rc}) - see {_ANALYZE_LOG} in the "
                f"trial folder. The bag itself is intact.")
        print(f"[study_recorder] analysis finished (exit {rc}) for {name}")
        self.analyze_proc = None

    def _on_close(self):
        if self.state == self.RECORDING:
            if not messagebox.askyesno(
                    "Recording in progress",
                    "A recording is still running. Stop it and quit?"):
                return
            self._stop_process()
        # The analysis is a detached process: closing the GUI does NOT kill it,
        # it just finishes unobserved (progress still goes to analyze.log).
        if self.analyze_proc is not None and self.analyze_proc.poll() is None:
            if not messagebox.askyesno(
                    "Analysis in progress",
                    f"analyze_trial.py is still running on "
                    f"'{os.path.basename(self.analyze_bag_dir or '')}'.\n\n"
                    "It will keep running in the background after this window "
                    "closes. Quit anyway?"):
                return
        self.root.destroy()


def main():
    if tk is None:
        print("[study_recorder] Tkinter is not available: "
              f"{_TK_IMPORT_ERROR}\nInstall it with:  sudo apt install python3-tk",
              file=sys.stderr)
        sys.exit(1)
    root = tk.Tk()
    RecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
