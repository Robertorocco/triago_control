#!/usr/bin/env python3
"""Export one offline_plotter trial bag (or any sqlite3 rosbag2 of the QP stack) to a MATLAB .mat.

Reads the bag through sqlite3 + rclpy.serialization directly, so it needs a sourced
ROS 2 environment for the message classes but NOT rosbag2_py. Every decodable topic
is stored as series.<topic>.{t, data} (JointState: names, position, velocity), and
the event-ordered derived signals offline_plotter.py computes live (reference-governor
raw/governed magnitudes, commanded/executed gripper path, tracking errors) are
recomputed here from the same field layouts and stored under derived.*. Time is
trial-relative: t0 is the record-trigger rising edge, or -- when that message was
missed by the bag recorder -- the falling edge minus the t_off logged in the trial's
trial_summary.txt, or the first message. The figures are drawn by
scripts/analysis/matlab/fig_thesis_hw.m.

    python3 -s export_offline_bag.py <bag_dir> [--out file.mat] [--t-off SECONDS]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_to_matlab import matlab_field, _dict_to_struct  # noqa: E402

TRIGGER_TOPIC = "/offline_plotter/record_trigger"
REF_TOPICS = {"right": "/arm_right/cartesian_reference", "left": "/arm_left/cartesian_reference"}
EE_TOPIC = "/qp_debug/ee_real"
GOV_TOPIC = "/qp_debug/governor"
ANG_VEL_EMA_ALPHA = 0.85   # same smoothing as offline_plotter.py's angular-velocity estimate


# ------------------------------------------------------------------ SO(3)
def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def log3(R):
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-6:
        return 0.5 * skew
    return skew * (theta / (2 * np.sin(theta)))


def geodesic_angle(rpy, R_real):
    if rpy is None or R_real is None:
        return 0.0
    R_err = rpy_to_matrix(*[float(v) for v in rpy]) @ np.asarray(R_real).T
    if np.trace(R_err) <= -1.0 + 1e-6:
        return float(np.pi)
    return float(np.linalg.norm(log3(R_err)))


# ------------------------------------------------------------------ bag reading
def _extract(msg, base):
    if base in ("Float64", "Float32", "Int32"):
        return np.array([float(msg.data)])
    if base == "Bool":
        return np.array([1.0 if msg.data else 0.0])
    if base in ("Float64MultiArray", "Float32MultiArray"):
        return np.asarray(msg.data, dtype=float)
    if base == "Pose":
        p, o = msg.position, msg.orientation
        return np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w])
    if base == "Twist":
        return np.array([msg.linear.x, msg.linear.y, msg.linear.z,
                         msg.angular.x, msg.angular.y, msg.angular.z])
    if base == "Wrench":
        return np.array([msg.force.x, msg.force.y, msg.force.z,
                         msg.torque.x, msg.torque.y, msg.torque.z])
    return None


def read_bag(bag_dir: str):
    """-> (events sorted by time [(t_ns, topic, payload)], {topic: type}), payload = ndarray|str|JointState."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    db3 = sorted(glob.glob(os.path.join(bag_dir, "*.db3")))
    if not db3:
        raise SystemExit(f"no *.db3 in {bag_dir} (mcap bags need rosbag2_py; not supported here)")
    events, types = [], {}
    for path in db3:
        con = sqlite3.connect(path)
        cur = con.cursor()
        cur.execute("SELECT id, name, type FROM topics")
        for tid, name, typ in cur.fetchall():
            types[name] = typ
            base = typ.split("/")[-1]
            try:
                cls = get_message(typ)
            except Exception:
                continue
            cur2 = con.cursor()
            cur2.execute("SELECT timestamp, data FROM messages WHERE topic_id=?", (tid,))
            for t_ns, data in cur2:
                try:
                    msg = deserialize_message(bytes(data), cls)
                except Exception:
                    break
                if base == "JointState":
                    payload = msg
                elif base == "String":
                    payload = str(msg.data)
                else:
                    payload = _extract(msg, base)
                    if payload is None:
                        break
                events.append((t_ns, name, payload))
        con.close()
    events.sort(key=lambda e: e[0])
    return events, types


# ------------------------------------------------------------------ t0 / t_off
def parse_summary_t_off(bag_dir: str):
    """t_off from the trial's own trial_summary.txt ('trajectory-finished marker at X')."""
    for cand in (os.path.join(bag_dir, "trial_summary.txt"),
                 os.path.join(os.path.dirname(os.path.abspath(bag_dir)), "trial_summary.txt")):
        if os.path.exists(cand):
            m = re.search(r"trajectory-finished marker at ([0-9.]+)", open(cand).read())
            if m:
                return float(m.group(1)), cand
    return None, None


def resolve_t0(events, bag_dir, t_off_override):
    rising = [t for t, top, v in events if top == TRIGGER_TOPIC and v[0] > 0.5]
    falling = [t for t, top, v in events if top == TRIGGER_TOPIC and v[0] < 0.5]
    if rising:
        t0 = rising[0]
        t_off = (falling[0] - t0) * 1e-9 if falling else None
        return t0, t_off, "trigger rising edge"
    if falling:
        t_off, src = (t_off_override, "--t-off") if t_off_override is not None else parse_summary_t_off(bag_dir)
        if t_off is not None:
            return falling[0] - int(round(t_off * 1e9)), t_off, f"falling edge minus t_off from {src}"
    return events[0][0], t_off_override, "first message"


# ------------------------------------------------------------------ derived replay
class Replay:
    """offline_plotter.py's cb_real/gov_callback derivations, replayed event-ordered."""

    def __init__(self, t0):
        self.t0 = t0
        self.ref = {"right": None, "left": None}
        self.real_pos = {"right": None, "left": None}
        self.real_rpy = {"right": None, "left": None}
        self.prev_R = {"right": None, "left": None}
        self.w_ema = {"right": None, "left": None}
        self.prev_ang_t = None
        self.traj = {k: [] for k in ("t", "ref_pos_r", "real_pos_r", "ref_pos_l", "real_pos_l",
                                     "ref_rpy_r", "real_rpy_r", "ref_rpy_l", "real_rpy_l",
                                     "e_pos_r", "e_pos_l", "e_vel_r", "e_vel_l",
                                     "e_ang_r", "e_ang_l", "e_angvel_r", "e_angvel_l")}
        self.gov = {k: [] for k in ("t",
                                    "lin_vel_raw_r", "lin_vel_gov_r", "lin_vel_raw_l", "lin_vel_gov_l",
                                    "ang_vel_raw_r", "ang_vel_gov_r", "ang_vel_raw_l", "ang_vel_gov_l",
                                    "pos_err_raw_r", "pos_err_gov_r", "pos_err_raw_l", "pos_err_gov_l",
                                    "ori_err_raw_r", "ori_err_gov_r", "ori_err_raw_l", "ori_err_gov_l")}

    def _t(self, t_ns):
        return (t_ns - self.t0) * 1e-9

    def on_ref(self, side, data):
        if len(data) >= 12:
            self.ref[side] = np.asarray(data[:12], dtype=float)

    def on_ee(self, t_ns, real):
        if self.ref["right"] is None:
            return
        p = {"right": real[0:3], "left": real[6:9]}
        v = {"right": real[3:6], "left": real[9:12]}
        has_rpy = len(real) >= 18
        self.real_pos = {s: p[s].copy() for s in p}
        if has_rpy:
            self.real_rpy = {"right": real[12:15].copy(), "left": real[15:18].copy()}
        t_now = self._t(t_ns)
        R = {s: (rpy_to_matrix(*self.real_rpy[s]) if has_rpy else None) for s in p}
        w_est = {s: (self.w_ema[s] if self.w_ema[s] is not None else np.zeros(3)) for s in p}
        if has_rpy:
            if self.prev_R["right"] is not None and self.prev_ang_t is not None:
                dt = t_now - self.prev_ang_t
                if dt > 1e-4:
                    for s in p:
                        raw_w = log3(R[s] @ self.prev_R[s].T) / dt
                        self.w_ema[s] = raw_w if self.w_ema[s] is None else \
                            ANG_VEL_EMA_ALPHA * self.w_ema[s] + (1 - ANG_VEL_EMA_ALPHA) * raw_w
                        w_est[s] = self.w_ema[s]
            self.prev_R = dict(R)
            self.prev_ang_t = t_now
        T = self.traj
        T["t"].append(t_now)
        for s, k in (("right", "r"), ("left", "l")):
            ref = self.ref[s]
            if ref is None:
                ref = np.full(12, np.nan)
            T[f"ref_pos_{k}"].append(ref[0:3]); T[f"real_pos_{k}"].append(p[s])
            T[f"ref_rpy_{k}"].append(ref[3:6])
            T[f"real_rpy_{k}"].append(self.real_rpy[s] if has_rpy else np.full(3, np.nan))
            T[f"e_pos_{k}"].append(float(np.linalg.norm(ref[0:3] - p[s])))
            T[f"e_vel_{k}"].append(float(np.linalg.norm(ref[6:9] - v[s])))
            T[f"e_ang_{k}"].append(geodesic_angle(ref[3:6], R[s]) if not np.isnan(ref[3]) else np.nan)
            T[f"e_angvel_{k}"].append(float(np.linalg.norm(ref[9:12] - w_est[s])))

    def on_gov(self, t_ns, diff):
        if len(diff) < 24 or self.ref["right"] is None or self.ref["left"] is None:
            return
        G = self.gov
        G["t"].append(self._t(t_ns))
        for s, k, off in (("right", "r", 0), ("left", "l", 12)):
            ref = self.ref[s]
            d_pos, d_ori, d_vel, d_w = diff[off:off + 3], diff[off + 3:off + 6], diff[off + 6:off + 9], diff[off + 9:off + 12]
            raw_pos, raw_rpy, raw_vel, raw_w = ref[0:3], ref[3:6], ref[6:9], ref[9:12]
            gov_pos, gov_rpy, gov_vel, gov_w = raw_pos - d_pos, raw_rpy - d_ori, raw_vel - d_vel, raw_w - d_w
            G[f"lin_vel_raw_{k}"].append(float(np.linalg.norm(raw_vel)))
            G[f"lin_vel_gov_{k}"].append(float(np.linalg.norm(gov_vel)))
            G[f"ang_vel_raw_{k}"].append(float(np.linalg.norm(raw_w)))
            G[f"ang_vel_gov_{k}"].append(float(np.linalg.norm(gov_w)))
            real_p = self.real_pos[s] if self.real_pos[s] is not None else raw_pos
            G[f"pos_err_raw_{k}"].append(float(np.linalg.norm(raw_pos - real_p)))
            G[f"pos_err_gov_{k}"].append(float(np.linalg.norm(gov_pos - real_p)))
            R_real = rpy_to_matrix(*self.real_rpy[s]) if self.real_rpy[s] is not None else None
            G[f"ori_err_raw_{k}"].append(geodesic_angle(raw_rpy, R_real))
            G[f"ori_err_gov_{k}"].append(geodesic_angle(gov_rpy, R_real))


# ------------------------------------------------------------------ assembly
def _col(x):
    return np.asarray(x, dtype=float).reshape(-1, 1)


def build_mat(bag_dir: str, t_off_override=None):
    events, types = read_bag(bag_dir)
    if not events:
        raise SystemExit("bag decoded to zero messages")
    t0, t_off, t0_source = resolve_t0(events, bag_dir, t_off_override)

    per_topic: dict = {}
    replay = Replay(t0)
    for t_ns, topic, payload in events:
        t = (t_ns - t0) * 1e-9
        per_topic.setdefault(topic, []).append((t, payload))
        if topic == REF_TOPICS["right"]:
            replay.on_ref("right", payload)
        elif topic == REF_TOPICS["left"]:
            replay.on_ref("left", payload)
        elif topic == EE_TOPIC:
            replay.on_ee(t_ns, np.asarray(payload, dtype=float))
        elif topic == GOV_TOPIC:
            replay.on_gov(t_ns, np.asarray(payload, dtype=float))

    used: set = set()
    series = {}
    for topic, rows in per_topic.items():
        base = types[topic].split("/")[-1]
        t = _col([r[0] for r in rows])
        if base == "JointState":
            names = list(rows[0][1].name)
            idx = {n: i for i, n in enumerate(names)}
            def mat(attr):
                out = np.full((len(rows), len(names)), np.nan)
                for i, (_, m) in enumerate(rows):
                    vals = getattr(m, attr)
                    if len(vals) != len(m.name):
                        continue
                    for n, v in zip(m.name, vals):
                        if n in idx:
                            out[i, idx[n]] = v
                return out
            entry = {"t": t, "names": np.array(names, dtype=object),
                     "position": mat("position"), "velocity": mat("velocity"), "effort": mat("effort")}
        elif base == "String":
            entry = {"t": t, "data": np.array([r[1] for r in rows], dtype=object)}
        else:
            width = max(len(r[1]) for r in rows)
            data = np.full((len(rows), width), np.nan)
            for i, (_, v) in enumerate(rows):
                data[i, :len(v)] = v
            entry = {"t": t, "data": data}
        entry["type"] = types[topic]
        series[matlab_field(topic, used)] = entry

    derived = {}
    if replay.traj["t"]:
        derived["traj"] = {k: (_col(v) if np.ndim(v[0]) == 0 else np.asarray(v, dtype=float))
                           for k, v in replay.traj.items()}
    if replay.gov["t"]:
        derived["gov"] = {k: _col(v) for k, v in replay.gov.items()}

    meta = {
        "bag_dir": os.path.abspath(bag_dir),
        "t0_ns": float(t0),
        "t0_source": t0_source,
        "t_off_s": np.nan if t_off is None else float(t_off),
        "duration_s": (events[-1][0] - t0) * 1e-9,
        "topics": sorted(per_topic.keys()),
        "topic_fields": {k: v["type"] for k, v in series.items()},
    }
    return {"meta": _dict_to_struct(meta), "series": series, "derived": derived}


def main():
    import scipy.io as sio
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag_dir")
    ap.add_argument("--out", help="output .mat (default: <bag_dir>/../offline_export.mat)")
    ap.add_argument("--t-off", type=float, default=None,
                    help="trajectory-finished time [s] when the trigger rising edge is missing "
                         "(overrides trial_summary.txt)")
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.bag_dir)), "offline_export.mat")
    mat = build_mat(args.bag_dir, args.t_off)
    sio.savemat(out, mat, long_field_names=True, oned_as="column", do_compression=True)
    m = mat["meta"]
    print(f"[export_offline_bag] t0 from {m['t0_source']}, t_off={m['t_off_s']:.3f} s, "
          f"duration={m['duration_s']:.2f} s, {len(mat['series'])} topics, "
          f"derived: {sorted(mat['derived'].keys())}")
    print(f"[export_offline_bag] -> {out}")


if __name__ == "__main__":
    main()
