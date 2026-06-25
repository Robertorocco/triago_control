#!/usr/bin/env python3
"""PlotManager: belief histogram + twist comparison (radar + scrolling diff) UI.

Extracted from the ~90 lines of matplotlib setup in SharedControlNode.__init__
plus the update_plot method, per shared_autonomy_analysis.md Section 1 ("The
entire matplotlib setup belongs in a PlotManager class") and Section 4.

Bug fix applied here: the original update_plot called `.remove()` then `fill()`
on the radar's filled polygons every tick (10 Hz). matplotlib's PolyCollection
artists created by `fill()` are never fully garbage-collected when repeatedly
replaced this way -- the Axes' internal artist list and associated draw caches
keep growing, leaking memory over a long session. This version creates the
PolyCollection patches ONCE in __init__ and mutates their vertices in place via
`set_xy(...)` on every update, exactly as recommended in the analysis.
"""

import time
import threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt


class PlotManager:
    """Owns both matplotlib figures and performs thread-safe, leak-free updates."""

    COMPONENTS = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
    N_COMP = 6
    HISTORY_LEN = 150  # 150 x 0.1s = 15s of history at 10 Hz

    def __init__(self, target_keys, plot_lock=None, logger=None, freq_window_s=10.0):
        """Builds both figures up front; all artists that get updated every tick
        are created once here and mutated in place afterwards.

        Args:
            target_keys: list of goal key strings, used as the belief-bar x-axis.
            plot_lock: optional threading.Lock to share with the producer thread
                       (the control loop). If None, a private lock is created.
            logger: optional object exposing .info(msg) for frequency reporting
                    (e.g. a ROS node's get_logger()). If None, frequency is not logged.
            freq_window_s: reporting window (s) for the plotting-loop frequency monitor.
        """
        self.target_keys = list(target_keys)
        self.plot_lock = plot_lock if plot_lock is not None else threading.Lock()
        self.logger = logger
        self.freq_window_s = freq_window_s

        self._plot_ticks = 0
        self._plot_last_print = time.time()

        self._twist_history_vh = deque(maxlen=self.HISTORY_LEN)
        self._twist_history_pi = deque(maxlen=self.HISTORY_LEN)
        self._twist_history_label = deque(maxlen=self.HISTORY_LEN)

        # Staging slot: producer (control loop) writes, update() reads, under plot_lock.
        self._twist_snapshot = None  # dict: {'v_h', 'pi_star', 'goal_key'}
        self._latest_beliefs = {k: 1.0 / len(self.target_keys) for k in self.target_keys}
        self._excluded_goals = set()  # goals disabled from estimation -> drawn lighter

        plt.ion()
        self._build_belief_figure()
        self._build_twist_figure()

    # ------------------------------------------------------------------
    # Figure construction (run once)
    # ------------------------------------------------------------------
    def _build_belief_figure(self):
        self.fig, self.ax_beliefs = plt.subplots(1, 1, figsize=(8, 4))
        self.fig.canvas.manager.set_window_title('Intent Inference')

        init_vals = [self._latest_beliefs[k] for k in self.target_keys]
        self.bars = self.ax_beliefs.bar(self.target_keys, init_vals,
                                         color='#00BFFF', edgecolor='black')
        self.ax_beliefs.set_ylim(0, 1)
        self.ax_beliefs.set_ylabel('Probability')
        self.ax_beliefs.set_title('Intent Inference (Continuous Manifolds)')
        plt.tight_layout()

    def _build_twist_figure(self):
        self.fig2, (self.ax_radar, self.ax_diff) = plt.subplots(
            1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1, 2]})
        self.fig2.patch.set_facecolor('#0f0f1a')
        self.fig2.canvas.manager.set_window_title('Twist Command Monitor')

        # -- Radar (spider) axes --
        self.fig2.delaxes(self.ax_radar)
        self.ax_radar = self.fig2.add_subplot(121, projection='polar')
        self.ax_radar.set_facecolor('#0f0f1a')

        angles = np.linspace(0, 2 * np.pi, self.N_COMP, endpoint=False).tolist()
        angles += angles[:1]  # close the loop
        self._radar_angles = angles

        self.ax_radar.set_thetagrids(
            np.degrees(angles[:-1]), self.COMPONENTS, fontsize=9, color='#ccccdd')
        self.ax_radar.set_ylim(0, 1)
        self.ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
        self.ax_radar.set_yticklabels(['', '', '', ''], fontsize=7)
        self.ax_radar.grid(color='#334', linewidth=0.6)
        self.ax_radar.spines['polar'].set_color('#334')
        self.ax_radar.set_title('Per-DoF magnitude\n(normalised)', color='white',
                                 fontsize=9, pad=12)

        empty = [0.0] * (self.N_COMP + 1)
        self._radar_line_vh, = self.ax_radar.plot(
            angles, empty, 'o-', lw=1.8, color='#ff6b6b', markersize=3, label='v_h (user)')
        self._radar_line_pi, = self.ax_radar.plot(
            angles, empty, 'o-', lw=1.8, color='#4ecdc4', markersize=3, label='pi* (goal)')

        # Fix applied here: create the fill PolyCollections ONCE and mutate them
        # in place via set_xy() on every update instead of remove()+fill() every
        # tick (the original leaked PathCollection/PolyCollection artists at
        # 10 Hz -- see module docstring).
        self._radar_fill_vh = self.ax_radar.fill(
            angles, empty, alpha=0.20, color='#ff6b6b')[0]
        self._radar_fill_pi = self.ax_radar.fill(
            angles, empty, alpha=0.20, color='#4ecdc4')[0]

        self.ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
                              fontsize=8, framealpha=0.2, labelcolor='white')

        # -- Scrolling difference axes --
        self.ax_diff.set_facecolor('#0f0f1a')
        self.ax_diff.tick_params(colors='#aaa')
        for spine in self.ax_diff.spines.values():
            spine.set_edgecolor('#334')
        self.ax_diff.set_xlabel('time (s)', color='#aaa', fontsize=9)
        self.ax_diff.set_ylabel('v_h  -  pi*  (component)', color='#aaa', fontsize=9)
        self.ax_diff.set_title('Deviation from best-goal policy  (user twist - pi*)',
                                color='white', fontsize=9)
        self.ax_diff.axhline(0, color='#445', linewidth=0.8)
        self.ax_diff.set_xlim(-self.HISTORY_LEN * 0.1, 0)
        self.ax_diff.set_ylim(-0.25, 0.25)

        diff_colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#c77dff', '#ff9f43']
        self._diff_lines = []
        for comp, col in zip(self.COMPONENTS, diff_colors):
            ln, = self.ax_diff.plot([], [], lw=1.3, color=col, label=comp)
            self._diff_lines.append(ln)
        self.ax_diff.legend(loc='upper left', ncol=3, fontsize=8,
                             framealpha=0.15, labelcolor='white')
        self._diff_goal_text = self.ax_diff.text(
            0.99, 0.97, '', transform=self.ax_diff.transAxes,
            ha='right', va='top', fontsize=9, color='#4ecdc4',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                      edgecolor='#4ecdc4', alpha=0.8))

        self.fig2.tight_layout(pad=2.0)

    # ------------------------------------------------------------------
    # Producer-side API (called from the control loop thread)
    # ------------------------------------------------------------------
    def push_beliefs(self, beliefs: dict, excluded=None):
        """Thread-safe write of the latest belief distribution (+ excluded set)."""
        with self.plot_lock:
            self._latest_beliefs = dict(beliefs)
            self._excluded_goals = set(excluded) if excluded else set()

    def push_twist_snapshot(self, v_h, pi_star, goal_key):
        """Thread-safe write of the latest (v_h, pi*, goal_key) sample for the twist plot."""
        with self.plot_lock:
            self._twist_snapshot = {
                'v_h': np.asarray(v_h).copy(),
                'pi_star': np.asarray(pi_star).copy(),
                'goal_key': goal_key,
            }

    # ------------------------------------------------------------------
    # Consumer-side API (called from the main/UI thread)
    # ------------------------------------------------------------------
    def update(self):
        """Thread-safely updates both the belief histogram and the twist comparison window.

        Must be called from the main thread (matplotlib requirement); only the
        data hand-off is locked, all heavy drawing happens outside the lock.
        """
        self._plot_ticks += 1
        current_time = time.time()
        if self.logger is not None and (current_time - self._plot_last_print) >= self.freq_window_s:
            fps = self._plot_ticks / (current_time - self._plot_last_print)
            # self.logger.info(f"[FREQ] Plotting UI:  {fps:.1f} Hz")  # Disabled: not useful
            self._plot_ticks = 0
            self._plot_last_print = current_time

        with self.plot_lock:
            snap = self._twist_snapshot
            probs = [self._latest_beliefs[k] for k in self.target_keys]
            excluded = set(self._excluded_goals)

        self._update_belief_bars(probs, excluded)

        if snap is None:
            return
        self._update_twist_plot(snap)

    def _update_belief_bars(self, probs, excluded=frozenset()):
        max_idx = int(np.argmax(probs))
        for i, bar in enumerate(self.bars):
            key = self.target_keys[i]
            bar.set_height(probs[i])
            if key in excluded:
                # Goal disabled from estimation -> draw it light/faded.
                bar.set_color('#cccccc')
                bar.set_alpha(0.4)
                bar.set_edgecolor('#999999')
            else:
                bar.set_alpha(1.0)
                bar.set_color('red' if i == max_idx else '#00BFFF')
                bar.set_edgecolor('black')

    def _update_twist_plot(self, snap):
        v_h = snap['v_h']
        pi_star = snap['pi_star']
        goal_key = snap['goal_key']

        self._twist_history_vh.append(v_h)
        self._twist_history_pi.append(pi_star)
        self._twist_history_label.append(goal_key)

        # -- Radar: normalise both vectors by a common scale --
        combined_max = max(np.max(np.abs(v_h)), np.max(np.abs(pi_star)), 1e-6)
        vh_norm = (np.abs(v_h) / combined_max).tolist()
        pi_norm = (np.abs(pi_star) / combined_max).tolist()
        vh_norm += vh_norm[:1]
        pi_norm += pi_norm[:1]

        self._radar_line_vh.set_data(self._radar_angles, vh_norm)
        self._radar_line_pi.set_data(self._radar_angles, pi_norm)

        # In-place vertex update -- no remove()/fill() churn, no artist leak.
        verts_vh = np.column_stack((self._radar_angles, vh_norm))
        verts_pi = np.column_stack((self._radar_angles, pi_norm))
        self._radar_fill_vh.set_xy(verts_vh)
        self._radar_fill_pi.set_xy(verts_pi)

        # -- Scrolling difference lines --
        n = len(self._twist_history_vh)
        if n < 2:
            return

        vh_arr = np.array(self._twist_history_vh)   # (n, 6)
        pi_arr = np.array(self._twist_history_pi)   # (n, 6)
        diff = vh_arr - pi_arr                       # (n, 6)
        t_axis = np.linspace(-(n - 1) * 0.1, 0.0, n)

        for c, ln in enumerate(self._diff_lines):
            ln.set_data(t_axis, diff[:, c])

        d_max = max(np.max(np.abs(diff)), 0.02)
        self.ax_diff.set_ylim(-d_max * 1.15, d_max * 1.15)
        self.ax_diff.set_xlim(t_axis[0], 0)

        self._diff_goal_text.set_text(f'pi* goal: {goal_key}')

