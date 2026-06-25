#!/usr/bin/env python3
"""BeliefEstimator: EMA log-belief intent inference over a flat goal simplex.

Extracted from the monolithic SharedControlNode (update_belief / _blend_policies /
the self.beliefs / self.log_beliefs state) per shared_autonomy_analysis.md Section 4.

Thread safety: a single internal lock guards the belief dict, matching the
node's original use of plot_lock for the same purpose. The lock is private to
this class so callers never need to reason about locking order.
"""

import threading
import numpy as np


class BeliefEstimator:
    """Stateful EMA intent inference over a fixed set of goal keys (flat simplex)."""

    def __init__(self, target_keys, W, beta=0.04, ema_alpha=0.995):
        """Initializes uniform beliefs and stores the inference hyperparameters.

        Args:
            target_keys: list of goal key strings (e.g. ['Red_Top', 'Red_Side', ...]).
            W: 6x6 weighting matrix used in the policy-distance cost.
            beta: log-belief update step size.
            ema_alpha: exponential decay applied to the running log-belief.
        """
        self.target_keys = list(target_keys)
        self.W = W
        self.beta = beta
        self.ema_alpha = ema_alpha

        self._lock = threading.Lock()
        self.log_beliefs = {k: 0.0 for k in self.target_keys}
        self.beliefs = {k: 1.0 / len(self.target_keys) for k in self.target_keys}
        # Goals that are currently impossible (e.g. the already-grasped cylinder,
        # or the Platform goal while the gripper is empty). Excluded goals are
        # skipped in the cost/update, forced to probability 0, and never selected
        # as the active goal — but they remain in target_keys so the UI can still
        # display them (at 0). See set_excluded_goals.
        self._excluded = set()

    def reset(self):
        """Resets beliefs to uniform and zeros the log-belief accumulators.

        Call this on an arm switch (or any other event that invalidates the
        running intent estimate) to avoid carrying stale belief into a new context.
        """
        with self._lock:
            self.log_beliefs = {k: 0.0 for k in self.target_keys}
            self.beliefs = {k: 1.0 / len(self.target_keys) for k in self.target_keys}

    def get_beliefs(self):
        """Thread-safe snapshot (copy) of the current belief distribution."""
        with self._lock:
            return dict(self.beliefs)

    def set_excluded_goals(self, keys):
        """Mark a set of goals as currently impossible (probability forced to 0).

        Excluded goals keep appearing in target_keys / get_beliefs (so the UI can
        still show them) but at probability 0; they are skipped in update() and
        blend_policies() and never returned by get_active_goal(). Renormalizes the
        remaining (active) beliefs immediately so the distribution stays valid.

        Typical use: exclude the just-grasped cylinder's goals while HOLDING, and
        exclude the Platform goal whenever the gripper is empty.
        """
        with self._lock:
            self._excluded = set(keys)
            active = [k for k in self.target_keys if k not in self._excluded]
            for k in self._excluded:
                self.log_beliefs[k] = 0.0
                self.beliefs[k] = 0.0
            s = sum(self.beliefs[k] for k in active)
            if not active:
                return
            if s <= 1e-12:
                for k in active:
                    self.beliefs[k] = 1.0 / len(active)
            else:
                for k in active:
                    self.beliefs[k] /= s

    def get_excluded_goals(self):
        """Thread-safe snapshot (copy) of the currently-excluded goal set."""
        with self._lock:
            return set(self._excluded)

    def get_active_goal(self):
        """Thread-safe argmax goal key (over non-excluded goals) and its belief: (key, b_max)."""
        with self._lock:
            active = {k: v for k, v in self.beliefs.items() if k not in self._excluded}
            if not active:
                return None, 0.0
            key = max(active, key=active.get)
            return key, active[key]

    def update(self, v_h_curr, pi_stars, gain=1.0, pos_costs=None):
        """Performs one EMA log-belief update given the latest observed human twist.

        Args:
            v_h_curr: the most recent human/user 6D twist sample (np.ndarray).
            pi_stars: dict {goal_key: 6D policy twist} -- must contain all
                      non-excluded target_keys.
            gain: in [0, 1], scales BOTH the learning step AND the forgetting
                  decay. When the user is still (gain low), the belief shape is
                  nearly frozen (neither sharpens on noise NOR flattens to uniform).
                  When the user is actively moving, full responsiveness.
            pos_costs: optional dict {goal_key: float} giving a small proximity
                  bonus (lower = closer to that goal). Used to break the symmetry
                  when twist-costs are degenerate (aligned policies). If provided,
                  pos_costs are blended in at 10% weight relative to the main twist
                  cost so that a stationary user near a goal sees it slowly climb.
        """
        active = [k for k in self.target_keys if k not in self._excluded]
        if not active:
            return
        if not all(k in pi_stars for k in active):
            return

        g = float(np.clip(gain, 0.0, 1.0))
        beta_eff = self.beta * g
        # Scale the forgetting so the belief shape holds when the user is still.
        alpha_eff = 1.0 - (1.0 - self.ema_alpha) * g   # gain=0 -> alpha=1 (no decay)

        # Twist cost.
        raw = {k: float((v_h_curr - pi_stars[k]) @ self.W @ (v_h_curr - pi_stars[k]))
               for k in active}

        # Blend in position-distance cost (10% weight) to break degeneracy.
        if pos_costs is not None:
            for k in active:
                if k in pos_costs:
                    raw[k] = 0.90 * raw[k] + 0.10 * pos_costs[k]

        min_c = min(raw.values())
        max_c = max(raw.values())
        spread = max_c - min_c

        with self._lock:
            if spread < 1e-12:
                for k in active:
                    self.log_beliefs[k] *= alpha_eff
            else:
                for k in active:
                    norm_cost = (raw[k] - min_c) / spread
                    self.log_beliefs[k] = (alpha_eff * self.log_beliefs[k]
                                            - beta_eff * norm_cost)

            # Convert log-beliefs -> probabilities over the ACTIVE set only
            # (numerically stable softmax); excluded goals are pinned to 0.
            max_val = max(self.log_beliefs[k] for k in active)
            exps = {k: np.exp(self.log_beliefs[k] - max_val) for k in active}
            total = sum(exps.values())
            self.beliefs = {
                k: (exps[k] / total if k in active else 0.0)
                for k in self.target_keys
            }

    def blend_policies(self, policies):
        """Continuous belief-weighted convex blend of the per-goal policies.

        Note on the fix applied here: the original `_blend_policies` silently
        returned a partial (effectively zero-weighted) blend whenever a key from
        `target_keys` was missing in `policies`, degrading intent without warning.
        This version raises immediately so a missing policy is caught at the
        source rather than silently corrupting the blended command.
        """
        active = [k for k in self.target_keys if k not in self._excluded]
        missing = [k for k in active if k not in policies]
        if missing:
            raise KeyError(
                f"BeliefEstimator.blend_policies: missing policies for goals {missing}; "
                f"refusing to silently degrade the blend."
            )
        if not active:
            # Nothing is currently demandable -> command a zero twist.
            return np.zeros_like(next(iter(policies.values())))

        with self._lock:
            pi_blend = np.zeros_like(policies[active[0]])
            for k in active:
                pi_blend = pi_blend + self.beliefs[k] * policies[k]
        return pi_blend
