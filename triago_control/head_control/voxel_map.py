"""
VoxelMap — persistent multi-view point-cloud accumulation in the base frame.

WHY this exists:
    A single depth view only sees the near-facing arc of each cylinder and the
    unoccluded part of the table. That partial coverage is the root cause of the
    residual position/radius bias. By FUSING frames captured from many head
    poses into one voxel grid, we observe the full circumference of each object
    and fill in occluded table regions — turning a partial arc into a near-full
    ring, which the circle fit then nails.

HOW it works (decayed weighted voxel mean):
    Each occupied voxel stores a running SUM of point positions, a SUM of
    colours, and a WEIGHT (effective observation count). Every integration step:
        1. Decay all sums and weights by `decay` (so stale voxels fade — this is
           what lets a MOVED object disappear instead of smearing forever).
        2. Aggregate the incoming frame's points per voxel (vectorised) and add.
        3. Prune voxels whose weight fell below `w_min`.
    The fused point for a voxel is  Psum / W  (a decayed weighted average), and
    its colour is  Csum / W.

PERFORMANCE:
    Integration is vectorised with np.unique + np.add.at over the incoming
    frame; only a light per-unique-voxel dict lookup (a few thousand) runs in
    Python. The map stays small because we only ever feed it the cropped
    table-region cloud.
"""

import numpy as np

import triago_control.head_control.config as cfg


class VoxelMap:
    def __init__(self, leaf=None, decay=None, w_min=None, w_max=None):
        self.leaf = leaf if leaf is not None else cfg.VOXEL_MAP_LEAF
        self.decay = decay if decay is not None else cfg.VOXEL_MAP_DECAY
        self.w_min = w_min if w_min is not None else cfg.VOXEL_MAP_W_MIN
        self.w_max = w_max if w_max is not None else cfg.VOXEL_MAP_W_MAX

        # Parallel arrays + a key->row dict for O(1) merge.
        self.key2row = {}
        self.K = np.zeros((0, 3), dtype=np.int64)     # voxel integer keys
        self.Psum = np.zeros((0, 3))                  # decayed position sums
        self.Csum = np.zeros((0, 3))                  # decayed colour sums
        self.W = np.zeros((0,))                       # decayed weights

    # ------------------------------------------------------------------ #
    def integrate(self, points, colors):
        """Fuse one frame (already in base frame, cropped) into the map."""
        if points is None or len(points) == 0:
            # Still decay so stale voxels fade even with empty frames.
            self._decay_only()
            return

        # 1. Decay existing voxels.
        if len(self.W):
            self.Psum *= self.decay
            self.Csum *= self.decay
            self.W *= self.decay

        # 2. Aggregate the incoming frame per voxel (vectorised).
        keys = np.floor(points / self.leaf).astype(np.int64)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        U = len(uniq)
        psum = np.zeros((U, 3)); np.add.at(psum, inv, points)
        csum = np.zeros((U, 3)); np.add.at(csum, inv, colors.astype(np.float64))
        cnt = np.zeros(U);       np.add.at(cnt, inv, 1.0)

        # 3. Split uniq voxels into "already in map" vs "new".
        rows = np.full(U, -1, dtype=np.int64)
        new_local = []
        for i in range(U):
            k = (int(uniq[i, 0]), int(uniq[i, 1]), int(uniq[i, 2]))
            r = self.key2row.get(k, -1)
            if r >= 0:
                rows[i] = r
            else:
                new_local.append(i)

        ex = rows >= 0
        if np.any(ex):
            er = rows[ex]
            self.Psum[er] += psum[ex]
            self.Csum[er] += csum[ex]
            self.W[er] = np.minimum(self.w_max, self.W[er] + cnt[ex])

        if new_local:
            ni = np.array(new_local)
            base = len(self.W)
            self.K = np.vstack([self.K, uniq[ni]])
            self.Psum = np.vstack([self.Psum, psum[ni]])
            self.Csum = np.vstack([self.Csum, csum[ni]])
            self.W = np.concatenate([self.W, np.minimum(self.w_max, cnt[ni])])
            for j, i in enumerate(new_local):
                k = (int(uniq[i, 0]), int(uniq[i, 1]), int(uniq[i, 2]))
                self.key2row[k] = base + j

        # 4. Prune faded voxels.
        self._prune()

    def _decay_only(self):
        if len(self.W):
            self.Psum *= self.decay
            self.Csum *= self.decay
            self.W *= self.decay
            self._prune()

    def _prune(self):
        keep = self.W >= self.w_min
        if keep.all():
            return
        self.K = self.K[keep]
        self.Psum = self.Psum[keep]
        self.Csum = self.Csum[keep]
        self.W = self.W[keep]
        # Rebuild the index (compact arrays changed row positions).
        self.key2row = {
            (int(k[0]), int(k[1]), int(k[2])): i for i, k in enumerate(self.K)
        }

    # ------------------------------------------------------------------ #
    def get_cloud(self, w_thresh=None):
        """Return (points, colors) of the fused map above a weight threshold."""
        if w_thresh is None:
            w_thresh = cfg.VOXEL_MAP_QUERY_W
        if len(self.W) == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
        keep = self.W >= w_thresh
        if not np.any(keep):
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
        w = self.W[keep][:, None]
        pts = (self.Psum[keep] / w).astype(np.float32)
        cols = np.clip(self.Csum[keep] / w, 0, 255).astype(np.uint8)
        return pts, cols

    def size(self):
        return len(self.W)

    def reset(self):
        self.__init__(self.leaf, self.decay, self.w_min, self.w_max)
