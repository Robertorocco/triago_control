"""
obb_detector.py — Z-locked 2D-PCA oriented-bounding-box fit per cluster.

PROVENANCE
    Ported from a colleague's ROS2/C++ `tabletop_perception_node` (PCL-based),
    which makes NO shape assumption about the object at all (unlike
    `object_detector.py`'s upright-cylinder circle fit). See config.py §14
    for the full rationale, provenance, and honest caveats of this port —
    read that before treating this as a replacement for the cylinder pipeline.

ALGORITHM (mirrors the C++ node's `cloudCallback` per-cluster block exactly):
    1. Flatten the cluster to z=0 ("Z-locked" PCA — objects are assumed to
       stand upright on the table, so we only ever need the IN-PLANE
       orientation; locking Z avoids PCA picking a tilted 3D axis on a
       squat/noisy cluster where the vertical extent is small and noisy
       relative to the footprint).
    2. Compute the 2x2 XY covariance of the flattened cluster; eigen-decompose
       it (ascending eigenvalue order, exactly `Eigen::SelfAdjointEigenSolver`'s
       convention) to get the two in-plane principal axes.
    3. Build the 3x3 rotation `eigenVectorsPCA` with the two in-plane
       eigenvectors as columns 0/2 and the vertical axis re-derived via
       `col(1) = col(2) x col(0)` (mirrors the C++ node's own column
       assignment order exactly — do not "clean up" this ordering, it is
       part of the faithful port).
    4. Project the FULL 3D cluster into that rotated frame, take its AABB,
       and recover the box's centre (mean of AABB min/max, rotated back) and
       dimensions (AABB extent along each local axis).
    5. Orientation is the rotation matrix `eigenVectorsPCA` itself, exposed
       here as a 3x3 numpy array (converted to a quaternion where a topic
       needs one — see `rotation_matrix_to_quaternion`).

WHY THIS IS A SEPARATE MODULE from object_detector.py's cylinder fit (and not
a variant/flag inside it): the two make fundamentally different assumptions
(shape-free OBB vs. axis-known cylinder) and have different bias profiles
(see config.py §14). Keeping them as independent, side-by-side estimators
lets both be run and compared on the exact same input clusters.
"""

from dataclasses import dataclass, field

import numpy as np

import triago_control.head_control.config as cfg


@dataclass
class OrientedBox:
    """One frame's raw OBB fit for a single cluster (pre-tracker)."""
    position: np.ndarray                    # (3,) base_footprint, box centre
    rotation: np.ndarray                     # (3,3) columns = box local axes
    dimensions: np.ndarray                   # (3,) full extents along local axes
    n_points: int = 0
    mean_rgb: np.ndarray = field(default_factory=lambda: np.zeros(3))


def fit_oriented_box(cluster_pts: np.ndarray, cluster_cols: np.ndarray = None) -> OrientedBox:
    """Fit a Z-locked-PCA oriented bounding box to one cluster.

    Parameters
    ----------
    cluster_pts : (N, 3) float, base_footprint frame
    cluster_cols: (N, 3) uint8, optional (only used to report a mean colour)

    Returns
    -------
    OrientedBox, or None if the cluster is degenerate (<3 points).

    Mirrors, step for step, the C++ node's block:
        pcl::compute3DCentroid (3D + flattened-2D)
        pcl::computeCovarianceMatrixNormalized (on the flattened cloud)
        Eigen::SelfAdjointEigenSolver on the 2x2 block
        eigenVectorsPCA construction (col(1) = col(2) x col(0))
        pcl::transformPointCloud + pcl::getMinMax3D in the rotated frame
        meanDiagonal / obb_position / obb_orientation recovery
    """
    n = len(cluster_pts)
    if n < 3:
        return None

    # --- 3D centroid (used as the projection's translation origin) -------
    centroid_3d = cluster_pts.mean(axis=0)

    # --- Flatten to z=0 ("Z-locked" — see module docstring) ---------------
    cloud_2d = cluster_pts.copy()
    cloud_2d[:, 2] = 0.0
    centroid_2d = cloud_2d.mean(axis=0)

    # --- 2x2 XY covariance (normalized, i.e. divided by N — matches
    # pcl::computeCovarianceMatrixNormalized) on the flattened cloud. -------
    centered_2d = cloud_2d[:, :2] - centroid_2d[:2]
    cov_2d = (centered_2d.T @ centered_2d) / float(n)

    # --- Eigendecomposition, ASCENDING eigenvalue order (matches
    # Eigen::SelfAdjointEigenSolver's documented convention exactly — numpy's
    # eigh already returns ascending order for a symmetric 2x2, so no extra
    # sort is needed, but we sort explicitly to make the convention explicit
    # and robust to any future numpy behaviour change). ---------------------
    eigvals, eigvecs_2d = np.linalg.eigh(cov_2d)
    order = np.argsort(eigvals)
    eigvecs_2d = eigvecs_2d[:, order]

    # --- Build the 3x3 rotation exactly like the C++ node:
    #   eigenVectorsPCA = Identity
    #   eigenVectorsPCA.block<2,2>(0,0) = eig_vecs2d   (columns 0 and 1 <- 2D)
    #   eigenVectorsPCA.col(1) = eigenVectorsPCA.col(2).cross(eigenVectorsPCA.col(0))
    # i.e. column 0 = first (smaller-eigenvalue) in-plane eigenvector,
    #      column 2 = world +Z (from the Identity block, untouched),
    #      column 1 = col(2) x col(0)  (re-derived, NOT the second eigenvector
    #      — this is the C++ node's own convention, preserved faithfully).
    eigen_vectors_pca = np.eye(3)
    eigen_vectors_pca[0:2, 0:2] = eigvecs_2d
    col0 = eigen_vectors_pca[:, 0]
    col2 = eigen_vectors_pca[:, 2]
    col1 = np.cross(col2, col0)
    norm1 = np.linalg.norm(col1)
    if norm1 > 1e-9:
        col1 = col1 / norm1
    eigen_vectors_pca[:, 1] = col1

    # --- Projection transform: rotate into the PCA frame, translate by
    # -R^T @ centroid_3d (matches projectionTransform in the C++ node). -----
    R = eigen_vectors_pca.T   # rows = box axes, i.e. world->box rotation
    t = -(R @ centroid_3d)
    cloud_projected = cluster_pts @ R.T + t

    # --- AABB in the projected (box-local) frame ---------------------------
    min_pt = cloud_projected.min(axis=0)
    max_pt = cloud_projected.max(axis=0)
    dimensions = max_pt - min_pt

    # --- Recover world-frame centre and confirm orientation -----------------
    # meanDiagonal = 0.5*(max+min) in the LOCAL frame;
    # obb_position = eigenVectorsPCA @ meanDiagonal + centroid_3d
    mean_diagonal = 0.5 * (max_pt + min_pt)
    obb_position = eigen_vectors_pca @ mean_diagonal + centroid_3d

    mean_rgb = (
        cluster_cols.astype(np.float64).mean(axis=0)
        if cluster_cols is not None and len(cluster_cols) > 0
        else np.zeros(3)
    )

    return OrientedBox(
        position=obb_position,
        rotation=eigen_vectors_pca,
        dimensions=dimensions,
        n_points=n,
        mean_rgb=mean_rgb,
    )


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion [x, y, z, w].

    Standard Shepperd's method (numerically robust across all rotation
    magnitudes, matches Eigen::Quaternionf's own internal construction from
    a rotation matrix). Used only where a downstream consumer (e.g. a
    PoseStamped / Marker) needs a quaternion — the tracker itself keeps the
    rotation as a plain 3x3 matrix throughout, mirroring the C++ node's own
    `Eigen::Quaternionf obb_orientation(eigenVectorsPCA)` conversion point.
    """
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    q = np.array([x, y, z, w])
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
