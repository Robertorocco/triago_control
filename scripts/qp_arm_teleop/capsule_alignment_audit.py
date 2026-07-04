#!/usr/bin/env python3
# capsule_alignment_audit.py
"""
Quantifies capsule-vs-visual-mesh MISALIGNMENT for every arm/head link,
BEFORE touching any cfg.CAPSULE_RADIUS or per-link offset by hand.

WHY THIS SCRIPT EXISTS
    "The capsule radius is bigger than the link, but the mesh still pokes
    outside it in some spots" is the classic signature of a LATERAL
    MISALIGNMENT, not an undersized radius. CollisionManager.calculate_offsets
    builds each capsule as a straight segment directly between joint(i) and
    joint(i+1), snapped to the dominant axis (see its docstring). The actual
    CAD mesh's true centerline is not always collinear with that joint-to-
    joint line -- it can kink or sit laterally offset -- and growing the
    radius only helps uniformly around that (possibly off-center) axis; it
    cannot fix a genuine sideways offset, which is why a bigger radius can
    still leave a protrusion at one specific spot along the link while being
    comfortably oversized everywhere else.

WHAT THIS SCRIPT DOES
    For every link in a chain (right/left/head arm), it:
      1. Calls the REAL CollisionManager.calculate_offsets() -- the exact
         same function main_qp_controller.py calls -- to get that link's
         capsule segment (placement + length), guaranteeing zero drift
         between this diagnostic and production behavior.
      2. Loads the link's VISUAL mesh vertices (from the URDF, same package
         search paths as visualization_engine.py's Meshcat model) and
         transforms them into the SAME joint-relative frame the capsule
         segment lives in (both the capsule's placement and a mesh
         GeometryObject's placement are relative to parentJoint -- see
         calculate_offsets' own "Store placement exactly relative to the
         JOINT" comment).
      3. For every mesh VERTEX, computes point-to-segment distance to the
         capsule's core line, minus the capsule radius. This is EXACT (not
         approximate): a flat-triangle mesh surface point can never be
         farther from the line than the farthest VERTEX of its triangle
         (point-to-line distance is a convex function, so its maximum over a
         triangle is attained at a vertex) -- so checking vertices alone
         correctly captures the worst-case protrusion.
      4. Reports, per link: capsule radius/length, worst-case protrusion
         (mm, positive = mesh pokes outside), and WHERE along the capsule's
         own axis (t=0 at the proximal joint, t=1 at the distal joint) that
         worst point occurs -- this "where" is the key diagnostic: a
         protrusion concentrated near one end usually means the segment's
         straight-line approximation misses a kink near that joint
         (translational fix), while a protrusion spread evenly along the
         whole link is more often a genuinely undersized radius.

USAGE
    python3 capsule_alignment_audit.py --urdf /path/to/robot.urdf
    python3 capsule_alignment_audit.py --urdf triago_extracted.urdf --chain right
    python3 capsule_alignment_audit.py --urdf triago_extracted.urdf --flag-threshold-mm 0

    No ROS/Gazebo needs to be running -- this is pure Pinocchio/hppfcl on a
    URDF file, using pin.neutral(model) (the same pose calculate_offsets
    itself uses), so it is a static, offline audit. If you want the audit to
    reflect the LIVE URDF (e.g. after a xacro arg change) rather than the
    committed triago_extracted.urdf, dump it first:
        ros2 param get /robot_state_publisher robot_description > /tmp/live.urdf
    (some ROS distros prefix the output with "String value is:" -- strip
    that line if present) then pass --urdf /tmp/live.urdf.

READING THE OUTPUT
    - protrusion_mm <= 0  : the mesh is fully contained in the capsule at
                              every vertex -- nothing to fix for this link.
    - protrusion_mm > 0   : flagged "POKES OUT" -- the capsule needs either
                              a bigger radius (if spread across t) or a
                              translational/rotational offset correction
                              (if concentrated near t=0 or t=1).
    - t_along_capsule     : 0.0..1.0 position along the capsule's own axis
                              where the worst vertex was found.

NOTE ON hppfcl API PORTABILITY
    Mesh-vertex extraction from an hppfcl BVH/Convex geometry differs
    slightly across hppfcl versions/bindings. `_get_vertices` tries every
    known accessor pattern defensively. If a link reports "no extractable
    mesh vertices" for EVERY link (not just primitive-shape links, which is
    expected -- see below), the installed hppfcl's python API likely uses a
    surface this script doesn't yet try; report the printed error and the
    output of `python3 -c "import hppfcl; g=hppfcl.BVHModelOBBRSS(); print(dir(g))"`
    so the accessor can be extended.

    Links whose <visual> is a primitive shape (box/cylinder/sphere) instead
    of a mesh are reported as "no extractable mesh vertices" too -- this is
    EXPECTED (there is no mesh to extract), not a bug. Every arm/head link in
    the current TRIAGo URDF uses a real .stl mesh visual, so this should not
    show up for arm_*/gripper_*_base_link/arm_head_* links in practice.
"""

import argparse

import numpy as np
import pinocchio as pin
try:
    import hppfcl
except ImportError:
    import pinocchio.hppfcl as hppfcl

import triago_control.qp_controller.config as cfg
from triago_control.qp_controller.collision_manager import CollisionManager


# Mirrors EXACTLY the (chain, tool_link_name) pairs main_qp_controller.py
# passes to CollisionManager.calculate_offsets -- keep these in sync with
# that file if it ever changes which literal tool-link names it uses.
CHAINS = {
    'right': (cfg.RIGHT_CHAIN, 'gripper_right_base_link'),
    'left':  (cfg.LEFT_CHAIN,  'gripper_left_base_link'),
    'head':  (cfg.HEAD_CHAIN,  cfg.HEAD_TOOL_LINK),
}


def _get_vertices(geom):
    """Extract an (N,3) vertex array from an hppfcl mesh geometry, trying
    every known python-binding accessor pattern (see module docstring's
    NOTE ON hppfcl API PORTABILITY). Returns None if none apply (e.g. the
    shape is a primitive, or a binding surface this script doesn't yet try)."""
    candidates = []
    if hasattr(geom, 'vertices'):
        try:
            candidates.append(np.asarray(geom.vertices()))
        except Exception:
            pass
    if hasattr(geom, 'num_vertices') and hasattr(geom, 'vertex'):
        try:
            n = geom.num_vertices
            candidates.append(np.array([np.asarray(geom.vertex(i)) for i in range(n)]))
        except Exception:
            pass
    if hasattr(geom, 'points'):
        try:
            candidates.append(np.asarray(geom.points()))
        except Exception:
            pass
    for arr in candidates:
        if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] > 0:
            return arr
    return None


def _point_segment_distance_and_t(p, a, b):
    """Distance from point p to segment [a,b], plus the normalized position
    t in [0,1] of the closest point along the segment (0=a, 1=b)."""
    ab = b - a
    length_sq = float(np.dot(ab, ab))
    if length_sq < 1e-12:
        return float(np.linalg.norm(p - a)), 0.0
    t = float(np.dot(p - a, ab) / length_sq)
    t_clamped = float(np.clip(t, 0.0, 1.0))
    closest = a + t_clamped * ab
    return float(np.linalg.norm(p - closest)), t_clamped


def _find_link_geoms(model, vmodel, link_name):
    """Every visual GeometryObject belonging to `link_name`, robust to
    fixed-joint collapsing (several URDF links can share one Pinocchio
    joint -- matching by parentFrame's NAME, not parentJoint, tells us
    exactly which URDF link each geometry came from)."""
    geoms = []
    for g in vmodel.geometryObjects:
        try:
            if model.frames[g.parentFrame].name == link_name:
                geoms.append(g)
        except Exception:
            continue
    return geoms


def audit_chain(model, data, vmodel, chain, tool_link_name, capsule_radius):
    # Reuse the REAL production function -- zero drift vs. what
    # main_qp_controller.py actually builds into the collision model.
    col = CollisionManager(model, data)
    offsets = col.calculate_offsets(chain, tool_link_name)

    results = []
    for link_name in chain:
        if link_name not in offsets:
            results.append({'link': link_name, 'error': 'capsule not computed (frame missing?)'})
            continue

        placement, length = offsets[link_name]
        z_axis = placement.rotation @ np.array([0., 0., 1.])
        p_mid = placement.translation
        a = p_mid - (length / 2.0) * z_axis
        b = p_mid + (length / 2.0) * z_axis

        link_geoms = _find_link_geoms(model, vmodel, link_name)
        if not link_geoms:
            results.append({'link': link_name, 'error': 'no visual geometry found for this link'})
            continue

        worst = None
        for g in link_geoms:
            verts_local = _get_vertices(g.geometry)
            if verts_local is None:
                continue  # primitive shape or unsupported hppfcl binding -- see module docstring
            R, t = g.placement.rotation, g.placement.translation
            verts_joint = (R @ verts_local.T + t.reshape(3, 1)).T
            for v in verts_joint:
                dist, t_along = _point_segment_distance_and_t(v, a, b)
                protrusion = dist - capsule_radius
                if worst is None or protrusion > worst['protrusion']:
                    worst = {'protrusion': protrusion, 't': t_along, 'mesh_name': g.name}

        if worst is None:
            results.append({'link': link_name, 'error': 'visual geometry has no extractable mesh vertices'})
            continue

        results.append({
            'link': link_name,
            'capsule_radius': capsule_radius,
            'capsule_length': length,
            'protrusion_mm': worst['protrusion'] * 1000.0,
            't_along_capsule': worst['t'],
            'mesh_name': worst['mesh_name'],
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Audit capsule-vs-visual-mesh alignment for the arm/head chains, "
                    "reusing the EXACT SAME CollisionManager.calculate_offsets() the "
                    "real controller uses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "If you don't have a URDF file handy, dump the live one with:\n"
            "  ros2 param get /robot_state_publisher robot_description > /tmp/live.urdf\n"
            "(strip a leading \"String value is:\" line if your ROS distro adds one), "
            "then pass --urdf /tmp/live.urdf. Or just point at the repo's own "
            "triago_extracted.urdf for a static, offline audit (no ROS/Gazebo needed)."
        ))
    parser.add_argument('--urdf', required=True, help="Path to the robot URDF file.")
    parser.add_argument('--chain', choices=['right', 'left', 'head', 'all'], default='all')
    parser.add_argument('--flag-threshold-mm', type=float, default=0.0,
                        help="Only print links whose worst-case protrusion exceeds this "
                             "many mm (default 0.0 -- show every link, even fully-contained ones).")
    args = parser.parse_args()

    model = pin.buildModelFromUrdf(args.urdf)
    data = model.createData()
    vmodel = pin.buildGeomFromUrdf(model, args.urdf, pin.GeometryType.VISUAL,
                                   package_dirs=cfg.MESH_PATHS)

    chains_to_run = list(CHAINS.keys()) if args.chain == 'all' else [args.chain]

    print("=" * 92)
    print(" CAPSULE ALIGNMENT AUDIT")
    print(f" cfg.CAPSULE_RADIUS = {cfg.CAPSULE_RADIUS * 1000:.2f} mm")
    print("=" * 92)

    flagged = []
    for chain_key in chains_to_run:
        chain, tool_link = CHAINS[chain_key]
        if not model.existFrame(chain[0]):
            print(f"\n[{chain_key}] chain not found in this URDF -- skipped.")
            continue

        print(f"\n--- {chain_key.upper()} chain ---")
        results = audit_chain(model, data, vmodel, chain, tool_link, cfg.CAPSULE_RADIUS)
        header = f"{'link':<20} {'radius_mm':>10} {'length_mm':>10} {'protrusion_mm':>15} {'t_along':>8}  mesh"
        print(header)
        print("-" * len(header))
        for r in results:
            if 'error' in r:
                print(f"{r['link']:<20} {'--':>10} {'--':>10}  ERROR: {r['error']}")
                continue
            if r['protrusion_mm'] < args.flag_threshold_mm:
                continue
            flag = "  <-- POKES OUT" if r['protrusion_mm'] > 0 else ""
            print(f"{r['link']:<20} {r['capsule_radius']*1000:>10.2f} {r['capsule_length']*1000:>10.2f} "
                  f"{r['protrusion_mm']:>15.2f} {r['t_along_capsule']:>8.2f}  {r['mesh_name']}{flag}")
            if r['protrusion_mm'] > 0:
                flagged.append(r)

    if flagged:
        flagged.sort(key=lambda r: -r['protrusion_mm'])
        print("\n" + "=" * 92)
        print(" WORST OFFENDERS (mesh pokes outside its capsule)")
        print("=" * 92)
        for r in flagged[:10]:
            where = ('near proximal joint (t~0)' if r['t_along_capsule'] < 0.33 else
                    'near distal joint (t~1)' if r['t_along_capsule'] > 0.66 else 'mid-link')
            print(f"  {r['link']:<20} pokes out by {r['protrusion_mm']:6.2f} mm "
                  f"at t={r['t_along_capsule']:.2f} ({where})")
        print("\n  Interpretation:")
        print("  - Protrusion concentrated near t=0 or t=1 -> the straight joint-to-joint")
        print("    segment misses a kink/offset near that end. A TRANSLATIONAL offset fix")
        print("    (shift that link's capsule midpoint sideways, perpendicular to its axis)")
        print("    will generally fix more of this than growing the radius.")
        print("  - Protrusion spread evenly across t (including mid-link) -> more likely a")
        print("    genuinely undersized radius for that link.")
    else:
        print("\nNo protrusions found -- every visual mesh vertex sits inside its capsule "
              f"(radius {cfg.CAPSULE_RADIUS * 1000:.2f} mm).")


if __name__ == '__main__':
    main()
