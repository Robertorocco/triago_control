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
    python3 capsule_alignment_audit.py --urdf triago_extracted.urdf --suggest-fix

    No ROS/Gazebo needs to be running -- this is pure Pinocchio/hppfcl on a
    URDF file, using pin.neutral(model) (the same pose calculate_offsets
    itself uses), so it is a static, offline audit. If you want the audit to
    reflect the LIVE URDF (e.g. after a xacro arg change) rather than the
    committed triago_extracted.urdf, dump it first:
        ros2 param get /robot_state_publisher robot_description > /tmp/live.urdf
    (some ROS distros prefix the output with "String value is:" -- strip
    that line if present) then pass --urdf /tmp/live.urdf.

    --suggest-fix (2026-07-04): for every flagged (protrusion > 0) link,
    computes the CLOSED-FORM optimal correction -- a lateral offset
    (perpendicular re-centering) plus an axial extension on whichever
    end(s) the mesh overshoots the joint-to-joint segment -- and prints it
    as a ready-to-paste config.py dict (CAPSULE_OFFSET_OVERRIDES), together
    with an immediate numerical verification (recomputes worst-case
    protrusion WITH the fix applied; should print "[OK: fully contained]").
    See _compute_capsule_fix's docstring for the exact derivation. This
    computes the fix, it does NOT modify collision_manager.py or config.py
    -- copy the printed dict over yourself, or hand the output to whoever
    is applying the change, so the actual CBF collision geometry is never
    touched without an explicit review step.

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


def _collect_link_vertices(model, vmodel, link_name):
    """All visual mesh vertices for `link_name`, in joint-local frame (the
    SAME frame calculate_offsets' placement/length live in). Returns
    (verts (N,3) array, mesh_names_seen) or (None, []) if no extractable
    mesh vertices were found (primitive-shape visual, or unsupported hppfcl
    binding -- see module docstring)."""
    link_geoms = _find_link_geoms(model, vmodel, link_name)
    all_verts = []
    mesh_names = []
    for g in link_geoms:
        verts_local = _get_vertices(g.geometry)
        if verts_local is None:
            continue
        R, t = g.placement.rotation, g.placement.translation
        verts_joint = (R @ verts_local.T + t.reshape(3, 1)).T
        all_verts.append(verts_joint)
        mesh_names.append(g.name)
    if not all_verts:
        return None, []
    return np.vstack(all_verts), mesh_names


def _compute_capsule_fix(verts, a, z_axis, length, radius):
    """Computes the CLOSED-FORM optimal capsule correction (lateral offset +
    axial extension) that contains every vertex in `verts`, for a FIXED
    radius and FIXED axis direction (z_axis) -- only the axis's lateral
    position and its two endpoints are free.

    Two independent sub-problems (a capsule = a line segment + a fixed
    radius, with HEMISPHERICAL end caps):

    1. LATERAL OFFSET: for each vertex, decompose (v - a) into its axial
       component (along z_axis) and its perpendicular (radial) remainder.
       The centroid of all vertices' perpendicular remainders is the
       re-centering offset that minimizes the mean-squared radial distance
       -- a simple, robust, closed-form choice (not the theoretically
       optimal min-enclosing-circle center, but with CAPSULE_RADIUS=60mm
       typically far larger than the arm's real cross-section radius, this
       gap is not the limiting factor in practice -- verified numerically
       below via the recomputed "after fix" worst-case protrusion).

    2. AXIAL EXTENSION: a capsule's rounded end caps mean a vertex sitting
       PAST the segment's endpoint is covered as long as
       sqrt(axial_overshoot^2 + radial_distance^2) <= radius. For every
       vertex, this gives the loosest boundary the segment endpoint could
       still be at and still contain that vertex:
           s_start <= s_p + sqrt(radius^2 - r_p^2)   (proximal bound)
           s_end   >= s_p - sqrt(radius^2 - r_p^2)   (distal bound)
       Taking the min/max of these bounds over ALL vertices gives the
       exact minimal segment extension needed (see inline derivation in
       the code review -- this is NOT an approximation, it is the tight
       closed-form solution for a fixed-radius, fixed-axis capsule).
       Clamped to only ever EXTEND (never shrink) the original [0, length]
       segment, so this fix can only add coverage, never remove it.

    Returns a dict: lateral_offset (3-vector, joint-local frame), radius
    (possibly larger than the input `radius` if even perfect re-centering
    can't fit within it), proximal_extension, distal_extension (meters,
    >= 0), and new_length.
    """
    rel = verts - a                                  # (N,3), relative to proximal joint origin
    s = rel @ z_axis                                 # (N,) axial coordinate
    perp = rel - np.outer(s, z_axis)                 # (N,3) perpendicular remainder, already _|_ z_axis

    lateral_offset = perp.mean(axis=0)               # centroid re-centering (see docstring)
    r_new = np.linalg.norm(perp - lateral_offset, axis=1)   # (N,) radial distance AFTER re-centering

    radius_needed = float(np.max(r_new))
    effective_radius = max(radius, radius_needed)    # never shrink below the configured radius

    # Axial extension bounds (see docstring derivation). Guard the sqrt
    # argument: for the (now rare, since effective_radius covers the worst
    # r_new by construction) case r_new > effective_radius, that vertex
    # cannot be capped at all axially by ANY extension -- exclude it from
    # the axial bound (its radial excess is already reflected in
    # radius_needed above, which effective_radius already absorbed).
    covered = r_new <= effective_radius + 1e-9
    margin_sq = np.clip(effective_radius**2 - r_new**2, 0.0, None)
    margin = np.sqrt(margin_sq)

    s_start_bound = np.min(s[covered] + margin[covered]) if np.any(covered) else 0.0
    s_end_bound = np.max(s[covered] - margin[covered]) if np.any(covered) else length

    # Only ever EXTEND the original [0, length] segment, never shrink it.
    s_start = min(0.0, s_start_bound)
    s_end = max(length, s_end_bound)

    return {
        'lateral_offset': lateral_offset,
        'radius': effective_radius,
        'proximal_extension': max(0.0, -s_start),
        'distal_extension': max(0.0, s_end - length),
        'new_length': s_end - s_start,
        's_start': s_start,
        's_end': s_end,
    }


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


def audit_chain(model, data, vmodel, chain, tool_link_name, capsule_radius, apply_overrides=True):
    # Reuse the REAL production functions -- zero drift vs. what
    # main_qp_controller.py actually builds into the collision model.
    #
    # BUGFIX (2026-07-04): this used to call ONLY calculate_offsets() and
    # then check against the raw global `capsule_radius` -- i.e. it audited
    # the PRE-override geometry even after CAPSULE_OFFSET_OVERRIDES was
    # added and wired into build_collision_model via
    # CollisionManager._apply_capsule_override. That made re-running this
    # script after applying the fix silently show the OLD, unfixed numbers
    # (byte-identical to the very first run) -- the fix was correctly in
    # production code, but this diagnostic was never checking it. Now calls
    # _apply_capsule_override too (the exact same call build_collision_model
    # makes), so the audited geometry is ACTUALLY what gets shipped.
    col = CollisionManager(model, data)
    offsets = col.calculate_offsets(chain, tool_link_name)

    results = []
    for link_name in chain:
        if link_name not in offsets:
            results.append({'link': link_name, 'error': 'capsule not computed (frame missing?)'})
            continue

        placement, length = offsets[link_name]
        if apply_overrides:
            placement, length, capsule_radius_for_link = col._apply_capsule_override(
                link_name, placement, length)
        else:
            capsule_radius_for_link = capsule_radius
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
                protrusion = dist - capsule_radius_for_link
                if worst is None or protrusion > worst['protrusion']:
                    worst = {'protrusion': protrusion, 't': t_along, 'mesh_name': g.name}

        if worst is None:
            results.append({'link': link_name, 'error': 'visual geometry has no extractable mesh vertices'})
            continue

        results.append({
            'link': link_name,
            'capsule_radius': capsule_radius_for_link,
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
    parser.add_argument('--flag-threshold-mm', type=float, default=None,
                        help="Only print links whose worst-case protrusion exceeds this "
                             "many mm. Default: show EVERY link (including fully-contained "
                             "ones) -- omit this flag to see the complete picture.")
    parser.add_argument('--suggest-fix', action='store_true',
                        help="Also compute the closed-form optimal per-link correction "
                             "(lateral offset + axial extension) for every flagged link, "
                             "and print it as a ready-to-paste config.py dict, PLUS a "
                             "verification pass confirming the fix actually contains every "
                             "mesh vertex (worst-case protrusion after fix should be <= 0). "
                             "Always computed from RAW (pre-override) geometry -- see "
                             "--no-overrides below; combining --suggest-fix with an "
                             "ALREADY-populated cfg.CAPSULE_OFFSET_OVERRIDES is meaningless "
                             "(it would propose a fresh fix on top of an already-fixed link).")
    parser.add_argument('--no-overrides', action='store_true',
                        help="Audit the RAW calculate_offsets() geometry, ignoring "
                             "cfg.CAPSULE_OFFSET_OVERRIDES entirely (pre-fix numbers). "
                             "Default: apply overrides, matching EXACTLY what "
                             "build_collision_model ships in production -- use the default "
                             "to verify a fix; use --no-overrides to see the original "
                             "problem or to regenerate a fix from scratch.")
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
        results = audit_chain(model, data, vmodel, chain, tool_link, cfg.CAPSULE_RADIUS,
                              apply_overrides=not args.no_overrides)
        header = f"{'link':<20} {'radius_mm':>10} {'length_mm':>10} {'protrusion_mm':>15} {'t_along':>8}  mesh"
        print(header)
        print("-" * len(header))
        for r in results:
            if 'error' in r:
                print(f"{r['link']:<20} {'--':>10} {'--':>10}  ERROR: {r['error']}")
                continue
            # BUGFIX (2026-07-04): --flag-threshold-mm used to default to 0.0
            # and this comparison silently SKIPPED (never printed) any link
            # whose protrusion was below the threshold -- including negative
            # ("fully fine") ones, contradicting the documented default of
            # "show every link". Now only filters when the flag is EXPLICITLY
            # passed (default None = show everything, always).
            if args.flag_threshold_mm is not None and r['protrusion_mm'] < args.flag_threshold_mm:
                continue
            flag = "  <-- POKES OUT" if r['protrusion_mm'] > 0 else ""
            print(f"{r['link']:<20} {r['capsule_radius']*1000:>10.2f} {r['capsule_length']*1000:>10.2f} "
                  f"{r['protrusion_mm']:>15.2f} {r['t_along_capsule']:>8.2f}  {r['mesh_name']}{flag}")
            if r['protrusion_mm'] > 0:
                flagged.append({**r, 'chain': chain_key})

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

        if args.suggest_fix:
            print("\n" + "=" * 92)
            print(" SUGGESTED FIX (closed-form: lateral offset + axial extension per link)")
            print("=" * 92)
            print(" Ready-to-paste CAPSULE_OFFSET_OVERRIDES dict for config.py.")
            print(" Each entry is applied by calculate_offsets AFTER the existing dominant-axis")
            print(" snap -- see the accompanying collision_manager.py patch. Verified below by")
            print(" recomputing worst-case protrusion WITH the fix applied (should be <= 0).\n")

            override_lines = ["CAPSULE_OFFSET_OVERRIDES = {"]
            for r in flagged:
                chain_key = r['chain']
                chain, tool_link = CHAINS[chain_key]
                link_name = r['link']

                # ALWAYS from RAW geometry (see --suggest-fix's help text) --
                # this proposes a fresh fix from scratch, regardless of
                # whether --no-overrides was passed for the audit table above.
                col = CollisionManager(model, data)
                offsets = col.calculate_offsets(chain, tool_link)
                placement, length = offsets[link_name]
                if link_name in cfg.CAPSULE_OFFSET_OVERRIDES:
                    print(f"  [!] NOTE: {link_name} already has a CAPSULE_OFFSET_OVERRIDES "
                          f"entry -- this proposes a FRESH fix from raw geometry, ignoring "
                          f"it. Do not blindly add both.")
                z_axis = placement.rotation @ np.array([0., 0., 1.])
                a = placement.translation - (length / 2.0) * z_axis

                verts, _ = _collect_link_vertices(model, vmodel, link_name)
                if verts is None:
                    continue
                fix = _compute_capsule_fix(verts, a, z_axis, length, cfg.CAPSULE_RADIUS)

                # --- Verification: recompute worst-case protrusion WITH the fix ---
                a_fixed = a + fix['lateral_offset'] + fix['s_start'] * z_axis
                b_fixed = a + fix['lateral_offset'] + fix['s_end'] * z_axis
                worst_after = max(
                    _point_segment_distance_and_t(v, a_fixed, b_fixed)[0] - fix['radius']
                    for v in verts
                )

                lo = fix['lateral_offset']
                override_lines.append(
                    f"    '{link_name}': {{'lateral_offset': [{lo[0]*1000:+.2f}, {lo[1]*1000:+.2f}, {lo[2]*1000:+.2f}],  # mm, joint-local\n"
                    f"                     'proximal_extension': {fix['proximal_extension']*1000:.2f},  # mm\n"
                    f"                     'distal_extension': {fix['distal_extension']*1000:.2f}}},   # mm"
                )
                print(f"  {link_name:<20} lateral_offset=[{lo[0]*1000:+6.2f}, {lo[1]*1000:+6.2f}, {lo[2]*1000:+6.2f}] mm  "
                      f"prox_ext={fix['proximal_extension']*1000:5.2f}mm  dist_ext={fix['distal_extension']*1000:5.2f}mm  "
                      f"radius_needed={fix['radius']*1000:.2f}mm  "
                      f"-> worst_after_fix={worst_after*1000:+.3f}mm"
                      f"{'  [OK: fully contained]' if worst_after <= 1e-6 else '  [!] STILL POKES OUT'}")
            override_lines.append("}")
            print("\n" + "\n".join(override_lines))
    else:
        print("\nNo protrusions found -- every visual mesh vertex sits inside its capsule "
              f"(radius {cfg.CAPSULE_RADIUS * 1000:.2f} mm).")


if __name__ == '__main__':
    main()
