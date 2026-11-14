"""
Geometry & triangulation utilities for Maker3D Catan biome terrain solids.

Provides:
1. `warp_grid_to_slopes`: Slope-guided grid concentration that pulls interior
   vertices slightly toward steep contour steps while preserving safe spacing.
2. `contour_aligned_edge_flip`: Post-Delaunay planar diagonal flipping that aligns
   triangle edges with elevation contours (minimizing |ΔZ| across diagonals) while
   strictly preserving planar 2D convexity and watertight χ=2 topology.
3. `build_contour_aligned_hex_solid`: Drop-in upgrade to nanobanana_to_catan_hex
   watertight hex solid builder using contour-aligned triangulation.
"""

from pathlib import Path
import sys
import numpy as np
from scipy.spatial import Delaunay

SKILL_SCRIPTS = Path(__file__).resolve().parents[2] / ".agents" / "skills" / "nanobanana-3d-sculpt" / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

import nanobanana_to_catan_hex as nb  # noqa: E402


def cap_extreme_slopes(Z, dist_edge, min_step_dx=0.25, max_grad=1.85, passes=2):
    """
    Softly limits extreme near-vertical step cliffs (e.g. >~62 deg) to cleanly
    sculpted printable scarps without flattening plateaus or blurring low relief.
    Only modifies pixels exceeding `max_grad` inside the sculpted inner zone.
    """
    Zc = Z.copy()
    mask = dist_edge >= 2.0
    max_dz = float(max_grad) * float(min_step_dx)
    for _ in range(passes):
        pad = np.pad(Zc, 1, mode="edge")
        n4 = 0.25 * (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:])
        grad = np.maximum(
            np.maximum(np.abs(Zc - pad[:-2, 1:-1]), np.abs(Zc - pad[2:, 1:-1])),
            np.maximum(np.abs(Zc - pad[1:-1, :-2]), np.abs(Zc - pad[1:-1, 2:])),
        )
        steep = mask & (grad > max_dz)
        if not np.any(steep):
            break
        w = np.clip((grad - max_dz) / (max_dz * 1.5 + 1e-6), 0.0, 0.55)
        Zc = np.where(steep, Zc * (1.0 - w) + n4 * w, Zc)
    return Zc


def warp_grid_to_slopes(X, Y, Z, dist_edge, dist_vert, iters=3, strength=0.35):
    """
    Concentrates sample grid coordinates toward steep terrain slopes so cliff
    terraces receive tighter XY vertex spacing. Caller MUST re-sample Z and
    distance fields at the returned (Xw, Yw) coordinates.
    """
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])
    Xw = X.copy()
    Yw = Y.copy()

    gy, gx = np.gradient(Z, dy, dx)
    slope = np.hypot(gx, gy)
    interior = (dist_edge >= 2.2) & (dist_vert >= 5.0)
    weight = np.where(interior, np.clip(slope / 1.4, 0.0, 2.5), 0.0)

    # Smooth pull field slightly so vertices converge cleanly without kinks
    from scipy.ndimage import gaussian_filter
    w_smooth = gaussian_filter(weight, sigma=1.2)
    wy, wx = np.gradient(w_smooth, dy, dx)
    w_norm = np.hypot(wx, wy) + 1e-6
    ux = np.where(interior, wx / w_norm * np.clip(w_smooth, 0.0, 1.5), 0.0)
    uy = np.where(interior, wy / w_norm * np.clip(w_smooth, 0.0, 1.5), 0.0)

    step = float(strength) * dx / max(iters, 1)
    for _ in range(iters):
        nx_cand = Xw + step * ux
        ny_cand = Yw + step * uy
        # Enforce strictly positive monotone row/col spacing (>= 0.28 * dx)
        min_dx = 0.28 * dx
        min_dy = 0.28 * dy
        # Clamp per-cell displacement to keep monotonicity guaranteed
        nx_cand = np.clip(nx_cand, X - 0.34 * dx, X + 0.34 * dx)
        ny_cand = np.clip(ny_cand, Y - 0.34 * dy, Y + 0.34 * dy)
        nx_cand[:, 0] = X[:, 0]
        nx_cand[:, -1] = X[:, -1]
        ny_cand[0, :] = Y[0, :]
        ny_cand[-1, :] = Y[-1, :]
        # Enforce monotone row/col check
        ok_x = np.all(np.diff(nx_cand, axis=1) >= min_dx)
        ok_y = np.all(np.diff(ny_cand, axis=0) >= min_dy)
        if ok_x:
            Xw = nx_cand
        if ok_y:
            Yw = ny_cand

    return Xw, Yw


def _cross2(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def contour_aligned_edge_flip(verts_xyz, tri_top, dist_top=None, passes=4, min_gain=0.015):
    """
    Flips interior quad diagonals of a 2D-XY Delaunay triangulation so triangle
    edges align with height contours (minimizing |Z_a - Z_d| vs |Z_b - Z_c|)
    instead of cutting diagonally across steep terraced walls (sawtooth artifact).

    Guarantees:
    - Preserves exact 2D planar embedding (strict positive XY triangle areas).
    - Preserves vertex count, triangle count, boundary loop, and χ=2 watertightness.
    """
    tris = tri_top.copy()
    x = verts_xyz[:, 0]
    y = verts_xyz[:, 1]
    z = verts_xyz[:, 2]

    # Ensure every top triangle is positively oriented (CCW) in XY
    c = _cross2(x[tris[:, 0]], y[tris[:, 0]], x[tris[:, 1]], y[tris[:, 1]], x[tris[:, 2]], y[tris[:, 2]])
    neg = c < 0.0
    if np.any(neg):
        tris[neg, 1], tris[neg, 2] = tris[neg, 2].copy(), tris[neg, 1].copy()

    n_verts = len(verts_xyz)
    n_tris = len(tris)
    total_flipped = 0

    for _ in range(passes):
        # Collect all 3 half-edges per triangle
        t_ids = np.repeat(np.arange(n_tris, dtype=np.int32), 3)
        a_arr = tris[:, [2, 0, 1]].ravel()  # opposite vertex in t
        b_arr = tris[:, [0, 1, 2]].ravel()  # directed edge start
        c_arr = tris[:, [1, 2, 0]].ravel()  # directed edge end

        u = np.minimum(b_arr, c_arr).astype(np.int64)
        v = np.maximum(b_arr, c_arr).astype(np.int64)
        keys = u * n_verts + v
        order = np.argsort(keys, kind="mergesort")
        ks = keys[order]

        # Shared interior edges appear consecutively
        same = np.flatnonzero(ks[:-1] == ks[1:])
        if len(same) == 0:
            break

        # Guard against non-manifold edge triples if any ever occurred (take strictly paired)
        is_pair = np.ones(len(same), dtype=bool)
        if len(same) > 1:
            triple_mask = same[1:] == (same[:-1] + 1)
            is_pair[1:] &= ~triple_mask
            is_pair[:-1] &= ~triple_mask
        same = same[is_pair]
        if len(same) == 0:
            break

        h0 = order[same]
        h1 = order[same + 1]

        t0 = t_ids[h0]
        t1 = t_ids[h1]
        a = a_arr[h0]
        b = b_arr[h0]
        c_v = c_arr[h0]
        d = a_arr[h1]

        # Keep flips strictly inside the terrain interior (avoid tiny chamfer rim boundary)
        mask = t0 != t1
        if dist_top is not None:
            mask &= (dist_top[b] >= 1.5) & (dist_top[c_v] >= 1.5)

        dz_old = np.abs(z[b] - z[c_v])
        dz_new = np.abs(z[a] - z[d])
        gain = dz_old - dz_new
        mask &= gain > float(min_gain)
        if not np.any(mask):
            break

        idx_cand = np.flatnonzero(mask)
        # Filter strictly convex quads in XY: both (a,b,d) and (a,d,c_v) > min_area
        area0 = _cross2(x[a[idx_cand]], y[a[idx_cand]], x[b[idx_cand]], y[b[idx_cand]], x[d[idx_cand]], y[d[idx_cand]])
        area1 = _cross2(x[a[idx_cand]], y[a[idx_cand]], x[d[idx_cand]], y[d[idx_cand]], x[c_v[idx_cand]], y[c_v[idx_cand]])
        convex = (area0 > 1e-6) & (area1 > 1e-6)
        idx_cand = idx_cand[convex]
        if len(idx_cand) == 0:
            break

        # Sort best elevation alignment improvements first
        best_ord = idx_cand[np.argsort(-gain[idx_cand])]
        used = np.zeros(n_tris, dtype=bool)
        flips_this_pass = 0

        t0_arr = t0[best_ord].tolist()
        t1_arr = t1[best_ord].tolist()
        a_l = a[best_ord].tolist()
        b_l = b[best_ord].tolist()
        c_l = c_v[best_ord].tolist()
        d_l = d[best_ord].tolist()

        for ti0, ti1, va, vb, vc, vd in zip(t0_arr, t1_arr, a_l, b_l, c_l, d_l):
            if used[ti0] or used[ti1]:
                continue
            used[ti0] = True
            used[ti1] = True
            tris[ti0, 0] = va
            tris[ti0, 1] = vb
            tris[ti0, 2] = vd
            tris[ti1, 0] = va
            tris[ti1, 1] = vd
            tris[ti1, 2] = vc
            flips_this_pass += 1

        total_flipped += flips_this_pass
        if flips_this_pass == 0:
            break

    return tris, total_flipped


def build_contour_aligned_hex_solid(
    X, Y, Z, dist_edge, n_bnd=144, n_mag=32, include_magnet=True, flip_passes=4
):
    """
    Watertight hex prism solid builder matching `nanobanana_to_catan_hex.build_solid_hex_mesh`,
    upgraded with contour-aligned Delaunay edge flipping to eliminate sawtooth cliffs.
    """
    mask_in = dist_edge >= 1.25
    pts_int = np.column_stack([X[mask_in], Y[mask_in], Z[mask_in]]).astype(np.float64)
    in_dist = dist_edge[mask_in]

    v_angles = [(30.0 + 60.0 * k) * np.pi / 180.0 for k in range(6)]
    corners = [(nb.R * np.cos(a), nb.R * np.sin(a)) for a in v_angles]
    pts_per_edge = n_bnd // 6
    bnd_xy = np.array([
        [corners[e][0] + (s / float(pts_per_edge)) * (corners[(e + 1) % 6][0] - corners[e][0]),
         corners[e][1] + (s / float(pts_per_edge)) * (corners[(e + 1) % 6][1] - corners[e][1])]
        for e in range(6) for s in range(pts_per_edge)
    ], dtype=np.float64)
    n_b = len(bnd_xy)
    pts_top_bnd = np.column_stack([bnd_xy, np.full(n_b, nb.Z_BASE, dtype=np.float64)])
    bnd_dist = np.zeros(n_b, dtype=np.float64)

    all_top_verts = np.vstack([pts_int, pts_top_bnd])
    top_dist = np.concatenate([in_dist, bnd_dist])
    top_bnd_start = len(pts_int)
    bot_bnd_start = len(all_top_verts)

    tri_top = Delaunay(all_top_verts[:, :2]).simplices.astype(np.int32)
    tri_top, n_flips = contour_aligned_edge_flip(all_top_verts, tri_top, dist_top=top_dist, passes=flip_passes)

    p0, p1, p2 = all_top_verts[tri_top[:, 0]], all_top_verts[tri_top[:, 1]], all_top_verts[tri_top[:, 2]]
    flip = ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])) < 0
    tri_top[flip, 1], tri_top[flip, 2] = tri_top[flip, 2], tri_top[flip, 1]

    pts_bot_bnd = np.column_stack([bnd_xy, np.zeros(n_b, dtype=np.float64)])
    idx_e = np.arange(n_b, dtype=np.int32)
    idx_en = (idx_e + 1) % n_b
    t0, t1 = top_bnd_start + idx_e, top_bnd_start + idx_en
    b0, b1 = bot_bnd_start + idx_e, bot_bnd_start + idx_en
    tris_skirt = np.vstack([np.column_stack([t0, b0, b1]), np.column_stack([t0, b1, t1])]).astype(np.int32)

    if not include_magnet:
        tri_bot_local = Delaunay(bnd_xy).simplices.astype(np.int32)
        tris_bot = bot_bnd_start + tri_bot_local
        vb0, vb1, vb2 = bnd_xy[tri_bot_local[:, 0]], bnd_xy[tri_bot_local[:, 1]], bnd_xy[tri_bot_local[:, 2]]
        flip_b = ((vb1[:, 0] - vb0[:, 0]) * (vb2[:, 1] - vb0[:, 1]) - (vb1[:, 1] - vb0[:, 1]) * (vb2[:, 0] - vb0[:, 0])) > 0
        tris_bot[flip_b, 1], tris_bot[flip_b, 2] = tris_bot[flip_b, 2], tris_bot[flip_b, 1]
        verts = np.vstack([all_top_verts, pts_bot_bnd]).astype(np.float32)
        faces = np.vstack([tri_top, tris_bot, tris_skirt]).astype(np.int32)
        return verts, faces, n_flips

    mag_ang = np.linspace(0.0, 2.0 * np.pi, n_mag, endpoint=False)
    mag_xy = np.column_stack([nb.MAG_RADIUS * np.cos(mag_ang), nb.MAG_RADIUS * np.sin(mag_ang)])
    mag_bot_start = bot_bnd_start + n_b
    mag_ceil_start = mag_bot_start + n_mag
    mag_ctr_idx = mag_ceil_start + n_mag

    pts_mag_bot = np.column_stack([mag_xy, np.zeros(n_mag, dtype=np.float64)])
    pts_mag_ceil = np.column_stack([mag_xy, np.full(n_mag, nb.MAG_DEPTH, dtype=np.float64)])
    pt_mag_ctr = np.array([[0.0, 0.0, nb.MAG_DEPTH]], dtype=np.float64)

    ann_pts = np.vstack([bnd_xy, mag_xy])
    tri_ann = Delaunay(ann_pts).simplices
    tri_ann = tri_ann[np.hypot(ann_pts[tri_ann].mean(axis=1)[:, 0], ann_pts[tri_ann].mean(axis=1)[:, 1]) > nb.MAG_RADIUS * 1.001]
    l2g = np.concatenate([np.arange(bot_bnd_start, bot_bnd_start + n_b, dtype=np.int32),
                          np.arange(mag_bot_start, mag_bot_start + n_mag, dtype=np.int32)])
    tris_bot = l2g[tri_ann]
    vb0, vb1, vb2 = ann_pts[tri_ann[:, 0]], ann_pts[tri_ann[:, 1]], ann_pts[tri_ann[:, 2]]
    flip_b = ((vb1[:, 0] - vb0[:, 0]) * (vb2[:, 1] - vb0[:, 1]) - (vb1[:, 1] - vb0[:, 1]) * (vb2[:, 0] - vb0[:, 0])) > 0
    tris_bot[flip_b, 1], tris_bot[flip_b, 2] = tris_bot[flip_b, 2], tris_bot[flip_b, 1]

    jm = np.arange(n_mag, dtype=np.int32)
    jmn = (jm + 1) % n_mag
    mb0, mb1 = mag_bot_start + jm, mag_bot_start + jmn
    mc0, mc1 = mag_ceil_start + jm, mag_ceil_start + jmn
    tris_mag_wall = np.vstack([np.column_stack([mc0, mb1, mb0]), np.column_stack([mc0, mc1, mb1])]).astype(np.int32)
    tris_mag_ceil = np.column_stack([np.full(n_mag, mag_ctr_idx, dtype=np.int32), mc1, mc0]).astype(np.int32)

    verts = np.vstack([all_top_verts, pts_bot_bnd, pts_mag_bot, pts_mag_ceil, pt_mag_ctr]).astype(np.float32)
    faces = np.vstack([tri_top, tris_bot, tris_skirt, tris_mag_wall, tris_mag_ceil]).astype(np.int32)

    return verts, faces, n_flips
