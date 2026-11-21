"""
Montaña Granito V5 — Geometry-First 360° Catan Mountain Sculptor (360° Dual-View Nano Banana Fusion).

Reproduces the winning Geometry-First V4 architecture tailored specifically to the new
granite twin-spire & terracotta-gorge hero concept (montana_granito_v5):
  Stage 1 (--stage clay):
    Synthesizes the bespoke 360° granitic alpine massif aligned to hero_cam_params.npy,
    optimizes skyline silhouette parity (MAE < 2.0 px vs hero_sq.png), and renders:
      - v5_clay_vs_hero.png
      - v5_clay_6views.png (0..300 deg)
      - clay_back_180.png (1024x1024 studio template for Nano Banana 180-deg paint-over)
  Stage 2 (--stage guide):
    Computes de-lit front hero projection (hero_delit_2k.npz) + 2048x2048 orthographic
    semantic clay guide (clay_ortho_2k.png, ortho_hybrid_guide_2k.png).
  Stage 3 (--stage fuse):
    Fuses:
      1) Front hero_delit (0 deg, hero_sq.png)
      2) Back Nano Banana geometry-locked paint-over (180 deg, refs/back_hero_nb.png)
      3) Dual-view slope-aligned rigid Voronoi exemplar quilting (side transitions)
    through a normalized multi-view 2-band Laplacian pyramid + machined brushed gunmetal
    bezel & pad rings, exporting montana_granito_v5_hm_2k.npy & montana_granito_v5_albedo_4k.png.
"""
import argparse
import math
import os
import sys
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from montana_sculpt_v4 import (
    R,
    APOTHEM,
    Z_BASE,
    ROAD_MARGIN,
    PAD_RADIUS,
    hex_dist,
    vert_dist,
    smoothstep,
    smax,
    smooth_fbm,
    ridge_fbm,
    faceted_ridge,
    rock_outcrop,
    project_pts_vec,
    ray_march_heightfield,
    extract_hero_swatch_banks,
    synthesize_slope_aligned_exemplar,
    render_clay_audits,
)

BIOME_DIR = os.path.join(SCRIPT_DIR, "montana_granito_v5")
REFS_DIR = os.path.join(BIOME_DIR, "refs")
WORK_DIR = os.path.join(BIOME_DIR, "work")
RELIEF_MM = 45.2


def smin(a, b, k=2.0):
    """Smooth minimum helper for carving rocky gorges cleanly without sharp steps."""
    return -smax(-a, -b, k=k)


def river_gorge_depth(X, Y, pts_xy_w_d, sharp_edge=0.75):
    """Carve a winding V/U canyon channel along polyline segments [(x, y, half_width, depth), ...]."""
    carve = np.zeros_like(X, dtype=np.float64)
    for i in range(len(pts_xy_w_d) - 1):
        x0, y0, w0, d0 = pts_xy_w_d[i]
        x1, y1, w1, d1 = pts_xy_w_d[i + 1]
        vx, vy = x1 - x0, y1 - y0
        l2 = vx * vx + vy * vy + 1e-9
        t = np.clip(((X - x0) * vx + (Y - y0) * vy) / l2, 0.0, 1.0)
        px = x0 + t * vx
        py = y0 + t * vy
        dist = np.hypot(X - px, Y - py)
        w = w0 * (1.0 - t) + w1 * t
        d = d0 * (1.0 - t) + d1 * t
        prof = np.clip(1.0 - (dist / (w + 1e-6)) ** sharp_edge, 0.0, 1.0)
        carve = smax(carve, d * (prof ** 1.25), k=0.9)
    return carve


def build_v5_granite_dem(N=1024):
    """
    Build bespoke 360-deg granitic twin-spire Catan mountain DEM matched to hero_sq.png:
      - Camera azimuth is ~30 deg (looking from bottom-right toward top-left: screen +u ~ +30 deg)
      - Twin towering jagged summit spires: (-17.6, +15.5) & (-3.7, +14.7), peaking Z ~ 70.5 mm
      - Left & right upper buttresses: (-27, +2) -> (-41, -7) & (+9, +21) -> (+25, +13)
      - Rich fractured 180-deg north/back massif (-24..+28, +26..+42) with secondary spurs
      - Front-left granite bastion (+1, -21) & front-right bastion (+30, -7)
      - Central high-altitude protected glacial bowl (+6.6, -2.2) opening into
      - S-curve terracotta mule canyon exiting toward (+30.8, -30.0) on front-right rim
      - Left angular talus scree chute (-24.1, -24.5) cascading to mossy meadow apron.
    """
    lin = np.linspace(-R, R, N)
    X, Y = np.meshgrid(lin, lin)
    dist_h = hex_dist(X, Y)
    dist_v = vert_dist(X, Y)
    taper = smoothstep(ROAD_MARGIN + 0.15, ROAD_MARGIN + 5.8, dist_h) * smoothstep(PAD_RADIUS + 0.15, PAD_RADIUS + 5.6, dist_v)

    base_rock = 0.0

    # --- 1. TWIN CENTRAL GRANITE SPIRES & CROWN BUTTRESSES ---
    spires = [
        # Primary main spire (-17.8, +15.6) + razor shoulder descending right-ward toward saddle
        (-17.8, 15.6, 46.2, -13.0, 16.0, 35.5, 12.6, 10.0, 0.96, 11, 0.21),
        (-17.8, 15.6, 45.8, -26.8, 2.0, 29.0, 12.2, 9.8, 0.94, 12, 0.19),
        (-26.8, 2.0, 29.0, -39.5, -6.5, 11.5, 9.6, 6.8, 0.92, 13, 0.18),
        # Right twin peak (-3.6, +14.8) with deep V-cleft separating from main spire, plunging right
        (-3.6, 14.8, 42.2, 8.8, 21.0, 29.5, 11.6, 9.0, 0.95, 14, 0.20),
        (8.8, 21.0, 29.2, 24.5, 13.5, 14.2, 9.2, 6.5, 0.92, 16, 0.18),
        (24.5, 13.5, 14.2, 36.5, 4.0, 5.2, 6.4, 4.4, 0.90, 17, 0.16),

        # --- 2. FRONT-LEFT & FRONT-RIGHT GRANITE BASTIONS ---
        # Connected continuous jagged bastion massifs framing the gorge cleanly
        (-18.5, -11.0, 22.0, -9.5, -13.5, 24.5, 9.5, 9.0, 0.95, 23, 0.20),
        (-9.5, -13.5, 24.5, 1.0, -20.6, 25.0, 9.4, 8.6, 0.96, 21, 0.22),
        (1.0, -20.6, 24.5, 10.2, -25.0, 14.8, 8.2, 5.8, 0.93, 22, 0.20),
        # Right gorge wall bastion (+20, +3) -> (+30.2, -6.8) -> (+38, -16)
        (8.8, 21.0, 28.5, 21.5, 2.5, 23.0, 9.0, 8.2, 0.93, 26, 0.19),
        (21.5, 2.5, 23.0, 30.2, -6.8, 21.2, 8.6, 7.6, 0.95, 24, 0.21),
        (30.2, -6.8, 20.8, 38.0, -16.5, 9.5, 7.2, 4.8, 0.92, 25, 0.18),

        # --- 3. NORTH / BACK (180-deg) CONTINUOUS FRACTURED MASSIF SPURS ---
        # Integrated ridged buttresses (zero standalone pointed cones!)
        (-17.8, 15.6, 44.5, -19.5, 31.5, 27.5, 11.8, 8.4, 0.95, 31, 0.21),
        (-19.5, 31.5, 27.2, -26.5, 42.5, 9.8, 8.2, 5.2, 0.92, 32, 0.19),
        (-3.6, 14.8, 40.8, -1.5, 32.0, 25.8, 11.0, 8.0, 0.95, 33, 0.22),
        (-1.5, 32.0, 25.5, 3.5, 43.0, 9.0, 7.8, 5.0, 0.92, 34, 0.19),
        (8.8, 21.0, 28.6, 17.5, 33.5, 19.5, 9.4, 6.8, 0.94, 35, 0.20),
        (17.5, 33.5, 19.2, 27.0, 40.5, 7.8, 6.6, 4.6, 0.90, 36, 0.16),
        (-28.0, 12.0, 22.8, -38.5, 23.5, 10.2, 8.8, 5.4, 0.93, 37, 0.19),
        (-12.5, 26.0, 28.0, -13.0, 41.5, 10.5, 8.6, 5.2, 0.92, 38, 0.18),
    ]
    for s in spires:
        base_rock = smax(base_rock, faceted_ridge(X, Y, *s), k=1.65)

    # Secondary integrated summit crest crowns (merged tightly with main ridge lines, no isolated pillars)
    outcrops = [
        (-17.8, 15.6, 46.5, 6.6, 5.6, 24.0, 0.96),
        (-3.6, 14.8, 42.4, 6.2, 5.2, -18.0, 0.95),
        (-26.8, 2.2, 29.8, 6.6, 5.4, 38.0, 0.93),
        (9.0, 21.2, 29.8, 6.0, 5.0, -32.0, 0.93),
        (1.2, -20.5, 25.0, 6.4, 5.2, 18.0, 0.95),
        (30.3, -6.7, 21.4, 6.0, 5.0, -42.0, 0.94),
    ]
    for oc in outcrops:
        base_rock = smax(base_rock, rock_outcrop(X, Y, *oc), k=1.8)

    # Left angular talus scree ridges (crisp slanted diagonal buttress ribs running toward (-34, -22), zero dome!)
    scree_spurs = [
        (-26.0, -2.0, 22.0, -34.0, -16.0, 9.5, 6.5, 4.2, 0.92, 61, 0.22),
        (-19.0, -10.5, 20.5, -28.5, -24.0, 7.5, 6.2, 3.8, 0.91, 62, 0.24),
        (-23.0, -16.0, 11.5, -35.0, -25.0, 3.2, 5.0, 3.0, 0.88, 63, 0.18),
    ]
    for sr in scree_spurs:
        base_rock = smax(base_rock, faceted_ridge(X, Y, *sr), k=1.5)

    # Concave upper glacier shelf behind the front ridges (+5.5, +6.5 -> fades out BEFORE front gorge y=-6)
    glacier_shelf = 13.8 * np.exp(-(((X - 4.5) / 11.5) ** 2 + ((Y - 7.5) / 8.5) ** 2))
    base_rock = smax(base_rock, glacier_shelf, k=2.8)

    # Broad faceted granite cliff dihedrals + rock ledges
    ang_rad = np.deg2rad(30.0)
    U = X * math.cos(ang_rad) + Y * math.sin(ang_rad)
    V = -X * math.sin(ang_rad) + Y * math.cos(ang_rad)
    flute_broad = ridge_fbm(U * 1.15, V * 0.65, [0.16, 0.34], [0.55, 0.45], seed=505)
    flute_cross = ridge_fbm(U * 0.60, V * 1.15, [0.19, 0.42], [0.58, 0.42], seed=509)
    crag_detail = 0.56 * flute_broad + 0.44 * flute_cross

    gy_b, gx_b = np.gradient(base_rock, 2 * R / (N - 1), 2 * R / (N - 1))
    slope_b = np.hypot(gx_b, gy_b)
    cliff_w = smoothstep(0.42, 1.38, slope_b) * smoothstep(2.8, 10.5, base_rock)
    relief = base_rock + cliff_w * (2.3 * (crag_detail - 0.36))

    # Gently carve winding terracotta canyon floor (level gentle valley, NOT an inverted U cushion!)
    gorge_pts = [
        (7.5, 2.5, 4.4, 2.4),
        (12.5, -4.5, 5.0, 6.2),
        (17.5, -13.5, 5.6, 8.8),
        (24.5, -21.5, 6.2, 9.6),
        (30.8, -30.5, 6.8, 9.8),
    ]
    carve_g = river_gorge_depth(X, Y, gorge_pts, sharp_edge=0.92)
    relief = np.maximum(0.40, relief - carve_g)

    # Carve clean V-notch on north-side rear massif for 180-deg silhouette depth
    north_gorge = [
        (-8.5, 22.0, 4.2, 3.8),
        (-7.5, 31.0, 5.0, 6.4),
        (-5.5, 41.5, 5.8, 7.2),
    ]
    relief = np.maximum(0.40, relief - river_gorge_depth(X, Y, north_gorge, sharp_edge=0.92))

    # Subtle natural rolling meadow floor (broad low-frequency organic soil)
    meadow_noise = smooth_fbm(X, Y, [0.10, 0.22], [0.62, 0.38], seed=533)
    meadow_base = 1.15 * smoothstep(ROAD_MARGIN + 0.15, ROAD_MARGIN + 3.6, dist_h) * smoothstep(PAD_RADIUS + 0.15, PAD_RADIUS + 3.6, dist_v)
    meadow_base *= (0.60 + 0.75 * meadow_noise)
    relief = np.maximum(relief, meadow_base)

    # Low rugged mossy rock cobbles along scree toe & canyon mouth
    rng_b = np.random.default_rng(577)
    for _ in range(18):
        bx = float(rng_b.uniform(-36.0, 36.0))
        by = float(rng_b.uniform(-36.0, 36.0))
        d_c = math.hypot(bx, by)
        if 20.0 < d_c < 43.0:
            bh = float(rng_b.uniform(1.1, 2.2))
            brx = float(rng_b.uniform(1.5, 2.6))
            bry = float(rng_b.uniform(1.2, 2.2))
            bang = float(rng_b.uniform(0.0, 180.0))
            relief = smax(relief, rock_outcrop(X, Y, bx, by, bh, brx, bry, ang_deg=bang, sharp=0.82), k=0.55)

    # Normalize peak relief strictly inside target [44.8..45.5 mm] above Z_BASE
    rel_taper = np.maximum(0.0, relief) * taper
    cur_max = float(rel_taper.max())
    if cur_max > 1e-4:
        rel_taper *= (RELIEF_MM / cur_max)

    Z = Z_BASE + rel_taper
    return X, Y, Z, dist_h, dist_v, taper


def optimize_v5_skyline(X, Y, Z, taper, cam_p, fg_hero, iters=18):
    """Align silhouette skyline cleanly to hero_sq.png while locking peak <= 70.6 mm."""
    N = Z.shape[0]
    ys_fg, xs_fg = np.where(fg_hero)
    x_min_f, x_max_f = int(xs_fg.min()) + 40, int(xs_fg.max()) - 40
    hero_sky = np.full(1024, 1024, dtype=np.float32)
    for x in range(1024):
        col = np.where(fg_hero[:, x])[0]
        if len(col) > 0:
            hero_sky[x] = float(col.min())

    Z_cur = Z.copy()
    valid_x = (np.arange(1024) >= x_min_f) & (np.arange(1024) <= x_max_f) & (hero_sky < 520)
    last_mae = 999.0
    for it in range(iters):
        _, any_hit, _, hx, hy, hz, _, _, _, _ = ray_march_heightfield(Z_cur, cam_p, out_w=512, out_h=512)
        c_sky = np.full(512, 512.0, dtype=np.float32)
        hx_s = np.zeros(512, dtype=np.float32)
        hy_s = np.zeros(512, dtype=np.float32)
        for u2 in range(512):
            col = np.where(any_hit[:, u2])[0]
            if len(col) > 0:
                r2 = int(col[0])
                c_sky[u2] = r2 * 2.0
                hx_s[u2] = hx[r2, u2]
                hy_s[u2] = hy[r2, u2]
        c_sky_1k = np.interp(np.arange(1024), np.arange(512) * 2.0, c_sky)
        hx_1k = np.interp(np.arange(1024), np.arange(512) * 2.0, hx_s)
        hy_1k = np.interp(np.arange(1024), np.arange(512) * 2.0, hy_s)

        err_px = c_sky_1k - hero_sky
        mae = float(np.mean(np.abs(err_px[valid_x])))
        last_mae = mae
        if it in (0, 4, 8, 12, iters - 1):
            print(f"  [skyline iter {it:02d}] MAE={mae:.2f} px (Z_max={Z_cur.max():.2f} mm)")
        if mae < 1.35:
            break

        dz_accum = np.zeros((N, N), dtype=np.float64)
        w_accum = np.zeros((N, N), dtype=np.float64)
        for x1 in range(x_min_f, x_max_f, 3):
            if not valid_x[x1]:
                continue
            d_px = err_px[x1]
            if abs(d_px) < 0.7:
                continue
            # Camera pixel Y increases downward -> positive err_px means our 3D peak is too low!
            delta_z = np.clip(d_px * 0.145, -1.8, 1.8)
            cx, cy = hx_1k[x1], hy_1k[x1]
            r2 = (X - cx) ** 2 + (Y - cy) ** 2
            g = np.exp(-r2 / (2.0 * 4.0 ** 2))
            dz_accum += delta_z * g
            w_accum += g
        mask = w_accum > 1e-4
        dz = np.zeros_like(Z_cur)
        dz[mask] = dz_accum[mask] / np.maximum(w_accum[mask], 1.0)
        dz = ndi.gaussian_filter(dz, sigma=3.5)
        Z_cur = np.maximum(Z_BASE, Z_cur + 0.55 * dz * taper)
        Z_cur = np.clip(Z_cur, Z_BASE, Z_BASE + 46.8)
    return Z_cur, last_mae


def compute_cam_p_180(cam_p):
    """Exact 180-degree opposite camera parameters aligned to ray_march_heightfield(yaw_offset_deg=180.0)."""
    p180 = cam_p.copy()
    p180[0] = cam_p[0] + 180.0
    p180[3] = -cam_p[3]
    p180[4] = -cam_p[4]
    return p180


def render_clay_turnaround(Z, cam_p, out_path, yaw_offset_deg=180.0, size=1024):
    """Render clean high-res studio clay view on neutral #E5E7E9 floor for Nano Banana edit pass."""
    img, any_hit, hit_terrain, _, _, _, _, _, _, _ = ray_march_heightfield(
        Z, cam_p, out_w=size, out_h=size, tex_rgb=None, yaw_offset_deg=yaw_offset_deg, sun_rel=True
    )
    out = np.full((size, size, 3), np.array([0.895, 0.902, 0.910], dtype=np.float32), dtype=np.float32)
    out[hit_terrain] = img[hit_terrain]
    plinth = any_hit & ~hit_terrain
    out[plinth] = np.array([0.16, 0.175, 0.195], dtype=np.float32)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(out_path)
    return out_path


def project_view_onto_dem(Z_2k, X, Y, dist_h, dist_v, dx, view_rgb, view_cam_p, vis_thresh=0.06, front_mode=True):
    """Compute de-lit view projection & clean occlusion-checked visibility weights onto Z_2k."""
    _, _, _, hx, hy, hz, cam_xyz, nx_g, ny_g, nz_g = ray_march_heightfield(Z_2k, view_cam_p, out_w=1024, out_h=1024)
    u_all, v_all, _ = project_pts_vec(view_cam_p, X, Y, Z_2k, 1024.0, 1024.0)
    cam_dist_1k = np.sqrt((hx - cam_xyz[0]) ** 2 + (hy - cam_xyz[1]) ** 2 + (hz - cam_xyz[2]) ** 2)
    dist_grid = np.sqrt((X - cam_xyz[0]) ** 2 + (Y - cam_xyz[1]) ** 2 + (Z_2k - cam_xyz[2]) ** 2)
    ui = np.clip(np.round(u_all), 0, 1023).astype(np.int32)
    vi = np.clip(np.round(v_all), 0, 1023).astype(np.int32)

    vdir = cam_xyz[None, None, :] - np.stack([X, Y, Z_2k], axis=-1)
    vdir /= np.linalg.norm(vdir, axis=-1, keepdims=True)
    ndotv = nx_g * vdir[..., 0] + ny_g * vdir[..., 1] + nz_g * vdir[..., 2]

    lum_v = 0.2126 * view_rgb[..., 0] + 0.7152 * view_rgb[..., 1] + 0.0722 * view_rgb[..., 2]
    sat_v = np.max(view_rgb, axis=-1) - np.min(view_rgb, axis=-1)
    # Exclude background bright studio floor (>0.82 lum, <0.05 sat) and outer slate bezel rails
    is_bg = (lum_v > 0.80) & (sat_v < 0.045)
    outer_rim = (dist_h < 7.2) | (dist_v < 12.2)
    is_rail = outer_rim & (sat_v[vi, ui] < 0.095) & (lum_v[vi, ui] < 0.62)

    vis_raw = (
        (dist_grid <= cam_dist_1k[vi, ui] + 0.55)
        & (~is_bg[vi, ui])
        & (dist_h >= 4.45)
        & (dist_v >= 10.65)
        & (ndotv > vis_thresh)
        & (~is_rail)
    )
    vis_er = ndi.binary_erosion(vis_raw, iterations=1)
    lbl_v, n_v = ndi.label(vis_er)
    if n_v > 0:
        cnts = np.bincount(lbl_v.ravel()); cnts[0] = 0
        vis_clean = (lbl_v == np.argmax(cnts))
    else:
        vis_clean = vis_er

    vis_w = ndi.gaussian_filter(vis_clean.astype(np.float32), sigma=3.8)
    vis_w = np.clip((vis_w - 0.12) / 0.76, 0.0, 1.0)
    vis_w *= smoothstep(4.8, 7.6, dist_h) * smoothstep(10.55, 11.8, dist_v) * smoothstep(vis_thresh, vis_thresh + 0.24, ndotv)

    fx = np.clip(u_all, 0, 1022.99); fy = np.clip(v_all, 0, 1022.99)
    x0 = fx.astype(np.int32); y0 = fy.astype(np.int32)
    tx_f = (fx - x0)[..., None]; ty_f = (fy - y0)[..., None]
    proj = (
        view_rgb[y0, x0] * (1 - tx_f) * (1 - ty_f)
        + view_rgb[y0, x0 + 1] * tx_f * (1 - ty_f)
        + view_rgb[y0 + 1, x0] * (1 - tx_f) * ty_f
        + view_rgb[y0 + 1, x0 + 1] * tx_f * ty_f
    )

    # Symmetric de-lighting matched to cam orientation
    if front_mode:
        ld_s = np.array([0.66, -0.35, 0.66])
    else:
        ld_s = np.array([-0.66, 0.35, 0.66])
    ld_s /= np.linalg.norm(ld_s)
    gy_sm, gx_sm = np.gradient(ndi.gaussian_filter(Z_2k, 2.4 / dx), dx, dx)
    nsm_len = np.sqrt(1.0 + gx_sm * gx_sm + gy_sm * gy_sm)
    ndotl_sm = np.clip((-gx_sm * ld_s[0] - gy_sm * ld_s[1] + ld_s[2]) / nsm_len, 0.0, 1.0)
    illum_macro = np.clip(0.68 + 0.40 * ndotl_sm, 0.58, 1.08)
    delit = np.clip(proj / (illum_macro[..., None] ** 0.44), 0.0, 1.0).astype(np.float32)
    return delit, vis_w.astype(np.float32), vis_clean


def run_stage_clay():
    os.makedirs(WORK_DIR, exist_ok=True)
    cam_p = np.load(os.path.join(REFS_DIR, "hero_cam_params.npy"))
    hero_rgb = np.asarray(Image.open(os.path.join(REFS_DIR, "hero_sq.png")).convert("RGB"), dtype=np.float32) / 255.0
    fg_hero = np.load(os.path.join(WORK_DIR, "hero_fg_clean.npy"))

    print("1/4 Building V5 granite twin-spire 360-deg DEM (1024x1024)...")
    X, Y, Z0, dist_h, dist_v, taper = build_v5_granite_dem(1024)
    print(f"    Initial Z range: [{Z0.min():.2f}, {Z0.max():.2f}] mm")

    print("2/4 Optimizing skyline parity vs hero_sq.png...")
    Z_opt, mae = optimize_v5_skyline(X, Y, Z0, taper, cam_p, fg_hero, iters=18)
    # Lock exact Z_PEAK = Z_BASE + RELIEF_MM
    rel_opt = np.maximum(0.0, Z_opt - Z_BASE)
    if rel_opt.max() > 0:
        Z_opt = Z_BASE + rel_opt * (RELIEF_MM / float(rel_opt.max()))
    np.save(os.path.join(WORK_DIR, "Z_geom_v5.npy"), Z_opt.astype(np.float32))
    print(f"    Saved Z_geom_v5.npy: Z=[{Z_opt.min():.2f}, {Z_opt.max():.2f}] mm, skyline MAE={mae:.2f} px")

    print("3/4 Rendering 0-deg vs Hero & 6-view clay turnaround audits...")
    render_clay_audits_v5(Z_opt, cam_p, hero_rgb, fg_hero)

    print("4/4 Rendering 1024x1024 back (180-deg) clay studio template for Nano Banana edit...")
    back_clay_path = os.path.join(REFS_DIR, "clay_back_180.png")
    render_clay_turnaround(Z_opt, cam_p, back_clay_path, yaw_offset_deg=180.0, size=1024)
    front_clay_path = os.path.join(REFS_DIR, "clay_front_0.png")
    render_clay_turnaround(Z_opt, cam_p, front_clay_path, yaw_offset_deg=0.0, size=1024)
    print(f"[Stage Clay Done] -> {back_clay_path}")


def render_clay_audits_v5(Z, cam_p, hero_rgb, fg_hero):
    img0, any0, _, _, _, _, _, _, _, _ = ray_march_heightfield(Z, cam_p, out_w=512, out_h=512)
    h512 = np.asarray(Image.fromarray((hero_rgb * 255).astype(np.uint8)).resize((512, 512), Image.LANCZOS), dtype=np.float32) / 255.0
    ov = img0.copy()
    fg512 = ndi.zoom(fg_hero.astype(np.float32), 0.5, order=0) > 0.5
    rim = fg512 ^ ndi.binary_erosion(fg512, iterations=2)
    ov[rim] = np.array([1.0, 0.25, 0.2], dtype=np.float32)
    comp = np.concatenate([h512, img0, ov], axis=1)
    Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).save(os.path.join(WORK_DIR, "v5_clay_vs_hero.png"))

    views = []
    for ang in [0, 60, 120, 180, 240, 300]:
        im, _, _, _, _, _, _, _, _, _ = ray_march_heightfield(Z, cam_p, out_w=420, out_h=420, yaw_offset_deg=ang, sun_rel=True)
        pil_v = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(pil_v)
        dr.text((12, 12), f"{ang} deg", fill=(20, 25, 30))
        views.append(np.asarray(pil_v))
    grid = np.concatenate([np.concatenate(views[:3], axis=1), np.concatenate(views[3:], axis=1)], axis=0)
    Image.fromarray(grid).save(os.path.join(WORK_DIR, "v5_clay_6views.png"))


def run_stage_guide():
    os.makedirs(WORK_DIR, exist_ok=True)
    cam_p = np.load(os.path.join(REFS_DIR, "hero_cam_params.npy"))
    hero_rgb = np.asarray(Image.open(os.path.join(REFS_DIR, "hero_sq.png")).convert("RGB"), dtype=np.float32) / 255.0
    Z_1k = np.load(os.path.join(WORK_DIR, "Z_geom_v5.npy")).astype(np.float64)

    N2 = 2048
    Z_2k = ndi.zoom(Z_1k, N2 / Z_1k.shape[0], order=3)
    lin = np.linspace(-R, R, N2)
    X, Y = np.meshgrid(lin, lin)
    dist_h = hex_dist(X, Y)
    dist_v = vert_dist(X, Y)
    dx = 2.0 * R / (N2 - 1)

    hero_delit, vis_w, vis_clean = project_view_onto_dem(Z_2k, X, Y, dist_h, dist_v, dx, hero_rgb, cam_p, front_mode=True)

    gy, gx = np.gradient(Z_2k, dx, dx)
    nz = 1.0 / np.sqrt(1.0 + gx * gx + gy * gy)
    nx_n, ny_n = -gx * nz, -gy * nz
    slope_deg = np.degrees(np.arccos(np.clip(nz, 0.0, 1.0)))
    rel_z = np.maximum(0.0, Z_2k - Z_BASE)
    ndotl_top = np.clip(nx_n * (-0.40) + ny_n * 0.50 + nz * 0.77, 0.0, 1.0)
    elev_t = np.clip(rel_z / RELIEF_MM, 0.0, 1.0)
    clay_s = np.clip(0.36 + 0.50 * ndotl_top + 0.14 * elev_t, 0.0, 1.0)
    in_ter = smoothstep(ROAD_MARGIN, ROAD_MARGIN + 1.0, dist_h) * smoothstep(PAD_RADIUS, PAD_RADIUS + 1.0, dist_v)
    clay_rgb = np.stack([clay_s, clay_s, clay_s], axis=-1) * (0.38 + 0.62 * in_ter[..., None])

    z_sm = ndi.gaussian_filter(Z_2k, sigma=28.0)
    curv = Z_2k - z_sm
    snow_hint = smoothstep(8.0, 18.0, rel_z - 3.5 * curv) * (1.0 - smoothstep(37.0, 44.0, slope_deg)) * in_ter
    meadow_hint = (1.0 - smoothstep(3.2, 10.5, rel_z)) * (1.0 - smoothstep(28.0, 41.0, slope_deg)) * in_ter
    scree_hint = smoothstep(-0.2, -1.4, curv) * smoothstep(2.0, 11.0, rel_z) * (1.0 - smoothstep(28.0, 44.0, slope_deg)) * in_ter
    hint_col = np.array([0.48, 0.49, 0.50], dtype=np.float32)[None, None, :] * clay_s[..., None] * 1.40
    hint_col = hint_col * (1.0 - 0.75 * meadow_hint[..., None]) + (np.array([0.36, 0.42, 0.22]) * (0.75 + 0.50 * clay_s[..., None])) * (0.75 * meadow_hint[..., None])
    hint_col = hint_col * (1.0 - 0.70 * scree_hint[..., None]) + (np.array([0.74, 0.49, 0.32]) * (0.75 + 0.50 * clay_s[..., None])) * (0.70 * scree_hint[..., None])
    hint_col = hint_col * (1.0 - 0.85 * snow_hint[..., None]) + (np.array([0.93, 0.95, 0.98]) * (0.85 + 0.20 * clay_s[..., None])) * (0.85 * snow_hint[..., None])

    slate = np.array([0.14, 0.15, 0.17], dtype=np.float32)
    collar = 1.0 - smoothstep(ROAD_MARGIN, ROAD_MARGIN + 0.8, dist_h) * smoothstep(PAD_RADIUS, PAD_RADIUS + 0.8, dist_v)
    hybrid = np.clip(hero_delit * vis_w[..., None] + hint_col * (1.0 - vis_w[..., None]), 0.0, 1.0)
    hybrid = hybrid * (1.0 - collar[..., None]) + slate * collar[..., None]

    Image.fromarray(np.flipud((np.clip(clay_rgb, 0, 1) * 255).astype(np.uint8))).save(os.path.join(WORK_DIR, "clay_ortho_2k.png"))
    Image.fromarray(np.flipud((np.clip(hybrid, 0, 1) * 255).astype(np.uint8))).save(os.path.join(WORK_DIR, "ortho_hybrid_guide_2k.png"))
    print(f"[Guide Ready] front_vis_frac={vis_clean[dist_h >= 3.8].mean():.3f} -> clay_ortho_2k.png & ortho_hybrid_guide_2k.png")


def run_stage_fuse():
    os.makedirs(WORK_DIR, exist_ok=True)
    cam_p = np.load(os.path.join(REFS_DIR, "hero_cam_params.npy"))
    hero_rgb = np.asarray(Image.open(os.path.join(REFS_DIR, "hero_sq.png")).convert("RGB"), dtype=np.float32) / 255.0
    back_nb_path = os.path.join(REFS_DIR, "back_hero_nb.png")
    back_rgb = None
    if os.path.exists(back_nb_path):
        im_b = Image.open(back_nb_path).convert("RGB")
        if im_b.size != (1024, 1024):
            im_b = im_b.resize((1024, 1024), Image.LANCZOS)
        back_rgb = np.asarray(im_b, dtype=np.float32) / 255.0
        print(f"    Loaded 180-deg Nano Banana back view: {back_nb_path}")
    else:
        print("    [Warning] back_hero_nb.png not found -> synthesizing back purely from exemplar bank!")

    Z_1k = np.load(os.path.join(WORK_DIR, "Z_geom_v5.npy")).astype(np.float64)
    N2 = 2048
    Z_2k = ndi.zoom(Z_1k, N2 / Z_1k.shape[0], order=3)
    lin = np.linspace(-R, R, N2)
    X, Y = np.meshgrid(lin, lin)
    dist_h = hex_dist(X, Y)
    dist_v = vert_dist(X, Y)
    dx = 2.0 * R / (N2 - 1)

    front_delit, w_front, vc_f = project_view_onto_dem(Z_2k, X, Y, dist_h, dist_v, dx, hero_rgb, cam_p, vis_thresh=0.08, front_mode=True)
    if back_rgb is not None:
        cam_p_180 = compute_cam_p_180(cam_p)
        back_delit, w_back, vc_b = project_view_onto_dem(Z_2k, X, Y, dist_h, dist_v, dx, back_rgb, cam_p_180, vis_thresh=0.10, front_mode=False)
        # Ensure front dominates where both cameras overlap near crest edges
        w_back = w_back * np.clip(1.0 - 1.35 * w_front, 0.0, 1.0)
        # Mild color temperature normalization between back_delit & front_delit rock tones
        swatches = extract_hero_swatch_banks(hero_rgb, back_rgb)
    else:
        back_delit = np.zeros_like(front_delit)
        w_back = np.zeros_like(w_front)
        swatches = extract_hero_swatch_banks(hero_rgb, None)

    ex_360 = synthesize_slope_aligned_exemplar(X, Y, Z_2k, dist_h, dist_v, swatches)

    # Combine front_delit (w_front) + back_delit (w_back) + slope-aligned exemplar (w_ex) via normalized 2-band Laplacian pyramid
    w_views = np.clip(w_front + w_back, 0.0, 1.0)
    w_ex = np.clip(1.0 - w_views, 0.0, 1.0)
    wf = np.where(w_views > 1e-4, w_front / np.maximum(w_views, 1e-4), 0.5)
    view_comb = front_delit * wf[..., None] + back_delit * (1.0 - wf[..., None])

    v_bin = (w_views > 0.04).astype(np.float32)
    v_blur = ndi.gaussian_filter(v_bin, 8.0)
    v_num = ndi.gaussian_filter(view_comb * v_bin[..., None], (8.0, 8.0, 0.0))
    ex_low = ndi.gaussian_filter(ex_360, (8.0, 8.0, 0.0))
    v_low = np.where(v_blur[..., None] > 0.05, v_num / np.maximum(v_blur[..., None], 1e-4), ex_low)

    w_low = np.clip(ndi.gaussian_filter(w_views, sigma=6.5), 0.0, 1.0)[..., None]
    w_hi = w_views[..., None]
    fused_low = v_low * w_low + ex_low * (1.0 - w_low)
    fused_hi = (view_comb - v_low) * w_hi + (ex_360 - ex_low) * (1.0 - w_hi)
    albedo_2k = np.clip(fused_low + fused_hi, 0.0, 1.0)

    # Machined brushed gunmetal slate hex bezel + crisp bevel catchlight & recessed pad rings (0 mm Z change)
    rel_z = np.maximum(0.0, Z_2k - Z_BASE)
    on_border = np.maximum(
        1.0 - smoothstep(ROAD_MARGIN + 0.50, ROAD_MARGIN + 1.15, dist_h),
        1.0 - smoothstep(PAD_RADIUS + 0.40, PAD_RADIUS + 1.00, dist_v),
    )
    flat_mask = 1.0 - smoothstep(0.14, 0.65, rel_z)
    collar = np.clip(on_border * flat_mask, 0.0, 1.0)
    f_bez = ridge_fbm(X, Y, [0.4, 1.5], [0.5, 0.5], seed=888)
    slate_col = np.array([0.165, 0.174, 0.188], dtype=np.float32)[None, None, :] * (0.95 + 0.10 * f_bez[..., None])
    rim_hi = np.exp(-0.5 * ((dist_h - 0.90) / 0.30) ** 2) * 0.045
    curb_sh = np.exp(-0.5 * ((dist_h - 3.60) / 0.42) ** 2) * 0.055
    pad_groove = np.exp(-0.5 * ((dist_v - 8.2) / 0.32) ** 2) * 0.042
    pad_hi = np.exp(-0.5 * ((dist_v - 9.0) / 0.28) ** 2) * 0.030
    slate_col = np.clip(slate_col + (rim_hi - curb_sh - pad_groove + pad_hi)[..., None], 0.06, 0.30)
    albedo_2k = albedo_2k * (1.0 - collar[..., None]) + slate_col * collar[..., None]

    # Export canonical 2K .npy heightmap & 4K crisp PNG albedo
    hm_norm = np.clip((Z_2k - Z_BASE) / RELIEF_MM, 0.0, 1.0) * 255.0
    np.save(os.path.join(WORK_DIR, "montana_granito_v5_hm_2k.npy"), hm_norm.astype(np.float32))
    Image.fromarray(np.flipud(np.clip(hm_norm * 257.0, 0, 65535).astype(np.uint16)), mode="I;16").save(
        os.path.join(WORK_DIR, "montana_granito_v5_hm_16bit.png")
    )

    alb_4k = Image.fromarray(np.flipud((np.clip(albedo_2k, 0, 1) * 255).astype(np.uint8))).resize((4096, 4096), Image.LANCZOS)
    alb_4k_path = os.path.join(WORK_DIR, "montana_granito_v5_albedo_4k.png")
    alb_4k.save(alb_4k_path)

    # Render textured parity & 6-view turnaround audits
    img0, _, _, _, _, _, _, _, _, _ = ray_march_heightfield(Z_2k, cam_p, out_w=512, out_h=512, tex_rgb=albedo_2k, sun_rel=True)
    h512 = np.asarray(Image.fromarray((hero_rgb * 255).astype(np.uint8)).resize((512, 512), Image.LANCZOS), dtype=np.float32) / 255.0
    comp = np.concatenate([h512, img0], axis=1)
    Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).save(os.path.join(WORK_DIR, "v5_textured_vs_hero.png"))

    views = []
    for ang in [0, 60, 120, 180, 240, 300]:
        im, _, _, _, _, _, _, _, _, _ = ray_march_heightfield(Z_2k, cam_p, out_w=420, out_h=420, tex_rgb=albedo_2k, yaw_offset_deg=ang, sun_rel=True)
        pil_v = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(pil_v)
        dr.text((12, 12), f"{ang} deg", fill=(20, 25, 30))
        views.append(np.asarray(pil_v))
    grid = np.concatenate([np.concatenate(views[:3], axis=1), np.concatenate(views[3:], axis=1)], axis=0)
    Image.fromarray(grid).save(os.path.join(WORK_DIR, "v5_textured_6views.png"))

    frac_f = float(vc_f[dist_h >= 3.8].mean())
    frac_b = float(vc_b[dist_h >= 3.8].mean()) if back_rgb is not None else 0.0
    print(f"[Stage Fuse Done] front_vis={frac_f:.3f}, back_vis={frac_b:.3f} -> {alb_4k_path}")


def main():
    p = argparse.ArgumentParser(description="Montaña Granito V5 Sculptor")
    p.add_argument("--stage", choices=["clay", "guide", "fuse"], required=True)
    args = p.parse_args()
    if args.stage == "clay":
        run_stage_clay()
    elif args.stage == "guide":
        run_stage_guide()
    else:
        run_stage_fuse()


if __name__ == "__main__":
    main()
