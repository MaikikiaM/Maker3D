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
      - Camera azimuth ~30 deg (looking from (+168,-291) toward (0,0)):
        screen-left=(-X,-Y), screen-right=(+X,+Y), back=(-X,+Y), front=(+X,-Y)
      - Twin towering jagged granite spires: (-15.0, +12.0, ~45 mm) & (-2.0, +12.0, ~40 mm)
      - Broad solid alpine massif core + carved U-shaped glacial cirque (+4, 0) visible from 0 deg!
      - Lower front-left granite bastion (-12,-10 -> -4,-19, ~18 mm) & right gorge wall (+22,-4 -> +31,-13, ~17 mm)
      - S-curve terracotta gorge exiting (+6,-3) -> (+24,-30)
      - Left angular talus scree chute (-26,-8 -> -23,-23) & crisp 360-deg north/back massif.
    """
    lin = np.linspace(-R, R, N)
    X, Y = np.meshgrid(lin, lin)
    dist_h = hex_dist(X, Y)
    dist_v = vert_dist(X, Y)
    taper = smoothstep(ROAD_MARGIN + 0.35, ROAD_MARGIN + 6.4, dist_h) * smoothstep(PAD_RADIUS + 0.30, PAD_RADIUS + 6.0, dist_v)

    # 1. Organic alpine meadow & foothill base (zero flat slab)
    hill_n = smooth_fbm(X, Y, [0.09, 0.22, 0.48], [0.48, 0.34, 0.18], seed=533)
    in_hex = smoothstep(ROAD_MARGIN + 0.2, ROAD_MARGIN + 4.8, dist_h) * smoothstep(PAD_RADIUS + 0.2, PAD_RADIUS + 4.8, dist_v)
    Z_floor = in_hex * (0.95 + 1.55 * hill_n)

    # 2. Broad pyramidal massif core (gives solid rock cohesion like V4, zero isolated thin fins!)
    r_core = np.hypot((X + 8.0) / 1.08, (Y - 10.5) / 0.96)
    Z_body = 34.5 * np.power(np.maximum(0.0, 1.0 - r_core / 41.0), 1.38)
    Z_body = smax(Z_body, 24.0 * np.power(np.maximum(0.0, 1.0 - np.hypot(X + 22.0, Y - 1.0) / 28.0), 1.40), k=4.5)
    Z_body = smax(Z_body, 23.5 * np.power(np.maximum(0.0, 1.0 - np.hypot(X - 12.0, Y - 8.5) / 29.0), 1.40), k=4.5)
    Z_body = smax(Z_body, 22.5 * np.power(np.maximum(0.0, 1.0 - np.hypot(X + 7.0, Y - 24.5) / 30.0), 1.42), k=4.5)
    # Cohesive lower foothill shelves under front-left & front-right bastions (keeps them grounded, not floating fins!)
    Z_body = smax(Z_body, 14.5 * np.power(np.maximum(0.0, 1.0 - np.hypot(X + 10.0, Y + 13.5) / 21.0), 1.45), k=3.5)
    Z_body = smax(Z_body, 14.0 * np.power(np.maximum(0.0, 1.0 - np.hypot(X - 24.5, Y + 8.5) / 20.0), 1.45), k=3.5)
    base_core = smax(Z_floor, Z_body, k=2.2)

    # Carve wide open central snow cirque (+4.5, +1.5) & secondary north glacial valleys BEFORE placing sharp arêtes
    cirque_center = 11.5 * np.exp(-0.5 * (((X - 4.5) / 9.8) ** 2 + ((Y - 1.5) / 8.8) ** 2))
    cirque_left_pocket = 5.0 * np.exp(-0.5 * (((X + 18.5) / 4.8) ** 2 + ((Y + 14.0) / 5.2) ** 2))
    cirque_north_w = 5.8 * np.exp(-0.5 * (((X + 16.5) / 7.8) ** 2 + ((Y - 26.5) / 8.8) ** 2))
    cirque_north_e = 5.8 * np.exp(-0.5 * (((X - 9.0) / 7.8) ** 2 + ((Y - 27.0) / 8.8) ** 2))
    base_core = smax(Z_floor + 1.8, base_core - (cirque_center + cirque_left_pocket + cirque_north_w + cirque_north_e), k=2.4)

    # 3. Broad interlocking prismatic granite arêtes (V4-style w=10..15mm, rib_amp=0.04 -> solid rocky massifs!)
    spires = [
        # Main Left Spire (-15.0, +12.0) & jagged shoulders
        (-15.0, 12.0, 45.2, -20.5, 6.5, 32.0, 15.0, 12.0, 0.90, 11, 0.04),
        (-20.5, 6.5, 32.0, -27.5, -2.5, 22.0, 12.5, 9.8, 0.91, 12, 0.04),
        (-27.5, -2.5, 22.0, -37.0, -11.5, 9.5, 9.8, 6.4, 0.96, 13, 0.03),
        (-15.0, 12.0, 44.5, -9.2, 12.2, 34.5, 14.0, 11.5, 0.90, 14, 0.04),
        # Main Right Twin Spire (-2.0, +12.0) & right alpine shoulders (matches skyline right crags!)
        (-2.0, 12.0, 40.6, -7.8, 12.0, 32.0, 13.8, 11.2, 0.90, 15, 0.04),
        (-2.0, 12.0, 40.4, 8.0, 16.0, 28.5, 13.5, 11.2, 0.90, 16, 0.04),
        (8.0, 16.0, 28.5, 18.0, 11.0, 24.5, 11.8, 9.8, 0.90, 17, 0.04),
        (18.0, 11.0, 24.5, 27.5, 3.5, 16.0, 9.8, 7.6, 0.93, 18, 0.03),
        (27.5, 3.5, 16.0, 35.5, -4.5, 6.5, 7.6, 5.2, 0.98, 19, 0.03),
        # Mid-right secondary jagged crag behind snow bowl
        (10.5, 8.5, 22.0, 19.5, 3.5, 20.5, 10.2, 8.4, 0.90, 20, 0.04),

        # Front-Left Bastion: shifted slightly down-left so central snow cirque opens wide!
        (-15.5, -10.5, 18.5, -10.5, -16.5, 18.0, 10.2, 9.0, 0.89, 21, 0.04),
        (-10.5, -16.5, 18.0, -3.5, -22.5, 14.0, 9.0, 7.2, 0.91, 22, 0.04),
        (-3.5, -22.5, 14.0, 4.5, -27.5, 7.2, 7.2, 5.2, 0.96, 23, 0.03),
        (-15.5, -10.5, 17.8, -21.5, -6.5, 17.2, 9.5, 8.0, 0.91, 24, 0.04),
        (-15.5, -14.0, 16.2, -18.5, -20.5, 10.2, 8.5, 6.4, 0.91, 29, 0.04),

        # Front-Right Bastion & Gorge Framing (pulled slightly inward so right corner isn't flat!)
        (18.5, -1.0, 19.5, 24.0, -7.5, 18.2, 10.5, 9.0, 0.89, 25, 0.04),
        (24.0, -7.5, 18.2, 30.5, -15.5, 11.8, 9.0, 6.8, 0.93, 26, 0.04),
        (30.5, -15.5, 11.8, 35.5, -23.0, 5.2, 6.8, 4.6, 0.98, 27, 0.03),
        (20.5, -7.5, 16.2, 26.5, -14.5, 11.5, 8.2, 6.2, 0.91, 30, 0.04),
        # Low inner-left canyon curb separating lower meadow from terracotta path
        (9.5, -17.5, 10.0, 16.0, -23.0, 7.2, 7.0, 5.2, 0.93, 28, 0.03),

        # 360-deg North / Back Interlocking Granite Massif (broad connected spurs, full 240/120 balance!)
        (-15.0, 12.0, 44.0, -17.5, 24.5, 28.5, 14.5, 11.2, 0.91, 31, 0.04),
        (-17.5, 24.5, 28.5, -23.0, 36.0, 13.5, 11.2, 7.8, 0.94, 32, 0.03),
        (-23.0, 36.0, 13.5, -27.0, 43.0, 5.5, 7.8, 5.0, 1.00, 33, 0.03),
        (-2.0, 12.0, 39.8, -3.0, 25.0, 26.5, 14.0, 11.0, 0.91, 34, 0.04),
        (-3.0, 25.0, 26.5, -1.5, 36.5, 12.8, 11.0, 7.6, 0.94, 35, 0.03),
        (-1.5, 36.5, 12.8, 2.0, 43.0, 5.0, 7.6, 4.8, 1.00, 36, 0.03),
        (8.0, 16.0, 28.0, 14.5, 27.5, 19.8, 11.8, 9.0, 0.92, 37, 0.04),
        (14.5, 27.5, 19.8, 22.5, 37.0, 8.8, 9.0, 6.0, 0.96, 38, 0.03),
        (-24.5, 9.5, 24.0, -34.0, 18.0, 14.0, 11.5, 8.0, 0.93, 39, 0.03),
        (-34.0, 18.0, 14.0, -41.0, 23.5, 5.8, 8.0, 5.2, 1.00, 40, 0.03),
        # North-west & north-east connecting cross-spurs (eliminates empty apron in 240 deg!)
        (-10.5, 21.5, 25.5, -11.0, 34.5, 13.2, 10.5, 7.4, 0.93, 41, 0.03),
        (4.5, 22.5, 22.0, 9.0, 34.5, 11.2, 10.0, 7.0, 0.94, 42, 0.03),
        (18.0, 11.0, 23.5, 26.5, 19.0, 15.0, 10.0, 7.2, 0.92, 43, 0.03),
        (26.5, 19.0, 15.0, 35.5, 24.5, 6.0, 7.2, 4.8, 0.98, 44, 0.03),
    ]
    Z_ridges = np.zeros_like(Z_floor)
    for s in spires:
        Z_ridges = smax(Z_ridges, faceted_ridge(X, Y, *s), k=1.8)

    # Prismatic chiseled rock outcrops (vertical granite pillar clusters)
    outcrops = [
        (-15.2, 12.2, 45.4, 7.2, 6.2, 22.0, 0.88),
        (-2.2, 12.1, 40.6, 6.8, 5.8, -18.0, 0.88),
        (-12.2, -10.5, 19.0, 6.5, 5.6, 28.0, 0.86),
        (-7.5, -16.0, 17.5, 5.8, 5.0, 16.0, 0.86),
        (24.0, -8.5, 16.8, 6.2, 5.4, -36.0, 0.86),
        (13.5, -19.5, 10.4, 5.2, 4.5, -28.0, 0.86),
        (-17.5, 25.0, 27.5, 6.8, 5.8, 12.0, 0.88),
        (-3.5, 26.0, 26.0, 6.5, 5.5, -14.0, 0.88),
        (17.5, 10.5, 22.8, 5.8, 5.0, -24.0, 0.88),
    ]
    Z_out = np.zeros_like(Z_floor)
    for oc in outcrops:
        Z_out = smax(Z_out, rock_outcrop(X, Y, *oc), k=1.4)

    # Left angular talus scree chute (-26, -6) -> (-22, -24)
    scree_spurs = [
        (-25.5, -5.0, 16.5, -31.5, -15.5, 8.8, 7.8, 5.2, 0.90, 61, 0.04),
        (-21.0, -12.0, 12.5, -27.5, -22.5, 5.8, 7.2, 4.6, 0.90, 62, 0.04),
    ]
    for sr in scree_spurs:
        Z_ridges = smax(Z_ridges, faceted_ridge(X, Y, *sr), k=1.6)

    Z_m = smax(base_core, Z_ridges, k=2.2)
    Z_m = smax(Z_m, Z_out, k=1.4)

    # 4. Crisp multi-directional granite dihedral flutes + high-frequency rock facets across ALL rock slopes!
    in_snow_bowl = (
        np.exp(-0.5 * (((X - 3.5) / 7.0) ** 2 + ((Y - 1.0) / 6.8) ** 2))
        + 0.75 * np.exp(-0.5 * (((X + 18.5) / 4.0) ** 2 + ((Y + 14.0) / 4.4) ** 2))
        + 0.70 * np.exp(-0.5 * (((X + 16.5) / 5.5) ** 2 + ((Y - 26.5) / 6.5) ** 2))
        + 0.70 * np.exp(-0.5 * (((X - 9.0) / 5.5) ** 2 + ((Y - 27.0) / 6.5) ** 2))
    )
    flute_damp = np.clip(1.0 - 0.74 * in_snow_bowl, 0.22, 1.0)
    fl1 = ridge_fbm(X, Y, [0.34, 0.82, 1.85, 3.8], [0.42, 0.28, 0.18, 0.12], seed=505)
    fl2 = ridge_fbm(X * 0.85 + Y * 0.35, -X * 0.35 + Y * 0.85, [0.52, 1.38, 2.95], [0.52, 0.32, 0.16], seed=509)
    Z_m = Z_m + smoothstep(2.4, 14.0, Z_m) * flute_damp * (2.15 * (fl1 - 0.44) + 1.60 * (fl2 - 0.45))

    # 5. Carve open S-curve terracotta canyon exiting front-right (+5,-2) -> (+24,-30)
    gorge_pts = [
        (4.5, 1.0, 6.2, 3.2),
        (10.5, -6.5, 5.2, 4.8),
        (16.5, -14.5, 4.8, 5.6),
        (21.5, -22.5, 5.0, 5.2),
        (25.5, -31.0, 5.6, 4.4),
    ]
    carve_g = river_gorge_depth(X, Y, gorge_pts, sharp_edge=0.95)
    Z_m = np.maximum(Z_floor * 0.85 + 0.6, Z_m - carve_g)

    # Subtle north valley notch (for 180-deg interest without deep artificial slot)
    north_gorge = [
        (-9.5, 22.0, 4.2, 2.8),
        (-8.5, 31.5, 4.8, 4.2),
        (-6.5, 41.0, 5.4, 3.6),
    ]
    Z_m = np.maximum(Z_floor * 0.85 + 0.6, Z_m - river_gorge_depth(X, Y, north_gorge, sharp_edge=0.92))

    # 6. Small angular rocks on left scree chute & foreground meadow
    rng_b = np.random.default_rng(577)
    for _ in range(22):
        bx = float(rng_b.uniform(-34.0, 34.0))
        by = float(rng_b.uniform(-34.0, 34.0))
        d_c = math.hypot(bx, by)
        if 22.0 < d_c < 41.5 and (by < -10.0 or bx < -16.0):
            bh = float(rng_b.uniform(0.9, 1.8))
            brx = float(rng_b.uniform(1.1, 2.0))
            bry = float(rng_b.uniform(0.9, 1.8))
            bang = float(rng_b.uniform(0.0, 180.0))
            Z_m = smax(Z_m, rock_outcrop(X, Y, bx, by, bh, brx, bry, ang_deg=bang, sharp=0.80), k=0.45)

    # Normalize relief strictly inside [44.8..45.2 mm] above Z_BASE
    rel_taper = np.maximum(0.0, Z_m) * taper
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


def compute_cam_p_angle(cam_p, yaw_offset_deg):
    """Exact camera parameters at yaw_offset_deg matching ray_march_heightfield(yaw_offset_deg=...)."""
    p = cam_p.copy()
    p[0] = cam_p[0] + yaw_offset_deg
    cyaw = math.cos(math.radians(yaw_offset_deg))
    syaw = math.sin(math.radians(yaw_offset_deg))
    tx, ty = cam_p[3], cam_p[4]
    p[3] = tx * cyaw - ty * syaw
    p[4] = tx * syaw + ty * cyaw
    return p


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


def align_nb_view_to_clay(nb_rgb, target_any_hit):
    """Align a Nano Banana 1024x1024 studio render to the exact 3D ray-marched silhouette (scale + shift)."""
    lum = 0.2126 * nb_rgb[..., 0] + 0.7152 * nb_rgb[..., 1] + 0.0722 * nb_rgb[..., 2]
    sat = np.max(nb_rgb, axis=-1) - np.min(nb_rgb, axis=-1)
    fg = (lum < 0.84) | (sat > 0.045)
    fg = ndi.binary_opening(fg, iterations=1)
    lbl, n_l = ndi.label(fg)
    if n_l > 0:
        cnts = np.bincount(lbl.ravel())
        cnts[0] = 0
        fg = lbl == int(np.argmax(cnts))
    if not np.any(fg) or not np.any(target_any_hit):
        return nb_rgb, 0.0, 0.0

    iou_before = float(np.sum(fg & target_any_hit)) / max(float(np.sum(fg | target_any_hit)), 1.0)
    ty, tx = np.where(target_any_hit)
    fy, fx = np.where(fg)
    # Match bbox + slight margin cleanly
    th = max(float(ty.max() - ty.min()), 1.0)
    tw = max(float(tx.max() - tx.min()), 1.0)
    fh = max(float(fy.max() - fy.min()), 1.0)
    fw = max(float(fx.max() - fx.min()), 1.0)
    sy = th / fh
    sx = tw / fw
    # Keep near-isotropic scale to prevent aspect distortion
    sc = 0.5 * (sx + sy)
    cy_t, cx_t = 0.5 * (ty.min() + ty.max()), 0.5 * (tx.min() + tx.max())
    cy_f, cx_f = 0.5 * (fy.min() + fy.max()), 0.5 * (fx.min() + fx.max())

    yy, xx = np.mgrid[0:1024, 0:1024].astype(np.float32)
    src_y = (yy - cy_t) / sc + cy_f
    src_x = (xx - cx_t) / sc + cx_f
    aligned = np.stack(
        [ndi.map_coordinates(nb_rgb[..., c], [src_y, src_x], order=1, mode="nearest") for c in range(3)],
        axis=-1,
    ).astype(np.float32)
    fg_al = ndi.map_coordinates(fg.astype(np.float32), [src_y, src_x], order=0, mode="constant", cval=0.0) > 0.5
    iou_after = float(np.sum(fg_al & target_any_hit)) / max(float(np.sum(fg_al | target_any_hit)), 1.0)
    return aligned, iou_before, iou_after


def project_view_clean(Z_2k, X, Y, dist_h, dist_v, dx, view_rgb, view_cam_p, min_ndotv=0.22, erode_px=3, is_hero=False):
    """
    Project a 1024x1024 view onto Z_2k with strict silhouette-eroded masking (zero studio/bezel halos,
    zero snow-deletion hacks) returning (delit, vis_w, ndotv, vis_clean).
    """
    _, any_hit_1k, hit_ter_1k, hx, hy, hz, cam_xyz, nx_g, ny_g, nz_g = ray_march_heightfield(
        Z_2k, view_cam_p, out_w=1024, out_h=1024
    )
    if not is_hero:
        view_rgb, iou_b, iou_a = align_nb_view_to_clay(view_rgb, any_hit_1k)
        print(f"    Aligned view to clay silhouette: IoU {iou_b:.4f} -> {iou_a:.4f}")

    # Erode 2D terrain hit silhouette so border pixels never sample background sky or plinth bezel!
    hit_ter_safe = ndi.binary_erosion(hit_ter_1k, iterations=erode_px)

    u_all, v_all, _ = project_pts_vec(view_cam_p, X, Y, Z_2k, 1024.0, 1024.0)
    cam_dist_1k = np.sqrt((hx - cam_xyz[0]) ** 2 + (hy - cam_xyz[1]) ** 2 + (hz - cam_xyz[2]) ** 2)
    dist_grid = np.sqrt((X - cam_xyz[0]) ** 2 + (Y - cam_xyz[1]) ** 2 + (Z_2k - cam_xyz[2]) ** 2)
    ui = np.clip(np.round(u_all), 0, 1023).astype(np.int32)
    vi = np.clip(np.round(v_all), 0, 1023).astype(np.int32)

    vdir = cam_xyz[None, None, :] - np.stack([X, Y, Z_2k], axis=-1)
    vdir /= np.linalg.norm(vdir, axis=-1, keepdims=True)
    ndotv = nx_g * vdir[..., 0] + ny_g * vdir[..., 1] + nz_g * vdir[..., 2]

    # Guard outer rim against dark slate plinth pixels & low-altitude studio sky bleed (never touching high snow!)
    lum_v = 0.2126 * view_rgb[..., 0] + 0.7152 * view_rgb[..., 1] + 0.0722 * view_rgb[..., 2]
    sat_v = np.max(view_rgb, axis=-1) - np.min(view_rgb, axis=-1)
    rel_z_m = np.maximum(0.0, Z_2k - Z_BASE)
    warm_chroma = view_rgb[vi, ui, 0] + view_rgb[vi, ui, 1] - 2.0 * view_rgb[vi, ui, 2]
    is_slate = (warm_chroma < 0.05) & (lum_v[vi, ui] < 0.50) & (sat_v[vi, ui] < 0.11) & (rel_z_m < 4.5) & ((dist_h < 12.0) | (dist_v < 15.5))
    is_low_sky = (lum_v[vi, ui] > 0.66) & (sat_v[vi, ui] < 0.06) & (rel_z_m < 4.8) & ((not is_hero) or (dist_h < 9.0))

    vis_raw = (
        hit_ter_safe[vi, ui]
        & (dist_grid <= cam_dist_1k[vi, ui] + 0.35)
        & (dist_h >= 4.5)
        & (dist_v >= 10.8)
        & (ndotv > min_ndotv)
        & (~is_slate)
        & (~is_low_sky)
    )
    vis_er = ndi.binary_erosion(vis_raw, iterations=2)
    lbl_v, n_v = ndi.label(vis_er)
    if n_v > 0:
        cnts = np.bincount(lbl_v.ravel())
        keep_ids = np.where((cnts >= 100) & (np.arange(len(cnts)) > 0))[0]
        vis_clean = np.isin(lbl_v, keep_ids)
    else:
        vis_clean = vis_er

    vis_w = ndi.gaussian_filter(vis_clean.astype(np.float32), sigma=2.8)
    vis_w = np.clip((vis_w - 0.08) / 0.84, 0.0, 1.0)
    vis_w *= smoothstep(4.2, 6.2, dist_h) * smoothstep(10.5, 11.8, dist_v) * smoothstep(min_ndotv, min_ndotv + 0.24, ndotv)

    fx = np.clip(u_all, 0, 1022.99)
    fy = np.clip(v_all, 0, 1022.99)
    x0 = fx.astype(np.int32)
    y0 = fy.astype(np.int32)
    tx_f = (fx - x0)[..., None]
    ty_f = (fy - y0)[..., None]
    proj = (
        view_rgb[y0, x0] * (1 - tx_f) * (1 - ty_f)
        + view_rgb[y0, x0 + 1] * tx_f * (1 - ty_f)
        + view_rgb[y0 + 1, x0] * (1 - tx_f) * ty_f
        + view_rgb[y0 + 1, x0 + 1] * tx_f * ty_f
    )

    # Gentle camera-relative de-lighting matching ray_march_heightfield(sun_rel=True)
    az_r = math.radians(view_cam_p[0])
    el_r = math.radians(view_cam_p[1])
    fwd = np.array([-math.cos(el_r) * math.sin(az_r), math.cos(el_r) * math.cos(az_r), -math.sin(el_r)])
    right_r = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right_r /= np.linalg.norm(right_r)
    up_r = np.cross(right_r, fwd)
    ld_s = right_r * 0.65 + up_r * 0.10 - fwd * 0.45 + np.array([0.0, 0.0, 0.58])
    ld_s /= np.linalg.norm(ld_s)
    gy_sm, gx_sm = np.gradient(ndi.gaussian_filter(Z_2k, 2.4 / dx), dx, dx)
    nsm_len = np.sqrt(1.0 + gx_sm * gx_sm + gy_sm * gy_sm)
    ndotl_sm = np.clip((-gx_sm * ld_s[0] - gy_sm * ld_s[1] + ld_s[2]) / nsm_len, 0.0, 1.0)
    illum_macro = np.clip(0.70 + 0.36 * ndotl_sm, 0.62, 1.06)
    delit = np.clip(proj / (illum_macro[..., None] ** 0.36), 0.0, 1.0).astype(np.float32)
    return delit, vis_w.astype(np.float32), np.clip(ndotv, 0.0, 1.0).astype(np.float32), vis_clean


def render_partial_view(Z_2k, cam_p, cur_albedo, cur_w, out_path, yaw_offset_deg, size=1024):
    """Render 1024x1024 guide at yaw_offset_deg showing painted regions + soft neutral gray clay on unpainted terrain + gunmetal plinth rim."""
    img_clay, any_hit, hit_terrain, hx, hy, _, _, nx_g, ny_g, nz_g = ray_march_heightfield(
        Z_2k, cam_p, out_w=size, out_h=size, tex_rgb=None, yaw_offset_deg=yaw_offset_deg, sun_rel=True
    )
    out = np.full((size, size, 3), np.array([0.895, 0.902, 0.910], dtype=np.float32), dtype=np.float32)
    plinth = any_hit & ~hit_terrain
    out[plinth] = np.array([0.16, 0.175, 0.195], dtype=np.float32)

    if np.any(hit_terrain):
        hxh = hx[hit_terrain]
        hyh = hy[hit_terrain]
        dh = hex_dist(hxh, hyh)
        dv = vert_dist(hxh, hyh)
        is_rim = (dh < ROAD_MARGIN + 0.15) | (dv < PAD_RADIUS + 0.15)

        th, tw = cur_albedo.shape[:2]
        t_span = (tw - 1) / (2.0 * R)
        tix = np.clip(np.round((hxh + R) * t_span), 0, tw - 1).astype(np.int32)
        tiy = np.clip(np.round((hyh + R) * t_span), 0, th - 1).astype(np.int32)
        pw = np.clip(cur_w[tiy, tix], 0.0, 1.0)[:, None]
        # Soft studio light so clay shadows are gentle and Nano Banana never bakes dark cast shadows
        raw_ndotl = np.clip((img_clay[hit_terrain, 0] / 0.84 - 0.34) / 0.72, 0.0, 1.0)
        painted_lit = np.clip(cur_albedo[tiy, tix] * (0.84 + 0.18 * raw_ndotl)[:, None], 0.0, 1.0)
        clay_px = np.full((len(hxh), 3), 0.76, dtype=np.float32) * (0.62 + 0.42 * raw_ndotl)[:, None]
        ter_col = painted_lit * pw + clay_px * (1.0 - pw)
        # Render outer road collar & 6 pads in crisp dark charcoal gunmetal matching hero_reference.jpg
        gunmetal = np.array([0.20, 0.22, 0.24], dtype=np.float32) * (0.88 + 0.16 * raw_ndotl)[:, None]
        ter_col[is_rim] = gunmetal[is_rim]
        out[hit_terrain] = ter_col

    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(out_path)
    return out_path


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


def extract_v5_swatch_banks(hero_rgb, back_rgb=None):
    """Extract verified pure-material swatches directly from montana_granito_v5 refs (zero background bleed!)."""
    def prep_swatch(src_rgb, u0, u1, v0, v1, target_sz=256, sharpen=True):
        crop = src_rgb[v0:v1, u0:u1].astype(np.float32)
        pil = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8)).resize((target_sz, target_sz), Image.LANCZOS)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        low = ndi.gaussian_filter(arr, (48.0, 48.0, 0.0))
        mean_col = np.mean(arr, axis=(0, 1), keepdims=True)
        balanced = np.clip(arr - 0.50 * (low - mean_col), 0.0, 1.0)
        if not sharpen:
            return ndi.gaussian_filter(balanced, (2.2, 2.2, 0.0))
        b2 = ndi.gaussian_filter(balanced, (1.4, 1.4, 0.0))
        return np.clip(balanced + 0.30 * (balanced - b2), 0.0, 1.0)

    granite_swatches = [
        prep_swatch(hero_rgb, 452, 488, 345, 385, 256),  # Cool light-grey upper peak granite facets
        prep_swatch(hero_rgb, 445, 485, 505, 548, 256),  # Deep charcoal-slate mid bastion facets
        prep_swatch(hero_rgb, 625, 668, 515, 558, 256),  # Right bastion cold slate-grey stone
    ]
    if back_rgb is not None:
        granite_swatches.append(prep_swatch(back_rgb, 438, 478, 455, 505, 256))
        granite_swatches.append(prep_swatch(back_rgb, 542, 585, 455, 505, 256))

    scree_swatches = [
        prep_swatch(hero_rgb, 315, 370, 515, 555, 256),  # Angular grey-buff talus scree rubble
        prep_swatch(hero_rgb, 558, 602, 578, 622, 256),  # Warm dusty gorge stone transition
        prep_swatch(hero_rgb, 542, 588, 618, 650, 256),  # Terracotta earth path floor
    ]
    meadow_swatches = [
        prep_swatch(hero_rgb, 295, 355, 578, 612, 256),  # Left olive-moss alpine meadow
        prep_swatch(hero_rgb, 668, 722, 568, 602, 256),  # Right olive-moss alpine turf
        prep_swatch(hero_rgb, 395, 455, 608, 638, 256),  # Foreground olive-green meadow with pebbles
    ]
    # Pure solid alpine glacial snow without dark rock corners
    snow_c = np.asarray(
        Image.fromarray((np.clip(hero_rgb[452:484, 536:568], 0, 1) * 255).astype(np.uint8)).resize((256, 256), Image.LANCZOS),
        dtype=np.float32,
    ) / 255.0
    snow_c = np.maximum(snow_c, 0.89)
    snow_swatches = [ndi.gaussian_filter(snow_c, (2.0, 2.0, 0.0))]
    return {
        "granite": granite_swatches,
        "scree": scree_swatches,
        "meadow": meadow_swatches,
        "snow": snow_swatches,
    }


def synthesize_v5_exemplar_360(X, Y, Z, dist_h, dist_v, swatches):
    """
    Seam-free rigid Voronoi slope-aligned texture synthesis tuned for cold fractured granite:
      - Wider 2.4mm Voronoi boundary blending eliminates circular patch rims
      - Side flanks stay 100% crisp fractured granite (never blanketed in white snow dots!)
      - Protected snow ONLY settles inside gentle (<32 deg) elevated (>10 mm) concave pockets.
    """
    H, W = Z.shape
    dx = 2.0 * R / (W - 1)
    gy_s, gx_s = np.gradient(ndi.gaussian_filter(Z, 2.6 / dx), dx, dx)
    gmag = np.hypot(gx_s, gy_s) + 1e-6
    dx_dn_grid = -gx_s / gmag
    dy_dn_grid = -gy_s / gmag

    gy, gx = np.gradient(ndi.gaussian_filter(Z, 0.35 / dx), dx, dx)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    rel_z = np.maximum(0.0, Z - Z_BASE)
    z_sm = ndi.gaussian_filter(Z, sigma=max(2.6 / dx, 4.0))
    curv = Z - z_sm
    cav = Z - ndi.gaussian_filter(Z, sigma=max(0.75 / dx, 2.0))

    def cell_rand(ix, iy, salt):
        h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ np.int64(salt * 83492791)
        h = ((h * 2654435761) >> 8) & 0xFFFF
        return h.astype(np.float32) / 65535.0

    def quilt_bank(bank, cell_mm=9.5, scale_mm=11.0, align_slope=True, seed=10):
        gx_c = np.floor(X / cell_mm).astype(np.int32)
        gy_c = np.floor(Y / cell_mm).astype(np.int32)
        nb = len(bank)
        sz = bank[0].shape[0]
        d1 = np.full((H, W), 1e9, dtype=np.float32)
        d2 = np.full((H, W), 1e9, dtype=np.float32)
        col1 = np.zeros((H, W, 3), dtype=np.float32)
        col2 = np.zeros((H, W, 3), dtype=np.float32)

        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                cx_i = gx_c + ox
                cy_i = gy_c + oy
                jx = (cx_i + 0.18 + 0.64 * cell_rand(cx_i, cy_i, seed + 1)) * cell_mm
                jy = (cy_i + 0.18 + 0.64 * cell_rand(cx_i, cy_i, seed + 2)) * cell_mm
                rx = X - jx
                ry = Y - jy
                dist = np.hypot(rx, ry).astype(np.float32)

                if align_slope:
                    si_x = np.clip(np.round((jx + R) / dx).astype(np.int32), 0, W - 1)
                    si_y = np.clip(np.round((jy + R) / dx).astype(np.int32), 0, H - 1)
                    sdx_dn = dx_dn_grid[si_y, si_x]
                    sdy_dn = dy_dn_grid[si_y, si_x]
                    jit_a = (cell_rand(cx_i, cy_i, seed + 5) - 0.5) * 0.55
                    ca, sa = np.cos(jit_a), np.sin(jit_a)
                    jx_dn = sdx_dn * ca - sdy_dn * sa
                    jy_dn = sdx_dn * sa + sdy_dn * ca
                    jx_rt, jy_rt = -jy_dn, jx_dn
                    u_loc = 0.5 + (rx * jx_rt + ry * jy_rt) / (scale_mm * 1.45)
                    v_loc = 0.5 + (rx * jx_dn + ry * jy_dn) / (scale_mm * 1.45)
                else:
                    ang = cell_rand(cx_i, cy_i, seed + 3) * (2.0 * math.pi)
                    ca, sa = np.cos(ang), np.sin(ang)
                    u_loc = 0.5 + (rx * ca + ry * sa) / (scale_mm * 1.45)
                    v_loc = 0.5 + (-rx * sa + ry * ca) / (scale_mm * 1.45)

                su = np.clip((np.clip(u_loc, 0.04, 0.96) * (sz - 1)).astype(np.int32), 0, sz - 1)
                sv = np.clip((np.clip(v_loc, 0.04, 0.96) * (sz - 1)).astype(np.int32), 0, sz - 1)
                idx_s = (np.floor(cell_rand(cx_i, cy_i, seed + 4) * nb).astype(np.int32)) % nb
                c_samp = np.zeros((H, W, 3), dtype=np.float32)
                for bi in range(nb):
                    m = idx_s == bi
                    if np.any(m):
                        c_samp[m] = bank[bi][sv[m], su[m]]

                b1 = dist < d1
                b2 = (~b1) & (dist < d2)
                if np.any(b1):
                    d2[b1] = d1[b1]; col2[b1] = col1[b1]
                    d1[b1] = dist[b1]; col1[b1] = c_samp[b1]
                if np.any(b2):
                    d2[b2] = dist[b2]; col2[b2] = c_samp[b2]

        gap = d2 - d1
        t_seam = np.clip(0.5 - 0.48 * (gap / 2.4), 0.0, 0.5)[..., None]
        return np.clip(col1 * (1.0 - t_seam) + col2 * t_seam, 0.0, 1.0)

    print("  [Exemplar V5] Quilting cold steel-grey fractured granite & alpine foothills...")
    tex_granite = quilt_bank(swatches["granite"], cell_mm=9.5, scale_mm=11.0, align_slope=True, seed=101)
    f_fine = ridge_fbm(X, Y, [0.85, 1.85], [0.55, 0.45], seed=709)
    crk_micro = 1.0 - 0.14 * smoothstep(24.0, 44.0, slope_deg) * smoothstep(0.18, 0.50, 1.0 - f_fine)
    cav_shade = np.clip(1.0 + 0.22 * np.clip(cav, -1.2, 1.0), 0.72, 1.14)
    tex_granite = np.clip((tex_granite - 0.44) * 1.12 + 0.44, 0.0, 1.0)
    tex_granite = np.clip(tex_granite * (crk_micro * cav_shade)[..., None], 0.0, 1.0)

    tex_scree = quilt_bank(swatches["scree"], cell_mm=8.5, scale_mm=10.0, align_slope=True, seed=202)
    tex_meadow = quilt_bank(swatches["meadow"], cell_mm=9.5, scale_mm=10.5, align_slope=False, seed=303)
    tex_snow = np.array([0.93, 0.945, 0.96], dtype=np.float32)[None, None, :] * (0.97 + 0.03 * f_fine[..., None])

    f_mac = ridge_fbm(X, Y, [0.10, 0.24], [0.6, 0.4], seed=505)
    f_det = ridge_fbm(X, Y, [0.45, 1.15], [0.6, 0.4], seed=506)
    in_area = smoothstep(ROAD_MARGIN + 0.35, ROAD_MARGIN + 1.25, dist_h) * smoothstep(PAD_RADIUS + 0.30, PAD_RADIUS + 1.25, dist_v)

    rim_turf = (1.0 - smoothstep(4.2, 7.6, dist_h)) * (1.0 - smoothstep(1.2, 4.8, rel_z))
    meadow_m = np.maximum(
        (1.0 - smoothstep(3.5, 9.5, rel_z + 2.0 * (f_mac - 0.5))) * (1.0 - smoothstep(29.0, 40.0, slope_deg)),
        rim_turf,
    ) * in_area
    col = tex_granite * (1.0 - meadow_m[..., None]) + tex_meadow * meadow_m[..., None]

    inner_scree = smoothstep(ROAD_MARGIN + 2.2, ROAD_MARGIN + 4.8, dist_h) * smoothstep(PAD_RADIUS + 1.5, PAD_RADIUS + 4.0, dist_v)
    scree_m = smoothstep(-0.22, -1.25, curv + 0.35 * (f_det - 0.5)) * smoothstep(2.2, 8.2, rel_z) * (1.0 - smoothstep(14.5, 24.0, rel_z)) * (1.0 - smoothstep(29.0, 40.0, slope_deg)) * inner_scree
    col = col * (1.0 - 0.72 * scree_m[..., None]) + tex_scree * (0.72 * scree_m[..., None])

    # Strictly keep snow ONLY inside true elevated concave bowls + sheltered central glacial cirque (0.0000% on cliffs >=44 deg!)
    central_bowl = np.exp(-((((X - 5.5) / 11.8) ** 2 + ((Y - 2.5) / 9.8) ** 2) ** 1.4))
    bowl_snow = central_bowl * smoothstep(6.0, 9.8, rel_z) * (1.0 - smoothstep(37.0, 42.5, slope_deg))
    snow_couloir = np.maximum(
        bowl_snow,
        smoothstep(10.5, 16.0, rel_z)
        * smoothstep(-0.15, -1.20, curv + 0.25 * (f_mac - 0.5))
        * (1.0 - smoothstep(28.0, 35.0, slope_deg)),
    ) * inner_scree
    col = col * (1.0 - snow_couloir[..., None]) + tex_snow * snow_couloir[..., None]
    return np.clip(col, 0.0, 1.0)


def load_dem_2k():
    Z_1k = np.load(os.path.join(WORK_DIR, "Z_geom_v5.npy")).astype(np.float64)
    N2 = 2048
    Z_2k = ndi.zoom(Z_1k, N2 / Z_1k.shape[0], order=3)
    lin = np.linspace(-R, R, N2)
    X, Y = np.meshgrid(lin, lin)
    dist_h = hex_dist(X, Y)
    dist_v = vert_dist(X, Y)
    dx = 2.0 * R / (N2 - 1)
    return Z_2k, X, Y, dist_h, dist_v, dx


def load_available_views():
    """Load 0-deg hero + any Nano Banana completed views (180, 60, 120, 240, 300) in REFS_DIR."""
    views = []
    hero_rgb = np.asarray(Image.open(os.path.join(REFS_DIR, "hero_sq.png")).convert("RGB"), dtype=np.float32) / 255.0
    views.append((0.0, hero_rgb, True))
    for ang in [180, 60, 300, 120, 240]:
        p = os.path.join(REFS_DIR, f"view_{ang:03d}_nb.png")
        if os.path.exists(p):
            im = Image.open(p).convert("RGB")
            if im.size != (1024, 1024):
                im = im.resize((1024, 1024), Image.LANCZOS)
            views.append((float(ang), np.asarray(im, dtype=np.float32) / 255.0, False))
    return views


def blend_views_winner_takes_most(Z_2k, X, Y, dist_h, dist_v, dx, cam_p, views):
    """
    Harmonize exposure/color across views on overlapping facets and blend via sharp
    winner-takes-most normal-facing weights (ndotv^3.2) — zero halos, zero Voronoi blur!
    """
    N2 = Z_2k.shape[0]
    acc_num = np.zeros((N2, N2, 3), dtype=np.float32)
    acc_den = np.zeros((N2, N2), dtype=np.float32)
    vis_max = np.zeros((N2, N2), dtype=np.float32)

    for ang, rgb, is_hero in views:
        vcam = cam_p if ang == 0.0 else compute_cam_p_angle(cam_p, ang)
        delit, vis_w, ndotv, vc = project_view_clean(
            Z_2k, X, Y, dist_h, dist_v, dx, rgb, vcam, min_ndotv=(0.14 if is_hero else 0.22), erode_px=(3 if is_hero else 5), is_hero=is_hero
        )
        # Winner-takes-most weight: perpendicular views dominate cleanly, with ground-truth 0-deg boosted on front
        wt = vis_w * np.power(np.clip(ndotv, 0.0, 1.0), 3.4)
        if is_hero:
            wt *= 2.6

        if not is_hero and np.any(acc_den > 0.08):
            cur_est = acc_num / np.maximum(acc_den[..., None], 1e-5)
            ov = (acc_den > 0.12) & (wt > 0.08)
            if int(np.sum(ov)) >= 500:
                med_ref = np.median(cur_est[ov], axis=0)
                med_new = np.median(delit[ov], axis=0)
                gain = np.clip(med_ref / np.maximum(med_new, 0.05), 0.85, 1.18)
                delit = np.clip(delit * gain[None, None, :], 0.0, 1.0)
                print(f"    Color-matched {ang:03.0f} deg gain={np.round(gain, 3)}")

        acc_num += delit * wt[..., None]
        acc_den += wt
        vis_max = np.maximum(vis_max, vis_w)
        cov_i = float(vc[dist_h >= 4.2].mean())
        print(f"    Projected {ang:03.0f} deg (is_hero={is_hero}): view_cov={cov_i:.3f}")

    # Smooth continuous transition weight across covered regions
    w_cov = np.clip(ndi.gaussian_filter(np.clip(acc_den / 0.10, 0.0, 1.0), sigma=2.0), 0.0, 1.0)
    w_cov = np.maximum(w_cov, vis_max)
    albedo_views = np.where(acc_den[..., None] > 1e-4, acc_num / np.maximum(acc_den[..., None], 1e-4), 0.5)
    return np.clip(albedo_views, 0.0, 1.0).astype(np.float32), np.clip(w_cov, 0.0, 1.0).astype(np.float32), acc_den


def run_stage_guide_180():
    os.makedirs(WORK_DIR, exist_ok=True)
    cam_p = np.load(os.path.join(REFS_DIR, "hero_cam_params.npy"))
    Z_2k, X, Y, dist_h, dist_v, dx = load_dem_2k()
    hero_rgb = np.asarray(Image.open(os.path.join(REFS_DIR, "hero_sq.png")).convert("RGB"), dtype=np.float32) / 255.0
    alb_cur, w_cur, _ = blend_views_winner_takes_most(Z_2k, X, Y, dist_h, dist_v, dx, cam_p, [(0.0, hero_rgb, True)])
    p180 = os.path.join(REFS_DIR, "partial_view_180.png")
    render_partial_view(Z_2k, cam_p, alb_cur, w_cur, p180, yaw_offset_deg=180.0, size=1024)
    in_ter = (dist_h >= 4.4) & (dist_v >= 10.7)
    print(f"[Guide-180 Done] total_cov={float((w_cur > 0.3)[in_ter].mean()):.3f} -> {p180}")


def run_stage_guide_sides():
    os.makedirs(WORK_DIR, exist_ok=True)
    cam_p = np.load(os.path.join(REFS_DIR, "hero_cam_params.npy"))
    Z_2k, X, Y, dist_h, dist_v, dx = load_dem_2k()
    views = load_available_views()
    alb_cur, w_cur, _ = blend_views_winner_takes_most(Z_2k, X, Y, dist_h, dist_v, dx, cam_p, views)
    for ang in [60, 300, 120, 240]:
        out_p = os.path.join(REFS_DIR, f"partial_view_{ang:03d}.png")
        render_partial_view(Z_2k, cam_p, alb_cur, w_cur, out_p, yaw_offset_deg=float(ang), size=1024)
    in_ter = (dist_h >= 4.4) & (dist_v >= 10.7)
    print(f"[Guide-Sides Done] views={len(views)}, total_cov={float((w_cur > 0.3)[in_ter].mean()):.3f} -> partial_view_{{060,120,240,300}}.png")


def run_stage_guide_top():
    os.makedirs(WORK_DIR, exist_ok=True)
    cam_p = np.load(os.path.join(REFS_DIR, "hero_cam_params.npy"))
    Z_2k, X, Y, dist_h, dist_v, dx = load_dem_2k()
    views = load_available_views()
    alb_cur, w_cur, _ = blend_views_winner_takes_most(Z_2k, X, Y, dist_h, dist_v, dx, cam_p, views)
    # Render top-down 2K hybrid showing painted regions + shaded clay in any blind holes
    gy, gx = np.gradient(Z_2k, dx, dx)
    nz = 1.0 / np.sqrt(1.0 + gx * gx + gy * gy)
    nx_n, ny_n = -gx * nz, -gy * nz
    ndotl_top = np.clip(nx_n * (-0.40) + ny_n * 0.50 + nz * 0.77, 0.0, 1.0)
    clay_s = np.clip(0.40 + 0.52 * ndotl_top, 0.0, 1.0)
    clay_rgb = np.stack([clay_s, clay_s, clay_s], axis=-1)
    top_img = alb_cur * w_cur[..., None] + clay_rgb * (1.0 - w_cur[..., None])
    slate = np.array([0.16, 0.175, 0.195], dtype=np.float32)
    collar = 1.0 - smoothstep(ROAD_MARGIN, ROAD_MARGIN + 0.8, dist_h) * smoothstep(PAD_RADIUS, PAD_RADIUS + 0.8, dist_v)
    top_img = top_img * (1.0 - collar[..., None]) + slate * collar[..., None]
    out_p = os.path.join(WORK_DIR, "partial_ortho_top.png")
    Image.fromarray(np.flipud((np.clip(top_img, 0, 1) * 255).astype(np.uint8))).resize((1024, 1024), Image.LANCZOS).save(out_p)
    in_ter = (dist_h >= 4.4) & (dist_v >= 10.7)
    print(f"[Guide-Top Done] total_cov={float((w_cur > 0.3)[in_ter].mean()):.3f} -> {out_p}")


def run_stage_fuse():
    os.makedirs(WORK_DIR, exist_ok=True)
    cam_p = np.load(os.path.join(REFS_DIR, "hero_cam_params.npy"))
    hero_rgb = np.asarray(Image.open(os.path.join(REFS_DIR, "hero_sq.png")).convert("RGB"), dtype=np.float32) / 255.0
    Z_2k, X, Y, dist_h, dist_v, dx = load_dem_2k()

    views = load_available_views()
    print(f"  [Fuse] Blending {len(views)} camera views via Winner-Takes-Most...")
    albedo_views, w_cov, acc_den = blend_views_winner_takes_most(Z_2k, X, Y, dist_h, dist_v, dx, cam_p, views)

    # Optional top-down zenithal fill if provided
    top_nb_path = os.path.join(REFS_DIR, "view_top_nb.png")
    if os.path.exists(top_nb_path):
        im_t = Image.open(top_nb_path).convert("RGB").resize((2048, 2048), Image.LANCZOS)
        top_rgb = np.flipud(np.asarray(im_t, dtype=np.float32) / 255.0)
        print(f"    Applied top-down Nano Banana fill for occluded crevices: {top_nb_path}")
        albedo_views = albedo_views * w_cov[..., None] + top_rgb * (1.0 - w_cov[..., None])
        w_cov = np.maximum(w_cov, smoothstep(ROAD_MARGIN + 0.6, ROAD_MARGIN + 1.5, dist_h) * smoothstep(PAD_RADIUS + 0.6, PAD_RADIUS + 1.5, dist_v))

    # Clean local nearest-neighbor + gentle multi-scale propagation ONLY for tiny residual blind crevices
    in_ter = (dist_h >= 4.2) & (dist_v >= 10.6)
    cov_pct = float((w_cov > 0.25)[in_ter].mean()) * 100.0
    print(f"    Multi-view photographic coverage across terrain: {cov_pct:.2f}%")

    valid_px = (acc_den > 0.06) & in_ter
    if np.any(valid_px) and not np.all(valid_px[in_ter]):
        # Harmonic multi-scale diffusion inpainting for tiny residual occluded crevices (zero polygon edges!)
        _, nn_idx = ndi.distance_transform_edt(~valid_px, return_indices=True)
        fill_col = albedo_views[nn_idx[0], nn_idx[1]].copy()
        for sig in (10.0, 5.0, 2.5):
            sm_fill = ndi.gaussian_filter(fill_col, (sig, sig, 0.0))
            fill_col = np.where(valid_px[..., None], albedo_views, sm_fill)
        f_crev = ridge_fbm(X, Y, [0.95, 2.25], [0.55, 0.45], seed=709)
        fill_col = np.clip(fill_col * (0.92 + 0.12 * f_crev[..., None]), 0.0, 1.0)
        w_smooth = np.clip(ndi.gaussian_filter(valid_px.astype(np.float32), sigma=3.0), 0.0, 1.0)
        w_blend = np.maximum(w_cov, w_smooth)
        albedo_2k = albedo_views * w_blend[..., None] + fill_col * (1.0 - w_blend[..., None])
    else:
        albedo_2k = albedo_views

    # Machined brushed gunmetal slate hex bezel + crisp bevel catchlight & recessed pad rings (0 mm Z change)
    rel_z = np.maximum(0.0, Z_2k - Z_BASE)
    on_border = np.maximum(
        1.0 - smoothstep(ROAD_MARGIN + 0.35, ROAD_MARGIN + 1.00, dist_h),
        1.0 - smoothstep(PAD_RADIUS + 0.30, PAD_RADIUS + 0.90, dist_v),
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

    # Export canonical 2K .npy heightmap (np.flipud to match Y-down PNG convention of build_catan_biome.py!) & 4K sharpened PNG albedo
    hm_norm = np.clip((Z_2k - Z_BASE) / RELIEF_MM, 0.0, 1.0) * 255.0
    np.save(os.path.join(WORK_DIR, "montana_granito_v5_hm_2k.npy"), np.flipud(hm_norm).astype(np.float32))
    Image.fromarray(np.flipud(np.clip(hm_norm * 257.0, 0, 65535).astype(np.uint16)), mode="I;16").save(
        os.path.join(WORK_DIR, "montana_granito_v5_hm_16bit.png")
    )

    alb_4k = Image.fromarray(np.flipud((np.clip(albedo_2k, 0, 1) * 255).astype(np.uint8))).resize((4096, 4096), Image.LANCZOS)
    a4 = np.asarray(alb_4k, dtype=np.float32) / 255.0
    blur4 = ndi.gaussian_filter(a4, (1.3, 1.3, 0.0))
    a4_sharp = np.clip(a4 + 0.42 * (a4 - blur4), 0.0, 1.0)
    alb_4k_path = os.path.join(WORK_DIR, "montana_granito_v5_albedo_4k.png")
    Image.fromarray((a4_sharp * 255.0).astype(np.uint8)).save(alb_4k_path)

    # Render textured parity & 6-view turnaround audits
    img0, _, _, _, _, _, _, _, _, _ = ray_march_heightfield(Z_2k, cam_p, out_w=512, out_h=512, tex_rgb=albedo_2k, sun_rel=True)
    h512 = np.asarray(Image.fromarray((hero_rgb * 255).astype(np.uint8)).resize((512, 512), Image.LANCZOS), dtype=np.float32) / 255.0
    comp = np.concatenate([h512, img0], axis=1)
    Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).save(os.path.join(WORK_DIR, "v5_textured_vs_hero.png"))

    views_6 = []
    for ang in [0, 60, 120, 180, 240, 300]:
        im, _, _, _, _, _, _, _, _, _ = ray_march_heightfield(Z_2k, cam_p, out_w=420, out_h=420, tex_rgb=albedo_2k, yaw_offset_deg=ang, sun_rel=True)
        pil_v = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(pil_v)
        dr.text((12, 12), f"{ang} deg", fill=(20, 25, 30))
        views_6.append(np.asarray(pil_v))
    grid = np.concatenate([np.concatenate(views_6[:3], axis=1), np.concatenate(views_6[3:], axis=1)], axis=0)
    Image.fromarray(grid).save(os.path.join(WORK_DIR, "v5_textured_6views.png"))
    print(f"[Stage Fuse Done] cov={cov_pct:.2f}% -> {alb_4k_path}")


def main():
    p = argparse.ArgumentParser(description="Montaña Granito V5 Sculptor")
    p.add_argument("--stage", choices=["clay", "guide", "guide-180", "guide-sides", "guide-top", "fuse"], required=True)
    args = p.parse_args()
    if args.stage == "clay":
        run_stage_clay()
    elif args.stage in ("guide", "guide-180"):
        run_stage_guide_180()
    elif args.stage == "guide-sides":
        run_stage_guide_sides()
    elif args.stage == "guide-top":
        run_stage_guide_top()
    else:
        run_stage_fuse()


if __name__ == "__main__":
    main()
