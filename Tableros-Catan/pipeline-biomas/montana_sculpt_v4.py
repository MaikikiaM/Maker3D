"""
Stage-based Master Pipeline for Bioma 1 (Montana) — Exact Nano Banana Parity across 360 deg.
Stages:
  1. --stage clay   : Build 360-deg faceted alpine DEM, optimize hero skyline, render Clay Hero & 6-View Clay audits.
  2. --stage guide  : Project de-lit front hero RGB onto canonical UV + generate top-down orthographic hybrid guide for Nano Banana completion.
  3. --stage fuse   : Fuse front hero_delit + Nano Banana 360-deg orthographic albedo + carve 2D-to-3D microrelief across all 360 deg.
  4. --stage build  : Compile watertight STL Master (<=10M tris) + <50MB WebGL HTML via build_catan_biome.py & assemble_viewer_v2.py.
"""
import argparse
import math
import os
import sys
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MONTANA_DIR = os.path.join(REPO_ROOT, "Tableros-Catan", "pipeline-biomas", "montana")
REFS_DIR = os.path.join(MONTANA_DIR, "refs")
WORK_DIR = os.path.join(MONTANA_DIR, "work_v4")

R = 64.0
APOTHEM = R * math.sqrt(3.0) * 0.5  # 55.42562584 mm
Z_BASE = 25.0
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00
RELIEF_MM = 45.2


def hex_dist(X, Y):
    ax, ay = np.abs(X), np.abs(Y)
    return APOTHEM - np.maximum(ax, 0.5 * ax + (math.sqrt(3.0) * 0.5) * ay)


def vert_dist(X, Y):
    dv = np.full_like(X, 999.0, dtype=np.float64)
    for k in range(6):
        a = math.radians(30.0 + 60.0 * k)
        dv = np.minimum(dv, np.hypot(X - R * math.cos(a), Y - R * math.sin(a)))
    return dv


def smoothstep(e0, e1, x):
    if abs(e1 - e0) < 1e-8:
        return np.where(x >= e0, 1.0, 0.0)
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smax(a, b, k=2.0):
    h = np.clip(0.5 + 0.5 * (a - b) / k, 0.0, 1.0)
    return a * h + b * (1.0 - h) + k * h * (1.0 - h) * 0.5


def smooth_fbm(X, Y, scales, weights, seed=42):
    rng = np.random.default_rng(seed)
    out = np.zeros_like(X, dtype=np.float64)
    for sc, wt in zip(scales, weights):
        ang = rng.uniform(0, 2 * math.pi, 4)
        ph = rng.uniform(0, 2 * math.pi, 4)
        n = (
            np.sin(X * sc * math.cos(ang[0]) + Y * sc * math.sin(ang[0]) + ph[0])
            + np.cos(X * sc * math.cos(ang[1]) + Y * sc * math.sin(ang[1]) + ph[1])
            + np.sin(X * sc * math.cos(ang[2]) - Y * sc * math.sin(ang[2]) + ph[2])
            + np.cos(X * sc * math.cos(ang[3]) - Y * sc * math.sin(ang[3]) + ph[3])
        ) / 4.0
        out += wt * (0.5 + 0.5 * n)
    return out / sum(weights)


def ridge_fbm(X, Y, scales, weights, seed=42):
    rng = np.random.default_rng(seed)
    out = np.zeros_like(X, dtype=np.float64)
    for sc, wt in zip(scales, weights):
        ang = rng.uniform(0, 2 * math.pi, 3)
        ph = rng.uniform(0, 2 * math.pi, 3)
        n = (
            np.sin(X * sc * math.cos(ang[0]) + Y * sc * math.sin(ang[0]) + ph[0])
            + np.cos(X * sc * math.cos(ang[1]) + Y * sc * math.sin(ang[1]) + ph[1])
            + np.sin(X * sc * math.cos(ang[2]) - Y * sc * math.sin(ang[2]) + ph[2])
        ) / 3.0
        out += wt * (1.0 - np.abs(n))
    return out / sum(weights)


def faceted_ridge(X, Y, x0, y0, h0, x1, y1, h1, w0, w1, sharp=0.96, seed_zig=0, rib_amp=0.14):
    """Sharp polyhedral granite arête with faceted cross-section and downhill rock ribs."""
    vx, vy = x1 - x0, y1 - y0
    L2 = max(vx * vx + vy * vy, 1e-6)
    L = math.sqrt(L2)
    t = np.clip(((X - x0) * vx + (Y - y0) * vy) / L2, 0.0, 1.0)
    px, py = x0 + t * vx, y0 + t * vy
    nx, ny = -vy / L, vx / L
    # Angular zig along spine
    zig = (
        1.15 * np.sin(t * 7.5 + seed_zig)
        + 0.48 * np.sin(t * 17.0 + seed_zig * 2.3)
        + 0.20 * (1.0 - np.abs(np.sin(t * 28.0 + seed_zig * 3.1)))
    )
    dx_s = X - (px + nx * zig * t * (1.0 - t) * 1.15)
    dy_s = Y - (py + ny * zig * t * (1.0 - t) * 1.15)
    # Signed distance across ridge for asymmetric faceted granite planar walls + side ribs
    s_side = dx_s * nx + dy_s * ny
    d = np.sqrt(dx_s * dx_s + dy_s * dy_s + 0.015)
    w = w0 * (1.0 - t) + w1 * t
    h = h0 * (1.0 - t) + h1 * t
    # Lateral downhill rib modulation (creates sharp granite fins jutting sideways off the main ridge!)
    rib_wave = 1.0 - np.abs(np.sin(t * max(L * 0.65, 3.0) + seed_zig * 1.7 + 0.35 * np.sign(s_side)))
    w_eff = w * (1.0 + rib_amp * (rib_wave - 0.35) * np.clip(np.abs(s_side) / (0.35 * w + 0.5), 0.0, 1.0))
    # Crisp V-crest profile with smooth foot transition into valley/flanks (zero cookie-cutter rim crease!)
    rn = d / np.maximum(w_eff, 0.5)
    foot = smoothstep(1.0, 0.68, rn)
    profile = np.power(np.maximum(0.0, 1.0 - rn), sharp) * (0.35 + 0.65 * foot)
    return np.maximum(0.0, h * profile)


def rock_outcrop(X, Y, cx, cy, h, rx, ry, ang_deg=0.0, sharp=0.88):
    """Angular faceted rock spire or foreground outcrop block."""
    ca, sa = math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))
    dx, dy = X - cx, Y - cy
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    # Diamond/hex faceted L1+L2 distance for chiseled planar rock faces
    d_l2 = np.hypot(u / rx, v / ry)
    d_l1 = (np.abs(u / rx) + np.abs(v / ry)) * 0.72
    d = 0.58 * d_l2 + 0.42 * d_l1
    return np.maximum(0.0, h * np.power(np.maximum(0.0, 1.0 - d), sharp))


def project_pts_vec(params, X, Y, Z, W=1024.0, H=1024.0):
    az_deg, el_deg, dist, tx, ty, tz, fov_deg, roll_deg = params
    az, el = math.radians(az_deg), math.radians(el_deg)
    cam = np.array([tx + dist * math.cos(el) * math.sin(az), ty - dist * math.cos(el) * math.cos(az), tz + dist * math.sin(el)])
    fwd = np.array([tx, ty, tz]) - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    right_r = cr * right + sr * up
    up_r = -sr * right + cr * up
    rx, ry, rz = X - cam[0], Y - cam[1], Z - cam[2]
    xc = rx * right_r[0] + ry * right_r[1] + rz * right_r[2]
    yc = rx * up_r[0] + ry * up_r[1] + rz * up_r[2]
    zc = rx * fwd[0] + ry * fwd[1] + rz * fwd[2]
    f = (H * 0.5) / math.tan(math.radians(fov_deg * 0.5))
    return W * 0.5 + f * (xc / zc), H * 0.5 - f * (yc / zc), zc


def ray_march_heightfield(Z_grid, params, out_w=512, out_h=512, tex_rgb=None, yaw_offset_deg=0.0, sun_rel=True):
    N = Z_grid.shape[0]
    p_rot = params.copy()
    p_rot[0] = params[0] + yaw_offset_deg
    az_deg, el_deg, dist, tx, ty, tz, fov_deg, roll_deg = p_rot
    az, el = math.radians(az_deg), math.radians(el_deg)
    cyaw, syaw = math.cos(math.radians(yaw_offset_deg)), math.sin(math.radians(yaw_offset_deg))
    rtx = tx * cyaw - ty * syaw
    rty = tx * syaw + ty * cyaw
    cam = np.array([rtx + dist * math.cos(el) * math.sin(az), rty - dist * math.cos(el) * math.cos(az), tz + dist * math.sin(el)])
    fwd = np.array([rtx, rty, tz]) - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    right_r = cr * right + sr * up
    up_r = -sr * right + cr * up
    f = (out_h * 0.5) / math.tan(math.radians(fov_deg * 0.5))

    u = np.arange(out_w, dtype=np.float32) + 0.5
    v = np.arange(out_h, dtype=np.float32) + 0.5
    uu, vv = np.meshgrid(u, v)
    dx_d = (uu - out_w * 0.5) / f
    dy_d = -(vv - out_h * 0.5) / f
    rx = fwd[0] + dx_d * right_r[0] + dy_d * up_r[0]
    ry = fwd[1] + dx_d * right_r[1] + dy_d * up_r[1]
    rz = fwd[2] + dx_d * right_r[2] + dy_d * up_r[2]
    rlen = np.sqrt(rx * rx + ry * ry + rz * rz)
    rx /= rlen; ry /= rlen; rz /= rlen

    z_hi = float(Z_grid.max()) + 1.0
    z_lo = Z_BASE - 9.4
    t_min = np.maximum(10.0, (z_hi - cam[2]) / rz)
    t_max = (z_lo - cam[2]) / rz

    n_steps = 165
    hit_t = np.full((out_h, out_w), np.inf, dtype=np.float32)
    hit_terrain = np.zeros((out_h, out_w), dtype=bool)
    hit_plinth = np.zeros((out_h, out_w), dtype=bool)
    inv_span = (N - 1) / (2.0 * R)
    ts_frac = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    prev_t = t_min
    for k in range(1, n_steps):
        tk = t_min + ts_frac[k] * (t_max - t_min)
        px = cam[0] + tk * rx; py = cam[1] + tk * ry; pz = cam[2] + tk * rz
        hd = hex_dist(px, py)
        in_hex = hd >= 0.0
        ix = np.clip((px + R) * inv_span, 0, N - 1.001).astype(np.int32)
        iy = np.clip((py + R) * inv_span, 0, N - 1.001).astype(np.int32)
        below = in_hex & (pz <= Z_grid[iy, ix]) & (~hit_terrain) & (~hit_plinth)
        if np.any(below):
            lo_t = prev_t[below]; hi_t = tk[below]
            bx0, by0, bz0 = cam[0], cam[1], cam[2]
            brx, bry, brz = rx[below], ry[below], rz[below]
            for _ in range(7):
                mt = 0.5 * (lo_t + hi_t)
                mpx, mpy, mpz = bx0 + mt * brx, by0 + mt * bry, bz0 + mt * brz
                mhd = hex_dist(mpx, mpy)
                mix = np.clip((mpx + R) * inv_span, 0, N - 1.001).astype(np.int32)
                miy = np.clip((mpy + R) * inv_span, 0, N - 1.001).astype(np.int32)
                m_below = (mhd >= 0.0) & (mpz <= Z_grid[miy, mix])
                hi_t = np.where(m_below, mt, hi_t)
                lo_t = np.where(m_below, lo_t, mt)
            ft = 0.5 * (lo_t + hi_t)
            hit_t[below] = ft
            fpz = cam[2] + ft * brz
            fhd = hex_dist(cam[0] + ft * brx, cam[1] + ft * bry)
            is_side_wall = (fpz < Z_BASE - 0.12) & (fhd < 0.8)
            is_ter = np.zeros_like(below); is_pl = np.zeros_like(below)
            is_ter[below] = ~is_side_wall
            is_pl[below] = is_side_wall
            hit_terrain |= is_ter; hit_plinth |= is_pl
        prev_t = tk

    any_hit = hit_terrain | hit_plinth
    hx = cam[0] + hit_t * rx; hy = cam[1] + hit_t * ry; hz = cam[2] + hit_t * rz
    gy, gx = np.gradient(Z_grid, 2 * R / (N - 1), 2 * R / (N - 1))
    n_len = np.sqrt(1.0 + gx * gx + gy * gy)
    nx_g, ny_g, nz_g = -gx / n_len, -gy / n_len, 1.0 / n_len

    img = np.full((out_h, out_w, 3), 0.84, dtype=np.float32)
    ld = (right_r * 0.65 + up_r * 0.10 - fwd * 0.45 + np.array([0, 0, 0.58])) if sun_rel else np.array([0.66, -0.35, 0.66], dtype=np.float32)
    ld = ld / np.linalg.norm(ld)
    if np.any(hit_terrain):
        fx = np.clip((hx[hit_terrain] + R) * inv_span, 0, N - 1.001)
        fy = np.clip((hy[hit_terrain] + R) * inv_span, 0, N - 1.001)
        ix0, iy0 = fx.astype(np.int32), fy.astype(np.int32)
        ndotl = np.clip(nx_g[iy0, ix0] * ld[0] + ny_g[iy0, ix0] * ld[1] + nz_g[iy0, ix0] * ld[2], 0.0, 1.0)
        if tex_rgb is not None:
            th, tw = tex_rgb.shape[:2]
            t_span = (tw - 1) / (2.0 * R)
            tix = np.clip(np.round((hx[hit_terrain] + R) * t_span), 0, tw - 1).astype(np.int32)
            tiy = np.clip(np.round((hy[hit_terrain] + R) * t_span), 0, th - 1).astype(np.int32)
            img[hit_terrain] = tex_rgb[tiy, tix] * (0.56 + 0.50 * ndotl)[:, None]
        else:
            img[hit_terrain] = np.array([0.84, 0.86, 0.89], dtype=np.float32) * (0.34 + 0.72 * ndotl)[:, None]
    if np.any(hit_plinth):
        img[hit_plinth] = np.array([0.13, 0.14, 0.15], dtype=np.float32)
    return img, any_hit, hit_terrain, hx, hy, hz, cam, nx_g, ny_g, nz_g


def build_base_alpine_dem(N=1024):
    lin = np.linspace(-R, R, N)
    X, Y = np.meshgrid(lin, lin)
    dist_h = hex_dist(X, Y)
    dist_v = vert_dist(X, Y)
    in_hex = smoothstep(3.6, 7.0, dist_h) * smoothstep(10.2, 13.2, dist_v)

    # Natural organic undulating emerald meadow + flanking valley slopes (zero smooth sausage tubes!)
    hill_n = smooth_fbm(X, Y, [0.09, 0.21, 0.45, 0.95], [0.42, 0.30, 0.18, 0.10], seed=204)
    hill_r = ridge_fbm(X, Y, [0.22, 0.55], [0.6, 0.4], seed=205)
    valley_ramp = 4.5 * smoothstep(-45.0, -4.0, Y) * np.exp(-0.5 * (X / 30.0) ** 2)
    bank_L = 2.6 * (0.75 + 0.45 * hill_r) * np.exp(-0.5 * (((X + 10.5) / 6.8) ** 2 + ((Y + 24.0) / 11.5) ** 2))
    bank_R = 2.8 * (0.75 + 0.45 * hill_r) * np.exp(-0.5 * (((X - 8.5) / 6.8) ** 2 + ((Y + 23.5) / 11.5) ** 2))
    Z_floor = in_hex * (0.9 + 1.6 * hill_n + valley_ramp + bank_L + bank_R)

    # Compact central massif hubs (broad solid summit pyramid + lower smooth open north valleys)
    r_core = np.hypot((X - 1.2) / 1.12, (Y - 12.2) / 0.92)
    Z_body = 33.0 * np.power(np.maximum(0.0, 1.0 - r_core / 39.5), 1.40)
    Z_body = smax(Z_body, 26.2 * np.power(np.maximum(0.0, 1.0 - np.hypot(X + 20.5, Y - 7.5) / 29.5), 1.36), k=4.5)
    Z_body = smax(Z_body, 27.6 * np.power(np.maximum(0.0, 1.0 - np.hypot(X - 21.5, Y - 9.5) / 30.0), 1.36), k=4.5)
    Z_body = smax(Z_body, 15.5 * np.power(np.maximum(0.0, 1.0 - np.hypot(X - 0.0, Y - 24.5) / 25.0), 1.55), k=4.5)
    Z_base_terrain = smax(Z_floor, Z_body, k=2.2)

    # Broad sweeping U-shaped glacial amphitheaters in BOTH front AND north cirque bowls (<40 deg floors for snow!)
    cirque_FL = 3.0 * np.exp(-0.5 * (((X + 10.2) / 8.8) ** 2 + ((Y + 2.0) / 11.5) ** 2))
    cirque_FR = 3.2 * np.exp(-0.5 * (((X - 12.6) / 9.0) ** 2 + ((Y + 1.0) / 11.8) ** 2))
    cirque_NL = 5.5 * np.exp(-0.5 * (((X + 11.8) / 7.8) ** 2 + ((Y - 26.5) / 9.6) ** 2))
    cirque_NR = 5.5 * np.exp(-0.5 * (((X - 12.2) / 7.8) ** 2 + ((Y - 26.5) / 9.6) ** 2))
    Z_base_terrain = smax(Z_floor, Z_base_terrain - (cirque_FL + cirque_FR + cirque_NL + cirque_NR), k=2.4)

    # Crisp landmark-calibrated ridges (continuous broad central pyramid -> V-notch -> sharp Right Spire)
    ridges = [
        # 1. Dominant Central Pyramid (+0.7,+13.9,45.2) + solid Right Shoulder + Central Prow
        (0.7, 13.9, 45.2,   5.2, 13.2, 37.8,   15.5, 13.5, 0.95, 1.0, 0.14),
        (5.2, 13.2, 37.8,  11.5, 11.8, 26.5,   13.5, 11.2, 0.96, 1.5, 0.13),
        (11.5, 11.8, 26.5, 19.6, 11.0, 38.6,   11.8, 14.5, 0.90, 8.4, 0.15),
        (0.7, 13.9, 45.2,   2.4, 11.5, 36.2,   13.5, 11.0, 0.92, 1.1, 0.15),
        (2.4, 11.5, 36.2,   0.7, 6.0,  24.8,   10.5, 8.2,  0.92, 2.2, 0.15),
        (0.7, 6.0,  24.8,   0.9, -3.8, 13.0,   8.2,  5.8,  0.95, 3.3, 0.13),
        (0.9, -3.8, 13.0,   0.2, -11.5, 5.2,   5.8,  4.2,  1.02, 4.1, 0.10),
        # Central-left pyramid shoulder & Inner-Left Fin (-7.5, +11.6) & Mid-Left Rib (-15.0, +10.9)
        (0.7, 13.9, 45.2,  -4.2, 12.8, 36.8,   14.5, 12.5, 0.94, 4.6, 0.14),
        (-4.2, 12.8, 36.8, -7.5, 11.6, 33.6,   12.5, 10.8, 0.92, 4.8, 0.14),
        (-7.5, 11.6, 33.6, -7.7, 4.1,  19.8,   9.8,  7.2,  0.94, 5.5, 0.14),
        (-7.7, 4.1,  19.8, -6.5, -3.5, 9.2,    7.2,  5.0,  1.02, 6.2, 0.11),
        (-7.5, 11.6, 33.6,-15.0, 10.9, 32.8,   12.0, 11.5, 0.92, 6.9, 0.14),
        (-15.0,10.9, 32.8,-13.5, 1.5,  17.5,   9.5,  7.2,  0.98, 7.4, 0.12),

        # 2. Left Massif (-21.1, +6.4, 36.2) -> Left Wing (-32.8, -6.4, 22.2) -> Far Buttress (-40.4, -16.9, 9.0)
        (-15.0, 10.9, 32.8,-21.1, 6.4,  36.2,   13.5, 15.0, 0.90, 8.8, 0.15),
        (-21.1, 6.4,  36.2,-32.8, -6.4, 22.2,   15.0, 12.2, 0.94, 9.9, 0.15),
        (-32.8, -6.4, 22.2,-40.4, -16.9, 9.0,   12.2, 8.4,  1.04, 10.7, 0.12),
        (-21.1, 6.4,  33.0,-41.5, 2.5,  12.8,   14.0, 9.2,  1.04, 11.5, 0.13),
        # Front-Left Copper Jagged Ridge! (-32.8,-6.4) -> (-27.8,-18.5) -> (-21.0,-25.0) -> (-12.9,-30.6)
        (-32.8, -6.4, 19.8,-27.8, -18.5, 13.2,  10.0, 8.0,  0.90, 12.3, 0.16),
        (-27.8,-18.5, 13.2,-21.0, -25.0, 8.8,   8.0,  6.4,  0.90, 13.4, 0.18),
        (-21.0,-25.0, 8.8, -12.9, -30.6, 4.8,   6.4,  4.8,  0.98, 14.5, 0.14),
        # Left inner flank rib
        (-21.1, 6.4,  32.0,-18.0, -6.0, 17.5,   11.0, 8.0,  0.96, 15.2, 0.14),
        (-18.0, -6.0, 17.5,-13.0, -17.0, 8.2,   8.0,  5.6,  1.04, 15.9, 0.12),

        # 3. Right Massif (+19.6,+11.0, 38.6) -> (+24.8,+9.8, 35.8) -> (+33.1,-1.4, 22.4) -> (+40.1,-14.4, 9.5)
        (19.6, 11.0,  38.6, 24.8, 9.8,  35.8,   14.8, 14.5, 0.89, 17.8, 0.15),
        (24.8, 9.8,   35.8, 33.1, -1.4, 22.4,   14.5, 12.2, 0.94, 18.9, 0.15),
        (33.1, -1.4,  22.4, 40.1, -14.4, 9.5,   12.2, 8.4,  1.04, 19.6, 0.12),
        (33.1, -1.4,  20.0, 44.0, -7.5, 8.4,    11.5, 7.8,  1.06, 20.4, 0.12),
        # Front-Right Diagonal Copper Jagged Spine! (+33.1,-1.4) -> (+30.2,-18.8) -> (+19.9,-26.2) -> (+9.8,-34.1)
        (33.1, -1.4,  20.2, 30.2, -18.8, 12.4,  10.0, 8.0,  0.90, 21.1, 0.16),
        (30.2, -18.8, 12.4, 19.9, -26.2, 8.0,   8.0,  6.2,  0.90, 22.2, 0.18),
        (19.9, -26.2, 8.0,  9.8,  -34.1, 4.4,   6.2,  4.6,  0.98, 23.3, 0.14),
        # Diagonal Copper Scree Avalanche Ridge spilling into central fork (+10.5,-6.5)->(+5.8,-12.5)->(+1.2,-17.5)
        (10.5, -6.5,  9.5,  5.8, -12.5, 6.8,    5.6,  4.4,  0.92, 23.8, 0.15),
        (5.8, -12.5,  6.8,  1.2, -17.5, 4.4,    4.4,  3.5,  0.98, 24.1, 0.14),

        # 4. Crisp 360-deg North Back Arêtes & Radial Buttress Spurs (trimmed widths so twin NW/NE glacial valleys sweep cleanly!)
        (0.7, 13.9,   45.2, -0.4, 24.8, 26.8,   11.6, 8.8,  0.92, 24.6, 0.14),
        (-0.4, 24.8,  26.8,  0.8, 34.5, 14.2,   8.8,  6.8,  0.96, 25.7, 0.13),
        (0.8, 34.5,   14.2, -0.3, 41.8, 5.2,    6.8,  4.8,  1.05, 26.5, 0.10),
        (-21.1, 6.4,  36.2,-22.5, 19.5, 26.5,   11.8, 9.2,  0.92, 27.3, 0.14),
        (-22.5, 19.5, 26.5,-24.5, 30.8, 14.5,   9.2,  7.0,  0.96, 28.1, 0.13),
        (-24.5, 30.8, 14.5,-19.5, 39.2, 5.4,    7.0,  4.8,  1.06, 28.9, 0.10),
        (-21.1, 6.4,  31.5,-32.5, 16.0, 17.0,   11.6, 8.4,  0.96, 29.7, 0.13),
        (-32.5, 16.0, 17.0,-39.5, 9.0,  6.8,    8.4,  5.6,  1.08, 30.5, 0.10),
        (19.6, 11.0,  37.5, 23.0, 21.0, 27.2,   11.8, 9.2,  0.92, 31.3, 0.14),
        (23.0, 21.0,  27.2, 25.2, 31.2, 14.8,   9.2,  7.0,  0.96, 32.1, 0.13),
        (25.2, 31.2,  14.8, 20.0, 39.5, 5.4,    7.0,  4.8,  1.06, 32.9, 0.10),
        (24.8, 9.8,   31.0, 33.4, 16.5, 16.8,   11.6, 8.4,  0.96, 33.7, 0.13),
        (33.4, 16.5,  16.8, 40.2, 8.6,  6.6,    8.4,  5.6,  1.08, 34.5, 0.10),
        # Gentle flanking scree ridges border outer north pass
        (-12.5, 35.5, 7.2, -10.5, 41.2, 3.8,    6.6,  4.6,  1.02, 37.2, 0.11),
        (13.0, 35.8,  7.4,  11.0, 41.2, 3.8,    6.6,  4.6,  1.02, 38.4, 0.11),
    ]

    Z_ridges = np.zeros_like(Z_floor)
    for r in ridges:
        Z_ridges = smax(Z_ridges, faceted_ridge(X, Y, *r), k=1.4)

    # Crisp foreground rocky scree cascade, corner rock cluster & foothill outcrops
    outcrops = [
        (9.1, -7.9,   7.4, 5.4, 4.0, -32.0),
        (5.5, -12.5,  6.0, 4.8, 3.4, -38.0),
        (2.0, -17.0,  4.6, 4.2, 2.9, -25.0),
        (37.6, -28.2, 4.8, 5.2, 3.8, 25.0),
        (33.5, -26.0, 3.8, 4.0, 3.0, 15.0),
        (-23.0,-22.5, 4.6, 4.6, 3.2, 42.0),
        (-15.5,-28.8, 3.9, 4.4, 2.8, 36.0),
        (-28.0,-30.5, 3.2, 3.8, 2.6, 20.0),
        (24.0, -23.0, 4.4, 4.6, 3.2, -35.0),
        (14.5, -30.5, 3.8, 4.2, 2.8, -42.0),
        (-11.0, 34.5, 5.2, 4.8, 3.4, 25.0),
        (11.5, 35.0,  5.2, 4.8, 3.4, -25.0),
    ]
    Z_out = np.zeros_like(Z_floor)
    for oc in outcrops:
        Z_out = np.maximum(Z_out, rock_outcrop(X, Y, *oc))

    Z_m = smax(Z_base_terrain, Z_ridges, k=1.9)
    Z_m = smax(Z_m, Z_out, k=1.1)

    # Angular granite dihedral flutes (calmed inside smooth concave snow cirque bowls)
    in_snow_bowl = (
        np.exp(-0.5 * (((X + 10.2) / 6.6) ** 2 + ((Y + 2.5) / 8.0) ** 2))
        + np.exp(-0.5 * (((X - 12.6) / 7.0) ** 2 + ((Y + 1.5) / 8.2) ** 2))
        + 0.90 * np.exp(-0.5 * (((X + 10.5) / 6.5) ** 2 + ((Y - 26.0) / 8.0) ** 2))
        + 0.90 * np.exp(-0.5 * (((X - 11.2) / 6.5) ** 2 + ((Y - 26.0) / 8.0) ** 2))
    )
    flute_damp = np.clip(1.0 - 0.72 * in_snow_bowl, 0.20, 1.0)
    fl1 = ridge_fbm(X, Y, [0.30, 0.72, 1.58, 3.1], [0.46, 0.28, 0.16, 0.10], seed=314)
    fl2 = ridge_fbm(X * 0.85 + Y * 0.35, -X * 0.35 + Y * 0.85, [0.48, 1.25, 2.5], [0.55, 0.30, 0.15], seed=777)
    Z_m = Z_m + smoothstep(3.2, 18.0, Z_m) * flute_damp * (1.80 * (fl1 - 0.44) + 1.30 * (fl2 - 0.45))

    # Carve clean organic incised S-meandering front canyon gorge (zero smooth tube lips!)
    t_g = np.clip((2.0 - Y) / 45.0, 0.0, 1.0)
    cx_g = (
        3.5 * (1.0 - t_g)
        - 3.8 * np.sin(t_g * math.pi * 0.85)
        - 3.2 * t_g
        - 1.4 * np.sin(t_g * math.pi * 2.2) * smoothstep(0.35, 0.85, t_g)
    )
    dist_g = np.abs(X - cx_g)
    gully_w = 1.65 + 1.25 * t_g
    gully_cut = smoothstep(7.5, -12.0, Y) * smoothstep(-48.5, -35.0, Y) * np.exp(-0.5 * (dist_g / gully_w) ** 2)
    Z_m = Z_m * (1.0 - 0.72 * gully_cut) + 0.55 * (0.72 * gully_cut)

    # Gentle natural North valley chute (zero abrupt vertical slot!)
    t_ng = np.clip((Y - 21.0) / 24.0, 0.0, 1.0)
    cx_ng = 2.4 * np.sin(t_ng * math.pi * 1.1)
    ng_cut = smoothstep(20.0, 27.0, Y) * smoothstep(47.0, 38.0, Y) * np.exp(-0.5 * ((X - cx_ng) / (3.8 + 1.8 * t_ng)) ** 2)
    Z_m = Z_m * (1.0 - 0.24 * ng_cut) + (Z_floor * 0.70 + 1.0) * (0.24 * ng_cut)

    # Compact hex apron respecting ROAD=3.5mm and PAD=10.0mm cleanly with wide smooth foothill setback
    taper = smoothstep(4.25, 8.6, dist_h) * smoothstep(10.45, 14.2, dist_v)
    Z = Z_BASE + Z_m * taper
    return X, Y, Z, dist_h, dist_v, taper


def optimize_hero_skyline(X, Y, Z, taper, cam_p, fg_hero, iters=15):
    top_rim = np.column_stack([
        R * np.cos(np.radians([270, 210, 150, 90, 30, 330])),
        R * np.sin(np.radians([270, 210, 150, 90, 30, 330])),
        np.full(6, Z_BASE),
    ])
    tu, tv, _ = project_pts_vec(cam_p, top_rim[:, 0], top_rim[:, 1], top_rim[:, 2], 512.0, 512.0)
    y_bezel = np.full(512, 512.0)
    for col in range(512):
        if tu[2] <= col <= tu[3]:
            y_bezel[col] = tv[2] + (col - tu[2]) / max(tu[3] - tu[2], 1.0) * (tv[3] - tv[2])
        elif tu[3] < col <= tu[4]:
            y_bezel[col] = tv[3] + (col - tu[3]) / max(tu[4] - tu[3], 1.0) * (tv[4] - tv[3])

    y_sky_h_full = np.array([np.where(fg_hero[:, c])[0].min() if fg_hero[:, c].any() else 1024 for c in range(1024)])
    y_sky_target = y_sky_h_full[::2].astype(np.float32) * 0.5

    for it in range(iters):
        _, sil_r, _, hx, hy, _, _, _, _, _ = ray_march_heightfield(Z, cam_p, out_w=512, out_h=512)
        y_sky_r = np.array([np.where(sil_r[:, c])[0].min() if sil_r[:, c].any() else 512 for c in range(512)], dtype=np.float32)
        cols = (np.arange(512) >= int(tu[2]) + 6) & (np.arange(512) <= int(tu[4]) - 6) & (
            (y_sky_target < y_bezel - 3.0) | (y_sky_r < y_bezel - 3.0)
        )
        dy_px = y_sky_r[cols] - y_sky_target[cols]
        mae = float(np.mean(np.abs(dy_px)))
        print(f"  [Skyline iter {it}] MAE={mae:.2f}px max={np.max(np.abs(dy_px)):.1f}px Zmax={Z.max():.2f}")
        if mae < 0.90:
            break
        col_x = np.arange(512)[cols]
        dZ_sum = np.zeros_like(Z, dtype=np.float64)
        w_sum = np.zeros_like(Z, dtype=np.float64)
        sig_xy = 6.2 if it < 6 else 4.3
        for ci, c_idx in enumerate(col_x):
            err_px = dy_px[ci]
            if abs(err_px) < 0.40:
                continue
            row_hit = int(np.clip(y_sky_r[c_idx] + 2, 0, 511))
            if not sil_r[row_hit, c_idx]:
                continue
            xw, yw = float(hx[row_hit, c_idx]), float(hy[row_hit, c_idx])
            dz_mm = np.clip(err_px * 0.30, -4.0, 4.0)
            rad = np.exp(-0.5 * (((X - xw) / sig_xy) ** 2 + ((Y - yw) / (sig_xy * 1.40)) ** 2))
            dZ_sum += dz_mm * rad
            w_sum += rad
        delta = np.where(w_sum > 1e-4, dZ_sum / np.maximum(w_sum, 1e-4), 0.0)
        delta = ndi.gaussian_filter(delta, 4.4) * smoothstep(2.0, 9.5, Z - Z_BASE) * taper
        Z = np.clip(Z + 0.68 * delta, Z_BASE, Z_BASE + 46.0)

    return Z


def render_clay_audits(Z, cam_p, hero_rgb, fg_hero):
    os.makedirs(WORK_DIR, exist_ok=True)
    top_rim = np.column_stack([
        R * np.cos(np.radians([270, 210, 150, 90, 30, 330])),
        R * np.sin(np.radians([270, 210, 150, 90, 30, 330])),
        np.full(6, Z_BASE),
    ])
    tu, tv, _ = project_pts_vec(cam_p, top_rim[:, 0], top_rim[:, 1], top_rim[:, 2], 1024.0, 1024.0)
    y_bezel = np.full(1024, 1024.0)
    for col in range(1024):
        if tu[2] <= col <= tu[3]:
            y_bezel[col] = tv[2] + (col - tu[2]) / max(tu[3] - tu[2], 1.0) * (tv[3] - tv[2])
        elif tu[3] < col <= tu[4]:
            y_bezel[col] = tv[3] + (col - tu[3]) / max(tu[4] - tu[3], 1.0) * (tv[4] - tv[3])

    clay_1024, sil_1024, _, _, _, _, _, _, _, _ = ray_march_heightfield(Z, cam_p, out_w=1024, out_h=1024, sun_rel=False)
    y_sky_h = np.array([np.where(fg_hero[:, c])[0].min() if fg_hero[:, c].any() else 1024 for c in range(1024)])
    y_sky_r = np.array([np.where(sil_1024[:, c])[0].min() if sil_1024[:, c].any() else 1024 for c in range(1024)])
    cols = (np.arange(1024) >= int(tu[2]) + 8) & (np.arange(1024) <= int(tu[4]) - 8) & (
        (y_sky_h < y_bezel - 4.0) | (y_sky_r < y_bezel - 4.0)
    )
    mae = float(np.mean(np.abs(y_sky_r[cols] - y_sky_h[cols])))
    print(f"[Clay Audit] 1024px Hero Skyline MAE = {mae:.2f} px (above bezel), Z_max = {Z.max():.2f} mm")

    pair = np.hstack([
        (np.clip(clay_1024, 0, 1) * 255).astype(np.uint8),
        (np.clip(hero_rgb, 0, 1) * 255).astype(np.uint8),
    ])
    out_pair = os.path.join(WORK_DIR, "v4_clay_vs_hero.png")
    Image.fromarray(pair).resize((1400, 700), Image.LANCZOS).save(out_pair)

    v6 = []
    for ang in [0, -60, -120, -180, -240, -300]:
        ci, _, _, _, _, _, _, _, _, _ = ray_march_heightfield(Z, cam_p, out_w=460, out_h=460, yaw_offset_deg=ang, sun_rel=True)
        pil = Image.fromarray((np.clip(ci, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(pil)
        try:
            fnt = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
        except Exception:
            fnt = ImageFont.load_default()
        dr.text((14, 12), f"{(-ang) % 360} deg", fill=(30, 40, 52), font=fnt)
        v6.append(np.asarray(pil))
    out_6 = os.path.join(WORK_DIR, "v4_clay_6views.png")
    Image.fromarray(np.vstack([np.hstack(v6[:3]), np.hstack(v6[3:])])).save(out_6)
    print(f"Saved {out_pair} and {out_6}")


def run_stage_guide():
    """Project de-lit front hero onto Z_2k and generate 2048x2048 orthographic hybrid guide for Nano Banana 360 completion."""
    os.makedirs(WORK_DIR, exist_ok=True)
    cam_p = np.load(os.path.join(REFS_DIR, "hero_cam_params.npy"))
    hero_rgb = np.asarray(Image.open(os.path.join(REFS_DIR, "hero_nanobanana.jpg")).convert("RGB"), dtype=np.float32) / 255.0
    Z_1k = np.load(os.path.join(WORK_DIR, "Z_geom_v4.npy")).astype(np.float64)

    N2 = 2048
    Z_2k = ndi.zoom(Z_1k, N2 / Z_1k.shape[0], order=3)
    lin = np.linspace(-R, R, N2)
    X, Y = np.meshgrid(lin, lin)
    dist_h = hex_dist(X, Y)
    dist_v = vert_dist(X, Y)
    dx = 2.0 * R / (N2 - 1)

    _, _, _, hx, hy, hz, cam_xyz, nx_g, ny_g, nz_g = ray_march_heightfield(Z_2k, cam_p, out_w=1024, out_h=1024)
    u_all, v_all, _ = project_pts_vec(cam_p, X, Y, Z_2k, 1024.0, 1024.0)
    cam_dist_1k = np.sqrt((hx - cam_xyz[0]) ** 2 + (hy - cam_xyz[1]) ** 2 + (hz - cam_xyz[2]) ** 2)
    dist_grid = np.sqrt((X - cam_xyz[0]) ** 2 + (Y - cam_xyz[1]) ** 2 + (Z_2k - cam_xyz[2]) ** 2)
    ui = np.clip(np.round(u_all), 0, 1023).astype(np.int32)
    vi = np.clip(np.round(v_all), 0, 1023).astype(np.int32)

    vdir = cam_xyz[None, None, :] - np.stack([X, Y, Z_2k], axis=-1)
    vdir /= np.linalg.norm(vdir, axis=-1, keepdims=True)
    ndotv = nx_g * vdir[..., 0] + ny_g * vdir[..., 1] + nz_g * vdir[..., 2]

    # Strict unstretched contiguous front visibility (zero bezel bleed, zero isolated rear fragments!)
    vis_raw = (dist_grid <= cam_dist_1k[vi, ui] + 0.35) & (dist_h >= 5.2) & (dist_v >= 11.4) & (ndotv > 0.25)
    # Reject accidental dark bezel pixels picked up near outer edge
    lum_hp = 0.2126 * hero_rgb[..., 0] + 0.7152 * hero_rgb[..., 1] + 0.0722 * hero_rgb[..., 2]
    sat_hp = np.max(hero_rgb, axis=-1) - np.min(hero_rgb, axis=-1)
    is_bezel_px = (lum_hp < 0.34) & (sat_hp < 0.055)
    vis_raw &= ~((dist_h < 8.5) & is_bezel_px[vi, ui])
    vis_er = ndi.binary_erosion(vis_raw, iterations=4)
    lbl_v, n_v = ndi.label(vis_er)
    if n_v > 0:
        cnts = np.bincount(lbl_v.ravel()); cnts[0] = 0
        vis_clean = (lbl_v == np.argmax(cnts))
    else:
        vis_clean = vis_er
    vis_w = ndi.gaussian_filter(vis_clean.astype(np.float32), sigma=6.5)
    vis_w = np.clip((vis_w - 0.14) / 0.72, 0.0, 1.0) * smoothstep(4.6, 5.8, dist_h) * smoothstep(10.8, 12.0, dist_v)

    fx = np.clip(u_all, 0, 1022.99); fy = np.clip(v_all, 0, 1022.99)
    x0 = fx.astype(np.int32); y0 = fy.astype(np.int32)
    tx_f = (fx - x0)[..., None]; ty_f = (fy - y0)[..., None]
    hero_proj = (
        hero_rgb[y0, x0] * (1 - tx_f) * (1 - ty_f) + hero_rgb[y0, x0 + 1] * tx_f * (1 - ty_f)
        + hero_rgb[y0 + 1, x0] * (1 - tx_f) * ty_f + hero_rgb[y0 + 1, x0 + 1] * tx_f * ty_f
    )

    # Gentle macro de-lighting matched to studio key light (+X, -Y/+Z) preserving crisp granite cracks & bright snow
    ld_s = np.array([0.66, -0.35, 0.66]); ld_s /= np.linalg.norm(ld_s)
    gy_sm, gx_sm = np.gradient(ndi.gaussian_filter(Z_2k, 2.4 / dx), dx, dx)
    nsm_len = np.sqrt(1.0 + gx_sm * gx_sm + gy_sm * gy_sm)
    ndotl_sm = np.clip((-gx_sm * ld_s[0] - gy_sm * ld_s[1] + nsm_len * 0.0 + ld_s[2]) / nsm_len, 0.0, 1.0)
    illum_macro = np.clip(0.66 + 0.42 * ndotl_sm, 0.56, 1.08)
    hero_delit = np.clip(hero_proj / (illum_macro[..., None] ** 0.46), 0.0, 1.0).astype(np.float32)

    # Top-down shaded clay (for Geometry-Locked Nano Banana orthographic paint-over!)
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

    # Semantic softly-tinted clay for unseen north/back faces so Nano Banana paints crisp granite, snow & moss
    z_sm = ndi.gaussian_filter(Z_2k, sigma=28.0)
    curv = Z_2k - z_sm
    snow_hint = smoothstep(8.0, 18.0, rel_z - 3.5 * curv) * (1.0 - smoothstep(37.0, 44.0, slope_deg)) * in_ter
    meadow_hint = (1.0 - smoothstep(3.2, 10.5, rel_z)) * (1.0 - smoothstep(28.0, 41.0, slope_deg)) * in_ter
    scree_hint = smoothstep(-0.2, -1.4, curv) * smoothstep(2.0, 11.0, rel_z) * (1.0 - smoothstep(28.0, 44.0, slope_deg)) * in_ter
    hint_col = np.array([0.52, 0.45, 0.37], dtype=np.float32)[None, None, :] * clay_s[..., None] * 1.45
    hint_col = hint_col * (1.0 - 0.75 * meadow_hint[..., None]) + (np.array([0.32, 0.44, 0.18]) * (0.75 + 0.50 * clay_s[..., None])) * (0.75 * meadow_hint[..., None])
    hint_col = hint_col * (1.0 - 0.70 * scree_hint[..., None]) + (np.array([0.74, 0.46, 0.25]) * (0.75 + 0.50 * clay_s[..., None])) * (0.70 * scree_hint[..., None])
    hint_col = hint_col * (1.0 - 0.85 * snow_hint[..., None]) + (np.array([0.92, 0.95, 0.98]) * (0.85 + 0.20 * clay_s[..., None])) * (0.85 * snow_hint[..., None])

    slate = np.array([0.13, 0.14, 0.155], dtype=np.float32)
    collar = 1.0 - smoothstep(ROAD_MARGIN, ROAD_MARGIN + 0.8, dist_h) * smoothstep(PAD_RADIUS, PAD_RADIUS + 0.8, dist_v)
    hybrid = np.clip(hero_delit * vis_w[..., None] + hint_col * (1.0 - vis_w[..., None]), 0.0, 1.0)
    hybrid = hybrid * (1.0 - collar[..., None]) + slate * collar[..., None]

    # Flip Y for PNG saving (row 0 = +R North)
    Image.fromarray(np.flipud((np.clip(clay_rgb, 0, 1) * 255).astype(np.uint8))).save(os.path.join(WORK_DIR, "clay_ortho_2k.png"))
    Image.fromarray(np.flipud((np.clip(hybrid, 0, 1) * 255).astype(np.uint8))).save(os.path.join(WORK_DIR, "ortho_hybrid_guide_2k.png"))
    np.savez_compressed(
        os.path.join(WORK_DIR, "hero_delit_2k.npz"),
        Z_2k=Z_2k.astype(np.float32),
        hero_delit=hero_delit.astype(np.float32),
        vis_w=vis_w.astype(np.float32),
    )
    print(f"[Guide Ready] front_vis_frac={vis_clean[dist_h >= 3.8].mean():.3f} -> clay_ortho_2k.png & ortho_hybrid_guide_2k.png")


def run_stage_clay():
    os.makedirs(WORK_DIR, exist_ok=True)
    cam_p = np.load(os.path.join(REFS_DIR, "hero_cam_params.npy"))
    fg_hero = np.load(os.path.join(REFS_DIR, "hero_fg_clean.npy"))
    hero_rgb = np.asarray(Image.open(os.path.join(REFS_DIR, "hero_nanobanana.jpg")).convert("RGB"), dtype=np.float32) / 255.0

    X, Y, Z, dist_h, dist_v, taper = build_base_alpine_dem(N=1024)
    Z = optimize_hero_skyline(X, Y, Z, taper, cam_p, fg_hero, iters=12)
    np.save(os.path.join(WORK_DIR, "Z_geom_v4.npy"), Z.astype(np.float32))
    render_clay_audits(Z, cam_p, hero_rgb, fg_hero)


def extract_hero_swatch_banks(hero_rgb, turn_180_rgb=None):
    """Extract crisp unblurred pure-material exemplar texture banks from verified coordinate-picked boxes."""
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
        prep_swatch(hero_rgb, 534, 574, 276, 330, 256),  # Warm sunlit golden-tan chiseled granite cliff (clean interior)
        prep_swatch(hero_rgb, 315, 360, 295, 348, 256),  # Sunlit left peak warm granite
        prep_swatch(hero_rgb, 468, 502, 258, 306, 256),  # Cool slate-grey central-left shadow granite
    ]
    if turn_180_rgb is not None:
        th, tw = turn_180_rgb.shape[:2]
        if tw >= 400 and th >= 350:
            granite_swatches.append(prep_swatch(turn_180_rgb, int(tw * 0.47), int(tw * 0.54), int(th * 0.27), int(th * 0.44), 256))
            granite_swatches.append(prep_swatch(turn_180_rgb, int(tw * 0.67), int(tw * 0.74), int(th * 0.29), int(th * 0.43), 256))

    scree_swatches = [
        prep_swatch(hero_rgb, 535, 572, 496, 536, 256),  # Center copper scree avalanche stones
        prep_swatch(hero_rgb, 550, 586, 458, 494, 256),  # Upper central copper scree chute
        prep_swatch(hero_rgb, 542, 576, 512, 548, 256),  # Fractured copper scree boulders
    ]
    meadow_swatches = [
        prep_swatch(hero_rgb, 528, 588, 666, 718, 256),  # Emerald alpine meadow with tiny pebbles
        prep_swatch(hero_rgb, 486, 540, 686, 732, 256),  # Lush lower valley green turf
        prep_swatch(hero_rgb, 474, 524, 672, 716, 256),  # Foreground emerald turf
    ]
    snow_swatches = [
        prep_swatch(hero_rgb, 574, 622, 392, 438, 256, sharpen=False),  # Pure bright glacier bowl
        prep_swatch(hero_rgb, 602, 636, 404, 436, 256, sharpen=False),  # Pure right glacier snowfield
        prep_swatch(hero_rgb, 446, 472, 386, 412, 256, sharpen=False),  # Pure clean central snow basin
    ]
    return {
        "granite": granite_swatches,
        "scree": scree_swatches,
        "meadow": meadow_swatches,
        "snow": snow_swatches,
    }


def synthesize_slope_aligned_exemplar(X, Y, Z, dist_h, dist_v, swatches):
    """
    Sharp single-winner Voronoi patch quilting (narrow 1.1mm seam feather, zero multi-cell blur)
    using rigid per-seed downhill orthonormal frames (det J == 1 everywhere -> ZERO UV fold swirls!).
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

    def quilt_bank(bank, cell_mm=10.5, scale_mm=10.5, align_slope=True, seed=10):
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
                    # Evaluate downhill tangent at the SEED center (jx, jy) -> constant rigid frame per Voronoi cell!
                    si_x = np.clip(np.round((jx + R) / dx).astype(np.int32), 0, W - 1)
                    si_y = np.clip(np.round((jy + R) / dx).astype(np.int32), 0, H - 1)
                    sdx_dn = dx_dn_grid[si_y, si_x]
                    sdy_dn = dy_dn_grid[si_y, si_x]
                    jit_a = (cell_rand(cx_i, cy_i, seed + 5) - 0.5) * 0.65
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

                uu = np.clip(u_loc, 0.03, 0.97)
                vv = np.clip(v_loc, 0.03, 0.97)
                su = np.clip((uu * (sz - 1)).astype(np.int32), 0, sz - 1)
                sv = np.clip((vv * (sz - 1)).astype(np.int32), 0, sz - 1)

                idx_s = (np.floor(cell_rand(cx_i, cy_i, seed + 4) * nb).astype(np.int32)) % nb
                c_samp = np.zeros((H, W, 3), dtype=np.float32)
                for bi in range(nb):
                    m = idx_s == bi
                    if np.any(m):
                        c_samp[m] = bank[bi][sv[m], su[m]]

                better1 = dist < d1
                better2 = (~better1) & (dist < d2)
                if np.any(better1):
                    d2[better1] = d1[better1]
                    col2[better1] = col1[better1]
                    d1[better1] = dist[better1]
                    col1[better1] = c_samp[better1]
                if np.any(better2):
                    d2[better2] = dist[better2]
                    col2[better2] = c_samp[better2]

        gap = d2 - d1
        t_seam = np.clip(0.5 - 0.45 * (gap / 1.15), 0.0, 0.5)[..., None]
        return np.clip(col1 * (1.0 - t_seam) + col2 * t_seam, 0.0, 1.0)

    print("  [Exemplar] Quilting crisp rigid-frame slope-aligned granite...")
    tex_granite = quilt_bank(swatches["granite"], cell_mm=8.5, scale_mm=10.0, align_slope=True, seed=101)
    # Subtle geometry-driven V-cleft AO & crisp micro-striation (no fake black stripe worms!)
    f_fine = ridge_fbm(X, Y, [1.65, 3.4], [0.55, 0.45], seed=709)
    crk_micro = 1.0 - 0.11 * smoothstep(26.0, 44.0, slope_deg) * smoothstep(0.18, 0.50, 1.0 - f_fine)
    cav_shade = np.clip(1.0 + 0.18 * np.clip(cav, -1.2, 1.0), 0.76, 1.12)
    tex_granite = np.clip((tex_granite - 0.42) * 1.10 + 0.42, 0.0, 1.0)
    tex_granite = np.clip(tex_granite * (crk_micro * cav_shade)[..., None], 0.0, 1.0)

    print("  [Exemplar] Quilting copper scree, emerald meadow & snow...")
    tex_scree = quilt_bank(swatches["scree"], cell_mm=7.5, scale_mm=8.8, align_slope=True, seed=202)
    tex_meadow = quilt_bank(swatches["meadow"], cell_mm=8.5, scale_mm=9.5, align_slope=False, seed=303)
    tex_snow = quilt_bank(swatches["snow"], cell_mm=9.5, scale_mm=11.0, align_slope=False, seed=404)

    f_mac = ridge_fbm(X, Y, [0.10, 0.24], [0.6, 0.4], seed=505)
    f_det = ridge_fbm(X, Y, [0.45, 1.15], [0.6, 0.4], seed=506)
    in_area = smoothstep(ROAD_MARGIN + 0.35, ROAD_MARGIN + 1.25, dist_h) * smoothstep(PAD_RADIUS + 0.30, PAD_RADIUS + 1.25, dist_v)

    # 1. Lush emerald alpine meadow across all gentle lower foothills & perimeter border (prevents rock strips on bezel lip!)
    rim_turf = (1.0 - smoothstep(4.2, 7.6, dist_h)) * (1.0 - smoothstep(1.2, 4.8, rel_z))
    meadow_m = np.maximum(
        (1.0 - smoothstep(3.8, 10.2, rel_z + 2.2 * (f_mac - 0.5))) * (1.0 - smoothstep(31.0, 42.0, slope_deg)),
        rim_turf,
    ) * in_area
    col = tex_granite * (1.0 - meadow_m[..., None]) + tex_meadow * meadow_m[..., None]

    # 2. Copper scree chutes in concave gullies & talus slopes (clear of outer road border)
    inner_scree = smoothstep(ROAD_MARGIN + 2.2, ROAD_MARGIN + 4.8, dist_h) * smoothstep(PAD_RADIUS + 1.5, PAD_RADIUS + 4.0, dist_v)
    scree_m = smoothstep(-0.18, -1.20, curv + 0.35 * (f_det - 0.5)) * smoothstep(2.2, 8.8, rel_z) * (1.0 - smoothstep(17.5, 29.0, rel_z)) * (1.0 - smoothstep(31.0, 44.0, slope_deg)) * inner_scree
    col = col * (1.0 - 0.88 * scree_m[..., None]) + tex_scree * (0.88 * scree_m[..., None])

    # 3. Broad Clean Glacial Snow Couloirs inside elevated North/Side cirques (rel_z > 5.5mm so meadows stay green around rim, zero >=44 deg snow!)
    snow_couloir = (
        smoothstep(5.5, 11.5, rel_z)
        * smoothstep(5.2, 11.5, rel_z - 3.6 * curv + 2.0 * (f_mac - 0.5))
        * (1.0 - smoothstep(0.05, 0.55, curv))
        * (1.0 - smoothstep(38.0, 43.4, slope_deg))
        * inner_scree
    )
    col = col * (1.0 - snow_couloir[..., None]) + tex_snow * snow_couloir[..., None]
    return np.clip(col, 0.0, 1.0)


def run_stage_fuse():
    """Fuse front hero_delit + crisp slope-aligned Nano Banana exemplar 360-deg completion & render 4K audits."""
    os.makedirs(WORK_DIR, exist_ok=True)
    cam_p = np.load(os.path.join(REFS_DIR, "hero_cam_params.npy"))
    fg_hero = np.load(os.path.join(REFS_DIR, "hero_fg_clean.npy"))
    hero_rgb = np.asarray(Image.open(os.path.join(REFS_DIR, "hero_nanobanana.jpg")).convert("RGB"), dtype=np.float32) / 255.0
    t180_path = os.path.join(REFS_DIR, "turn_view_180.png")
    t180_rgb = (np.asarray(Image.open(t180_path).convert("RGB"), dtype=np.float32) / 255.0) if os.path.exists(t180_path) else None
    Z_1k = np.load(os.path.join(WORK_DIR, "Z_geom_v4.npy")).astype(np.float64)

    N2 = 2048
    Z_2k = ndi.zoom(Z_1k, N2 / Z_1k.shape[0], order=3)
    lin = np.linspace(-R, R, N2)
    X, Y = np.meshgrid(lin, lin)
    dist_h = hex_dist(X, Y)
    dist_v = vert_dist(X, Y)
    dx = 2.0 * R / (N2 - 1)

    _, _, _, hx, hy, hz, cam_xyz, nx_g, ny_g, nz_g = ray_march_heightfield(Z_2k, cam_p, out_w=1024, out_h=1024)
    u_all, v_all, _ = project_pts_vec(cam_p, X, Y, Z_2k, 1024.0, 1024.0)
    cam_dist_1k = np.sqrt((hx - cam_xyz[0]) ** 2 + (hy - cam_xyz[1]) ** 2 + (hz - cam_xyz[2]) ** 2)
    dist_grid = np.sqrt((X - cam_xyz[0]) ** 2 + (Y - cam_xyz[1]) ** 2 + (Z_2k - cam_xyz[2]) ** 2)
    ui = np.clip(np.round(u_all), 0, 1023).astype(np.int32)
    vi = np.clip(np.round(v_all), 0, 1023).astype(np.int32)

    vdir = cam_xyz[None, None, :] - np.stack([X, Y, Z_2k], axis=-1)
    vdir /= np.linalg.norm(vdir, axis=-1, keepdims=True)
    ndotv = nx_g * vdir[..., 0] + ny_g * vdir[..., 1] + nz_g * vdir[..., 2]

    fg_safe = ndi.binary_erosion(fg_hero, iterations=2)
    lum_hp = 0.2126 * hero_rgb[..., 0] + 0.7152 * hero_rgb[..., 1] + 0.0722 * hero_rgb[..., 2]
    sat_hp = np.max(hero_rgb, axis=-1) - np.min(hero_rgb, axis=-1)
    is_bezel_px = (lum_hp < 0.29) & (sat_hp < 0.055)

    # Preserve front hero peaks & foreground foothills cleanly while blocking ALL neutral grey outer hex rails (lum < 0.62 & sat < 0.095)
    outer_rim = (dist_h < 7.6) | (dist_v < 12.6)
    is_rail_px = outer_rim & (sat_hp[vi, ui] < 0.095) & (lum_hp[vi, ui] < 0.62)
    vis_raw = (
        (dist_grid <= cam_dist_1k[vi, ui] + 0.55)
        & fg_safe[vi, ui]
        & (dist_h >= 4.45)
        & (dist_v >= 10.65)
        & (ndotv > 0.06)
        & ~is_rail_px
    )
    vis_er = ndi.binary_erosion(vis_raw, iterations=1)
    lbl_v, n_v = ndi.label(vis_er)
    if n_v > 0:
        cnts = np.bincount(lbl_v.ravel()); cnts[0] = 0
        vis_clean = (lbl_v == np.argmax(cnts))
    else:
        vis_clean = vis_er
    vis_w = ndi.gaussian_filter(vis_clean.astype(np.float32), sigma=3.8)
    side_skirt = np.where((Y < -24.0) & (np.abs(X) < 34.0), smoothstep(4.35, 5.4, dist_h), smoothstep(5.8, 8.2, dist_h))
    vis_w = np.clip((vis_w - 0.12) / 0.76, 0.0, 1.0) * side_skirt * smoothstep(10.55, 11.6, dist_v)

    fx = np.clip(u_all, 0, 1022.99); fy = np.clip(v_all, 0, 1022.99)
    x0 = fx.astype(np.int32); y0 = fy.astype(np.int32)
    tx_f = (fx - x0)[..., None]; ty_f = (fy - y0)[..., None]
    hero_proj = (
        hero_rgb[y0, x0] * (1 - tx_f) * (1 - ty_f) + hero_rgb[y0, x0 + 1] * tx_f * (1 - ty_f)
        + hero_rgb[y0 + 1, x0] * (1 - tx_f) * ty_f + hero_rgb[y0 + 1, x0 + 1] * tx_f * ty_f
    )

    ld_s = np.array([0.66, -0.35, 0.66]); ld_s /= np.linalg.norm(ld_s)
    gy_sm, gx_sm = np.gradient(ndi.gaussian_filter(Z_2k, 2.4 / dx), dx, dx)
    nsm_len = np.sqrt(1.0 + gx_sm * gx_sm + gy_sm * gy_sm)
    ndotl_sm = np.clip((-gx_sm * ld_s[0] - gy_sm * ld_s[1] + ld_s[2]) / nsm_len, 0.0, 1.0)
    illum_macro = np.clip(0.66 + 0.42 * ndotl_sm, 0.56, 1.08)
    hero_delit = np.clip(hero_proj / (illum_macro[..., None] ** 0.46), 0.0, 1.0).astype(np.float32)

    swatches = extract_hero_swatch_banks(hero_rgb, t180_rgb)
    back_ex = synthesize_slope_aligned_exemplar(X, Y, Z_2k, dist_h, dist_v, swatches)

    # Normalized 2-band Laplacian pyramid blend (prevents background/bezel pixels behind vis_w==0 from causing crest halos!)
    v_bin = (vis_w > 0.04).astype(np.float32)
    v_blur = ndi.gaussian_filter(v_bin, 8.0)
    h_num = ndi.gaussian_filter(hero_delit * v_bin[..., None], (8.0, 8.0, 0.0))
    b_low = ndi.gaussian_filter(back_ex, (8.0, 8.0, 0.0))
    h_low = np.where(v_blur[..., None] > 0.05, h_num / np.maximum(v_blur[..., None], 1e-4), b_low)
    w_low = np.clip(ndi.gaussian_filter(vis_w, sigma=6.5), 0.0, 1.0)[..., None]
    w_hi = vis_w[..., None]
    fused_low = h_low * w_low + b_low * (1.0 - w_low)
    fused_hi = (hero_delit - h_low) * w_hi + (back_ex - b_low) * (1.0 - w_hi)
    albedo_2k = np.clip(fused_low + fused_hi, 0.0, 1.0)

    # Clean machined brushed gunmetal slate hex bezel + crisp bevel catchlight & recessed pad rings (0mm Z change)
    rel_z = np.maximum(0.0, Z_2k - Z_BASE)
    on_border = np.maximum(
        1.0 - smoothstep(ROAD_MARGIN + 0.50, ROAD_MARGIN + 1.15, dist_h),
        1.0 - smoothstep(PAD_RADIUS + 0.40, PAD_RADIUS + 1.00, dist_v),
    )
    flat_mask = 1.0 - smoothstep(0.14, 0.65, rel_z)
    collar = np.clip(on_border * flat_mask, 0.0, 1.0)
    f_bez = ridge_fbm(X, Y, [0.4, 1.5], [0.5, 0.5], seed=888)
    slate_col = np.array([0.158, 0.164, 0.175], dtype=np.float32)[None, None, :] * (0.95 + 0.10 * f_bez[..., None])
    rim_hi = np.exp(-0.5 * ((dist_h - 0.90) / 0.30) ** 2) * 0.045
    curb_sh = np.exp(-0.5 * ((dist_h - 3.60) / 0.42) ** 2) * 0.055
    pad_groove = np.exp(-0.5 * ((dist_v - 8.2) / 0.32) ** 2) * 0.040
    pad_hi = np.exp(-0.5 * ((dist_v - 9.0) / 0.28) ** 2) * 0.028
    slate_col = np.clip(slate_col + (rim_hi - curb_sh - pad_groove + pad_hi)[..., None], 0.06, 0.28)
    albedo_2k = albedo_2k * (1.0 - collar[..., None]) + slate_col * collar[..., None]

    # Snow probability mask for 3D micro-sculpting (keep snow smooth drifts, carve ALL rocks/scree/boulders 3D!)
    sat_a = np.max(albedo_2k, axis=-1) - np.min(albedo_2k, axis=-1)
    lum_a = 0.2126 * albedo_2k[..., 0] + 0.7152 * albedo_2k[..., 1] + 0.0722 * albedo_2k[..., 2]
    snow_mask = smoothstep(0.68, 0.84, lum_a) * (1.0 - smoothstep(0.10, 0.20, sat_a)) * smoothstep(3.5, 5.0, dist_h)

    # Preview 2D-to-3D micro-relief rock sculpting on Z_2k for rapid visual check
    l_loc = np.maximum(ndi.gaussian_filter(lum_a, 24.0), 0.14)
    lum_eq = np.clip((lum_a / l_loc) * 0.55, 0.0, 1.35)
    lum_sc = 0.45 * lum_a + 0.55 * lum_eq
    l_f = ndi.gaussian_filter(lum_sc, 1.4)
    l_m = ndi.gaussian_filter(lum_sc, 8.5)
    crev = np.clip((l_m - l_f) - 0.032, 0.0, 0.55) * 1.95
    dome = ndi.gaussian_filter(np.clip((l_f - l_m) * 2.25, 0.0, 0.68), 1.6)
    core_m = smoothstep(3.7, 5.8, dist_h) * smoothstep(10.2, 12.6, dist_v)
    Z_prev = Z_2k + core_m * (1.0 - 0.90 * snow_mask) * (0.55 * dome - 0.98 * crev)
    Z_prev = Z_BASE + (Z_prev - Z_BASE) * (RELIEF_MM / max(float((Z_prev - Z_BASE).max()), 1e-4))

    # Render Hero 3-Panel Audit (Clay | Hero Original | Textured 3D) + 6-View 360-deg Textured Turnaround
    clay_1k, _, _, _, _, _, _, _, _, _ = ray_march_heightfield(Z_prev, cam_p, out_w=1024, out_h=1024, sun_rel=False)
    tex_1k, _, _, _, _, _, _, _, _, _ = ray_march_heightfield(Z_prev, cam_p, out_w=1024, out_h=1024, tex_rgb=albedo_2k, sun_rel=False)
    trip = np.hstack([
        (np.clip(clay_1k, 0, 1) * 255).astype(np.uint8),
        (np.clip(hero_rgb, 0, 1) * 255).astype(np.uint8),
        (np.clip(tex_1k, 0, 1) * 255).astype(np.uint8),
    ])
    out_trip = os.path.join(WORK_DIR, "v4_textured_vs_hero.png")
    Image.fromarray(trip).resize((1536, 512), Image.LANCZOS).save(out_trip)

    v6 = []
    for ang in [0, -60, -120, -180, -240, -300]:
        ri, _, _, _, _, _, _, _, _, _ = ray_march_heightfield(Z_prev, cam_p, out_w=460, out_h=460, tex_rgb=albedo_2k, yaw_offset_deg=ang, sun_rel=True)
        pil = Image.fromarray((np.clip(ri, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(pil)
        try:
            fnt = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
        except Exception:
            fnt = ImageFont.load_default()
        dr.text((14, 12), f"{(-ang) % 360} deg", fill=(30, 40, 52), font=fnt)
        v6.append(np.asarray(pil))
    out_6 = os.path.join(WORK_DIR, "v4_textured_6views.png")
    Image.fromarray(np.vstack([np.hstack(v6[:3]), np.hstack(v6[3:])])).save(out_6)

    # Save crisp 4K Albedo + 16-bit PNG & float32 .npy heightmap (lossless precision for Stage 4 STL/WebGL!)
    alb_flip = Image.fromarray(np.flipud(np.clip(albedo_2k * 255.0, 0, 255).astype(np.uint8)))
    alb_4k = alb_flip.resize((4096, 4096), Image.LANCZOS)
    a4 = np.asarray(alb_4k, dtype=np.float32) / 255.0
    blur4 = ndi.gaussian_filter(a4, (1.3, 1.3, 0.0))
    a4_sharp = np.clip(a4 + 0.48 * (a4 - blur4), 0.0, 1.0)
    out_alb4k = os.path.join(WORK_DIR, "montana_v4_albedo_4k.png")
    Image.fromarray((a4_sharp * 255.0).astype(np.uint8)).save(out_alb4k)

    # Save exact lossless heightmap (normalized [0..255] float32 .npy AND 16-bit PNG)
    hm_norm_255 = np.clip((Z_2k - Z_BASE) / RELIEF_MM * 255.0, 0.0, 255.0).astype(np.float32)
    np.save(os.path.join(WORK_DIR, "montana_v4_hm_2k.npy"), np.flipud(hm_norm_255))
    hm_u16 = np.clip(np.flipud(hm_norm_255) / 255.0 * 65535.0, 0, 65535).astype(np.uint16)
    Image.fromarray(hm_u16).save(os.path.join(WORK_DIR, "montana_v4_hm_16bit.png"))
    print(f"[Fuse OK] Saved {out_trip}, {out_6}, {out_alb4k}, and montana_v4_hm_2k.npy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["clay", "guide", "fuse", "build"], required=True)
    args = p.parse_args()
    if args.stage == "clay":
        run_stage_clay()
    elif args.stage == "guide":
        run_stage_guide()
    elif args.stage == "fuse":
        run_stage_fuse()


if __name__ == "__main__":
    main()
