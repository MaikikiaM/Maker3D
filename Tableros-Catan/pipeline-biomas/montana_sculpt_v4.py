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
    # Crisp V-crest profile
    profile = np.power(np.maximum(0.0, 1.0 - d / np.maximum(w_eff, 0.5)), sharp)
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

    # Natural organic undulating emerald meadow + crisp canyon-flanking valley banks
    hill_n = smooth_fbm(X, Y, [0.09, 0.21, 0.45, 0.95], [0.42, 0.30, 0.18, 0.10], seed=204)
    valley_ramp = 4.8 * smoothstep(-45.0, -4.0, Y) * np.exp(-0.5 * (X / 29.0) ** 2)
    bank_L = 3.4 * np.exp(-0.5 * (((X + 9.2) / 4.8) ** 2 + ((Y + 25.0) / 10.5) ** 2))
    bank_R = 3.6 * np.exp(-0.5 * (((X - 6.8) / 4.8) ** 2 + ((Y + 24.5) / 11.0) ** 2))
    bank_F = 2.4 * np.exp(-0.5 * (((X - 2.5) / 8.5) ** 2 + ((Y + 37.0) / 6.0) ** 2))
    Z_floor = in_hex * (0.9 + 1.5 * hill_n + valley_ramp + bank_L + bank_R + bank_F)

    # Compact central massif hubs (broad solid summit pyramid + smooth north core)
    r_core = np.hypot((X - 1.2) / 1.12, (Y - 12.2) / 0.92)
    Z_body = 33.0 * np.power(np.maximum(0.0, 1.0 - r_core / 39.5), 1.40)
    Z_body = smax(Z_body, 26.2 * np.power(np.maximum(0.0, 1.0 - np.hypot(X + 20.5, Y - 7.5) / 29.5), 1.36), k=4.5)
    Z_body = smax(Z_body, 27.6 * np.power(np.maximum(0.0, 1.0 - np.hypot(X - 21.5, Y - 9.5) / 30.0), 1.36), k=4.5)
    Z_body = smax(Z_body, 24.5 * np.power(np.maximum(0.0, 1.0 - np.hypot(X - 0.0, Y - 20.5) / 31.0), 1.40), k=4.5)
    Z_base_terrain = smax(Z_floor, Z_body, k=2.2)

    # Gentle broad glacial amphitheater sweep only in front cirque basins
    cirque_FL = 3.0 * np.exp(-0.5 * (((X + 10.2) / 8.8) ** 2 + ((Y + 2.0) / 11.5) ** 2))
    cirque_FR = 3.2 * np.exp(-0.5 * (((X - 12.6) / 9.0) ** 2 + ((Y + 1.0) / 11.8) ** 2))
    Z_base_terrain = smax(Z_floor, Z_base_terrain - (cirque_FL + cirque_FR), k=2.0)

    # Crisp landmark-calibrated ridges (continuous broad central pyramid -> V-notch -> Right Spire)
    ridges = [
        # 1. Dominant Central Pyramid (+0.7,+13.9,45.2) + solid Right Shoulder (5.2,13.2,37.8) + Central Prow
        (0.7, 13.9, 45.2,   5.2, 13.2, 37.8,   15.5, 13.5, 0.95, 1.0, 0.14),
        (5.2, 13.2, 37.8,  11.5, 11.8, 26.8,   13.5, 11.5, 0.96, 1.5, 0.13),
        (11.5, 11.8, 26.8, 19.6, 11.0, 37.5,   12.0, 15.0, 0.92, 8.4, 0.14),
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

        # 3. Right Massif (+19.6,+11.0, 37.5) -> (+24.8,+9.8, 35.2) -> (+33.1,-1.4, 22.4) -> (+40.1,-14.4, 9.5)
        (19.6, 11.0,  37.5, 24.8, 9.8,  35.2,   15.0, 14.8, 0.90, 17.8, 0.15),
        (24.8, 9.8,   35.2, 33.1, -1.4, 22.4,   14.8, 12.2, 0.94, 18.9, 0.15),
        (33.1, -1.4,  22.4, 40.1, -14.4, 9.5,   12.2, 8.4,  1.04, 19.6, 0.12),
        (33.1, -1.4,  20.0, 44.0, -7.5, 8.4,    11.5, 7.8,  1.06, 20.4, 0.12),
        # Front-Right Diagonal Copper Jagged Spine! (+33.1,-1.4) -> (+30.2,-18.8) -> (+19.9,-26.2) -> (+9.8,-34.1)
        (33.1, -1.4,  20.2, 30.2, -18.8, 12.4,  10.0, 8.0,  0.90, 21.1, 0.16),
        (30.2, -18.8, 12.4, 19.9, -26.2, 8.0,   8.0,  6.2,  0.90, 22.2, 0.18),
        (19.9, -26.2, 8.0,  9.8,  -34.1, 4.4,   6.2,  4.6,  0.98, 23.3, 0.14),

        # 4. Crisp 360-deg North Back Arêtes & Radial Buttress Spurs (zero ring creases!)
        (0.7, 13.9,   45.2, -0.8, 25.0, 32.2,   14.8, 12.8, 0.92, 24.6, 0.16),
        (-0.8, 25.0,  32.2,  1.5, 35.5, 19.5,   12.8, 9.8,  0.96, 25.7, 0.15),
        (1.5, 35.5,   19.5, -0.5, 43.5, 7.8,    9.8,  6.6,  1.06, 26.5, 0.12),
        (-21.1, 6.4,  36.2,-17.5, 21.5, 30.5,   15.0, 12.8, 0.92, 27.3, 0.15),
        (-17.5, 21.5, 30.5,-22.5, 32.5, 18.2,   12.8, 9.6,  0.96, 28.1, 0.15),
        (-22.5, 32.5, 18.2,-18.0, 41.0, 7.4,    9.6,  6.5,  1.08, 28.9, 0.12),
        (-21.1, 6.4,  32.0,-32.5, 17.0, 19.2,   13.5, 10.0, 0.96, 29.7, 0.14),
        (-32.5, 17.0, 19.2,-40.5, 9.5,  8.6,    10.0, 7.0,  1.08, 30.5, 0.12),
        (19.6, 11.0,  37.5, 20.5, 22.5, 31.2,   15.0, 12.8, 0.92, 31.3, 0.15),
        (20.5, 22.5,  31.2, 24.5, 33.0, 18.5,   12.8, 9.6,  0.96, 32.1, 0.15),
        (24.5, 33.0,  18.5, 19.5, 41.5, 7.2,    9.6,  6.5,  1.08, 32.9, 0.12),
        (24.8, 9.8,   31.5, 33.5, 17.5, 18.8,   13.5, 10.0, 0.96, 33.7, 0.14),
        (33.5, 17.5,  18.8, 41.5, 9.0,  8.2,    10.0, 7.0,  1.08, 34.5, 0.12),
        # North-central saddles & diagonal granite buttresses
        (-0.8, 25.0,  30.2,-17.5, 21.5, 28.8,   13.0, 12.0, 0.94, 35.3, 0.14),
        (-0.8, 25.0,  30.2, 20.5, 22.5, 29.2,   13.0, 12.0, 0.94, 36.1, 0.14),
        # Subsidiary North-West & North-East foothills spurs into north meadow
        (-17.5, 21.5, 24.0, -9.0, 34.0, 11.5,   9.5,  7.0,  0.98, 37.2, 0.14),
        (20.5, 22.5,  24.5, 10.5, 34.5, 11.8,   9.5,  7.0,  0.98, 38.4, 0.14),
    ]

    Z_ridges = np.zeros_like(Z_floor)
    for r in ridges:
        Z_ridges = np.maximum(Z_ridges, faceted_ridge(X, Y, *r))

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
        + 0.75 * np.exp(-0.5 * (((X + 10.5) / 6.0) ** 2 + ((Y - 26.5) / 7.2) ** 2))
        + 0.75 * np.exp(-0.5 * (((X - 11.0) / 6.0) ** 2 + ((Y - 26.5) / 7.2) ** 2))
    )
    flute_damp = np.clip(1.0 - 0.72 * in_snow_bowl, 0.20, 1.0)
    fl1 = ridge_fbm(X, Y, [0.30, 0.72, 1.58, 3.1], [0.46, 0.28, 0.16, 0.10], seed=314)
    fl2 = ridge_fbm(X * 0.85 + Y * 0.35, -X * 0.35 + Y * 0.85, [0.48, 1.25, 2.5], [0.55, 0.30, 0.15], seed=777)
    Z_m = Z_m + smoothstep(3.2, 18.0, Z_m) * flute_damp * (1.80 * (fl1 - 0.44) + 1.30 * (fl2 - 0.45))

    # Carve crisp incised S-meandering front canyon gorge between raised meadow banks
    t_g = np.clip((2.0 - Y) / 45.0, 0.0, 1.0)
    cx_g = (
        3.5 * (1.0 - t_g)
        - 3.8 * np.sin(t_g * math.pi * 0.85)
        - 3.2 * t_g
        - 1.4 * np.sin(t_g * math.pi * 2.2) * smoothstep(0.35, 0.85, t_g)
    )
    dist_g = np.abs(X - cx_g)
    gully_w = 1.35 + 1.15 * t_g
    gully_cut = smoothstep(2.5, -5.0, Y) * smoothstep(-47.5, -36.0, Y) * np.exp(-0.5 * (dist_g / gully_w) ** 2)
    # Subtle raised flanking canyon lips along both sides of the S-gorge
    gully_lips = 1.35 * smoothstep(2.5, -6.0, Y) * smoothstep(-46.5, -35.0, Y) * np.exp(-0.5 * ((dist_g - gully_w * 2.2) / (1.4 * gully_w)) ** 2)
    Z_m = (Z_m + gully_lips) * (1.0 - 0.86 * gully_cut) + 0.30 * (0.86 * gully_cut)

    # Gentle natural North valley chute (zero abrupt vertical slot!)
    t_ng = np.clip((Y - 21.0) / 24.0, 0.0, 1.0)
    cx_ng = 2.4 * np.sin(t_ng * math.pi * 1.1)
    ng_cut = smoothstep(20.0, 27.0, Y) * smoothstep(47.0, 38.0, Y) * np.exp(-0.5 * ((X - cx_ng) / (3.8 + 1.8 * t_ng)) ** 2)
    Z_m = Z_m * (1.0 - 0.24 * ng_cut) + (Z_floor * 0.70 + 1.0) * (0.24 * ng_cut)

    # Compact hex apron respecting ROAD=3.5mm and PAD=10.0mm cleanly
    taper = smoothstep(3.85, 7.8, dist_h) * smoothstep(10.25, 13.8, dist_v)
    Z = Z_BASE + Z_m * taper
    return X, Y, Z, dist_h, dist_v, taper


def optimize_hero_skyline(X, Y, Z, taper, cam_p, fg_hero, iters=9):
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
        sig_xy = 5.8 if it < 4 else 4.2
        for ci, c_idx in enumerate(col_x):
            err_px = dy_px[ci]
            if abs(err_px) < 0.45:
                continue
            row_hit = int(np.clip(y_sky_r[c_idx] + 2, 0, 511))
            if not sil_r[row_hit, c_idx]:
                continue
            xw, yw = float(hx[row_hit, c_idx]), float(hy[row_hit, c_idx])
            dz_mm = np.clip(err_px * 0.28, -4.0, 4.0)
            rad = np.exp(-0.5 * (((X - xw) / sig_xy) ** 2 + ((Y - yw) / (sig_xy * 1.25)) ** 2))
            dZ_sum += dz_mm * rad
            w_sum += rad
        delta = np.where(w_sum > 1e-4, dZ_sum / np.maximum(w_sum, 1e-4), 0.0)
        delta = ndi.gaussian_filter(delta, 2.8) * smoothstep(2.0, 9.5, Z - Z_BASE) * taper
        Z = np.clip(Z + 0.65 * delta, Z_BASE, Z_BASE + 46.0)

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


def run_stage_clay():
    os.makedirs(WORK_DIR, exist_ok=True)
    cam_p = np.load(os.path.join(REFS_DIR, "hero_cam_params.npy"))
    fg_hero = np.load(os.path.join(REFS_DIR, "hero_fg_clean.npy"))
    hero_rgb = np.asarray(Image.open(os.path.join(REFS_DIR, "hero_nanobanana.jpg")).convert("RGB"), dtype=np.float32) / 255.0

    X, Y, Z, dist_h, dist_v, taper = build_base_alpine_dem(N=1024)
    Z = optimize_hero_skyline(X, Y, Z, taper, cam_p, fg_hero, iters=9)
    np.save(os.path.join(WORK_DIR, "Z_geom_v4.npy"), Z.astype(np.float32))
    render_clay_audits(Z, cam_p, hero_rgb, fg_hero)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["clay", "guide", "fuse", "build"], required=True)
    args = p.parse_args()
    if args.stage == "clay":
        run_stage_clay()


if __name__ == "__main__":
    main()
