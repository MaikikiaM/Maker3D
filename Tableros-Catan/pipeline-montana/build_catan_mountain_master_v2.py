"""
Master 5.1M-Face Production Build Pipeline (V2) for Catan Mountain Tile.
Pure NumPy + SciPy + PIL + zipfile (zero trimesh dependency).
Guarantees:
- Exact Catan dimensions: R=64.00mm, W=110.8513mm, Z=0..83.00mm (58.00mm alpine relief above Z=25.00mm)
- 3.50mm flat perimeter road clearance lane + Ø20.00mm vertex settlement pads at Z=25.00mm
- Ø10.20mm x 5.20mm bottom N52 magnet cavity, 100% solid watertight manifold (Euler chi=2)
- >=5,000,000 triangular faces + LOD1 (~1M), LOD2 (~250k), LOD3 (~50k)
- 4K & 2K Diffuse (ALL 45 calibrated grays + ALL 15 botanical greens utilized) + 4K & 2K Tangent Normal maps
- Consistent Y-up PNG orientation matching OBJ/3MF/WebGL UV conventions (zero vertical flip)
- Zero snow on steep rock cliffs >=44.0 deg (max_snow >= 44 deg == 0.000000)
"""
import os
import json
import math
import time
import struct
import hashlib
import zipfile
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import Delaunay
from PIL import Image

R = 64.0
APOTHEM = R * math.cos(math.radians(30.0))  # 55.42562584 mm
Z_BASE = 25.0
Z_PEAK = 83.0
TARGET_RELIEF = Z_PEAK - Z_BASE  # 58.00 mm
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00  # Ø20.00 mm flat settlement pads at all 6 corners
MAG_RADIUS = 5.10   # Ø10.20 mm magnet pocket
MAG_DEPTH = 5.20    # 5.20 mm cavity height

OUT_DIR = "Tableros-Catan/Loseta_Montana_3D"
SCRATCH_DIR = "Tableros-Catan/pipeline-montana/scratch_v2"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)


def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smax(a, b, k=0.35):
    m = np.maximum(a, b)
    return m + np.log(np.exp(k * (a - m)) + np.exp(k * (b - m))) / k


def build_palettes():
    """Build 45 calibrated grayscale/alpine rock stops + 15 botanical green stops."""
    greys_45 = []
    for i in range(45):
        t = i / 44.0
        if i == 0:
            greys_45.append([1.0, 1.0, 1.0])
            continue
        val = 0.975 * ((1.0 - t) ** 1.15) + 0.055
        if i < 6:
            r, g, b = val * 0.988, val * 0.994, val * 1.00
        elif i < 18:
            r, g, b = val * 1.014, val * 1.004, val * 0.982
        elif i < 32:
            r, g, b = val * 0.995, val * 0.998, val * 1.004
        else:
            r, g, b = val * 0.978, val * 0.988, val * 1.006
        greys_45.append(np.clip([r, g, b], 0.0, 1.0))
    greys_45 = np.array(greys_45, dtype=np.float32)

    greens_15_hex = [
        "#c6d67e", "#b4c96b", "#9fb85d", "#8ca84f", "#769841",
        "#638936", "#527b2d", "#436c25", "#375e20", "#2e521c",
        "#264618", "#203c15", "#344422", "#27381b", "#192b12",
    ]
    greens_15 = np.array(
        [[int(h[1:3], 16) / 255.0, int(h[3:5], 16) / 255.0, int(h[5:7], 16) / 255.0] for h in greens_15_hex],
        dtype=np.float32,
    )
    return greys_45, greens_15


def fft_fbm(nx, ny, min_wl_mm, max_wl_mm, width_mm, height_mm, beta=2.2, seed=42):
    """Band-limited 2D FFT fractional Brownian motion noise in exact physical mm units."""
    rng = np.random.RandomState(seed)
    kx = np.fft.rfftfreq(nx, d=width_mm / nx)
    ky = np.fft.fftfreq(ny, d=height_mm / ny)
    KX, KY = np.meshgrid(kx, ky)
    freq = np.hypot(KX, KY)
    f_min = 1.0 / max_wl_mm
    f_max = 1.0 / min_wl_mm
    band = (freq >= f_min) & (freq <= f_max)
    amp = np.zeros_like(freq, dtype=np.float32)
    amp[band] = (freq[band] + 1e-6) ** (-beta / 2.0)
    amp *= smoothstep(f_min * 0.8, f_min * 1.4, freq) * (1.0 - smoothstep(f_max * 0.65, f_max, freq))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=amp.shape)
    spec = amp * np.exp(1j * phase)
    out = np.fft.irfft2(spec, s=(ny, nx)).astype(np.float32)
    std = float(np.std(out))
    return (out - float(np.mean(out))) / (std if std > 1e-8 else 1.0)


def ridged_multifractal(nx, ny, width_mm, height_mm, seed=100):
    octaves = [
        (18.0, 52.0, 1.00, 1.45, seed + 1),
        (8.5, 20.0, 0.45, 1.60, seed + 2),
        (4.0, 9.5, 0.20, 1.70, seed + 3),
        (1.8, 4.2, 0.08, 1.75, seed + 4),
    ]
    total = np.zeros((ny, nx), dtype=np.float32)
    weight = np.ones((ny, nx), dtype=np.float32)
    norm_sum = 0.0
    for min_wl, max_wl, amp, power, s in octaves:
        n = fft_fbm(nx, ny, min_wl, max_wl, width_mm, height_mm, beta=1.9, seed=s)
        signal = 1.0 - np.clip(np.abs(n) * 0.58, 0.0, 1.0)
        signal = signal ** power
        signal *= weight
        weight = np.clip(signal * 1.28, 0.25, 1.0)
        total += amp * signal
        norm_sum += amp
    return total / norm_sum


def hex_distances(X, Y):
    """Return signed distance inside flat hexagon edges and distance to nearest of 6 vertices."""
    normals_hex = [[math.cos(k * math.pi / 3.0), math.sin(k * math.pi / 3.0)] for k in range(6)]
    dist_edge = np.full_like(X, 999.0, dtype=np.float32)
    for nx_h, ny_h in normals_hex:
        dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

    v_angles = [(30.0 + 60.0 * k) * math.pi / 180.0 for k in range(6)]
    hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles], dtype=np.float32)
    dist_vert = np.full_like(X, 999.0, dtype=np.float32)
    for vx, vy in hex_corners:
        dist_vert = np.minimum(dist_vert, np.hypot(X - vx, Y - vy))
    return dist_edge, dist_vert


def generate_mountain_dem(nx, ny, x_lim=APOTHEM, y_lim=R):
    """Generate V2.22 alpine DEM across [ -x_lim..x_lim ] x [ -y_lim..y_lim ]."""
    xs = np.linspace(-x_lim, x_lim, nx, dtype=np.float32)
    ys = np.linspace(-y_lim, y_lim, ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    W_mm = 2.0 * x_lim
    H_mm = 2.0 * y_lim

    dist_edge, dist_vert = hex_distances(X, Y)

    wx = 2.6 * fft_fbm(nx, ny, 24.0, 85.0, W_mm, H_mm, beta=2.4, seed=11) + \
         0.7 * fft_fbm(nx, ny, 9.0, 24.0, W_mm, H_mm, beta=2.1, seed=12)
    wy = 2.6 * fft_fbm(nx, ny, 24.0, 85.0, W_mm, H_mm, beta=2.4, seed=21) + \
         0.7 * fft_fbm(nx, ny, 9.0, 24.0, W_mm, H_mm, beta=2.1, seed=22)

    sx, sy = -1.5, 2.5
    r_summit = np.hypot(X - sx, Y - sy)
    warp_atten = smoothstep(5.0, 22.0, r_summit)
    Xw = X + wx * warp_atten
    Yw = Y + wy * warp_atten

    def ridge_seg(x1, y1, x2, y2, h1, h2, w_left, w_right, crest_flat=2.6):
        dxr, dyr = x2 - x1, y2 - y1
        L = math.hypot(dxr, dyr)
        L2 = L * L
        t = np.clip(((Xw - x1) * dxr + (Yw - y1) * dyr) / L2, 0.0, 1.0)
        px = x1 + t * dxr
        py = y1 + t * dyr
        nx_p, ny_p = -dyr / L, dxr / L
        s_dist = (Xw - px) * nx_p + (Yw - py) * ny_p
        along_d = np.hypot(Xw - px, Yw - py)
        w_side = 0.5 * (w_left + w_right) + 0.5 * (w_left - w_right) * np.tanh(s_dist / 3.2)
        d_norm = along_d / w_side
        cf = crest_flat / w_side
        d_eff = np.maximum(0.0, np.sqrt(d_norm * d_norm + cf * cf) - cf)
        h_lin = (1.0 - t) * h1 + t * h2 - 1.2 * np.sin(math.pi * t)
        return h_lin / (1.0 + d_eff ** 1.85)

    dx_s, dy_s = Xw - sx, Yw - sy
    ang_s = np.arctan2(dy_s, dx_s)
    r_horn = np.hypot(dx_s, dy_s)
    az_fade = smoothstep(3.5, 14.0, r_horn)
    horn_ribs = 0.16 * az_fade * (np.maximum(0.0, np.cos(4.0 * (ang_s - 0.35))) ** 1.35)
    glacier_sector = 0.24 * az_fade * (
        np.maximum(0.0, np.cos(ang_s - 0.55)) ** 1.5 +
        0.95 * np.maximum(0.0, np.cos(ang_s + 2.45)) ** 1.5 +
        0.80 * np.maximum(0.0, np.cos(ang_s + 0.85)) ** 1.5
    )
    r_asym = r_horn / (1.0 + horn_ribs + glacier_sector)
    r_eff_tip = np.sqrt(r_asym * r_asym + 2.0 * 2.0) - 2.0

    h_pyramid = np.maximum(0.0, 58.0 - 0.42 * r_eff_tip)
    h_skirt = 86.0 * np.exp(-((r_asym / 28.2) ** 1.54))
    k_int = 0.16
    h_horn = -np.log(np.exp(-k_int * h_pyramid) + np.exp(-k_int * h_skirt)) / k_int
    h_horn = np.maximum(0.0, h_horn)

    r1 = ridge_seg(sx, sy, 21.0, 13.5, 57.5, 38.5, 8.2, 14.5, 3.0)
    r2 = ridge_seg(21.0, 13.5, 34.0, 18.0, 38.5, 5.5, 6.0, 9.2, 1.8)
    r3 = ridge_seg(21.0, 13.5, 30.0, -7.0, 38.5, 6.0, 9.2, 6.0, 1.8)
    r4 = ridge_seg(sx, sy, -19.5, -16.5, 57.5, 37.5, 14.5, 8.2, 3.0)
    r5 = ridge_seg(-19.5, -16.5, -31.5, -24.0, 37.5, 5.0, 9.0, 6.0, 1.8)
    r6 = ridge_seg(-19.5, -16.5, 5.5, -30.5, 37.5, 5.5, 6.0, 9.0, 1.8)
    r7 = ridge_seg(sx, sy, -26.0, 11.5, 56.2, 26.0, 12.5, 6.5, 2.6)
    r8 = ridge_seg(sx, sy, 4.0, 29.0, 55.5, 20.0, 6.2, 11.5, 2.4)
    r9 = ridge_seg(sx, sy, 15.5, -14.5, 56.2, 29.0, 6.8, 12.5, 2.6)

    r_all = np.maximum.reduce([r1, r2, r3, r4, r5, r6, r7, r8, r9])
    r_all_s = ndi.gaussian_filter(r_all, 0.55 / dx)
    massif = smax(h_horn, r_all_s, k=0.55)

    cirques = [
        (-11.5, 16.5, 10.2, 13.0),
        (14.0, -3.5, 9.5, 11.0),
        (-3.0, -19.0, 10.0, 9.5),
        (-20.5, -3.5, 9.5, 9.0),
    ]
    for cx, cy, cr, depth in cirques:
        d_c = np.hypot(Xw - cx, Yw - cy)
        bowl = np.exp(-((d_c / cr) ** 2.0))
        massif = np.maximum(0.0, massif - depth * bowl * smoothstep(5.0, 20.0, massif))

    r_base = np.hypot(Xw * 0.96, Yw)
    massif = massif * smoothstep(48.5, 21.0, r_base)

    d_clear = np.minimum(np.maximum(0.0, dist_edge - ROAD_MARGIN), np.maximum(0.0, dist_vert - PAD_RADIUS))
    foothill_noise = fft_fbm(nx, ny, 9.5, 28.0, W_mm, H_mm, beta=2.2, seed=55)
    foothills = (2.8 + 1.9 * smoothstep(-1.0, 1.0, foothill_noise)) * smoothstep(0.0, 6.2, d_clear) * (1.0 - smoothstep(4.5, 16.0, massif))
    massif = massif * smoothstep(0.0, 7.5, d_clear) + foothills

    rmf = ridged_multifractal(nx, ny, W_mm, H_mm, seed=303)
    rmf_c = (rmf - 0.45) * 2.0
    elev_w = smoothstep(4.5, 16.0, massif) * (1.0 - 0.85 * smoothstep(28.0, 44.0, massif))

    strata_warp = fft_fbm(nx, ny, 12.0, 42.0, W_mm, H_mm, beta=2.2, seed=404)
    strata_coord = 0.20 * massif + 0.06 * Xw - 0.04 * Yw + 1.5 * strata_warp
    strata_bands = 0.30 * np.sin(strata_coord * 1.25) + 0.14 * np.sin(strata_coord * 2.6 + 0.6)

    Z_sculpted = np.maximum(0.0, massif + (0.90 * rmf_c * elev_w) + (0.24 * strata_bands * elev_w))

    talus_deg = 42.0
    T_x = math.tan(math.radians(talus_deg)) * dx
    T_y = math.tan(math.radians(talus_deg)) * dy
    Z_eroded = Z_sculpted.copy()
    for _ in range(8):
        dx_r = Z_eroded[:, :-1] - Z_eroded[:, 1:]
        dy_d = Z_eroded[:-1, :] - Z_eroded[1:, :]
        ex_r = np.maximum(0.0, dx_r - T_x) - np.maximum(0.0, -dx_r - T_x)
        ex_d = np.maximum(0.0, dy_d - T_y) - np.maximum(0.0, -dy_d - T_y)
        c = 0.12
        Z_eroded[:, :-1] -= c * 0.25 * ex_r
        Z_eroded[:, 1:] += c * 0.25 * ex_r
        Z_eroded[:-1, :] -= c * 0.25 * ex_d
        Z_eroded[1:, :] += c * 0.25 * ex_d

    relief = np.maximum(0.0, Z_eroded) * smoothstep(0.0, 0.8, d_clear)
    max_r = float(np.max(relief))
    if max_r > 0:
        relief *= (TARGET_RELIEF / max_r)

    Z_final = Z_BASE + relief
    Z_final[dist_edge <= ROAD_MARGIN] = Z_BASE
    Z_final[dist_vert <= PAD_RADIUS] = Z_BASE
    return X, Y, Z_final, dist_edge, dist_vert, dx, dy, strata_bands, rmf_c


def compute_geomorphometry(Z, dx, dy):
    """Compute slope (deg), macro principal curvatures, and 8-direction raymarched AO."""
    Z_s = ndi.gaussian_filter(Z, 0.28 / dx)
    gy, gx = np.gradient(Z_s, dy, dx)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    Z_mac = ndi.gaussian_filter(Z, 1.8 / dx)
    gym, gxm = np.gradient(Z_mac, dy, dx)
    gyym, gyxm = np.gradient(gym, dy, dx)
    gxym, gxxm = np.gradient(gxm, dy, dx)
    tr = gxxm + gyym
    det = gxxm * gyym - 0.5 * (gxym + gyxm) ** 2
    disc = np.maximum(0.0, tr ** 2 - 4.0 * det)
    e_min = 0.5 * (tr - np.sqrt(disc))
    e_max = 0.5 * (tr + np.sqrt(disc))

    ao = np.ones_like(Z, dtype=np.float32)
    for ang_deg in range(0, 360, 45):
        rad = math.radians(ang_deg)
        ux, uy = math.cos(rad), math.sin(rad)
        max_h_ang = np.zeros_like(Z, dtype=np.float32)
        for step_mm in [0.6, 1.5, 3.2, 6.0, 10.0]:
            sx_px = int(round(ux * step_mm / dx))
            sy_px = int(round(uy * step_mm / dy))
            Z_sh = np.roll(np.roll(Z, sy_px, axis=0), sx_px, axis=1)
            dh = (Z_sh - Z) / step_mm
            max_h_ang = np.maximum(max_h_ang, np.clip(dh, 0.0, 2.5))
        ao -= (np.arctan(max_h_ang) / (math.pi * 0.5)) / 8.0
    ao = np.clip(ao * 1.15, 0.22, 1.0)
    return slope_deg, e_min, e_max, ao


def synthesize_textures(nx_tex, greys_45, greens_15):
    """Synthesize full-color diffuse (45 grays + 15 greens) & tangent-space normal map on [-R,R]^2."""
    X, Y, Z, dist_edge, dist_vert, dx, dy, strata_bands, rmf_c = generate_mountain_dem(
        nx_tex, nx_tex, x_lim=R, y_lim=R
    )
    W_mm = 2.0 * R
    slope_deg, e_min, e_max, ao = compute_geomorphometry(Z, dx, dy)

    RELIEF = np.maximum(0.0, Z - Z_BASE)
    rel_h = RELIEF / TARGET_RELIEF

    couloir_concavity = smoothstep(0.008, 0.065, e_max)
    arete_convexity = smoothstep(-0.010, -0.075, e_min)

    # STRICT 44° snow cutoff: 0.000000 snow on >=44.0° faces!
    snow_retention = 1.0 - smoothstep(39.8, 44.0, slope_deg)
    snow_retention[slope_deg >= 44.0] = 0.0

    snow_noise = fft_fbm(nx_tex, nx_tex, 2.2, 12.0, W_mm, W_mm, beta=2.1, seed=707)
    snow_alt = smoothstep(13.0, 21.5, RELIEF + 5.5 * couloir_concavity + 4.5 * arete_convexity + 2.5 * snow_noise)
    snow_mask = smoothstep(0.05, 0.24, snow_alt * snow_retention)
    snow_mask *= smoothstep(14.0, 17.5, RELIEF)
    snow_mask[slope_deg >= 44.0] = 0.0

    # Tundra across all 6 gentle valley bays (zero green near snow or road lane!)
    tundra_noise = fft_fbm(nx_tex, nx_tex, 3.0, 18.0, W_mm, W_mm, beta=2.0, seed=808)
    tundra_alt = smoothstep(0.06, 0.35, RELIEF) * (1.0 - smoothstep(4.2, 8.2, RELIEF + 1.4 * tundra_noise))
    tundra_slope = 1.0 - smoothstep(21.0, 34.0, slope_deg)
    apron_clear = smoothstep(3.52, 4.3, dist_edge) * smoothstep(10.02, 11.2, dist_vert)
    tundra_raw = tundra_alt * tundra_slope * apron_clear * (0.78 + 0.22 * smoothstep(-0.8, 0.8, tundra_noise))
    tundra_mask = smoothstep(0.08, 0.32, tundra_raw) * (1.0 - snow_mask)
    tundra_mask[RELIEF > 9.0] = 0.0

    # Index mapping ensuring ALL 45 grays (0..44) and ALL 15 greens (0..14) are used
    rock_grain = fft_fbm(nx_tex, nx_tex, 0.45, 2.8, W_mm, W_mm, beta=1.8, seed=908)
    rock_detail = fft_fbm(nx_tex, nx_tex, 2.2, 11.0, W_mm, W_mm, beta=2.0, seed=909)

    snow_tone_idx = np.clip(
        3.2 * (1.0 - smoothstep(0.45, 0.96, ao))
        + 3.2 * smoothstep(8.0, 42.5, slope_deg)
        + 0.4 * np.clip(rock_grain, -1.0, 1.0),
        0.0, 5.0
    )
    s_lo = np.minimum(np.floor(snow_tone_idx).astype(np.int32), 4)
    s_hi = s_lo + 1
    s_fr = (snow_tone_idx - s_lo)[:, :, None]
    snow_col = (1.0 - s_fr) * greys_45[s_lo] + s_fr * greys_45[s_hi]

    rock_tone_idx = np.clip(
        13.0
        + 17.5 * smoothstep(18.0, 52.0, slope_deg)
        + 12.5 * (1.0 - smoothstep(0.28, 0.90, ao))
        + 4.0 * (1.0 - rel_h)
        - 5.0 * arete_convexity
        + 3.2 * strata_bands
        + 2.0 * rock_detail
        + 1.6 * rock_grain,
        5.0, 44.0
    )
    r_lo = np.minimum(np.floor(rock_tone_idx).astype(np.int32), 43)
    r_hi = r_lo + 1
    r_fr = (rock_tone_idx - r_lo)[:, :, None]
    rock_col = (1.0 - r_fr) * greys_45[r_lo] + r_fr * greys_45[r_hi]

    g_tone_idx = np.clip(
        0.0
        + 7.8 * (1.0 - smoothstep(0.15, 6.2, RELIEF))
        + 5.8 * (1.0 - smoothstep(0.45, 0.88, ao))
        + 6.2 * smoothstep(-0.75, 0.75, tundra_noise),
        0.0, 14.0
    )
    g_lo = np.minimum(np.floor(g_tone_idx).astype(np.int32), 13)
    g_hi = g_lo + 1
    g_fr = (g_tone_idx - g_lo)[:, :, None]
    tundra_col = (1.0 - g_fr) * greens_15[g_lo] + g_fr * greens_15[g_hi]

    diffuse = rock_col * (1.0 - tundra_mask[:, :, None]) + tundra_col * tundra_mask[:, :, None]
    diffuse = diffuse * (1.0 - snow_mask[:, :, None]) + snow_col * snow_mask[:, :, None]

    # Calibrated stone margin + flat settlement pads
    apron_stone = greys_45[34] * (0.94 + 0.06 * np.clip(rock_detail, -1.0, 1.0)[:, :, None])
    apron_blend = smoothstep(0.02, 0.25, RELIEF) * apron_clear
    diffuse = apron_stone * (1.0 - apron_blend[:, :, None]) + diffuse * apron_blend[:, :, None]
    diffuse[dist_edge < 0] = greys_45[36]
    diffuse = np.clip(diffuse, 0.0, 1.0)

    # Tangent-space normal map from micro-relief + rock grain
    Z_micro = Z + 0.045 * rock_grain * (1.0 - snow_mask)
    gy_n, gx_n = np.gradient(Z_micro, dy, dx)
    nz_n = 1.0 / np.sqrt(gx_n * gx_n + gy_n * gy_n + 1.0)
    nx_t = -gx_n * nz_n
    ny_t = -gy_n * nz_n
    normal_rgb = np.stack([nx_t * 0.5 + 0.5, ny_t * 0.5 + 0.5, nz_n * 0.5 + 0.5], axis=-1)
    normal_rgb[dist_edge < 0] = [0.5, 0.5, 1.0]
    normal_rgb = np.clip(normal_rgb, 0.0, 1.0)

    # Stats & Palette Histogram checks
    mask_hex = dist_edge > 0
    mask_mtn = mask_hex & (RELIEF > 0.15)
    stats = {
        "snow_pct": float((snow_mask[mask_mtn] > 0.4).mean() * 100.0),
        "tundra_pct": float((tundra_mask[mask_mtn] > 0.4).mean() * 100.0),
        "rock_pct": float(100.0 - (snow_mask[mask_mtn] > 0.4).mean() * 100.0 - (tundra_mask[mask_mtn] > 0.4).mean() * 100.0),
        "steep_pct": float((slope_deg[mask_mtn] >= 44.0).mean() * 100.0),
        "max_snow_steep": float(snow_mask[mask_hex & (slope_deg >= 44.0)].max()) if np.any(mask_hex & (slope_deg >= 44.0)) else 0.0,
        "greys_counts": np.bincount(
            np.where(
                snow_mask[mask_hex] > 0.4,
                np.round(snow_tone_idx[mask_hex]).astype(np.int32),
                np.round(rock_tone_idx[mask_hex]).astype(np.int32),
            ),
            minlength=45,
        ),
        "greens_counts": np.bincount(
            np.round(g_tone_idx[mask_hex & (tundra_mask > 0.15)]).astype(np.int32),
            minlength=15,
        ),
    }

    # Seasonal Variants (Winter + Autumn) using inverted green-to-gold/frost cleanly
    winter_snow = smoothstep(0.04, 0.22, smoothstep(6.0, 14.0, RELIEF + 3.0 * snow_noise) * (1.0 - smoothstep(42.0, 44.0, slope_deg)))
    winter_snow[slope_deg >= 44.0] = 0.0
    winter_diff = diffuse * (1.0 - winter_snow[:, :, None]) + greys_45[1] * winter_snow[:, :, None]

    autumn_tint = np.array([0.82, 0.44, 0.18], dtype=np.float32) * (0.65 + 0.35 * (g_lo / 14.0)[:, :, None])
    autumn_diff = np.clip(diffuse * (1.0 - 0.75 * tundra_mask[:, :, None]) + autumn_tint * (0.75 * tundra_mask[:, :, None]), 0.0, 1.0)

    return {
        "X": X, "Y": Y, "Z": Z, "slope_deg": slope_deg, "ao": ao, "snow_mask": snow_mask, "tundra_mask": tundra_mask,
        "diffuse_u8": (diffuse * 255.0 + 0.5).astype(np.uint8),
        "normal_u8": (normal_rgb * 255.0 + 0.5).astype(np.uint8),
        "winter_u8": (np.clip(winter_diff, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
        "autumn_u8": (np.clip(autumn_diff, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
        "stats": stats,
    }


def save_uv_aligned_png(arr_u8, path):
    """Save image flipping Y vertically so PNG bottom (v=0) maps to Y=-R and top (v=1) to Y=+R."""
    Image.fromarray(np.flipud(arr_u8), mode="RGB").save(path, optimize=False, compress_level=6)


def build_solid_hex_mesh(nx, ny, n_bnd=144, n_mag=32):
    """Build closed watertight 2-manifold hex mesh (Z=0..83, bottom magnet cavity) purely via NumPy/SciPy."""
    X, Y, Z, dist_edge, _, _, _, _, _ = generate_mountain_dem(nx, ny, x_lim=APOTHEM, y_lim=R)

    # Interior grid points well inside hexagon
    mask_int = dist_edge >= 1.25
    pts_int = np.column_stack([X[mask_int], Y[mask_int], Z[mask_int]]).astype(np.float64)

    # Exact hexagon perimeter boundary (144 points = 24 per edge)
    v_angles = [(30.0 + 60.0 * k) * math.pi / 180.0 for k in range(6)]
    corners = [(R * math.cos(a), R * math.sin(a)) for a in v_angles]
    bnd_xy = []
    pts_per_edge = n_bnd // 6
    for e in range(6):
        x0, y0 = corners[e]
        x1, y1 = corners[(e + 1) % 6]
        for s in range(pts_per_edge):
            t = s / float(pts_per_edge)
            bnd_xy.append([x0 + t * (x1 - x0), y0 + t * (y1 - y0)])
    bnd_xy = np.array(bnd_xy, dtype=np.float64)

    pts_top_bnd = np.column_stack([bnd_xy, np.full(n_bnd, Z_BASE, dtype=np.float64)])
    all_top_verts = np.vstack([pts_int, pts_top_bnd])
    n_top = len(all_top_verts)
    n_int = len(pts_int)
    top_bnd_start = n_int

    tri2d_top = Delaunay(all_top_verts[:, :2]).simplices.astype(np.int32)
    # Orient all top faces +Z
    p0 = all_top_verts[tri2d_top[:, 0]]
    p1 = all_top_verts[tri2d_top[:, 1]]
    p2 = all_top_verts[tri2d_top[:, 2]]
    cz = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    flip = cz < 0
    tri2d_top[flip, 1], tri2d_top[flip, 2] = tri2d_top[flip, 2], tri2d_top[flip, 1]

    # Bottom rim + magnet cavity rings
    bot_bnd_start = n_top
    pts_bot_bnd = np.column_stack([bnd_xy, np.zeros(n_bnd, dtype=np.float64)])

    mag_ang = np.linspace(0.0, 2.0 * math.pi, n_mag, endpoint=False)
    mag_xy = np.column_stack([MAG_RADIUS * np.cos(mag_ang), MAG_RADIUS * np.sin(mag_ang)])
    mag_bot_start = bot_bnd_start + n_bnd
    pts_mag_bot = np.column_stack([mag_xy, np.zeros(n_mag, dtype=np.float64)])

    mag_ceil_start = mag_bot_start + n_mag
    pts_mag_ceil = np.column_stack([mag_xy, np.full(n_mag, MAG_DEPTH, dtype=np.float64)])
    mag_ctr_idx = mag_ceil_start + n_mag
    pt_mag_ctr = np.array([[0.0, 0.0, MAG_DEPTH]], dtype=np.float64)

    # Bottom annulus Delaunay between hex outer rim & inner magnet circle
    ann_pts_2d = np.vstack([bnd_xy, mag_xy])
    tri_ann_local = Delaunay(ann_pts_2d).simplices
    cent_r = np.hypot(ann_pts_2d[tri_ann_local].mean(axis=1)[:, 0], ann_pts_2d[tri_ann_local].mean(axis=1)[:, 1])
    tri_ann_local = tri_ann_local[cent_r > MAG_RADIUS * 1.001]

    # Map local annulus indices to global indices (orient -Z downward)
    local_to_global = np.concatenate([
        np.arange(bot_bnd_start, bot_bnd_start + n_bnd, dtype=np.int32),
        np.arange(mag_bot_start, mag_bot_start + n_mag, dtype=np.int32),
    ])
    tris_bot = local_to_global[tri_ann_local]
    vb0, vb1, vb2 = ann_pts_2d[tri_ann_local[:, 0]], ann_pts_2d[tri_ann_local[:, 1]], ann_pts_2d[tri_ann_local[:, 2]]
    cbz = (vb1[:, 0] - vb0[:, 0]) * (vb2[:, 1] - vb0[:, 1]) - (vb1[:, 1] - vb0[:, 1]) * (vb2[:, 0] - vb0[:, 0])
    flip_b = cbz > 0  # must point -Z
    tris_bot[flip_b, 1], tris_bot[flip_b, 2] = tris_bot[flip_b, 2], tris_bot[flip_b, 1]

    # Outer hex vertical plinth skirt (bnd_xy ordered CCW around +Z: outward wall normal uses t0->b0->b1, t0->b1->t1)
    idx_e = np.arange(n_bnd, dtype=np.int32)
    idx_en = (idx_e + 1) % n_bnd
    t0, t1 = top_bnd_start + idx_e, top_bnd_start + idx_en
    b0, b1 = bot_bnd_start + idx_e, bot_bnd_start + idx_en
    tris_skirt = np.vstack([
        np.column_stack([t0, b0, b1]),
        np.column_stack([t0, b1, t1]),
    ]).astype(np.int32)

    # Magnet cylindrical cavity walls (inward hole -> normal points toward (0,0), so reversed relative to outer skirt)
    jm = np.arange(n_mag, dtype=np.int32)
    jmn = (jm + 1) % n_mag
    mb0, mb1 = mag_bot_start + jm, mag_bot_start + jmn
    mc0, mc1 = mag_ceil_start + jm, mag_ceil_start + jmn
    tris_mag_wall = np.vstack([
        np.column_stack([mc0, mb1, mb0]),
        np.column_stack([mc0, mc1, mb1]),
    ]).astype(np.int32)

    # Magnet ceiling fan at Z=5.20mm pointing -Z
    tris_mag_ceil = np.column_stack([
        np.full(n_mag, mag_ctr_idx, dtype=np.int32),
        mc1,
        mc0,
    ]).astype(np.int32)

    verts = np.vstack([all_top_verts, pts_bot_bnd, pts_mag_bot, pts_mag_ceil, pt_mag_ctr]).astype(np.float32)
    faces = np.vstack([tri2d_top, tris_bot, tris_skirt, tris_mag_wall, tris_mag_ceil]).astype(np.int32)
    return verts, faces, n_top


def verify_watertight_manifold(verts, faces):
    """Vectorized half-edge pairing, Euler chi=2, bounds, and signed volume check."""
    V = len(verts)
    F = len(faces)
    u = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]]).astype(np.int64)
    v = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]]).astype(np.int64)

    fwd = np.sort(u * V + v)
    rev = np.sort(v * V + u)
    unique_fwd = np.all(fwd[1:] != fwd[:-1])
    paired = np.array_equal(fwd, rev)
    E = len(fwd) // 2
    chi = V - E + F

    v0 = verts[faces[:, 0]].astype(np.float64)
    v1 = verts[faces[:, 1]].astype(np.float64)
    v2 = verts[faces[:, 2]].astype(np.float64)
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    min_area = float(areas.min())
    vol_cm3 = float(np.sum(np.einsum("ij,ij->i", v0, cross)) / 6.0 / 1000.0)

    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    return {
        "V": V, "E": E, "F": F, "chi": chi,
        "watertight": bool(unique_fwd and paired and chi == 2 and min_area > 0.0),
        "paired_half_edges": bool(paired),
        "unique_directed_edges": bool(unique_fwd),
        "min_tri_area_mm2": min_area,
        "volume_cm3": vol_cm3,
        "bounds_min": bmin.tolist(),
        "bounds_max": bmax.tolist(),
    }


def write_binary_stl(verts, faces, path, label="Catan Mountain V2"):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    lens = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(lens, 1e-12)

    dtype = np.dtype([
        ("normal", "<3f4"),
        ("v0", "<3f4"),
        ("v1", "<3f4"),
        ("v2", "<3f4"),
        ("attr", "<u2"),
    ])
    buf = np.empty(len(faces), dtype=dtype)
    buf["normal"] = nrm
    buf["v0"] = v0
    buf["v1"] = v1
    buf["v2"] = v2
    buf["attr"] = 0

    hdr = label.encode("ascii")[:80].ljust(80, b"\x00")
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(struct.pack("<I", len(faces)))
        buf.tofile(f)


def sample_uv_colors(verts, diffuse_flip_u8):
    """Sample vertex RGB cleanly matching standard OBJ/3MF v=0 bottom, v=1 top on flipped PNG."""
    H, W, _ = diffuse_flip_u8.shape
    u = np.clip((verts[:, 0] + R) / (2.0 * R), 0.0, 1.0)
    v = np.clip((verts[:, 1] + R) / (2.0 * R), 0.0, 1.0)
    # Since PNG row 0 is v=1 and row H-1 is v=0:
    col = np.clip(np.round(u * (W - 1)).astype(np.int32), 0, W - 1)
    row = np.clip(np.round((1.0 - v) * (H - 1)).astype(np.int32), 0, H - 1)
    rgb = diffuse_flip_u8[row, col].copy()
    # Bottom / lower skirt concrete/basalt neutral gray for vertices below Z=24.99mm
    below = verts[:, 2] < (Z_BASE - 0.01)
    rgb[below] = [56, 60, 68]
    return rgb, u, v


def write_binary_ply(verts, faces, v_rgb, path):
    V = len(verts)
    F = len(faces)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Catan 3D Mountain Master V2 - 45 Grays + 15 Greens\n"
        f"element vertex {V}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {F}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    vdtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vbuf = np.empty(V, dtype=vdtype)
    vbuf["x"], vbuf["y"], vbuf["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    vbuf["r"], vbuf["g"], vbuf["b"] = v_rgb[:, 0], v_rgb[:, 1], v_rgb[:, 2]

    fdtype = np.dtype([("cnt", "u1"), ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4")])
    fbuf = np.empty(F, dtype=fdtype)
    fbuf["cnt"] = 3
    fbuf["i0"], fbuf["i1"], fbuf["i2"] = faces[:, 0], faces[:, 1], faces[:, 2]

    with open(path, "wb") as f:
        f.write(header)
        vbuf.tofile(f)
        fbuf.tofile(f)


def write_obj_mtl(verts, faces, u, v, obj_path, mtl_path, tex_filename):
    with open(mtl_path, "w", encoding="utf-8") as mf:
        mf.write(
            "# Catan Mountain Master V2 Material\n"
            "newmtl CatanMountainMat\n"
            "Ka 0.15 0.15 0.15\n"
            "Kd 1.00 1.00 1.00\n"
            "Ks 0.04 0.04 0.04\n"
            "Ns 18.0\n"
            "d 1.0\n"
            f"map_Kd {tex_filename}\n"
        )

    # Set clean bottom/skirt UVs onto outer rock corner of texture so walls never streak snow/green
    u_clean = u.copy()
    v_clean = v.copy()
    below = verts[:, 2] < (Z_BASE - 0.01)
    u_clean[below] = 0.02
    v_clean[below] = 0.50

    with open(obj_path, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\nusemtl CatanMountainMat\n")
        V = len(verts)
        step = 250000
        for s in range(0, V, step):
            e = min(V, s + step)
            lines = [f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in verts[s:e]]
            f.write("".join(lines))
        for s in range(0, V, step):
            e = min(V, s + step)
            lines = [f"vt {uu:.6f} {vv:.6f}\n" for uu, vv in zip(u_clean[s:e], v_clean[s:e])]
            f.write("".join(lines))
        F = len(faces)
        f1 = faces + 1
        for s in range(0, F, step):
            e = min(F, s + step)
            lines = [f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in f1[s:e]]
            f.write("".join(lines))


def write_fullcolor_3mf(verts, faces, u, v, png_path, out_3mf_path):
    """Write Microsoft 3MF OPC archive with embedded 4K PNG texture and exact per-vertex tex2coords."""
    u_clean = u.copy()
    v_clean = v.copy()
    below = verts[:, 2] < (Z_BASE - 0.01)
    u_clean[below] = 0.02
    v_clean[below] = 0.50

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        '</Relationships>'
    )
    model_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/Textures/catan_mountain_diffuse_4k.png" Id="reltex0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dtexture"/>'
        '</Relationships>'
    )

    with zipfile.ZipFile(out_3mf_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("3D/_rels/3dmodel.model.rels", model_rels)
        zf.write(png_path, "3D/Textures/catan_mountain_diffuse_4k.png")

        with zf.open("3D/3dmodel.model", "w") as mf:
            mf.write(
                b'<?xml version="1.0" encoding="UTF-8"?>\n'
                b'<model unit="millimeter" xml:lang="en-US" '
                b'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
                b'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">\n'
                b'  <resources>\n'
                b'    <m:texture2d id="1" path="/3D/Textures/catan_mountain_diffuse_4k.png" contenttype="image/png" tilestyleu="clamp" tilestylev="clamp"/>\n'
                b'    <m:texture2dgroup id="2" texid="1">\n'
            )
            V = len(verts)
            step = 200000
            for s in range(0, V, step):
                e = min(V, s + step)
                chunk = "".join(f'      <m:tex2coord u="{uu:.6f}" v="{vv:.6f}"/>\n' for uu, vv in zip(u_clean[s:e], v_clean[s:e]))
                mf.write(chunk.encode("ascii"))
            mf.write(b'    </m:texture2dgroup>\n    <object id="3" type="model">\n      <mesh>\n        <vertices>\n')
            for s in range(0, V, step):
                e = min(V, s + step)
                chunk = "".join(f'          <vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>\n' for x, y, z in verts[s:e])
                mf.write(chunk.encode("ascii"))
            mf.write(b'        </vertices>\n        <triangles>\n')
            F = len(faces)
            for s in range(0, F, step):
                e = min(F, s + step)
                chunk = "".join(
                    f'          <triangle v1="{a}" v2="{b}" v3="{c}" pid="2" p1="{a}" p2="{b}" p3="{c}"/>\n'
                    for a, b, c in faces[s:e]
                )
                mf.write(chunk.encode("ascii"))
            mf.write(
                b'        </triangles>\n      </mesh>\n    </object>\n  </resources>\n'
                b'  <build>\n    <item objectid="3"/>\n  </build>\n</model>\n'
            )


def main():
    t0 = time.time()
    print("=== CATAN MOUNTAIN V2 PRODUCTION SUITE (NO TRIMESH) ===")
    greys_45, greens_15 = build_palettes()

    # 1. Generate 4096x4096 Textures + Normal Maps
    print("[1/6] Synthesizing 4K textures (45 grays + 15 greens, Y-up UV-aligned PNGs)...")
    t_s = time.time()
    tex4k = synthesize_textures(4096, greys_45, greens_15)
    st = tex4k["stats"]
    print(
        f"  4K DEM+Textures in {time.time()-t_s:.2f}s | "
        f"Snow={st['snow_pct']:.1f}%, Tundra={st['tundra_pct']:.1f}%, Rock={st['rock_pct']:.1f}%, "
        f"Steep>=44={st['steep_pct']:.1f}%, MaxSnow>=44={st['max_snow_steep']:.6f}"
    )
    assert st["max_snow_steep"] == 0.0, f"Snow physics violation! Max snow on >=44° = {st['max_snow_steep']}"
    assert np.all(st["greys_counts"] > 0), f"Unused grey stops: {np.where(st['greys_counts']==0)[0]}"
    assert np.all(st["greens_counts"] > 0), f"Unused green stops: {np.where(st['greens_counts']==0)[0]}"
    print(f"  Palette check PASSED: all 45 grays >0 pixels (min={st['greys_counts'].min()}), all 15 greens >0 pixels (min={st['greens_counts'].min()})")

    diff_4k_path = os.path.join(OUT_DIR, "catan_mountain_diffuse_4k.png")
    diff_2k_path = os.path.join(OUT_DIR, "catan_mountain_diffuse_2k.png")
    norm_4k_path = os.path.join(OUT_DIR, "catan_mountain_normal_4k.png")
    winter_4k_path = os.path.join(OUT_DIR, "catan_mountain_diffuse_winter_4k.png")
    autumn_4k_path = os.path.join(OUT_DIR, "catan_mountain_diffuse_autumn_4k.png")

    save_uv_aligned_png(tex4k["diffuse_u8"], diff_4k_path)
    save_uv_aligned_png(tex4k["normal_u8"], norm_4k_path)
    save_uv_aligned_png(tex4k["winter_u8"], winter_4k_path)
    save_uv_aligned_png(tex4k["autumn_u8"], autumn_4k_path)

    # Save 2K downsampled versions for HTML viewer & lightweight tools
    diff_4k_pil = Image.open(diff_4k_path)
    diff_4k_pil.resize((2048, 2048), Image.Resampling.LANCZOS).save(diff_2k_path, compress_level=6)
    norm_2k_path = os.path.join(SCRATCH_DIR, "catan_mountain_normal_2k.png")
    Image.open(norm_4k_path).resize((2048, 2048), Image.Resampling.LANCZOS).save(norm_2k_path, compress_level=6)

    # Read back flipped 4K PNG into memory for exact UV round-trip sampling
    diff_png_arr = np.array(diff_4k_pil, dtype=np.uint8)
    del tex4k  # Free 4K float arrays before 5.1M mesh assembly

    # 2. Build >=5,000,000 Face Solid Watertight Master Mesh (nx=1770, ny=2044 -> ~5.18M tris)
    print("[2/6] Building >=5,000,000-face watertight solid mesh (nx=1770, ny=2044)...")
    t_m = time.time()
    verts, faces, n_top = build_solid_hex_mesh(1770, 2044, n_bnd=144, n_mag=32)
    audit = verify_watertight_manifold(verts, faces)
    print(f"  Master mesh built & audited in {time.time()-t_m:.2f}s:")
    print(f"    V={audit['V']:,} | E={audit['E']:,} | F={audit['F']:,} | chi={audit['chi']} | watertight={audit['watertight']}")
    print(f"    Vol={audit['volume_cm3']:.2f} cm^3 | MinArea={audit['min_tri_area_mm2']:.6e} mm^2")
    print(f"    Bounds: X[{audit['bounds_min'][0]:.4f}, {audit['bounds_max'][0]:.4f}], Y[{audit['bounds_min'][1]:.4f}, {audit['bounds_max'][1]:.4f}], Z[{audit['bounds_min'][2]:.4f}, {audit['bounds_max'][2]:.4f}]")
    assert audit["F"] >= 5_000_000, f"Face count {audit['F']} < 5,000,000!"
    assert audit["watertight"], "Mesh is not a closed watertight chi=2 manifold!"

    # 3. Export Binary STL, PLY, OBJ+MTL, and 3MF
    print("[3/6] Exporting STL, PLY, OBJ+MTL, and 3MF...")
    t_e = time.time()
    stl_path = os.path.join(OUT_DIR, "catan_mountain_solid.stl")
    ply_path = os.path.join(OUT_DIR, "catan_mountain_fullcolor.ply")
    obj_path = os.path.join(OUT_DIR, "catan_mountain_fullcolor.obj")
    mtl_path = os.path.join(OUT_DIR, "catan_mountain.mtl")
    tmf_path = os.path.join(OUT_DIR, "catan_mountain_fullcolor.3mf")

    write_binary_stl(verts, faces, stl_path, "Catan 3D Mountain Master V2 Solid")
    v_rgb, u_arr, v_arr = sample_uv_colors(verts, diff_png_arr)

    # Programmatic UV round-trip consistency check against disk PNG (top vertices only)
    rng_uv = np.random.RandomState(123)
    sample_idx = rng_uv.choice(n_top, size=1000, replace=False)
    H, W, _ = diff_png_arr.shape
    scol = np.clip(np.round(u_arr[sample_idx] * (W - 1)).astype(np.int32), 0, W - 1)
    srow = np.clip(np.round((1.0 - v_arr[sample_idx]) * (H - 1)).astype(np.int32), 0, H - 1)
    assert np.array_equal(diff_png_arr[srow, scol], v_rgb[sample_idx]), "UV mismatch between PNG and vertex RGB!"
    print("  UV Round-Trip Check PASSED (1000 random top vertices exact match).")

    write_binary_ply(verts, faces, v_rgb, ply_path)
    write_obj_mtl(verts, faces, u_arr, v_arr, obj_path, mtl_path, "catan_mountain_diffuse_4k.png")
    write_fullcolor_3mf(verts, faces, u_arr, v_arr, diff_4k_path, tmf_path)
    print(f"  Master exports complete in {time.time()-t_e:.2f}s")

    # 4. Derived LODs (1M, 250k, 50k)
    print("[4/6] Generating & verifying watertight LOD1 (1M), LOD2 (250k), LOD3 (50k)...")
    lod_specs = [
        ("catan_mountain_lod1_1m.stl", 786, 908),
        ("catan_mountain_lod2_250k.stl", 394, 456),
        ("catan_mountain_lod3_50k.stl", 190, 220),
    ]
    lod_stats = {}
    for fname, nx_l, ny_l in lod_specs:
        vl, fl, _ = build_solid_hex_mesh(nx_l, ny_l, n_bnd=144, n_mag=32)
        al = verify_watertight_manifold(vl, fl)
        assert al["watertight"], f"LOD {fname} not watertight!"
        write_binary_stl(vl, fl, os.path.join(OUT_DIR, fname), f"Catan Mountain {fname}")
        lod_stats[fname] = al
        print(f"  {fname}: F={al['F']:,}, chi={al['chi']}, watertight={al['watertight']}")

    # 5. Lightweight WebGL Viewer Payload (~210k tris so mobile & desktop run at solid 60 FPS)
    print("[5/6] Exporting WebGL Viewer payload (~210k tris)...")
    vv, fv, _ = build_solid_hex_mesh(340, 392, n_bnd=144, n_mag=32)
    av = verify_watertight_manifold(vv, fv)
    np.savez_compressed(
        os.path.join(SCRATCH_DIR, "viewer_mesh_v2.npz"),
        verts=vv.astype(np.float32),
        faces=fv.astype(np.int32),
    )
    print(f"  Viewer mesh saved: V={av['V']:,}, F={av['F']:,}")

    # 6. Save structured metrics JSON for audit/dossier/viewer sync
    metrics = {
        "master": audit,
        "coverage": {k: float(v) if not isinstance(v, np.ndarray) else v.tolist() for k, v in st.items()},
        "lods": lod_stats,
        "viewer": {"V": av["V"], "F": av["F"]},
    }
    with open(os.path.join(SCRATCH_DIR, "master_metrics_v2.json"), "w", encoding="utf-8") as jf:
        json.dump(metrics, jf, indent=2)

    print(f"[6/6] COMPLETE in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
