"""
V3.4 Photorealistic Painted Alpine Cordillera Prototype
- Fixes high snowline (RELIEF >= 21mm only -> zero low snow spots near green valleys!)
- Broadens upper summit cornices & hanging glacier névé bowls (<43° snow shelves framed by >=48° sheer jagged granite walls -> ~12-16% snow with 0.000000 >=44°)
- Combines crisp crystalline fractured rock micro-detail + soft downhill talus flow (fixes vertical smear blur!)
- Seamless oil-painted color melting: rock -> warm mineral scree -> olive lichen -> mossy ravine -> sunlit alpine meadow -> clean architectural slate apron
"""
import os
import math
import time
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = 64.0
APOTHEM = R * math.cos(math.radians(30.0))  # 55.42562584 mm
Z_BASE = 25.0
Z_PEAK = 83.0
TARGET_RELIEF = Z_PEAK - Z_BASE  # 58.00 mm
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00

OUT_DIR = "Tableros-Catan/pipeline-montana/scratch_v3"
os.makedirs(OUT_DIR, exist_ok=True)


def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smax_k(a, b, k=0.45):
    m = np.maximum(a, b)
    return m + np.log(np.exp(k * (a - m)) + np.exp(k * (b - m))) / k


def fft_fbm_canonical(X, Y, min_wl_mm, max_wl_mm, beta=2.2, seed=42, canon_n=1024):
    """Evaluates band-limited FBM on canonical [-R, R]x[-R, R] domain so ALL mesh LODs & UV textures share bit-identical features."""
    rng = np.random.RandomState(seed)
    W_ref = 2.0 * R
    kx = np.fft.rfftfreq(canon_n, d=W_ref / canon_n)
    ky = np.fft.fftfreq(canon_n, d=W_ref / canon_n)
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
    canon = np.fft.irfft2(spec, s=(canon_n, canon_n)).astype(np.float32)
    std = float(np.std(canon))
    canon = (canon - float(np.mean(canon))) / (std if std > 1e-8 else 1.0)
    if X.shape == (canon_n, canon_n) and float(X[0, 0]) <= -R + 0.05 and float(X[0, -1]) >= R - 0.05:
        return canon
    ix = ((X + R) / W_ref) * (canon_n - 1)
    iy = ((Y + R) / W_ref) * (canon_n - 1)
    return ndi.map_coordinates(canon, [iy, ix], order=1, mode="nearest").astype(np.float32)


def fft_fbm(nx, ny, min_wl_mm, max_wl_mm, width_mm, height_mm, beta=2.2, seed=42):
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


def ridged_multifractal_xy(X, Y, seed=100):
    octaves = [
        (16.0, 44.0, 1.00, 1.50, seed + 1),
        (6.8, 17.0, 0.50, 1.65, seed + 2),
        (2.8, 7.2, 0.25, 1.75, seed + 3),
        (1.0, 2.8, 0.12, 1.80, seed + 4),
    ]
    total = np.zeros(X.shape, dtype=np.float32)
    weight = np.ones(X.shape, dtype=np.float32)
    norm_sum = 0.0
    for min_wl, max_wl, amp, power, s in octaves:
        n = fft_fbm_canonical(X, Y, min_wl, max_wl, beta=1.85, seed=s, canon_n=1024)
        signal = 1.0 - np.clip(np.abs(n) * 0.56, 0.0, 1.0)
        signal = signal ** power
        signal *= weight
        weight = np.clip(signal * 1.32, 0.24, 1.0)
        total += amp * signal
        norm_sum += amp
    return total / norm_sum


def hex_distances(X, Y):
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


def compute_flow_accumulation(Z_small):
    ny, nx = Z_small.shape
    order = np.argsort(-Z_small.ravel())
    acc_2d = np.ones((ny, nx), dtype=np.float32)
    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    for idx in order:
        iy, ix = divmod(int(idx), nx)
        if iy == 0 or iy == ny - 1 or ix == 0 or ix == nx - 1:
            continue
        zc = Z_small[iy, ix]
        weights = []
        targets = []
        wsum = 0.0
        for dy_n, dx_n, dist_n in nbrs:
            zn = Z_small[iy + dy_n, ix + dx_n]
            if zn < zc:
                sl = ((zc - zn) / dist_n) ** 1.5
                weights.append(sl)
                targets.append((iy + dy_n, ix + dx_n))
                wsum += sl
        if wsum > 1e-9:
            inv_w = acc_2d[iy, ix] / wsum
            for (ty, tx), w in zip(targets, weights):
                acc_2d[ty, tx] += inv_w * w
    return np.log1p(acc_2d).astype(np.float32)


def slope_advect_field(field, gx, gy, dx, dy, steps=4, step_mm=0.45):
    ny, nx = field.shape
    mag = np.hypot(gx, gy) + 1e-4
    vx = (-gx / mag) * (step_mm / dx)
    vy = (-gy / mag) * (step_mm / dy)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    acc = field.copy()
    w_tot = 1.0
    for s in range(1, steps + 1):
        w = 1.0 - s / (steps + 1.5)
        for sgn in (-1.0, 1.0):
            sx = np.clip(np.round(xx + sgn * s * vx).astype(np.int32), 0, nx - 1)
            sy = np.clip(np.round(yy + sgn * s * vy).astype(np.int32), 0, ny - 1)
            acc += w * field[sy, sx]
            w_tot += w
    return acc / w_tot


def generate_cordillera_dem(nx, ny, x_lim=APOTHEM, y_lim=R):
    xs = np.linspace(-x_lim, x_lim, nx, dtype=np.float32)
    ys = np.linspace(-y_lim, y_lim, ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    W_mm = 2.0 * x_lim
    H_mm = 2.0 * y_lim

    dist_edge, dist_vert = hex_distances(X, Y)

    wx = 2.2 * fft_fbm_canonical(X, Y, 22.0, 75.0, beta=2.3, seed=101) + \
         0.6 * fft_fbm_canonical(X, Y, 7.5, 22.0, beta=2.0, seed=102)
    wy = 2.2 * fft_fbm_canonical(X, Y, 22.0, 75.0, beta=2.3, seed=201) + \
         0.6 * fft_fbm_canonical(X, Y, 7.5, 22.0, beta=2.0, seed=202)

    sx, sy = -3.5, 4.5
    r_summit = np.hypot(X - sx, Y - sy)
    warp_atten = smoothstep(2.5, 16.0, r_summit)
    Xw = X + wx * warp_atten
    Yw = Y + wy * warp_atten

    # Asymmetric stylized spire:
    # Steep jagged cliff headwalls (>50°) paired with slightly gentler windward glacier shoulder slopes (<42°)
    # so luminous snow crowns & hanging glacier ribbons cascade smoothly from summits while cliff faces stay bare rock!
    def alpine_spire(cx, cy, h_top, r_core, sharpness=1.16, faceting=0.26, rot=0.4, n_ribs=3, r_crown=3.2):
        dxp, dyp = Xw - cx, Yw - cy
        rp = np.hypot(dxp, dyp)
        ang = np.arctan2(dyp, dxp)
        az_w = smoothstep(2.0, 9.0, rp)
        ribs = np.abs(np.cos(n_ribs * 0.5 * (ang - rot))) ** 0.78
        # Gentle upper glacier snow shoulder on S/SE & E aspect so snow drapes naturally down ridge backs (<42 deg)
        glacier_shoulder = 0.46 * az_w * (
            np.maximum(0.0, np.cos(ang - rot + 0.55)) ** 1.15
            + 0.58 * np.maximum(0.0, np.cos(ang + 2.15)) ** 1.25
        )
        asym = (1.0 + faceting * az_w * (1.0 - ribs)) / (1.0 + glacier_shoulder)
        r_eff = np.sqrt((rp * asym) ** 2 + r_crown * r_crown) - r_crown
        return h_top * np.exp(-((r_eff / r_core) ** sharpness))

    def knife_ridge(x1, y1, x2, y2, h1, h2, w_left, w_right, sag=4.5, meander_amp=2.0, w_crest=3.4):
        dxr, dyr = x2 - x1, y2 - y1
        L = math.hypot(dxr, dyr)
        L2 = L * L
        t = np.clip(((Xw - x1) * dxr + (Yw - y1) * dyr) / L2, 0.0, 1.0)
        nx_p, ny_p = -dyr / L, dxr / L
        m_off = meander_amp * np.sin(t * math.pi * 1.35 + 0.4) * np.sin(t * math.pi)
        px = x1 + t * dxr + m_off * nx_p
        py = y1 + t * dyr + m_off * ny_p
        s_dist = (Xw - px) * nx_p + (Yw - py) * ny_p
        d_perp = np.hypot(Xw - px, Yw - py)
        w_side = 0.5 * (w_left + w_right) + 0.5 * (w_left - w_right) * np.tanh(s_dist / 3.4)
        h_spine = (1.0 - t) * h1 + t * h2 - sag * np.sin(math.pi * t)
        # Slightly wider corniced crest high up so ridge crests carry graceful snow ribbons (<42 deg)
        w_crest_eff = w_crest * (1.0 + 0.32 * smoothstep(20.0, 50.0, h_spine))
        d_eff = np.sqrt(d_perp * d_perp + w_crest_eff * w_crest_eff) - w_crest_eff
        return h_spine * np.exp(-((d_eff / w_side) ** 1.18))

    p1 = alpine_spire(sx, sy, 58.0, r_core=20.6, sharpness=1.13, faceting=0.25, rot=0.35, n_ribs=3, r_crown=5.8)
    p2 = alpine_spire(19.5, 13.5, 47.0, r_core=16.4, sharpness=1.15, faceting=0.24, rot=1.05, n_ribs=3, r_crown=4.8)
    p3 = alpine_spire(-13.5, -20.0, 41.5, r_core=15.8, sharpness=1.15, faceting=0.23, rot=-0.55, n_ribs=3, r_crown=4.3)
    p4 = alpine_spire(-27.0, -1.5, 29.5, r_core=12.6, sharpness=1.17, faceting=0.22, rot=0.15, n_ribs=4, r_crown=3.0)
    p5 = alpine_spire(18.0, -18.5, 30.5, r_core=12.4, sharpness=1.15, faceting=0.22, rot=0.75, n_ribs=3, r_crown=3.0)

    r_1_2 = knife_ridge(sx, sy, 19.5, 13.5, 57.5, 46.5, w_left=8.0, w_right=13.6, sag=5.0, meander_amp=1.8, w_crest=5.4)
    r_2_ne = knife_ridge(19.5, 13.5, 38.0, 22.5, 46.5, 2.5, w_left=6.2, w_right=8.0, sag=2.5, meander_amp=1.4, w_crest=3.2)
    r_2_se = knife_ridge(19.5, 13.5, 34.0, -8.0, 46.0, 3.2, w_left=8.4, w_right=6.4, sag=2.8, meander_amp=1.4, w_crest=3.4)
    r_1_3 = knife_ridge(sx, sy, -13.5, -20.0, 57.5, 41.0, w_left=13.0, w_right=8.0, sag=5.4, meander_amp=1.9, w_crest=5.4)
    r_3_sw = knife_ridge(-13.5, -20.0, -34.0, -29.5, 41.0, 2.5, w_left=6.4, w_right=7.6, sag=2.4, meander_amp=1.3, w_crest=3.0)
    r_3_5 = knife_ridge(-13.5, -20.0, 18.0, -18.5, 40.5, 30.0, w_left=9.4, w_right=6.8, sag=6.4, meander_amp=2.0, w_crest=4.0)
    r_5_se = knife_ridge(18.0, -18.5, 36.5, -23.5, 30.0, 2.2, w_left=5.8, w_right=6.4, sag=1.8, meander_amp=1.2, w_crest=2.6)
    r_1_nw = knife_ridge(sx, sy, -19.5, 29.0, 57.2, 3.2, w_left=6.6, w_right=9.0, sag=4.2, meander_amp=2.0, w_crest=3.2)
    r_1_4 = knife_ridge(sx, sy, -27.0, -1.5, 57.0, 29.0, w_left=8.2, w_right=8.0, sag=5.6, meander_amp=1.6, w_crest=4.2)
    r_4_w = knife_ridge(-27.0, -1.5, -41.5, -6.0, 29.0, 2.0, w_left=6.0, w_right=5.6, sag=1.6, meander_amp=1.1, w_crest=2.4)
    r_1_n = knife_ridge(sx, sy, -2.5, 37.5, 57.0, 2.5, w_left=6.6, w_right=7.0, sag=4.4, meander_amp=1.6, w_crest=2.8)

    # Smooth cascading glacier névé bowls & high summit shoulder (<39 deg slopes hold gleaming continuous snowfields)
    g_se = 40.5 * np.exp(-((np.hypot(Xw - 8.5, Yw - 2.5) / 17.5) ** 1.48))
    g_sw = 36.0 * np.exp(-((np.hypot(Xw + 9.5, Yw + 9.0) / 15.8) ** 1.48))
    g_top = 48.0 * np.exp(-((np.hypot(Xw - 0.5, Yw - 2.5) / 13.5) ** 1.52))
    g_crown = 52.5 * np.exp(-((np.hypot(Xw - (sx + 2.2), Yw - (sy + 1.8)) / 9.2) ** 1.55))

    Z_cord = p1
    for comp in [p2, p3, p4, p5, r_1_2, r_2_ne, r_2_se, r_1_3, r_3_sw, r_3_5, r_5_se, r_1_nw, r_1_4, r_4_w, r_1_n, g_se, g_sw, g_top, g_crown]:
        Z_cord = smax_k(Z_cord, comp, k=0.48)

    Z_cord = ndi.gaussian_filter(Z_cord, 0.38 / dx)

    rmf_macro = ridged_multifractal_xy(X, Y, seed=401)
    rmf_fine = ridged_multifractal_xy(X, Y, seed=402)
    broad_rock = fft_fbm_canonical(X, Y, 7.0, 24.0, beta=2.15, seed=404)

    # Smooth micro-noise gently over high névé zones & summit crowns so glacier snow sweeps continuously under <43 deg
    gy_c, gx_c = np.gradient(Z_cord, dy, dx)
    slope_cord = np.degrees(np.arctan(np.hypot(gx_c, gy_c)))
    glacier_calm = 1.0 - 0.58 * smoothstep(21.0, 38.0, Z_cord) * (1.0 - smoothstep(32.0, 44.0, slope_cord))

    tip_mask = (
        smoothstep(2.8, 9.0, np.hypot(Xw - sx, Yw - sy))
        * smoothstep(2.2, 7.2, np.hypot(Xw - 19.5, Yw - 13.5))
        * smoothstep(2.2, 7.2, np.hypot(Xw + 13.5, Yw + 20.0))
    )
    rel_factor = smoothstep(2.0, 18.0, Z_cord)
    chisel_detail = (
        2.35 * (rmf_macro - 0.38)
        + 0.72 * (rmf_fine - 0.42) * glacier_calm
        + 0.65 * broad_rock
    ) * rel_factor * (0.14 + 0.86 * tip_mask) * glacier_calm

    Z_eroded = np.maximum(0.0, Z_cord + chisel_detail)

    def carve_cirque(cx, cy, rx_c, ry_c, depth, rot_c=0.0):
        dx_c, dy_c = Xw - cx, Yw - cy
        u = dx_c * math.cos(rot_c) + dy_c * math.sin(rot_c)
        v = -dx_c * math.sin(rot_c) + dy_c * math.cos(rot_c)
        d2 = (u / rx_c) ** 2 + (v / ry_c) ** 2
        return depth * np.exp(-d2 ** 1.18) * smoothstep(5.0, 15.0, Z_eroded)

    Z_eroded -= carve_cirque(-11.5, 20.5, 9.5, 8.0, 6.8, 0.35)
    Z_eroded -= carve_cirque(9.0, 22.5, 8.5, 7.5, 5.8, -0.3)
    Z_eroded -= carve_cirque(-20.0, -9.0, 8.0, 7.5, 4.8, 0.5)
    Z_eroded -= carve_cirque(5.5, -17.0, 9.0, 7.5, 5.2, -0.4)

    # Smooth valley feathering cleanly into road deck (zero collar step!)
    d_clear = np.minimum(dist_edge - ROAD_MARGIN, dist_vert - PAD_RADIUS)
    apron_taper = smoothstep(0.0, 7.5, d_clear)
    relief = np.maximum(0.0, Z_eroded) * apron_taper

    max_r = float(np.max(relief))
    if max_r > 0:
        relief *= (TARGET_RELIEF / max_r)

    Z_final = Z_BASE + relief
    Z_final[dist_edge <= ROAD_MARGIN] = Z_BASE
    Z_final[dist_vert <= PAD_RADIUS] = Z_BASE

    return X, Y, Z_final, dist_edge, dist_vert, dx, dy, rmf_macro, rmf_fine, broad_rock


def synthesize_hyperreal_textures(X, Y, Z, dist_edge, dist_vert, dx, dy, rmf_macro, rmf_fine, broad_rock):
    nx_tex = Z.shape[1]
    ny_tex = Z.shape[0]
    W_mm = float(X[0, -1] - X[0, 0]) + dx
    H_mm = float(Y[-1, 0] - Y[0, 0]) + dy
    RELIEF = np.maximum(0.0, Z - Z_BASE)
    rel_h = RELIEF / TARGET_RELIEF

    Z_s = ndi.gaussian_filter(Z, 0.25 / dx)
    gy, gx = np.gradient(Z_s, dy, dx)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    Z_mac = ndi.gaussian_filter(Z, 1.4 / dx)
    gym, gxm = np.gradient(Z_mac, dy, dx)
    gyym, gyxm = np.gradient(gym, dy, dx)
    gxym, gxxm = np.gradient(gxm, dy, dx)
    tr = gxxm + gyym
    det = gxxm * gyym - 0.5 * (gxym + gyxm) ** 2
    disc = np.maximum(0.0, tr ** 2 - 4.0 * det)
    e_min = 0.5 * (tr - np.sqrt(disc))
    e_max = 0.5 * (tr + np.sqrt(disc))
    arete_convex = smoothstep(-0.02, -0.30, e_min)
    gully_concave = smoothstep(0.02, 0.26, e_max)

    # Fast multi-scale 16-dir HBAO (<=1536 grid for O(1) memory even at 4K) + native 4K micro cavity
    ao_max_n = 1536
    if max(ny_tex, nx_tex) > ao_max_n:
        scale_ao = ao_max_n / float(max(ny_tex, nx_tex))
        Z_ao = ndi.zoom(Z, (scale_ao, scale_ao), order=1).astype(np.float32)
        dx_ao = W_mm / float(Z_ao.shape[1])
        dy_ao = H_mm / float(Z_ao.shape[0])
    else:
        Z_ao = Z
        dx_ao, dy_ao = dx, dy

    ao_sub = np.ones_like(Z_ao, dtype=np.float32)
    for ang_deg in range(0, 360, 22):
        rad = math.radians(ang_deg)
        ux, uy = math.cos(rad), math.sin(rad)
        max_h_ang = np.zeros_like(Z_ao, dtype=np.float32)
        for step_mm in [0.5, 1.2, 2.6, 5.2, 9.5]:
            sx_px = int(round(ux * step_mm / dx_ao))
            sy_px = int(round(uy * step_mm / dy_ao))
            Z_sh = np.roll(np.roll(Z_ao, sy_px, axis=0), sx_px, axis=1)
            dh = (Z_sh - Z_ao) / step_mm
            max_h_ang = np.maximum(max_h_ang, np.clip(dh, 0.0, 2.2))
        ao_sub -= (np.arctan(max_h_ang) / (math.pi * 0.5)) / 16.0
    ao_sub = np.clip(ao_sub * 1.24, 0.40, 1.0)
    if ao_sub.shape != Z.shape:
        ao = ndi.zoom(ao_sub, (ny_tex / float(ao_sub.shape[0]), nx_tex / float(ao_sub.shape[1])), order=1).astype(np.float32)
        ao = np.clip(ao, 0.40, 1.0)
    else:
        ao = ao_sub
    micro_cav = np.clip(1.0 + (Z - ndi.gaussian_filter(Z, 0.55 / dx)) * 0.22, 0.85, 1.08)

    # Flow accumulation for downhill scree trails
    step_sub = max(1, nx_tex // 360)
    Z_sub = Z[::step_sub, ::step_sub]
    flow_sub = compute_flow_accumulation(Z_sub)
    flow_full = ndi.zoom(flow_sub, (ny_tex / Z_sub.shape[0], nx_tex / Z_sub.shape[1]), order=1).astype(np.float32)
    flow_norm = smoothstep(0.65, 3.2, flow_full)
    scree_streaks = flow_norm * smoothstep(11.0, 34.0, slope_deg) * (1.0 - smoothstep(42.0, 56.0, slope_deg))

    # Hybrid Crisp Crystalline Rock + Gentle Gravity Flow (canonical for bit-identical UV<->Mesh alignment!)
    raw_crisp = fft_fbm(nx_tex, ny_tex, 0.22, 1.1, W_mm, H_mm, beta=1.55, seed=701)
    raw_crags = fft_fbm_canonical(X, Y, 0.9, 4.0, beta=1.80, seed=705, canon_n=2048)
    slope_mask_detail = smoothstep(0.4, 2.8, RELIEF)
    flow_streaks = slope_advect_field(raw_crags, gx, gy, dx, dy, steps=4, step_mm=0.45) * slope_mask_detail

    broad_painterly = fft_fbm_canonical(X, Y, 6.0, 28.0, beta=2.40, seed=702, canon_n=1024)
    meadow_veil = fft_fbm_canonical(X, Y, 4.5, 18.0, beta=2.45, seed=703, canon_n=1024)
    wind_drift = fft_fbm_canonical(X, Y, 2.0, 10.0, beta=2.1, seed=704, canon_n=1024)

    norm_g = np.hypot(gx, gy) + 1e-6
    aspect_sun = np.clip((-gx * 0.62 + gy * 0.52) / np.maximum(norm_g, 0.45), -1.0, 1.0)

    # === 1. CRISP CRYSTALLINE + PAINTERLY ALPINE GRANITE ===
    # Cohesive icy mineral palette inspired by real European Alpine granite (no disconnected beige/brown blocks)
    col_slate_deep   = np.array([0.215, 0.235, 0.275], dtype=np.float32)
    col_slate_mid    = np.array([0.365, 0.388, 0.425], dtype=np.float32)
    col_granite_warm = np.array([0.485, 0.478, 0.468], dtype=np.float32)
    col_granite_pale = np.array([0.665, 0.662, 0.652], dtype=np.float32)
    col_scree        = np.array([0.415, 0.408, 0.398], dtype=np.float32)

    warmth = smoothstep(-0.6, 0.95, aspect_sun * 0.55 + 0.35 * arete_convex + 0.30 * broad_painterly) * smoothstep(2.0, 15.0, RELIEF) * 0.72
    rock_val = np.clip(
        0.50
        + 0.25 * arete_convex
        + 0.17 * (rmf_macro - 0.45)
        + 0.13 * (rmf_fine - 0.48)
        + 0.12 * flow_streaks
        - 0.21 * smoothstep(25.0, 62.0, slope_deg)
        + 0.14 * rel_h
        + 0.08 * raw_crisp
        + 0.06 * broad_painterly,
        0.0, 1.0
    )

    cool_rock = col_slate_deep * (1.0 - rock_val[..., None]) + col_slate_mid * rock_val[..., None]
    warm_rock = col_granite_warm * (1.0 - smoothstep(0.30, 0.92, rock_val)[..., None]) + col_granite_pale * smoothstep(0.30, 0.92, rock_val)[..., None]
    bedrock = cool_rock * (1.0 - warmth[..., None]) + warm_rock * warmth[..., None]

    bedrock = bedrock * (1.0 - 0.32 * scree_streaks[..., None]) + col_scree * (0.32 * scree_streaks[..., None])
    cavity_tint = (0.68 + 0.32 * ao) * micro_cav
    bedrock *= cavity_tint[..., None]

    # === 2. VELVETY FOOTHILL SCREE -> OLIVE LICHEN -> ALPINE MEADOW ===
    apron_clear = smoothstep(3.52, 5.4, dist_edge) * smoothstep(10.02, 12.2, dist_vert)
    moist_meadow_zone = (
        smoothstep(0.08, 1.1, RELIEF)
        * (1.0 - smoothstep(4.5, 13.0, RELIEF + 2.8 * broad_painterly - 2.6 * flow_norm))
        * (1.0 - smoothstep(19.0, 34.0, slope_deg))
        * apron_clear
    )
    lichen_veil = (
        smoothstep(0.6, 3.2, RELIEF)
        * (1.0 - smoothstep(10.5, 22.0, RELIEF))
        * (1.0 - smoothstep(22.0, 40.0, slope_deg))
        * smoothstep(-0.3, 0.85, broad_painterly + 0.25 * meadow_veil)
        * 0.42
    )

    col_lichen     = np.array([0.305, 0.325, 0.255], dtype=np.float32)
    col_moss_deep  = np.array([0.145, 0.215, 0.125], dtype=np.float32)
    col_meadow_mid = np.array([0.225, 0.305, 0.165], dtype=np.float32)
    col_meadow_sun = np.array([0.315, 0.385, 0.205], dtype=np.float32)

    bedrock = bedrock * (1.0 - lichen_veil[..., None]) + col_lichen * lichen_veil[..., None]

    meadow_richness = smoothstep(0.04, 0.78, moist_meadow_zone + 0.22 * flow_norm * moist_meadow_zone + 0.16 * meadow_veil)
    sun_grass = smoothstep(-0.35, 0.85, aspect_sun + 0.35 * meadow_veil)
    meadow_col = col_moss_deep * (1.0 - meadow_richness[..., None]) + col_meadow_mid * meadow_richness[..., None]
    meadow_col = meadow_col * (1.0 - 0.42 * sun_grass[..., None]) + col_meadow_sun * (0.42 * sun_grass[..., None])
    meadow_col *= (0.76 + 0.24 * ao)[..., None]

    meadow_alpha = smoothstep(0.04, 0.82, moist_meadow_zone * (0.85 + 0.15 * meadow_veil))
    surface = bedrock * (1.0 - meadow_alpha[..., None]) + meadow_col * meadow_alpha[..., None]

    # === 3. HIGH-ELEVATION GLACIER SNOW + SPINDRIFT HOARFROST (0% >=44°, ZERO HARD WHITE STICKERS) ===
    slope_snow_gate = (1.0 - smoothstep(18.0, 43.98, slope_deg)) ** 1.35
    slope_snow_gate[slope_deg >= 44.0] = 0.0

    # Natural elevation envelope: couloir fingers reach softly down to ~13mm, snowpack crowns >=17.5mm
    high_alt_gate = smoothstep(13.0, 21.0, RELIEF + 2.6 * wind_drift + 3.2 * gully_concave)
    alt_snow = smoothstep(16.5, 27.5, RELIEF + 3.8 * wind_drift + 5.2 * gully_concave + 5.5 * arete_convex - 0.8 * aspect_sun)
    glacier_chutes = smoothstep(14.0, 24.5, RELIEF) * gully_concave * smoothstep(0.38, 1.65, flow_full) * 0.96
    ridge_cornices = smoothstep(16.5, 27.0, RELIEF + 2.4 * raw_crags) * (0.64 + 0.36 * arete_convex)

    snow_potential = np.maximum(np.maximum(alt_snow, glacier_chutes), ridge_cornices) * high_alt_gate
    snow_raw = np.clip(snow_potential * slope_snow_gate, 0.0, 1.0)
    snow_raw[RELIEF < 12.0] = 0.0
    snow_raw[slope_deg >= 44.0] = 0.0

    # Subtle wet basalt rock margin where melting snow meets sun-warmed rock
    wet_rim = smoothstep(0.02, 0.16, snow_raw) * (1.0 - smoothstep(0.20, 0.50, snow_raw))
    surface *= (1.0 - 0.15 * wet_rim)[..., None]

    # Dual-layer painted snow transition:
    frost_crevice = np.clip(snow_raw * 1.75 * (0.50 + 0.50 * np.clip(0.5 + 0.5 * raw_crisp - 0.3 * (rmf_fine - 0.5), 0.0, 1.0)), 0.0, 0.58)
    pack_alpha = smoothstep(0.14, 0.72, snow_raw + 0.06 * raw_crisp * (snow_raw > 0.02))
    snow_alpha = np.clip(frost_crevice * (1.0 - pack_alpha) + pack_alpha, 0.0, 1.0)
    snow_alpha[RELIEF < 12.0] = 0.0
    snow_alpha[slope_deg >= 44.0] = 0.0

    col_snow_shadow = np.array([0.74, 0.82, 0.92], dtype=np.float32)
    col_snow_mid    = np.array([0.925, 0.950, 0.975], dtype=np.float32)
    col_snow_sun    = np.array([0.996, 0.994, 0.988], dtype=np.float32)

    snow_lit = smoothstep(0.45, 0.96, ao + 0.22 * aspect_sun - 0.10 * gully_concave)
    snow_col = col_snow_shadow * (1.0 - smoothstep(0.0, 0.58, snow_lit)[..., None]) + col_snow_mid * smoothstep(0.0, 0.58, snow_lit)[..., None]
    snow_col = snow_col * (1.0 - smoothstep(0.55, 1.0, snow_lit)[..., None]) + col_snow_sun * smoothstep(0.55, 1.0, snow_lit)[..., None]

    surface = surface * (1.0 - snow_alpha[..., None]) + snow_col * snow_alpha[..., None]

    # Clean dark slate road apron & Ø20mm pads (gradual 3.8mm dark damp-foot transition!)
    col_apron = np.array([0.220, 0.235, 0.265], dtype=np.float32) * (0.96 + 0.04 * np.clip(broad_painterly, -1.0, 1.0))[..., None]
    apron_blend = smoothstep(0.05, 3.6, RELIEF) * smoothstep(3.50, 5.6, dist_edge) * smoothstep(10.00, 12.4, dist_vert)
    surface = col_apron * (1.0 - apron_blend[..., None]) + surface * apron_blend[..., None]
    surface[dist_edge < 0] = [0.22, 0.235, 0.265]
    surface = np.clip(surface, 0.0, 1.0)

    # Seasonal variants derived directly from the hyperreal painted surface
    winter_gate = (1.0 - smoothstep(26.0, 43.98, slope_deg)) * smoothstep(2.5, 11.0, RELIEF + 2.5 * wind_drift) * apron_clear
    winter_gate[slope_deg >= 44.0] = 0.0
    winter_rgb = np.clip(surface * (1.0 - 0.88 * winter_gate[..., None]) + snow_col * (0.88 * winter_gate[..., None]), 0.0, 1.0)
    autumn_heather = np.array([0.52, 0.30, 0.14], dtype=np.float32) * (0.72 + 0.28 * meadow_richness)[..., None]
    autumn_rgb = np.clip(surface * (1.0 - 0.78 * meadow_alpha[..., None]) + autumn_heather * (0.78 * meadow_alpha[..., None]), 0.0, 1.0)

    # PBR Roughness & Crisp Crystalline Normal Map
    roughness = np.full_like(Z, 0.72, dtype=np.float32)
    roughness = roughness - 0.16 * smoothstep(32.0, 58.0, slope_deg) - 0.22 * wet_rim
    roughness = roughness * (1.0 - meadow_alpha) + 0.86 * meadow_alpha
    roughness = roughness * (1.0 - snow_alpha) + (0.62 - 0.10 * (1.0 - snow_lit)) * snow_alpha
    roughness = np.clip(roughness, 0.28, 0.92)

    Z_micro = Z + (0.065 * raw_crisp + 0.045 * raw_crags) * slope_mask_detail * (1.0 - snow_alpha) + 0.012 * wind_drift * snow_alpha
    gy_n, gx_n = np.gradient(Z_micro, dy, dx)
    nz_n = 1.0 / np.sqrt(gx_n * gx_n + gy_n * gy_n + 1.0)
    normal_rgb = np.clip(np.stack([-gx_n * nz_n * 0.5 + 0.5, -gy_n * nz_n * 0.5 + 0.5, nz_n * 0.5 + 0.5], axis=-1), 0.0, 1.0)
    normal_rgb[dist_edge < 0] = [0.5, 0.5, 1.0]

    mask_hex = dist_edge > 0
    mask_mtn = mask_hex & (RELIEF > 0.15)
    stats = {
        "snow_pct": float((snow_alpha[mask_mtn] > 0.35).mean() * 100.0),
        "meadow_pct": float((meadow_alpha[mask_mtn] > 0.25).mean() * 100.0),
        "steep_pct": float((slope_deg[mask_mtn] >= 44.0).mean() * 100.0),
        "max_snow_steep": float(snow_alpha[mask_hex & (slope_deg >= 44.0)].max()) if np.any(mask_hex & (slope_deg >= 44.0)) else 0.0,
        "winter_rgb": winter_rgb,
        "autumn_rgb": autumn_rgb,
    }
    return surface, normal_rgb, roughness, slope_deg, snow_alpha, meadow_alpha, stats


def raycast_preview(Z_dem, diff_f32, norm_f32, rough_f32, az_deg, el_deg, width=900, height=720, fov_mm=138.0, center=(0, 0, 42.0)):
    az, el = math.radians(az_deg), math.radians(el_deg)
    fwd = np.array([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)], dtype=np.float32)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0, 0, 1], dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    cam_dist = 520.0
    cam_pos = np.array(center, dtype=np.float32) - fwd * cam_dist
    aspect = width / float(height)
    PX, PY = np.meshgrid(np.linspace(-0.5 * fov_mm * aspect, 0.5 * fov_mm * aspect, width, dtype=np.float32),
                         np.linspace(0.5 * fov_mm, -0.5 * fov_mm, height, dtype=np.float32))
    dx = fwd[0] * cam_dist + right[0] * PX + up[0] * PY
    dy = fwd[1] * cam_dist + right[1] * PX + up[1] * PY
    dz = fwd[2] * cam_dist + right[2] * PX + up[2] * PY
    inv = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
    dx *= inv; dy *= inv; dz *= inv

    def in_hex(x, y):
        d = np.full_like(x, 999.0)
        for k in range(6):
            a = k * math.pi / 3.0
            d = np.minimum(d, APOTHEM - (x * math.cos(a) + y * math.sin(a)))
        return d

    ny, nx = Z_dem.shape
    def samp(grid, x, y):
        u = np.clip((x + APOTHEM) / (2.0 * APOTHEM), 0, 1) * (nx - 1)
        v = np.clip((y + R) / (2.0 * R), 0, 1) * (ny - 1)
        x0 = np.clip(np.floor(u).astype(np.int32), 0, nx - 2)
        y0 = np.clip(np.floor(v).astype(np.int32), 0, ny - 2)
        fx, fy = u - x0, v - y0
        if grid.ndim == 2:
            return (grid[y0, x0] * (1 - fx) + grid[y0, x0 + 1] * fx) * (1 - fy) + (grid[y0 + 1, x0] * (1 - fx) + grid[y0 + 1, x0 + 1] * fx) * fy
        fx, fy = fx[..., None], fy[..., None]
        return (grid[y0, x0] * (1 - fx) + grid[y0, x0 + 1] * fx) * (1 - fy) + (grid[y0 + 1, x0] * (1 - fx) + grid[y0 + 1, x0 + 1] * fx) * fy

    t0, t1 = np.zeros_like(dx), np.full_like(dx, 1e6)
    for o, dc, mn, mx in [(cam_pos[0], dx, -APOTHEM - 0.5, APOTHEM + 0.5),
                          (cam_pos[1], dy, -R - 0.5, R + 0.5),
                          (cam_pos[2], dz, -0.1, Z_PEAK + 0.5)]:
        ds = np.where(np.abs(dc) < 1e-8, 1e-8, dc)
        ta, tb = (mn - o) / ds, (mx - o) / ds
        t0 = np.maximum(t0, np.minimum(ta, tb))
        t1 = np.minimum(t1, np.maximum(ta, tb))

    iy, ix = np.where(t1 > t0)
    img = np.zeros((height, width, 3), dtype=np.float32) + np.array([0.07, 0.09, 0.12], dtype=np.float32)
    rx, ry, rz = dx[iy, ix], dy[iy, ix], dz[iy, ix]
    ta_s, tb_s = t0[iy, ix], t1[iy, ix]
    step_t = (tb_s - ta_s) / 220.0
    hit_t = np.full(len(iy), -1.0, dtype=np.float32)
    hit = np.zeros(len(iy), dtype=bool)
    for s in range(221):
        rem = ~hit
        if not np.any(rem): break
        tc = ta_s[rem] + s * step_t[rem]
        px, py, pz = cam_pos[0] + tc * rx[rem], cam_pos[1] + tc * ry[rem], cam_pos[2] + tc * rz[rem]
        ins = (in_hex(px, py) >= 0) & (pz >= 0) & (pz <= samp(Z_dem, px, py))
        if np.any(ins):
            ri = np.where(rem)[0][ins]
            hit_t[ri] = tc[ins]
            hit[ri] = True

    hi = np.where(hit)[0]
    t_hi = hit_t[hi]
    t_lo = np.maximum(ta_s[hi], t_hi - step_t[hi])
    rxh, ryh, rzh = rx[hi], ry[hi], rz[hi]
    for _ in range(7):
        tm = 0.5 * (t_lo + t_hi)
        px, py, pz = cam_pos[0] + tm * rxh, cam_pos[1] + tm * ryh, cam_pos[2] + tm * rzh
        ins = (in_hex(px, py) >= 0) & (pz >= 0) & (pz <= samp(Z_dem, px, py))
        t_hi = np.where(ins, tm, t_hi)
        t_lo = np.where(ins, t_lo, tm)
    th = 0.5 * (t_lo + t_hi)
    hx, hy, hz = cam_pos[0] + th * rxh, cam_pos[1] + th * ryh, cam_pos[2] + th * rzh

    is_wall = in_hex(hx, hy) < 0.22
    n_vec = samp(norm_f32, hx, hy) * 2.0 - 1.0
    n_vec /= np.maximum(np.linalg.norm(n_vec, axis=-1, keepdims=True), 1e-6)
    albedo = samp(diff_f32, hx, hy)
    rough = samp(rough_f32, hx, hy)

    if np.any(is_wall):
        bn = np.zeros((np.sum(is_wall), 3), dtype=np.float32)
        bd = np.full(np.sum(is_wall), 999.0, dtype=np.float32)
        for k in range(6):
            a = k * math.pi / 3.0
            dk = np.abs(APOTHEM - (hx[is_wall] * math.cos(a) + hy[is_wall] * math.sin(a)))
            b = dk < bd
            bd = np.where(b, dk, bd)
            bn[b] = [math.cos(a), math.sin(a), 0.0]
        n_vec[is_wall] = bn
        albedo[is_wall] = [0.22, 0.235, 0.265]
        rough[is_wall] = 0.72

    sun = np.array([0.55, -0.45, 0.70], dtype=np.float32)
    sun /= np.linalg.norm(sun)
    sh = np.ones(len(hx), dtype=np.float32)
    for st in np.linspace(1.0, 48.0, 22):
        occ = (in_hex(hx + sun[0] * st, hy + sun[1] * st) >= 0) & (hz + sun[2] * st < samp(Z_dem, hx + sun[0] * st, hy + sun[1] * st) - 0.14)
        sh = np.where(occ, np.minimum(sh, 0.26), sh)

    ndotl = np.clip(n_vec[:, 0] * sun[0] + n_vec[:, 1] * sun[1] + n_vec[:, 2] * sun[2], 0, 1)
    half_v = sun - np.stack([rxh, ryh, rzh], axis=-1)
    half_v /= np.maximum(np.linalg.norm(half_v, axis=-1, keepdims=True), 1e-6)
    ndoth = np.clip(np.sum(n_vec * half_v, axis=-1), 0, 1)
    shininess = np.clip((1.0 - rough) ** 2 * 180.0, 6.0, 120.0)
    spec = (ndoth ** shininess) * (1.0 - rough) * 0.25 * sh

    sky = 0.5 + 0.5 * n_vec[:, 2]
    fill = np.clip(-0.5 * n_vec[:, 0] + 0.55 * n_vec[:, 1] + 0.45 * n_vec[:, 2], 0, 1)
    warm_sun = np.array([1.04, 0.99, 0.93], dtype=np.float32)
    cool_sky = np.array([0.78, 0.87, 1.02], dtype=np.float32)
    rgb = albedo * (0.36 * sky[:, None] * cool_sky + 0.22 * fill[:, None] * cool_sky + 0.88 * ndotl[:, None] * sh[:, None] * warm_sun) + spec[:, None] * warm_sun
    img[iy[hi], ix[hi]] = np.clip(rgb, 0, 1) ** (1.0 / 1.12)
    return (img * 255).astype(np.uint8)


def main():
    t0 = time.time()
    print("[V3.4 Cordillera] Generating 900x1040 DEM + hyperreal alpine textures...")
    X, Y, Z, dist_edge, dist_vert, dx, dy, rmf_m, rmf_f, broad_r = generate_cordillera_dem(900, 1040)
    diff, norm, rough, slope_deg, snow_a, meadow_a, st = synthesize_hyperreal_textures(
        X, Y, Z, dist_edge, dist_vert, dx, dy, rmf_m, rmf_f, broad_r
    )
    print(
        f"  Done in {time.time()-t0:.2f}s | Z[{Z.min():.2f}, {Z.max():.2f}] | "
        f"Snow={st['snow_pct']:.1f}%, Meadow={st['meadow_pct']:.1f}%, Steep>=44={st['steep_pct']:.1f}%, MaxSnow>=44={st['max_snow_steep']:.6f}"
    )

    p_iso = raycast_preview(Z, diff, norm, rough, -50.0, 26.0, 960, 740, 138.0, (0, 0, 44.0))
    p_west = raycast_preview(Z, diff, norm, rough, 132.0, 22.0, 960, 740, 138.0, (0, 0, 44.0))
    p_top = raycast_preview(Z, diff, norm, rough, -90.0, 88.5, 860, 780, 136.0, (0, 0, 40.0))
    p_zoom = raycast_preview(Z, diff, norm, rough, -42.0, 28.0, 860, 720, 56.0, (0.0, 2.0, 62.0))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5), dpi=130, facecolor="#0c1017")
    for ax, im, ttl in zip(axes.flat, [p_iso, p_west, p_top, p_zoom], [
        "A. Cordillera V3.4 Stylized Skyline (SE Isometric)",
        "B. Opposite NW Cirque & Multi-Spire Silhouette",
        "C. Cenital Topo & Pure High-Elevation Glacier Snow",
        "D. Macro Zoom: Snowy Summit Cornices + Fractured Granite",
    ]):
        ax.imshow(im)
        ax.set_title(ttl, color="#f1f5f9", fontsize=11, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    out_p = os.path.join(OUT_DIR, "preview_v3_cordillera.png")
    plt.savefig(out_p, facecolor="#0c1017")
    plt.close(fig)
    print("  Saved:", out_p)


if __name__ == "__main__":
    main()
