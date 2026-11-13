import os
import math
import time
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== PROTOTYPE V2.10: CRISP PYRAMIDAL MATTERHORN + 22% SNOW ARÊTES + VALLEY TUNDRA ===")
t0 = time.time()

R = 64.0
APOTHEM = R * math.cos(math.radians(30.0))  # 55.4256258 mm
Z_BASE = 25.0
Z_PEAK = 83.0
TARGET_RELIEF = Z_PEAK - Z_BASE  # 58.00 mm

os.makedirs("Tableros-Catan/pipeline-montana/scratch_v2", exist_ok=True)

# 1. Palettes: 45 Greys + 15 Greens
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
greens_15 = np.array([[int(h[1:3], 16)/255.0, int(h[3:5], 16)/255.0, int(h[5:7], 16)/255.0] for h in greens_15_hex], dtype=np.float32)

np.save("Tableros-Catan/pipeline-montana/scratch_v2/greys_45.npy", greys_45)
np.save("Tableros-Catan/pipeline-montana/scratch_v2/greens_15.npy", greens_15)

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def smax(a, b, k=0.35):
    m = np.maximum(a, b)
    return m + np.log(np.exp(k * (a - m)) + np.exp(k * (b - m))) / k

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
    std = np.std(out)
    return (out - np.mean(out)) / (std if std > 1e-8 else 1.0)

def ridged_multifractal(nx, ny, width_mm, height_mm, seed=100):
    octaves = [
        (18.0, 52.0, 1.00, 1.45, seed + 1),
        (8.5,  20.0, 0.45, 1.60, seed + 2),
        (4.0,   9.5, 0.20, 1.70, seed + 3),
        (1.8,   4.2, 0.08, 1.75, seed + 4),
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

def generate_dem(nx, ny):
    xs = np.linspace(-APOTHEM, APOTHEM, nx, dtype=np.float32)
    ys = np.linspace(-R, R, ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    W_mm = 2.0 * APOTHEM
    H_mm = 2.0 * R

    normals_hex = [[math.cos(k * math.pi / 3.0), math.sin(k * math.pi / 3.0)] for k in range(6)]
    dist_edge = np.full_like(X, 999.0)
    for nx_h, ny_h in normals_hex:
        dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

    v_angles = [(30.0 + 60.0 * k) * math.pi / 180.0 for k in range(6)]
    hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles])
    dist_vert = np.full_like(X, 999.0)
    for vx, vy in hex_corners:
        dist_vert = np.minimum(dist_vert, np.hypot(X - vx, Y - vy))

    # Organic domain warp (gentle near summit so snow-cap facets keep natural <41 deg slopes)
    wx = 2.6 * fft_fbm(nx, ny, 24.0, 85.0, W_mm, H_mm, beta=2.4, seed=11) + \
         0.7 * fft_fbm(nx, ny, 9.0, 24.0, W_mm, H_mm, beta=2.1, seed=12)
    wy = 2.6 * fft_fbm(nx, ny, 24.0, 85.0, W_mm, H_mm, beta=2.4, seed=21) + \
         0.7 * fft_fbm(nx, ny, 9.0, 24.0, W_mm, H_mm, beta=2.1, seed=22)

    sx, sy = -1.5, 2.5
    r_summit = np.hypot(X - sx, Y - sy)
    warp_atten = smoothstep(5.0, 22.0, r_summit)
    Xw = X + wx * warp_atten
    Yw = Y + wy * warp_atten

    # Asymmetric alpine ridge: gently crowned snow-bearing crest (<40 deg)
    # plunging steeply (50-66 deg) into flanking bare-rock cliff walls.
    def ridge_seg(x1, y1, x2, y2, h1, h2, w_left, w_right, crest_flat=3.0):
        dxr, dyr = x2 - x1, y2 - y1
        L = math.hypot(dxr, dyr)
        L2 = L * L
        t = np.clip(((Xw - x1) * dxr + (Yw - y1) * dyr) / L2, 0.0, 1.0)
        px = x1 + t * dxr
        py = y1 + t * dyr
        nx_p, ny_p = -dyr / L, dxr / L
        s_dist = (Xw - px) * nx_p + (Yw - py) * ny_p
        along_d = np.hypot(Xw - px, Yw - py)
        # C-infinity smooth asymmetric width (eliminates vertical fault-line discontinuities beyond endpoints!)
        w_side = 0.5 * (w_left + w_right) + 0.5 * (w_left - w_right) * np.tanh(s_dist / 3.2)
        d_norm = along_d / w_side
        cf = crest_flat / w_side
        d_eff = np.maximum(0.0, np.sqrt(d_norm * d_norm + cf * cf) - cf)
        h_lin = (1.0 - t) * h1 + t * h2 - 1.2 * np.sin(math.pi * t)
        return h_lin / (1.0 + d_eff ** 1.85)

    # Main Concave Pyramidal Matterhorn Peak (crisp 1.2mm alpine tip, exponent 1.28 pyramidal facets)
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
    # Crisp Matterhorn pyramidal summit (30-39 deg snowy crest facets connecting solidly through apex!)
    h_pyramid = np.maximum(0.0, 58.0 - 0.42 * r_eff_tip)
    # Compact granite cliff buttresses curving cleanly into broad green foothills -> Snow ~15-18%, Tundra ~11-13%!
    h_skirt = 86.0 * np.exp(-((r_asym / 28.2) ** 1.54))
    k_int = 0.16
    h_horn = -np.log(np.exp(-k_int * h_pyramid) + np.exp(-k_int * h_skirt)) / k_int
    h_horn = np.maximum(0.0, h_horn)

    # 4 Sweeping Continuous Arêtes descending smoothly from summit down into granite spurs
    r1 = ridge_seg(sx, sy, 21.0, 13.5, 57.5, 38.5, 8.2, 14.5, 3.0)
    r2 = ridge_seg(21.0, 13.5, 34.0, 18.0, 38.5, 5.5, 6.0, 9.2, 1.8)
    r3 = ridge_seg(21.0, 13.5, 30.0, -7.0, 38.5, 6.0, 9.2, 6.0, 1.8)
    r4 = ridge_seg(sx, sy, -19.5, -16.5, 57.5, 37.5, 14.5, 8.2, 3.0)
    r5 = ridge_seg(-19.5, -16.5, -31.5, -24.0, 37.5, 5.0, 9.0, 6.0, 1.8)
    r6 = ridge_seg(-19.5, -16.5, 5.5, -30.5, 37.5, 5.5, 6.0, 9.0, 1.8)
    r7 = ridge_seg(sx, sy, -26.0, 11.5, 56.2, 26.0, 12.5, 6.5, 2.6)
    r8 = ridge_seg(sx, sy, 4.0, 29.0, 55.5, 20.0, 6.2, 11.5, 2.4)
    r9 = ridge_seg(sx, sy, 15.5, -14.5, 56.2, 29.0, 6.8, 12.5, 2.6)

    # Single parallel max + mild curvature rounding eliminates the 5-ridge log-sum-exp summit wart!
    r_all = np.maximum.reduce([r1, r2, r3, r4, r5, r6, r7, r8, r9])
    r_all_s = ndi.gaussian_filter(r_all, 0.55 / dx)
    massif = smax(h_horn, r_all_s, k=0.55)

    # Excavate deep sheer bare-rock North/East/South glacial cirque cliffs (>50 deg dark granite faces!)
    cirques = [
        (-11.5, 16.5, 10.2, 13.0),  # Dramatic steep NW Matterhorn North Face!
        (14.0, -3.5, 9.5, 11.0),   # Steep East Cliff Wall
        (-3.0, -19.0, 10.0, 9.5),  # Steep South Headwall
        (-20.5, -3.5, 9.5, 9.0),   # Steep West Cirque
    ]
    for cx, cy, cr, depth in cirques:
        d_c = np.hypot(Xw - cx, Yw - cy)
        bowl = np.exp(-((d_c / cr) ** 2.0))
        massif = np.maximum(0.0, massif - depth * bowl * smoothstep(5.0, 20.0, massif))

    # Smooth wide radial taper so mountain spurs fade naturally well inside perimeter -> ZERO hex cliff wall!
    r_base = np.hypot(Xw * 0.96, Yw)
    radial_fade = smoothstep(48.5, 21.0, r_base)
    massif = massif * radial_fade

    # Gentle green alluvial foothill meadows (6-20 deg) filling broad valley bays around ALL 6 edges
    d_clear = np.minimum(np.maximum(0.0, dist_edge - 3.50), np.maximum(0.0, dist_vert - 10.00))
    foothill_noise = fft_fbm(nx, ny, 9.5, 28.0, W_mm, H_mm, beta=2.2, seed=55)
    foothills = (2.8 + 1.9 * smoothstep(-1.0, 1.0, foothill_noise)) * smoothstep(0.0, 6.2, d_clear) * (1.0 - smoothstep(4.5, 16.0, massif))
    massif = massif * smoothstep(0.0, 7.5, d_clear) + foothills

    # Ridged Multifractal rock ribs + strata (gentler on upper snowy crests so snow stays connected!)
    rmf = ridged_multifractal(nx, ny, W_mm, H_mm, seed=303)
    rmf_c = (rmf - 0.45) * 2.0
    elev_w = smoothstep(4.5, 16.0, massif) * (1.0 - 0.85 * smoothstep(28.0, 44.0, massif))

    strata_warp = fft_fbm(nx, ny, 12.0, 42.0, W_mm, H_mm, beta=2.2, seed=404)
    strata_coord = 0.20 * massif + 0.06 * Xw - 0.04 * Yw + 1.5 * strata_warp
    strata_bands = 0.30 * np.sin(strata_coord * 1.25) + 0.14 * np.sin(strata_coord * 2.6 + 0.6)

    Z_sculpted = massif + (0.90 * rmf_c * elev_w) + (0.24 * strata_bands * elev_w)
    Z_sculpted = np.maximum(0.0, Z_sculpted)

    # Mild thermal talus relaxation
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
        Z_eroded[:, 1:]  += c * 0.25 * ex_r
        Z_eroded[:-1, :] -= c * 0.25 * ex_d
        Z_eroded[1:, :]  += c * 0.25 * ex_d

    relief = np.maximum(0.0, Z_eroded) * smoothstep(0.0, 0.8, d_clear)

    max_r = np.max(relief)
    if max_r > 0:
        relief *= (TARGET_RELIEF / max_r)

    Z_final = Z_BASE + relief
    Z_final[dist_edge <= 3.50] = Z_BASE
    Z_final[dist_vert <= 10.00] = Z_BASE
    return X, Y, Z_final, dist_edge, dist_vert, dx, dy

nx_p, ny_p = 1024, 1182
X, Y, Z, dist_edge, dist_vert, dx, dy = generate_dem(nx_p, ny_p)
print(f"DEM generated in {time.time()-t0:.2f}s. Z range: [{Z.min():.3f}, {Z.max():.3f}] mm")

Z_slope_s = ndi.gaussian_filter(Z, 0.28 / dx)
gy, gx = np.gradient(Z_slope_s, dy, dx)
slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx, Ny, Nz = -gx / N_norm, -gy / N_norm, 1.0 / N_norm

Z_mac = ndi.gaussian_filter(Z, 1.8 / dx)
gym, gxm = np.gradient(Z_mac, dy, dx)
gyym, gyxm = np.gradient(gym, dy, dx)
gxym, gxxm = np.gradient(gxm, dy, dx)
tr = gxxm + gyym
det = gxxm * gyym - 0.5 * (gxym + gyxm)**2
disc = np.maximum(0.0, tr**2 - 4.0 * det)
e_min = 0.5 * (tr - np.sqrt(disc))
e_max = 0.5 * (tr + np.sqrt(disc))

ao = np.ones_like(Z)
for ang_deg in range(0, 360, 45):
    rad = math.radians(ang_deg)
    ux, uy = math.cos(rad), math.sin(rad)
    max_h_ang = np.zeros_like(Z)
    for step_mm in [0.6, 1.5, 3.2, 6.0, 10.0]:
        sx_px = int(round(ux * step_mm / dx))
        sy_px = int(round(uy * step_mm / dy))
        Z_sh = np.roll(np.roll(Z, sy_px, axis=0), sx_px, axis=1)
        dh = (Z_sh - Z) / step_mm
        max_h_ang = np.maximum(max_h_ang, np.clip(dh, 0.0, 2.5))
    ao -= (np.arctan(max_h_ang) / (math.pi * 0.5)) / 8.0
ao = np.clip(ao * 1.15, 0.22, 1.0)

RELIEF = np.maximum(0.0, Z - Z_BASE)
rel_h = RELIEF / TARGET_RELIEF

couloir_concavity = smoothstep(0.008, 0.065, e_max)
arete_convexity = smoothstep(-0.010, -0.075, e_min)

# STRICT SNOW RETENTION PHYSICS: 0.000 at slope >= 44.0 deg!
snow_retention = 1.0 - smoothstep(39.8, 44.0, slope_deg)
snow_retention[slope_deg >= 44.0] = 0.0

W_mm, H_mm = 2.0 * APOTHEM, 2.0 * R
snow_noise = fft_fbm(nx_p, ny_p, 2.2, 12.0, W_mm, H_mm, beta=2.1, seed=707)

# Calibrated snowline (~16-20% high-altitude peak & connected arête snow, ZERO low puddle bowls!)
snow_alt = smoothstep(13.5, 22.0, RELIEF + 5.5 * couloir_concavity + 4.5 * arete_convexity + 2.5 * snow_noise)
snow_mask = snow_alt * snow_retention
snow_mask = smoothstep(0.05, 0.24, snow_mask)
snow_mask *= smoothstep(14.5, 17.8, RELIEF)
snow_mask[slope_deg >= 44.0] = 0.0

# Botanical Tundra (15 Greens) on gentle alluvial foothills all around base (~11-13%, Z <= 33.0 mm)
tundra_noise = fft_fbm(nx_p, ny_p, 3.0, 18.0, W_mm, H_mm, beta=2.0, seed=808)
tundra_alt = smoothstep(0.06, 0.35, RELIEF) * (1.0 - smoothstep(4.2, 7.8, RELIEF + 1.4 * tundra_noise))
tundra_slope = 1.0 - smoothstep(21.0, 34.0, slope_deg)
apron_clear = smoothstep(3.55, 4.4, dist_edge) * smoothstep(10.05, 11.2, dist_vert)
tundra_raw = tundra_alt * tundra_slope * apron_clear * (0.78 + 0.22 * smoothstep(-0.8, 0.8, tundra_noise))
tundra_mask = smoothstep(0.08, 0.32, tundra_raw) * (1.0 - snow_mask)
tundra_mask[RELIEF > 9.0] = 0.0

# Snow using greys_45 indices 0..5
snow_tone_idx = np.clip(0.1 + 2.6 * (1.0 - smoothstep(0.35, 0.90, ao)) + 2.0 * smoothstep(18.0, 42.0, slope_deg), 0.0, 4.999)
s_lo = np.floor(snow_tone_idx).astype(np.int32)
s_hi = np.minimum(s_lo + 1, 5)
s_fr = (snow_tone_idx - s_lo)[:, :, None]
snow_col = (1.0 - s_fr) * greys_45[s_lo] + s_fr * greys_45[s_hi]

# Rock using greys_45 indices 6..44 (leaves 0..5 for snow/firn, hits every bright quartzite & dark basalt tone)
rock_grain = fft_fbm(nx_p, ny_p, 0.7, 3.5, W_mm, H_mm, beta=1.8, seed=908)
rock_detail = fft_fbm(nx_p, ny_p, 2.5, 12.0, W_mm, H_mm, beta=2.0, seed=909)
rock_tone_idx = (
    12.5
    + 17.0 * smoothstep(18.0, 56.0, slope_deg)
    + 11.5 * (1.0 - smoothstep(0.26, 0.90, ao))
    + 3.5 * (1.0 - rel_h)
    - 4.5 * arete_convexity
    + 2.4 * rock_detail
    + 1.8 * rock_grain
)
rock_tone_idx = np.clip(rock_tone_idx, 5.5, 43.999)
r_lo = np.floor(rock_tone_idx).astype(np.int32)
r_hi = np.minimum(r_lo + 1, 44)
r_fr = (rock_tone_idx - r_lo)[:, :, None]
rock_col = (1.0 - r_fr) * greys_45[r_lo] + r_fr * greys_45[r_hi]

# Tundra using ALL 15 greens (0..14)
g_tone_idx = (
    0.2
    + 6.8 * (1.0 - smoothstep(0.2, 7.2, RELIEF))
    + 5.8 * (1.0 - smoothstep(0.32, 0.90, ao))
    + 3.5 * smoothstep(-1.0, 1.0, tundra_noise)
)
g_tone_idx = np.clip(g_tone_idx, 0.0, 13.999)
g_lo = np.floor(g_tone_idx).astype(np.int32)
g_hi = np.minimum(g_lo + 1, 14)
g_fr = (g_tone_idx - g_lo)[:, :, None]
tundra_col = (1.0 - g_fr) * greens_15[g_lo] + g_fr * greens_15[g_hi]

diffuse = rock_col * (1.0 - tundra_mask[:, :, None]) + tundra_col * tundra_mask[:, :, None]
diffuse = diffuse * (1.0 - snow_mask[:, :, None]) + snow_col * snow_mask[:, :, None]

apron_stone = greys_45[34] * (0.94 + 0.06 * np.clip(rock_detail, -1.0, 1.0)[:, :, None])
apron_blend = smoothstep(0.02, 0.25, RELIEF) * apron_clear
diffuse = apron_stone * (1.0 - apron_blend[:, :, None]) + diffuse * apron_blend[:, :, None]
diffuse[dist_edge < 0] = greys_45[39]
diffuse = np.clip(diffuse, 0.0, 1.0)

mask_hex = dist_edge > 0
mask_mtn = mask_hex & (RELIEF > 0.15)
snow_pct = (snow_mask[mask_mtn] > 0.4).mean() * 100.0
tundra_pct = (tundra_mask[mask_mtn] > 0.4).mean() * 100.0
steep_frac = (slope_deg[mask_mtn] >= 44.0).mean() * 100.0
steep_snow_max = snow_mask[mask_hex & (slope_deg >= 44.0)].max() if np.any(mask_hex & (slope_deg >= 44.0)) else 0.0
print(f"Coverage (mountain relief): Snow={snow_pct:.1f}%, Tundra={tundra_pct:.1f}%, Rock={100-snow_pct-tundra_pct:.1f}%")
print(f"Steep faces >=44 deg: {steep_frac:.1f}% | Max snow on >=44 deg: {steep_snow_max:.6f}")

L1 = np.array([0.55, -0.45, 0.70], dtype=np.float32); L1 /= np.linalg.norm(L1)
L2 = np.array([-0.50, 0.45, 0.72], dtype=np.float32); L2 /= np.linalg.norm(L2)
d1 = np.clip(Nx*L1[0] + Ny*L1[1] + Nz*L1[2], 0.0, 1.0)
d2 = np.clip(Nx*L2[0] + Ny*L2[1] + Nz*L2[2], 0.0, 1.0)
shade = np.clip(0.58 + 0.28 * d1 + 0.14 * d2, 0.0, 1.0) * (0.68 + 0.32 * ao)
lit = np.clip(diffuse * shade[:, :, None], 0.0, 1.0)
lit[dist_edge < 0] = [0.04, 0.06, 0.09]

fig = plt.figure(figsize=(16, 14), facecolor='#090d16')

ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(lit, origin='lower')
ax1.set_title(f"1. Vista Cenital (Nieve {snow_pct:.1f}% | Tundra {tundra_pct:.1f}%)", color='#38bdf8', fontsize=12, fontweight='bold')
ax1.axis('off')

ax2 = fig.add_subplot(2, 2, 2, projection='3d', facecolor='#090d16')
st = 3
sx_g, sy_g, sz_g = X[::st, ::st], Y[::st, ::st], Z[::st, ::st]
rgba = np.zeros((sx_g.shape[0], sx_g.shape[1], 4))
rgba[:, :, :3] = lit[::st, ::st]
rgba[:, :, 3] = np.where(dist_edge[::st, ::st] >= -0.2, 1.0, 0.0)
ax2.plot_surface(sx_g, sy_g, sz_g, facecolors=rgba, rstride=1, cstride=1, shade=False, antialiased=True)
ax2.view_init(elev=28, azim=-50)
ax2.set_box_aspect([1.0, 1.15, 0.70])
ax2.set_title("2. Isometrica SE (Elev 28)", color='#38bdf8', fontsize=12, fontweight='bold')
ax2.axis('off')

ax3 = fig.add_subplot(2, 2, 3, projection='3d', facecolor='#090d16')
ax3.plot_surface(sx_g, sy_g, sz_g, facecolors=rgba, rstride=1, cstride=1, shade=False, antialiased=True)
ax3.view_init(elev=22, azim=125)
ax3.set_box_aspect([1.0, 1.15, 0.70])
ax3.set_title("3. Cara Norte / Escarpada (Elev 22)", color='#38bdf8', fontsize=12, fontweight='bold')
ax3.axis('off')

ax4 = fig.add_subplot(2, 2, 4)
im4 = ax4.imshow(np.where(mask_hex, slope_deg, np.nan), origin='lower', cmap='magma', vmin=0, vmax=72)
plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Slope (deg)')
ax4.set_title("4. Mapa de Pendientes (Base Suave + Paredes >=44°)", color='#38bdf8', fontsize=12, fontweight='bold')
ax4.axis('off')

plt.tight_layout()
preview_path = "Tableros-Catan/pipeline-montana/scratch_v2/preview_v2_4panel.png"
plt.savefig(preview_path, dpi=180, bbox_inches='tight', facecolor='#090d16')
plt.close()
print(f"Saved preview: {preview_path} in {time.time()-t0:.2f}s")
