import math
import time
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

print("=== GENERATING FLAWLESS CHISELED ALPINE GEOMORPHOLOGY V2 ===")
t0 = time.time()

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
Z_BASE = 25.0
Z_PEAK = 83.0
TARGET_RELIEF = Z_PEAK - Z_BASE # 58.00 mm

nx, ny = 1150, 1328
xs = np.linspace(-APOTHEM, APOTHEM, nx, dtype=np.float32)
ys = np.linspace(-R, R, ny, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)
dx = float(xs[1] - xs[0])
dy = float(ys[1] - ys[0])

# Hexagon distance field
normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

v_angles = [(30 + 60 * k) * math.pi / 180 for k in range(6)]
hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles])
dist_vert = np.full_like(X, 999.0)
for vx, vy in hex_corners:
    dist_vert = np.minimum(dist_vert, np.hypot(X - vx, Y - vy))

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# --- 1. PRIMARY ALPINE PEAKS (MATTERHORN HORN + SATELLITE PEAKS) ---
p_main_x, p_main_y = -3.5, 4.5
r_main = np.hypot(X - p_main_x, Y - p_main_y)
h_main = 58.0 * np.exp(-(r_main / 24.0)**1.32)

r_p2 = np.hypot(X - 16.0, Y - 15.0)
h_p2 = 47.0 * np.exp(-(r_p2 / 18.0)**1.35)

r_p3 = np.hypot(X - (-14.0), Y - (-16.0))
h_p3 = 41.5 * np.exp(-(r_p3 / 18.5)**1.35)

r_p4 = np.hypot(X - (-25.0), Y - (-5.0))
h_p4 = 28.5 * np.exp(-(r_p4 / 15.0)**1.35)

# --- 2. RADIATING KNIFE-EDGE ARÊTES ---
ridges = [
    {"x1": p_main_x, "y1": p_main_y, "x2": 16.0, "y2": 15.0, "h1": 58.0, "h2": 47.0, "w": 6.5},
    {"x1": p_main_x, "y1": p_main_y, "x2": -14.0, "y2": -16.0, "h1": 58.0, "h2": 41.5, "w": 6.8},
    {"x1": p_main_x, "y1": p_main_y, "x2": -25.0, "y2": -5.0, "h1": 58.0, "h2": 28.5, "w": 6.5},
    {"x1": p_main_x, "y1": p_main_y, "x2": -32.0, "y2": 32.0, "h1": 58.0, "h2": 12.0, "w": 7.0},
    {"x1": 16.0, "y1": 15.0, "x2": 36.0, "y2": 24.0, "h1": 47.0, "h2": 10.0, "w": 6.8},
    {"x1": 16.0, "y1": 15.0, "x2": 30.0, "y2": -12.0, "h1": 47.0, "h2": 14.0, "w": 7.2},
    {"x1": -14.0, "y1": -16.0, "x2": -32.0, "y2": -26.0, "h1": 41.5, "h2": 10.0, "w": 7.0},
    {"x1": -14.0, "y1": -16.0, "x2": 14.0, "y2": -26.0, "h1": 41.5, "h2": 12.0, "w": 6.8},
]

H_ridges = np.zeros_like(X)
for rd in ridges:
    dx_r = rd["x2"] - rd["x1"]
    dy_r = rd["y2"] - rd["y1"]
    L2 = dx_r**2 + dy_r**2
    t_proj = np.clip(((X - rd["x1"])*dx_r + (Y - rd["y1"])*dy_r) / L2, 0.0, 1.0)
    px = rd["x1"] + t_proj * dx_r
    py = rd["y1"] + t_proj * dy_r
    dist_rd = np.hypot(X - px, Y - py)
    h_spine = (1.0 - t_proj) * rd["h1"] + t_proj * rd["h2"]
    h_contrib = h_spine * np.exp(-(dist_rd / rd["w"])**1.28)
    H_ridges = np.maximum(H_ridges, h_contrib)

massif = np.maximum(H_ridges, np.maximum(h_main, np.maximum(h_p2, np.maximum(h_p3, h_p4))))

# Glacial Cirques (concave excavation)
cirque_NW = np.exp(-((X - (-18.0))**2 + (Y - 18.0)**2) / (18.0**2)) * 11.0
cirque_E  = np.exp(-((X - 8.0)**2 + (Y - (-2.0))**2) / (16.0**2)) * 9.5
cirque_SW = np.exp(-((X - (-22.0))**2 + (Y - (-12.0))**2) / (15.0**2)) * 7.5
massif = np.maximum(0.0, massif - (cirque_NW + cirque_E + cirque_SW) * (massif / 58.0))

# --- 3. SCALE-INVARIANT SPECTRAL FRACTAL FRACTURING (NO BLISTERS, NO RIPPLES) ---
print("Generating scale-invariant spectral fractal bedrock via FFT...")
def generate_fBm_fft(width, height, beta=2.2, seed=123):
    rng = np.random.RandomState(seed)
    kx = np.fft.rfftfreq(width)
    ky = np.fft.fftfreq(height)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1.0
    amplitude = 1.0 / (K ** (beta / 2.0))
    amplitude[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, size=amplitude.shape)
    spectrum = amplitude * np.exp(1j * phase)
    noise = np.fft.irfft2(spectrum, s=(height, width)).astype(np.float32)
    return (noise - np.mean(noise)) / np.std(noise)

fBm_macro = generate_fBm_fft(nx, ny, beta=2.35, seed=101) # Large structural crags
fBm_micro = generate_fBm_fft(nx, ny, beta=1.95, seed=202) # Fine rock texture

# Chiseled Faceting: We sharpen the fractal by slope-clipping (terrace-shear)
# Instead of adding round bumps, we use directional derivative sharpening
gy_m, gx_m = np.gradient(fBm_macro)
sharpened_crags = fBm_macro - 0.45 * np.hypot(gx_m, gy_m) # Sharpen ridges, hollow cirques

# Combined rock sculpture (amplitude is carefully tuned so it creates crisp rock faces without cellulite!)
rock_crags = 1.6 * sharpened_crags + 0.45 * fBm_micro

# Modulate rock sculpture by elevation and cliff steepness
elev_weight = np.clip(massif / 58.0, 0.0, 1.0)**0.85
Z_sculpted = massif + rock_crags * elev_weight

# --- 4. PHYSICAL THERMAL EROSION (TALUS SCREE RELAXATION) ---
print("Simulating physical thermal erosion (talus scree cones at 37 deg)...")
talus_crit_deg = 36.5
T_x = np.tan(np.radians(talus_crit_deg)) * dx
T_y = np.tan(np.radians(talus_crit_deg)) * dy

Z_eroded = Z_sculpted.copy()
for it in range(16):
    diff_r = Z_eroded[:, :-1] - Z_eroded[:, 1:]
    diff_l = Z_eroded[:, 1:] - Z_eroded[:, :-1]
    diff_d = Z_eroded[:-1, :] - Z_eroded[1:, :]
    diff_u = Z_eroded[1:, :] - Z_eroded[:-1, :]
    
    excess_r = np.maximum(0.0, diff_r - T_x)
    excess_l = np.maximum(0.0, diff_l - T_x)
    excess_d = np.maximum(0.0, diff_d - T_y)
    excess_u = np.maximum(0.0, diff_u - T_y)
    
    c = 0.22
    Z_eroded[:, :-1] -= c * 0.25 * excess_r
    Z_eroded[:, 1:]  += c * 0.25 * excess_r
    Z_eroded[:, 1:]  -= c * 0.25 * excess_l
    Z_eroded[:, :-1] += c * 0.25 * excess_l
    Z_eroded[:-1, :] -= c * 0.25 * excess_d
    Z_eroded[1:, :]  += c * 0.25 * excess_d
    Z_eroded[1:, :]  -= c * 0.25 * excess_u
    Z_eroded[:-1, :] += c * 0.25 * excess_u

# --- 5. PERIMETER ROAD CLEARANCE & MONOLITHIC NORM ---
print("Applying road clearance and base integration...")
edge_clear = smoothstep(1.8, 8.5, dist_edge)
vert_clear = smoothstep(9.5, 14.5, dist_vert)
total_clearance = edge_clear * vert_clear

relief_cleared = Z_eroded * total_clearance
max_r = np.max(relief_cleared)
if max_r > 0:
    relief_cleared = relief_cleared * (TARGET_RELIEF / max_r)

Z_final = Z_BASE + relief_cleared
Z_final[dist_edge < 0] = Z_BASE

# Ensure apex reaches exactly 83.00 mm
ix_p = int(np.round((p_main_x - (-APOTHEM)) / (2 * APOTHEM) * (nx - 1)))
iy_p = int(np.round((p_main_y - (-R)) / (2 * R) * (ny - 1)))
Z_final[iy_p, ix_p] = Z_PEAK

print(f"Topography generated! Range: [{np.min(Z_final):.2f}, {np.max(Z_final):.2f}] mm in {time.time()-t0:.2f}s")

np.save('scratch/Z_top_master_2m.npy', Z_final)
print("Saved scratch/Z_top_master_2m.npy")

# Generate hillshade
gy, gx = np.gradient(Z_final, dy, dx)
slope = np.hypot(gx, gy)
aspect = np.arctan2(gy, -gx)
az = 315.0 * np.pi / 180.0
alt = 45.0 * np.pi / 180.0
shaded = np.sin(alt) + np.cos(alt) * slope * np.cos(az - aspect)
shaded = (shaded - shaded.min()) / (shaded.max() - shaded.min())
im = Image.fromarray((shaded * 255.0).astype(np.uint8))
im.save('scratch/flawless_dem_hillshade.png')
print("Saved scratch/flawless_dem_hillshade.png")
