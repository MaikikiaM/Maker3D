import math
import os
import time
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

print("=== GENERATING SEASONAL 4K TEXTURE SUITE (WINTER & AUTUMN) ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D"
dim = 4096
R = 64.0
APOTHEM = R * math.cos(math.radians(30))

xs = np.linspace(-APOTHEM, APOTHEM, dim, dtype=np.float32)
ys = np.linspace(-R, R, dim, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)

normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

Z_master = np.load('scratch/Z_top_master_2m.npy')
import scipy.interpolate as interp
ny_m, nx_m = Z_master.shape
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

# Resample Z
chunk_size = dim // 4
Z_chunks = []
for c_idx in range(4):
    y_c = ys[c_idx*chunk_size : (c_idx+1)*chunk_size]
    X_c, Y_c = np.meshgrid(xs, y_c)
    pts = np.column_stack((Y_c.ravel(), X_c.ravel()))
    Z_chunks.append(interp_z(pts).reshape((chunk_size, dim)))
Z = np.vstack(Z_chunks)
RELIEF = np.maximum(0.0, Z - 25.0)

dx = float(xs[1] - xs[0]); dy = float(ys[1] - ys[0])
gy, gx = np.gradient(Z, dy, dx)
slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
laplace = ndi.laplace(Z)

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# Base noise
noise_low = np.random.RandomState(42).randn(dim // 8, dim // 8).astype(np.float32)
noise_mid = np.random.RandomState(101).randn(dim // 4, dim // 4).astype(np.float32)
rock_f1 = ndi.zoom(ndi.gaussian_filter(noise_low, 3.0), 8.0, order=1)
rock_f2 = ndi.zoom(ndi.gaussian_filter(noise_mid, 2.0), 4.0, order=1)
rock_noise = 0.65 * rock_f1 + 0.35 * rock_f2

# -------------------------------------------------------------
# 1. WINTER MASSIF (GLACIAL SNOWPACK & FROST)
# -------------------------------------------------------------
print("Synthesizing Winter Glacial Massif 4K Texture...")
# Deep winter: snow descends down to Z = 26mm
winter_snow_alt = smoothstep(1.5, 18.0, RELIEF)
winter_snow_shed = 1.0 - smoothstep(28.0, 44.0, slope_deg) # STRICT: 0% snow on cliffs >= 44 deg
winter_snow_alpha = np.clip(winter_snow_alt * winter_snow_shed + 0.05 * rock_noise, 0.0, 1.0)
winter_snow_alpha = smoothstep(0.15, 0.55, winter_snow_alpha) * winter_snow_shed

winter_rock_rgb = np.zeros((dim, dim, 3), dtype=np.float32)
# Cool dark frosted slate rock (Tones 35-44 of greys)
winter_rock_rgb[:, :, 0] = 0.22 + 0.04 * rock_noise
winter_rock_rgb[:, :, 1] = 0.24 + 0.04 * rock_noise
winter_rock_rgb[:, :, 2] = 0.27 + 0.04 * rock_noise

winter_snow_rgb = np.array([0.98, 0.99, 1.00], dtype=np.float32)
winter_ice_rgb = np.array([0.80, 0.84, 0.88], dtype=np.float32) # Cold steel firn grey

# Ice in couloirs
winter_ice_alpha = smoothstep(-1.0, -0.1, laplace) * smoothstep(5.0, 25.0, RELIEF) * 0.75

winter_rgb = winter_rock_rgb.copy()
for c in range(3):
    winter_rgb[:, :, c] = (1.0 - winter_ice_alpha) * winter_rgb[:, :, c] + winter_ice_alpha * winter_ice_rgb[c]
for c in range(3):
    winter_rgb[:, :, c] = (1.0 - winter_snow_alpha) * winter_rgb[:, :, c] + winter_snow_alpha * winter_snow_rgb[c]

# Foundation
road_margin = smoothstep(1.5, 3.5, dist_edge)
for c in range(3):
    winter_rgb[:, :, c] = road_margin * winter_rgb[:, :, c] + (1.0 - road_margin) * 0.12
winter_rgb[dist_edge < 0] = [0.08, 0.09, 0.11]
winter_rgb = np.clip(winter_rgb, 0.0, 1.0)

im_winter = Image.fromarray((winter_rgb[::-1] * 255.0).astype(np.uint8))
winter_png = os.path.join(out_dir, "catan_mountain_diffuse_winter_4k.png")
im_winter.save(winter_png, optimize=True)
print(f"Saved Winter 4K: {winter_png} ({os.path.getsize(winter_png)/1024/1024:.2f} MB)")

# -------------------------------------------------------------
# 2. AUTUMN TUNDRA (GOLDEN LARCH & OCHRE GRANITE)
# -------------------------------------------------------------
print("Synthesizing Autumn Golden Tundra 4K Texture...")
# Warm golden granite
autumn_rock_rgb = np.zeros((dim, dim, 3), dtype=np.float32)
autumn_rock_rgb[:, :, 0] = 0.62 + 0.08 * rock_noise # Warm ochre tone
autumn_rock_rgb[:, :, 1] = 0.58 + 0.07 * rock_noise
autumn_rock_rgb[:, :, 2] = 0.50 + 0.06 * rock_noise

# First autumn dusting of snow only on highest summits (Z > 30mm relief)
autumn_snow_alpha = smoothstep(28.0, 48.0, RELIEF) * (1.0 - smoothstep(40.0, 65.0, slope_deg))

# Golden-orange alpine larch and tundra
autumn_tundra_alt = smoothstep(0.1, 1.5, RELIEF) * (1.0 - smoothstep(5.0, 14.0, RELIEF))
autumn_tundra_slope = 1.0 - smoothstep(12.0, 26.0, slope_deg)
autumn_tundra_alpha = np.clip(autumn_tundra_alt * autumn_tundra_slope * (0.55 + 0.45 * rock_f1), 0.0, 1.0)
autumn_tundra_alpha = smoothstep(0.15, 0.60, autumn_tundra_alpha)

# Golden yellow and amber tundra color
autumn_tundra_color = np.array([0.76, 0.58, 0.18], dtype=np.float32)

autumn_rgb = autumn_rock_rgb.copy()
for c in range(3):
    autumn_rgb[:, :, c] = (1.0 - autumn_tundra_alpha) * autumn_rgb[:, :, c] + autumn_tundra_alpha * autumn_tundra_color[c]
for c in range(3):
    autumn_rgb[:, :, c] = (1.0 - autumn_snow_alpha) * autumn_rgb[:, :, c] + autumn_snow_alpha * 0.98

for c in range(3):
    autumn_rgb[:, :, c] = road_margin * autumn_rgb[:, :, c] + (1.0 - road_margin) * 0.12
autumn_rgb[dist_edge < 0] = [0.08, 0.09, 0.11]
autumn_rgb = np.clip(autumn_rgb, 0.0, 1.0)

im_autumn = Image.fromarray((autumn_rgb[::-1] * 255.0).astype(np.uint8))
autumn_png = os.path.join(out_dir, "catan_mountain_diffuse_autumn_4k.png")
im_autumn.save(autumn_png, optimize=True)
print(f"Saved Autumn 4K: {autumn_png} ({os.path.getsize(autumn_png)/1024/1024:.2f} MB)")

print(f"Seasonal suite generated in {time.time()-t0:.2f}s")
