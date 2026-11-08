import math
import os
import time
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

print("=== GENERATING 4096x4096 ULTRA-HD DIFFUSE TEXTURE MAP ===")
t0 = time.time()

# Texture dimensions
dim = 4096
R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm

xs = np.linspace(-APOTHEM, APOTHEM, dim, dtype=np.float32)
ys = np.linspace(-R, R, dim, dtype=np.float32)

# Load master Z topography
Z_master = np.load('scratch/Z_top_master_final.npy')
ny_m, nx_m = Z_master.shape
import scipy.interpolate as interp
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

print(f"Interpolating Z topography on {dim}x{dim} grid...")
# Compute in 4 vertical chunks to be memory efficient and fast
chunk_size = dim // 4
Z_chunks = []
for c_idx in range(4):
    y_chunk = ys[c_idx*chunk_size : (c_idx+1)*chunk_size]
    X_c, Y_c = np.meshgrid(xs, y_chunk)
    pts = np.column_stack((Y_c.ravel(), X_c.ravel()))
    z_c = interp_z(pts).reshape((chunk_size, dim))
    Z_chunks.append(z_c)
Z = np.vstack(Z_chunks)
print(f"Z grid interpolated in {time.time()-t0:.2f}s. Z in [{np.min(Z):.2f}, {np.max(Z):.2f}] mm")

# Meshgrid for coordinates
X, Y = np.meshgrid(xs, ys)

# Distance to hexagon edges
normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF = np.maximum(0.0, Z - Z_BASE)

# Gradient and derivatives for lighting and slope
dx = float(xs[1] - xs[0]); dy = float(ys[1] - ys[0])
gy, gx = np.gradient(Z, dy, dx)
slope = np.hypot(gx, gy)
slope_deg = np.degrees(np.arctan(slope))
laplace = ndi.laplace(Z)

print(f"Gradients computed in {time.time()-t0:.2f}s.")

# Smoothstep helper
def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# -------------------------------------------------------------
# 1. CONTINUOUS GEOLOGICAL ROCK FORMULATION
# -------------------------------------------------------------
print("Synthesizing multi-octave geological rock field...")
# Downsampled noise upscaled smoothly for fast computation of 4K noise
noise_low = np.random.RandomState(42).randn(dim // 8, dim // 8).astype(np.float32)
noise_mid = np.random.RandomState(101).randn(dim // 4, dim // 4).astype(np.float32)
noise_fine = np.random.RandomState(888).randn(dim // 2, dim // 2).astype(np.float32)

rock_f1 = ndi.zoom(ndi.gaussian_filter(noise_low, 3.0), 8.0, order=1)
rock_f2 = ndi.zoom(ndi.gaussian_filter(noise_mid, 2.0), 4.0, order=1)
rock_f3 = ndi.zoom(ndi.gaussian_filter(noise_fine, 1.5), 2.0, order=1)

rock_noise = 0.50 * rock_f1 + 0.32 * rock_f2 + 0.18 * rock_f3
rock_noise = (rock_noise - np.mean(rock_noise)) / np.std(rock_noise)

# Geological strata angle (tilted metamorphic banding)
strata = 0.06 * np.sin((X * 0.22 + Y * 0.12 + Z * 0.28) * 1.5)

# Tonal variation
cliff_factor = smoothstep(24.0, 56.0, slope_deg)
alt_factor = smoothstep(10.0, 55.0, RELIEF)
t_rock = np.clip(0.50 - 0.38 * cliff_factor + 0.32 * alt_factor + 0.07 * rock_noise + strata, 0.0, 1.0)

# Colors
slate_rgb = np.array([0.20, 0.22, 0.25], dtype=np.float32)
granite_rgb = np.array([0.56, 0.58, 0.61], dtype=np.float32)
quartz_rgb = np.array([0.68, 0.66, 0.62], dtype=np.float32)

rock_rgb = np.zeros((dim, dim, 3), dtype=np.float32)
for c in range(3):
    rock_rgb[:, :, c] = np.where(
        t_rock < 0.5,
        (1.0 - t_rock*2.0) * slate_rgb[c] + (t_rock*2.0) * granite_rgb[c],
        (1.0 - (t_rock-0.5)*2.0) * granite_rgb[c] + ((t_rock-0.5)*2.0) * quartz_rgb[c]
    )

# -------------------------------------------------------------
# 2. CONTINUOUS SNOW & COULOIRS
# -------------------------------------------------------------
print("Synthesizing continuous snow cover and couloirs...")
snow_alt = smoothstep(18.0, 36.0, RELIEF)
snow_shed = 1.0 - smoothstep(36.0, 66.0, slope_deg)
couloir_flow = smoothstep(-0.8, -0.05, laplace) * smoothstep(10.0, 30.0, RELIEF)

snow_alpha = np.clip(snow_alt * (0.35 + 0.65 * snow_shed) + 0.45 * couloir_flow + 0.08 * rock_f2, 0.0, 1.0)
snow_alpha = smoothstep(0.22, 0.72, snow_alpha)

snow_base_rgb = np.zeros((dim, dim, 3), dtype=np.float32)
snow_shade_factor = smoothstep(15.0, 50.0, slope_deg)
snow_base_rgb[:, :, 0] = (1.0 - 0.12 * snow_shade_factor) * 0.97
snow_base_rgb[:, :, 1] = (1.0 - 0.07 * snow_shade_factor) * 0.98
snow_base_rgb[:, :, 2] = 1.00

# -------------------------------------------------------------
# 3. CONTINUOUS GLACIAL ICE
# -------------------------------------------------------------
print("Synthesizing glacial ice cirques...")
ice_basin = smoothstep(14.0, 35.0, RELIEF) * smoothstep(-1.2, -0.2, laplace) * (1.0 - snow_alt * 0.6)
ice_alpha = np.clip(ice_basin + 0.08 * rock_f2, 0.0, 1.0)
ice_alpha = smoothstep(0.24, 0.68, ice_alpha) * (1.0 - snow_alpha * 0.85)
ice_rgb = np.array([0.42, 0.72, 0.88], dtype=np.float32)

# -------------------------------------------------------------
# 4. CONTINUOUS TUNDRA & LICHEN (SOFT FEATHERING)
# -------------------------------------------------------------
print("Synthesizing alpine tundra and lichen colonies...")
tundra_alt = smoothstep(0.1, 1.5, RELIEF) * (1.0 - smoothstep(3.5, 9.5, RELIEF))
tundra_slope = 1.0 - smoothstep(12.0, 24.0, slope_deg)
tundra_pattern = smoothstep(0.15, 0.75, 0.65 * rock_f1 + 0.35 * rock_f2)

tundra_alpha = np.clip(tundra_alt * tundra_slope * tundra_pattern * 1.5, 0.0, 1.0)
tundra_alpha = smoothstep(0.18, 0.62, tundra_alpha) * (1.0 - snow_alpha)

moss_deep = np.array([0.22, 0.44, 0.18], dtype=np.float32)
lichen_edge = np.array([0.55, 0.58, 0.26], dtype=np.float32)
tundra_color = np.zeros((dim, dim, 3), dtype=np.float32)
for c in range(3):
    tundra_color[:, :, c] = tundra_alpha * moss_deep[c] + (1.0 - tundra_alpha) * lichen_edge[c]

# -------------------------------------------------------------
# 5. CONTINUOUS BLEND
# -------------------------------------------------------------
print("Blending multi-material continuous spectral color layers...")
diffuse_rgb = rock_rgb.copy()
for c in range(3):
    diffuse_rgb[:, :, c] = (1.0 - tundra_alpha) * diffuse_rgb[:, :, c] + tundra_alpha * tundra_color[:, :, c]
for c in range(3):
    diffuse_rgb[:, :, c] = (1.0 - ice_alpha) * diffuse_rgb[:, :, c] + ice_alpha * ice_rgb[c]
for c in range(3):
    diffuse_rgb[:, :, c] = (1.0 - snow_alpha) * diffuse_rgb[:, :, c] + snow_alpha * snow_base_rgb[:, :, c]

# Road margins & Base
road_margin = smoothstep(1.5, 3.5, dist_edge)
base_black = np.array([0.12, 0.13, 0.15], dtype=np.float32)
for c in range(3):
    diffuse_rgb[:, :, c] = (1.0 - (1.0 - road_margin)) * diffuse_rgb[:, :, c] + (1.0 - road_margin) * base_black[c]

# Outside hex is transparent or black
mask_in_hex = dist_edge >= 0.0
diffuse_rgb[~mask_in_hex] = [0.08, 0.09, 0.11]
diffuse_rgb = np.clip(diffuse_rgb, 0.0, 1.0)

# Save as PNG
out_png = r"Tableros Catan\Loseta_Montana_3D\catan_mountain_diffuse_4k.png"
img_u8 = (diffuse_rgb * 255.0).astype(np.uint8)
im = Image.fromarray(img_u8[::-1]) # Flip Y for standard UV orientation
im.save(out_png, optimize=True)
print(f"Saved 4096x4096 Diffuse Texture: {out_png} ({os.path.getsize(out_png)/1024/1024:.2f} MB)")

# Also save a 2048x2048 version for ultra-fast WebGL viewer loading
im_2k = im.resize((2048, 2048), Image.Resampling.LANCZOS)
out_2k = r"Tableros Catan\Loseta_Montana_3D\catan_mountain_diffuse_2k.png"
im_2k.save(out_2k, optimize=True)
print(f"Saved 2048x2048 Diffuse Texture: {out_2k} ({os.path.getsize(out_2k)/1024/1024:.2f} MB)")

# Save diffuse array to scratch
np.save('scratch/diffuse_rgb_4k.npy', diffuse_rgb)
print(f"Total texture generation time: {time.time()-t0:.2f}s")
