import os
import math
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from PIL import Image

Z_master = np.load('scratch/Z_top_master_2m.npy')
dim = 2048
R = 64.0
APOTHEM = R * math.cos(math.radians(30))
Z_BASE = 25.0
Z_PEAK = 83.0

xs = np.linspace(-APOTHEM, APOTHEM, dim, dtype=np.float32)
ys = np.linspace(-R, R, dim, dtype=np.float32)
dx = float(xs[1] - xs[0])
dy = float(ys[1] - ys[0])

import scipy.interpolate as interp
ny_m, nx_m = Z_master.shape
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

X, Y = np.meshgrid(xs, ys)
pts = np.column_stack((Y.ravel(), X.ravel()))
Z = interp_z(pts).reshape((dim, dim))

# Gentle smooth of Z for normal calculation to eliminate stucco noise
Z_smooth = ndi.gaussian_filter(Z, 2.0)
RELIEF = np.maximum(0.0, Z - Z_BASE)
total_relief = Z_PEAK - Z_BASE # 58.0 mm
rel_h = RELIEF / total_relief # 0.0 to 1.0

# Derivatives from smoothed Z
gy, gx = np.gradient(Z_smooth, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm
Ny = -gy / N_norm
Nz = 1.0 / N_norm
slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

laplace = ndi.laplace(Z_smooth)
laplace_s = ndi.gaussian_filter(laplace, 2.0)

# Hex boundary mask
normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

greys_45 = np.load('scratch/greys_45.npy')
greens_15 = np.load('scratch/greens_15.npy')

# -------------------------------------------------------------
# ICONIC 5K PEAK SNOW MANTLE (ALPAMAYO / MONT BLANC STYLE)
# -------------------------------------------------------------
# Snowline sits at rel_h = 0.20 (Z = 36.6 mm, relief = 11.6 mm)
# Above rel_h = 0.40 (Z = 48.2 mm), permanent glaciers and snowfields cover the entire mountain!
snow_elevation = smoothstep(0.18, 0.42, rel_h)

# Glacial tongues reach deep down valleys (laplace_s < 0) down to rel_h = 0.08 (Z = 29.6 mm)
glacial_tongues = smoothstep(0.02, -0.22, laplace_s) * smoothstep(0.06, 0.30, rel_h)

# Macro snowfield coverage
macro_snow = np.clip(snow_elevation + 0.65 * glacial_tongues, 0.0, 1.0)

# Rock exposure ONLY on:
# 1. Knife-edge arêtes (positive laplacian - wind blows snow off ridges)
arete_rock = smoothstep(0.03, 0.15, laplace_s) * (1.0 - smoothstep(0.70, 0.95, rel_h) * 0.7)

# 2. Extreme vertical cliffs (slopes > 70 deg)
cliff_rock = smoothstep(68.0, 78.0, slope_deg) * (1.0 - smoothstep(0.60, 0.90, rel_h) * 0.5)

# Overall rock exposure (protruding through the snow like Matterhorn / Alpamayo ribs)
rock_exposure = np.clip(arete_rock * 0.85 + cliff_rock * 0.75, 0.0, 1.0)

# Snow mask:
snow_mask = macro_snow * (1.0 - rock_exposure)
snow_mask = smoothstep(0.15, 0.55, snow_mask)

# Sun lighting
az_sun = 310.0 * math.pi / 180.0
alt_sun = 46.0 * math.pi / 180.0
sun_dir = np.array([math.cos(alt_sun) * math.sin(az_sun), math.cos(alt_sun) * math.cos(az_sun), math.sin(alt_sun)], dtype=np.float32)
sun_dot = np.clip(Nx * sun_dir[0] + Ny * sun_dir[1] + Nz * sun_dir[2], 0.0, 1.0)

# Snow color: Radiant White + Subtle Cyan Ice in glacial hollows
col_snow_sun = np.array([0.98, 0.99, 1.00], dtype=np.float32) # Pure radiant alpine white
col_snow_shade = np.array([0.86, 0.91, 0.97], dtype=np.float32) # Shaded alpine blue-white
col_glacier_ice = np.array([0.75, 0.90, 0.98], dtype=np.float32) # Glacial cyan

ice_hollows = smoothstep(0.0, -0.30, laplace_s) * smoothstep(0.15, 0.60, rel_h)

snow_col = np.zeros((dim, dim, 3), dtype=np.float32)
for c in range(3):
    s = (1.0 - 0.20 * (1.0 - sun_dot)) * col_snow_sun[c] + 0.20 * (1.0 - sun_dot) * col_snow_shade[c]
    snow_col[:, :, c] = (1.0 - 0.25 * ice_hollows) * s + 0.25 * ice_hollows * col_glacier_ice[c]

# Rock color: Alpine Granite (Tone 14 to Tone 34)
rock_shade = 0.25 + 0.40 * smoothstep(30.0, 75.0, slope_deg) - 0.20 * sun_dot + 0.20 * smoothstep(0.05, -0.25, laplace_s)
rock_shade = np.clip(rock_shade, 0.0, 1.0)
rock_grey_idx = 14.0 + rock_shade * 20.0 # tone 14 to tone 34
rock_grey_idx = np.clip(rock_grey_idx, 0.0, 43.999)

r_lo = np.floor(rock_grey_idx).astype(np.int32)
r_hi = np.minimum(r_lo + 1, 44)
r_frac = (rock_grey_idx - r_lo)[:, :, np.newaxis]
rock_col = (1.0 - r_frac) * greys_45[r_lo] + r_frac * greys_45[r_hi]

# Base Skirt Tundra (15 Greens):
tundra_pot = smoothstep(0.005, 0.02, rel_h) * (1.0 - smoothstep(0.04, 0.10, rel_h))
tundra_slope = 1.0 - smoothstep(6.0, 24.0, slope_deg)
tundra_mask = np.clip(tundra_pot * tundra_slope * (1.0 - snow_mask), 0.0, 1.0)
tundra_mask = smoothstep(0.15, 0.55, tundra_mask)

green_t = np.clip(0.35 + 0.40 * smoothstep(-0.2, 0.1, -laplace_s) - 0.25 * sun_dot, 0.0, 1.0)
g_idx = np.clip(green_t * 14.0, 0.0, 13.999)
g_lo = np.floor(g_idx).astype(np.int32)
g_hi = np.minimum(g_lo + 1, 14)
g_frac = (g_idx - g_lo)[:, :, np.newaxis]
tundra_col = (1.0 - g_frac) * greens_15[g_lo] + g_frac * greens_15[g_hi]

# Composite
diffuse = rock_col.copy()
for c in range(3):
    diffuse[:, :, c] = (1.0 - tundra_mask) * diffuse[:, :, c] + tundra_mask * tundra_col[:, :, c]
    diffuse[:, :, c] = (1.0 - snow_mask) * diffuse[:, :, c] + snow_mask * snow_col[:, :, c]

# Clean base road
road_border = smoothstep(0.0, 0.015, rel_h)
road_grey = greys_45[22]
for c in range(3):
    diffuse[:, :, c] = (1.0 - road_border) * road_grey[c] + road_border * diffuse[:, :, c]

# 3D Shading
ao = 1.0 - np.clip(-laplace_s * 0.30, 0.0, 0.40)
sun_c = np.array([1.06, 1.02, 0.96])
sky_c = np.array([0.74, 0.86, 0.98])
amb_c = np.array([0.28, 0.30, 0.35])

diff_sky = np.clip(Nz * 0.7 + Ny * 0.5, 0.0, 1.0)

# Save raw diffuse map directly
Image.fromarray((np.clip(diffuse, 0.0, 1.0) * 255.0).astype(np.uint8)).save("scratch/raw_diffuse_view.png")
print("Saved scratch/raw_diffuse_view.png!")

diff_sun = sun_dot
lit = diffuse * (
    diff_sun[:, :, None] * sun_c[None, None, :] * 0.72 +
    diff_sky[:, :, None] * sky_c[None, None, :] * 0.35 +
    amb_c[None, None, :] * ao[:, :, None] * 0.38
)
lit[dist_edge < 0] = [0.06, 0.08, 0.12]
lit = np.clip(lit, 0.0, 1.0)

# Isometric 3D Render
fig = plt.figure(figsize=(13, 11), facecolor='#090d16')
ax = fig.add_subplot(111, projection='3d', facecolor='#090d16')
step = 4
surf_x = X[::step, ::step]; surf_y = Y[::step, ::step]; surf_z = Z[::step, ::step]
rgba_sub = np.zeros((surf_x.shape[0], surf_x.shape[1], 4))
rgba_sub[:, :, :3] = lit[::step, ::step]
rgba_sub[:, :, 3] = 1.0
rgba_sub[dist_edge[::step, ::step] < -0.2, 3] = 0.0

ax.plot_surface(surf_x, surf_y, surf_z, facecolors=rgba_sub, rstride=1, cstride=1, shade=False, antialiased=True)
ax.view_init(elev=34, azim=-52)
ax.set_box_aspect([1.0, 1.15, 0.72])
ax.set_title("Loseta Montaña Catan - 5K Glaciated Peak (Alpamayo / Mont Blanc)", fontsize=14, fontweight='bold', color='#38bdf8', pad=14)
ax.axis('off')
plt.tight_layout()
plt.savefig("scratch/test_5k_v2_isometric.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

# Ortho
fig, ax = plt.subplots(figsize=(8, 8), facecolor='#090d16')
ax.imshow(lit, origin='lower')
ax.axis('off')
plt.tight_layout()
plt.savefig("scratch/test_5k_v2_ortho.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

print("Snow coverage on mountain (>0.7):", (diffuse[dist_edge > 0] > 0.7).mean())
print("Saved scratch/test_5k_v2_isometric.png and scratch/test_5k_v2_ortho.png!")
