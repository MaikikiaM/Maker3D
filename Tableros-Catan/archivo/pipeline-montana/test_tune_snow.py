import os
import math
import numpy as np
import scipy.ndimage as ndi
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
RELIEF = np.maximum(0.0, Z - Z_BASE)
total_relief = Z_PEAK - Z_BASE
rel_h = RELIEF / total_relief

# Derivatives
gy, gx = np.gradient(Z, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm
Ny = -gy / N_norm
Nz = 1.0 / N_norm
slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

laplace = ndi.laplace(Z)
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
# DESIGNING THE ICONIC 5K REGIME (MONT BLANC / ALPAMAYO)
# -------------------------------------------------------------
# 1. Snow Line & Elevation:
# Snow starts at rel_h = 0.28 (Z = 41.2 mm), with glacial tongues reaching down to rel_h = 0.12 (Z = 32 mm)
# Above rel_h = 0.50 (Z = 54 mm), snow is the dominant mantle!
snow_elevation = smoothstep(0.22, 0.48, rel_h)

# Glacial tongues flowing down concave valley troughs
glacial_tongues = smoothstep(0.02, -0.25, laplace_s) * smoothstep(0.08, 0.35, rel_h)

# High summit snow dominance (Mont Blanc dome effect)
summit_cap = smoothstep(0.50, 0.75, rel_h)

base_snow = np.clip(snow_elevation + 0.55 * glacial_tongues + 0.35 * summit_cap, 0.0, 1.0)

# 2. Rock Exposure (Arêtes and Vertical Cliffs):
# Real 5k mountains: Rock breaks through snow along knife-edge arêtes (positive laplacian)
# and on extreme sheer cliffs (> 70 degrees).
# Even on 60° slopes, snow/ice clings firmly!
cliff_rock = smoothstep(68.0, 78.0, slope_deg) * (1.0 - 0.60 * summit_cap)
arete_rock = smoothstep(0.05, 0.22, laplace_s) * (1.0 - 0.70 * summit_cap)

# Total rock breakthrough
rock_breakthrough = np.clip(cliff_rock * 0.90 + arete_rock * 0.75, 0.0, 1.0)

# Final Snow Mask:
snow_mask = np.clip(base_snow * (1.0 - rock_breakthrough), 0.0, 1.0)
snow_mask = smoothstep(0.20, 0.60, snow_mask)

# 3. Snow Coloration:
# Dazzling alpine white, soft shaded blue-grey, cyan glacial ice
az_sun = 310.0 * math.pi / 180.0
alt_sun = 46.0 * math.pi / 180.0
sun_dir = np.array([math.cos(alt_sun) * math.sin(az_sun), math.cos(alt_sun) * math.cos(az_sun), math.sin(alt_sun)], dtype=np.float32)
sun_dot = np.clip(Nx * sun_dir[0] + Ny * sun_dir[1] + Nz * sun_dir[2], 0.0, 1.0)

col_snow_sun = np.array([0.98, 0.99, 1.00], dtype=np.float32) # Pure radiant alpine white
col_snow_shade = np.array([0.84, 0.89, 0.96], dtype=np.float32) # Shaded alpine blue-white
col_glacier_ice = np.array([0.72, 0.88, 0.96], dtype=np.float32) # Glacial cyan

ice_hollows = smoothstep(0.0, -0.30, laplace_s) * smoothstep(0.15, 0.60, rel_h)

snow_col = np.zeros((dim, dim, 3), dtype=np.float32)
for c in range(3):
    s = (1.0 - 0.22 * (1.0 - sun_dot)) * col_snow_sun[c] + 0.22 * (1.0 - sun_dot) * col_snow_shade[c]
    snow_col[:, :, c] = (1.0 - 0.30 * ice_hollows) * s + 0.30 * ice_hollows * col_glacier_ice[c]

# 4. Alpine Granite & Bedrock (The 45 Greys):
# Majestic mountain granite with silver arêtes and deep clefts
rock_shade = 0.30 + 0.35 * smoothstep(35.0, 75.0, slope_deg) - 0.20 * sun_dot + 0.20 * smoothstep(0.05, -0.25, laplace_s)
rock_shade = np.clip(rock_shade, 0.0, 1.0)
rock_grey_idx = 14.0 + rock_shade * 22.0 # tone 14 to tone 36
rock_grey_idx = np.clip(rock_grey_idx, 0.0, 43.999)

r_lo = np.floor(rock_grey_idx).astype(np.int32)
r_hi = np.minimum(r_lo + 1, 44)
r_frac = (rock_grey_idx - r_lo)[:, :, np.newaxis]
rock_col = (1.0 - r_frac) * greys_45[r_lo] + r_frac * greys_45[r_hi]

# 5. Alpine Base Skirt (15 Greens):
tundra_pot = smoothstep(0.005, 0.02, rel_h) * (1.0 - smoothstep(0.05, 0.12, rel_h))
tundra_slope = 1.0 - smoothstep(8.0, 26.0, slope_deg)
tundra_mask = np.clip(tundra_pot * tundra_slope * (1.0 - snow_mask), 0.0, 1.0)
tundra_mask = smoothstep(0.15, 0.60, tundra_mask)

green_t = np.clip(0.35 + 0.40 * smoothstep(-0.2, 0.1, -laplace_s) - 0.25 * sun_dot, 0.0, 1.0)
g_idx = np.clip(green_t * 14.0, 0.0, 13.999)
g_lo = np.floor(g_idx).astype(np.int32)
g_hi = np.minimum(g_lo + 1, 14)
g_frac = (g_idx - g_lo)[:, :, np.newaxis]
tundra_col = (1.0 - g_frac) * greens_15[g_lo] + g_frac * greens_15[g_hi]

# 6. Composite
diffuse = rock_col.copy()
for c in range(3):
    diffuse[:, :, c] = (1.0 - tundra_mask) * diffuse[:, :, c] + tundra_mask * tundra_col[:, :, c]
    diffuse[:, :, c] = (1.0 - snow_mask) * diffuse[:, :, c] + snow_mask * snow_col[:, :, c]

road_border = smoothstep(0.0, 0.015, rel_h)
road_grey = greys_45[20]
for c in range(3):
    diffuse[:, :, c] = (1.0 - road_border) * road_grey[c] + road_border * diffuse[:, :, c]

# Fast 3D Lighting & Shading for Inspection
ao = 1.0 - np.clip(-laplace_s * 0.35, 0.0, 0.45)
sun_c = np.array([1.05, 1.01, 0.95])
sky_c = np.array([0.72, 0.85, 0.98])
amb_c = np.array([0.25, 0.28, 0.34])

diff_sun = np.clip(Nx * sun_dir[0] + Ny * sun_dir[1] + Nz * sun_dir[2], 0.0, 1.0)
diff_sky = np.clip(Nz * 0.7 + Ny * 0.5, 0.0, 1.0)

lit = diffuse * (
    diff_sun[:, :, None] * sun_c[None, None, :] * 0.70 +
    diff_sky[:, :, None] * sky_c[None, None, :] * 0.35 +
    amb_c[None, None, :] * ao[:, :, None] * 0.40
)
lit[dist_edge < 0] = [0.06, 0.08, 0.12]
lit = np.clip(lit, 0.0, 1.0)

# Save test render
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 10), facecolor='#090d16')
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
ax.set_title("Test 5K Glaciated Peak (Mont Blanc / Alpamayo Regime)", fontsize=14, fontweight='bold', color='#38bdf8', pad=14)
ax.axis('off')
plt.tight_layout()
plt.savefig("scratch/test_5k_isometric.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

# Also save top ortho
fig, ax = plt.subplots(figsize=(8, 8), facecolor='#090d16')
ax.imshow(lit, origin='lower')
ax.axis('off')
plt.tight_layout()
plt.savefig("scratch/test_5k_ortho.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

print("Snow coverage (>0.7):", (diffuse > 0.7).mean())
print("Saved scratch/test_5k_isometric.png and scratch/test_5k_ortho.png!")
