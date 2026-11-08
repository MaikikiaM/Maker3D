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
Z_smooth = ndi.gaussian_filter(Z, 2.0)

RELIEF = np.maximum(0.0, Z - Z_BASE)
total_relief = Z_PEAK - Z_BASE # 58.0 mm
rel_h = RELIEF / total_relief # 0.0 to 1.0

# Derivatives
gy, gx = np.gradient(Z_smooth, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm
Ny = -gy / N_norm
Nz = 1.0 / N_norm
slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

laplace = ndi.laplace(Z_smooth)
laplace_s = ndi.gaussian_filter(laplace, 2.5)

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
# 1. ICONIC 5K ALPINE SNOWPACK & GLACIAL REGIME (ALPAMAYO / MONT BLANC)
# -------------------------------------------------------------
# Permanent snowfield starts at rel_h = 0.22 (Z = 37.8 mm), reaches 100% on summit dome
snow_elev = smoothstep(0.18, 0.44, rel_h)

# Glacial tongues extending down concave valley hollows (laplace_s < 0)
glacial_tongues = smoothstep(0.01, -0.20, laplace_s) * smoothstep(0.06, 0.32, rel_h)

# Summit cap dominance (Z >= 65 mm)
summit_cap = smoothstep(0.60, 0.88, rel_h)

macro_snow = np.clip(snow_elev + 0.60 * glacial_tongues + 0.25 * summit_cap, 0.0, 1.0)

# Rock breakthrough:
# Real 5k mountains (Matterhorn, Mont Blanc, Ama Dablam) have sharp dark granite arêtes!
# 1. Arête ridges: sharp positive curvature (laplace_s > 0.025)
arete_rock = smoothstep(0.02, 0.12, laplace_s) * (1.0 - summit_cap * 0.50)

# 2. Sheer rock walls & headwalls: slopes > 65 deg
cliff_rock = smoothstep(64.0, 75.0, slope_deg) * (1.0 - summit_cap * 0.45)

# Total rock ribs protruding through snowpack
rock_breakthrough = np.clip(arete_rock * 0.95 + cliff_rock * 0.85, 0.0, 1.0)

# Final snow coverage mask
snow_mask = np.clip(macro_snow * (1.0 - rock_breakthrough), 0.0, 1.0)
snow_mask = smoothstep(0.18, 0.58, snow_mask)

# 2. Colors
# Snow: Dazzling alpine white + subtle cyan glacial ice in hollows
col_snow_sun = np.array([0.99, 0.99, 1.00], dtype=np.float32)
col_snow_shade = np.array([0.88, 0.93, 0.98], dtype=np.float32)
col_glacier_ice = np.array([0.72, 0.88, 0.98], dtype=np.float32)

ice_hollows = smoothstep(0.0, -0.28, laplace_s) * smoothstep(0.15, 0.65, rel_h)
snow_col = np.zeros((dim, dim, 3), dtype=np.float32)
for c in range(3):
    snow_col[:, :, c] = (1.0 - 0.28 * ice_hollows) * col_snow_sun[c] + 0.28 * ice_hollows * col_glacier_ice[c]

# Alpine Bedrock (45 Greys):
# Bold, dark granite ribs that contrast dramatically against the white snow!
# Arêtes: Tone 20-26 (medium cold granite)
# Sheer cliffs: Tone 28-36 (dark slate/basalt)
rock_shade = 0.35 + 0.45 * smoothstep(35.0, 75.0, slope_deg) + 0.20 * smoothstep(0.05, -0.25, laplace_s)
rock_shade = np.clip(rock_shade, 0.0, 1.0)
rock_grey_idx = 18.0 + rock_shade * 20.0 # tone 18 to tone 38
rock_grey_idx = np.clip(rock_grey_idx, 0.0, 43.999)

r_lo = np.floor(rock_grey_idx).astype(np.int32)
r_hi = np.minimum(r_lo + 1, 44)
r_frac = (rock_grey_idx - r_lo)[:, :, np.newaxis]
rock_col = (1.0 - r_frac) * greys_45[r_lo] + r_frac * greys_45[r_hi]

# Base Skirt Tundra (15 Greens):
# Rich botanical apron at Z in [25.1, 29.5] mm
tundra_pot = smoothstep(0.005, 0.02, rel_h) * (1.0 - smoothstep(0.045, 0.11, rel_h))
tundra_slope = 1.0 - smoothstep(6.0, 26.0, slope_deg)
tundra_mask = np.clip(tundra_pot * tundra_slope * (1.0 - snow_mask), 0.0, 1.0)
tundra_mask = smoothstep(0.12, 0.50, tundra_mask)

green_t = np.clip(0.35 + 0.45 * smoothstep(-0.2, 0.1, -laplace_s), 0.0, 1.0)
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

road_border = smoothstep(0.0, 0.015, rel_h)
road_grey = greys_45[22]
for c in range(3):
    diffuse[:, :, c] = (1.0 - road_border) * road_grey[c] + road_border * diffuse[:, :, c]

# Studio Three-Point Lighting
L_key = np.array([0.45, -0.55, 0.70], dtype=np.float32)
L_key /= np.linalg.norm(L_key)
diff_key = np.clip(Nx * L_key[0] + Ny * L_key[1] + Nz * L_key[2], 0.0, 1.0)

L_fill = np.array([-0.60, 0.40, 0.69], dtype=np.float32)
L_fill /= np.linalg.norm(L_fill)
diff_fill = np.clip(Nx * L_fill[0] + Ny * L_fill[1] + Nz * L_fill[2], 0.0, 1.0)

diff_top = np.clip(Nz, 0.0, 1.0)

col_key = np.array([1.06, 1.03, 0.98])
col_fill = np.array([0.72, 0.85, 0.98])
col_amb = np.array([0.42, 0.44, 0.48])

lit = diffuse * (
    diff_key[:, :, None] * col_key[None, None, :] * 0.66 +
    diff_fill[:, :, None] * col_fill[None, None, :] * 0.28 +
    diff_top[:, :, None] * col_amb[None, None, :] * 0.36
)
lit[dist_edge < 0] = [0.06, 0.08, 0.12]
lit = np.clip(lit, 0.0, 1.0)

# Save Studio Isometric
fig = plt.figure(figsize=(13, 11), facecolor='#090d16')
ax = fig.add_subplot(111, projection='3d', facecolor='#090d16')
step = 4
surf_x = X[::step, ::step]; surf_y = Y[::step, ::step]; surf_z = Z_smooth[::step, ::step]
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
plt.savefig("scratch/test_studio_v3_isometric.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

# Save Studio Ortho
fig, ax = plt.subplots(figsize=(8, 8), facecolor='#090d16')
ax.imshow(lit, origin='lower')
ax.axis('off')
plt.tight_layout()
plt.savefig("scratch/test_studio_v3_ortho.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

print("V3 Studio render saved! Snow coverage:", (diffuse[dist_edge > 0] > 0.7).mean())
