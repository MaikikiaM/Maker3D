import math
import numpy as np
import scipy.ndimage as ndi
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from PIL import Image

# 1. Load DEM
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

# Derivatives & Geomorphometry
gy, gx = np.gradient(Z_smooth, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm
Ny = -gy / N_norm
Nz = 1.0 / N_norm
slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

# Curvature smoothed to capture solid continuous geological structures
laplace = ndi.laplace(Z_smooth)
laplace_ridge = ndi.gaussian_filter(laplace, 4.0) # macro ridges
laplace_detail = ndi.gaussian_filter(laplace, 1.8) # micro details

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
# 2. MAJESTIC 5K GLACIATED MANTLE & CHISELED GRANITE ARÊTES
# -------------------------------------------------------------
# Solid snowpack coverage:
# Snow starts at rel_h = 0.12 (Z = 32 mm)
snow_base = smoothstep(0.10, 0.35, rel_h)

# Glacial tongues flowing down hollows (laplace_ridge < 0) deep into foothills
glacial_tongues = smoothstep(0.01, -0.06, laplace_ridge) * smoothstep(0.04, 0.28, rel_h)

# Summit cap: pristine, blinding white snow dome (Z >= 55 mm)
summit_cap = smoothstep(0.50, 0.85, rel_h)

# Thick continuous snow blanket
snow_blanket = np.clip(snow_base + 0.65 * glacial_tongues + 0.40 * summit_cap, 0.0, 1.0)

# True Alpine Rock Exposure:
# Continuous, solid granite arêtes along the knife edges (laplace_ridge > 0.02)
arete_spines = smoothstep(0.018, 0.065, laplace_ridge)
# Sheer vertical cliffs (> 66 degrees)
cliff_walls = smoothstep(65.0, 75.0, slope_deg)

# Combine into clean, structural geological rock ribs
rock_ribs = np.clip(arete_spines * 0.95 + cliff_walls * 0.85, 0.0, 1.0)
# Summit cap keeps thick snow except on the sharpest crest
rock_ribs *= (1.0 - summit_cap * 0.60)

# Clean, cohesive snow mask (NO noisy speckling!)
snow_mask = np.clip(snow_blanket * (1.0 - rock_ribs), 0.0, 1.0)
snow_mask = smoothstep(0.18, 0.60, snow_mask)

# -------------------------------------------------------------
# 3. COLOR PALETTES
# -------------------------------------------------------------
# Snow: Blinding alpine white + subtle glacial blue/cyan in hollows
col_snow_sun = np.array([0.99, 0.99, 1.00], dtype=np.float32)
col_glacier_ice = np.array([0.76, 0.90, 0.98], dtype=np.float32)
ice_hollows = smoothstep(0.0, -0.07, laplace_ridge) * smoothstep(0.10, 0.65, rel_h)

snow_col = np.zeros((dim, dim, 3), dtype=np.float32)
for c in range(3):
    snow_col[:, :, c] = (1.0 - 0.20 * ice_hollows) * col_snow_sun[c] + 0.20 * ice_hollows * col_glacier_ice[c]

# Alpine Bedrock (45 Greys):
# Chiseled cold granite with deep contrast
rock_shade = 0.25 + 0.45 * smoothstep(30.0, 70.0, slope_deg) + 0.30 * smoothstep(0.03, -0.05, laplace_detail)
rock_shade = np.clip(rock_shade, 0.0, 1.0)
rock_grey_idx = 16.0 + rock_shade * 24.0 # tone 16 to tone 40
rock_grey_idx = np.clip(rock_grey_idx, 0.0, 43.999)

r_lo = np.floor(rock_grey_idx).astype(np.int32)
r_hi = np.minimum(r_lo + 1, 44)
r_frac = (rock_grey_idx - r_lo)[:, :, np.newaxis]
rock_col = (1.0 - r_frac) * greys_45[r_lo] + r_frac * greys_45[r_hi]

# Base Skirt Foothills (15 Greens):
# Vibrant alpine pastures wrapping around the base (Z in [25.0, 32.5] mm)
tundra_pot = smoothstep(0.002, 0.015, rel_h) * (1.0 - smoothstep(0.08, 0.20, rel_h))
tundra_slope = 1.0 - smoothstep(12.0, 38.0, slope_deg)
tundra_mask = np.clip(tundra_pot * tundra_slope * (1.0 - snow_mask * 0.95), 0.0, 1.0)
tundra_mask = smoothstep(0.08, 0.45, tundra_mask)

green_t = np.clip(0.30 + 0.50 * smoothstep(-0.06, 0.03, -laplace_detail), 0.0, 1.0)
g_idx = np.clip(green_t * 14.0, 0.0, 13.999)
g_lo = np.floor(g_idx).astype(np.int32)
g_hi = np.minimum(g_lo + 1, 14)
g_frac = (g_idx - g_lo)[:, :, np.newaxis]
tundra_col = (1.0 - g_frac) * greens_15[g_lo] + g_frac * greens_15[g_hi]

# Composite Diffuse (Albedo)
diffuse = rock_col.copy()
for c in range(3):
    diffuse[:, :, c] = (1.0 - tundra_mask) * diffuse[:, :, c] + tundra_mask * tundra_col[:, :, c]
    diffuse[:, :, c] = (1.0 - snow_mask) * diffuse[:, :, c] + snow_mask * snow_col[:, :, c]

# Hex Flat Border (Tone 22 grey)
road_border = smoothstep(0.0, 0.012, rel_h) * smoothstep(0.2, 1.5, dist_edge)
road_grey = greys_45[22]
for c in range(3):
    diffuse[:, :, c] = (1.0 - road_border) * road_grey[c] + road_border * diffuse[:, :, c]

# Save pure diffuse
diff_masked = diffuse.copy()
diff_masked[dist_edge < 0] = [0.07, 0.09, 0.13]
Image.fromarray((np.clip(diff_masked, 0, 1) * 255).astype(np.uint8)).save("scratch/test_diffuse_v7.png")

# Balanced Multi-Light Studio Rig
L1 = np.array([0.5, -0.6, 0.65], dtype=np.float32); L1 /= np.linalg.norm(L1)
d1 = np.clip(Nx * L1[0] + Ny * L1[1] + Nz * L1[2], 0.0, 1.0)
L2 = np.array([-0.5, -0.6, 0.65], dtype=np.float32); L2 /= np.linalg.norm(L2)
d2 = np.clip(Nx * L2[0] + Ny * L2[1] + Nz * L2[2], 0.0, 1.0)
L3 = np.array([0.0, 0.5, 0.75], dtype=np.float32); L3 /= np.linalg.norm(L3)
d3 = np.clip(Nx * L3[0] + Ny * L3[1] + Nz * L3[2], 0.0, 1.0)
d_amb = np.clip(Nz * 0.5 + 0.5, 0.0, 1.0)

curv_high = smoothstep(0.015, 0.06, laplace_detail) * 0.10
lit_factor = 0.38 * d1 + 0.32 * d2 + 0.22 * d3 + 0.25 * d_amb
lit = np.clip(diffuse * lit_factor[:, :, None] + curv_high[:, :, None], 0.0, 1.0)

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
ax.set_title("Loseta Montana Catan - 5K Glaciated Peak (Alpamayo / Mont Blanc)", fontsize=14, fontweight='bold', color='#38bdf8', pad=14)
ax.axis('off')
plt.tight_layout()
plt.savefig("scratch/test_studio_v7_isometric.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

# Save Studio Ortho (Top View)
fig, ax = plt.subplots(figsize=(8, 8), facecolor='#090d16')
ortho_lit = diffuse * (0.65 + 0.35 * np.clip(Nz[:, :, None], 0, 1))
ortho_lit[dist_edge < 0] = [0.06, 0.08, 0.12]
ax.imshow(np.clip(ortho_lit, 0, 1), origin='lower')
ax.axis('off')
plt.tight_layout()
plt.savefig("scratch/test_studio_v7_ortho.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

snow_pct = (snow_mask[dist_edge > 0] > 0.5).mean() * 100
tundra_pct = (tundra_mask[dist_edge > 0] > 0.5).mean() * 100
rock_pct = 100.0 - snow_pct - tundra_pct
print(f"V7 Done! Snow: {snow_pct:.1f}%, Tundra: {tundra_pct:.1f}%, Rock: {rock_pct:.1f}%")
