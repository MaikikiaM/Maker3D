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
Z_smooth = ndi.gaussian_filter(Z, 1.8)

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

# Curvature at multiple scales
laplace_fine = ndi.gaussian_filter(ndi.laplace(Z_smooth), 1.5)
laplace_med = ndi.gaussian_filter(ndi.laplace(Z_smooth), 3.5)

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
# 2. ICONIC 5K GEOMORPHIC SNOW & ROCK SYSTEM (MATTERHORN / MONT BLANC / ALPAMAYO)
# -------------------------------------------------------------
# Permanent alpine snow starts at Z = 32mm (rel_h = 0.12)
# In glacial cirques / hollows (laplace_med < 0), snow flows down to Z = 27mm (rel_h = 0.04)
cirques = smoothstep(0.005, -0.06, laplace_med)
elevation_snow = smoothstep(0.10, 0.40, rel_h)
glacial_flow = cirques * smoothstep(0.03, 0.25, rel_h)

# Summit dome / snowcap
summit_cap = smoothstep(0.55, 0.90, rel_h)

# Base snow capacity
snow_potential = np.clip(elevation_snow * 0.95 + glacial_flow * 0.70 + summit_cap * 0.40, 0.0, 1.0)

# Rock exposure on ridges and steep faces:
# Razor arêtes (sharp positive curvature)
arete_crest = smoothstep(0.015, 0.07, laplace_fine)
# Steep rock buttresses (slope > 50 deg)
steep_buttress = smoothstep(48.0, 68.0, slope_deg)
# Combined rock skeleton
rock_exposure = np.clip(arete_crest * 0.85 + steep_buttress * 0.85, 0.0, 1.0)
# Rock breaks through snowpack, but summit cap has high snow adhesion
rock_exposure *= (1.0 - summit_cap * 0.45)

# Micro fluting on snow ramps
flute_pattern = np.sin((X * 0.7 + Y * 0.7) * 45.0) * 0.06
snow_adhesion = smoothstep(0.35, 0.75, 1.0 - rock_exposure)

# Final Snow Mask (0 = pure rock/tundra, 1 = deep snow/ice)
snow_mask = np.clip(snow_potential * snow_adhesion + flute_pattern * snow_potential, 0.0, 1.0)
snow_mask = smoothstep(0.18, 0.65, snow_mask)

# -------------------------------------------------------------
# 3. COLOR PALETTES & TEXTURING
# -------------------------------------------------------------
# Snow: Dazzling alpine white + subtle glacial blue/cyan in hollows
col_snow_sun = np.array([0.99, 0.99, 1.00], dtype=np.float32)
col_glacier_ice = np.array([0.76, 0.90, 0.98], dtype=np.float32)
ice_hollows = smoothstep(0.0, -0.08, laplace_med) * smoothstep(0.10, 0.70, rel_h)

snow_col = np.zeros((dim, dim, 3), dtype=np.float32)
for c in range(3):
    snow_col[:, :, c] = (1.0 - 0.22 * ice_hollows) * col_snow_sun[c] + 0.22 * ice_hollows * col_glacier_ice[c]

# Alpine Bedrock (45 Greys):
# Dark chiseled granite ribs, shaded crevices, lit rock spurs
rock_shade = 0.25 + 0.50 * smoothstep(30.0, 70.0, slope_deg) + 0.25 * smoothstep(0.04, -0.08, laplace_fine)
rock_shade = np.clip(rock_shade, 0.0, 1.0)
rock_grey_idx = 16.0 + rock_shade * 24.0 # tone 16 to tone 40
rock_grey_idx = np.clip(rock_grey_idx, 0.0, 43.999)

r_lo = np.floor(rock_grey_idx).astype(np.int32)
r_hi = np.minimum(r_lo + 1, 44)
r_frac = (rock_grey_idx - r_lo)[:, :, np.newaxis]
rock_col = (1.0 - r_frac) * greys_45[r_lo] + r_frac * greys_45[r_hi]

# Base Skirt Foothills (15 Greens):
# Rich alpine meadows at Z in [25.0, 33.0] mm
tundra_pot = smoothstep(0.002, 0.015, rel_h) * (1.0 - smoothstep(0.08, 0.20, rel_h))
tundra_slope = 1.0 - smoothstep(12.0, 38.0, slope_deg)
tundra_mask = np.clip(tundra_pot * tundra_slope * (1.0 - snow_mask * 0.95), 0.0, 1.0)
tundra_mask = smoothstep(0.08, 0.45, tundra_mask)

green_t = np.clip(0.30 + 0.50 * smoothstep(-0.08, 0.04, -laplace_fine), 0.0, 1.0)
g_idx = np.clip(green_t * 14.0, 0.0, 13.999)
g_lo = np.floor(g_idx).astype(np.int32)
g_hi = np.minimum(g_lo + 1, 14)
g_frac = (g_idx - g_lo)[:, :, np.newaxis]
tundra_col = (1.0 - g_frac) * greens_15[g_lo] + g_frac * greens_15[g_hi]

# Composite Diffuse
diffuse = rock_col.copy()
for c in range(3):
    diffuse[:, :, c] = (1.0 - tundra_mask) * diffuse[:, :, c] + tundra_mask * tundra_col[:, :, c]
    diffuse[:, :, c] = (1.0 - snow_mask) * diffuse[:, :, c] + snow_mask * snow_col[:, :, c]

# Hex flat border (Tone 22)
road_border = smoothstep(0.0, 0.012, rel_h) * smoothstep(0.2, 1.5, dist_edge)
road_grey = greys_45[22]
for c in range(3):
    diffuse[:, :, c] = (1.0 - road_border) * road_grey[c] + road_border * diffuse[:, :, c]

# Save pure diffuse
diff_masked = diffuse.copy()
diff_masked[dist_edge < 0] = [0.07, 0.09, 0.13]
Image.fromarray((np.clip(diff_masked, 0, 1) * 255).astype(np.uint8)).save("scratch/test_diffuse_v6.png")

# Balanced Multi-Light Studio Rig
L1 = np.array([0.5, -0.6, 0.65], dtype=np.float32); L1 /= np.linalg.norm(L1)
d1 = np.clip(Nx * L1[0] + Ny * L1[1] + Nz * L1[2], 0.0, 1.0)
L2 = np.array([-0.5, -0.6, 0.65], dtype=np.float32); L2 /= np.linalg.norm(L2)
d2 = np.clip(Nx * L2[0] + Ny * L2[1] + Nz * L2[2], 0.0, 1.0)
L3 = np.array([0.0, 0.5, 0.75], dtype=np.float32); L3 /= np.linalg.norm(L3)
d3 = np.clip(Nx * L3[0] + Ny * L3[1] + Nz * L3[2], 0.0, 1.0)
d_amb = np.clip(Nz * 0.5 + 0.5, 0.0, 1.0)

curv_high = smoothstep(0.015, 0.06, laplace_fine) * 0.12
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
plt.savefig("scratch/test_studio_v6_isometric.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

# Save Studio Ortho (Top View)
fig, ax = plt.subplots(figsize=(8, 8), facecolor='#090d16')
ortho_lit = diffuse * (0.65 + 0.35 * np.clip(Nz[:, :, None], 0, 1))
ortho_lit[dist_edge < 0] = [0.06, 0.08, 0.12]
ax.imshow(np.clip(ortho_lit, 0, 1), origin='lower')
ax.axis('off')
plt.tight_layout()
plt.savefig("scratch/test_studio_v6_ortho.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

snow_pct = (snow_mask[dist_edge > 0] > 0.5).mean() * 100
tundra_pct = (tundra_mask[dist_edge > 0] > 0.5).mean() * 100
rock_pct = 100.0 - snow_pct - tundra_pct
print(f"V6 Done! Snow: {snow_pct:.1f}%, Tundra: {tundra_pct:.1f}%, Rock: {rock_pct:.1f}%")
