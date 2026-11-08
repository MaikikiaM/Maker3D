import math
import numpy as np
import scipy.ndimage as ndi
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from PIL import Image

# 1. Load DEM and Raymarched AO
Z_master = np.load('scratch/Z_top_master_2m.npy')
ao_master = np.load('scratch/ao_raymarched_16dir.npy')
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
interp_ao = interp.RegularGridInterpolator((ys_m, xs_m), ao_master, bounds_error=False, fill_value=0.8)

X, Y = np.meshgrid(xs, ys)
pts = np.column_stack((Y.ravel(), X.ravel()))
Z = interp_z(pts).reshape((dim, dim))
AO = interp_ao(pts).reshape((dim, dim))

# Macro smooth for geological features (sigma = 3.5 mm in physical units)
# dx is ~0.054 mm, so sigma_pix = 3.5 / 0.054 = 65 pixels
sigma_macro = int(2.5 / dx)
Z_macro = ndi.gaussian_filter(Z, sigma_macro)

# Medium smooth for local facets (sigma = 0.8 mm)
sigma_med = int(0.8 / dx)
Z_med = ndi.gaussian_filter(Z, sigma_med)

RELIEF = np.maximum(0.0, Z - Z_BASE)
total_relief = Z_PEAK - Z_BASE # 58.0 mm
rel_h = RELIEF / total_relief # 0.0 to 1.0

# Gradients on medium smooth
gy, gx = np.gradient(Z_med, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm
Ny = -gy / N_norm
Nz = 1.0 / N_norm
slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

# Second derivatives on macro smooth for geological skeleton
gy_m, gx_m = np.gradient(Z_macro, dy, dx)
gyy_m, gyx_m = np.gradient(gy_m, dy, dx)
gxy_m, gxx_m = np.gradient(gx_m, dy, dx)

tr = gxx_m + gyy_m
det = gxx_m * gyy_m - 0.5 * (gxy_m + gyx_m)**2
disc = np.maximum(0.0, tr**2 - 4.0 * det)
k1 = 0.5 * (tr + np.sqrt(disc))  # Ridge curvature (positive = ridge spine)
k2 = 0.5 * (tr - np.sqrt(disc))  # Valley curvature (negative = couloir / cirque)

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

# ------------------------------------------------------------------
# 2. GLACIATED MASSIC GEOMORPHIC RULES (Mont Blanc / Alpamayo / Ama Dablam)
# ------------------------------------------------------------------
# 1) CIRQUES & COULOIRS (Glacial flow channels):
# Concave valleys (k2 < 0) collect deep snow and ice
cirque_strength = smoothstep(-0.02, -0.25, k2)

# 2) ARÊTES & RIDGES (Structural granite ribs):
# Convex ridges (k1 > 0) shed snow and expose dark granite
ridge_strength = smoothstep(0.02, 0.25, k1)

# 3) CLIFF FACES:
# Steep slopes shed snow (snow starts sloughing at 42 deg, can't hold at 56 deg)
cliff_strength = smoothstep(42.0, 56.0, slope_deg)

# 4) ELEVATION DRIVERS:
# Permanent snowline starts around Z = 32 mm (rel_h = 0.12)
# Reaches 100% potential at upper elevations (rel_h > 0.45, Z > 51 mm)
elevation_snow = smoothstep(0.10, 0.45, rel_h)

# Glacial tongues flow far down couloirs (down to rel_h = 0.05, Z = 28 mm!)
glacial_tongue = cirque_strength * smoothstep(0.05, 0.35, rel_h)

# Base snow potential: combines general elevation with deep glacial valleys
snow_potential = np.clip(elevation_snow + 0.85 * glacial_tongue, 0.0, 1.0)

# Rock exposure: ridges and steep cliffs cut through the snow
# But at very high elevations (summit cap, rel_h > 0.85), heavy snow caps soften rock
rock_exposure = np.clip(0.65 * ridge_strength + 0.70 * cliff_strength, 0.0, 1.0)
rock_exposure *= (1.0 - 0.45 * smoothstep(0.75, 0.95, rel_h))

# Snow coverage mask:
snow_mask = np.clip(snow_potential - 0.75 * rock_exposure, 0.0, 1.0)
snow_mask = smoothstep(0.18, 0.65, snow_mask)

# Ensure summit peak has glorious snow
summit_boost = smoothstep(0.85, 0.98, rel_h) * (1.0 - smoothstep(62.0, 75.0, slope_deg))
snow_mask = np.clip(snow_mask + 0.4 * summit_boost, 0.0, 1.0)

# ------------------------------------------------------------------
# 3. COLOR PALETTES & TEXTURING
# ------------------------------------------------------------------
# Snow color: Pristine alpine snow with glacial cyan in deep couloirs
col_snow_pure = np.array([0.98, 0.98, 1.00], dtype=np.float32)
col_glacier_cyan = np.array([0.75, 0.88, 0.96], dtype=np.float32)

# Glacial ice appears in deep cirques and hollows
ice_weight = cirque_strength * smoothstep(0.12, 0.70, rel_h)
snow_depth = 0.90 + 0.10 * smoothstep(0.35, 0.85, AO)

snow_col = np.zeros((dim, dim, 3), dtype=np.float32)
for c in range(3):
    base_s = (1.0 - 0.22 * ice_weight) * col_snow_pure[c] + (0.22 * ice_weight) * col_glacier_cyan[c]
    snow_col[:, :, c] = base_s * snow_depth

# Rock color: Rich alpine granite using 45 calibrated greys
# Flatter ledges get slightly lighter grey (talus/scree), steep walls get dark charcoal granite
rock_tone = 14.0 + 18.0 * smoothstep(25.0, 65.0, slope_deg) + 10.0 * (1.0 - smoothstep(0.3, 0.8, AO))
rock_tone = np.clip(rock_tone, 8.0, 43.999)

r_lo = np.floor(rock_tone).astype(np.int32)
r_hi = np.minimum(r_lo + 1, 44)
r_frac = (rock_tone - r_lo)[:, :, np.newaxis]
rock_col = (1.0 - r_frac) * greys_45[r_lo] + r_frac * greys_45[r_hi]

# Tundra / Alpine pasture (15 botanical greens):
# Concentrated on lower gentle slopes (Z in [25.0, 34.0] mm, slope < 32 deg)
tundra_pot = smoothstep(0.002, 0.012, rel_h) * (1.0 - smoothstep(0.06, 0.18, rel_h))
tundra_slope = 1.0 - smoothstep(10.0, 32.0, slope_deg)
tundra_mask = np.clip(tundra_pot * tundra_slope * (1.0 - snow_mask * 0.9), 0.0, 1.0)
tundra_mask = smoothstep(0.08, 0.40, tundra_mask)

# Green tone variety (lower valleys richer, upper scree more mossy/olive)
g_tone = np.clip(0.20 + 0.60 * smoothstep(0.01, 0.14, rel_h) + 0.20 * (1.0 - smoothstep(0.4, 0.8, AO)), 0.0, 1.0)
g_idx = np.clip(g_tone * 14.0, 0.0, 13.999)
g_lo = np.floor(g_idx).astype(np.int32)
g_hi = np.minimum(g_lo + 1, 14)
g_frac = (g_idx - g_lo)[:, :, np.newaxis]
tundra_col = (1.0 - g_frac) * greens_15[g_lo] + g_frac * greens_15[g_hi]

# Composite diffuse albedo
diffuse = rock_col.copy()
for c in range(3):
    diffuse[:, :, c] = (1.0 - tundra_mask) * diffuse[:, :, c] + tundra_mask * tundra_col[:, :, c]
    diffuse[:, :, c] = (1.0 - snow_mask) * diffuse[:, :, c] + snow_mask * snow_col[:, :, c]

# Hex flat skirt border (Tone 22 grey)
road_border = smoothstep(0.0, 0.010, rel_h) * smoothstep(0.3, 1.8, dist_edge)
road_grey = greys_45[22]
for c in range(3):
    diffuse[:, :, c] = (1.0 - road_border) * road_grey[c] + road_border * diffuse[:, :, c]

# Save preview images
diff_masked = diffuse.copy()
diff_masked[dist_edge < 0] = [0.07, 0.09, 0.13]
Image.fromarray((np.clip(diff_masked, 0, 1) * 255).astype(np.uint8)).save("scratch/test_diffuse_v10.png")

# Sky Dome + Studio directional rig for preview render
L1 = np.array([0.5, -0.5, 0.7], dtype=np.float32); L1 /= np.linalg.norm(L1)
d1 = np.clip(Nx * L1[0] + Ny * L1[1] + Nz * L1[2], 0.0, 1.0)
L2 = np.array([-0.5, 0.4, 0.7], dtype=np.float32); L2 /= np.linalg.norm(L2)
d2 = np.clip(Nx * L2[0] + Ny * L2[1] + Nz * L2[2], 0.0, 1.0)

d_dome = np.clip(0.65 + 0.22 * d1 + 0.13 * d2, 0.0, 1.0)
lit = np.clip(diffuse * d_dome[:, :, None], 0.0, 1.0)

# Save Studio Isometric
fig = plt.figure(figsize=(13, 11), facecolor='#090d16')
ax = fig.add_subplot(111, projection='3d', facecolor='#090d16')
step = 4
surf_x = X[::step, ::step]; surf_y = Y[::step, ::step]; surf_z = Z_med[::step, ::step]
rgba_sub = np.zeros((surf_x.shape[0], surf_x.shape[1], 4))
rgba_sub[:, :, :3] = lit[::step, ::step]
rgba_sub[:, :, 3] = 1.0
rgba_sub[dist_edge[::step, ::step] < -0.2, 3] = 0.0

ax.plot_surface(surf_x, surf_y, surf_z, facecolors=rgba_sub, rstride=1, cstride=1, shade=False, antialiased=True)
ax.view_init(elev=34, azim=-52)
ax.set_box_aspect([1.0, 1.15, 0.72])
ax.set_title("Loseta Montana Catan - Iconic 5K Glaciated Peak (V10)", fontsize=14, fontweight='bold', color='#38bdf8', pad=14)
ax.axis('off')
plt.tight_layout()
plt.savefig("scratch/test_studio_v10_isometric.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

# Save Studio Ortho (Top View)
fig, ax = plt.subplots(figsize=(8, 8), facecolor='#090d16')
ortho_lit = diffuse * (0.65 + 0.35 * np.clip(Nz[:, :, None], 0, 1))
ortho_lit[dist_edge < 0] = [0.06, 0.08, 0.12]
ax.imshow(np.clip(ortho_lit, 0, 1), origin='lower')
ax.axis('off')
plt.tight_layout()
plt.savefig("scratch/test_studio_v10_ortho.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

snow_pct = (snow_mask[dist_edge > 0] > 0.5).mean() * 100
tundra_pct = (tundra_mask[dist_edge > 0] > 0.5).mean() * 100
rock_pct = 100.0 - snow_pct - tundra_pct
print(f"V10 Done! Snow: {snow_pct:.1f}%, Tundra: {tundra_pct:.1f}%, Rock: {rock_pct:.1f}%")
