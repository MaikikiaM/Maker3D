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

# Multi-scale smoothing
Z_macro = ndi.gaussian_filter(Z, 2.5 / dx)  # Geological macro form (~2.5mm)
Z_med = ndi.gaussian_filter(Z, 0.8 / dx)    # Intermediate ridges (~0.8mm)

RELIEF = np.maximum(0.0, Z - Z_BASE)
total_relief = Z_PEAK - Z_BASE # 58.0 mm
rel_h = RELIEF / total_relief # 0.0 to 1.0

# Gradients & Slopes
gy, gx = np.gradient(Z_med, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm
Ny = -gy / N_norm
Nz = 1.0 / N_norm
slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

# Hessian on macro geometry
gy_m, gx_m = np.gradient(Z_macro, dy, dx)
gyy_m, gyx_m = np.gradient(gy_m, dy, dx)
gxy_m, gxx_m = np.gradient(gx_m, dy, dx)

tr = gxx_m + gyy_m
det = gxx_m * gyy_m - 0.5 * (gxy_m + gyx_m)**2
disc = np.maximum(0.0, tr**2 - 4.0 * det)
# Eigenvalues:
# e_min < 0: convexity (arêtes / ridges / horns)
# e_max > 0: concavity (cirques / couloirs / glacial troughs)
e_min = 0.5 * (tr - np.sqrt(disc))
e_max = 0.5 * (tr + np.sqrt(disc))

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
# 2. ICONIC 5K ALPINE GLACIER SYSTEM (Mont Blanc / Cotopaxi / Alpamayo)
# ------------------------------------------------------------------
# 1) Couloirs & Cirques (Glacial channels where e_max > 0):
couloir_concavity = smoothstep(0.015, 0.12, e_max)

# 2) Macro Arêtes & Spines (Convex rock ribs where e_min < 0):
arete_convexity = smoothstep(-0.015, -0.12, e_min)

# 3) Sheer vertical rock walls (cliffs > 58 deg shed snow completely):
sheer_cliff = smoothstep(52.0, 68.0, slope_deg)

# 4) Permanent Alpine Snowfield:
# Snow starts at elevation rel_h = 0.10 (Z = 31 mm) and becomes dominant above rel_h = 0.28 (Z = 41 mm)
snow_elevation = smoothstep(0.08, 0.32, rel_h)

# Glacial tongues extending down the couloirs almost to the base (rel_h = 0.04, Z = 27.5 mm)
glacial_tongues = couloir_concavity * smoothstep(0.03, 0.30, rel_h)

# Upper Massif Snow Blanket:
# The entire upper massif (rel_h > 0.50) is covered in snow, EXCEPT along sheer arêtes and cliffs
upper_snowpack = smoothstep(0.35, 0.70, rel_h)

# Total snow potential (massive coverage!)
total_snow_pot = np.clip(snow_elevation * 0.85 + glacial_tongues * 0.85 + upper_snowpack * 0.30, 0.0, 1.0)

# Rock breakthrough:
# Dark granite ribs cut through the snow along the knife edges and sheer walls
rock_spine = np.clip(0.65 * arete_convexity + 0.60 * sheer_cliff, 0.0, 1.0)
# Summit cap keeps pure snow over the top, softening rock ribs right at the peak
rock_spine *= (1.0 - 0.55 * smoothstep(0.75, 0.95, rel_h))

# Final Snow Mask:
snow_mask = np.clip(total_snow_pot - 0.70 * rock_spine, 0.0, 1.0)
snow_mask = smoothstep(0.12, 0.48, snow_mask)

# Ensure the summit dome and highest crags are resplendent white
summit_pure = smoothstep(0.82, 0.98, rel_h) * (1.0 - smoothstep(65.0, 75.0, slope_deg))
snow_mask = np.clip(snow_mask + 0.45 * summit_pure, 0.0, 1.0)

# ------------------------------------------------------------------
# 3. COLOR PALETTES & TEXTURING
# ------------------------------------------------------------------
# Snow Color:
# - Brilliant alpine snow (sunlit, clean, crystalline)
# - Subtle glacial aquamarine/ice-blue in deep cirques and crevices
col_snow_pure = np.array([0.985, 0.985, 1.000], dtype=np.float32)
col_glacier_cyan = np.array([0.76, 0.89, 0.97], dtype=np.float32)

ice_tint = couloir_concavity * smoothstep(0.15, 0.60, rel_h)
snow_depth_ao = 0.92 + 0.08 * smoothstep(0.3, 0.85, AO)

snow_col = np.zeros((dim, dim, 3), dtype=np.float32)
for c in range(3):
    base_s = (1.0 - 0.20 * ice_tint) * col_snow_pure[c] + (0.20 * ice_tint) * col_glacier_cyan[c]
    snow_col[:, :, c] = base_s * snow_depth_ao

# Bedrock Color (45 Calibrated Greys):
# Bold, rich dark granite arêtes.
# High slope / steep crevices get deep charcoal (tones 28-38), flat scree gets tone 16-22
rock_tone_val = 14.0 + 16.0 * smoothstep(25.0, 65.0, slope_deg) + 12.0 * (1.0 - smoothstep(0.3, 0.85, AO))
rock_tone_val = np.clip(rock_tone_val, 8.0, 42.999)

r_lo = np.floor(rock_tone_val).astype(np.int32)
r_hi = np.minimum(r_lo + 1, 44)
r_frac = (rock_tone_val - r_lo)[:, :, np.newaxis]
rock_col = (1.0 - r_frac) * greys_45[r_lo] + r_frac * greys_45[r_hi]

# Alpine Tundra / Pasture (15 Botanical Greens):
# Covers lower slopes below the snowline (rel_h in [0.002, 0.12], slopes < 36 deg)
tundra_pot = smoothstep(0.002, 0.015, rel_h) * (1.0 - smoothstep(0.06, 0.14, rel_h))
tundra_slope = 1.0 - smoothstep(12.0, 36.0, slope_deg)
tundra_mask = np.clip(tundra_pot * tundra_slope * (1.0 - snow_mask * 0.95), 0.0, 1.0)
tundra_mask = smoothstep(0.08, 0.40, tundra_mask)

# Botanical green indexing
g_tone = np.clip(0.25 + 0.50 * smoothstep(0.01, 0.10, rel_h) + 0.25 * (1.0 - smoothstep(0.4, 0.8, AO)), 0.0, 1.0)
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

# Save pure diffuse
diff_masked = diffuse.copy()
diff_masked[dist_edge < 0] = [0.07, 0.09, 0.13]
Image.fromarray((np.clip(diff_masked, 0, 1) * 255).astype(np.uint8)).save("scratch/test_diffuse_v11.png")

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
ax.set_title("Loseta Montana Catan - Iconic 5K Glaciated Peak (V11)", fontsize=14, fontweight='bold', color='#38bdf8', pad=14)
ax.axis('off')
plt.tight_layout()
plt.savefig("scratch/test_studio_v11_isometric.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

# Save Studio Ortho (Top View)
fig, ax = plt.subplots(figsize=(8, 8), facecolor='#090d16')
ortho_lit = diffuse * (0.65 + 0.35 * np.clip(Nz[:, :, None], 0, 1))
ortho_lit[dist_edge < 0] = [0.06, 0.08, 0.12]
ax.imshow(np.clip(ortho_lit, 0, 1), origin='lower')
ax.axis('off')
plt.tight_layout()
plt.savefig("scratch/test_studio_v11_ortho.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

snow_pct = (snow_mask[dist_edge > 0] > 0.5).mean() * 100
tundra_pct = (tundra_mask[dist_edge > 0] > 0.5).mean() * 100
rock_pct = 100.0 - snow_pct - tundra_pct
print(f"V11 Done! Snow: {snow_pct:.1f}%, Tundra: {tundra_pct:.1f}%, Rock: {rock_pct:.1f}%")
