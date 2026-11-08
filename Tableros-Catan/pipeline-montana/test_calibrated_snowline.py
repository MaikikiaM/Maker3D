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
Z_macro = ndi.gaussian_filter(Z, 2.2 / dx)  # Geological macro form (~2.2mm)
Z_med = ndi.gaussian_filter(Z, 0.6 / dx)    # Intermediate crags (~0.6mm)

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
e_min = 0.5 * (tr - np.sqrt(disc))  # < 0: Convex ridges & peaks
e_max = 0.5 * (tr + np.sqrt(disc))  # > 0: Concave cirques & valleys

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
# EXACT SNOWLINE CALIBRATION REQUESTED BY USER:
# "la cota de nieve de hace 2 versiones era la ideal"
# Snow starts at Z ~ 43-46 mm (RELIEF ~ 18-21 mm)
# Full snowpack at Z >= 60 mm (RELIEF >= 35 mm)
# Couloirs descend down to Z ~ 41 mm (RELIEF ~ 16 mm)
# ZERO SNOW below Z = 39 mm!
# ------------------------------------------------------------------
# Couloirs & cirques (concave flow channels)
couloir_concavity = smoothstep(0.010, 0.08, e_max)

# Macro arêtes & spines (convex rock ribs)
arete_convexity = smoothstep(-0.010, -0.09, e_min)

# Cliff walls (steep rock shedding snow)
cliff_face = smoothstep(46.0, 64.0, slope_deg)

# The Ideal Snowline:
# Relief in mm:
# relief 18.0 mm = Z = 43.0 mm
# relief 36.0 mm = Z = 61.0 mm
snow_elevation = smoothstep(18.0, 36.0, RELIEF)

# Couloir tongues flowing down into mid-mountain (relief 15 to 30 mm, Z = 40 to 55 mm)
glacial_tongues = couloir_concavity * smoothstep(14.0, 30.0, RELIEF)

# Upper Massif snow blanket (relief > 32 mm, Z > 57 mm)
upper_snowpack = smoothstep(30.0, 48.0, RELIEF)

# Combined snow potential:
snow_pot = np.clip(snow_elevation * 0.85 + glacial_tongues * 0.70 + upper_snowpack * 0.35, 0.0, 1.0)

# Rock ribs breakthrough (arêtes and sheer cliffs):
rock_spine = np.clip(0.70 * arete_convexity + 0.65 * cliff_face, 0.0, 1.0)
rock_spine *= (1.0 - 0.50 * smoothstep(44.0, 56.0, RELIEF)) # summit cap retains snow

# Snow mask:
snow_mask = np.clip(snow_pot - 0.65 * rock_spine, 0.0, 1.0)
snow_mask = smoothstep(0.18, 0.60, snow_mask)

# Absolute cutoff: ZERO snow below Z = 40.0 mm (RELIEF = 15.0 mm)
snow_mask *= smoothstep(14.0, 20.0, RELIEF)

# Resplendent summit pure snow
summit_pure = smoothstep(46.0, 57.0, RELIEF) * (1.0 - smoothstep(64.0, 75.0, slope_deg))
snow_mask = np.clip(snow_mask + 0.45 * summit_pure, 0.0, 1.0)

# ------------------------------------------------------------------
# ALPINE PASTURES (TUNDRA) & TALUS SCREE
# ------------------------------------------------------------------
# Meadows wrap the entire lower base (Z in [25.0, 35.0] mm, slopes < 36 deg)
tundra_pot = smoothstep(0.001, 0.015, rel_h) * (1.0 - smoothstep(0.08, 0.18, rel_h))
tundra_slope = 1.0 - smoothstep(10.0, 36.0, slope_deg)
tundra_mask = np.clip(tundra_pot * tundra_slope * (1.0 - snow_mask * 0.95), 0.0, 1.0)
tundra_mask = smoothstep(0.06, 0.38, tundra_mask)

# ------------------------------------------------------------------
# COLOR PALETTES & TEXTURING (45 Greys + 15 Greens)
# ------------------------------------------------------------------
col_snow_pure = np.array([0.985, 0.988, 1.000], dtype=np.float32)
col_glacier_cyan = np.array([0.76, 0.89, 0.97], dtype=np.float32)

ice_tint = couloir_concavity * smoothstep(18.0, 42.0, RELIEF)
snow_depth_ao = 0.92 + 0.08 * smoothstep(0.3, 0.85, AO)

snow_col = np.zeros((dim, dim, 3), dtype=np.float32)
for c in range(3):
    base_s = (1.0 - 0.20 * ice_tint) * col_snow_pure[c] + (0.20 * ice_tint) * col_glacier_cyan[c]
    snow_col[:, :, c] = base_s * snow_depth_ao

# Bedrock & Scree (45 Calibrated Greys):
rock_tone_val = 14.0 + 16.0 * smoothstep(25.0, 65.0, slope_deg) + 12.0 * (1.0 - smoothstep(0.3, 0.85, AO))
rock_tone_val = np.clip(rock_tone_val, 8.0, 42.999)

r_lo = np.floor(rock_tone_val).astype(np.int32)
r_hi = np.minimum(r_lo + 1, 44)
r_frac = (rock_tone_val - r_lo)[:, :, np.newaxis]
rock_col = (1.0 - r_frac) * greys_45[r_lo] + r_frac * greys_45[r_hi]

# Botanical Greens (15 Greens):
g_tone = np.clip(0.20 + 0.55 * smoothstep(0.01, 0.12, rel_h) + 0.25 * (1.0 - smoothstep(0.4, 0.8, AO)), 0.0, 1.0)
g_idx = np.clip(g_tone * 14.0, 0.0, 13.999)
g_lo = np.floor(g_idx).astype(np.int32)
g_hi = np.minimum(g_lo + 1, 14)
g_frac = (g_idx - g_lo)[:, :, np.newaxis]
tundra_col = (1.0 - g_frac) * greens_15[g_lo] + g_frac * greens_15[g_hi]

# Composite Diffuse Albedo
diffuse = rock_col.copy()
for c in range(3):
    diffuse[:, :, c] = (1.0 - tundra_mask) * diffuse[:, :, c] + tundra_mask * tundra_col[:, :, c]
    diffuse[:, :, c] = (1.0 - snow_mask) * diffuse[:, :, c] + snow_mask * snow_col[:, :, c]

# Hex flat skirt border (Tone 22 grey)
road_border = smoothstep(0.0, 0.008, rel_h) * smoothstep(0.3, 1.8, dist_edge)
road_grey = greys_45[22]
for c in range(3):
    diffuse[:, :, c] = (1.0 - road_border) * road_grey[c] + road_border * diffuse[:, :, c]

# Preview lighting (balanced studio dome)
L1 = np.array([0.5, -0.5, 0.7], dtype=np.float32); L1 /= np.linalg.norm(L1)
d1 = np.clip(Nx * L1[0] + Ny * L1[1] + Nz * L1[2], 0.0, 1.0)
L2 = np.array([-0.5, 0.4, 0.7], dtype=np.float32); L2 /= np.linalg.norm(L2)
d2 = np.clip(Nx * L2[0] + Ny * L2[1] + Nz * L2[2], 0.0, 1.0)

d_dome = np.clip(0.66 + 0.22 * d1 + 0.12 * d2, 0.0, 1.0)
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
ax.set_title("Loseta Montana Catan - Cota de Nieve Calibrada (Ideal)", fontsize=14, fontweight='bold', color='#38bdf8', pad=14)
ax.axis('off')
plt.tight_layout()
plt.savefig("scratch/test_calibrated_snow_isometric.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

# Save Studio Ortho (Top View)
fig, ax = plt.subplots(figsize=(8, 8), facecolor='#090d16')
ortho_lit = diffuse * (0.75 + 0.25 * np.clip(Nz[:, :, None], 0, 1))
ortho_lit[dist_edge < 0] = [0.06, 0.08, 0.12]
ax.imshow(np.clip(ortho_lit, 0, 1), origin='lower')
ax.axis('off')
plt.tight_layout()
plt.savefig("scratch/test_calibrated_snow_ortho.png", dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

mask_hex = dist_edge > 0
snow_pct = (snow_mask[mask_hex] > 0.4).mean() * 100
tundra_pct = (tundra_mask[mask_hex] > 0.4).mean() * 100
rock_pct = 100.0 - snow_pct - tundra_pct
min_z_snow = Z[mask_hex & (snow_mask > 0.1)].min()
print(f"Calibrated Snow Done! Snow: {snow_pct:.1f}%, Tundra: {tundra_pct:.1f}%, Rock: {rock_pct:.1f}%")
print(f"Cota minima de nieve: Z = {min_z_snow:.1f} mm (Relieve: {min_z_snow - 25.0:.1f} mm)")
