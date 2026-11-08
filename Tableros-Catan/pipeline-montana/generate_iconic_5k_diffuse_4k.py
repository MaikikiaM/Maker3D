import os
import math
import time
import numpy as np
import scipy.ndimage as ndi
import scipy.interpolate as interp
from PIL import Image

print("=================================================================")
print("  BAKING 4K DIFFUSE ALBEDO MAP: ICONIC 5K GLACIATED MASSIF       ")
print("=================================================================")
t0 = time.time()

out_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D"
os.makedirs(out_dir, exist_ok=True)

# 1. Load DEM and Raymarched AO
Z_master = np.load('scratch/Z_top_master_2m.npy')
ao_master = np.load('scratch/ao_raymarched_16dir.npy')

dim = 4096
R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256258 mm
Z_BASE = 25.0
Z_PEAK = 83.0
total_relief = Z_PEAK - Z_BASE # 58.00 mm

xs = np.linspace(-APOTHEM, APOTHEM, dim, dtype=np.float32)
ys = np.linspace(-R, R, dim, dtype=np.float32)
dx = float(xs[1] - xs[0])
dy = float(ys[1] - ys[0])

ny_m, nx_m = Z_master.shape
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)

print(f"Interpolating Master DEM ({nx_m}x{ny_m}) to 4K ({dim}x{dim})...")
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)
interp_ao = interp.RegularGridInterpolator((ys_m, xs_m), ao_master, bounds_error=False, fill_value=0.8)

X, Y = np.meshgrid(xs, ys)
pts = np.column_stack((Y.ravel(), X.ravel()))
Z = interp_z(pts).reshape((dim, dim))
AO = interp_ao(pts).reshape((dim, dim))

print("Computing multi-scale differential geomorphometry...")
# Macro geological structure (sigma ~ 2.2mm)
sigma_macro = 2.2 / dx
Z_macro = ndi.gaussian_filter(Z, sigma_macro)

# Intermediate crags (sigma ~ 0.6mm)
sigma_med = 0.6 / dx
Z_med = ndi.gaussian_filter(Z, sigma_med)

RELIEF = np.maximum(0.0, Z - Z_BASE)
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

print("Evaluating Calibrated 22.7% Snowline Alpine Distribution...")
# ------------------------------------------------------------------
# EXACT COTA DE NIEVE (HACE 2 VERSIONES):
# Z starts at 44.4 mm (Relief 19.4 mm)
# Solid snow on summits Z >= 58 mm (Relief >= 33 mm)
# Couloirs flow down to Z ~ 42 mm (Relief ~ 17 mm)
# Below Z = 41 mm (Relief 16 mm), snow is strictly 0.0!
# Total snow coverage: ~22.7%
# ------------------------------------------------------------------
couloir_concavity = smoothstep(0.008, 0.07, e_max)
arete_convexity = smoothstep(-0.008, -0.08, e_min)
cliff_face = smoothstep(48.0, 64.0, slope_deg)

# Snow starts at relief 18.0 (Z = 43.0 mm) and fully blankets by relief 33.0 (Z = 58.0 mm)
snow_elevation = smoothstep(18.0, 33.0, RELIEF)
glacial_tongues = couloir_concavity * smoothstep(15.0, 28.0, RELIEF)
upper_snowpack = smoothstep(26.0, 42.0, RELIEF)

snow_pot = np.clip(snow_elevation * 0.90 + glacial_tongues * 0.75 + upper_snowpack * 0.30, 0.0, 1.0)

rock_spine = np.clip(0.65 * arete_convexity + 0.60 * cliff_face, 0.0, 1.0)
rock_spine *= (1.0 - 0.55 * smoothstep(40.0, 54.0, RELIEF))

snow_mask = np.clip(snow_pot - 0.55 * rock_spine, 0.0, 1.0)
snow_mask = smoothstep(0.15, 0.52, snow_mask)

# Strict hard floor: 0% snow below Z = 41.0 mm (RELIEF = 16.0 mm)
snow_mask *= smoothstep(15.0, 19.0, RELIEF)

# Summit dome
summit_pure = smoothstep(44.0, 56.0, RELIEF) * (1.0 - smoothstep(64.0, 75.0, slope_deg))
snow_mask = np.clip(snow_mask + 0.40 * summit_pure, 0.0, 1.0)

# Tundra
tundra_pot = smoothstep(0.001, 0.015, rel_h) * (1.0 - smoothstep(0.08, 0.18, rel_h))
tundra_slope = 1.0 - smoothstep(10.0, 36.0, slope_deg)
tundra_mask = np.clip(tundra_pot * tundra_slope * (1.0 - snow_mask * 0.95), 0.0, 1.0)
tundra_mask = smoothstep(0.06, 0.38, tundra_mask)

# ------------------------------------------------------------------
# 4. COLOR PALETTES & TEXTURING
# ------------------------------------------------------------------
print("Synthesizing albedo diffuse textures...")
# Snow Color:
# - Crystalline alpine white
# - Subtle glacial ice-blue tint in deep cirques
col_snow_pure = np.array([0.985, 0.988, 1.000], dtype=np.float32)
col_glacier_cyan = np.array([0.76, 0.89, 0.97], dtype=np.float32)

ice_tint = couloir_concavity * smoothstep(18.0, 42.0, RELIEF)
snow_depth_ao = 0.92 + 0.08 * smoothstep(0.3, 0.85, AO)

snow_col = np.zeros((dim, dim, 3), dtype=np.float32)
for c in range(3):
    base_s = (1.0 - 0.20 * ice_tint) * col_snow_pure[c] + (0.20 * ice_tint) * col_glacier_cyan[c]
    snow_col[:, :, c] = base_s * snow_depth_ao

# Bedrock & Scree (45 Calibrated Greys):
# Bold dark granite ribs contrasting with lighter talus scree
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

# Hex flat skirt border (Tone 22 grey: #808080)
road_border = smoothstep(0.0, 0.008, rel_h) * smoothstep(0.3, 1.8, dist_edge)
road_grey = greys_45[22]
for c in range(3):
    diffuse[:, :, c] = (1.0 - road_border) * road_grey[c] + road_border * diffuse[:, :, c]

# Foundation background outside hexagon
diffuse_clean = np.clip(diffuse, 0.0, 1.0)
diffuse_clean[dist_edge < 0] = [0.12, 0.13, 0.15]

# Save numpy array for mesh coloring pipeline
np.save('scratch/diffuse_rgb_4k.npy', diffuse_clean)
print("Saved scratch/diffuse_rgb_4k.npy")

# Save 4K PNG
diff_u8 = (diffuse_clean * 255.0).astype(np.uint8)
img_4k = Image.fromarray(diff_u8)
path_4k = os.path.join(out_dir, "catan_mountain_diffuse_4k.png")
img_4k.save(path_4k, optimize=True)
print(f"Saved 4K Diffuse Map: {path_4k} ({os.path.getsize(path_4k)/1024/1024:.2f} MB)")

# Downscale to 2K bicubic for WebGL viewer performance
print("Generating 2K Diffuse Map for WebGL viewer...")
img_2k = img_4k.resize((2048, 2048), Image.Resampling.LANCZOS)
path_2k = os.path.join(out_dir, "catan_mountain_diffuse_2k.png")
img_2k.save(path_2k, optimize=True)
print(f"Saved 2K Diffuse Map: {path_2k} ({os.path.getsize(path_2k)/1024/1024:.2f} MB)")

snow_pct = (snow_mask[dist_edge > 0] > 0.5).mean() * 100
tundra_pct = (tundra_mask[dist_edge > 0] > 0.5).mean() * 100
rock_pct = 100.0 - snow_pct - tundra_pct
print(f"Summary: Snow: {snow_pct:.1f}%, Tundra: {tundra_pct:.1f}%, Rock: {rock_pct:.1f}%")
print(f"Done in {time.time()-t0:.2f}s")
