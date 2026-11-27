import os
import math
import time
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

print("=== EXPERIMENTING WITH TRUE 5K GLACIATED PEAK TOPOGRAPHY & TEXTURE ===")

dim_tex = 2048 # fast iteration
R = 64.0
APOTHEM = R * math.cos(math.radians(30))
Z_BASE = 25.0
Z_PEAK = 83.0

xs = np.linspace(-APOTHEM, APOTHEM, dim_tex, dtype=np.float32)
ys = np.linspace(-R, R, dim_tex, dtype=np.float32)
dx = float(xs[1] - xs[0])
dy = float(ys[1] - ys[0])

Z_master = np.load('scratch/Z_top_master_2m.npy')
import scipy.interpolate as interp
ny_m, nx_m = Z_master.shape
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

X, Y = np.meshgrid(xs, ys)
pts = np.column_stack((Y.ravel(), X.ravel()))
Z = interp_z(pts).reshape((dim_tex, dim_tex))
RELIEF = np.maximum(0.0, Z - Z_BASE)
total_relief = Z_PEAK - Z_BASE # 58.0 mm
rel_h = RELIEF / total_relief # 0.0 to 1.0

# Derivatives
gy, gx = np.gradient(Z, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm
Ny = -gy / N_norm
Nz = 1.0 / N_norm
slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

# Curvature
laplace = ndi.laplace(Z)
# Smooth laplace slightly to avoid pixel noise
laplace_smooth = ndi.gaussian_filter(laplace, 2.0)

# Distance to hexagon boundary
normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

greys_45 = np.load('scratch/greys_45.npy')
greens_15 = np.load('scratch/greens_15.npy')

# 1. Glacial Snow Model
# Snow zone:
# Above rel_h = 0.65 (Z >= 62.7 mm): Summit snowpack (85-98% coverage)
# Between rel_h = 0.25 and 0.65 (Z = 39.5 to 62.7 mm): Main glacial firn zone
# Glacial tongues flow down concave valleys (laplace_smooth < -0.05) down to rel_h = 0.12 (Z = 32 mm)

# Elevation snow potential
snow_potential = smoothstep(0.20, 0.55, rel_h)

# Valleys retain snow much lower (glacial tongues)
glacial_tongue = smoothstep(0.0, -0.30, laplace_smooth) * smoothstep(0.10, 0.35, rel_h)
snow_potential = np.clip(snow_potential + 0.60 * glacial_tongue, 0.0, 1.0)

# High summit mantle
summit_mantle = smoothstep(0.55, 0.85, rel_h)
snow_potential = np.maximum(snow_potential, summit_mantle)

# Rock exposure rules:
# Rock ONLY appears on:
# 1. Sheer vertical rock cliffs: slope > 64 deg on upper mountain, slope > 52 deg on lower mountain
cliff_thresh = 52.0 + 16.0 * rel_h # from 52 deg at base up to 68 deg near summit!
is_cliff = smoothstep(cliff_thresh - 6.0, cliff_thresh + 6.0, slope_deg)

# 2. Razor-sharp convex arêtes (laplace_smooth > 0.08)
is_arete = smoothstep(0.04, 0.16, laplace_smooth) * (1.0 - summit_mantle * 0.5)

# Snow retention
snow_retention = (1.0 - 0.90 * is_cliff) * (1.0 - 0.75 * is_arete)

# Smooth snow coverage mask
snow_mask = np.clip(snow_potential * snow_retention, 0.0, 1.0)
# Make transitions crisp and natural like real snowpacks
snow_mask = smoothstep(0.25, 0.75, snow_mask)

# 2. Snow Color
col_sun = greys_45[0]      # Arctic White
col_shade = greys_45[3]    # Pale alpine white-blue
col_ice = np.array([0.76, 0.88, 0.96], dtype=np.float32) # Glacial blue in couloirs

az_sun = 310.0 * math.pi / 180.0
alt_sun = 46.0 * math.pi / 180.0
sun_dir = np.array([math.cos(alt_sun) * math.sin(az_sun), math.cos(alt_sun) * math.cos(az_sun), math.sin(alt_sun)], dtype=np.float32)
sun_dot = np.clip(Nx * sun_dir[0] + Ny * sun_dir[1] + Nz * sun_dir[2], 0.0, 1.0)

ice_presence = smoothstep(0.0, -0.25, laplace_smooth) * smoothstep(0.2, 0.6, rel_h)

snow_col = np.zeros((dim_tex, dim_tex, 3), dtype=np.float32)
for c in range(3):
    base_s = (1.0 - 0.20 * (1.0 - sun_dot)) * col_sun[c] + 0.20 * (1.0 - sun_dot) * col_shade[c]
    snow_col[:, :, c] = (1.0 - 0.25 * ice_presence) * base_s + 0.25 * ice_presence * col_ice[c]

# 3. Alpine Rock Color (45 Greys)
# We want rich alpine granite tones, NOT black coal!
# Arêtes and high sunlit rock: Tones 12-18 (medium-light granite)
# Main faces: Tones 20-28 (cold mountain granite)
# Deep cracks / shadows: Tones 32-38
rock_shade = 0.35 + 0.35 * smoothstep(40.0, 75.0, slope_deg) - 0.25 * sun_dot + 0.20 * smoothstep(0.05, -0.25, laplace_smooth)
rock_shade = np.clip(rock_shade, 0.0, 1.0)
rock_grey_idx = 14.0 + rock_shade * 22.0 # tone 14 to tone 36!
rock_grey_idx = np.clip(rock_grey_idx, 0.0, 43.999)

r_lo = np.floor(rock_grey_idx).astype(np.int32)
r_hi = np.minimum(r_lo + 1, 44)
r_frac = (rock_grey_idx - r_lo)[:, :, np.newaxis]
rock_col = (1.0 - r_frac) * greys_45[r_lo] + r_frac * greys_45[r_hi]

# 4. Alpine Skirt Tundra (15 Greens)
tundra_pot = smoothstep(0.005, 0.02, rel_h) * (1.0 - smoothstep(0.05, 0.12, rel_h))
tundra_slope = 1.0 - smoothstep(10.0, 28.0, slope_deg)
tundra_mask = np.clip(tundra_pot * tundra_slope * (1.0 - snow_mask), 0.0, 1.0)
tundra_mask = smoothstep(0.15, 0.60, tundra_mask)

green_t = np.clip(0.35 + 0.40 * smoothstep(-0.2, 0.1, -laplace_smooth) - 0.25 * sun_dot, 0.0, 1.0)
g_idx = np.clip(green_t * 14.0, 0.0, 13.999)
g_lo = np.floor(g_idx).astype(np.int32)
g_hi = np.minimum(g_lo + 1, 14)
g_frac = (g_idx - g_lo)[:, :, np.newaxis]
tundra_col = (1.0 - g_frac) * greens_15[g_lo] + g_frac * greens_15[g_hi]

# 5. Composite
diffuse = rock_col.copy()
for c in range(3):
    diffuse[:, :, c] = (1.0 - tundra_mask) * diffuse[:, :, c] + tundra_mask * tundra_col[:, :, c]
    diffuse[:, :, c] = (1.0 - snow_mask) * diffuse[:, :, c] + snow_mask * snow_col[:, :, c]

# Base border
road_border = smoothstep(0.0, 0.015, rel_h)
road_grey = greys_45[20]
for c in range(3):
    diffuse[:, :, c] = (1.0 - road_border) * road_grey[c] + road_border * diffuse[:, :, c]

diffuse_out = np.clip(diffuse * 255.0, 0.0, 255.0).astype(np.uint8)
img = Image.fromarray(diffuse_out)
img.save("scratch/test_majestic_diffuse.png")
print("Saved scratch/test_majestic_diffuse.png!")
print("Snow coverage (>0.7):", (diffuse > 0.7).mean())
