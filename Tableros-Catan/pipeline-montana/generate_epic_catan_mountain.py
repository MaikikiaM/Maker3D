import math
import numpy as np
import scipy.ndimage as ndi
import trimesh
import os

print("=== STARTING EPIC CATAN MOUNTAIN GENERATION (CMYKWOG-B) ===")

# 1. Load high-res DEMs
m_z15 = np.load('scratch/matterhorn_z15_768.npy')
t_z15 = np.load('scratch/teton_z15_768.npy')

# Hexagon specifications
R = 64.0 # Vertex radius
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF_H = Z_PEAK - Z_BASE # 58.0 mm

# High-resolution grid
nx, ny = 351, 405
xs = np.linspace(-APOTHEM, APOTHEM, nx)
ys = np.linspace(-R, R, ny)
X, Y = np.meshgrid(xs, ys)

# Hexagon edge distances and vertex distances
normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    d = APOTHEM - (X * nx_h + Y * ny_h)
    dist_edge = np.minimum(dist_edge, d)

v_angles = [(30 + 60 * k) * math.pi / 180 for k in range(6)]
verts_hex = [[R * math.cos(a), R * math.sin(a)] for a in v_angles]
dist_vert = np.full_like(X, 999.0)
for vx, vy in verts_hex:
    dv = np.hypot(X - vx, Y - vy)
    dist_vert = np.minimum(dist_vert, dv)

# 2. Resample and combine LiDAR DEMs
from scipy.ndimage import zoom

# Grand Teton Cathedral Massif (452, 388)
y0_t, x0_t = 452, 388
w_t = 160
crop_t = t_z15[y0_t - w_t : y0_t + w_t, x0_t - w_t : x0_t + w_t]
res_t = zoom(crop_t, (ny / (2 * w_t), nx / (2 * w_t)), order=3)
norm_t = (res_t - res_t.min()) / (res_t.max() - res_t.min())

# Matterhorn Pyramid Horn (426, 245)
y0_m, x0_m = 426, 245
w_m = 110
crop_m = m_z15[y0_m - w_m : y0_m + w_m, x0_m - w_m : x0_m + w_m]
res_m = zoom(crop_m, (ny / (2 * w_m), nx / (2 * w_m)), order=3)
norm_m = (res_m - res_m.min()) / (res_m.max() - res_m.min())

# Blending: Matterhorn central horn + Grand Teton Cathedral ridges
r_dist = np.hypot(X, Y)
horn_weight = np.clip(1.0 - (r_dist / 34.0)**2.0, 0.0, 1.0) * 0.45
composite = norm_t * (1.0 - horn_weight) + np.maximum(norm_t, norm_m) * horn_weight
composite = (composite - composite.min()) / (composite.max() - composite.min())

# 3. Alpine ridge sharpening & steepness
blur = ndi.gaussian_filter(composite, sigma=1.2)
composite_sharp = np.clip(composite + 1.3 * (composite - blur), 0.0, 1.0)
composite_sharp = np.power(composite_sharp, 1.22)
composite_sharp = composite_sharp / composite_sharp.max()

# 4. Smoothstep perimeter clearances
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# Road clearance: 3.5mm shelf at Z=25mm
# Vertex clearance: 10mm radius (20mm diam) at Z=25mm
edge_factor = smoothstep(3.5, 12.0, dist_edge)
vert_factor = smoothstep(10.0, 16.0, dist_vert)
clearance = edge_factor * vert_factor

# Top relief height above Z=25mm
relief = composite_sharp * clearance * RELIEF_H
Z_top = Z_BASE + relief
# Outside hexagon, clamp to Z_BASE
Z_top[dist_edge < 0] = Z_BASE

print(f"Topography generated: Z min = {Z_top.min():.2f} mm, Z max = {Z_top.max():.2f} mm")

# 5. Slope, Aspect, and Geological Feature Extraction
dx = xs[1] - xs[0]
dy = ys[1] - ys[0]
gy, gx = np.gradient(Z_top, dy, dx)
slope = np.hypot(gx, gy) # tangent of slope angle
slope_deg = np.degrees(np.arctan(slope))
laplace = ndi.laplace(Z_top)

# 6. Dendritic Mineral Vein Synthesis (Hematite Iron Ore / Chalcopyrite Copper)
# Natural fractures follow shear stress lines across the rock face
fault1 = np.abs(Y - (0.65 * X - 6.0) + 3.5 * np.sin(X * 0.14))
fault2 = np.abs(Y - (-0.45 * X + 10.0) + 3.0 * np.cos(Y * 0.16))
fault3 = np.abs(X - (0.3 * Y - 12.0) + 2.5 * np.sin(Y * 0.12))

np.random.seed(777)
fracture_noise = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=2.0) * 2.5
v1 = (fault1 + fracture_noise) < 1.8
v2 = (fault2 + fracture_noise) < 1.6
v3 = (fault3 + fracture_noise) < 1.4

vein_network = (v1 | v2 | v3) & (relief > 4.0) & (relief < 48.0) & (slope > 0.35)
vein_mask = ndi.binary_dilation(vein_network, iterations=1)

# 7. Snow Accumulation & Ice Field Simulation
# Snow settles on high elevations where slope < 45 degrees, plus couloirs
snow_elev_factor = smoothstep(30.0, 50.0, relief) # snow above 55-75mm total Z
snow_slope_factor = 1.0 - smoothstep(35.0, 52.0, slope_deg) # sheds snow on cliffs > 50 deg
# Couloir retention (concave areas retain snow)
couloir_factor = np.clip(-laplace * 0.5, 0.0, 1.0)
snow_prob = snow_elev_factor * (0.75 * snow_slope_factor + 0.25 * couloir_factor)
snow_mask = (snow_prob > 0.42) & (relief > 28.0) & (~vein_mask)

# Glacial ice: high altitude hollows/couloirs
ice_mask = (relief > 36.0) & (couloir_factor > 0.3) & (~snow_mask) & (~vein_mask)

# Alpine tundra / moss: lower perimeter depressions (Z 25.0 to 29.0mm)
tundra_mask = (relief > 0.1) & (relief < 6.5) & (dist_edge > 2.0) & (~vein_mask)

# Granite cliff / slate crevice: steep shaded slopes
cliff_mask = (slope_deg > 48.0) & (relief > 5.0) & (~snow_mask) & (~vein_mask)

print("Geological layers computed:")
print(f"  Vein pixels: {np.sum(vein_mask)}")
print(f"  Snow pixels: {np.sum(snow_mask)}")
print(f"  Ice pixels:  {np.sum(ice_mask)}")
print(f"  Tundra:      {np.sum(tundra_mask)}")
print(f"  Cliffs:      {np.sum(cliff_mask)}")

np.save('scratch/Z_top_master.npy', Z_top)
np.save('scratch/vein_mask.npy', vein_mask)
np.save('scratch/snow_mask.npy', snow_mask)
np.save('scratch/tundra_mask.npy', tundra_mask)
np.save('scratch/ice_mask.npy', ice_mask)
print("Saved master arrays to scratch/")
