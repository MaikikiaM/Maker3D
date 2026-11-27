import math
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

print("=== PHASE 1: TESTING CONTINUOUS INFINITE RGB COLOR FIELD SYNTHESIS ===")

# High resolution grid for texture and vertex color evaluation
nx, ny = 1024, 1024
R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
xs = np.linspace(-APOTHEM, APOTHEM, nx)
ys = np.linspace(-R, R, ny)
X, Y = np.meshgrid(xs, ys)

normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

# Load high-res Z top
Z_master = np.load('scratch/Z_top_master_final.npy')
ny_m, nx_m = Z_master.shape
import scipy.interpolate as interp
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m)
ys_m = np.linspace(-R, R, ny_m)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

# Resample Z on 1024x1024
pts = np.column_stack((Y.ravel(), X.ravel()))
Z = interp_z(pts).reshape((ny, nx))
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF = np.maximum(0.0, Z - Z_BASE)

# Gradient, Slope, and Curvature
dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
gy, gx = np.gradient(Z, dy, dx)
slope = np.hypot(gx, gy)
slope_deg = np.degrees(np.arctan(slope))
laplace = ndi.laplace(Z)

print(f"Resampled Z grid: {ny}x{nx}, Relief in [{np.min(RELIEF):.2f}, {np.max(RELIEF):.2f}] mm")

# -------------------------------------------------------------
# 1. CONTINUOUS BASE ROCK COLOR FIELD (NO DISCRETE STEPS)
# -------------------------------------------------------------
# Macro variation: elevation + slope + aspect
np.random.seed(42)
rock_f1 = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=28.0)
rock_f2 = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=12.0)
rock_f3 = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=4.0)

# Natural fractal rock noise
rock_noise = 0.55 * rock_f1 + 0.30 * rock_f2 + 0.15 * rock_f3
rock_noise = (rock_noise - np.mean(rock_noise)) / np.std(rock_noise)

# Base Rock Palette (Smooth RGB Interpolation):
# Dark Slate / Basalt: [0.18, 0.20, 0.23]
# Deep Shadow Crevice: [0.12, 0.13, 0.15]
# Neutral Alpine Granite: [0.55, 0.57, 0.60]
# Warm Feldspar Quartzite: [0.68, 0.65, 0.60]
# High Weathered Peak Rock: [0.72, 0.73, 0.75]

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# Slope darkening (steep crags = dark slate cliffs, flat surfaces = lighter granite)
cliff_factor = smoothstep(22.0, 55.0, slope_deg)
# Elevation weathering (higher peaks have lighter quartz granite)
alt_factor = smoothstep(10.0, 55.0, RELIEF)
# Micro grain modulation
grain = 0.08 * rock_noise

# Base rock RGB as continuous linear blend
# Dark slate base:
slate_rgb = np.array([0.20, 0.22, 0.25])
# Granite midtone:
granite_rgb = np.array([0.56, 0.58, 0.61])
# Warm quartzite highlight:
quartz_rgb = np.array([0.68, 0.66, 0.62])

# Blend rock color continuously:
t_rock = np.clip(0.55 - 0.40 * cliff_factor + 0.35 * alt_factor + grain, 0.0, 1.0)
rock_rgb = np.zeros((ny, nx, 3), dtype=np.float32)
for c in range(3):
    # Two-stage smooth blend: slate -> granite -> quartz
    rock_rgb[:, :, c] = np.where(
        t_rock < 0.5,
        (1.0 - t_rock*2.0) * slate_rgb[c] + (t_rock*2.0) * granite_rgb[c],
        (1.0 - (t_rock-0.5)*2.0) * granite_rgb[c] + ((t_rock-0.5)*2.0) * quartz_rgb[c]
    )

print("Base rock field generated.")

# -------------------------------------------------------------
# 2. CONTINUOUS SNOW MANTLE FIELD (WEIGHT IN [0, 1])
# -------------------------------------------------------------
# High altitude snow probability
snow_alt = smoothstep(18.0, 38.0, RELIEF)

# Steep slope shedding (snow falls off cliffs > 45 deg, but smoothly!)
snow_shed = 1.0 - smoothstep(38.0, 68.0, slope_deg)

# Couloir and gully accumulation (negative laplacian = concavity)
couloir_flow = smoothstep(-0.8, -0.05, laplace) * smoothstep(12.0, 30.0, RELIEF)

# Combine into continuous snow thickness / coverage alpha in [0, 1]
np.random.seed(123)
snow_noise = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=6.0) * 0.12
snow_alpha = np.clip(snow_alt * (0.35 + 0.65 * snow_shed) + 0.45 * couloir_flow + snow_noise, 0.0, 1.0)
# Feather the edges so there is ZERO hard boundary
snow_alpha = smoothstep(0.20, 0.75, snow_alpha)

# Snow Color with Subtle Blue Ambient Subsurface Glow in Shade:
# Sunlit fresh snow: [0.97, 0.98, 1.00]
# Shaded snow / firn: [0.85, 0.91, 0.98]
snow_base_rgb = np.zeros((ny, nx, 3), dtype=np.float32)
snow_shade_factor = smoothstep(15.0, 48.0, slope_deg)
snow_base_rgb[:, :, 0] = (1.0 - 0.12 * snow_shade_factor) * 0.97
snow_base_rgb[:, :, 1] = (1.0 - 0.07 * snow_shade_factor) * 0.98
snow_base_rgb[:, :, 2] = 1.00

# -------------------------------------------------------------
# 3. CONTINUOUS GLACIAL ICE FIELD (WEIGHT IN [0, 1])
# -------------------------------------------------------------
# Glacial ice forms in shaded, deep cirque basins
ice_basin = smoothstep(14.0, 35.0, RELIEF) * smoothstep(-1.2, -0.2, laplace) * (1.0 - snow_alt * 0.6)
ice_noise = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=8.0) * 0.10
ice_alpha = np.clip(ice_basin + ice_noise, 0.0, 1.0)
ice_alpha = smoothstep(0.25, 0.70, ice_alpha) * (1.0 - snow_alpha * 0.85)

ice_rgb = np.array([0.42, 0.72, 0.88], dtype=np.float32)

# -------------------------------------------------------------
# 4. CONTINUOUS ALPINE TUNDRA & LICHEN FIELD (WEIGHT IN [0, 1])
# -------------------------------------------------------------
# Tundra grows in gentle slopes (<24 deg) near the base (RELIEF between 0.2 and 8.0 mm)
tundra_alt = smoothstep(0.1, 1.5, RELIEF) * (1.0 - smoothstep(4.0, 11.0, RELIEF))
tundra_slope = 1.0 - smoothstep(12.0, 26.0, slope_deg)

# Fractal lichen colony growth pattern
np.random.seed(888)
tundra_f1 = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=16.0)
tundra_f2 = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=5.0)
tundra_pattern = smoothstep(0.1, 0.8, 0.65 * tundra_f1 + 0.35 * tundra_f2)

tundra_alpha = np.clip(tundra_alt * tundra_slope * tundra_pattern * 1.5, 0.0, 1.0)
tundra_alpha = smoothstep(0.15, 0.65, tundra_alpha) * (1.0 - snow_alpha)

# Lichen color gradient: deep forest moss center -> golden xanthoria edge
moss_deep = np.array([0.22, 0.44, 0.18])
lichen_edge = np.array([0.55, 0.58, 0.26])
tundra_color = np.zeros((ny, nx, 3), dtype=np.float32)
for c in range(3):
    tundra_color[:, :, c] = tundra_alpha * moss_deep[c] + (1.0 - tundra_alpha) * lichen_edge[c]

# -------------------------------------------------------------
# 5. CONTINUOUS COMPOSITION (INFINITE COLOR SPECTRUM)
# -------------------------------------------------------------
# Start from rock
final_rgb = rock_rgb.copy()

# Blend tundra smoothly:
for c in range(3):
    final_rgb[:, :, c] = (1.0 - tundra_alpha) * final_rgb[:, :, c] + tundra_alpha * tundra_color[:, :, c]

# Blend glacial ice smoothly:
for c in range(3):
    final_rgb[:, :, c] = (1.0 - ice_alpha) * final_rgb[:, :, c] + ice_alpha * ice_rgb[c]

# Blend snow smoothly:
for c in range(3):
    final_rgb[:, :, c] = (1.0 - snow_alpha) * final_rgb[:, :, c] + snow_alpha * snow_base_rgb[:, :, c]

# Foundation Base, Road Margins (3.5mm) and Vertex Pads
road_margin = smoothstep(1.5, 3.5, dist_edge)
base_black = np.array([0.12, 0.13, 0.15], dtype=np.float32)
for c in range(3):
    final_rgb[:, :, c] = (1.0 - (1.0 - road_margin)) * final_rgb[:, :, c] + (1.0 - road_margin) * base_black[c]

final_rgb[dist_edge < 0] = [0.06, 0.08, 0.10]
final_rgb = np.clip(final_rgb, 0.0, 1.0)

# Check number of unique RGB colors in the 1024x1024 image
u_colors = len(np.unique((final_rgb.reshape(-1, 3) * 255).astype(np.uint8), axis=0))
print(f"TOTAL UNIQUE 24-BIT RGB COLORS GENERATED: {u_colors:,} (INFINITE SMOOTH SPECTRUM!)")

# Plot macro preview and extreme zoom preview
fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#090d16')
axes[0].imshow(final_rgb, origin='lower')
axes[0].set_title("Full Color Continuo 1024x1024 (Espectro Infinito)", color='white', fontsize=12)
axes[0].axis('off')

# Extreme zoom on the snow/rock/tundra boundary
mid_y, mid_x = int(ny * 0.42), int(nx * 0.52)
zoom_crop = final_rgb[mid_y-80:mid_y+80, mid_x-80:mid_x+80]
axes[1].imshow(zoom_crop, origin='lower', interpolation='bicubic')
axes[1].set_title("Super-Zoom 1000% en Transición Nieve/Roca/Tundra (Cero Píxeles)", color='white', fontsize=12)
axes[1].axis('off')

plt.tight_layout()
plt.savefig('scratch/infinite_fullcolor_preview.png', dpi=200, facecolor='#090d16')
print("Saved scratch/infinite_fullcolor_preview.png")
