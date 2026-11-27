import math
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

Z_top = np.load('scratch/Z_top_refined.npy')
ny, nx = Z_top.shape
R = 64.0
APOTHEM = R * math.cos(math.radians(30))
xs = np.linspace(-APOTHEM, APOTHEM, nx)
ys = np.linspace(-R, R, ny)
X, Y = np.meshgrid(xs, ys)

normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
gy, gx = np.gradient(Z_top, dy, dx)
slope = np.hypot(gx, gy)
slope_deg = np.degrees(np.arctan(slope))
laplace = ndi.laplace(Z_top)

az = 315.0 * math.pi / 180.0; alt = 48.0 * math.pi / 180.0
sun_x = math.cos(alt)*math.sin(az); sun_y = math.cos(alt)*math.cos(az); sun_z = math.sin(alt)
norm = np.sqrt(gx**2 + gy**2 + 1.0)
hs = np.clip((-gx/norm)*sun_x + (-gy/norm)*sun_y + (1.0/norm)*sun_z, 0.0, 1.0)

final_relief = Z_top - 25.0

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# REALISTIC ALPINE SNOW COVERAGE:
# 1. Permanent Snow Zone: Relief > 28mm (total Z > 53mm).
# On real alpine peaks, snow clings to all slopes up to 60 degrees.
high_snow = smoothstep(25.0, 40.0, final_relief) * (1.0 - smoothstep(55.0, 72.0, slope_deg) * 0.7)
# 2. Couloir snow: chutes of snow extending down to relief 15mm in concave gullies
couloir_snow = smoothstep(12.0, 30.0, final_relief) * np.clip(-laplace * 0.8, 0.0, 1.0) * (1.0 - smoothstep(45.0, 60.0, slope_deg) * 0.5)

snow_score = np.maximum(high_snow, couloir_snow)
snow_mask = (snow_score > 0.35) & (final_relief > 12.0)

# 3. Glacial Blue Ice (Cyan T1):
# Glacial firn/ice in deep cirque basins under the snow, and margins of snowfields
ice_score = smoothstep(15.0, 32.0, final_relief) * np.clip(-laplace * 0.6, 0.0, 1.0) * (1.0 - snow_score * 0.7)
ice_mask = (ice_score > 0.22) & (final_relief > 14.0) & (~snow_mask)

# 4. Alpine Tundra & Lichen (Green T5):
# Natural organic patches in the low valley landings (Z 25.0 to 28.0mm, slope < 25 deg)
np.random.seed(888)
tundra_noise = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=3.5)
tundra_score = smoothstep(6.5, 0.5, final_relief) * (1.0 - smoothstep(16.0, 28.0, slope_deg)) * (tundra_noise > 0.1)
tundra_mask = (tundra_score > 0.30) & (final_relief > 0.1) & (dist_edge > 3.2) & (~snow_mask)

# 5. Dark Slate Shadow Cliffs (Black T3):
# Steep cliffs (> 42 deg) and crevasse shadows
cliff_mask = (slope_deg > 40.0) & (final_relief > 1.5) & (~snow_mask) & (~ice_mask)

# 6. Sunlit Granite Rock Face (Grey T4):
granite_mask = (final_relief > 0.05) & (~snow_mask) & (~ice_mask) & (~tundra_mask) & (~cliff_mask)

# 7. Road Shelf & Vertex Pads:
shelf_mask = (final_relief <= 0.05) & (dist_edge >= 0)

# Build RGB image
rgb = np.zeros((ny, nx, 3), dtype=np.float32)
rgb[granite_mask] = [0.58, 0.60, 0.63]
rgb[cliff_mask] = [0.24, 0.26, 0.28]
rgb[ice_mask] = [0.44, 0.72, 0.86]
rgb[snow_mask] = [0.96, 0.97, 0.99]
rgb[tundra_mask] = [0.30, 0.52, 0.26]
rgb[shelf_mask] = [0.18, 0.20, 0.23]
rgb[dist_edge < 0] = [0.06, 0.08, 0.12]

lit_rgb = rgb * (0.35 + 0.65 * hs[:, :, None])
lit_rgb[dist_edge < 0] = [0.06, 0.08, 0.12]
lit_rgb = np.clip(lit_rgb, 0.0, 1.0)

plt.figure(figsize=(10, 10), facecolor='#090d16')
plt.imshow(lit_rgb, origin='lower')
plt.title('Calibrated Alpine Snow & Granite Painting', color='white', fontsize=12)
plt.axis('off')
plt.tight_layout()
plt.savefig('scratch/calibrated_snow_preview.png', dpi=200, facecolor='#090d16')
print("Saved scratch/calibrated_snow_preview.png")

print(f"Snow pixels: {np.sum(snow_mask)} ({100*np.sum(snow_mask)/np.sum(dist_edge>=0):.1f}%)")
print(f"Ice pixels:  {np.sum(ice_mask)} ({100*np.sum(ice_mask)/np.sum(dist_edge>=0):.1f}%)")
print(f"Granite:     {np.sum(granite_mask)} ({100*np.sum(granite_mask)/np.sum(dist_edge>=0):.1f}%)")
print(f"Cliffs:      {np.sum(cliff_mask)} ({100*np.sum(cliff_mask)/np.sum(dist_edge>=0):.1f}%)")
print(f"Tundra:      {np.sum(tundra_mask)} ({100*np.sum(tundra_mask)/np.sum(dist_edge>=0):.1f}%)")
