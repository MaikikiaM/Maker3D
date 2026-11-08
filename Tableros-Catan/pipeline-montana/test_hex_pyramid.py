import math
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

print("=== TESTING HEXAGON-MATCHED ALPINE ENVELOPE WITH SECONDARY SPIRES ===")

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF_H = Z_PEAK - Z_BASE # 58.0 mm

# Dense evaluation grid: 451 x 521
nx, ny = 451, 521
xs = np.linspace(-APOTHEM, APOTHEM, nx)
ys = np.linspace(-R, R, ny)
X, Y = np.meshgrid(xs, ys)

# Exact Hexagon Edge Distance
normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

v_angles = [(30 + 60 * k) * math.pi / 180 for k in range(6)]
hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles])
dist_vert = np.full_like(X, 999.0)
for vx, vy in hex_corners:
    dist_vert = np.minimum(dist_vert, np.hypot(X - vx, Y - vy))

# Load LiDAR DEMs
t_z15 = np.load('scratch/teton_z15_768.npy')
m_z15 = np.load('scratch/matterhorn_z15_768.npy')

from scipy.ndimage import zoom
w_t = 150
crop_t = t_z15[452 - w_t : 452 + w_t, 388 - w_t : 388 + w_t]
dem_t = zoom(crop_t, (ny / (2*w_t), nx / (2*w_t)), order=3)
dem_t = (dem_t - dem_t.min()) / (dem_t.max() - dem_t.min())

w_m = 100
crop_m = m_z15[426 - w_m : 426 + w_m, 245 - w_m : 245 + w_m]
dem_m = zoom(crop_m, (ny / (2*w_m), nx / (2*w_m)), order=3)
dem_m = (dem_m - dem_m.min()) / (dem_m.max() - dem_m.min())

# Remove DEM raster quantization steps with bilateral/Gaussian filtering
dem_t_smooth = ndi.gaussian_filter(dem_t, sigma=1.2)

# Hexagon-matched normalized center distance:
# 1.0 at center (0,0), drops to 0.0 at the hex perimeter
# We account for the 3.5mm road shelf: active mountain domain is dist_edge > 3.5mm
dist_from_road = np.clip(dist_edge - 3.5, 0.0, None)
max_dist = APOTHEM - 3.5 # ~51.9 mm
eta_hex = np.clip(dist_from_road / max_dist, 0.0, 1.0)

# Natural alpine pyramid slope curve:
# eta_hex goes from 0 at road shelf to 1 at center.
# Power 1.4 creates a dramatic concave slope (steep at summit, tapering into gentle talus)
pyramid_envelope = np.power(eta_hex, 1.45)

# Vertex pad clearance: 10mm radius (20mm diam) circles at the 6 vertices
def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

vert_factor = smoothstep(10.0, 15.0, dist_vert)
road_factor = smoothstep(0.0, 8.0, dist_from_road)
clearance = vert_factor * road_factor

# Multi-peak structure:
# 1. Matterhorn Central Horn (pinnacle reaching full 58mm)
r_center = np.hypot(X - (-2.0), Y - 3.0)
horn_weight = np.clip(1.0 - (r_center / 26.0)**1.8, 0.0, 1.0)
composite_dem = dem_t_smooth * (1.0 - horn_weight * 0.45) + np.maximum(dem_t_smooth, dem_m) * (horn_weight * 0.45)
composite_dem = (composite_dem - composite_dem.min()) / (composite_dem.max() - composite_dem.min())

# Combine DEM crags with the Pyramidal Hexagon Envelope
# Base relief: envelope * 0.65 + envelope * composite_dem * 0.35
# This guarantees a pyramidal silhouette while preserving rich real LiDAR crags and couloirs!
mountain_sculpture = (0.50 * pyramid_envelope + 0.50 * (pyramid_envelope * composite_dem))
mountain_sculpture = mountain_sculpture / mountain_sculpture.max()

# Knife-Edge Arêtes Accentuation (Unsharp mask on the mountain sculpture)
blur_sculpt = ndi.gaussian_filter(mountain_sculpture, sigma=1.8)
sculpt_sharp = np.clip(mountain_sculpture + 1.5 * (mountain_sculpture - blur_sculpt), 0.0, 1.0)
sculpt_sharp = sculpt_sharp / sculpt_sharp.max()

# Final Relief scaled to 58.0 mm
relief = sculpt_sharp * clearance * RELIEF_H
Z_top = Z_BASE + relief
Z_top[dist_edge < 0] = Z_BASE

# Hillshade
dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
gy, gx = np.gradient(Z_top, dy, dx)
az = 315.0 * np.pi / 180.0; alt = 48.0 * np.pi / 180.0
sun_x = np.cos(alt)*np.sin(az); sun_y = np.cos(alt)*np.cos(az); sun_z = np.sin(alt)
norm = np.sqrt(gx**2 + gy**2 + 1.0)
hs = np.clip((-gx/norm)*sun_x + (-gy/norm)*sun_y + (1.0/norm)*sun_z, 0.0, 1.0)

fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor='#090d16')
axes[0].imshow(Z_top, cmap='terrain', origin='lower')
axes[0].set_title('Topografía Hexagonal Alpina (Z 25-83mm)', color='white')
axes[0].axis('off')

axes[1].imshow(hs, cmap='gray', origin='lower')
axes[1].set_title('Sombreado de Relieve (Cero Panettone, Aristas Nítidas)', color='white')
axes[1].axis('off')

mid_y = ny // 2
axes[2].plot(xs, Z_top[mid_y, :], color='#38bdf8', lw=2.5, label='Perfil E-W')
axes[2].axhline(25.0, color='#94a3b8', linestyle='--', label='Zócalo Base (25mm)')
axes[2].set_ylim(20, 90)
axes[2].set_title('Perfil Transversal (Pendiente Piramidal Suave)', color='white')
axes[2].set_facecolor('#0f172a')
axes[2].tick_params(colors='white')
axes[2].legend(facecolor='#1e293b', labelcolor='white')
axes[2].grid(True, color='#334155', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('scratch/hex_pyramid_preview.png', dpi=200, facecolor='#090d16')
print("Saved scratch/hex_pyramid_preview.png")
