import math
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

print("=== TESTING HYBRID: REAL LIDAR TOPOGRAPHY + PYRAMIDAL ALPINE ENVELOPE ===")

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF_H = Z_PEAK - Z_BASE # 58.0 mm

# High-resolution grid: 451 x 521
nx, ny = 451, 521
xs = np.linspace(-APOTHEM, APOTHEM, nx)
ys = np.linspace(-R, R, ny)
X, Y = np.meshgrid(xs, ys)

# Load Grand Teton and Matterhorn DEMs
t_z15 = np.load('scratch/teton_z15_768.npy')
m_z15 = np.load('scratch/matterhorn_z15_768.npy')

# Teton Cathedral Group centered on (452, 388)
from scipy.ndimage import zoom
w_t = 140
crop_t = t_z15[452 - w_t : 452 + w_t, 388 - w_t : 388 + w_t]
res_t = zoom(crop_t, (ny / (2*w_t), nx / (2*w_t)), order=3)

# Matterhorn Horn centered on (426, 245)
w_m = 95
crop_m = m_z15[426 - w_m : 426 + w_m, 245 - w_m : 245 + w_m]
res_m = zoom(crop_m, (ny / (2*w_m), nx / (2*w_m)), order=3)

# Normalize both
dem_t = (res_t - res_t.min()) / (res_t.max() - res_t.min())
dem_m = (res_m - res_m.min()) / (res_m.max() - res_m.min())

# Filter out raster quantization banding using Gaussian smoothing on the high-frequency band
# High-pass filter with gentle cutoff removes staircasing while keeping crags
t_smooth = ndi.gaussian_filter(dem_t, sigma=1.5)
t_crags = dem_t - t_smooth
dem_filtered = t_smooth + np.clip(t_crags, -0.05, 0.05)

# Blend Matterhorn horn at center
r = np.hypot(X, Y)
horn_mask = np.clip(1.0 - (r / 28.0)**2.0, 0.0, 1.0)**1.5
massif_raw = dem_filtered * (1.0 - horn_mask * 0.4) + np.maximum(dem_filtered, dem_m) * (horn_mask * 0.4)
massif_raw = (massif_raw - massif_raw.min()) / (massif_raw.max() - massif_raw.min())

# THE PYRAMIDAL ALPINE ENVELOPE:
# To eliminate the "panettone" cylinder:
# We multiply by an exponential pyramidal falloff that starts tapering immediately from the center!
# Envelope is 1.0 at r=0, and smoothly reaches 0.0 at r=50mm
R_envelope = 50.0
r_envelope = np.clip(r / R_envelope, 0.0, 1.0)
# Power 1.5 ensures steep soaring summit and gentle broad talus base
envelope = (1.0 - r_envelope)**1.55

# Apply envelope to create genuine soaring mountain pyramid
massif_pyramid = massif_raw * envelope

# Ridge sharpening: unsharp masking to accentuate knife-edge arêtes
blur = ndi.gaussian_filter(massif_pyramid, sigma=1.8)
massif_sharp = np.clip(massif_pyramid + 1.4 * (massif_pyramid - blur), 0.0, 1.0)
massif_sharp = massif_sharp / massif_sharp.max()

# Clearance: Road shelf (3.5mm) and vertex pads (20mm diam)
normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

v_angles = [(30 + 60 * k) * math.pi / 180 for k in range(6)]
hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles])
dist_vert = np.full_like(X, 999.0)
for vx, vy in hex_corners:
    dist_vert = np.minimum(dist_vert, np.hypot(X - vx, Y - vy))

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

edge_factor = smoothstep(3.5, 10.0, dist_edge)
vert_factor = smoothstep(10.0, 14.0, dist_vert)
clearance = edge_factor * vert_factor

relief = massif_sharp * clearance * RELIEF_H
Z_surface = Z_BASE + relief
Z_surface[dist_edge < 0] = Z_BASE

# Hillshade
dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
gy, gx = np.gradient(Z_surface, dy, dx)
az = 315.0 * np.pi / 180.0; alt = 48.0 * np.pi / 180.0
sun_x = np.cos(alt)*np.sin(az); sun_y = np.cos(alt)*np.cos(az); sun_z = np.sin(alt)
norm = np.sqrt(gx**2 + gy**2 + 1.0)
hs = np.clip((-gx/norm)*sun_x + (-gy/norm)*sun_y + (1.0/norm)*sun_z, 0.0, 1.0)

fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor='#090d16')
axes[0].imshow(Z_surface, cmap='terrain', origin='lower')
axes[0].set_title('Topografía Híbrida LiDAR + Envolvente Piramidal', color='white')
axes[0].axis('off')

axes[1].imshow(hs, cmap='gray', origin='lower')
axes[1].set_title('Sombreado de Relieve (Cero Panettone, Aristas Vivas)', color='white')
axes[1].axis('off')

mid_y = ny // 2
axes[2].plot(xs, Z_surface[mid_y, :], color='#38bdf8', lw=2.5, label='Perfil E-W Transversal')
axes[2].axhline(25.0, color='#94a3b8', linestyle='--', label='Zócalo Base (25mm)')
axes[2].set_ylim(20, 90)
axes[2].set_title('Perfil Transversal (Pendiente Continua Suave)', color='white')
axes[2].set_facecolor('#0f172a')
axes[2].tick_params(colors='white')
axes[2].legend(facecolor='#1e293b', labelcolor='white')
axes[2].grid(True, color='#334155', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('scratch/hybrid_pyramid_preview.png', dpi=200, facecolor='#090d16')
print("Saved scratch/hybrid_pyramid_preview.png")
