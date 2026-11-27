import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

# Load real DEM crop (400x400)
dem = np.load('scratch/matterhorn_dem_crop.npy')
H_dem, W_dem = dem.shape
cy, cx = H_dem // 2, W_dem // 2 # summit is near center

# Normalization of DEM relief:
# Summit is at 4354.5m, valleys around 2400m
dem_min = 2300.0
dem_max = dem.max()
dem_norm = np.clip((dem - dem_min) / (dem_max - dem_min), 0.0, 1.0)

# Catan Hexagon parameters
R = 64.0 # mm
apothem = R * np.cos(np.pi / 6) # 55.4256 mm
z_base = 25.0 # mm
z_peak = 83.0 # mm
target_relief = z_peak - z_base # 58.0 mm

# High-resolution sampling grid (301 x 347)
nx = 301
ny = int(nx * (2 * R) / (2 * apothem))
xs = np.linspace(-apothem, apothem, nx)
ys = np.linspace(-R, R, ny)
X, Y = np.meshgrid(xs, ys)

# Hexagon distance to edges
angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
normals = np.column_stack([np.cos(angles), np.sin(angles)])
d_edges = np.zeros((6, ny, nx))
for k in range(6):
    d_edges[k] = apothem - (X * normals[k, 0] + Y * normals[k, 1])
dist_edge = np.min(d_edges, axis=0)
mask_hex = dist_edge >= 0.0

# 6 vertices
v_angles = (30.0 + 60.0 * np.arange(6)) * np.pi / 180.0
vx = R * np.cos(v_angles)
vy = R * np.sin(v_angles)
dist_vert = np.full(X.shape, 999.0)
for k in range(6):
    dist_vert = np.minimum(dist_vert, np.hypot(X - vx[k], Y - vy[k]))
    
def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
    
# Smooth clearance envelope
clearance = smoothstep(3.2, 7.5, dist_edge) * smoothstep(9.5, 13.5, dist_vert)

# Map hex coordinates (X, Y in mm) to DEM pixel coordinates
# Hex extent: radius 64mm. Map 64mm to ~160 pixels in DEM (radius of ~3km)
scale_pix_per_mm = 2.45
pixel_x = cx + X * scale_pix_per_mm
pixel_y = cy - Y * scale_pix_per_mm # invert Y for image coordinates

# Sample DEM with bicubic interpolation
coords = np.array([pixel_y.ravel(), pixel_x.ravel()])
sampled_norm = map_coordinates(dem_norm, coords, order=3, mode='nearest').reshape(X.shape)

# Subtract base elevation so the foothills reach 0 at the boundary
sampled_norm = np.maximum(0.0, sampled_norm - 0.08)
sampled_norm = sampled_norm / (np.max(sampled_norm) + 1e-8)

# Natural mountain tapering at outer boundary
r_xy = np.hypot(X, Y)
falloff = np.exp(-(r_xy / 54.0)**3.0)
raw_relief = sampled_norm * falloff * target_relief

# Re-normalize summit to exact 58.0 mm
raw_relief = raw_relief * (target_relief / np.max(raw_relief))

# Apply clearance factor (3.5mm flat road shelf + vertex pads)
sculpted_relief = raw_relief * clearance
Z_top = z_base + sculpted_relief

print('Matterhorn DEM mapped to Catan hex:')
print(f'  Min Z (inside): {np.min(Z_top[mask_hex]):.2f} mm')
print(f'  Max Z (summit): {np.max(Z_top[mask_hex]):.2f} mm')

# Render 3D preview
fig = plt.figure(figsize=(11, 9), facecolor='#070b14')
ax = fig.add_subplot(111, projection='3d', facecolor='#070b14')

Z_plot = np.copy(Z_top)
Z_plot[~mask_hex] = np.nan

# Calculate slope for realistic shading
gy, gx = np.gradient(Z_top, ys[1]-ys[0], xs[1]-xs[0])
slope = np.hypot(gx, gy)

surf = ax.plot_surface(X, Y, Z_plot, cmap='copper', vmin=25, vmax=83,
                       linewidth=0, antialiased=True, rcount=180, ccount=180)
ax.view_init(elev=35, azim=220)
ax.set_axis_off()
plt.tight_layout()
plt.savefig('scratch/matterhorn_dem_catan_preview.png', dpi=200, facecolor='#070b14')
print('Saved preview to scratch/matterhorn_dem_catan_preview.png')
