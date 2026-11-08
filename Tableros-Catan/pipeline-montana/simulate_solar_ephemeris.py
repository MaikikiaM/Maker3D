import os
import math
import time
import shutil
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from PIL import Image

print("=== ALPINE SOLAR EPHEMERIS & RAY-CAST HORIZON SHADOWS ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"
os.makedirs(out_dir, exist_ok=True)

R = 64.0
APOTHEM = R * math.cos(math.radians(30))

Z_master = np.load('scratch/Z_top_master_2m.npy')
diffuse_4k = np.load('scratch/diffuse_4k_array.npy')
ny_m, nx_m = Z_master.shape

res = 1024
xs = np.linspace(-APOTHEM, APOTHEM, res, dtype=np.float32)
ys = np.linspace(-R, R, res, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)
dx = float(xs[1] - xs[0])
dy = float(ys[1] - ys[0])

import scipy.interpolate as interp
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

pts = np.column_stack((Y.ravel(), X.ravel()))
Z = interp_z(pts).reshape((res, res))

im_diff = Image.fromarray((diffuse_4k * 255.0).astype(np.uint8))
diffuse_res = np.array(im_diff.resize((res, res), Image.Resampling.LANCZOS)).astype(np.float32) / 255.0

# Surface normals
gy, gx = np.gradient(Z, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm
Ny = -gy / N_norm
Nz = 1.0 / N_norm

normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

laplace = ndi.laplace(Z)
ao = 1.0 - np.clip(-laplace * 0.40, 0.0, 0.50)

# Ray-marching Horizon Shadow Function
def compute_horizon_shadow(Z_grid, az_deg, alt_deg, num_steps=64, step_size_px=2.5):
    az_rad = math.radians(az_deg)
    alt_rad = math.radians(alt_deg)
    
    # Vector towards sun in grid coordinates (x, y)
    cos_az = math.sin(az_rad) # X direction
    sin_az = math.cos(az_rad) # Y direction
    tan_alt = math.tan(alt_rad)
    
    # Distance per pixel step in world units
    ds_world = step_size_px * dx
    dz_step = ds_world * tan_alt
    
    ny_g, nx_g = Z_grid.shape
    shadow = np.ones((ny_g, nx_g), dtype=np.float32)
    
    # Precompute sample offsets
    for step in range(1, num_steps + 1):
        off_x = int(round(step * step_size_px * cos_az))
        off_y = int(round(step * step_size_px * sin_az))
        
        # Roll / slice coordinates
        y_src = np.clip(np.arange(ny_g) + off_y, 0, ny_g - 1)
        x_src = np.clip(np.arange(nx_g) + off_x, 0, nx_g - 1)
        
        sampled_Z = Z_grid[np.ix_(y_src, x_src)]
        ray_Z = Z_grid + step * dz_step
        
        # If occluding terrain is higher than ray, point is in shadow
        is_occluded = sampled_Z > ray_Z
        shadow[is_occluded] = 0.0
        
    # Smooth shadow boundary slightly
    shadow_smooth = ndi.gaussian_filter(shadow, 1.2)
    return shadow_smooth

scenarios = [
    {
        "title": "1. Amanecer Alpino (Alpenglow - 06:30 AM)",
        "az": 70.0, "alt": 14.0,
        "sun_color": np.array([1.18, 0.72, 0.55]),
        "sky_color": np.array([0.45, 0.55, 0.82]),
        "amb": 0.18
    },
    {
        "title": "2. Mediodía Solar Cenital (12:30 PM)",
        "az": 180.0, "alt": 65.0,
        "sun_color": np.array([1.08, 1.05, 0.96]),
        "sky_color": np.array([0.65, 0.82, 1.05]),
        "amb": 0.32
    },
    {
        "title": "3. Atardecer Golden Hour (19:30 PM)",
        "az": 290.0, "alt": 15.0,
        "sun_color": np.array([1.22, 0.82, 0.45]),
        "sky_color": np.array([0.55, 0.50, 0.75]),
        "amb": 0.20
    },
    {
        "title": "4. Luz de Luna & Hora Azul (21:30 PM)",
        "az": 135.0, "alt": 38.0,
        "sun_color": np.array([0.55, 0.75, 1.05]),
        "sky_color": np.array([0.25, 0.35, 0.60]),
        "amb": 0.15
    }
]

fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor='#090d16')
axes = axes.ravel()

for idx, sc in enumerate(scenarios):
    print(f"Simulating {sc['title']} (Az: {sc['az']}°, Alt: {sc['alt']}°)...")
    sh = compute_horizon_shadow(Z, sc['az'], sc['alt'])
    
    az_rad = math.radians(sc['az'])
    alt_rad = math.radians(sc['alt'])
    Lx = math.cos(alt_rad) * math.sin(az_rad)
    Ly = math.cos(alt_rad) * math.cos(az_rad)
    Lz = math.sin(alt_rad)
    
    ndotl = np.clip(Nx * Lx + Ny * Ly + Nz * Lz, 0.0, 1.0)
    direct_light = ndotl * sh
    
    ambient_light = sc['amb'] * ao
    sky_light = np.clip(Nz * 0.5 + 0.5, 0.0, 1.0) * 0.35
    
    lit_color = diffuse_res * (
        direct_light[:, :, None] * sc['sun_color'][None, None, :] * 0.85 +
        sky_light[:, :, None] * sc['sky_color'][None, None, :] * 0.40 +
        ambient_light[:, :, None] * sc['sky_color'][None, None, :] * 0.50
    )
    lit_color[dist_edge < 0] = [0.06, 0.08, 0.12]
    lit_color = np.clip(lit_color, 0.0, 1.0)
    
    axes[idx].imshow(lit_color, origin='lower')
    axes[idx].set_title(sc['title'], color='#38bdf8', fontsize=12, fontweight='bold', pad=10)
    axes[idx].axis('off')

plt.suptitle("Estudio de Iluminación Solar Alpina & Sombras Arrojadas Ray-Casting\n(Física de Nieve 0% en Caras Escarpadas y Espectro de 45 Grises + 15 Verdes)",
             color='#e2e8f0', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()

out_ephemeris = os.path.join(out_dir, "render_alpine_solar_ephemeris_study.png")
plt.savefig(out_ephemeris, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy(out_ephemeris, os.path.join(brain_dir, "render_alpine_solar_ephemeris_study.png"))
print(f"Saved: {out_ephemeris} and copied to brain")
print(f"Solar ephemeris simulation completed in {time.time()-t0:.2f}s.")
