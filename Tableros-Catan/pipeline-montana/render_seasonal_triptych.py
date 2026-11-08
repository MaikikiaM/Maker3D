import os
import math
import time
import shutil
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from PIL import Image

print("=== RENDERING 3-SEASON FULL-COLOR COMPARISON TRIPTYCH ===")
t0 = time.time()

out_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

Z_master = np.load('scratch/Z_top_master_2m.npy')
import scipy.interpolate as interp
ny_m, nx_m = Z_master.shape

R = 64.0
APOTHEM = R * math.cos(math.radians(30))

res = 1024
xs = np.linspace(-APOTHEM, APOTHEM, res, dtype=np.float32)
ys = np.linspace(-R, R, res, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)

xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

pts = np.column_stack((Y.ravel(), X.ravel()))
Z = interp_z(pts).reshape((res, res))

normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

dx = float(xs[1] - xs[0]); dy = float(ys[1] - ys[0])
gy, gx = np.gradient(Z, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm
Ny = -gy / N_norm
Nz = 1.0 / N_norm

laplace = ndi.laplace(Z)
ao = 1.0 - np.clip(-laplace * 0.40, 0.0, 0.50)

# Dual light
az_sun = 310.0 * math.pi / 180.0; alt_sun = 46.0 * math.pi / 180.0
Lx_sun = math.cos(alt_sun) * math.sin(az_sun); Ly_sun = math.cos(alt_sun) * math.cos(az_sun); Lz_sun = math.sin(alt_sun)
diff_sun = np.clip(Nx * Lx_sun + Ny * Ly_sun + Nz * Lz_sun, 0.0, 1.0)

az_sky = 130.0 * math.pi / 180.0; alt_sky = 60.0 * math.pi / 180.0
Lx_sky = math.cos(alt_sky) * math.sin(az_sky); Ly_sky = math.cos(alt_sky) * math.cos(az_sky); Lz_sky = math.sin(alt_sky)
diff_sky = np.clip(Nx * Lx_sky + Ny * Ly_sky + Nz * Lz_sky, 0.0, 1.0)

# Load the 3 textures
tex_summer = Image.open(r"Tableros Catan\Loseta_Montana_3D\catan_mountain_diffuse_4k.png").resize((res, res), Image.Resampling.LANCZOS)
tex_winter = Image.open(r"Tableros Catan\Loseta_Montana_3D\catan_mountain_diffuse_winter_4k.png").resize((res, res), Image.Resampling.LANCZOS)
tex_autumn = Image.open(r"Tableros Catan\Loseta_Montana_3D\catan_mountain_diffuse_autumn_4k.png").resize((res, res), Image.Resampling.LANCZOS)

seasons = [
    ("1. Primavera / Verano Alpino (Master)", (np.array(tex_summer).astype(np.float32) / 255.0)[::-1]),
    ("2. Invierno Glaciar (Manto Profundo de Nieve)", (np.array(tex_winter).astype(np.float32) / 255.0)[::-1]),
    ("3. Otoño Dorado (Lárices y Pátina Ocre)", (np.array(tex_autumn).astype(np.float32) / 255.0)[::-1])
]

# Surface coords for 3D plot
step = 2
surf_x = X[::step, ::step]
surf_y = Y[::step, ::step]
surf_z = Z[::step, ::step]
dist_sub = dist_edge[::step, ::step]

fig = plt.figure(figsize=(19, 7), facecolor='#090d16')

for idx, (title, diff_arr) in enumerate(seasons):
    # Flip array vertically to match Z coordinate orientation
    diff_arr = diff_arr[::-1]
    lit_s = diff_arr * (
        diff_sun[:, :, None] * np.array([1.06, 1.00, 0.92]) * 0.72 +
        diff_sky[:, :, None] * np.array([0.70, 0.84, 0.98]) * 0.35 +
        np.array([0.22, 0.25, 0.32]) * ao[:, :, None] * 0.42
    )
    lit_s[dist_edge < 0] = [0.06, 0.08, 0.12]
    lit_s = np.clip(lit_s, 0.0, 1.0)
    
    rgba_s = np.zeros((surf_x.shape[0], surf_x.shape[1], 4))
    rgba_s[:, :, :3] = lit_s[::step, ::step]
    rgba_s[:, :, 3] = 1.0
    rgba_s[dist_sub < -0.2, 3] = 0.0
    
    ax = fig.add_subplot(1, 3, idx + 1, projection='3d', facecolor='#090d16')
    ax.plot_surface(surf_x, surf_y, surf_z, facecolors=rgba_s, rstride=1, cstride=1, shade=False, antialiased=True)
    ax.view_init(elev=32, azim=-52)
    ax.set_box_aspect([1.0, 1.15, 0.70])
    ax.set_title(title, color='#38bdf8', fontsize=11.5, fontweight='bold', pad=10)
    ax.axis('off')

plt.suptitle("TRÍPTICO ESTACIONAL FULL-COLOR: 3 PALETAS CONTINUAS DISPONIBLES EN 4K",
             color='#38bdf8', fontsize=14, fontweight='bold', y=0.98)
out_trip = os.path.join(out_dir, "render_mountain_seasonal_triptych.png")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(out_trip, dpi=180, bbox_inches='tight', facecolor='#090d16')
plt.close()
print(f"Saved Seasonal Triptych: {out_trip}")

shutil.copy(out_trip, os.path.join(brain_dir, os.path.basename(out_trip)))
print(f"Total time: {time.time()-t0:.2f}s")
