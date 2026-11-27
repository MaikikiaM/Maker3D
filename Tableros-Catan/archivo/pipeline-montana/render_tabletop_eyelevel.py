import os
import math
import time
import shutil
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from PIL import Image

print("=== RENDERING 3-PANEL TABLETOP PLAYER EYE-LEVEL PERSPECTIVES ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

R = 64.0
APOTHEM = R * math.cos(math.radians(30))

Z_master = np.load('scratch/Z_top_master_2m.npy')
diffuse_4k = np.load('scratch/diffuse_4k_array.npy')
ny_m, nx_m = Z_master.shape

res = 1024
xs = np.linspace(-APOTHEM, APOTHEM, res, dtype=np.float32)
ys = np.linspace(-R, R, res, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)
dx = float(xs[1] - xs[0]); dy = float(ys[1] - ys[0])

import scipy.interpolate as interp
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

pts = np.column_stack((Y.ravel(), X.ravel()))
Z = interp_z(pts).reshape((res, res))

im_diff = Image.fromarray((diffuse_4k * 255.0).astype(np.uint8))
diffuse_res = np.array(im_diff.resize((res, res), Image.Resampling.LANCZOS)).astype(np.float32) / 255.0

normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

gy, gx = np.gradient(Z, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm; Ny = -gy / N_norm; Nz = 1.0 / N_norm

laplace = ndi.laplace(Z)
ao = 1.0 - np.clip(-laplace * 0.40, 0.0, 0.50)

# Shading function for arbitrary sun angle
def shade_terrain(az_sun_deg, alt_sun_deg, sun_col, sky_col):
    az_rad = math.radians(az_sun_deg); alt_rad = math.radians(alt_sun_deg)
    Lx = math.cos(alt_rad) * math.sin(az_rad); Ly = math.cos(alt_rad) * math.cos(az_rad); Lz = math.sin(alt_rad)
    ndotl = np.clip(Nx * Lx + Ny * Ly + Nz * Lz, 0.0, 1.0)
    lit = diffuse_res * (
        ndotl[:, :, None] * sun_col[None, None, :] * 0.75 +
        (Nz[:, :, None] * 0.5 + 0.5) * sky_col[None, None, :] * 0.35 +
        np.array([0.22, 0.25, 0.32])[None, None, :] * ao[:, :, None] * 0.42
    )
    lit[dist_edge < 0] = [0.06, 0.08, 0.12]
    return np.clip(lit, 0.0, 1.0)

views = [
    {
        "title": "A. Perspectiva Ojo de Jugador (Elev: 18°, Azim: -45°)\nVista Frontal desde la Silla de Juego",
        "elev": 18, "azim": -45,
        "sun_az": 315, "sun_alt": 45,
        "sun_col": np.array([1.06, 1.00, 0.92]), "sky_col": np.array([0.70, 0.84, 0.98])
    },
    {
        "title": "B. Perspectiva Contrapicada Pared Norte (Elev: 14°, Azim: 60°)\nPared de Roca Escarpada 100% Expuesta (0% Nieve)",
        "elev": 14, "azim": 60,
        "sun_az": 200, "sun_alt": 40,
        "sun_col": np.array([1.10, 1.02, 0.90]), "sky_col": np.array([0.65, 0.80, 1.00])
    },
    {
        "title": "C. Amanecer Rasante sobre Aristas (Elev: 22°, Azim: -130°)\nIluminación Rasante sobre Crestas y Corredores Couloirs",
        "elev": 22, "azim": -130,
        "sun_az": 75, "sun_alt": 18,
        "sun_col": np.array([1.20, 0.78, 0.55]), "sky_col": np.array([0.50, 0.60, 0.85])
    }
]

fig = plt.figure(figsize=(20, 8.5), facecolor='#090d16')
step = 3
surf_x = X[::step, ::step]; surf_y = Y[::step, ::step]; surf_z = Z[::step, ::step]
dist_sub = dist_edge[::step, ::step]

for idx, v in enumerate(views):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d', facecolor='#090d16')
    lit_v = shade_terrain(v["sun_az"], v["sun_alt"], v["sun_col"], v["sky_col"])
    
    rgba = np.zeros((surf_x.shape[0], surf_x.shape[1], 4))
    rgba[:, :, :3] = lit_v[::step, ::step]
    rgba[:, :, 3] = 1.0
    rgba[dist_sub < -0.2, 3] = 0.0
    
    ax.plot_surface(surf_x, surf_y, surf_z, facecolors=rgba, rstride=1, cstride=1, shade=False, antialiased=True)
    ax.view_init(elev=v["elev"], azim=v["azim"])
    ax.set_box_aspect([1.0, 1.15, 0.70])
    ax.set_title(v["title"], color='#38bdf8', fontsize=11, fontweight='bold', pad=18)
    ax.axis('off')

plt.suptitle("Vistas Cinemáticas a Ojo de Jugador sobre Tablero Catan (Altura Máxima 83.00 mm)",
             color='#e2e8f0', fontsize=15, fontweight='bold', y=0.96)
plt.tight_layout(rect=[0, 0, 1, 0.92])

out_eyelevel = os.path.join(out_dir, "render_mountain_tabletop_eyelevel.png")
plt.savefig(out_eyelevel, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy(out_eyelevel, os.path.join(brain_dir, "render_mountain_tabletop_eyelevel.png"))
print(f"Saved: {out_eyelevel} and copied to brain")
print(f"Tabletop render finished in {time.time()-t0:.2f}s.")
