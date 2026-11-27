import os
import math
import time
import shutil
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from PIL import Image

print("=== GENERATING 6-PLAYER BOARD PERSPECTIVES (HEX EDGE EYE-LEVEL) ===")
t0 = time.time()

out_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

# Load diffuse texture and Z
diffuse_4k = np.load('scratch/diffuse_4k_array.npy')
dim_tex = diffuse_4k.shape[0]

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

im_diff = Image.fromarray((diffuse_4k * 255.0).astype(np.uint8))
im_diff_res = im_diff.resize((res, res), Image.Resampling.LANCZOS)
diffuse_res = np.array(im_diff_res).astype(np.float32) / 255.0

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

lit = diffuse_res * (
    diff_sun[:, :, None] * np.array([1.06, 1.00, 0.92]) * 0.72 +
    diff_sky[:, :, None] * np.array([0.70, 0.84, 0.98]) * 0.35 +
    np.array([0.22, 0.25, 0.32]) * ao[:, :, None] * 0.42
)
lit[dist_edge < 0] = [0.06, 0.08, 0.12]
lit = np.clip(lit, 0.0, 1.0)

# Prepare surface coordinates
step = 2
surf_x = X[::step, ::step]
surf_y = Y[::step, ::step]
surf_z = Z[::step, ::step]
dist_sub = dist_edge[::step, ::step]

rgba = np.zeros((surf_x.shape[0], surf_x.shape[1], 4))
rgba[:, :, :3] = lit[::step, ::step]
rgba[:, :, 3] = 1.0
rgba[dist_sub < -0.2, 3] = 0.0

# 6 Player Perspectives (0, 60, 120, 180, 240, 300 degrees around hex)
angles = [0, 60, 120, 180, 240, 300]
player_titles = [
    "Jugador 1 (Este - Espolón Catedral)",
    "Jugador 2 (Noreste - Arista Hörnli)",
    "Jugador 3 (Noroeste - Cumbre Matterhorn 83mm)",
    "Jugador 4 (Oeste - Espolón Sombrío)",
    "Jugador 5 (Suroeste - Diente Sur)",
    "Jugador 6 (Sureste - Pradera Tundra y Talud)"
]

fig = plt.figure(figsize=(18, 12), facecolor='#090d16')

for idx, (ang, title) in enumerate(zip(angles, player_titles)):
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d', facecolor='#090d16')
    ax.plot_surface(surf_x, surf_y, surf_z, facecolors=rgba, rstride=1, cstride=1, shade=False, antialiased=True)
    ax.view_init(elev=26, azim=ang)
    ax.set_box_aspect([1.0, 1.15, 0.70])
    ax.set_title(title, color='#38bdf8', fontsize=10.5, fontweight='bold', pad=8)
    ax.axis('off')

plt.suptitle("PERSPECTIVAS DESDE LOS 6 LADOS DEL TABLERO DE CATAN (FULL-COLOR INFINITO)",
             color='#38bdf8', fontsize=14, fontweight='bold', y=0.98)
out_6p = os.path.join(out_dir, "render_mountain_6_player_perspectives.png")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(out_6p, dpi=180, bbox_inches='tight', facecolor='#090d16')
plt.close()
print(f"Saved 6-Player Perspectives: {out_6p}")

shutil.copy(out_6p, os.path.join(brain_dir, os.path.basename(out_6p)))
print(f"Total time: {time.time()-t0:.2f}s")
