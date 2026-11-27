import os
import math
import time
import shutil
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from PIL import Image

print("=== GENERATING HIGH-PRECISION TOPOGRAPHICAL SURVEY MAP WITH 2MM ISOLINES ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

R = 64.0
APOTHEM = R * math.cos(math.radians(30))

Z_master = np.load('scratch/Z_top_master_2m.npy')
diffuse_4k = np.load('scratch/diffuse_4k_array.npy')
ny_m, nx_m = Z_master.shape

res = 2048
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

# Normals & Hillshade
gy, gx = np.gradient(Z, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm; Ny = -gy / N_norm; Nz = 1.0 / N_norm

az_sun = 315.0 * math.pi / 180.0; alt_sun = 45.0 * math.pi / 180.0
Lx = math.cos(alt_sun) * math.sin(az_sun); Ly = math.cos(alt_sun) * math.cos(az_sun); Lz = math.sin(alt_sun)
ndotl = np.clip(Nx * Lx + Ny * Ly + Nz * Lz, 0.0, 1.0)

lit = diffuse_res * (ndotl[:, :, None] * 0.70 + 0.35)

normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))
lit[dist_edge < 0] = [0.06, 0.08, 0.12]
lit = np.clip(lit, 0.0, 1.0)

fig, ax = plt.subplots(figsize=(12, 12), facecolor='#090d16')
ax.set_facecolor('#090d16')

# Base hillshade
ax.imshow(lit, extent=[-APOTHEM, APOTHEM, -R, R], origin='lower')

# Standard Contours (every 2.0 mm)
levels_fine = np.arange(26.0, 83.0, 2.0)
cs_fine = ax.contour(X, Y, Z, levels=levels_fine, colors=['#38bdf8'], linewidths=[0.6], alpha=0.55)

# Index Contours (every 10.0 mm)
levels_index = np.arange(30.0, 85.0, 10.0)
cs_index = ax.contour(X, Y, Z, levels=levels_index, colors=['#ffffff'], linewidths=[1.4], alpha=0.9)
ax.clabel(cs_index, inline=True, fmt='Z=%1.0f mm', fontsize=8.5, colors='#ffffff')

# Outer hex border
v_angles = (30 + 60 * np.arange(7)) * np.pi / 180.0
ax.plot(R * np.cos(v_angles), R * np.sin(v_angles), color='#38bdf8', lw=2.5)

# Summit marker
apex_idx = np.unravel_index(np.argmax(Z), Z.shape)
ax.plot(X[apex_idx], Y[apex_idx], marker='^', color='#f43f5e', markersize=11, markeredgecolor='#ffffff', markeredgewidth=1.5)
ax.text(X[apex_idx] + 3.0, Y[apex_idx] + 2.0, f"Cima Matterhorn\nZ = 83.00 mm", color='#f8fafc', fontsize=9.5, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor='#f43f5e', lw=1.2))

ax.set_xlim(-APOTHEM - 3, APOTHEM + 3)
ax.set_ylim(-R - 3, R + 3)
ax.set_aspect('equal')
ax.set_title("Carta Topográfica Cartográfica: Curvas de Nivel Cada 2.0 mm\n(Equidistancia: 2.0 mm / Curvas Maestras: 10.0 mm / Relieve Alpino: 58.00 mm)",
             color='#38bdf8', fontsize=12.5, fontweight='bold', pad=12)
ax.axis('off')

plt.tight_layout()
out_topo = os.path.join(out_dir, "render_topo_contour_map.png")
plt.savefig(out_topo, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy(out_topo, os.path.join(brain_dir, "render_topo_contour_map.png"))
print(f"Saved: {out_topo} and copied to brain")
print(f"Topographical survey finished in {time.time()-t0:.2f}s.")
