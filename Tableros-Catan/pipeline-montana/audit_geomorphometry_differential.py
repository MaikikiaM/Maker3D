import os
import math
import time
import shutil
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from PIL import Image

print("=== DIFFERENTIAL GEOMETRY & CURVATURE AUDIT (GAUSSIAN & MEAN CURVATURE) ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"
os.makedirs(out_dir, exist_ok=True)

R = 64.0
APOTHEM = R * math.cos(math.radians(30))

Z_master = np.load('scratch/Z_top_master_2m.npy')
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

# First derivatives
zy, zx = np.gradient(Z, dy, dx)

# Second derivatives
zyy, zyx = np.gradient(zy, dy, dx)
zxy, zxx = np.gradient(zx, dy, dx)

# Differential Geometry of Monge Patch z = f(x, y):
# Metric tensor (First fundamental form):
# E = 1 + zx^2, F = zx*zy, G = 1 + zy^2
# Normal vector: N = (-zx, -zy, 1) / sqrt(1 + zx^2 + zy^2)
denom = np.sqrt(1.0 + zx**2 + zy**2)
denom4 = (1.0 + zx**2 + zy**2)**2

# Second fundamental form:
# L = zxx / denom, M = zxy / denom, N = zyy / denom
# Gaussian Curvature K = (L*N - M^2) / (E*G - F^2) = (zxx*zyy - zxy^2) / (1 + zx^2 + zy^2)^2
K_gauss = (zxx * zyy - zxy**2) / denom4

# Mean Curvature H = (E*N - 2*F*M + G*L) / (2 * (E*G - F^2))
H_mean = ((1.0 + zx**2) * zyy - 2.0 * zx * zy * zxy + (1.0 + zy**2) * zxx) / (2.0 * (denom**3))

# Hexagon mask
normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

inside_hex = dist_edge >= 0.5

K_plot = np.where(inside_hex, np.clip(K_gauss, -0.05, 0.05), np.nan)
H_plot = np.where(inside_hex, np.clip(H_mean, -0.25, 0.25), np.nan)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), facecolor='#090d16')
ax1.set_facecolor('#0f172a')
ax2.set_facecolor('#0f172a')

# 1. Gaussian Curvature Plot
im1 = ax1.imshow(K_plot, origin='lower', cmap='coolwarm', vmin=-0.04, vmax=0.04)
ax1.set_title("Curvatura Gaussiana K (mm⁻²)\n[K > 0: Picos/Cúspides | K < 0: Collados/Sillas de Montar]",
              color='#38bdf8', fontsize=11.5, fontweight='bold', pad=10)
ax1.axis('off')
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label("Curvatura Gaussiana K", color='#e2e8f0', fontsize=10)
cbar1.ax.tick_params(colors='#94a3b8')

# 2. Mean Curvature Plot
im2 = ax2.imshow(H_plot, origin='lower', cmap='seismic', vmin=-0.15, vmax=0.15)
ax2.set_title("Curvatura Media H (mm⁻¹)\n[H > 0: Aristas/Crestas Afiladas | H < 0: Circos/Canales Couloirs]",
              color='#38bdf8', fontsize=11.5, fontweight='bold', pad=10)
ax2.axis('off')
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("Curvatura Media H", color='#e2e8f0', fontsize=10)
cbar2.ax.tick_params(colors='#94a3b8')

plt.suptitle("Auditoría de Geometría Diferencial de Superficie Monge: Curvaturas Intrínsicas\n(Confirmación de Aristas y Circos Glaciares Naturales)",
             color='#e2e8f0', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

out_curv = os.path.join(out_dir, "render_geomorphometry_curvatures.png")
plt.savefig(out_curv, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy(out_curv, os.path.join(brain_dir, "render_geomorphometry_curvatures.png"))
print(f"Saved: {out_curv} and copied to brain")
print(f"Differential geometry audit finished in {time.time()-t0:.2f}s.")
