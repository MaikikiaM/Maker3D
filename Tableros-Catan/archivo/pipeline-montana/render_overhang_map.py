import os
import math
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print("=== GENERATING 3D PRINTABILITY OVERHANG MAP ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

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

gy, gx = np.gradient(Z, dy, dx)
slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

inside_hex = dist_edge >= 0.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), facecolor='#090d16')
ax1.set_facecolor('#0f172a')
ax2.set_facecolor('#0f172a')

# 2D Slope Heatmap
cmap = plt.colormaps['magma'].copy()
slope_display = np.where(inside_hex, slope_deg, np.nan)
im = ax1.imshow(slope_display, origin='lower', cmap=cmap, vmin=0, vmax=75)
ax1.set_title("Mapa de Pendientes de la Montaña (0° a 75°+)", color='#38bdf8', fontsize=12, fontweight='bold', pad=10)
ax1.axis('off')
cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Pendiente (°)", color='#e2e8f0', fontsize=10, fontweight='bold')
cbar.ax.tick_params(colors='#94a3b8')

# Slope Category Histogram & Supportless Printability Analysis
slopes_valid = slope_deg[inside_hex]
bins = [0, 30, 45, 60, 90]
labels = [
    f"< 30°: Valles y Laderas Suaves\n({np.mean(slopes_valid < 30)*100:.1f}% de la superficie)",
    f"30°-45°: Derrubios y Conos\n({np.mean((slopes_valid>=30)&(slopes_valid<45))*100:.1f}%)",
    f"45°-60°: Contrafuertes Rocosos\n({np.mean((slopes_valid>=45)&(slopes_valid<60))*100:.1f}%)",
    f"> 60°: Paredes de Roca Escarpada\n({np.mean(slopes_valid >= 60)*100:.1f}%)"
]
counts = [
    np.sum(slopes_valid < 30),
    np.sum((slopes_valid >= 30) & (slopes_valid < 45)),
    np.sum((slopes_valid >= 45) & (slopes_valid < 60)),
    np.sum(slopes_valid >= 60)
]
colors_cat = ['#10b981', '#38bdf8', '#f59e0b', '#f43f5e']

ax2.barh(labels, counts, color=colors_cat, edgecolor='#cbd5e1', lw=1.2, height=0.55)
ax2.set_title("Distribución de Imprimibilidad FDM sin Soportes\n(Geometría Cónica Naturalmente Auto-Soportada)", color='#38bdf8', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel("Número de Píxeles en Superficie", color='#e2e8f0', fontsize=10, fontweight='bold')
ax2.tick_params(colors='#94a3b8', labelsize=9.5)
ax2.grid(True, color='#334155', linestyle=':', alpha=0.6, axis='x')

plt.suptitle("Análisis CAD de Imprimibilidad y Pendientes Críticas de Fabricación Aditiva",
             color='#e2e8f0', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

out_overhang = os.path.join(out_dir, "render_overhang_printability_map.png")
plt.savefig(out_overhang, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy(out_overhang, os.path.join(brain_dir, "render_overhang_printability_map.png"))
print(f"Saved: {out_overhang} and copied to brain")
print(f"Overhang map finished in {time.time()-t0:.2f}s.")
