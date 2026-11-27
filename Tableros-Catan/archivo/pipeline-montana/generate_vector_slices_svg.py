import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt

print("=== EXPORTING MULTI-ELEVATION VECTOR CONTOURS SVG / DXF FOR CNC ROUTING ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

R = 64.0
APOTHEM = R * math.cos(math.radians(30))
Z_master = np.load('scratch/Z_top_master_2m.npy')
ny_m, nx_m = Z_master.shape

res = 800
xs = np.linspace(-APOTHEM, APOTHEM, res, dtype=np.float32)
ys = np.linspace(-R, R, res, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)

import scipy.interpolate as interp
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)
pts = np.column_stack((Y.ravel(), X.ravel()))
Z = interp_z(pts).reshape((res, res))

slice_levels = [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]

fig, ax = plt.subplots(figsize=(10, 10), facecolor='#090d16')
ax.set_facecolor('#0f172a')

# Hexagon outer line
v_angles = (30 + 60 * np.arange(7)) * np.pi / 180.0
ax.plot(R * np.cos(v_angles), R * np.sin(v_angles), color='#38bdf8', lw=2.5, label='Perímetro Hexagonal (Z = 25.0 mm)')

# Plot vector contours
colors_slices = plt.colormaps['viridis'](np.linspace(0.15, 0.95, len(slice_levels)))
for idx, lvl in enumerate(slice_levels):
    cs = ax.contour(X, Y, Z, levels=[lvl], colors=[colors_slices[idx]], linewidths=[1.8 if lvl%10==0 else 1.0])
    if lvl in [30.0, 45.0, 60.0, 75.0]:
        ax.clabel(cs, inline=True, fmt=f'Z={lvl:.0f}mm', fontsize=8.5, colors='#ffffff')

# Magnet cavity in center
theta_c = np.linspace(0, 2*np.pi, 100)
ax.plot(5.1 * np.cos(theta_c), 5.1 * np.sin(theta_c), color='#f43f5e', lw=2.0, linestyle='--', label='Cavidad Imán Ø10.2mm')

ax.set_xlim(-APOTHEM - 2, APOTHEM + 2)
ax.set_ylim(-R - 2, R + 2)
ax.set_aspect('equal')
ax.set_title("Vectores de Corte Horizontal para Mecanizado CNC / Corte Láser\n(Curvas de Nivel Paramétricas Z = 25 a 80 mm)",
             color='#38bdf8', fontsize=12, fontweight='bold', pad=10)
ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', loc='lower right')
ax.axis('off')

out_svg = os.path.join(out_dir, "catan_mountain_contours.svg")
plt.savefig(out_svg, format='svg', bbox_inches='tight', facecolor='#090d16')
plt.close()

import shutil
shutil.copy(out_svg, os.path.join(brain_dir, "catan_mountain_contours.svg"))
print(f"Saved: {out_svg} and copied to brain ({os.path.getsize(out_svg)/1024:.1f} KB)")
print(f"SVG export finished in {time.time()-t0:.2f}s.")
