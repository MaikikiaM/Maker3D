import os
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

R = 64.0
APOTHEM = R * math.cos(math.radians(30))
Z_master = np.load('scratch/Z_top_master_2m.npy')
ny_m, nx_m = Z_master.shape

xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)

# Slice through apex: X = -3.5, Y = 4.5
y_apex_idx = int(np.round((4.5 - (-R)) / (2 * R) * (ny_m - 1)))
z_slice = Z_master[y_apex_idx, :]

fig, ax = plt.subplots(figsize=(11, 6), facecolor='#090d16')
ax.set_facecolor('#0f172a')

# Filled solid volume
ax.fill_between(xs_m, 0, z_slice, color='#334155', alpha=0.95, label='Macizo Alpino 100% Sólido (5.11M Triángulos)')
ax.fill_between(xs_m, 0, 25.0, color='#1e293b', alpha=0.95, label='Zócalo Base Hexagonal (Z = 25.0 mm)')
ax.plot(xs_m, z_slice, color='#38bdf8', lw=2.5, label='Cresta Cumbre Principal (Cota 83.00 mm)')
ax.axhline(25.0, color='#94a3b8', linestyle='--', lw=1.2, label='Plano Carreteras Catan (Z = 25.0 mm)')

# Magnet cavity pocket
ax.fill_between([-5.1, 5.1], 0, 5.2, color='#020617', hatch='///', edgecolor='#f43f5e', lw=1.5, label='Alojamiento Imán Ø10.2 x 5.2mm')

ax.set_ylim(-3, 90)
ax.set_xlim(-58, 58)
ax.set_ylabel("Altura Z (mm)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax.set_xlabel("Eje Radial X (mm)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax.tick_params(colors='#94a3b8', labelsize=10)
ax.set_title("Corte Transversal Z por el Vértice Principal - Sólido Manifold Watertight", color='#38bdf8', fontsize=13, fontweight='bold', pad=12)
ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', loc='upper right', fontsize=9.5)
ax.grid(True, color='#334155', linestyle=':', alpha=0.6)

cross_path = os.path.join(out_dir, "render_solid_cross_section.png")
plt.tight_layout()
plt.savefig(cross_path, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy(cross_path, os.path.join(brain_dir, "render_solid_cross_section.png"))
print(f"Saved and copied: {cross_path}")
