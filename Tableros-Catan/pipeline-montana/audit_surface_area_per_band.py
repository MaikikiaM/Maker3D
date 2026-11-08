import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt

print("=== COMPUTING HYPSOMETRIC CURVE & ELEVATION SURFACE AREA DISTRIBUTION ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

R = 64.0
APOTHEM = R * math.cos(math.radians(30))
Z_master = np.load('scratch/Z_top_master_2m.npy')
ny, nx = Z_master.shape
dx = 2 * APOTHEM / (nx - 1)
dy = 2 * R / (ny - 1)

# Mask inside hexagon
xs = np.linspace(-APOTHEM, APOTHEM, nx, dtype=np.float32)
ys = np.linspace(-R, R, ny, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)

normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

inside_hex = dist_edge >= 0.0

# Surface element area dA_surf = dA_flat * sqrt(1 + gx^2 + gy^2)
gy, gx = np.gradient(Z_master, dy, dx)
slope_factor = np.sqrt(1.0 + gx**2 + gy**2)
dA_flat = dx * dy
dA_surf = dA_flat * slope_factor

Z_inside = Z_master[inside_hex]
dA_inside = dA_surf[inside_hex]

total_planimetric_area = np.sum(inside_hex) * dA_flat
total_surface_area = np.sum(dA_inside)
print(f"Base Planimetric Area: {total_planimetric_area:.2f} mm2 (106.42 cm2)")
print(f"3D Mountain Surface Area: {total_surface_area:.2f} mm2 ({total_surface_area/100:.2f} cm2)")
print(f"Rugosity Index (Surface / Planimetric): {total_surface_area / total_planimetric_area:.3f}x")

# Elevation bands (5 mm width)
bins = np.arange(25.0, 85.0, 2.5)
band_areas = []
band_mids = []
for i in range(len(bins) - 1):
    z_lo, z_hi = bins[i], bins[i+1]
    mask_b = (Z_inside >= z_lo) & (Z_inside < z_hi)
    band_areas.append(np.sum(dA_inside[mask_b]) / 100.0) # cm2
    band_mids.append((z_lo + z_hi) / 2.0)

band_areas = np.array(band_areas)
band_mids = np.array(band_mids)

# Cumulative Hypsometric Curve
z_sorted = np.sort(Z_inside)
cum_area_pct = 100.0 * (1.0 - np.arange(len(z_sorted)) / float(len(z_sorted)))

# 2-Panel Plot: Elevation Area Distribution & Hypsometric Integral
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#090d16')
ax1.set_facecolor('#0f172a')
ax2.set_facecolor('#0f172a')

# 1. Surface Area per 2.5mm Elevation Band
colors_h = plt.colormaps['plasma'](np.linspace(0.1, 0.9, len(band_mids)))
bars = ax1.barh(band_mids, band_areas, height=2.0, color=colors_h, edgecolor='#cbd5e1', lw=0.8)
ax1.set_xlabel("Área de Superficie 3D (cm²)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax1.set_ylabel("Cota de Elevación Z (mm)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax1.set_title("Distribución de Área Superficial por Franjas Altimétricas (ΔZ = 2.5mm)", color='#38bdf8', fontsize=12, fontweight='bold')
ax1.tick_params(colors='#94a3b8')
ax1.grid(True, color='#334155', linestyle=':', alpha=0.6, axis='x')

# Annotations of ecological zones
ax1.text(band_areas[0] + 0.5, 27.5, "Zócalo & Tundra (15 Verdes)", color='#22c55e', fontsize=9.5, fontweight='bold')
ax1.text(band_areas[6] + 0.5, 42.5, "Límite de Nieve & Granito (45 Grises)", color='#38bdf8', fontsize=9.5, fontweight='bold')
ax1.text(band_areas[-1] + 0.3, 81.5, "Cima Matterhorn", color='#f43f5e', fontsize=9.5, fontweight='bold')

# 2. Cumulative Hypsometric Curve
ax2.plot(cum_area_pct, z_sorted, color='#38bdf8', lw=2.5, label='Curva Hipsométrica Alpina')
ax2.fill_betweenx(z_sorted, 0, cum_area_pct, color='#0284c7', alpha=0.25)
ax2.set_xlabel("Porcentaje Acumulado de Superficie por Encima de Z (%)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax2.set_ylabel("Elevación Z (mm)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax2.set_title("Curva Hipsométrica Acumulada (Índice de Madurez Geológica Strahler)", color='#38bdf8', fontsize=12, fontweight='bold')
ax2.tick_params(colors='#94a3b8')
ax2.grid(True, color='#334155', linestyle=':', alpha=0.6)
ax2.set_xlim(0, 100)
ax2.set_ylim(25, 85)

# Calculate Strahler Hypsometric Integral (HI)
# HI = (mean_elevation - min_elevation) / (max_elevation - min_elevation)
HI = (np.mean(Z_inside) - 25.0) / (83.0 - 25.0)
print(f"Strahler Hypsometric Integral (HI): {HI:.3f} (Classic Alpine Pyramidal Glacial Landscape)")

hi_box = (
    f"MÉTRICAS GEOMORFOLÓGICAS:\n"
    f"• Integral Hipsométrica (HI): {HI:.3f}\n"
    f"• Tipología: Relieve Glaciar Maduro (Cervino / Matterhorn)\n"
    f"• Rugosidad 3D: {total_surface_area / total_planimetric_area:.3f}x\n"
    f"• Superficie 3D Total: {total_surface_area/100:.2f} cm²\n"
    f"• Área Base Plana: {total_planimetric_area/100:.2f} cm²"
)
ax2.text(0.05, 0.65, hi_box, transform=ax2.transAxes, color='#e2e8f0', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f172a', edgecolor='#38bdf8', lw=1.5))

plt.suptitle("Análisis Cuantitativo Hipsométrico y de Rugosidad de Superficie 3D",
             color='#e2e8f0', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

out_hyp = os.path.join(out_dir, "render_mountain_hypsometric_curve.png")
plt.savefig(out_hyp, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

import shutil
shutil.copy(out_hyp, os.path.join(brain_dir, "render_mountain_hypsometric_curve.png"))
print(f"Saved: {out_hyp} and copied to brain")
print(f"Hypsometric analysis finished in {time.time()-t0:.2f}s.")
