import os
import math
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print("=== FDM SLICING SIMULATION & TOOLPATH OPTIMIZATION (5.11M SOLID) ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"
os.makedirs(out_dir, exist_ok=True)

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
Z_BASE = 25.0
Z_PEAK = 83.0

Z_master = np.load('scratch/Z_top_master_2m.npy')
ny, nx = Z_master.shape
dx = 2 * APOTHEM / (nx - 1)
dy = 2 * R / (ny - 1)
dA = dx * dy # mm2 per pixel

# Hexagon area
hex_area = 3 * math.sqrt(3) / 2 * (R ** 2) # 10641.72 mm2

# 1. Compute layer-by-layer cross-sectional area A(z) from Z=0 to Z=83 mm
layer_heights = [0.08, 0.12, 0.16, 0.20]
slicing_data = {}

for lh in layer_heights:
    layers_z = np.arange(lh, Z_PEAK + lh/2.0, lh)
    areas = []
    perimeters = []
    
    # Precompute mask for hexagon base (Z in [0, 25.0])
    # Inside hexagon, base has magnet cavity of diam 10.2mm (area pi * 5.1^2 = 81.71 mm2)
    mag_area = math.pi * (5.1 ** 2)
    
    for z in layers_z:
        if z <= 5.2:
            # Base with magnet cavity
            area_z = hex_area - mag_area
            perim_z = 6 * R + 2 * math.pi * 5.1
        elif z <= Z_BASE:
            # Solid base zocalo
            area_z = hex_area
            perim_z = 6 * R
        else:
            # Mountain terrain above z_base
            mask_solid = Z_master >= z
            area_z = np.sum(mask_solid) * dA
            # Approximate perimeter via edge pixels
            import scipy.ndimage as ndi
            eroded = ndi.binary_erosion(mask_solid)
            edge_count = np.sum(mask_solid ^ eroded)
            perim_z = edge_count * math.sqrt(dx * dy)
            
        areas.append(area_z)
        perimeters.append(perim_z)
        
    areas = np.array(areas)
    perimeters = np.array(perimeters)
    
    # Volume calculation via trapezoidal integration
    total_vol_mm3 = np.sum(areas * lh)
    total_vol_cm3 = total_vol_mm3 / 1000.0
    
    # Mass at 100% solid PLA (density = 1.24 g/cm3)
    mass_100_solid_g = total_vol_cm3 * 1.24
    
    # Practical FDM Infill: 4 walls (0.42mm nozzle -> 1.68mm shell), 5 top/bottom layers (0.8mm solid skin), 15% gyroid infill core
    # Shell volume ~ 45% of total volume, core infill ~ 15% of core volume (55%)
    # Effective density ~ 0.45 * 1.0 + 0.55 * 0.15 = 0.5325 (53.25% solid)
    effective_infill_ratio = 0.5325
    practical_mass_g = mass_100_solid_g * effective_infill_ratio
    filament_length_m = (practical_mass_g / (1.24 * math.pi * (0.875**2) / 1000.0)) / 1000.0
    
    # Print speed estimation (Bambu X1C / P1S profile)
    # Average volumetric flow rate: 16 mm3/s for PLA High Flow
    # Extrusion time = (practical_mass_g / 1.24 * 1000) / 16.0 seconds
    extrusion_time_s = (total_vol_mm3 * effective_infill_ratio) / 16.0
    # Add travel, z-hops, layer changes (~0.8s per layer)
    overhead_time_s = len(layers_z) * 0.85
    total_time_min = (extrusion_time_s + overhead_time_s) / 60.0
    
    slicing_data[lh] = {
        'layers_z': layers_z,
        'num_layers': len(layers_z),
        'areas': areas,
        'perimeters': perimeters,
        'volume_cm3': total_vol_cm3,
        'mass_solid_g': mass_100_solid_g,
        'mass_practical_g': practical_mass_g,
        'filament_m': filament_length_m,
        'print_time_min': total_time_min
    }
    print(f"Layer Height {lh:.2f} mm: {len(layers_z)} layers | Vol: {total_vol_cm3:.2f} cm3 | Practical Mass: {practical_mass_g:.1f} g | Time: {total_time_min/60.0:.2f} hrs ({total_time_min:.0f} min)")

# 2. Generate comprehensive slicing engineering diagram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#090d16')
ax1.set_facecolor('#0f172a')
ax2.set_facecolor('#0f172a')

# Subplot 1: Area profile A(z) and Perimeter P(z)
d016 = slicing_data[0.16]
ax1.plot(d016['areas'], d016['layers_z'], color='#38bdf8', lw=2.2, label='Área de Sección A(Z) [mm²]')
ax1.fill_betweenx(d016['layers_z'], 0, d016['areas'], color='#0284c7', alpha=0.25)
ax1.axhline(25.0, color='#f59e0b', linestyle='--', lw=1.5, label='Zócalo Base / Carreteras (Z = 25.0 mm)')
ax1.axhline(5.2, color='#f43f5e', linestyle=':', lw=1.5, label='Techo Imán Ø10.2 (Z = 5.2 mm)')
ax1.set_xlabel("Área Transversal (mm²)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax1.set_ylabel("Cota Z de Impresión (mm)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax1.set_title("Perfil de Sección Transversal Capa a Capa (Z=0 a 83 mm)", color='#38bdf8', fontsize=12, fontweight='bold')
ax1.tick_params(colors='#94a3b8')
ax1.grid(True, color='#334155', linestyle=':', alpha=0.6)
ax1.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', loc='center right')

# Subplot 2: Comparison of Layer Heights (Print Time vs Resolution)
lhs = [f"{lh:.2f} mm" for lh in layer_heights]
times_hrs = [slicing_data[lh]['print_time_min'] / 60.0 for lh in layer_heights]
layers_cnt = [slicing_data[lh]['num_layers'] for lh in layer_heights]

colors_bars = ['#a855f7', '#38bdf8', '#34d399', '#f59e0b']
bars = ax2.bar(lhs, times_hrs, color=colors_bars, edgecolor='#cbd5e1', lw=1.2, width=0.55)
ax2.set_ylabel("Tiempo Estimado de Impresión FDM (Horas)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax2.set_xlabel("Altura de Capa (Layer Height)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax2.set_title("Benchmark de Laminación FDM (Bambu Lab X1C / P1S High-Flow)", color='#38bdf8', fontsize=12, fontweight='bold')
ax2.tick_params(colors='#94a3b8')
ax2.grid(True, color='#334155', linestyle=':', alpha=0.6, axis='y')

for bar, n_l, t_h in zip(bars, layers_cnt, times_hrs):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, h + 0.15, f"{t_h:.1f}h\n({n_l} capas)",
             ha='center', va='bottom', color='#e2e8f0', fontsize=10, fontweight='bold')

out_slice_img = os.path.join(out_dir, "render_slicing_analysis_fdm.png")
plt.tight_layout()
plt.savefig(out_slice_img, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy(out_slice_img, os.path.join(brain_dir, "render_slicing_analysis_fdm.png"))
print(f"Saved: {out_slice_img} and copied to brain")
print(f"Slicing simulation finished in {time.time()-t0:.2f}s.")
