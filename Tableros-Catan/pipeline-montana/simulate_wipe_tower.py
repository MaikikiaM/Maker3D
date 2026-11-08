import os
import math
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

print("=== MULTI-MATERIAL FDM PURGE & FULL-COLOR 3MF MANUFACTURING AUDIT ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

# Multi-Material Technology Comparison:
# 1. True Full-Color Voxel Jetting (PolyJet / Mimaki 3DUJ / HP Multi Jet Fusion Color):
#    - Native format: 3MF / VRML / OBJ+MTL
#    - Color capability: 10+ Million colors (24-bit TrueColor)
#    - Purge waste: 0.00 grams (piezoelectric drop-on-demand inkjet nozzles)
#    - Print time: Constant per build tray (~3.5 hours for full bed)
#
# 2. Multi-Filament FDM with Purge Tower (Bambu AMS / Prusa MMU3 / Infimech):
#    - Native format: STL / 3MF multi-part
#    - Color capability: 4 to 8 discrete spools
#    - Purge waste: 150 - 350 mm3 per filament change
#    - Total purge mass: ~180 - 450 g (often exceeds model weight!)

Z_master = np.load('scratch/Z_top_master_2m.npy')
layer_h = 0.16 # mm
layers_z = np.arange(layer_h, 83.0, layer_h)

# Layer-by-layer material activity:
# Z < 25mm: Base (Dark Grey / Obsidian) -> 1 material (0 filament changes)
# 25mm <= Z < 32mm: Base + Tundra Green + Slate Rock -> 3 materials per layer (~2 changes/layer)
# 32mm <= Z < 42mm: Rock (Basalt + Granite + Quartz) -> 3 materials per layer (~2 changes/layer)
# 42mm <= Z < 83mm: Rock + Snow -> 2 materials per layer (1 change/layer)

changes_per_layer = []
for z in layers_z:
    if z < 25.0:
        c = 0
    elif z < 32.0:
        c = 2 # Green, Rock, Base
    elif z < 42.0:
        c = 2 # Rock tones
    else:
        c = 1 # Snow + Bare Cliff Rock
    changes_per_layer.append(c)

changes_per_layer = np.array(changes_per_layer)
total_changes = np.sum(changes_per_layer)
print(f"Total FDM Filament Changes: {total_changes:,}")

# Purge volume at standard 220 mm3 per swap
purge_vol_mm3 = total_changes * 220.0
purge_mass_g = (purge_vol_mm3 / 1000.0) * 1.24 # PLA density 1.24
print(f"Estimated FDM Purge Waste: {purge_mass_g:.1f} g")

# Comparison graphic: PolyJet 3MF vs Multi-Filament FDM
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#090d16')
ax1.set_facecolor('#0f172a')
ax2.set_facecolor('#0f172a')

# 1. Filament swaps vs Height Z
ax1.plot(changes_per_layer, layers_z, color='#38bdf8', lw=2.0, label='Cambios de Filamento por Capa')
ax1.fill_betweenx(layers_z, 0, changes_per_layer, color='#0284c7', alpha=0.3)
ax1.axhline(25.0, color='#f59e0b', linestyle='--', label='Zócalo Base (Cero Cambios / Monocolor)')
ax1.axhline(42.0, color='#ffffff', linestyle=':', label='Línea de Nieve (Roca + Nieve / 1 Cambio)')
ax1.set_xlabel("Número de Cambios de Herramienta por Capa", color='#e2e8f0', fontsize=11, fontweight='bold')
ax1.set_ylabel("Cota Z de Impresión (mm)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax1.set_title("Distribución Vertical de Cambios de Color en FDM", color='#38bdf8', fontsize=12, fontweight='bold')
ax1.tick_params(colors='#94a3b8')
ax1.grid(True, color='#334155', linestyle=':', alpha=0.6)
ax1.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', loc='upper right')

# 2. Material Waste & Efficiency Comparison
techs = [
    "PolyJet / Inkjet 3MF\n(Stratasys / Mimaki)",
    "FDM AMS / MMU3\n(Torre de Purga Estándar)",
    "FDM AMS / MMU3\n(Purga en Relleno Infill)",
    "FDM Monolítico\n(Pintura Manual sobre PLA)"
]
waste_g = [0.0, purge_mass_g, purge_mass_g * 0.42, 0.0]
model_g = [274.3, 274.3, 274.3, 274.3]

x_pos = np.arange(len(techs))
w_bars = 0.45

b_mod = ax2.bar(x_pos - w_bars/2, model_g, width=w_bars, label='Peso Loseta (g)', color='#38bdf8', edgecolor='#cbd5e1', lw=1.2)
b_wst = ax2.bar(x_pos + w_bars/2, waste_g, width=w_bars, label='Desperdicio Purga (g)', color='#f43f5e', edgecolor='#cbd5e1', lw=1.2)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(techs, color='#e2e8f0', fontsize=9.5)
ax2.set_ylabel("Masa de Material (Gramos)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax2.set_title("Comparativa de Eficiencia de Material por Tecnología de Impresión", color='#38bdf8', fontsize=12, fontweight='bold')
ax2.tick_params(colors='#94a3b8')
ax2.grid(True, color='#334155', linestyle=':', alpha=0.6, axis='y')
ax2.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', loc='upper left')

for bar in b_wst:
    h = bar.get_height()
    if h > 0:
        ax2.text(bar.get_x() + bar.get_width()/2.0, h + 5, f"{h:.0f} g", ha='center', va='bottom', color='#f43f5e', fontsize=10, fontweight='bold')
    else:
        ax2.text(bar.get_x() + bar.get_width()/2.0, 5, "0 g (Cero)", ha='center', va='bottom', color='#10b981', fontsize=10, fontweight='bold')

plt.suptitle("Análisis Industrial de Fabricación Aditiva Full-Color: Voxel Jetting vs Multi-Extrusión FDM",
             color='#e2e8f0', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

out_wipe = os.path.join(out_dir, "render_multi_material_wipe_tower.png")
plt.savefig(out_wipe, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy(out_wipe, os.path.join(brain_dir, "render_multi_material_wipe_tower.png"))
print(f"Saved: {out_wipe} and copied to brain")
print(f"Multi-material audit finished in {time.time()-t0:.2f}s.")
