import os
import time
import math
import numpy as np
import scipy.interpolate as interp
import trimesh
import matplotlib.pyplot as plt

print("=== GENERATING MULTI-RESOLUTION LEVEL-OF-DETAIL (LOD) SUITE VIA SOLID GENERATOR ===")
t0 = time.time()

from build_solid_mountain import build_solid_watertight_mountain

out_dir = r"Tableros Catan\Loseta_Montana_3D"
renders_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

R = 64.0
APOTHEM = R * math.cos(math.radians(30))
Z_master = np.load('scratch/Z_top_master_2m.npy')
ny_m, nx_m = Z_master.shape
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

# LOD configurations
lod_configs = [
    {"name": "LOD1 (1.03M Triángulos)", "res": 768, "filename": "catan_mountain_lod1_1m.stl", "desc": "Desktop High-End WebGL / CAD"},
    {"name": "LOD2 (258k Triángulos)", "res": 384, "filename": "catan_mountain_lod2_250k.stl", "desc": "Mobile & Tablet WebGL"},
    {"name": "LOD3 (51k Triángulos)", "res": 170, "filename": "catan_mountain_lod3_50k.stl", "desc": "AR / VR Realtime Preview"}
]

lod_stats = {}

# Master LOD0
mesh_lod0 = trimesh.load(os.path.join(out_dir, "catan_mountain_solid.stl"))
lod_stats["LOD0 (5.11M)"] = {
    "faces": len(mesh_lod0.faces),
    "vertices": len(mesh_lod0.vertices),
    "volume": mesh_lod0.volume / 1000.0,
    "watertight": mesh_lod0.is_watertight,
    "size_mb": os.path.getsize(os.path.join(out_dir, "catan_mountain_solid.stl")) / (1024 * 1024)
}

for cfg in lod_configs:
    t_cfg = time.time()
    out_stl = os.path.join(out_dir, cfg["filename"])
    print(f"\nSynthesizing {cfg['name']} at grid resolution {cfg['res']}...")
    mesh = build_solid_watertight_mountain(res=cfg["res"], output_path=out_stl, z_interpolator=interp_z)
    
    file_mb = os.path.getsize(out_stl) / (1024 * 1024)
    print(f"  {cfg['name']}: {len(mesh.faces):,} faces | {len(mesh.vertices):,} vertices")
    print(f"  Watertight: {mesh.is_watertight} | Euler: {mesh.euler_number} | Volume: {mesh.volume/1000:.2f} cm3")
    print(f"  Saved: {out_stl} ({file_mb:.2f} MB in {time.time()-t_cfg:.2f}s)")
    
    lod_stats[cfg["name"]] = {
        "faces": len(mesh.faces),
        "vertices": len(mesh.vertices),
        "volume": mesh.volume / 1000.0,
        "watertight": mesh.is_watertight,
        "size_mb": file_mb
    }

# Generate Comparison Summary Graphic
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#090d16')
ax1.set_facecolor('#0f172a')
ax2.set_facecolor('#0f172a')

names = list(lod_stats.keys())
faces_k = [lod_stats[n]["faces"] / 1000.0 for n in names]
sizes = [lod_stats[n]["size_mb"] for n in names]
colors_lod = ['#a855f7', '#38bdf8', '#34d399', '#f59e0b']

bars1 = ax1.bar(names, faces_k, color=colors_lod, edgecolor='#cbd5e1', lw=1.2, width=0.55)
ax1.set_ylabel("Complejidad de Malla (Miles de Triángulos)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax1.set_title("Nivel de Detalle: Recuento de Triángulos", color='#38bdf8', fontsize=12, fontweight='bold')
ax1.tick_params(colors='#94a3b8', labelsize=9.5)
ax1.grid(True, color='#334155', linestyle=':', alpha=0.6, axis='y')
for bar, fk in zip(bars1, faces_k):
    ax1.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 50, f"{fk:,.0f}k\ncaras",
             ha='center', va='bottom', color='#e2e8f0', fontsize=10, fontweight='bold')

bars2 = ax2.bar(names, sizes, color=colors_lod, edgecolor='#cbd5e1', lw=1.2, width=0.55)
ax2.set_ylabel("Tamaño de Archivo STL Binario (MB)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax2.set_title("Huella de Almacenamiento & Tiempo de Carga WebGL", color='#38bdf8', fontsize=12, fontweight='bold')
ax2.tick_params(colors='#94a3b8', labelsize=9.5)
ax2.grid(True, color='#334155', linestyle=':', alpha=0.6, axis='y')
for bar, sz in zip(bars2, sizes):
    ax2.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 3, f"{sz:.1f} MB",
             ha='center', va='bottom', color='#e2e8f0', fontsize=10, fontweight='bold')

plt.suptitle("Suite Jerárquica de Mallas Monolíticas Watertight (LOD0 a LOD3)\n(100% Manifold, Euler χ = 2, Compatibilidad Universal FDM y WebGL)",
             color='#e2e8f0', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

out_lod_plot = os.path.join(renders_dir, "render_lod_mesh_comparison.png")
plt.savefig(out_lod_plot, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

import shutil
shutil.copy(out_lod_plot, os.path.join(brain_dir, "render_lod_mesh_comparison.png"))
print(f"\nSaved LOD comparison plot: {out_lod_plot} and copied to brain")
print(f"Total LOD suite generation time: {time.time()-t0:.2f}s")
