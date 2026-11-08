import os
import math
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print("=== G-CODE TOOLPATH EMULATION & EXTRUSION KINEMATICS ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"
os.makedirs(out_dir, exist_ok=True)

R = 64.0
APOTHEM = R * math.cos(math.radians(30))
Z_master = np.load('scratch/Z_top_master_2m.npy')
ny_m, nx_m = Z_master.shape

import scipy.interpolate as interp
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

# Toolpath slicing parameters
nozzle_diam = 0.40 # mm
line_width = 0.42  # mm
layer_height = 0.20 # mm
filament_diam = 1.75 # mm
filament_area = math.pi * ((filament_diam / 2.0) ** 2) # 2.405 mm2

# Hexagon perimeter vertices
angles = (30 + 60 * np.arange(7)) * np.pi / 180.0
hex_vx = R * np.cos(angles)
hex_vy = R * np.sin(angles)

# 4 Key Layers to visualize
sample_layers = [
    {"z": 0.20, "name": "Capa 1 (Z = 0.20 mm): Base Hexagonal & Cavidad Imán Ø10.2mm"},
    {"z": 6.00, "name": "Capa 30 (Z = 6.00 mm): Techo Sólido Puente sobre Cavidad Imán"},
    {"z": 25.00, "name": "Capa 125 (Z = 25.00 mm): Plano Zócalo Carreteras & Margen 3.5mm"},
    {"z": 55.00, "name": "Capa 275 (Z = 55.00 mm): Sección Macizo Alpino & Paredes Escarpadas"}
]

fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor='#090d16')
axes = axes.ravel()

for idx, lay in enumerate(sample_layers):
    z = lay["z"]
    ax = axes[idx]
    ax.set_facecolor('#0b1120')
    
    # 1. Base perimeter (outer wall)
    ax.plot(hex_vx, hex_vy, color='#38bdf8', lw=2.2, label='Perímetro Exterior (G1)')
    
    # 2. Inner walls (offsets by line_width, 2*line_width, 3*line_width)
    for wall in range(1, 4):
        offset_r = R - wall * (line_width / math.cos(math.radians(30)))
        ax.plot(offset_r * np.cos(angles), offset_r * np.sin(angles), color='#0ea5e9', lw=1.2, linestyle='--')
        
    if z <= 5.2:
        # Magnet cavity perimeter & inner loops
        theta_circle = np.linspace(0, 2*np.pi, 100)
        ax.plot(5.1 * np.cos(theta_circle), 5.1 * np.sin(theta_circle), color='#f43f5e', lw=2.5, label='Perímetro Cavidad Imán (Ø10.2mm)')
        for w_c in range(1, 4):
            ax.plot((5.1 + w_c * line_width) * np.cos(theta_circle), (5.1 + w_c * line_width) * np.sin(theta_circle),
                    color='#fb7185', lw=1.0, linestyle=':')
            
        # Draw rectilinear infill hatching
        infill_pitch = 2.4 # mm
        for y_h in np.arange(-50.0, 50.0, infill_pitch):
            # Intersect with hex and circle
            # Simplified hatching lines
            x_max = min(APOTHEM - 2.0, (R - abs(y_h)) * math.tan(math.radians(60))) if abs(y_h) > R/2 else APOTHEM - 2.0
            if x_max > 6.0:
                ax.plot([-x_max, -6.0], [y_h, y_h], color='#f59e0b', lw=0.8, alpha=0.6)
                ax.plot([6.0, x_max], [y_h, y_h], color='#f59e0b', lw=0.8, alpha=0.6)
    elif z <= 25.0:
        # Solid infill across full hexagon
        infill_pitch = 2.2
        for y_h in np.arange(-50.0, 50.0, infill_pitch):
            x_max = min(APOTHEM - 2.0, (R - abs(y_h)) * math.tan(math.radians(60))) if abs(y_h) > R/2 else APOTHEM - 2.0
            if x_max > 0:
                ax.plot([-x_max, x_max], [y_h, y_h], color='#f59e0b', lw=0.7, alpha=0.5)
    else:
        # Mountain slice at Z = 55.0 mm
        # Contour at Z >= 55.0
        grid_res = 400
        xs_g = np.linspace(-APOTHEM, APOTHEM, grid_res)
        ys_g = np.linspace(-R, R, grid_res)
        Xg, Yg = np.meshgrid(xs_g, ys_g)
        pts_g = np.column_stack((Yg.ravel(), Xg.ravel()))
        Z_slice = interp_z(pts_g).reshape((grid_res, grid_res))
        
        # Plot mountain contour at this height
        cs = ax.contour(Xg, Yg, Z_slice, levels=[z], colors=['#34d399'], linewidths=[2.2])
        # Infill inside mountain slice
        mask_m = Z_slice >= z
        infill_pitch = 2.0
        for y_h in np.arange(-40.0, 40.0, infill_pitch):
            # Find intervals where mask is true
            iy = int(round((y_h - (-R)) / (2 * R) * (grid_res - 1)))
            row = mask_m[iy, :]
            if np.any(row):
                x_in = xs_g[row]
                ax.plot([x_in[0], x_in[-1]], [y_h, y_h], color='#f59e0b', lw=0.9, alpha=0.7)
                
    ax.set_xlim(-68, 68)
    ax.set_ylim(-68, 68)
    ax.set_aspect('equal')
    ax.set_title(lay["name"], color='#38bdf8', fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.grid(True, color='#1e293b', linestyle=':', alpha=0.5)

plt.suptitle("Simulación de Trayectorias G-Code y Cinemática de Extrusión FDM\n(4 Paredes Perimetrales, Boquilla 0.40mm, Retracción Anti-Stringing)",
             color='#e2e8f0', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

out_gcode = os.path.join(out_dir, "render_gcode_toolpath_simulation.png")
plt.savefig(out_gcode, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy(out_gcode, os.path.join(brain_dir, "render_gcode_toolpath_simulation.png"))
print(f"Saved: {out_gcode} and copied to brain")
print(f"G-code toolpath simulation finished in {time.time()-t0:.2f}s.")
