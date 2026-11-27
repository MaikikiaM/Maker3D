import os
import math
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

print("=== CATAN 7-HEX CLUSTER TESSELLATION & TOLERANCE SIMULATION ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"
os.makedirs(out_dir, exist_ok=True)

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
DIAM_FLAT = 2 * APOTHEM # 110.851 mm (center-to-center distance between adjacent tiles)

# 7 Hex Centers:
# Center 0: (0, 0) - Master 3D Mountain
# 6 Neighbors at angles 0, 60, 120, 180, 240, 300 degrees, at distance DIAM_FLAT
neighbor_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
hex_centers = [(0.0, 0.0)]
for ang in neighbor_angles:
    hex_centers.append((DIAM_FLAT * math.cos(ang), DIAM_FLAT * math.sin(ang)))

hex_types = [
    {"name": "MONTAÑA 3D MASTER\n(Mineral / Cota 83mm)", "color": "#38bdf8", "border": "#38bdf8", "is_3d": True},
    {"name": "PASTIZAL (Lana)\nZ = 25.0 mm", "color": "#15803d", "border": "#22c55e", "is_3d": False},
    {"name": "COLINAS (Arcilla)\nZ = 25.0 mm", "color": "#b45309", "border": "#f59e0b", "is_3d": False},
    {"name": "BOSQUE (Madera)\nZ = 25.0 mm", "color": "#047857", "border": "#10b981", "is_3d": False},
    {"name": "CAMPOS (Trigo)\nZ = 25.0 mm", "color": "#ca8a04", "border": "#eab308", "is_3d": False},
    {"name": "MONTAÑA 2 (Mineral)\nZ = 25.0 mm", "color": "#475569", "border": "#94a3b8", "is_3d": False},
    {"name": "DESIERTO (Vacío)\nZ = 25.0 mm", "color": "#78716c", "border": "#d6d3d1", "is_3d": False}
]

fig, ax = plt.subplots(figsize=(14, 14), facecolor='#090d16')
ax.set_facecolor('#0b1120')

# Corner angles for regular hexagon
v_angles = (30 + 60 * np.arange(6)) * np.pi / 180.0

for idx, (cx, cy) in enumerate(hex_centers):
    t_info = hex_types[idx]
    vx = cx + R * np.cos(v_angles)
    vy = cy + R * np.sin(v_angles)
    
    # Draw hex tile
    alpha_fill = 0.28 if t_info["is_3d"] else 0.18
    poly = patches.Polygon(np.column_stack([vx, vy]), closed=True,
                           facecolor=t_info["color"], alpha=alpha_fill,
                           edgecolor=t_info["border"], lw=2.5 if t_info["is_3d"] else 1.8, zorder=2)
    ax.add_patch(poly)
    
    # Center magnet pocket indicator
    mag = patches.Circle((cx, cy), radius=5.1, facecolor='#f43f5e', alpha=0.45, edgecolor='#f43f5e', lw=1.5, zorder=4)
    ax.add_patch(mag)
    
    # Tile label
    font_w = 'bold' if t_info["is_3d"] else 'normal'
    f_size = 11 if t_info["is_3d"] else 9.5
    ax.text(cx, cy - 3, t_info["name"], color='#f8fafc', fontsize=f_size, fontweight=font_w,
            ha='center', va='center', zorder=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor=t_info["border"], lw=1.2))

# Draw Road Connections along the 6 edges of the central mountain
for ang in neighbor_angles:
    # Midpoint of edge between center hex and neighbor
    mid_x = (DIAM_FLAT / 2.0) * math.cos(ang)
    mid_y = (DIAM_FLAT / 2.0) * math.sin(ang)
    # Perpendicular unit vector
    p_x = -math.sin(ang)
    p_y = math.cos(ang)
    road_len = 32.0 # mm
    ax.plot([mid_x - road_len/2 * p_x, mid_x + road_len/2 * p_x],
            [mid_y - road_len/2 * p_y, mid_y + road_len/2 * p_y],
            color='#38bdf8', lw=4.5, solid_capstyle='round', zorder=5)

# Draw Settlement / City 3-Way Junction Pads at the 6 vertices of the central mountain
v_corners_c0 = np.column_stack([R * np.cos(v_angles), R * np.sin(v_angles)])
for vx, vy in v_corners_c0:
    settle_pad = patches.Circle((vx, vy), radius=8.0, facecolor='#f59e0b', alpha=0.35, edgecolor='#f59e0b', lw=2.0, linestyle='--', zorder=6)
    ax.add_patch(settle_pad)
    # Settlement icon (house shape)
    ax.plot(vx, vy, marker='^', color='#f59e0b', markersize=9, zorder=7)

# Dimension annotation: Flat-to-flat distance 110.85 mm
ax.annotate("", xy=hex_centers[1], xytext=(0, 0),
            arrowprops=dict(arrowstyle="<->", color='#34d399', lw=2.5))
ax.text(hex_centers[1][0]/2.0, -9.0, f"D = 110.85 mm\n(Apotema × 2)", color='#34d399',
        fontsize=10.5, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor='#34d399', lw=1.2), zorder=15)

# Dimension annotation: Edge length 64.00 mm
ax.annotate("", xy=(v_corners_c0[0, 0], v_corners_c0[0, 1]), xytext=(v_corners_c0[1, 0], v_corners_c0[1, 1]),
            arrowprops=dict(arrowstyle="<->", color='#38bdf8', lw=2.5))
ax.text((v_corners_c0[0, 0] + v_corners_c0[1, 0])/2.0 + 12, (v_corners_c0[0, 1] + v_corners_c0[1, 1])/2.0,
        f"L = 64.00 mm", color='#38bdf8', fontsize=10.5, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor='#38bdf8', lw=1.2), zorder=15)

ax.set_xlim(-165, 165)
ax.set_ylim(-165, 165)
ax.set_aspect('equal')
ax.axis('off')

# Legend box with Catan board standards
legend_text = (
    "ESTÁNDARES DE INTEGRACIÓN EN TABLERO CATAN:\n"
    "• Distancia Lado a Lado (Paso Modular): 110.85 mm (Apotema × 2)\n"
    "• Longitud de Arista / Camino: 64.00 mm (Ancho calzada: 3.5 mm plana)\n"
    "• Diámetro Mayor Vértice a Vértice: 128.00 mm (Radio circunscrito R = 64.0 mm)\n"
    "• Nodos de Poblados / Ciudades: Círculo de apoyo Ø16mm plano a Z = 25.0 mm\n"
    "• Alojamiento Magnético: Imán cilíndrico Neodimio N52 Ø10.0 × 5.0 mm\n"
    "• Tolerancia de Ajuste Entre Losetas: 0.20 mm juego perimetral estándar"
)
ax.text(0.02, 0.03, legend_text, transform=ax.transAxes, color='#e2e8f0', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#0f172a', edgecolor='#38bdf8', lw=1.5))

plt.suptitle("Simulación de Ensamblaje y Tolerancias en Racimo Catan (7 Losetas)\n(Coplanaridad Exacta a Z = 25.0 mm y Asiento Plano para Poblados)",
             color='#e2e8f0', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

out_cluster = os.path.join(out_dir, "render_catan_7hex_cluster_assembly.png")
plt.savefig(out_cluster, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy(out_cluster, os.path.join(brain_dir, "render_catan_7hex_cluster_assembly.png"))
print(f"Saved: {out_cluster} and copied to brain")
print(f"Catan cluster simulation finished in {time.time()-t0:.2f}s.")
