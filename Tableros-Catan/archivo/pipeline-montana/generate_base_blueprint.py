import os
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("=== Generating High-Precision Base CAD Blueprint ===")

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"
os.makedirs(out_dir, exist_ok=True)

R = 64.0 # Radio del círculo circunscrito = longitud del lado del hexágono
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm (inradio)
DIAM_FLAT = 2 * APOTHEM # 110.851 mm (distancia entre lados opuestos)
DIAM_VERTEX = 2 * R # 128.00 mm (distancia entre vértices opuestos)

# Hexagon vertices
angles = np.array([(30 + 60 * k) * np.pi / 180 for k in range(6)])
vx = R * np.cos(angles)
vy = R * np.sin(angles)

fig, ax = plt.subplots(figsize=(10, 10), facecolor='#090d16')
ax.set_facecolor('#0b1120')

# Draw hex filled
hex_poly = patches.Polygon(np.column_stack([vx, vy]), closed=True, facecolor='#1e293b', edgecolor='#38bdf8', lw=2.5, zorder=2)
ax.add_patch(hex_poly)

# Draw 6 Equilateral Triangles (radial lines from center)
for k in range(6):
    ax.plot([0, vx[k]], [0, vy[k]], color='#f59e0b', linestyle='--', lw=1.5, zorder=3)

# Highlight one of the 6 equilateral triangles
tri_highlight = patches.Polygon([[0, 0], [vx[0], vy[0]], [vx[1], vy[1]]], closed=True, facecolor='#f59e0b', alpha=0.15, edgecolor='#f59e0b', lw=2.0, zorder=3)
ax.add_patch(tri_highlight)

# Draw central magnet cavity
mag_circle = patches.Circle((0, 0), radius=5.1, facecolor='#f43f5e', alpha=0.35, edgecolor='#f43f5e', lw=2.0, zorder=4)
ax.add_patch(mag_circle)
ax.plot(0, 0, marker='+', color='#f43f5e', markersize=14, markeredgewidth=2, zorder=5)

# Dimension 1: Lado del hexágono (L = 64.00 mm)
ax.annotate("", xy=(vx[1], vy[1]), xytext=(vx[2], vy[2]),
            arrowprops=dict(arrowstyle='<->', color='#38bdf8', lw=2.5))
mid_side = [(vx[1] + vx[2])/2, (vy[1] + vy[2])/2]
ax.text(mid_side[0] - 12, mid_side[1] + 4, f"Lado = {R:.2f} mm", color='#38bdf8', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor='#38bdf8', lw=1.5), zorder=10)

# Dimension 2: Distancia de Lado a Lado (110.85 mm)
mid_s0 = np.array([(vx[0] + vx[1])/2, (vy[0] + vy[1])/2])
mid_s3 = np.array([(vx[3] + vx[4])/2, (vy[3] + vy[4])/2])
ax.plot([mid_s0[0], mid_s3[0]], [mid_s0[1], mid_s3[1]], color='#34d399', linestyle=':', lw=2.0, zorder=4)
ax.annotate("", xy=mid_s0, xytext=mid_s3,
            arrowprops=dict(arrowstyle='<->', color='#34d399', lw=2.5))
ax.text(0, -14, f"Distancia Lado a Lado = {DIAM_FLAT:.2f} mm\n(2 × Apotema {APOTHEM:.2f} mm)",
        color='#34d399', fontsize=11, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f172a', edgecolor='#34d399', lw=1.5), zorder=10)

# Dimension 3: Distancia Vértice a Vértice (128.00 mm)
ax.annotate("", xy=(vx[0], vy[0]), xytext=(vx[3], vy[3]),
            arrowprops=dict(arrowstyle='<->', color='#a855f7', lw=2.2))
ax.text(26, 26, f"Vértice a Vértice = {DIAM_VERTEX:.2f} mm\n(2 × Radio {R:.2f} mm)",
        color='#c084fc', fontsize=10.5, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor='#a855f7', lw=1.5), zorder=10)

# Dimension 4: Imán Central
ax.text(0, 14, f"Alojamiento Imán\nØ10.20 × 5.20 mm",
        color='#fb7185', fontsize=10, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor='#f43f5e', lw=1.2), zorder=10)

# Dimension 5: Sector Triangular Equilátero
ax.text(mid_side[0] + 45, 12,
        f"Sector Triangular Equilátero (1 de 6):\n"
        f"• Base: {R:.2f} mm\n"
        f"• Lados radiales: {R:.2f} mm\n"
        f"• Altura (Apotema): {APOTHEM:.2f} mm\n"
        f"• Ángulos: 60° - 60° - 60°\n"
        f"• Área: {(0.5 * R * APOTHEM):.2f} mm²",
        color='#fbbf24', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f172a', edgecolor='#f59e0b', lw=1.5), zorder=10)

# Road margin indicator
road_circle = patches.RegularPolygon((0, 0), numVertices=6, radius=R - 3.5 / math.cos(math.radians(30)),
                                     orientation=math.radians(30), facecolor='none', edgecolor='#64748b', linestyle='--', lw=1.2)
ax.add_patch(road_circle)
ax.text(-APOTHEM + 3, -R + 10, "Carril Carreteras (3.5 mm plano a Z=25mm)", color='#94a3b8', fontsize=8.5, style='italic')

ax.set_xlim(-85, 95)
ax.set_ylim(-75, 75)
ax.set_aspect('equal')
ax.axis('off')

plt.title("Plano Acotado de Ingeniería - Base Hexagonal Loseta Montaña Catan",
          color='#38bdf8', fontsize=13.5, fontweight='bold', pad=15)

out_file = os.path.join(out_dir, "render_mountain_base_dimensions_blueprint.png")
plt.tight_layout()
plt.savefig(out_file, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy2(out_file, os.path.join(brain_dir, "render_mountain_base_dimensions_blueprint.png"))
print(f"Saved: {out_file} and copied to brain")
