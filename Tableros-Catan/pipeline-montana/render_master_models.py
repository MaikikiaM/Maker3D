import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import os

mesh = trimesh.load("Tableros Catan/Loseta_Montana_3D/catan_mountain_solid.stl")
verts = mesh.vertices
faces = mesh.faces

out_dir = "Tableros Catan/Loseta_Montana_3D/renders_cad"
os.makedirs(out_dir, exist_ok=True)

# 1. Perspective Isometric View
fig = plt.figure(figsize=(12, 10), facecolor='#070b14')
ax = fig.add_subplot(111, projection='3d', facecolor='#070b14')

# Subsample faces for clean visualization
step = 2
plot_faces = faces[::step]
face_z = np.mean(verts[plot_faces, 2], axis=1)

# Color shading: granite grey, snow white at top
v0 = verts[plot_faces[:, 0]]
v1 = verts[plot_faces[:, 1]]
v2 = verts[plot_faces[:, 2]]
fnorm = np.cross(v1 - v0, v2 - v0)
fnorm = fnorm / (np.linalg.norm(fnorm, axis=1, keepdims=True) + 1e-8)
h_fact = np.abs(fnorm[:, 2])

# Build RGB colors
colors = np.zeros((len(plot_faces), 4))
for i in range(len(plot_faces)):
    z = face_z[i]
    h = h_fact[i]
    if z < 24.9:
        colors[i] = [0.12, 0.14, 0.18, 1.0] # Dark base
    elif z >= 65.0 and h > 0.35:
        colors[i] = [0.96, 0.98, 1.0, 1.0] # Snow
    elif z >= 72.0:
        colors[i] = [0.98, 0.99, 1.0, 1.0] # Summit snow
    elif 25.0 <= z <= 28.0 and h > 0.6:
        colors[i] = [0.24, 0.45, 0.22, 1.0] # Alpine moss
    else:
        # Granite rock shading with slope
        shade = 0.38 + 0.32 * h + 0.15 * ((z - 25.0) / 58.0)
        colors[i] = [shade * 0.95, shade, shade * 1.05, 1.0]

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Triangles 3D coordinates
tri_coords = verts[plot_faces]

coll = Poly3DCollection(tri_coords, facecolors=colors, edgecolors='none', linewidths=0, antialiased=True)
ax.add_collection3d(coll)

# Set axis limits
ax.set_xlim(-65, 65)
ax.set_ylim(-65, 65)
ax.set_zlim(0, 85)

ax.view_init(elev=34, azim=215)
ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
iso_path = os.path.join(out_dir, "render_mountain_isometric.png")
plt.savefig(iso_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=0)
plt.close()
print("Saved isometric render:", iso_path)

# 2. Top-down view
fig2 = plt.figure(figsize=(10, 10), facecolor='#070b14')
ax2 = fig2.add_subplot(111, projection='3d', facecolor='#070b14')
coll2 = Poly3DCollection(tri_coords, facecolors=colors, edgecolors='none', linewidths=0, antialiased=True)
ax2.add_collection3d(coll2)
ax2.set_xlim(-65, 65)
ax2.set_ylim(-65, 65)
ax2.set_zlim(0, 85)
ax2.view_init(elev=90, azim=-90)
ax2.set_axis_off()
fig2.subplots_adjust(left=0, right=1, bottom=0, top=1)
top_path = os.path.join(out_dir, "render_mountain_top_ortho.png")
plt.savefig(top_path, dpi=200, bbox_inches='tight', facecolor=fig2.get_facecolor(), pad_inches=0)
plt.close()
print("Saved top ortho render:", top_path)
