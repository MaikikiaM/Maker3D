import trimesh
import numpy as np

m = trimesh.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco\Phase 2 Main Body.stl")
verts = m.vertices

# In the wing region: Y in [-10, 75], Z in [1830, 1915]
mask = (verts[:, 1] >= -10) & (verts[:, 1] <= 75) & (verts[:, 2] >= 1830) & (verts[:, 2] <= 1915) & (verts[:, 0] > 0)
wing_shell = verts[mask]

print(f"Points in wing region: {len(wing_shell)}")
print(f"X min = {wing_shell[:, 0].min():.1f}, X max = {wing_shell[:, 0].max():.1f}")

# Let's inspect X distribution for different Y and Z
for y in [-10, 10, 30, 50, 70]:
    for z in [1840, 1860, 1880, 1900]:
        sub = wing_shell[(np.abs(wing_shell[:, 1] - y) < 8) & (np.abs(wing_shell[:, 2] - z) < 8)]
        if len(sub) > 0:
            print(f"Y ≈ {y:2d}, Z ≈ {z:4d}: X in [{sub[:, 0].min():5.1f}, {sub[:, 0].max():5.1f}]")
