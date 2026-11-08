import numpy as np
import trimesh
import manifold3d as m3d
import os

helmet_mesh = trimesh.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco\Phase 2 Main Body.stl")
v_base = helmet_mesh.vertices.astype('float32')
f_base = helmet_mesh.faces.astype('uint32')
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))
print(f"Helmet loaded. Volume = {helmet_mani.volume():.1f} mm3")

# Let's inspect the ear and temple Y coordinates:
# Temple / brow ridge runs from Y ≈ -60 to Y ≈ +65. Ear begins at Y = 62.5.
# The wings should be located:
# Y from front of dome (Y ≈ -15 to +10) up to Y ≈ 75 (just in front of ear Y=62..92).
# Z for wings: from dome groove Z ≈ 1828 up to dome curve Z ≈ 1910.
print("Target wing Y range: [-15, 75] mm, Z range: [1828, 1910] mm.")
