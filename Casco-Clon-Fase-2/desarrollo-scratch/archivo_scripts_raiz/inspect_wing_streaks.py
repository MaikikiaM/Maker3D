import trimesh
import numpy as np

m_red = trimesh.load("test_mesh_red.stl")
verts = m_red.vertices

# Let's inspect the left wing on the red mesh: X < -50, Z > 1820, Y in [-20, 80]
wing_verts = verts[(verts[:, 0] < -50) & (verts[:, 2] > 1820) & (verts[:, 1] > -20) & (verts[:, 1] < 80)]
print(f"Wing red vertices: {len(wing_verts)}")

# Let's check the thickness and shape of the wing cutter:
# In test_new_geometry.py:
# cs_wing was extruded by 65.0 mm from X = 58.0 to X = 123.0 mm.
# But what is cs_wing_base - (s1 + s2 + s3)?
# Let's check where the slits s1, s2, s3 are!
# s1, s2, s3 are 2D cross sections in (Y, Z).
# When extruded along X, they are infinite horizontal prisms in X!
# As the helmet dome curves from X = -58 (inner) to X = -100 (outer), the helmet surface is a sphere!
# A 2D curve in (Y, Z) projected along X onto a sphere:
# At X = -58, Y and Z on the sphere are different than at X = -95!
# On a sphere: X^2 + Y^2 + (Z - Z_center)^2 = R^2!
# So for a given Y and Z, X is fixed on the sphere surface:
# X(Y, Z) = -sqrt(R^2 - Y^2 - (Z - Z_center)^2)
print("Analyzing dome surface curvature:")
m_base = trimesh.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco\Phase 2 Main Body.stl")
v_b = m_base.vertices
sub = v_b[(v_b[:, 0] < -50) & (v_b[:, 2] > 1830) & (v_b[:, 1] > -10) & (v_b[:, 1] < 70)]
print(f"Base dome surface pts in wing region: {len(sub)}")
print(f"X range: [{sub[:, 0].min():.1f}, {sub[:, 0].max():.1f}]")
print(f"Y range: [{sub[:, 1].min():.1f}, {sub[:, 1].max():.1f}]")
print(f"Z range: [{sub[:, 2].min():.1f}, {sub[:, 2].max():.1f}]")

# Let's find why there were streaks:
# At X = -58 mm, does the cutter have an abrupt end-cap face at X = -58 mm that intersects the dome surface?
# YES! If the cutter starts at X = -58.0 mm, any part of the helmet dome that has |X| < 58.0 mm will NOT be cut!
# And on the dome, as Y goes from -10 to +70 mm and Z goes up to 1910 mm,
# at Z = 1900 mm, the dome surface has X ≈ -50 to -65 mm!
# So the cutter's inner boundary at X = -58 mm CUTS RIGHT THROUGH THE WING AT THE TOP!
# Creating a vertical cut-off wall at X = -58 mm!
print("\nChecking dome X at Z = 1900 mm:")
top_pts = sub[sub[:, 2] > 1895]
print(f"At Z > 1895: X in [{top_pts[:, 0].min():.1f}, {top_pts[:, 0].max():.1f}]")
