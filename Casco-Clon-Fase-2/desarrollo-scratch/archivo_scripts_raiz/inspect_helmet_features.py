import trimesh
import numpy as np

m = trimesh.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco\Phase 2 Main Body.stl")
verts = m.vertices

print("Bounds:")
print("X:", verts[:, 0].min(), verts[:, 0].max())
print("Y:", verts[:, 1].min(), verts[:, 1].max())
print("Z:", verts[:, 2].min(), verts[:, 2].max())

# Let's inspect the rear profile at X=0 (sagittal slice)
mask_center = np.abs(verts[:, 0]) < 2.0
v_center = verts[mask_center]
# Sort by Z
v_center = v_center[v_center[:, 2].argsort()]

print("\nCenterline Z from 1700 to 1950 (rear and front):")
for z in np.arange(1700, 1950, 15):
    sub = v_center[(v_center[:, 2] >= z) & (v_center[:, 2] < z+15)]
    if len(sub) > 0:
        y_front = sub[:, 1].min()
        y_back = sub[:, 1].max()
        print(f"Z in [{z:4d}, {z+15:4d}]: Y_front = {y_front:6.1f}, Y_back = {y_back:6.1f}, count={len(sub)}")

# Let's check the rear dial/disc location:
# In the rear (+Y), where is the circular dial located?
# Usually around X=0, Y > 200, Z around 1800-1850.
rear_sub = verts[(np.abs(verts[:, 0]) < 30) & (verts[:, 1] > 200)]
print("\nRear feature around center:")
print(f"Z min={rear_sub[:, 2].min():.1f}, max={rear_sub[:, 2].max():.1f}, Y max={rear_sub[:, 1].max():.1f}")

# Let's find the dome groove / depression:
# In the dome (Z between 1800 and 1900), look at radial profile or Y profile at X=0 or X=60
