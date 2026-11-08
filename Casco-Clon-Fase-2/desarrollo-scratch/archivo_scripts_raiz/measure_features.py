import trimesh
import numpy as np

m = trimesh.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco\Phase 2 Main Body.stl")
verts = m.vertices

# Rear dial / circle analysis
# Where is the circle centered?
# Vertices with |X| < 35, Y > 200, Z in [1800, 1845]
rear_pts = verts[(np.abs(verts[:, 0]) < 35) & (verts[:, 1] > 205) & (verts[:, 2] > 1800) & (verts[:, 2] < 1845)]
print("Rear disc vertices count:", len(rear_pts))
z_center = (rear_pts[:, 2].min() + rear_pts[:, 2].max()) / 2
print(f"Z range of rear disc: [{rear_pts[:, 2].min():.2f}, {rear_pts[:, 2].max():.2f}], mid = {z_center:.2f}")
print(f"X range of rear disc: [{rear_pts[:, 0].min():.2f}, {rear_pts[:, 0].max():.2f}]")
print(f"Y range of rear disc: [{rear_pts[:, 1].min():.2f}, {rear_pts[:, 1].max():.2f}]")

# Now let's check the dome groove around the helmet at different X, Y
# The horizontal groove / seam between dome and brow/body:
# Let's inspect side profile at X=80 (temple/ear area)
side_mask = (np.abs(verts[:, 0] - 80) < 5) & (verts[:, 2] > 1780) & (verts[:, 2] < 1880)
side_pts = verts[side_mask]
side_pts = side_pts[side_pts[:, 2].argsort()]
print("\nSide profile at X ≈ 80:")
for z in np.arange(1780, 1880, 10):
    s = side_pts[(side_pts[:, 2] >= z) & (side_pts[:, 2] < z+10)]
    if len(s) > 0:
        print(f"Z [{z}, {z+10}]: Y min={s[:, 1].min():.1f}, max={s[:, 1].max():.1f}, count={len(s)}")

# Let's check where the ear cap sits
greeb_ear = trimesh.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco\Greeblies\Ear Piece Right.stl")
print("\nEar Piece Right bounds:")
print("X:", greeb_ear.vertices[:, 0].min(), greeb_ear.vertices[:, 0].max())
print("Y:", greeb_ear.vertices[:, 1].min(), greeb_ear.vertices[:, 1].max())
print("Z:", greeb_ear.vertices[:, 2].min(), greeb_ear.vertices[:, 2].max())

# Mouth / Vocoder bounds
mouth = trimesh.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco\Greeblies\Mouth Center Detail.stl")
print("\nMouth bounds:")
print("X:", mouth.vertices[:, 0].min(), mouth.vertices[:, 0].max())
print("Y:", mouth.vertices[:, 1].min(), mouth.vertices[:, 1].max())
print("Z:", mouth.vertices[:, 2].min(), mouth.vertices[:, 2].max())

# Aerator bounds
aer = trimesh.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco\Greeblies\Aerator Right - Mesh.stl")
print("\nAerator Right bounds:")
print("X:", aer.vertices[:, 0].min(), aer.vertices[:, 0].max())
print("Y:", aer.vertices[:, 1].min(), aer.vertices[:, 1].max())
print("Z:", aer.vertices[:, 2].min(), aer.vertices[:, 2].max())
