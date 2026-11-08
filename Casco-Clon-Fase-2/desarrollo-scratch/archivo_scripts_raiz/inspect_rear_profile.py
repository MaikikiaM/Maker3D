import trimesh
import numpy as np

m = trimesh.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco\Phase 2 Main Body.stl")
verts = m.vertices

# Rear profile at Y > 190, X between -50 and 50
print("Analyzing rear features:")
rear_pts = verts[(verts[:, 1] > 190)]
print(f"Rear pts count: {len(rear_pts)}")

# Let's inspect the ribs on the rear:
# The rear has two ribs flanking X=0. What are their X coordinates?
# In the range Z in [1650, 1750]:
lower_rear = rear_pts[(rear_pts[:, 2] > 1680) & (rear_pts[:, 2] < 1720)]
print(f"Lower rear Z in [1680, 1720]: Y range = [{lower_rear[:, 1].min():.1f}, {lower_rear[:, 1].max():.1f}]")

# Let's slice horizontally at Z = 1700, 1750, 1800, 1820, 1850
import json

for z in [1650, 1700, 1750, 1800, 1820, 1840, 1855, 1870]:
    slice_pts = verts[(np.abs(verts[:, 2] - z) < 2.0) & (verts[:, 1] > 180)]
    if len(slice_pts) > 0:
        print(f"Z ≈ {z}: count={len(slice_pts)}, X in [{slice_pts[:, 0].min():.1f}, {slice_pts[:, 0].max():.1f}], Y in [{slice_pts[:, 1].min():.1f}, {slice_pts[:, 1].max():.1f}]")
        # Find local maxima in Y (the ribs or features)
        # Sort by X
        s = slice_pts[slice_pts[:, 0].argsort()]
        # print Y along X
        xs = np.linspace(-60, 60, 13)
        y_profile = []
        for xi in xs:
            sub = s[np.abs(s[:, 0] - xi) < 5]
            if len(sub) > 0:
                y_profile.append(f"{sub[:, 1].max():.1f}")
            else:
                y_profile.append(" - ")
        print(f"   Profile Y across X=[-60..60]: {' '.join(y_profile)}")
