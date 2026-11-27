import os
import numpy as np
import trimesh

CASCO = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco"
main = trimesh.load(os.path.join(CASCO, "Phase 2 Main Body.stl"))
no_teeth = trimesh.load(os.path.join(CASCO, "Phase 2 Main Body - No Teeth.stl"))
tube_l = trimesh.load(os.path.join(CASCO, r"Templates\Tube Stripe Template - Left.stl"))
V = main.vertices
N = main.vertex_normals

# 1. Exact Frown / Teeth pockets!
# In Main Body, the recessed back walls of the 8 tooth pockets sit deeper (+Y) than the tooth dividers,
# OR differ from No-Teeth! Let's inspect vertices in X in [-45, 45], Y < -50, Z in [1695, 1742]
# where Main differs from No-Teeth!
from scipy.spatial import cKDTree
tree = cKDTree(no_teeth.vertices)
d, _ = tree.query(V, k=1)
tp = V[(d > 0.35) & (V[:, 1] < -45) & (V[:, 2] > 1695) & (V[:, 2] < 1745) & (np.abs(V[:, 0]) < 50)]
print(f"=== 1. EXACT TEETH VERTICES: {len(tp)} ===")
for x0 in range(0, 46, 4):
    s = tp[(tp[:, 0] >= x0) & (tp[:, 0] < x0 + 4)]
    if len(s):
        print(f"  X=[{x0:2d},{x0+4:2d}]: Z=[{s[:,2].min():6.2f}, {s[:,2].max():6.2f}], Y=[{s[:,1].min():6.2f}, {s[:,1].max():6.2f}]")

# Also inspect YZ profile of nose/frown/snout at X = 0, 10, 20, 30, 38
print("\n=== FROWN UPPER & LOWER LIP CREASES (X = 0, 10, 20, 30, 38) ===")
for xv in [0, 10, 20, 30, 38]:
    sec = main.section(plane_origin=[xv, 0, 0], plane_normal=[1, 0, 0]).vertices
    s = sec[(sec[:, 1] < -45) & (sec[:, 2] >= 1695) & (sec[:, 2] <= 1748)]
    # bin by Z every 1.5 mm and print frontmost Y
    zs = np.arange(1698, 1745, 2.0)
    prof = []
    for z0 in zs:
        b = s[(s[:, 2] >= z0) & (s[:, 2] < z0 + 2.0)]
        if len(b):
            prof.append(f"{z0+1:.0f}:{b[:,1].min():.1f}")
    print(f"X={xv:2d}: " + " ".join(prof))

# 2. Cheek Depression exact ridge lines in YZ (X > 65)
print("\n=== 2. CHEEK DEPRESSION BOUNDARIES IN YZ ===")
for yv in range(-20, 56, 5):
    sec = main.section(plane_origin=[0, yv, 0], plane_normal=[0, 1, 0]).vertices
    s = sec[(sec[:, 0] > 60) & (sec[:, 2] >= 1705) & (sec[:, 2] <= 1795)]
    if len(s):
        # Find X_max (outer rim above and below hollow) and X_min of outer skin inside hollow
        # Separate outer skin from inner helmet wall! Helmet wall thickness is ~3-4 mm, so inner wall has smaller X.
        # At each Z bin, outer skin is the max X!
        zbins = np.arange(1706, 1792, 3.0)
        x_outer = []
        for z0 in zbins:
            b = s[(s[:, 2] >= z0) & (s[:, 2] < z0 + 3.0)]
            if len(b):
                x_outer.append((z0 + 1.5, b[:, 0].max()))
        # Print min of x_outer (deepest part of hollow) and where x_outer jumps out
        if x_outer:
            arr = np.array(x_outer)
            z_min_x = arr[np.argmin(arr[:, 1])]
            print(f"Y={yv:3d}: deepest hollow at Z={z_min_x[0]:6.1f} (X={z_min_x[1]:5.1f}) | Z=1712 X={arr[arr[:,0]==1713.5,1][0] if any(arr[:,0]==1713.5) else 0:5.1f} | Z=1782 X={arr[arr[:,0]==1782.5,1][0] if any(arr[:,0]==1782.5) else 0:5.1f}")

# 3. Jaw Tube 3D centerline from Y = -70 to Y = +60!
print("\n=== 3. JAW TUBE 3D OUTER PROFILE & SLITS ===")
for yv in range(-65, 55, 10):
    sec = main.section(plane_origin=[0, yv, 0], plane_normal=[0, 1, 0]).vertices
    # Lower jaw tube on +X side: X > 20, Z < 1712
    s = sec[(sec[:, 0] > 20) & (sec[:, 2] < 1712) & (sec[:, 2] > 1616)]
    if len(s):
        print(f"Y={yv:3d}: X_max={s[:,0].max():6.2f} (at Z={s[np.argmax(s[:,0]),2]:6.1f}), Z_min={s[:,2].min():6.2f} (at X={s[np.argmin(s[:,2]),0]:5.1f})")

# Exact 1st tube ventilation slit front edge (Y, Z, X)!
m_slit = (V[:, 0] > 88) & (V[:, 1] >= -5) & (V[:, 1] <= 22) & (V[:, 2] >= 1682) & (V[:, 2] <= 1714)
ss, sns = V[m_slit], N[m_slit]
slit1 = ss[np.abs(sns[:, 1]) > 0.7]
print("\n1st Tube Ventilation Slit exact coordinates:")
for z0 in range(1686, 1712, 3):
    b = slit1[(slit1[:, 2] >= z0) & (slit1[:, 2] < z0 + 3)]
    if len(b):
        print(f"  Z={z0+1.5:6.1f}: Y_min={b[:,1].min():6.2f}, X={np.median(b[:,0]):6.2f}")
