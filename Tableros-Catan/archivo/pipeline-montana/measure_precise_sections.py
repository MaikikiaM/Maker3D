import os
import numpy as np
import trimesh
from PIL import Image

CASCO = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco"
OUT = r"c:\Users\migue\Proyectos IA varios\Miscelanea\scratch\geo"
FOTOS = os.path.join(CASCO, "Fotos")

# Crop rear photos at high zoom
im_r = Image.open(os.path.join(FOTOS, "WhatsApp Image 2026-10-17 at 17.40.57 (2).jpeg"))
w, h = im_r.size
im_r.crop((int(w*0.1), int(h*0.18), int(w*0.9), int(h*0.92))).save(os.path.join(OUT, "crop_rear_detail.png"))

main = trimesh.load(os.path.join(CASCO, "Phase 2 Main Body.stl"))
no_teeth = trimesh.load(os.path.join(CASCO, "Phase 2 Main Body - No Teeth.stl"))
aer_l = trimesh.load(os.path.join(CASCO, r"Greeblies\Aerator Left - Mesh.stl"))

# 1. Exact cross-sections of the Fin using trimesh.section!
print("=== 1. EXACT FIN SLICES (Y = 0, 50, 100, 150) ===")
for y_val in [-20, 0, 40, 80, 120, 160, 185]:
    sec = main.section(plane_origin=[0, y_val, 0], plane_normal=[0, 1, 0])
    if sec is None:
        continue
    pts = sec.vertices
    top = pts[(pts[:, 2] > 1875) & (np.abs(pts[:, 0]) < 24)]
    if len(top):
        # outer hull only (upper envelope)
        xs = np.round(top[:, 0], 1)
        zmax = top[:, 2].max()
        # find X width where Z > zmax - 15
        hi = top[top[:, 2] > zmax - 12]
        if len(hi):
            print(f"Y={y_val:4d}: Z_peak={zmax:7.2f}, fin X_span=[{hi[:,0].min():6.2f}, {hi[:,0].max():6.2f}]")

# Look closely at Y=40 fin outer profile:
sec40 = main.section(plane_origin=[0, 40, 0], plane_normal=[0, 1, 0]).vertices
fin40 = sec40[(sec40[:,2] > 1925) & (sec40[:,0] >= 0) & (sec40[:,0] <= 22)]
fin40 = fin40[np.argsort(fin40[:,0])]
print("Fin Y=40 right half profile (X, Z):")
for x_b in np.arange(0, 20, 1.0):
    s = fin40[(fin40[:,0] >= x_b) & (fin40[:,0] < x_b+1.0)]
    if len(s):
        print(f"  X={x_b+0.5:4.1f}: Z_max={s[:,2].max():.2f}")

# 2. Exact Frown / Teeth trough using section planes X = 0, 8, 16, 24, 32, 40, 48
print("\n=== 2. FROWN / TEETH TROUGH SLICES ACROSS X ===")
for x_val in [0, 6, 12, 18, 24, 30, 36, 42, 48]:
    sec = main.section(plane_origin=[x_val, 0, 0], plane_normal=[1, 0, 0]).vertices
    sec_nt = no_teeth.section(plane_origin=[x_val, 0, 0], plane_normal=[1, 0, 0]).vertices
    sub = sec[(sec[:,1] < -40) & (sec[:,2] > 1690) & (sec[:,2] < 1748)]
    sub_nt = sec_nt[(sec_nt[:,1] < -40) & (sec_nt[:,2] > 1690) & (sec_nt[:,2] < 1748)]
    # Front-most surface at each Z
    print(f"--- X = {x_val} ---")
    for z0 in range(1696, 1744, 3):
        a = sub[(sub[:,2] >= z0) & (sub[:,2] < z0+3)]
        b = sub_nt[(sub_nt[:,2] >= z0) & (sub_nt[:,2] < z0+3)]
        if len(a) and len(b):
            ya, yb = a[:,1].min(), b[:,1].min()
            mark = " <-- TOOTH!" if abs(ya - yb) > 0.4 else ""
            print(f"  Z={z0+1.5:6.1f}: Y_main={ya:7.2f}, Y_noteeth={yb:7.2f}{mark}")

# 3. Exact Rear profile slices at X = 0, 20, 45, 70 and horizontal slices at Z = 1660, 1700, 1750, 1790, 1820, 1840
print("\n=== 3. REAR SLICES ===")
sec_rear0 = main.section(plane_origin=[0, 0, 0], plane_normal=[1, 0, 0]).vertices
r0 = sec_rear0[(sec_rear0[:,1] > 180) & (sec_rear0[:,2] > 1785) & (sec_rear0[:,2] < 1870)]
for z0 in range(1790, 1866, 3):
    s = r0[(r0[:,2] >= z0) & (r0[:,2] < z0+3)]
    if len(s):
        print(f"X=0   Z={z0+1.5:6.1f}: Y_outer={s[:,1].max():7.2f}")

sec_rear40 = main.section(plane_origin=[40, 0, 0], plane_normal=[1, 0, 0]).vertices
r40 = sec_rear40[(sec_rear40[:,1] > 160) & (sec_rear40[:,2] > 1805) & (sec_rear40[:,2] < 1870)]
for z0 in range(1810, 1866, 3):
    s = r40[(r40[:,2] >= z0) & (r40[:,2] < z0+3)]
    if len(s):
        print(f"X=40  Z={z0+1.5:6.1f}: Y_outer={s[:,1].max():7.2f}")

print("\n--- Rear Horizontal Slices (Ribs & Ladder X coordinates) ---")
for z_val in [1655, 1690, 1730, 1770, 1795]:
    sec_z = main.section(plane_origin=[0, 0, z_val], plane_normal=[0, 0, 1]).vertices
    rz = sec_z[(sec_z[:,1] > 185) & (sec_z[:,0] >= 0) & (sec_z[:,0] <= 45)]
    line = []
    for x0 in range(0, 42, 2):
        s = rz[(rz[:,0] >= x0) & (rz[:,0] < x0+2)]
        if len(s):
            line.append(f"{x0+1:2d}:{s[:,1].max():.1f}")
    print(f"Z={z_val}: " + " ".join(line))

# 4. Aerator axis & Jaw tube center-line!
print("\n=== 4. AERATOR & JAW TUBE GEOMETRY ===")
av = aer_l.vertices
print("Aerator Left centroid:", av.mean(axis=0))
# PCA of aerator to get exact cylinder axis!
ac = av - av.mean(axis=0)
eigvals, eigvecs = np.linalg.eigh(ac.T @ ac)
print("Aerator Left principal axes (smallest eigval = symmetry/cylinder axis or thin axis):")
for val, vec in zip(eigvals, eigvecs.T):
    print(f"  eig={val:10.1f}, vec=[{vec[0]:+.4f}, {vec[1]:+.4f}, {vec[2]:+.4f}]")
