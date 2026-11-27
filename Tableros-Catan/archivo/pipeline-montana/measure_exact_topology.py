import os
import numpy as np
import trimesh

CASCO = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco"

main = trimesh.load(os.path.join(CASCO, "Phase 2 Main Body.stl"))
no_teeth = trimesh.load(os.path.join(CASCO, "Phase 2 Main Body - No Teeth.stl"))
dome = trimesh.load(os.path.join(CASCO, r"Non-Magnetic Assembly\Dome.stl"))
front = trimesh.load(os.path.join(CASCO, r"Non-Magnetic Assembly\Front - Visor Ridge.stl"))
back = trimesh.load(os.path.join(CASCO, r"Non-Magnetic Assembly\Back.stl"))
tube_l = trimesh.load(os.path.join(CASCO, r"Templates\Tube Stripe Template - Left.stl"))
visor_t = trimesh.load(os.path.join(CASCO, r"Templates\Visor Cutout Template.stl"))
aer_l = trimesh.load(os.path.join(CASCO, r"Greeblies\Aerator Left - Mesh.stl"))
ear_l = trimesh.load(os.path.join(CASCO, r"Greeblies\Ear Piece Left.stl"))
mouth = trimesh.load(os.path.join(CASCO, r"Greeblies\Mouth Center Detail.stl"))

for name, m in [("main", main), ("no_teeth", no_teeth), ("dome", dome), ("front", front),
                ("back", back), ("tube_l", tube_l), ("visor_t", visor_t),
                ("aer_l", aer_l), ("ear_l", ear_l), ("mouth", mouth)]:
    b = m.bounds
    print(f"{name:10s}: X=[{b[0,0]:7.2f}, {b[1,0]:7.2f}]  Y=[{b[0,1]:7.2f}, {b[1,1]:7.2f}]  Z=[{b[0,2]:7.2f}, {b[1,2]:7.2f}]")

V = main.vertices
N = main.vertex_normals

# 1. Inspect outer surface profile on the side/back for the recessed dome band!
print("\n--- REAR PROFILE (X in [-3, 3], Y > 150, Z in [1800, 1880]) ---")
mask = (np.abs(V[:, 0]) < 3.0) & (V[:, 1] > 150) & (V[:, 2] >= 1800) & (V[:, 2] <= 1880)
pts = V[mask]
for z0 in range(1800, 1878, 2):
    sub = pts[(pts[:, 2] >= z0) & (pts[:, 2] < z0 + 2)]
    if len(sub):
        print(f"Z={z0+1:4d}: Y_max={sub[:,1].max():7.2f}")

print("\n--- SIDE PROFILE AT Y=120 (Z in [1810, 1875], X > 60) ---")
mask = (np.abs(V[:, 1] - 120.0) < 2.0) & (V[:, 0] > 60) & (V[:, 2] >= 1810) & (V[:, 2] <= 1875)
pts = V[mask]
for z0 in range(1810, 1874, 2):
    sub = pts[(pts[:, 2] >= z0) & (pts[:, 2] < z0 + 2)]
    if len(sub):
        print(f"Z={z0+1:4d}: X_max={sub[:,0].max():7.2f}")

print("\n--- DOME RECESSED BAND START IN Y (Z in [1835, 1848], X > 50) ---")
mask = (V[:, 0] > 50) & (V[:, 2] >= 1836) & (V[:, 2] <= 1846) & (V[:, 1] >= -20) & (V[:, 1] <= 50)
pts = V[mask]
nrm = N[mask]
# Look for forward-facing step normal (ny < -0.3) or outer skin jump
outer = []
for y0 in range(-15, 45, 2):
    sub = pts[(pts[:, 1] >= y0) & (pts[:, 1] < y0 + 2)]
    if len(sub):
        outer.append((y0 + 1, sub[:, 0].max()))
for y_c, xm in outer:
    print(f"Y={y_c:3d}: X_max={xm:7.2f}")

print("\n--- FRONT BROW PROFILE (X in [25, 35], Y < -40, Z in [1805, 1865]) ---")
mask = (V[:, 0] >= 25) & (V[:, 0] <= 35) & (V[:, 1] < -40) & (V[:, 2] >= 1805) & (V[:, 2] <= 1865)
pts = V[mask]
for z0 in range(1806, 1864, 2):
    sub = pts[(pts[:, 2] >= z0) & (pts[:, 2] < z0 + 2)]
    if len(sub):
        print(f"Z={z0+1:4d}: Y_min={sub[:,1].min():7.2f}")
