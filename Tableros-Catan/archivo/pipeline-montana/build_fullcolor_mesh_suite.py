import os
import math
import time
import numpy as np
import scipy.ndimage as ndi
import scipy.interpolate as interp
from scipy.spatial import Delaunay
import trimesh
from PIL import Image

print("=================================================================")
print("  PHASE 2: BUILDING MASTER FULL-COLOR CONTINUOUS 3D MESH SUITE   ")
print("=================================================================")
t0 = time.time()

out_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D"
os.makedirs(out_dir, exist_ok=True)

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256258 mm
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF_H = Z_PEAK - Z_BASE # 58.00 mm

# Load master topography
Z_master = np.load('scratch/Z_top_master_final.npy')
ny_m, nx_m = Z_master.shape

xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m)
ys_m = np.linspace(-R, R, ny_m)

normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
v_angles = [(30 + 60 * k) * math.pi / 180 for k in range(6)]
hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles])

# Interpolator for exact continuous surface evaluation
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

# -------------------------------------------------------------
# 1. HIGH-RESOLUTION WATERTIGHT SOLID MESH
# -------------------------------------------------------------
print("--- Generating High-Density Monolithic Mesh ---")
# 192 boundary points (32 per edge)
pts_per_edge = 32
boundary_pts_2d = []
for k in range(6):
    p1 = hex_corners[k]
    p2 = hex_corners[(k + 1) % 6]
    for s in np.linspace(0, 1, pts_per_edge, endpoint=False):
        boundary_pts_2d.append((1.0 - s) * p1 + s * p2)
boundary_pts_2d = np.array(boundary_pts_2d)
N_b = len(boundary_pts_2d)

# Internal grid resolution: 281 x 325 (dx = 0.39mm)
res = 281
xs_grid = np.linspace(-APOTHEM + 0.8, APOTHEM - 0.8, res)
ys_grid = np.linspace(-R + 1.5, R - 1.5, int(res * (2 * R) / (2 * APOTHEM)))
X_g, Y_g = np.meshgrid(xs_grid, ys_grid)

d_edges = np.zeros((6, len(ys_grid), len(xs_grid)))
for k in range(6):
    d_edges[k] = APOTHEM - (X_g * normals[k][0] + Y_g * normals[k][1])
dist_edge_g = np.min(d_edges, axis=0)
mask_internal = dist_edge_g >= 0.8

int_x = list(X_g[mask_internal])
int_y = list(Y_g[mask_internal])

# Explicit critical summit vertices
critical_summits = [
    (-4.0, 6.0, 83.00),     # Main Summit Apex
    (18.0, 16.0, 75.00),    # East Cathedral Spire
    (-12.0, -18.0, 69.00),  # South Tooth Peak
    (-26.0, -4.0, 55.00)    # West Shoulder
]
for sx, sy, sz in critical_summits:
    int_x.append(sx)
    int_y.append(sy)

int_x = np.array(int_x)
int_y = np.array(int_y)
int_z = interp_z(np.column_stack((int_y, int_x)))
int_z[-4] = 83.00 # Exact apex

bound_x = boundary_pts_2d[:, 0]
bound_y = boundary_pts_2d[:, 1]
bound_z = np.full(N_b, Z_BASE)

all_top_pts = np.vstack([
    np.column_stack([bound_x, bound_y, bound_z]),
    np.column_stack([int_x, int_y, int_z])
])
N_top = len(all_top_pts)

d_top = Delaunay(all_top_pts[:, :2])
top_centers = np.mean(all_top_pts[d_top.simplices, :2], axis=1)
d_top_edges = np.zeros((6, len(top_centers)))
for k in range(6):
    d_top_edges[k] = APOTHEM - (top_centers[:, 0] * normals[k][0] + top_centers[:, 1] * normals[k][1])
mask_top = np.min(d_top_edges, axis=0) >= -0.05
top_faces = d_top.simplices[mask_top]

# Bottom with Magnet Pocket (Ø 10.2mm x 5.2mm deep)
N_mag = 36
r_mag = 5.1
mag_angles = np.linspace(0, 2 * math.pi, N_mag, endpoint=False)
mag_rim_pts = np.column_stack([r_mag * np.cos(mag_angles), r_mag * np.sin(mag_angles), np.zeros(N_mag)])
mag_ceil_pts = np.column_stack([r_mag * np.cos(mag_angles), r_mag * np.sin(mag_angles), np.full(N_mag, 5.2)])
mag_ceil_center = np.array([[0.0, 0.0, 5.2]])
bottom_bound_pts = np.column_stack([bound_x, bound_y, np.zeros(N_b)])

# Annulus sampling for bottom Delaunay
r_rings = np.linspace(8.0, 48.0, 8)
int_bot_pts = []
for r in r_rings:
    n_pts = int(r * 2.5)
    for a in np.linspace(0, 2 * math.pi, n_pts, endpoint=False):
        pt = np.array([r * math.cos(a), r * math.sin(a)])
        d_ed = [APOTHEM - (pt[0]*n[0] + pt[1]*n[1]) for n in normals]
        if min(d_ed) >= 2.0:
            int_bot_pts.append(pt)
int_bot_pts = np.array(int_bot_pts)

all_bot_2d = np.vstack([boundary_pts_2d, mag_rim_pts[:, :2], int_bot_pts])
d_bot = Delaunay(all_bot_2d)
bot_centers = np.mean(all_bot_2d[d_bot.simplices], axis=1)
r_c = np.hypot(bot_centers[:, 0], bot_centers[:, 1])
d_bot_edges = np.zeros((6, len(bot_centers)))
for k in range(6):
    d_bot_edges[k] = APOTHEM - (bot_centers[:, 0]*normals[k][0] + bot_centers[:, 1]*normals[k][1])
mask_bot = (np.min(d_bot_edges, axis=0) >= 0.0) & (r_c >= 5.1)
bot_ring_tris = d_bot.simplices[mask_bot]

offset_bot_bound = len(all_top_pts)
offset_mag_rim = offset_bot_bound + N_b
offset_int_bot = offset_mag_rim + N_mag
offset_mag_ceil = offset_int_bot + len(int_bot_pts)
idx_mag_center = offset_mag_ceil + N_mag

v_total = np.vstack([
    all_top_pts,
    bottom_bound_pts,
    mag_rim_pts,
    np.column_stack([int_bot_pts, np.zeros(len(int_bot_pts))]),
    mag_ceil_pts,
    mag_ceil_center
])

all_faces = list(top_faces)

# Vertical side walls (from Z=25 down to Z=0)
for i in range(N_b):
    top_curr = i
    top_next = (i + 1) % N_b
    bot_curr = offset_bot_bound + i
    bot_next = offset_bot_bound + (i + 1) % N_b
    all_faces.append([top_curr, bot_curr, bot_next])
    all_faces.append([top_curr, bot_next, top_next])

# Bottom ring faces
for tri in bot_ring_tris:
    mapped = []
    for idx in tri:
        if idx < N_b:
            mapped.append(offset_bot_bound + idx)
        elif idx < N_b + N_mag:
            mapped.append(offset_mag_rim + (idx - N_b))
        else:
            mapped.append(offset_int_bot + (idx - N_b - N_mag))
    all_faces.append([mapped[0], mapped[2], mapped[1]])

# Magnet pocket walls
for i in range(N_mag):
    rim_curr = offset_mag_rim + i
    rim_next = offset_mag_rim + (i + 1) % N_mag
    ceil_curr = offset_mag_ceil + i
    ceil_next = offset_mag_ceil + (i + 1) % N_mag
    all_faces.append([rim_curr, ceil_curr, ceil_next])
    all_faces.append([rim_curr, ceil_next, rim_next])

# Magnet ceiling cap
for i in range(N_mag):
    ceil_curr = offset_mag_ceil + i
    ceil_next = offset_mag_ceil + (i + 1) % N_mag
    all_faces.append([ceil_next, ceil_curr, idx_mag_center])

solid_mesh = trimesh.Trimesh(vertices=v_total, faces=all_faces, process=True)

print(f"Solid Monolithic Mesh created:")
print(f"  Watertight: {solid_mesh.is_watertight}")
print(f"  Volume: {solid_mesh.volume / 1000.0:.2f} cm3")
print(f"  Vertices: {len(solid_mesh.vertices):,}")
print(f"  Faces: {len(solid_mesh.faces):,}")
print(f"  Bounds: Z in [{solid_mesh.bounds[0][2]:.2f}, {solid_mesh.bounds[1][2]:.2f}] mm")

# Export watertight STL
stl_path = os.path.join(out_dir, "catan_mountain_solid.stl")
solid_mesh.export(stl_path)
print(f"Saved Master STL: {stl_path}")

# -------------------------------------------------------------
# 2. EVALUATE CONTINUOUS VERTEX COLORS & UV COORDINATES
# -------------------------------------------------------------
print("\n--- Evaluating Continuous 24-bit Vertex Colors & UV Coordinates ---")
verts = solid_mesh.vertices
vx = verts[:, 0]
vy = verts[:, 1]
vz = verts[:, 2]

# Planar UV projection across hexagon bounds:
# u in [0, 1] mapped from [-APOTHEM, APOTHEM]
# v in [0, 1] mapped from [-R, R]
u = np.clip((vx + APOTHEM) / (2 * APOTHEM), 0.0, 1.0)
v = np.clip((vy + R) / (2 * R), 0.0, 1.0)
uvs = np.column_stack([u, v])

# Load 4K diffuse texture array for exact color sampling
diffuse_4k = np.load('scratch/diffuse_rgb_4k.npy')
dim_tex = diffuse_4k.shape[0]

# Sample color at (u, v)
px = np.clip((u * (dim_tex - 1)).astype(int), 0, dim_tex - 1)
py = np.clip((v * (dim_tex - 1)).astype(int), 0, dim_tex - 1)
sampled_rgb = diffuse_4k[py, px] # shape (N_verts, 3)

# Vertices on side walls and bottom base (z < 25.0) get dark charcoal foundation color
is_foundation = vz < 24.95
sampled_rgb[is_foundation] = [0.12, 0.13, 0.15]

# Convert to uint8 (0-255)
colors_u8 = (sampled_rgb * 255.0).astype(np.uint8)
colors_rgba = np.column_stack([colors_u8, np.full(len(verts), 255, dtype=np.uint8)])

# Assign colors to solid_mesh visual
solid_mesh.visual = trimesh.visual.ColorVisuals(mesh=solid_mesh, vertex_colors=colors_rgba)

# -------------------------------------------------------------
# 3. EXPORT TRUE FULL-COLOR PLY
# -------------------------------------------------------------
ply_path = os.path.join(out_dir, "catan_mountain_fullcolor.ply")
solid_mesh.export(ply_path)
print(f"Saved Full-Color PLY (24-bit RGB per vertex): {ply_path} ({os.path.getsize(ply_path)/1024/1024:.2f} MB)")

# -------------------------------------------------------------
# 4. EXPORT TRUE FULL-COLOR OBJ + MTL + TEXTURE MAP
# -------------------------------------------------------------
obj_path = os.path.join(out_dir, "catan_mountain_fullcolor.obj")
mtl_path = os.path.join(out_dir, "catan_mountain.mtl")
tex_rel_name = "catan_mountain_diffuse_4k.png"

# Write MTL
with open(mtl_path, 'w', encoding='utf-8') as f:
    f.write("# Catan Mountain Full Color Material Definition\n")
    f.write("newmtl mountain_fullcolor\n")
    f.write("Ka 1.000 1.000 1.000\n")
    f.write("Kd 1.000 1.000 1.000\n")
    f.write("Ks 0.050 0.050 0.050\n")
    f.write("Ns 15.0\n")
    f.write(f"map_Kd {tex_rel_name}\n")
print(f"Saved MTL: {mtl_path}")

# Write OBJ with UVs
print(f"Writing Full-Color OBJ with {len(verts):,} vertices and {len(solid_mesh.faces):,} faces...")
with open(obj_path, 'w', encoding='utf-8') as f:
    f.write("# Catan Mountain Full Color Textured Model\n")
    f.write("mtllib catan_mountain.mtl\n")
    f.write("usemtl mountain_fullcolor\n")
    f.write("s 1\n")
    
    # Vertices
    for pt in verts:
        f.write(f"v {pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}\n")
    
    # Texture Coordinates
    for uv in uvs:
        f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
    
    # Vertex Normals
    normals_mesh = solid_mesh.vertex_normals
    for vn in normals_mesh:
        f.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")
    
    # Faces (1-indexed, v/vt/vn)
    for tri in solid_mesh.faces:
        v1, v2, v3 = tri[0] + 1, tri[1] + 1, tri[2] + 1
        f.write(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}\n")

print(f"Saved Full-Color OBJ: {obj_path} ({os.path.getsize(obj_path)/1024/1024:.2f} MB)")

# -------------------------------------------------------------
# 5. EXPORT FULL-COLOR 3MF WITH VERTEX COLORS
# -------------------------------------------------------------
mf3_path = os.path.join(out_dir, "catan_mountain_fullcolor.3mf")
solid_mesh.export(mf3_path)
print(f"Saved Full-Color 3MF: {mf3_path} ({os.path.getsize(mf3_path)/1024/1024:.2f} MB)")

print(f"\n=== PHASE 2 COMPLETE ({time.time()-t0:.2f}s) ===")
