import os
import math
import time
import zipfile
import shutil
import numpy as np
import scipy.spatial as spatial
import scipy.interpolate as interp
import trimesh

print("==================================================================")
print("  BUILDING FLAWLESS 5.11M WATERTIGHT SOLID & PRODUCTION SUITE     ")
print("==================================================================")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

R = 64.0
apothem = R * math.cos(math.radians(30))
z_base = 25.0
z_peak = 83.0

v_angles = (30.0 + 60.0 * np.arange(6)) * np.pi / 180.0
hex_vx = R * np.cos(v_angles)
hex_vy = R * np.sin(v_angles)

# Edges sampling (24 points per edge = 144 boundary points)
pts_per_edge = 24
boundary_pts_2d = []
for k in range(6):
    p1 = np.array([hex_vx[k], hex_vy[k]])
    p2 = np.array([hex_vx[(k+1)%6], hex_vy[(k+1)%6]])
    for s in np.linspace(0, 1, pts_per_edge, endpoint=False):
        boundary_pts_2d.append((1.0 - s) * p1 + s * p2)
boundary_pts_2d = np.array(boundary_pts_2d, dtype=np.float32)
N_b = len(boundary_pts_2d)

# Internal grid for > 5,000,000 faces: res = 1710
res = 1710
ny = int(res * (2 * R) / (2 * apothem))
xs = np.linspace(-apothem + 1.2, apothem - 1.2, res, dtype=np.float32)
ys = np.linspace(-R + 2.0, R - 2.0, ny, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)

angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
normals = np.column_stack([np.cos(angles), np.sin(angles)])
d_edges = np.zeros((6, len(ys), len(xs)), dtype=np.float32)
for k in range(6):
    d_edges[k] = apothem - (X * normals[k, 0] + Y * normals[k, 1])
dist_edge = np.min(d_edges, axis=0)
mask_internal = dist_edge >= 1.2

int_x = X[mask_internal]
int_y = Y[mask_internal]
print(f"Top surface interior vertices: {len(int_x):,}")

# Load flawless Z topography
Z_master = np.load('scratch/Z_top_master_2m.npy')
ny_m, nx_m = Z_master.shape
xs_m = np.linspace(-apothem, apothem, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

print("Interpolating Z elevations on 5M vertex grid...")
t_int = time.time()
int_z = interp_z(np.column_stack((int_y, int_x))).astype(np.float32)
print(f"Z interpolated in {time.time()-t_int:.2f}s")

bound_x = boundary_pts_2d[:, 0]
bound_y = boundary_pts_2d[:, 1]
bound_z = np.full(N_b, z_base, dtype=np.float32)

all_top_pts = np.vstack([
    np.column_stack([bound_x, bound_y, bound_z]),
    np.column_stack([int_x, int_y, int_z])
])

print(f"Triangulating {len(all_top_pts):,} top vertices via Delaunay...")
t_del = time.time()
d_top = spatial.Delaunay(all_top_pts[:, :2])
top_centers = np.mean(all_top_pts[d_top.simplices, :2], axis=1)
d_top_edges = np.zeros((6, len(top_centers)))
for k in range(6):
    d_top_edges[k] = apothem - (top_centers[:, 0] * normals[k, 0] + top_centers[:, 1] * normals[k, 1])
mask_top = np.min(d_top_edges, axis=0) >= -0.05
top_faces = d_top.simplices[mask_top]
print(f"Top Delaunay completed in {time.time()-t_del:.2f}s! Top faces: {len(top_faces):,}")

# Base & Magnet cavity
N_mag = 32
r_mag = 5.1
mag_angles = np.linspace(0, 2*np.pi, N_mag, endpoint=False)
mag_rim_pts = np.column_stack([r_mag * np.cos(mag_angles), r_mag * np.sin(mag_angles), np.zeros(N_mag, dtype=np.float32)])
mag_ceil_pts = np.column_stack([r_mag * np.cos(mag_angles), r_mag * np.sin(mag_angles), np.full(N_mag, 5.2, dtype=np.float32)])
mag_ceil_center = np.array([[0.0, 0.0, 5.2]], dtype=np.float32)
bottom_bound_pts = np.column_stack([bound_x, bound_y, np.zeros(N_b, dtype=np.float32)])

r_rings = np.linspace(8.0, 48.0, 8)
int_bot_pts = []
for r in r_rings:
    n_pts = int(r * 2.5)
    angs = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    for a in angs:
        pt = np.array([r * np.cos(a), r * np.sin(a)])
        d_ed = apothem - (pt[0]*normals[:, 0] + pt[1]*normals[:, 1])
        if np.min(d_ed) >= 2.0:
            int_bot_pts.append(pt)
int_bot_pts = np.array(int_bot_pts, dtype=np.float32)

all_bot_2d = np.vstack([boundary_pts_2d, mag_rim_pts[:, :2], int_bot_pts])
d_bot = spatial.Delaunay(all_bot_2d)

bot_centers = np.mean(all_bot_2d[d_bot.simplices], axis=1)
r_c = np.hypot(bot_centers[:, 0], bot_centers[:, 1])
d_bot_edges = np.zeros((6, len(bot_centers)))
for k in range(6):
    d_bot_edges[k] = apothem - (bot_centers[:, 0]*normals[k, 0] + bot_centers[:, 1]*normals[k, 1])
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
    np.column_stack([int_bot_pts, np.zeros(len(int_bot_pts), dtype=np.float32)]),
    mag_ceil_pts,
    mag_ceil_center
])

all_faces = list(top_faces)

# Side walls
for i in range(N_b):
    top_curr = i; top_next = (i + 1) % N_b
    bot_curr = offset_bot_bound + i; bot_next = offset_bot_bound + (i + 1) % N_b
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

# Magnet cylinder wall
for i in range(N_mag):
    rim_curr = offset_mag_rim + i; rim_next = offset_mag_rim + (i + 1) % N_mag
    ceil_curr = offset_mag_ceil + i; ceil_next = offset_mag_ceil + (i + 1) % N_mag
    all_faces.append([rim_curr, ceil_curr, ceil_next])
    all_faces.append([rim_curr, ceil_next, rim_next])

# Magnet ceiling cap
for i in range(N_mag):
    ceil_curr = offset_mag_ceil + i; ceil_next = offset_mag_ceil + (i + 1) % N_mag
    all_faces.append([ceil_next, ceil_curr, idx_mag_center])

all_faces = np.array(all_faces, dtype=np.int32)
print(f"Total Assembled Solid Mesh: {len(v_total):,} vertices, {len(all_faces):,} faces")

mesh = trimesh.Trimesh(vertices=v_total, faces=all_faces, process=True)
print(f"Watertight manifold: {mesh.is_watertight}")
print(f"Euler Characteristic: {mesh.euler_number}")
print(f"Volume: {mesh.volume / 1000.0:.2f} cm3")

assert mesh.is_watertight, "ERROR: Mesh is not watertight!"
assert mesh.euler_number == 2, f"ERROR: Euler number is {mesh.euler_number}, expected 2!"
assert len(mesh.faces) >= 5000000, f"ERROR: Faces {len(mesh.faces)} < 5,000,000!"

# 1. Export Solid STL
out_stl = os.path.join(out_dir, "catan_mountain_solid.stl")
mesh.export(out_stl)
print(f"Exported Master Solid STL: {out_stl} ({os.path.getsize(out_stl)/1024/1024:.2f} MB)")

# 2. Export 24-bit TrueColor PLY with continuous vertex colors
print("\n--- 2. Exporting 5.11M TrueColor PLY ---")
t_ply = time.time()
diffuse_4k = np.load('scratch/diffuse_4k_array.npy')
dim_tex = diffuse_4k.shape[0]

vx = mesh.vertices[:, 0]
vy = mesh.vertices[:, 1]
vz = mesh.vertices[:, 2]

u_tex = np.clip((vx - (-apothem)) / (2.0 * apothem), 0.0, 1.0)
v_tex = np.clip((vy - (-R)) / (2.0 * R), 0.0, 1.0)

px = np.clip((u_tex * (dim_tex - 1)).astype(np.int32), 0, dim_tex - 1)
py = np.clip((v_tex * (dim_tex - 1)).astype(np.int32), 0, dim_tex - 1)

vert_rgb = (diffuse_4k[py, px] * 255.0).astype(np.uint8)
vert_rgb[vz < 24.5] = [36, 38, 42]

color_mesh = mesh.copy()
color_mesh.visual.vertex_colors = np.column_stack([vert_rgb, np.full(len(vert_rgb), 255, dtype=np.uint8)])
out_ply = os.path.join(out_dir, "catan_mountain_fullcolor.ply")
color_mesh.export(out_ply)
print(f"Exported Full-Color PLY: {out_ply} ({os.path.getsize(out_ply)/1024/1024:.2f} MB in {time.time()-t_ply:.2f}s)")

# 3. Export Wavefront OBJ + MTL
print("\n--- 3. Exporting 5.11M Wavefront OBJ + MTL ---")
out_obj = os.path.join(out_dir, "catan_mountain_fullcolor.obj")
out_mtl = os.path.join(out_dir, "catan_mountain.mtl")

with open(out_mtl, 'w') as f_mtl:
    f_mtl.write("# Catan Mountain Full-Color Material\n")
    f_mtl.write("newmtl MountainMaterial\n")
    f_mtl.write("Ka 1.000 1.000 1.000\n")
    f_mtl.write("Kd 1.000 1.000 1.000\n")
    f_mtl.write("Ks 0.050 0.050 0.050\n")
    f_mtl.write("Ns 12.0\n")
    f_mtl.write("map_Kd catan_mountain_diffuse_4k.png\n")
    f_mtl.write("norm catan_mountain_normal_4k.png\n")

print(f"Writing OBJ file with {len(mesh.vertices):,} vertices and {len(mesh.faces):,} faces...")
t_obj = time.time()
with open(out_obj, 'w') as f_obj:
    f_obj.write("# Catan 3D Mountain Full-Color Tile (5.11M Triangles)\n")
    f_obj.write("mtllib catan_mountain.mtl\n")
    f_obj.write("o CatanMountain\n")
    for v in mesh.vertices:
        f_obj.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
    for u, v in zip(u_tex, v_tex):
        f_obj.write(f"vt {u:.6f} {v:.6f}\n")
    f_obj.write("usemtl MountainMaterial\n")
    f_obj.write("s 1\n")
    for face in mesh.faces:
        f0, f1, f2 = face[0] + 1, face[1] + 1, face[2] + 1
        f_obj.write(f"f {f0}/{f0} {f1}/{f1} {f2}/{f2}\n")
print(f"Exported Master OBJ: {out_obj} ({os.path.getsize(out_obj)/1024/1024:.2f} MB in {time.time()-t_obj:.2f}s)")

# 4. Export Spec-Compliant 3MF Archive
print("\n--- 4. Packaging 5.11M Full-Color 3MF Archive ---")
t_3mf = time.time()
out_3mf = os.path.join(out_dir, "catan_mountain_fullcolor.3mf")
with open(os.path.join(out_dir, "catan_mountain_diffuse_4k.png"), "rb") as f_png:
    png_bytes = f_png.read()

model_xml_path = "scratch/3dmodel.model"
with open(model_xml_path, 'w', encoding='utf-8') as f_m:
    f_m.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f_m.write('<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">\n')
    f_m.write('  <resources>\n')
    f_m.write('    <m:texture2d id="1" path="/3D/Textures/catan_mountain_diffuse_4k.png" contenttype="image/png" />\n')
    f_m.write('    <m:texture2dgroup id="2" texid="1">\n')
    for u, v in zip(u_tex, v_tex):
        f_m.write(f'      <m:tex2coord u="{u:.5f}" v="{v:.5f}" />\n')
    f_m.write('    </m:texture2dgroup>\n')
    f_m.write('    <object id="3" type="model">\n')
    f_m.write('      <mesh>\n')
    f_m.write('        <vertices>\n')
    for v in mesh.vertices:
        f_m.write(f'          <vertex x="{v[0]:.3f}" y="{v[1]:.3f}" z="{v[2]:.3f}" />\n')
    f_m.write('        </vertices>\n')
    f_m.write('        <triangles>\n')
    for face in mesh.faces:
        f0, f1, f2 = face[0], face[1], face[2]
        f_m.write(f'          <triangle v1="{f0}" v2="{f1}" v3="{f2}" pid="2" p1="{f0}" p2="{f1}" p3="{f2}" />\n')
    f_m.write('        </triangles>\n')
    f_m.write('      </mesh>\n')
    f_m.write('    </object>\n')
    f_m.write('  </resources>\n')
    f_m.write('  <build>\n')
    f_m.write('    <item objectid="3" />\n')
    f_m.write('  </build>\n')
    f_m.write('</model>\n')

with zipfile.ZipFile(out_3mf, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    z.write(model_xml_path, arcname="3D/3dmodel.model")
    z.writestr("3D/Textures/catan_mountain_diffuse_4k.png", png_bytes)
    z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/><Default Extension="png" ContentType="image/png"/></Types>')
    z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')

print(f"Exported Master 3MF: {out_3mf} ({os.path.getsize(out_3mf)/1024/1024:.2f} MB in {time.time()-t_3mf:.2f}s)")
os.remove(model_xml_path)

print(f"\nALL 5.11M PRODUCTION DELIVERABLES COMPLETED SUCCESSFULLY IN {time.time()-t0:.2f}s!")
