import math
import time
import sys
sys.path.append('.')
sys.path.append('scratch')
import numpy as np
import scipy.spatial as spatial
import trimesh

print("=== VERIFYING FULL WATERTIGHT ASSEMBLY WITH 2M+ FACES ===")
t0 = time.time()

import test_top_mesh as tm

top_verts = tm.top_verts
top_faces = tm.top_faces

# Boundary edges
e1 = top_faces[:, [0, 1]]
e2 = top_faces[:, [1, 2]]
e3 = top_faces[:, [2, 0]]
edges = np.vstack((e1, e2, e3))
edges_sorted = np.sort(edges, axis=1)
packed_undir = edges_sorted[:, 0].astype(np.int64) * 2000000 + edges_sorted[:, 1].astype(np.int64)
unq, counts = np.unique(packed_undir, return_counts=True)
b_packed_set = set(unq[counts == 1])

mask_b = np.array([p in b_packed_set for p in packed_undir])
b_edges = edges[mask_b]
next_map = dict(b_edges)

# Chain ordered loop
start_v = b_edges[0, 0]
curr = start_v
loop = [curr]
while True:
    curr = next_map[curr]
    if curr == start_v:
        break
    loop.append(curr)

N_loop = len(loop)
print(f"Boundary loop chained: {N_loop} vertices")

# 1. Side walls
# For each vertex in loop, create corresponding bottom vertex at Z=0
loop_arr = np.array(loop, dtype=np.int32)
bot_loop_verts = top_verts[loop_arr].copy()
bot_loop_verts[:, 2] = 0.0 # Z = 0

offset_bot = len(top_verts)
bot_loop_idx = np.arange(offset_bot, offset_bot + N_loop, dtype=np.int32)

# Wall faces:
# For step i: top_u = loop[i], top_v = loop[(i+1)%N]
#            bot_u = bot_loop_idx[i], bot_v = bot_loop_idx[(i+1)%N]
u_top = loop_arr
v_top = np.roll(loop_arr, -1)
u_bot = bot_loop_idx
v_bot = np.roll(bot_loop_idx, -1)

wall_tri1 = np.column_stack((u_top, u_bot, v_bot))
wall_tri2 = np.column_stack((u_top, v_bot, v_top))
wall_faces = np.vstack((wall_tri1, wall_tri2))

# 2. Magnet cavity in center (r=5.1mm, depth=5.2mm)
M_cav = 64
alphas = np.linspace(0, 2*np.pi, M_cav, endpoint=False, dtype=np.float32)
r_m = 5.1
h_m = 5.2

cav_bot_verts = np.column_stack((r_m * np.cos(alphas), r_m * np.sin(alphas), np.zeros(M_cav, dtype=np.float32)))
cav_top_verts = np.column_stack((r_m * np.cos(alphas), r_m * np.sin(alphas), np.full(M_cav, h_m, dtype=np.float32)))
cav_center_vert = np.array([[0.0, 0.0, h_m]], dtype=np.float32)

offset_cav_bot = offset_bot + N_loop
offset_cav_top = offset_cav_bot + M_cav
offset_cav_center = offset_cav_top + M_cav

idx_cav_bot = np.arange(offset_cav_bot, offset_cav_bot + M_cav, dtype=np.int32)
idx_cav_top = np.arange(offset_cav_top, offset_cav_top + M_cav, dtype=np.int32)
idx_cav_center = offset_cav_center

# Cavity walls: normal pointing inward toward cavity center (0, 0)
cb_curr = idx_cav_bot
cb_next = np.roll(idx_cav_bot, -1)
ct_curr = idx_cav_top
ct_next = np.roll(idx_cav_top, -1)

cav_wall_tri1 = np.column_stack((cb_curr, ct_curr, ct_next))
cav_wall_tri2 = np.column_stack((cb_curr, ct_next, cb_next))
cav_wall_faces = np.vstack((cav_wall_tri1, cav_wall_tri2))

# Cavity ceiling: normal pointing down (-Z)
cav_ceil_faces = np.column_stack((np.full(M_cav, idx_cav_center, dtype=np.int32), ct_curr, ct_next))

# 3. Bottom floor triangulation (between bot_loop and cav_bot)
print("Triangulating bottom floor with 2D Delaunay...")
pts2d_outer = bot_loop_verts[:, :2]
pts2d_inner = cav_bot_verts[:, :2]
pts2d_all = np.vstack((pts2d_outer, pts2d_inner))

delaunay = spatial.Delaunay(pts2d_all)
tri_cand = delaunay.simplices

# Filter triangles that are strictly inside outer loop and outside cavity circle
tri_centers = np.mean(pts2d_all[tri_cand], axis=1)
r_centers = np.hypot(tri_centers[:, 0], tri_centers[:, 1])

# Point in polygon for outer loop using distance to hex
R = 64.0; APOTHEM = R * math.cos(math.radians(30))
normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_c = np.full(len(tri_centers), 999.0)
for nx_h, ny_h in normals_hex:
    dist_c = np.minimum(dist_c, APOTHEM - (tri_centers[:, 0] * nx_h + tri_centers[:, 1] * ny_h))

floor_mask = (dist_c >= -0.1) & (r_centers >= (r_m - 0.05))
floor_tri = tri_cand[floor_mask]

# Remap indices to global mesh indices:
# 0..N_loop-1 -> bot_loop_idx
# N_loop..N_loop+M_cav-1 -> idx_cav_bot
remap = np.zeros(len(pts2d_all), dtype=np.int32)
remap[:N_loop] = bot_loop_idx
remap[N_loop:] = idx_cav_bot

floor_faces = remap[floor_tri]

# Ensure downward normal for bottom floor (-Z)
p0 = pts2d_all[floor_tri[:, 0]]
p1 = pts2d_all[floor_tri[:, 1]]
p2 = pts2d_all[floor_tri[:, 2]]
cross2d = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
# If cross > 0, triangle is CCW in 2D, normal is +Z. We want -Z, so swap vertices 1 and 2
flip = cross2d > 0
floor_faces[flip] = floor_faces[flip][:, [0, 2, 1]]

print(f"Bottom floor triangulated: {len(floor_faces)} faces")

# Combine all vertices and faces
all_verts = np.vstack((top_verts, bot_loop_verts, cav_bot_verts, cav_top_verts, cav_center_vert))
all_faces = np.vstack((top_faces, wall_faces, cav_wall_faces, cav_ceil_faces, floor_faces))

print(f"Total mesh assembled: {len(all_verts)} vertices, {len(all_faces)} faces in {time.time()-t0:.2f}s")

# Test with trimesh
mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
print(f"Trimesh created! is_watertight: {mesh.is_watertight}, Euler: {mesh.euler_number}, Volume: {mesh.volume/1000.0:.2f} cm3")
