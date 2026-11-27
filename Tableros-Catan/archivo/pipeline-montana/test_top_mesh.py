import math
import time
import numpy as np
import scipy.spatial as spatial
import trimesh

print("=== TESTING 2M+ WATERTIGHT SOLID MESH GENERATION ===")
t0 = time.time()

R = 64.0
APOTHEM = R * math.cos(math.radians(30))
nx, ny = 1150, 1328
xs = np.linspace(-APOTHEM, APOTHEM, nx, dtype=np.float32)
ys = np.linspace(-R, R, ny, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)

Z = np.load('scratch/Z_top_master_2m.npy')

normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

mask = dist_edge >= 0.0

# Identify quads where all 4 corners are inside
c00 = mask[:-1, :-1]
c10 = mask[:-1, 1:]
c11 = mask[1:, 1:]
c01 = mask[1:, :-1]
quad_mask = c00 & c10 & c11 & c01

print(f"Quads inside hexagon: {np.sum(quad_mask)}")
print(f"Top surface triangles: {np.sum(quad_mask) * 2}")

# Index mapping for vertices used by top surface
used_verts = np.zeros_like(mask, dtype=bool)
used_verts[:-1, :-1] |= quad_mask
used_verts[:-1, 1:]  |= quad_mask
used_verts[1:, 1:]   |= quad_mask
used_verts[1:, :-1]  |= quad_mask

vert_idx = np.full_like(mask, -1, dtype=np.int32)
num_v = np.sum(used_verts)
vert_idx[used_verts] = np.arange(num_v, dtype=np.int32)
print(f"Top surface vertices: {num_v}")

# Extract top triangles
qy, qx = np.where(quad_mask)
v00 = vert_idx[qy, qx]
v10 = vert_idx[qy, qx + 1]
v11 = vert_idx[qy + 1, qx + 1]
v01 = vert_idx[qy + 1, qx]

# Two triangles per quad: (v00, v10, v11) and (v00, v11, v01)
tri1 = np.column_stack((v00, v10, v11))
tri2 = np.column_stack((v00, v11, v01))
top_faces = np.vstack((tri1, tri2))

# Top vertices coordinates
top_verts = np.column_stack((X[used_verts], Y[used_verts], Z[used_verts]))

print(f"Top mesh assembled in {time.time()-t0:.2f}s: {len(top_verts)} vertices, {len(top_faces)} faces")
