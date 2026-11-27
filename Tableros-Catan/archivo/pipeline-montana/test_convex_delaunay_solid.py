import math
import time
import numpy as np
import scipy.spatial as spatial
import trimesh

print("=== TESTING EXACT CONVEX DELAUNAY SOLID (2.05M TRIANGLES) ===")
t0 = time.time()

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
Z_BASE = 25.0

# 1. Exact mathematical hexagon perimeter points
# 6 corners at angles 30, 90, 150, 210, 270, 330 deg
angles = [(30 + 60*k)*math.pi/180 for k in range(6)]
corners = np.array([[R*math.cos(a), R*math.sin(a)] for a in angles], dtype=np.float32)

pts_per_edge = 300
b_pts = []
for k in range(6):
    c1 = corners[k]
    c2 = corners[(k+1)%6]
    for t in np.linspace(0.0, 1.0, pts_per_edge, endpoint=False):
        b_pts.append((1.0 - t)*c1 + t*c2)
boundary_2d = np.array(b_pts, dtype=np.float32)
N_b = len(boundary_2d) # 1800 points
print(f"Boundary points: {N_b}")

# 2. Interior points on regular grid inside hexagon (d_edge >= 0.12 mm)
# To get ~1.02M interior points, grid size approx 1080 x 1246
nx, ny = 1080, 1246
xs = np.linspace(-APOTHEM, APOTHEM, nx, dtype=np.float32)
ys = np.linspace(-R, R, ny, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)

normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

mask_int = dist_edge >= 0.12
int_x = X[mask_int]
int_y = Y[mask_int]
print(f"Interior points: {len(int_x)}")

# Combine 2D points for top surface: boundary FIRST, then interior
top_pts_2d = np.vstack((boundary_2d, np.column_stack((int_x, int_y))))
N_top = len(top_pts_2d)
print(f"Total top points: {N_top}")

# Run Delaunay on top points
print("Running Delaunay triangulation on top points...")
t_del = time.time()
d_top = spatial.Delaunay(top_pts_2d)
top_faces = d_top.simplices
print(f"Delaunay completed in {time.time()-t_del:.2f}s! Top faces: {len(top_faces)}")

# Sample Z from our master 2M array
Z_master = np.load('scratch/Z_top_master_2m.npy')
ny_m, nx_m = Z_master.shape
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
import scipy.interpolate as interp
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

z_int = interp_z(np.column_stack((int_y, int_x)))
z_bound = np.full(N_b, Z_BASE, dtype=np.float32)
top_z = np.concatenate((z_bound, z_int))

top_verts_3d = np.column_stack((top_pts_2d, top_z))

# Ensure CCW orientation for top faces (+Z normal)
p0 = top_pts_2d[top_faces[:, 0]]
p1 = top_pts_2d[top_faces[:, 1]]
p2 = top_pts_2d[top_faces[:, 2]]
cross2d = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
flip = cross2d < 0
top_faces[flip] = top_faces[flip][:, [0, 2, 1]]

print(f"Top 3D mesh ready: {len(top_verts_3d)} verts, {len(top_faces)} faces")

# 3. Side walls
# Connect boundary top (0..N_b-1) to boundary bottom
bot_bound_3d = np.column_stack((boundary_2d, np.zeros(N_b, dtype=np.float32)))
offset_bot_bound = len(top_verts_3d)

wall_faces = []
for i in range(N_b):
    top_curr = i; top_next = (i + 1) % N_b
    bot_curr = offset_bot_bound + i; bot_next = offset_bot_bound + (i + 1) % N_b
    wall_faces.append([top_curr, bot_curr, bot_next])
    wall_faces.append([top_curr, bot_next, top_next])
wall_faces = np.array(wall_faces, dtype=np.int32)

# 4. Magnet Cavity (r=5.1mm, depth=5.2mm)
N_mag = 48
r_mag = 5.1
h_mag = 5.2
alphas = np.linspace(0, 2*np.pi, N_mag, endpoint=False, dtype=np.float32)

mag_rim_2d = np.column_stack((r_mag * np.cos(alphas), r_mag * np.sin(alphas)))
mag_rim_3d = np.column_stack((mag_rim_2d, np.zeros(N_mag, dtype=np.float32)))
mag_ceil_3d = np.column_stack((mag_rim_2d, np.full(N_mag, h_mag, dtype=np.float32)))
mag_ceil_center = np.array([[0.0, 0.0, h_mag]], dtype=np.float32)

# Bottom interior grid for smooth floor Delaunay
r_rings = np.linspace(8.0, 48.0, 12)
bot_int_pts = []
for r in r_rings:
    n_p = int(r * 3.0)
    for a in np.linspace(0, 2*np.pi, n_p, endpoint=False):
        pt = [r * math.cos(a), r * math.sin(a)]
        d_ed = [APOTHEM - (pt[0]*n[0] + pt[1]*n[1]) for n in normals_hex]
        if min(d_ed) >= 2.0:
            bot_int_pts.append(pt)
bot_int_pts = np.array(bot_int_pts, dtype=np.float32)

all_bot_2d = np.vstack((boundary_2d, mag_rim_2d, bot_int_pts))
d_bot = spatial.Delaunay(all_bot_2d)
bot_centers = np.mean(all_bot_2d[d_bot.simplices], axis=1)
r_c = np.hypot(bot_centers[:, 0], bot_centers[:, 1])
d_bot_edges = np.zeros((6, len(bot_centers)))
for k in range(6):
    d_bot_edges[k] = APOTHEM - (bot_centers[:, 0]*normals_hex[k][0] + bot_centers[:, 1]*normals_hex[k][1])
mask_bot = (np.min(d_bot_edges, axis=0) >= 0.0) & (r_c >= (r_mag - 0.05))
bot_tris = d_bot.simplices[mask_bot]

# Offsets for bottom mesh vertices
offset_mag_rim = offset_bot_bound + N_b
offset_bot_int = offset_mag_rim + N_mag
offset_mag_ceil = offset_bot_int + len(bot_int_pts)
offset_mag_center = offset_mag_ceil + N_mag

# Map floor triangle indices
floor_faces = []
for tri in bot_tris:
    mapped = []
    for idx in tri:
        if idx < N_b: mapped.append(offset_bot_bound + idx)
        elif idx < N_b + N_mag: mapped.append(offset_mag_rim + (idx - N_b))
        else: mapped.append(offset_bot_int + (idx - N_b - N_mag))
    # Invert to face downwards (-Z)
    floor_faces.append([mapped[0], mapped[2], mapped[1]])
floor_faces = np.array(floor_faces, dtype=np.int32)

# Cavity walls & ceiling
cav_wall_faces = []
for i in range(N_mag):
    rim_c = offset_mag_rim + i; rim_n = offset_mag_rim + (i + 1) % N_mag
    ceil_c = offset_mag_ceil + i; ceil_n = offset_mag_ceil + (i + 1) % N_mag
    cav_wall_faces.append([rim_c, ceil_c, ceil_n])
    cav_wall_faces.append([rim_c, ceil_n, rim_n])
cav_wall_faces = np.array(cav_wall_faces, dtype=np.int32)

cav_ceil_faces = []
for i in range(N_mag):
    ceil_c = offset_mag_ceil + i; ceil_n = offset_mag_ceil + (i + 1) % N_mag
    cav_ceil_faces.append([ceil_n, ceil_c, offset_mag_center])
cav_ceil_faces = np.array(cav_ceil_faces, dtype=np.int32)

# Assemble total vertices & faces
v_total = np.vstack((
    top_verts_3d,
    bot_bound_3d,
    mag_rim_3d,
    np.column_stack((bot_int_pts, np.zeros(len(bot_int_pts), dtype=np.float32))),
    mag_ceil_3d,
    mag_ceil_center
))

f_total = np.vstack((
    top_faces,
    wall_faces,
    floor_faces,
    cav_wall_faces,
    cav_ceil_faces
))

print(f"Total Mesh Assembled: {len(v_total):,} vertices, {len(f_total):,} faces in {time.time()-t0:.2f}s")

mesh = trimesh.Trimesh(vertices=v_total, faces=f_total, process=True)
print(f"Watertight: {mesh.is_watertight}")
print(f"Euler Number: {mesh.euler_number}")
print(f"Volume: {mesh.volume/1000.0:.2f} cm3")
