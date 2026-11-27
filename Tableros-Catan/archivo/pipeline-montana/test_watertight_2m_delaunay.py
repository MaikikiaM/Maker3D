import math
import time
import numpy as np
import scipy.spatial as spatial
import trimesh

print("=== VERIFYING 2M+ WATERTIGHT SOLID MESH ===")
t0 = time.time()

R = 64.0
APOTHEM = R * math.cos(math.radians(30))
pts_per_edge = 60
angles = [(30 + 60*k)*math.pi/180 for k in range(6)]
corners = np.array([[R*math.cos(a), R*math.sin(a)] for a in angles])

b_pts = []
for k in range(6):
    c1 = corners[k]; c2 = corners[(k+1)%6]
    for s in np.linspace(0.0, 1.0, pts_per_edge, endpoint=False):
        b_pts.append((1.0 - s)*c1 + s*c2)
boundary_2d = np.array(b_pts, dtype=np.float32)
N_b = len(boundary_2d) # 360

# Denser grid: 1100 x 1270 -> ~1.025M interior points -> > 2.05M top faces!
nx, ny = 1100, 1270
xs = np.linspace(-APOTHEM, APOTHEM, nx, dtype=np.float32)
ys = np.linspace(-R, R, ny, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)
normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

mask_int = dist_edge > 0.6
int_x = X[mask_int]; int_y = Y[mask_int]

# Interpolate Z
Z_master = np.load('scratch/Z_top_master_2m.npy')
ny_m, nx_m = Z_master.shape
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
import scipy.interpolate as interp
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

z_int = interp_z(np.column_stack((int_y, int_x)))
z_bound = np.full(N_b, 25.0, dtype=np.float32)

all_pts_2d = np.vstack((boundary_2d, np.column_stack((int_x, int_y))))
top_z = np.concatenate((z_bound, z_int))
top_verts = np.column_stack((all_pts_2d, top_z))

print(f"Top vertices: {len(top_verts):,}")

# Delaunay on top
d_top = spatial.Delaunay(all_pts_2d)
top_faces = d_top.simplices
# Orientation CCW (+Z)
p0 = all_pts_2d[top_faces[:, 0]]
p1 = all_pts_2d[top_faces[:, 1]]
p2 = all_pts_2d[top_faces[:, 2]]
cross2d = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
flip = cross2d < 0
top_faces[flip] = top_faces[flip][:, [0, 2, 1]]
print(f"Top faces: {len(top_faces):,}")

# Side walls
bot_bound_verts = np.column_stack((boundary_2d, np.zeros(N_b, dtype=np.float32)))
idx_bot_bound = len(top_verts)

wall_faces = []
for i in range(N_b):
    top_curr = i; top_next = (i + 1) % N_b
    bot_curr = idx_bot_bound + i; bot_next = idx_bot_bound + (i + 1) % N_b
    wall_faces.append([top_curr, bot_curr, bot_next])
    wall_faces.append([top_curr, bot_next, top_next])

# Magnet cavity
N_mag = 48
r_mag = 5.1
h_mag = 5.2
alphas = np.linspace(0, 2*np.pi, N_mag, endpoint=False, dtype=np.float32)
mag_rim_2d = np.column_stack((r_mag * np.cos(alphas), r_mag * np.sin(alphas)))
mag_rim_verts = np.column_stack((mag_rim_2d, np.zeros(N_mag, dtype=np.float32)))
mag_ceil_verts = np.column_stack((mag_rim_2d, np.full(N_mag, h_mag, dtype=np.float32)))
mag_ceil_center = np.array([[0.0, 0.0, h_mag]], dtype=np.float32)

idx_mag_rim = idx_bot_bound + N_b
idx_mag_ceil = idx_mag_rim + N_mag
idx_mag_center = idx_mag_ceil + N_mag

# Bottom annulus Delaunay
bot_2d = np.vstack((boundary_2d, mag_rim_2d))
d_bot = spatial.Delaunay(bot_2d)
cent_bot = bot_2d[d_bot.simplices].mean(axis=1)
r_cent = np.hypot(cent_bot[:, 0], cent_bot[:, 1])
d_edge_cent = np.full(len(cent_bot), 999.0)
for nx_h, ny_h in normals:
    d = APOTHEM - (cent_bot[:, 0] * nx_h + cent_bot[:, 1] * ny_h)
    d_edge_cent = np.minimum(d_edge_cent, d)
valid_bot = (r_cent >= r_mag * 0.98) & (d_edge_cent >= -0.01)

bot_faces = []
for s in d_bot.simplices[valid_bot]:
    mapped = []
    for idx in s:
        if idx < N_b:
            mapped.append(idx_bot_bound + idx)
        else:
            mapped.append(idx_mag_rim + (idx - N_b))
    # Downward normal (-Z)
    p0 = bot_2d[s[0]]
    p1 = bot_2d[s[1]]
    p2 = bot_2d[s[2]]
    cross_z = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
    if cross_z > 0:
        mapped = [mapped[0], mapped[2], mapped[1]]
    bot_faces.append(mapped)

# Magnet walls & ceiling
mag_wall_faces = []
for i in range(N_mag):
    r1 = idx_mag_rim + i; r2 = idx_mag_rim + (i + 1) % N_mag
    c1 = idx_mag_ceil + i; c2 = idx_mag_ceil + (i + 1) % N_mag
    mag_wall_faces.append([r1, c2, c1])
    mag_wall_faces.append([r1, r2, c2])

mag_ceil_faces = []
for i in range(N_mag):
    c1 = idx_mag_ceil + i; c2 = idx_mag_ceil + (i + 1) % N_mag
    mag_ceil_faces.append([c1, c2, idx_mag_center])

all_verts = np.vstack((top_verts, bot_bound_verts, mag_rim_verts, mag_ceil_verts, mag_ceil_center))
all_faces = np.vstack((top_faces, wall_faces, bot_faces, mag_wall_faces, mag_ceil_faces))

print(f"Total Vertices: {len(all_verts):,}")
print(f"Total Faces: {len(all_faces):,}")

mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=True)
print(f"Watertight: {mesh.is_watertight}")
print(f"Euler Number: {mesh.euler_number}")
print(f"Volume: {mesh.volume/1000.0:.2f} cm3")
