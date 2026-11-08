import numpy as np
import trimesh
from scipy.spatial import Delaunay
import scipy.interpolate as interp
import os

print("--- Testing Watertight Solid Mesh with Real LiDAR DEM ---")

Z_master = np.load('scratch/Z_top_master.npy')
ny_m, nx_m = Z_master.shape
R = 64.0
apothem = R * np.cos(np.pi / 6) # ~55.4256 mm
z_base = 25.0
z_peak = 83.0

xs_m = np.linspace(-apothem, apothem, nx_m)
ys_m = np.linspace(-R, R, ny_m)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

# Hexagon corners
v_angles = (30.0 + 60.0 * np.arange(6)) * np.pi / 180.0
hex_vx = R * np.cos(v_angles)
hex_vy = R * np.sin(v_angles)

# Perimeter points (32 per edge = 192 pts)
pts_per_edge = 32
boundary_pts_2d = []
for k in range(6):
    p1 = np.array([hex_vx[k], hex_vy[k]])
    p2 = np.array([hex_vx[(k+1)%6], hex_vy[(k+1)%6]])
    for s in np.linspace(0, 1, pts_per_edge, endpoint=False):
        boundary_pts_2d.append((1.0 - s) * p1 + s * p2)
boundary_pts_2d = np.array(boundary_pts_2d)
N_b = len(boundary_pts_2d)

# Internal grid: res=201
res = 201
xs = np.linspace(-apothem + 1.0, apothem - 1.0, res)
ys = np.linspace(-R + 1.8, R - 1.8, int(res * (2 * R) / (2 * apothem)))
X, Y = np.meshgrid(xs, ys)

angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
normals = np.column_stack([np.cos(angles), np.sin(angles)])
d_edges = np.zeros((6, len(ys), len(xs)))
for k in range(6):
    d_edges[k] = apothem - (X * normals[k, 0] + Y * normals[k, 1])
dist_edge = np.min(d_edges, axis=0)
mask_internal = dist_edge >= 1.0

int_x = X[mask_internal]
int_y = Y[mask_internal]
grid_query = np.column_stack((int_y, int_x))
int_z = interp_z(grid_query)

bound_x = boundary_pts_2d[:, 0]
bound_y = boundary_pts_2d[:, 1]
bound_z = np.full(N_b, z_base)

all_top_pts = np.vstack([
    np.column_stack([bound_x, bound_y, bound_z]),
    np.column_stack([int_x, int_y, int_z])
])

d_top = Delaunay(all_top_pts[:, :2])
top_centers = np.mean(all_top_pts[d_top.simplices, :2], axis=1)
d_top_edges = np.zeros((6, len(top_centers)))
for k in range(6):
    d_top_edges[k] = apothem - (top_centers[:, 0] * normals[k, 0] + top_centers[:, 1] * normals[k, 1])
mask_top = np.min(d_top_edges, axis=0) >= -0.05
top_faces = d_top.simplices[mask_top]

# Bottom Face with Magnet Pocket
N_mag = 36
r_mag = 5.1
mag_angles = np.linspace(0, 2*np.pi, N_mag, endpoint=False)
mag_rim_pts = np.column_stack([r_mag * np.cos(mag_angles), r_mag * np.sin(mag_angles), np.zeros(N_mag)])
mag_ceil_pts = np.column_stack([r_mag * np.cos(mag_angles), r_mag * np.sin(mag_angles), np.full(N_mag, 5.2)])
mag_ceil_center = np.array([[0.0, 0.0, 5.2]])
bottom_bound_pts = np.column_stack([bound_x, bound_y, np.zeros(N_b)])

# Rings on bottom face
r_rings = np.linspace(8.0, 48.0, 8)
int_bot_pts = []
for r in r_rings:
    n_pts = int(r * 2.5)
    angs = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    for a in angs:
        pt = np.array([r * np.cos(a), r * np.sin(a)])
        d_ed = apothem - (pt[0]*normals[:,0] + pt[1]*normals[:,1])
        if np.min(d_ed) >= 2.0:
            int_bot_pts.append(pt)
int_bot_pts = np.array(int_bot_pts)

all_bot_2d = np.vstack([boundary_pts_2d, mag_rim_pts[:, :2], int_bot_pts])
d_bot = Delaunay(all_bot_2d)
bot_centers = np.mean(all_bot_2d[d_bot.simplices], axis=1)
r_c = np.hypot(bot_centers[:, 0], bot_centers[:, 1])
d_bot_edges = np.zeros((6, len(bot_centers)))
for k in range(6):
    d_bot_edges[k] = apothem - (bot_centers[:, 0]*normals[k, 0] + bot_centers[:, 1]*normals[k, 1])
mask_bot = (np.min(d_bot_edges, axis=0) >= 0.0) & (r_c >= 5.1)
bot_ring_tris = d_bot.simplices[mask_bot]

# Master vertex table
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

# Vertical side walls
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

# Magnet cylinder wall
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

print("=== SOLID MESH VALIDATION ===")
print(f"  Vertices: {len(solid_mesh.vertices):,}")
print(f"  Faces: {len(solid_mesh.faces):,}")
print(f"  Is Watertight: {solid_mesh.is_watertight}")
print(f"  Volume: {solid_mesh.volume / 1000.0:.2f} cm3")
print(f"  Euler number: {solid_mesh.euler_number}")
print(f"  Z bounds: {solid_mesh.bounds[0][2]:.2f} mm to {solid_mesh.bounds[1][2]:.2f} mm")
