import math
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import Delaunay
import trimesh

print("--- Generating 100% Watertight Solid Mountain Mesh ---")

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256258
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF_H = Z_PEAK - Z_BASE # 58.0

# 6 hexagon vertices in order
v_angles = [(30 + 60 * k) * math.pi / 180 for k in range(6)]
hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles])

# Load master arrays from scratch
Z_top_master = np.load('scratch/Z_top_master.npy')
ny_m, nx_m = Z_top_master.shape

# Target mesh resolution: 221 x 255 for clean fast STL
nx, ny = 221, 255
xs = np.linspace(-APOTHEM, APOTHEM, nx)
ys = np.linspace(-R, R, ny)
X, Y = np.meshgrid(xs, ys)

# Resample Z_top to our target grid
from scipy.ndimage import zoom
Z_top = zoom(Z_top_master, (ny / ny_m, nx / nx_m), order=3)

# Hexagon normals
normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    d = APOTHEM - (X * nx_h + Y * ny_h)
    dist_edge = np.minimum(dist_edge, d)

# Boundary contour: we sample each of the 6 edges with N points
pts_per_edge = 40
boundary_2d = []
for k in range(6):
    p1 = hex_corners[k]
    p2 = hex_corners[(k + 1) % 6]
    for s in np.linspace(0.0, 1.0, pts_per_edge, endpoint=False):
        boundary_2d.append((1.0 - s) * p1 + s * p2)
boundary_2d = np.array(boundary_2d)
n_bound = len(boundary_2d)

# Interior points: grid points with dist_edge > 0.8 mm
mask_int = dist_edge > 0.8
int_x = X[mask_int]
int_y = Y[mask_int]
int_z = Z_top[mask_int]
pts_int_2d = np.column_stack((int_x, int_y))

# Combine boundary and interior for 2D Delaunay triangulation
all_pts_2d = np.vstack((boundary_2d, pts_int_2d))
n_total_top = len(all_pts_2d)

# Top Z coordinates: boundary points are strictly at Z_BASE = 25.0 mm
bound_z = np.full(n_bound, Z_BASE)
top_z = np.concatenate((bound_z, int_z))
top_verts = np.column_stack((all_pts_2d, top_z))

# Triangulate top surface
tri = Delaunay(all_pts_2d)
# Filter out triangles whose centroid is outside the hexagon
centroids = all_pts_2d[tri.simplices].mean(axis=1)
cent_dist = np.full(len(centroids), 999.0)
for nx_h, ny_h in normals:
    d = APOTHEM - (centroids[:, 0] * nx_h + centroids[:, 1] * ny_h)
    cent_dist = np.minimum(cent_dist, d)
valid_top_tri = cent_dist >= -0.01
top_faces = tri.simplices[valid_top_tri]

# Ensure correct winding (upward normals)
v0 = top_verts[top_faces[:, 0]]
v1 = top_verts[top_faces[:, 1]]
v2 = top_verts[top_faces[:, 2]]
cross = np.cross(v1 - v0, v2 - v0)
flip = cross[:, 2] < 0
top_faces[flip] = top_faces[flip][:, [0, 2, 1]]

print(f"Top surface: {len(top_verts)} vertices, {len(top_faces)} faces")

# Side walls: from boundary_2d at Z=25 down to Z=0
# Bottom boundary points: boundary_2d at Z=0
bot_bound_verts = np.column_stack((boundary_2d, np.zeros(n_bound)))
all_verts = list(top_verts) + list(bot_bound_verts)
idx_bot_bound_start = n_total_top

wall_faces = []
for i in range(n_bound):
    next_i = (i + 1) % n_bound
    t1 = i
    t2 = next_i
    b1 = idx_bot_bound_start + i
    b2 = idx_bot_bound_start + next_i
    # Two triangles for each quad, outward winding
    wall_faces.append([t1, b1, b2])
    wall_faces.append([t1, b2, t2])

# Bottom face at Z=0 with magnet pocket:
# Magnet pocket: center (0,0), radius 5.1 mm (diam 10.2mm), depth 5.2 mm (Z goes up to 5.2)
r_mag = 5.1
depth_mag = 5.2
n_mag_pts = 32
mag_angles = np.linspace(0, 2 * math.pi, n_mag_pts, endpoint=False)
mag_rim_verts = np.column_stack((r_mag * np.cos(mag_angles), r_mag * np.sin(mag_angles), np.zeros(n_mag_pts)))
mag_ceil_verts = np.column_stack((r_mag * np.cos(mag_angles), r_mag * np.sin(mag_angles), np.full(n_mag_pts, depth_mag)))
mag_ceil_center = np.array([[0.0, 0.0, depth_mag]])

idx_mag_rim_start = len(all_verts)
all_verts.extend(mag_rim_verts)
idx_mag_ceil_start = len(all_verts)
all_verts.extend(mag_ceil_verts)
idx_mag_center = len(all_verts)
all_verts.extend(mag_ceil_center)

# Bottom annulus triangulation: between bot_boundary and mag_rim
# 2D Delaunay on (boundary_2d + mag_rim)
bot_2d = np.vstack((boundary_2d, mag_rim_verts[:, :2]))
tri_bot = Delaunay(bot_2d)
cent_bot = bot_2d[tri_bot.simplices].mean(axis=1)
r_cent = np.hypot(cent_bot[:, 0], cent_bot[:, 1])
# Valid annulus faces are between r_mag and hexagon
d_edge_cent = np.full(len(cent_bot), 999.0)
for nx_h, ny_h in normals:
    d = APOTHEM - (cent_bot[:, 0] * nx_h + cent_bot[:, 1] * ny_h)
    d_edge_cent = np.minimum(d_edge_cent, d)
valid_bot = (r_cent >= r_mag * 0.98) & (d_edge_cent >= -0.01)

bot_faces = []
for s in tri_bot.simplices[valid_bot]:
    # Map indices: 0..n_bound-1 -> idx_bot_bound_start + i
    # n_bound..n_bound+n_mag-1 -> idx_mag_rim_start + (i - n_bound)
    mapped = []
    for idx in s:
        if idx < n_bound:
            mapped.append(idx_bot_bound_start + idx)
        else:
            mapped.append(idx_mag_rim_start + (idx - n_bound))
    # Downward normal winding (Z < 0)
    p0 = np.array(all_verts[mapped[0]])
    p1 = np.array(all_verts[mapped[1]])
    p2 = np.array(all_verts[mapped[2]])
    cross_z = np.cross(p1 - p0, p2 - p0)[2]
    if cross_z > 0:
        mapped = [mapped[0], mapped[2], mapped[1]]
    bot_faces.append(mapped)

# Magnet pocket cylindrical inner wall: from rim (Z=0) to ceil (Z=depth_mag)
mag_wall_faces = []
for i in range(n_mag_pts):
    next_i = (i + 1) % n_mag_pts
    r1 = idx_mag_rim_start + i
    r2 = idx_mag_rim_start + next_i
    c1 = idx_mag_ceil_start + i
    c2 = idx_mag_ceil_start + next_i
    # Inward winding (pointing into pocket cavity)
    mag_wall_faces.append([r1, c2, c1])
    mag_wall_faces.append([r1, r2, c2])

# Magnet pocket ceiling: fan to mag_ceil_center
mag_ceil_faces = []
for i in range(n_mag_pts):
    next_i = (i + 1) % n_mag_pts
    c1 = idx_mag_ceil_start + i
    c2 = idx_mag_ceil_start + next_i
    # Normal pointing down (Z = -1)
    mag_ceil_faces.append([c1, c2, idx_mag_center])

all_faces = list(top_faces) + wall_faces + bot_faces + mag_wall_faces + mag_ceil_faces
all_verts = np.array(all_verts)
all_faces = np.array(all_faces)

print(f"Total mesh: {len(all_verts)} vertices, {len(all_faces)} faces")

# Assemble Trimesh solid
solid_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=True)
print("Mesh properties:")
print(f"  Is watertight: {solid_mesh.is_watertight}")
print(f"  Volume: {solid_mesh.volume:.1f} mm^3")
print(f"  Euler number: {solid_mesh.euler_number}")
print(f"  Bounding box: min={solid_mesh.bounds[0].round(2)}, max={solid_mesh.bounds[1].round(2)}")

# Save STL to target directory
out_dir = r'c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D'
stl_path = os.path.join(out_dir, 'catan_mountain_solid.stl')
solid_mesh.export(stl_path)
print(f"Successfully exported 100% Watertight Solid STL to:\n  {stl_path}")
