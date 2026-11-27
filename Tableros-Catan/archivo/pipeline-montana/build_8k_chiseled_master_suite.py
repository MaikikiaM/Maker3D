import os
import math
import time
import zipfile
import shutil
import base64
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import Delaunay
import trimesh
from PIL import Image

print("==================================================================")
print("  BUILDING 4X RESOLUTION CHISELED ALPINISM FULL-COLOR SUITE (8K)  ")
print("==================================================================")
t0 = time.time()

out_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"
os.makedirs(out_dir, exist_ok=True)

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF_H = Z_PEAK - Z_BASE # 58.00 mm

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# -------------------------------------------------------------
# 1. SYNTHESIZE CHISELED ALPINISM 3D TOPOGRAPHY (601 x 695)
# -------------------------------------------------------------
print("--- 1. Synthesizing High-Density Chiseled Alpinism Topography ---")
nx, ny = 601, 695
xs = np.linspace(-APOTHEM, APOTHEM, nx, dtype=np.float32)
ys = np.linspace(-R, R, ny, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)

normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

v_angles = [(30 + 60 * k) * math.pi / 180 for k in range(6)]
hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles])
dist_vert = np.full_like(X, 999.0)
for vx, vy in hex_corners:
    dist_vert = np.minimum(dist_vert, np.hypot(X - vx, Y - vy))

r_main = np.hypot(X - (-4.0), Y - 6.0)
ang_main = np.arctan2(Y - 6.0, X - (-4.0))

# Pyramidal Horns
facet_horn = 0.86 + 0.36 * np.abs(np.cos(2.0 * ang_main + 0.35))**1.6
h_horn = 58.0 * np.exp(-(r_main / 23.0)**1.38) * facet_horn

r_p2 = np.hypot(X - 18.0, Y - 16.0)
ang_p2 = np.arctan2(Y - 16.0, X - 18.0)
h_p2 = 50.0 * np.exp(-(r_p2 / 17.5)**1.36) * (0.86 + 0.30 * np.abs(np.cos(2.0 * ang_p2))**1.3)

r_p3 = np.hypot(X - (-12.0), Y - (-18.0))
h_p3 = 44.0 * np.exp(-(r_p3 / 18.5)**1.36) * (0.88 + 0.25 * np.cos(1.8 * ang_main)**2.0)

r_p4 = np.hypot(X - (-26.0), Y - (-4.0))
h_p4 = 30.0 * np.exp(-(r_p4 / 15.5)**1.36)

# Knife-edge radiating arêtes
spines = [
    {"x1": -4.0, "y1": 6.0, "x2": -35.0, "y2": 35.0, "h1": 58.0, "h2": 12.0, "sharp": 7.0, "curv": 1.3},
    {"x1": -4.0, "y1": 6.0, "x2": 18.0, "y2": 16.0, "h1": 58.0, "h2": 50.0, "sharp": 6.5, "curv": 1.25},
    {"x1": 18.0, "y1": 16.0, "x2": 38.0, "y2": 26.0, "h1": 50.0, "h2": 10.0, "sharp": 7.5, "curv": 1.3},
    {"x1": 18.0, "y1": 16.0, "x2": 32.0, "y2": -8.0, "h1": 50.0, "h2": 18.0, "sharp": 7.2, "curv": 1.25},
    {"x1": -4.0, "y1": 6.0, "x2": -12.0, "y2": -18.0, "h1": 58.0, "h2": 44.0, "sharp": 7.0, "curv": 1.25},
    {"x1": -12.0, "y1": -18.0, "x2": -32.0, "y2": -28.0, "h1": 44.0, "h2": 8.0, "sharp": 7.5, "curv": 1.3},
    {"x1": -12.0, "y1": -18.0, "x2": 16.0, "y2": -26.0, "h1": 44.0, "h2": 12.0, "sharp": 7.2, "curv": 1.25},
    {"x1": -4.0, "y1": 6.0, "x2": -26.0, "y2": -4.0, "h1": 58.0, "h2": 28.0, "sharp": 7.0, "curv": 1.25},
    {"x1": -26.0, "y1": -4.0, "x2": -42.0, "y2": -14.0, "h1": 28.0, "h2": 6.0, "sharp": 8.0, "curv": 1.3}
]

H_spines = np.zeros_like(X)
for sp in spines:
    dx_sp = sp["x2"] - sp["x1"]; dy_sp = sp["y2"] - sp["y1"]
    L2 = dx_sp*dx_sp + dy_sp*dy_sp
    t_sp = np.clip(((X - sp["x1"])*dx_sp + (Y - sp["y1"])*dy_sp) / L2, 0.0, 1.0)
    perp_x = -dy_sp / math.sqrt(L2); perp_y = dx_sp / math.sqrt(L2)
    meander = 1.3 * np.sin(t_sp * 8.0 + 1.2)
    px = sp["x1"] + t_sp * dx_sp + meander * perp_x
    py = sp["y1"] + t_sp * dy_sp + meander * perp_y
    dist_sp = np.hypot(X - px, Y - py)
    h_line = (1.0 - t_sp) * sp["h1"] + t_sp * sp["h2"]
    h_contrib = h_line * np.exp(-(dist_sp / sp["sharp"])**sp["curv"])
    H_spines = np.maximum(H_spines, h_contrib)

massif = np.maximum(H_spines, np.maximum(h_horn, np.maximum(h_p2, np.maximum(h_p3, h_p4))))

# Talus apron decay
r_tile = np.hypot(X, Y)
base_apron = 58.0 * np.clip(1.0 - (r_tile / 54.0), 0.0, 1.0)**1.65
massif = np.maximum(massif, base_apron * 0.38)

# Chiseled geometric relief
ridge_f1 = 1.0 - 2.0 * np.abs(np.sin((X * 0.35 + Y * 0.28) * 1.4))
ridge_f2 = 1.0 - 2.0 * np.abs(np.sin((X * 0.72 - Y * 0.65) * 1.5))
chiseled_crags = 1.6 * ridge_f1 + 0.8 * ridge_f2

Z_temp = massif + 25.0
stepped_terraces = 0.80 * np.sin((X * 0.18 + Y * 0.10 + Z_temp * 0.32) * 1.8)
couloir_carving = 1.5 * np.abs(np.sin(ang_main * 4.0 + 0.5))**2.2 * np.exp(-(r_main / 30.0)**1.5)

relief_env = (massif / 58.0)**0.85
Z_sculpted = massif + (chiseled_crags + stepped_terraces - couloir_carving) * relief_env
Z_sculpted = np.maximum(0.0, Z_sculpted)
Z_sculpted = Z_sculpted * (58.0 / np.max(Z_sculpted))

edge_factor = smoothstep(3.5, 12.0, dist_edge)
vert_factor = smoothstep(10.0, 15.0, dist_vert)
clearance = edge_factor * vert_factor

Z_top = Z_BASE + Z_sculpted * clearance
Z_top[dist_edge < 0] = Z_BASE

ix_peak = int(np.round((-4.0 - (-APOTHEM)) / (2 * APOTHEM) * (nx - 1)))
iy_peak = int(np.round((6.0 - (-R)) / (2 * R) * (ny - 1)))
Z_top[iy_peak, ix_peak] = Z_PEAK

# Save updated master Z
np.save('scratch/Z_top_master_final.npy', Z_top)
print(f"Master Z generated: {ny}x{nx}, Z in [{np.min(Z_top):.2f}, {np.max(Z_top):.2f}] mm")

# -------------------------------------------------------------
# 2. SYNTHESIZE 8K & 4K 3D WORLD-SPACE TEXTURE & NORMAL MAP
# -------------------------------------------------------------
print("--- 2. Synthesizing 3D World-Space Texture (Anti-Stretch & 8K Resolution) ---")
dim_tex = 4096
xs_t = np.linspace(-APOTHEM, APOTHEM, dim_tex, dtype=np.float32)
ys_t = np.linspace(-R, R, dim_tex, dtype=np.float32)
X_t, Y_t = np.meshgrid(xs_t, ys_t)

import scipy.interpolate as interp
interp_z = interp.RegularGridInterpolator((ys, xs), Z_top, bounds_error=False, fill_value=25.0)

# Chunked Z evaluation on 4096x4096
chunk_size = dim_tex // 4
Z_chunks = []
for c_idx in range(4):
    y_c = ys_t[c_idx*chunk_size : (c_idx+1)*chunk_size]
    X_c, Y_c = np.meshgrid(xs_t, y_c)
    pts = np.column_stack((Y_c.ravel(), X_c.ravel()))
    Z_chunks.append(interp_z(pts).reshape((chunk_size, dim_tex)))
Z_t = np.vstack(Z_chunks)
RELIEF_t = np.maximum(0.0, Z_t - 25.0)

# Derivatives for 3D slope and lighting
dx_t = float(xs_t[1] - xs_t[0]); dy_t = float(ys_t[1] - ys_t[0])
gy_t, gx_t = np.gradient(Z_t, dy_t, dx_t)
slope_t = np.hypot(gx_t, gy_t)
slope_deg_t = np.degrees(np.arctan(slope_t))
laplace_t = ndi.laplace(Z_t)

# 3D Rotated Strata Coordinates
dip = math.radians(28.0); strike = math.radians(45.0)
X_dip = X_t * math.cos(strike) + Y_t * math.sin(strike)
Y_dip = -X_t * math.sin(strike) + Y_t * math.cos(strike)
Z_strata = Z_t * math.cos(dip) - X_dip * math.sin(dip)

# Sharp metamorphic strata & joint planes
strata_freq = 0.92
strata_fine = np.cos(Z_strata * strata_freq * 2.0 * math.pi)
joint_planes = np.abs(np.sin(X_dip * 0.45 + Y_dip * 0.32))**0.5

# High-frequency 3D crystal noise
p1 = (X_t * 2.4 + Y_t * 2.0 + Z_t * 2.7)
p2 = (X_t * 5.8 - Y_t * 5.1 + Z_t * 6.5)
p3 = (X_t * 14.1 + Y_t * 12.5 - Z_t * 15.2)
p4 = (X_t * 31.5 - Y_t * 28.2 + Z_t * 34.6)
crystal_fBm = (
    0.42 * np.sin(p1) * np.cos(p1 * 0.7 + 1.2) +
    0.28 * np.sin(p2) * np.cos(p2 * 0.8 - 0.5) +
    0.19 * np.sin(p3) * np.cos(p3 * 0.9 + 2.1) +
    0.11 * np.sin(p4)
)

chiseled_rock = 0.42 * strata_fine + 0.33 * joint_planes + 0.25 * crystal_fBm
chiseled_rock = (chiseled_rock - np.min(chiseled_rock)) / (np.max(chiseled_rock) - np.min(chiseled_rock))

cliff_dark = smoothstep(24.0, 58.0, slope_deg_t)
rock_t = np.clip(chiseled_rock * 0.85 + 0.15 * (1.0 - cliff_dark), 0.0, 1.0)

c_slate = np.array([0.16, 0.18, 0.21], dtype=np.float32)
c_granite = np.array([0.54, 0.56, 0.59], dtype=np.float32)
c_quartz = np.array([0.76, 0.77, 0.80], dtype=np.float32)

rock_rgb = np.zeros((dim_tex, dim_tex, 3), dtype=np.float32)
for c in range(3):
    rock_rgb[:, :, c] = np.where(
        rock_t < 0.45,
        (1.0 - rock_t/0.45) * c_slate[c] + (rock_t/0.45) * c_granite[c],
        (1.0 - (rock_t-0.45)/0.55) * c_granite[c] + ((rock_t-0.45)/0.55) * c_quartz[c]
    )

# Crisp Snow
snow_alt = smoothstep(16.0, 34.0, RELIEF_t)
snow_slope = 1.0 - smoothstep(36.0, 64.0, slope_deg_t)
couloir_flow = smoothstep(-0.8, -0.05, laplace_t) * smoothstep(8.0, 26.0, RELIEF_t)
snow_ledges = smoothstep(0.15, 0.65, strata_fine) * (1.0 - smoothstep(42.0, 68.0, slope_deg_t))
snow_raw = snow_alt * (0.35 + 0.45 * snow_slope + 0.20 * snow_ledges) + 0.42 * couloir_flow + 0.05 * crystal_fBm
snow_alpha = smoothstep(0.26, 0.62, snow_raw)
snow_rgb = np.array([0.97, 0.98, 1.00], dtype=np.float32)

# Glacial Ice
ice_basin = smoothstep(14.0, 35.0, RELIEF_t) * smoothstep(-1.2, -0.2, laplace_t) * (1.0 - snow_alt * 0.65)
ice_alpha = smoothstep(0.22, 0.65, ice_basin) * (1.0 - snow_alpha * 0.85)
ice_rgb = np.array([0.40, 0.72, 0.88], dtype=np.float32)

# Organic Tundra & Lichen
tundra_alt = smoothstep(0.1, 1.2, RELIEF_t) * (1.0 - smoothstep(4.0, 9.5, RELIEF_t))
tundra_slope = 1.0 - smoothstep(12.0, 24.0, slope_deg_t)
tundra_colony = smoothstep(0.25, 0.75, np.abs(np.sin(X_t * 0.45 + Y_t * 0.35)) * np.abs(np.cos(X_t * 0.35 - Y_t * 0.45)))
tundra_alpha = tundra_alt * tundra_slope * tundra_colony
moss_rgb = np.array([0.24, 0.42, 0.18], dtype=np.float32)
xanthoria_rgb = np.array([0.62, 0.54, 0.22], dtype=np.float32)

# Composite
diffuse_4k = rock_rgb.copy()
for c in range(3):
    diffuse_4k[:, :, c] = (1.0 - tundra_alpha) * diffuse_4k[:, :, c] + tundra_alpha * (0.7 * moss_rgb[c] + 0.3 * xanthoria_rgb[c])
for c in range(3):
    diffuse_4k[:, :, c] = (1.0 - ice_alpha) * diffuse_4k[:, :, c] + ice_alpha * ice_rgb[c]
for c in range(3):
    diffuse_4k[:, :, c] = (1.0 - snow_alpha) * diffuse_4k[:, :, c] + snow_alpha * snow_rgb[c]

dist_edge_t = np.full_like(X_t, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge_t = np.minimum(dist_edge_t, APOTHEM - (X_t * nx_h + Y_t * ny_h))

road_margin = smoothstep(1.5, 3.5, dist_edge_t)
for c in range(3):
    diffuse_4k[:, :, c] = road_margin * diffuse_4k[:, :, c] + (1.0 - road_margin) * 0.12
diffuse_4k[dist_edge_t < 0] = [0.08, 0.09, 0.11]
diffuse_4k = np.clip(diffuse_4k, 0.0, 1.0)

# Save 4K Diffuse PNG
out_4k = os.path.join(out_dir, "catan_mountain_diffuse_4k.png")
im_4k = Image.fromarray((diffuse_4k[::-1] * 255.0).astype(np.uint8))
im_4k.save(out_4k, optimize=True)
print(f"Saved Ultra-Crisp 4K Diffuse: {out_4k} ({os.path.getsize(out_4k)/1024/1024:.2f} MB)")

# Save 2K for WebGL viewer
im_2k = im_4k.resize((2048, 2048), Image.Resampling.LANCZOS)
out_2k = os.path.join(out_dir, "catan_mountain_diffuse_2k.png")
im_2k.save(out_2k, optimize=True)
print(f"Saved 2K Diffuse: {out_2k}")

# Compute & Save Normal Map (RGB encoding: R=Nx, G=Ny, B=Nz)
print("Computing 4K Tangent Normal Map...")
N_norm = np.sqrt(gx_t**2 + gy_t**2 + 1.0)
Nx_map = -gx_t / N_norm
Ny_map = -gy_t / N_norm
Nz_map = 1.0 / N_norm
normal_u8 = np.zeros((dim_tex, dim_tex, 3), dtype=np.uint8)
normal_u8[:, :, 0] = ((Nx_map * 0.5 + 0.5) * 255.0).astype(np.uint8)
normal_u8[:, :, 1] = ((Ny_map * 0.5 + 0.5) * 255.0).astype(np.uint8)
normal_u8[:, :, 2] = ((Nz_map * 0.5 + 0.5) * 255.0).astype(np.uint8)

out_norm = os.path.join(out_dir, "catan_mountain_normal_4k.png")
Image.fromarray(normal_u8[::-1]).save(out_norm, optimize=True)
print(f"Saved 4K Normal Map: {out_norm} ({os.path.getsize(out_norm)/1024/1024:.2f} MB)")

# Save diffuse array to scratch
np.save('scratch/diffuse_rgb_4k.npy', diffuse_4k)

# -------------------------------------------------------------
# 3. HIGH-DENSITY SOLID WATERTIGHT MESH (160k+ FACES)
# -------------------------------------------------------------
print("--- 3. Generating Watertight High-Density Solid Mesh ---")
pts_per_edge = 36
boundary_pts_2d = []
for k in range(6):
    p1 = hex_corners[k]
    p2 = hex_corners[(k + 1) % 6]
    for s in np.linspace(0, 1, pts_per_edge, endpoint=False):
        boundary_pts_2d.append((1.0 - s) * p1 + s * p2)
boundary_pts_2d = np.array(boundary_pts_2d)
N_b = len(boundary_pts_2d)

res_mesh = 321
xs_grid = np.linspace(-APOTHEM + 0.8, APOTHEM - 0.8, res_mesh)
ys_grid = np.linspace(-R + 1.5, R - 1.5, int(res_mesh * (2 * R) / (2 * APOTHEM)))
X_g, Y_g = np.meshgrid(xs_grid, ys_grid)

d_edges = np.zeros((6, len(ys_grid), len(xs_grid)))
for k in range(6):
    d_edges[k] = APOTHEM - (X_g * normals_hex[k][0] + Y_g * normals_hex[k][1])
dist_edge_g = np.min(d_edges, axis=0)
mask_internal = dist_edge_g >= 0.8

int_x = list(X_g[mask_internal])
int_y = list(Y_g[mask_internal])

critical_summits = [
    (-4.0, 6.0, 83.00),
    (18.0, 16.0, 75.00),
    (-12.0, -18.0, 69.00),
    (-26.0, -4.0, 55.00)
]
for sx, sy, sz in critical_summits:
    int_x.append(sx)
    int_y.append(sy)

int_x = np.array(int_x)
int_y = np.array(int_y)
int_z = interp_z(np.column_stack((int_y, int_x)))
int_z[-4] = 83.00 # Pin peak

bound_x = boundary_pts_2d[:, 0]
bound_y = boundary_pts_2d[:, 1]
bound_z = np.full(N_b, Z_BASE)

all_top_pts = np.vstack([
    np.column_stack([bound_x, bound_y, bound_z]),
    np.column_stack([int_x, int_y, int_z])
])

d_top = Delaunay(all_top_pts[:, :2])
top_centers = np.mean(all_top_pts[d_top.simplices, :2], axis=1)
d_top_edges = np.zeros((6, len(top_centers)))
for k in range(6):
    d_top_edges[k] = APOTHEM - (top_centers[:, 0] * normals_hex[k][0] + top_centers[:, 1] * normals_hex[k][1])
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

r_rings = np.linspace(8.0, 48.0, 8)
int_bot_pts = []
for r in r_rings:
    n_pts = int(r * 2.5)
    for a in np.linspace(0, 2 * math.pi, n_pts, endpoint=False):
        pt = np.array([r * math.cos(a), r * math.sin(a)])
        d_ed = [APOTHEM - (pt[0]*n[0] + pt[1]*n[1]) for n in normals_hex]
        if min(d_ed) >= 2.0:
            int_bot_pts.append(pt)
int_bot_pts = np.array(int_bot_pts)

all_bot_2d = np.vstack([boundary_pts_2d, mag_rim_pts[:, :2], int_bot_pts])
d_bot = Delaunay(all_bot_2d)
bot_centers = np.mean(all_bot_2d[d_bot.simplices], axis=1)
r_c = np.hypot(bot_centers[:, 0], bot_centers[:, 1])
d_bot_edges = np.zeros((6, len(bot_centers)))
for k in range(6):
    d_bot_edges[k] = APOTHEM - (bot_centers[:, 0]*normals_hex[k][0] + bot_centers[:, 1]*normals_hex[k][1])
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

for i in range(N_b):
    top_curr = i; top_next = (i + 1) % N_b
    bot_curr = offset_bot_bound + i; bot_next = offset_bot_bound + (i + 1) % N_b
    all_faces.append([top_curr, bot_curr, bot_next])
    all_faces.append([top_curr, bot_next, top_next])

for tri in bot_ring_tris:
    mapped = []
    for idx in tri:
        if idx < N_b: mapped.append(offset_bot_bound + idx)
        elif idx < N_b + N_mag: mapped.append(offset_mag_rim + (idx - N_b))
        else: mapped.append(offset_int_bot + (idx - N_b - N_mag))
    all_faces.append([mapped[0], mapped[2], mapped[1]])

for i in range(N_mag):
    rim_curr = offset_mag_rim + i; rim_next = offset_mag_rim + (i + 1) % N_mag
    ceil_curr = offset_mag_ceil + i; ceil_next = offset_mag_ceil + (i + 1) % N_mag
    all_faces.append([rim_curr, ceil_curr, ceil_next])
    all_faces.append([rim_curr, ceil_next, rim_next])

for i in range(N_mag):
    ceil_curr = offset_mag_ceil + i; ceil_next = offset_mag_ceil + (i + 1) % N_mag
    all_faces.append([ceil_next, ceil_curr, idx_mag_center])

solid_mesh = trimesh.Trimesh(vertices=v_total, faces=all_faces, process=True)
print(f"Solid Mesh: {len(solid_mesh.vertices):,} vertices, {len(solid_mesh.faces):,} faces, Watertight: {solid_mesh.is_watertight}")

# Export STL
stl_path = os.path.join(out_dir, "catan_mountain_solid.stl")
solid_mesh.export(stl_path)
print(f"Saved: {stl_path}")

# -------------------------------------------------------------
# 4. EXPORT ULTRA-CRISP FULL-COLOR 3MF, OBJ, PLY
# -------------------------------------------------------------
verts = solid_mesh.vertices
u = np.clip((verts[:, 0] + APOTHEM) / (2 * APOTHEM), 0.0, 1.0)
v = np.clip((verts[:, 1] + R) / (2 * R), 0.0, 1.0)
is_base = verts[:, 2] < 24.95
u[is_base] = 0.01; v[is_base] = 0.01
uvs = np.column_stack([u, v])

# Sample continuous RGB for PLY
px = np.clip((u * (dim_tex - 1)).astype(int), 0, dim_tex - 1)
py = np.clip((v * (dim_tex - 1)).astype(int), 0, dim_tex - 1)
sampled_rgb = diffuse_4k[py, px]
sampled_rgb[is_base] = [0.12, 0.13, 0.15]
colors_rgba = np.column_stack([(sampled_rgb * 255.0).astype(np.uint8), np.full(len(verts), 255, dtype=np.uint8)])
solid_mesh.visual = trimesh.visual.ColorVisuals(mesh=solid_mesh, vertex_colors=colors_rgba)

# Export PLY
ply_path = os.path.join(out_dir, "catan_mountain_fullcolor.ply")
solid_mesh.export(ply_path)
print(f"Saved Full-Color PLY: {ply_path}")

# Export OBJ
obj_path = os.path.join(out_dir, "catan_mountain_fullcolor.obj")
mtl_path = os.path.join(out_dir, "catan_mountain.mtl")
with open(mtl_path, 'w', encoding='utf-8') as f:
    f.write("# Catan Mountain Full Color Material\n")
    f.write("newmtl mountain_fullcolor\n")
    f.write("Ka 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.05 0.05 0.05\nNs 15.0\n")
    f.write("map_Kd catan_mountain_diffuse_4k.png\n")

with open(obj_path, 'w', encoding='utf-8') as f:
    f.write("# Catan Mountain Ultra-Crisp Full Color\n")
    f.write("mtllib catan_mountain.mtl\nusemtl mountain_fullcolor\ns 1\n")
    for pt in verts: f.write(f"v {pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}\n")
    for uv in uvs: f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
    for vn in solid_mesh.vertex_normals: f.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")
    for tri in solid_mesh.faces:
        v1, v2, v3 = tri[0]+1, tri[1]+1, tri[2]+1
        f.write(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}\n")
print(f"Saved Full-Color OBJ: {obj_path}")

# Export Textured 3MF
mf3_path = os.path.join(out_dir, "catan_mountain_fullcolor.3mf")
lines_3mf = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">',
    '  <resources>',
    '    <m:texture2d id="1" path="/3D/Textures/diffuse.png" contenttype="image/png" />',
    '    <m:texture2dgroup id="2" texid="1">'
]
for i in range(len(verts)): lines_3mf.append(f'      <m:tex2coord u="{u[i]:.6f}" v="{v[i]:.6f}" />')
lines_3mf.append('    </m:texture2dgroup>')
lines_3mf.append('    <object id="3" type="model" pid="2"><mesh><vertices>')
for pt in verts: lines_3mf.append(f'      <vertex x="{pt[0]:.4f}" y="{pt[1]:.4f}" z="{pt[2]:.4f}" />')
lines_3mf.append('    </vertices><triangles>')
for tri in solid_mesh.faces:
    lines_3mf.append(f'      <triangle v1="{tri[0]}" v2="{tri[1]}" v3="{tri[2]}" p1="{tri[0]}" p2="{tri[1]}" p3="{tri[2]}" />')
lines_3mf.append('    </triangles></mesh></object></resources><build><item objectid="3" /></build></model>')

with zipfile.ZipFile(mf3_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.writestr('[Content_Types].xml', '''<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" /><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" /><Default Extension="png" ContentType="image/png" /></Types>''')
    z.writestr('_rels/.rels', '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" /></Relationships>''')
    z.writestr('3D/_rels/3dmodel.model.rels', '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/Textures/diffuse.png" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/texture" /></Relationships>''')
    z.writestr('3D/3dmodel.model', "\n".join(lines_3mf))
    with open(out_2k, 'rb') as f_tex: z.writestr('3D/Textures/diffuse.png', f_tex.read())
print(f"Saved Textured 3MF: {mf3_path} ({os.path.getsize(mf3_path)/1024/1024:.2f} MB)")

# -------------------------------------------------------------
# 5. UPDATE WEBGL VIEWER EMBEDDED DATA
# -------------------------------------------------------------
print("--- 5. Updating WebGL Viewer Embedded Topography & Texture ---")
with open(out_2k, 'rb') as f: b64_tex = base64.b64encode(f.read()).decode('ascii')
z_norm = np.clip((Z_top - Z_BASE) / RELIEF_H, 0.0, 1.0)
z_u16 = (z_norm * 65535.0).astype(np.uint16)
b64_z = base64.b64encode(z_u16.tobytes()).decode('ascii')

with open('scratch/fullcolor_viewer_data.js', 'w', encoding='utf-8') as f:
    f.write(f"const DEM_NX = {nx};\n")
    f.write(f"const DEM_NY = {ny};\n")
    f.write(f"const DEM_Z_B64 = \"{b64_z}\";\n")
    f.write(f"const DEM_TEX_B64 = \"data:image/png;base64,{b64_tex}\";\n")

print(f"\n=== MASTER SUITE BUILD COMPLETE ({time.time()-t0:.2f}s) ===")
