import math
import os
import base64
import numpy as np
import scipy.ndimage as ndi
import scipy.interpolate as interp
from scipy.spatial import Delaunay
import trimesh
import matplotlib.pyplot as plt

print("==========================================================")
print("  MASTER CMYKWOG-B ALPINE CATAN MOUNTAIN EXPORT PIPELINE  ")
print("==========================================================")

# Target directories
out_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D"
parts_dir = os.path.join(out_dir, "piezas_cmykwog_separadas")
renders_dir = os.path.join(out_dir, "renders_cad")
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

os.makedirs(out_dir, exist_ok=True)
os.makedirs(parts_dir, exist_ok=True)
os.makedirs(renders_dir, exist_ok=True)

# 1. Hexagon Geometry Constants
R = 64.0 # Vertex radius
APOTHEM = R * math.cos(math.radians(30)) # 55.4256258 mm
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF_H = Z_PEAK - Z_BASE # 58.0 mm

# Load Master Topography and Masks
Z_master = np.load('scratch/Z_top_master.npy')
ny_m, nx_m = Z_master.shape
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m)
ys_m = np.linspace(-R, R, ny_m)
X_m, Y_m = np.meshgrid(xs_m, ys_m)

dx = xs_m[1] - xs_m[0]
dy = ys_m[1] - ys_m[0]
gy, gx = np.gradient(Z_master, dy, dx)
slope = np.hypot(gx, gy)
slope_deg = np.degrees(np.arctan(slope))
laplace = ndi.laplace(Z_master)

normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X_m, 999.0)
for nx_h, ny_h in normals:
    d = APOTHEM - (X_m * nx_h + Y_m * ny_h)
    dist_edge = np.minimum(dist_edge, d)

v_angles = [(30 + 60 * k) * math.pi / 180 for k in range(6)]
hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles])
dist_vert = np.full_like(X_m, 999.0)
for vx, vy in hex_corners:
    dv = np.hypot(X_m - vx, Y_m - vy)
    dist_vert = np.minimum(dist_vert, dv)

relief = Z_master - Z_BASE

# 2. Refined Geological Tool Layers (CMYKWOG-B)
np.random.seed(314)
n1 = ndi.gaussian_filter(np.random.randn(ny_m, nx_m), sigma=6.0)
n2 = ndi.gaussian_filter(np.random.randn(ny_m, nx_m), sigma=2.5) * 0.5
n3 = ndi.gaussian_filter(np.random.randn(ny_m, nx_m), sigma=1.0) * 0.25
fbm = (n1 + n2 + n3)

# Veins: Branching fractures across granite body
fault_field = np.sin((X_m * 0.08 + Y_m * 0.05 + fbm * 1.5) * 3.14)
ore_intensity = np.exp(-np.abs(fault_field) / 0.18) * np.clip((relief - 4.0) / 12.0, 0, 1) * np.clip((72.0 - Z_master) / 12.0, 0, 1)
ore_intensity[dist_edge < 4.0] = 0.0
ore_mask = ore_intensity > 0.55

# Snow: high elevation, low slope, couloir retention
snow_score = (relief / 58.0)**1.5 * 1.8 - (slope_deg / 45.0)**1.2 * 0.9 + np.clip(-laplace * 0.4, -0.3, 0.5)
snow_factor = np.clip(snow_score, 0.0, 1.0)
snow_mask = (snow_factor > 0.45) & (relief > 25.0) & (~ore_mask)

# Glacial Ice: shaded couloirs
ice_score = np.clip(-laplace * 0.6, 0.0, 1.0) * np.clip((relief - 28.0) / 20.0, 0, 1) * (1.0 - snow_factor)
ice_mask = (ice_score > 0.32) & (relief > 26.0) & (~snow_mask) & (~ore_mask)

# Alpine Tundra: lower depressions
tundra_score = np.clip((6.0 - relief) / 5.0, 0, 1) * np.clip((35.0 - slope_deg) / 20.0, 0, 1) * (dist_edge > 3.0)
tundra_mask = (tundra_score > 0.4) & (relief > 0.2) & (~ore_mask)

# Dark Slate Cliffs: steep shaded faces
cliff_mask = (slope_deg > 46.0) & (relief > 3.0) & (~snow_mask) & (~ore_mask) & (~ice_mask)

# Tool Layer IDs:
# 0 = Base / Road shelf (Tool B)
# 1 = Glacial Ice (Tool C)
# 3 = Dark Slate Cliffs (Tool K)
# 4 = Granite Rock Face (Tool W / Mid Grey)
# 5 = Alpine Tundra (Tool G / Green)
# 6 = Iron Ore Veins (Tool O / Orange)
# 7 = Alpine Snow (Tool W / White 0.2mm)
tool_map = np.full((ny_m, nx_m), 4, dtype=np.uint8) # default granite
tool_map[cliff_mask] = 3
tool_map[tundra_mask] = 5
tool_map[ice_mask] = 1
tool_map[snow_mask] = 7
tool_map[ore_mask] = 6
# Road shelf and vertex pads
shelf_mask = (relief < 0.05) & (dist_edge >= 0)
tool_map[shelf_mask] = 0
tool_map[dist_edge < 0] = 0

print("Layer distribution:")
for tid, name in [(0, 'Base / Shelf (Tool B)'), (1, 'Glacial Ice (Tool C)'), (3, 'Slate Cliffs (Tool K)'),
                  (4, 'Granite Rock (Tool W/Grey)'), (5, 'Tundra (Tool G)'), (6, 'Iron Ore (Tool O)'), (7, 'Snow (Tool W-0.2)')]:
    cnt = np.sum(tool_map == tid)
    pct = 100.0 * cnt / np.sum(dist_edge >= 0)
    print(f"  Tool {tid} ({name:25s}): {cnt:6d} px ({pct:5.1f}%)")

# 3. Build 100% Watertight Solid STL Mesh
print("\n--- Constructing Watertight Solid Monolithic Mesh ---")
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

pts_per_edge = 32
boundary_pts_2d = []
for k in range(6):
    p1 = hex_corners[k]
    p2 = hex_corners[(k + 1) % 6]
    for s in np.linspace(0, 1, pts_per_edge, endpoint=False):
        boundary_pts_2d.append((1.0 - s) * p1 + s * p2)
boundary_pts_2d = np.array(boundary_pts_2d)
N_b = len(boundary_pts_2d)

# Internal grid resolution: 201 x 233
res = 201
xs_grid = np.linspace(-APOTHEM + 1.0, APOTHEM - 1.0, res)
ys_grid = np.linspace(-R + 1.8, R - 1.8, int(res * (2 * R) / (2 * APOTHEM)))
X_g, Y_g = np.meshgrid(xs_grid, ys_grid)

d_edges = np.zeros((6, len(ys_grid), len(xs_grid)))
for k in range(6):
    d_edges[k] = APOTHEM - (X_g * normals[k][0] + Y_g * normals[k][1])
dist_edge_g = np.min(d_edges, axis=0)
mask_internal = dist_edge_g >= 1.0

int_x = X_g[mask_internal]
int_y = Y_g[mask_internal]
int_z = interp_z(np.column_stack((int_y, int_x)))

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
    d_top_edges[k] = APOTHEM - (top_centers[:, 0] * normals[k][0] + top_centers[:, 1] * normals[k][1])
mask_top = np.min(d_top_edges, axis=0) >= -0.05
top_faces = d_top.simplices[mask_top]

# Bottom with Magnet Socket (diam 10.2mm, depth 5.2mm)
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

# Vertical side walls (Z=25 down to Z=0)
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

# Magnet ceiling
for i in range(N_mag):
    ceil_curr = offset_mag_ceil + i
    ceil_next = offset_mag_ceil + (i + 1) % N_mag
    all_faces.append([ceil_next, ceil_curr, idx_mag_center])

solid_mesh = trimesh.Trimesh(vertices=v_total, faces=all_faces, process=True)

print("Solid Mesh Verification:")
print(f"  Watertight: {solid_mesh.is_watertight}")
print(f"  Volume: {solid_mesh.volume / 1000.0:.2f} cm3")
print(f"  Vertices: {len(solid_mesh.vertices):,}")
print(f"  Faces: {len(solid_mesh.faces):,}")
print(f"  Bounds: X [{solid_mesh.bounds[0][0]:.2f}, {solid_mesh.bounds[1][0]:.2f}], Y [{solid_mesh.bounds[0][1]:.2f}, {solid_mesh.bounds[1][1]:.2f}], Z [{solid_mesh.bounds[0][2]:.2f}, {solid_mesh.bounds[1][2]:.2f}]")

# Export Master Solid STL
stl_path = os.path.join(out_dir, "catan_mountain_solid.stl")
solid_mesh.export(stl_path)
print(f"Saved: {stl_path}")

# 4. Multi-Material Partitioning & 3MF Generation
print("\n--- Generating Multi-Material CMYKWOG 3MF & Tool STLs ---")
interp_tool = interp.RegularGridInterpolator((ys_m, xs_m), tool_map, method='nearest', bounds_error=False, fill_value=0)

# Evaluate tool ID for each face of the solid mesh
face_centers = np.mean(solid_mesh.vertices[solid_mesh.faces], axis=1)
fc_x = face_centers[:, 0]
fc_y = face_centers[:, 1]
fc_z = face_centers[:, 2]

face_tools = np.zeros(len(solid_mesh.faces), dtype=np.uint8)
# Faces with Z <= 25.01 or on bottom/side walls are strictly Tool 0 (Base)
is_base = (fc_z <= 25.05)
face_tools[is_base] = 0

# For faces on top surface (Z > 25.05), sample tool_map
top_mask_faces = ~is_base
face_tools[top_mask_faces] = interp_tool(np.column_stack((fc_y[top_mask_faces], fc_x[top_mask_faces])))

tool_names = {
    0: ("T0_Base_Zocalo_Negro", "Tool B (Negro)", [30, 32, 35]),
    1: ("T1_Hielo_Glaciar_Cian", "Tool C (Cian)", [104, 178, 191]),
    3: ("T3_Acantilados_Pizarra_Negro", "Tool K (Negro)", [45, 52, 58]),
    4: ("T4_Roca_Granito_Gris", "Tool W/Gris (Granito)", [125, 133, 140]),
    5: ("T5_Musgo_Tundra_Verde", "Tool G (Verde)", [67, 118, 56]),
    6: ("T6_Vetas_Hierro_Naranja", "Tool O (Naranja/Oro)", [209, 97, 26]),
    7: ("T7_Nieve_Alpina_Blanco_02mm", "Tool W (Blanco 0.2mm)", [238, 242, 246])
}

submeshes = {}
for tid, (fname, tlabel, color_rgb) in tool_names.items():
    sub_faces = np.where(face_tools == tid)[0]
    if len(sub_faces) > 0:
        sub_m = solid_mesh.submesh([sub_faces], append=True)
        sub_path = os.path.join(parts_dir, f"{fname}.stl")
        sub_m.export(sub_path)
        submeshes[fname] = sub_m
        print(f"  Exported {fname}.stl: {len(sub_m.faces):6d} faces ({tlabel})")

# Export Multi-Material 3MF
mf3_path = os.path.join(out_dir, "catan_mountain_cmykwog.3mf")
scene_3mf = trimesh.Scene()
for fname, sm in submeshes.items():
    scene_3mf.add_geometry(sm, node_name=fname)
scene_3mf.export(mf3_path)
print(f"Saved Multi-Body 3MF: {mf3_path}")

# 5. High-Resolution CAD Renders
print("\n--- Generating High-Resolution Presentation Renders ---")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
# Subsample for smooth 3D surface plot
step = 2
surf_x = X_m[::step, ::step]
surf_y = Y_m[::step, ::step]
surf_z = Z_master[::step, ::step]
tools_sub = tool_map[::step, ::step]

# Color array for surface
colors_arr = np.zeros((surf_x.shape[0], surf_x.shape[1], 4))
for tid, (_, _, c) in tool_names.items():
    m = (tools_sub == tid)
    colors_arr[m] = [c[0]/255.0, c[1]/255.0, c[2]/255.0, 1.0]

ax.plot_surface(surf_x, surf_y, surf_z, facecolors=colors_arr, rstride=1, cstride=1, shade=True, antialiased=True)
ax.view_init(elev=35, azim=-55)
ax.set_box_aspect([1.0, 1.15, 0.75])
ax.set_title("Master Catan Alpine Massif - Isometric CAD Render (CMYKWOG-B)", fontsize=13, fontweight='bold')
ax.axis('off')
iso_render_path = os.path.join(renders_dir, "render_mountain_isometric.png")
plt.tight_layout()
plt.savefig(iso_render_path, dpi=200, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print(f"Saved: {iso_render_path}")

# Orthographic Top View
fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0f172a')
az = 315.0 * np.pi / 180.0; alt = 48.0 * np.pi / 180.0
sun_x = np.cos(alt)*np.sin(az); sun_y = np.cos(alt)*np.cos(az); sun_z = np.sin(alt)
norm = np.sqrt(gx**2 + gy**2 + 1.0)
hs = np.clip((-gx/norm)*sun_x + (-gy/norm)*sun_y + (1.0/norm)*sun_z, 0.0, 1.0)

rgb = np.zeros((ny_m, nx_m, 3), dtype=np.float32)
for tid, (_, _, c) in tool_names.items():
    m = (tool_map == tid)
    rgb[m] = [c[0]/255.0, c[1]/255.0, c[2]/255.0]

lit = rgb * (0.35 + 0.65 * hs[:, :, None])
lit[dist_edge < 0] = [0.06, 0.09, 0.16]
ax.imshow(lit, origin='lower')
ax.set_title("Top Orthographic View - Full Color Geological Map", color='white', fontsize=12)
ax.axis('off')
top_render_path = os.path.join(renders_dir, "render_mountain_top_ortho.png")
plt.tight_layout()
plt.savefig(top_render_path, dpi=200, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print(f"Saved: {top_render_path}")

# Solid Cross Section Proof Render
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0f172a')
y_mid_idx = ny_m // 2
x_slice = xs_m
z_slice = Z_master[y_mid_idx, :]
ax.fill_between(x_slice, 0, z_slice, color='#334155', alpha=0.9, label='Cuerpo 100% Macizo (Monolítico)')
ax.fill_between(x_slice, 0, 25.0, color='#1e293b', alpha=0.9, label='Zócalo Base (Tool B - 25mm)')
ax.plot(x_slice, z_slice, color='#f8fafc', lw=2.0, label='Perfil Alpino DEM')
ax.axhline(25.0, color='#38bdf8', linestyle='--', lw=1.2, label='Plano Asentamiento Catan (Z=25mm)')
# Magnet pocket cross section
ax.fill_between([-5.1, 5.1], 0, 5.2, color='#020617', hatch='//', label='Alojamiento Imán Ø10.2 x 5.2mm')
ax.set_ylim(-2, 90)
ax.set_xlim(-60, 60)
ax.set_ylabel("Altura Z (mm)", color='white')
ax.set_xlabel("Distancia Radial X (mm)", color='white')
ax.tick_params(colors='white')
ax.set_title("Corte Transversal Z - 100% Sólido sin Huecos Internos", color='white', fontsize=12, fontweight='bold')
ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='white', loc='upper right')
ax.grid(True, color='#334155', linestyle=':', alpha=0.6)
cross_render_path = os.path.join(renders_dir, "render_solid_cross_section.png")
plt.tight_layout()
plt.savefig(cross_render_path, dpi=180, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print(f"Saved: {cross_render_path}")

# Copy renders to brain artifact folder
import shutil
for f in [iso_render_path, top_render_path, cross_render_path]:
    shutil.copy(f, os.path.join(brain_dir, os.path.basename(f)))
print("Copied renders to brain artifact folder")

# 6. Base64 Arrays for Three.js WebGL Viewer
z_norm = np.clip((Z_master - Z_BASE) / RELIEF_H, 0.0, 1.0)
z_u16 = (z_norm * 65535.0).astype(np.uint16)
b64_z = base64.b64encode(z_u16.tobytes()).decode('ascii')
b64_tools = base64.b64encode(tool_map.tobytes()).decode('ascii')

with open('scratch/mountain_b64_data.js', 'w', encoding='utf-8') as f:
    f.write(f"// Master LiDAR DEM and Tool Data\n")
    f.write(f"const DEM_NX = {nx_m};\n")
    f.write(f"const DEM_NY = {ny_m};\n")
    f.write(f"const DEM_Z_B64 = \"{b64_z}\";\n")
    f.write(f"const DEM_TOOLS_B64 = \"{b64_tools}\";\n")
print(f"Generated scratch/mountain_b64_data.js ({len(b64_z)/1024 + len(b64_tools)/1024:.1f} KB)")
print("\n=== PIPELINE EXECUTION COMPLETE ===")
