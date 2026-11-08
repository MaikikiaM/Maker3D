import numpy as np
import trimesh
from scipy.spatial import Delaunay
import os

def build_true_alpine_solid_mountain(res=201, output_stl="Tableros Catan/Loseta_Montana_3D/catan_mountain_solid.stl", output_3mf="Tableros Catan/Loseta_Montana_3D/catan_mountain_cmykwog.3mf"):
    os.makedirs(os.path.dirname(output_stl), exist_ok=True)
    
    R = 64.0
    apothem = R * np.cos(np.pi / 6) # ~55.4256 mm
    z_base = 25.0
    z_peak = 83.0
    max_relief = z_peak - z_base # 58.0 mm
    
    # 1. Hexagon boundary vertices (6 corners in CCW order)
    v_angles = (30.0 + 60.0 * np.arange(6)) * np.pi / 180.0
    hex_vx = R * np.cos(v_angles)
    hex_vy = R * np.sin(v_angles)
    
    # Perimeter sampling (32 pts/edge = 192 boundary pts)
    pts_per_edge = 32
    boundary_pts_2d = []
    for k in range(6):
        p1 = np.array([hex_vx[k], hex_vy[k]])
        p2 = np.array([hex_vx[(k+1)%6], hex_vy[(k+1)%6]])
        for s in np.linspace(0, 1, pts_per_edge, endpoint=False):
            boundary_pts_2d.append((1.0 - s) * p1 + s * p2)
    boundary_pts_2d = np.array(boundary_pts_2d) # N_b = 192
    N_b = len(boundary_pts_2d)
    
    # 2. Internal grid
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
    
    def smoothstep(e0, e1, v):
        t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
        
    def compute_alpine_relief(x_arr, y_arr, d_edge_arr):
        d_vert = np.full(x_arr.shape, 999.0)
        for k in range(6):
            d_vert = np.minimum(d_vert, np.hypot(x_arr - hex_vx[k], y_arr - hex_vy[k]))
            
        # Clearance: road shelf 3.5mm, vertex pads 10mm radius (20mm dia)
        clearance = smoothstep(3.5, 8.0, d_edge_arr) * smoothstep(10.0, 14.0, d_vert)
        
        # Major Alpine Ridge Spines (knife-edge ridges)
        spines = [
            (-4.0, 6.0, -28.0, 32.0, 58.0, 15.0, 10.0),   # NW ridge from main peak
            (-4.0, 6.0, 14.0, 14.0, 58.0, 47.0, 8.0),     # Ridge to East Peak
            (14.0, 14.0, 32.0, 22.0, 47.0, 12.0, 9.0),    # NE ridge
            (14.0, 14.0, 24.0, -8.0, 47.0, 25.0, 9.0),    # E ridge
            (-4.0, 6.0, -12.0, -18.0, 58.0, 38.0, 8.5),   # S spine to South Peak
            (-12.0, -18.0, -32.0, -26.0, 38.0, 10.0, 9.0),# SW ridge
            (-12.0, -18.0, 12.0, -26.0, 38.0, 14.0, 9.0), # SE ridge
        ]
        
        H_spines = np.zeros_like(x_arr)
        for x1, y1, x2, y2, h1, h2, sharp in spines:
            dx, dy = x2 - x1, y2 - y1
            L2 = dx*dx + dy*dy
            t = np.clip(((x_arr - x1)*dx + (y_arr - y1)*dy) / L2, 0.0, 1.0)
            px = x1 + t * dx
            py = y1 + t * dy
            dist_spine = np.hypot(x_arr - px, y_arr - py)
            h_line = (1.0 - t) * h1 + t * h2
            h_contrib = h_line * np.exp(-dist_spine / sharp)
            H_spines = np.maximum(H_spines, h_contrib)
            
        # Main Pyramidal Horn (Matterhorn)
        r_main = np.hypot(x_arr - (-4.0), y_arr - 6.0)
        ang_main = np.arctan2(y_arr - 6.0, x_arr - (-4.0))
        facet = np.abs(np.cos(1.5 * ang_main + 0.4))
        h_horn = 58.0 * np.exp(-r_main / 18.0) * (0.85 + 0.35 * facet**1.5)
        
        # Secondary Peak
        r_p2 = np.hypot(x_arr - 14.0, y_arr - 14.0)
        h_p2 = 47.0 * np.exp(-r_p2 / 14.0)
        
        # South Summit
        r_p3 = np.hypot(x_arr - (-12.0), y_arr - (-18.0))
        h_p3 = 38.0 * np.exp(-r_p3 / 15.0)
        
        terrain = np.maximum(H_spines, np.maximum(h_horn, np.maximum(h_p2, h_p3)))
        
        # Multifractal rock noise (non-aligned irrational angles to avoid periodic checkerboards)
        noise = np.zeros_like(x_arr)
        frequencies = [0.08, 0.16, 0.32, 0.65, 1.30, 2.60]
        amplitudes = [7.5, 4.2, 2.3, 1.2, 0.6, 0.25]
        angles_rot = [0.31, 1.15, 2.47, 0.78, 1.89, 2.92]
        
        for f, a, ang in zip(frequencies, amplitudes, angles_rot):
            Xr = x_arr * np.cos(ang) - y_arr * np.sin(ang)
            Yr = x_arr * np.sin(ang) + y_arr * np.cos(ang)
            n1 = 1.0 - np.abs(np.sin(Xr * f + 1.2) * np.cos(Yr * f * 1.1 + 0.7))
            n1 = n1 * n1 * n1
            noise += a * (n1 - 0.5)
            
        terrain = terrain + noise * (terrain / 58.0)**0.5
        terrain = np.maximum(0.0, terrain)
        
        # Normalize peak strictly to 58.0 mm
        p_max = np.max(terrain)
        if p_max > 0:
            terrain = terrain * (58.0 / p_max)
            
        sculpted = terrain * clearance
        return z_base + sculpted
        
    int_x = X[mask_internal]
    int_y = Y[mask_internal]
    int_z = compute_alpine_relief(int_x, int_y, dist_edge[mask_internal])
    
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
    
    # 3. Bottom Face Construction with Magnet Pocket
    N_mag = 36
    r_mag = 5.1 # dia 10.2mm
    mag_angles = np.linspace(0, 2*np.pi, N_mag, endpoint=False)
    mag_rim_pts = np.column_stack([r_mag * np.cos(mag_angles), r_mag * np.sin(mag_angles), np.zeros(N_mag)])
    mag_ceil_pts = np.column_stack([r_mag * np.cos(mag_angles), r_mag * np.sin(mag_angles), np.full(N_mag, 5.2)])
    mag_ceil_center = np.array([[0.0, 0.0, 5.2]])
    bottom_bound_pts = np.column_stack([bound_x, bound_y, np.zeros(N_b)])
    
    # Ring internal sampling for clean Delaunay on bottom face
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
    
    print("\n=== SOLID MONOLITHIC ALPINE MOUNTAIN BUILT ===")
    print(f"  Vertices: {len(solid_mesh.vertices):,}")
    print(f"  Triangles: {len(solid_mesh.faces):,}")
    print(f"  Watertight Solid: {solid_mesh.is_watertight}")
    print(f"  Volume: {solid_mesh.volume / 1000.0:.2f} cm3")
    print(f"  Z-Bounds: {solid_mesh.bounds[0][2]:.2f} mm to {solid_mesh.bounds[1][2]:.2f} mm")
    
    solid_mesh.export(output_stl)
    print(f"  Saved master solid STL: {output_stl}")
    
    # 4. Generate Multi-Body CMYKWOG 3MF Assembly
    generate_cmykwog_3mf(solid_mesh, output_3mf)
    
    return solid_mesh

def generate_cmykwog_3mf(mesh, output_3mf):
    verts = mesh.vertices
    faces = mesh.faces
    
    face_centers = np.mean(verts[faces], axis=1)
    fx = face_centers[:, 0]
    fy = face_centers[:, 1]
    fz = face_centers[:, 2]
    
    # Slope calculation for natural rock/snow/ore assignment
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fnorm = np.cross(v1 - v0, v2 - v0)
    fnorm = fnorm / (np.linalg.norm(fnorm, axis=1, keepdims=True) + 1e-8)
    # Cosine of slope angle relative to vertical (Z):
    # fnorm[:, 2] == 1 means flat horizontal surface
    # fnorm[:, 2] == 0 means vertical cliff
    horizontalness = np.abs(fnorm[:, 2])
    
    # Geological Ore Fault line: diagonal curve across mountain
    # distance to ore fault curve
    d_ore = np.abs(fy + 0.6 * fx + 5.0)
    
    face_labels = np.zeros(len(faces), dtype=int)
    
    for i in range(len(faces)):
        z = fz[i]
        x = fx[i]
        y = fy[i]
        h_factor = horizontalness[i]
        ore_dist = d_ore[i]
        
        # 0: Base Zocalo (Z <= 24.9 mm)
        if z < 24.9:
            face_labels[i] = 0 # Tool B (Base)
        # 3: Glacial Snow (High elevation + snow settling on flatter slopes and high couloirs)
        elif z >= 64.0 and h_factor > 0.4:
            face_labels[i] = 3 # Tool W (Snow - 0.2mm nozzle)
        elif z >= 72.0:
            face_labels[i] = 3 # Extreme summit snow
        # 2: Iron Ore Veins (following natural geological fault line on rocky faces)
        elif ore_dist < 4.5 and 30.0 <= z <= 58.0 and h_factor < 0.75:
            face_labels[i] = 2 # Tool O / Y (Iron Ore / Copper)
        # 4: Alpine Moss (Low elevation flatter terrain near boundary)
        elif 25.0 <= z <= 28.0 and h_factor > 0.65:
            face_labels[i] = 4 # Tool G (Alpine Moss)
        # 1: Granite Rock (Main exposed crags, cliffs, arêtes)
        else:
            face_labels[i] = 1 # Tool T3/K (Granite Rock)
            
    body_names = [
        "Body_0_Base_Zocalo_Tool_B",
        "Body_1_Granite_Rock_Tool_T3_K",
        "Body_2_Iron_Ore_Veins_Tool_O_Y",
        "Body_3_Alpine_Snow_Tool_W",
        "Body_4_Alpine_Moss_Tool_G"
    ]
    
    body_colors = [
        [35, 38, 42, 255],     # Dark charcoal / Base PLA
        [125, 128, 132, 255],  # Granite Grey
        [215, 95, 35, 255],    # Iron Ore Orange / Rust
        [250, 252, 255, 255],  # Alpine Snow White
        [60, 125, 50, 255],    # Alpine Moss Green
    ]
    
    scene = trimesh.Scene()
    parts_dir = "Tableros Catan/Loseta_Montana_3D/piezas_cmykwog_separadas"
    os.makedirs(parts_dir, exist_ok=True)
    
    print("\nCMYKWOG Geological Body Segmentation:")
    for b_id, name in enumerate(body_names):
        sub_faces = faces[face_labels == b_id]
        if len(sub_faces) == 0:
            continue
        u_v_idx, inv = np.unique(sub_faces, return_inverse=True)
        sub_verts = verts[u_v_idx]
        sub_f = inv.reshape(sub_faces.shape)
        
        sub_mesh = trimesh.Trimesh(vertices=sub_verts, faces=sub_f, process=False)
        sub_mesh.visual.face_colors = np.tile(body_colors[b_id], (len(sub_f), 1))
        
        stl_path = os.path.join(parts_dir, f"{name}.stl")
        sub_mesh.export(stl_path)
        scene.add_geometry(sub_mesh, node_name=name)
        print(f"  {name}: {len(sub_f):,} faces -> {stl_path}")
        
    scene.export(output_3mf)
    print(f"  Saved full-color 3MF assembly: {output_3mf}")

if __name__ == '__main__':
    build_true_alpine_solid_mountain(res=201)
