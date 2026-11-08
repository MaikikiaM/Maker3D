import numpy as np
import trimesh
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def build_production_mountain(res=181, is_coastal_port=False, magnet_mode="open", output_dir="scratch/production_models"):
    """
    Builds a high-precision production Catan Mountain tile.
    magnet_mode: "open" (flush bottom pocket) or "encapsulated" (internal sealed chamber with 0.8mm bridge)
    is_coastal_port: True adds the 2:1 Ore Port wooden pier along the NE edge (Edge 0)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    R = 64.0  # Outer radius (mm)
    apothem = R * np.cos(np.pi / 6)  # ~55.4256 mm
    z_base = 25.0  # Base zocalo height
    z_peak = 83.0  # Peak height
    target_relief = z_peak - z_base  # 58.0 mm
    
    # 1. Hexagon polygon boundary vertices (6 vertices in CCW order)
    # Pointy-topped hexagon: angles 30, 90, 150, 210, 270, 330 deg
    v_angles = (30.0 + 60.0 * np.arange(6)) * np.pi / 180.0
    hex_vx = R * np.cos(v_angles)
    hex_vy = R * np.sin(v_angles)
    
    # Edges sampling (32 points per edge = 192 boundary points for high fidelity)
    pts_per_edge = 32
    boundary_pts_2d = []
    for k in range(6):
        p1 = np.array([hex_vx[k], hex_vy[k]])
        p2 = np.array([hex_vx[(k+1)%6], hex_vy[(k+1)%6]])
        for s in np.linspace(0, 1, pts_per_edge, endpoint=False):
            boundary_pts_2d.append((1.0 - s) * p1 + s * p2)
    boundary_pts_2d = np.array(boundary_pts_2d) # N_b = 192
    N_b = len(boundary_pts_2d)
    
    # 2. Internal grid for top surface
    xs = np.linspace(-apothem + 1.0, apothem - 1.0, res)
    ys = np.linspace(-R + 1.8, R - 1.8, int(res * (2 * R) / (2 * apothem)))
    X, Y = np.meshgrid(xs, ys)
    
    # 6 edges normals
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    d_edges = np.zeros((6, len(ys), len(xs)))
    for k in range(6):
        d_edges[k] = apothem - (X * normals[k, 0] + Y * normals[k, 1])
    dist_edge = np.min(d_edges, axis=0)
    mask_internal = dist_edge >= 1.0
    
    def compute_terrain_elevation(x_arr, y_arr, d_edge_arr=None):
        if d_edge_arr is None:
            d_ed = np.zeros((6, len(x_arr)))
            for k in range(6):
                d_ed[k] = apothem - (x_arr * normals[k, 0] + y_arr * normals[k, 1])
            d_edge_arr = np.min(d_ed, axis=0)
            
        d_vert = np.full(x_arr.shape, 999.0)
        for k in range(6):
            d_vert = np.minimum(d_vert, np.hypot(x_arr - hex_vx[k], y_arr - hex_vy[k]))
            
        # Clearance Factor: 3.5mm shelf, 10mm vertex radius (dia 20mm)
        edge_factor = smoothstep(3.5, 7.5, d_edge_arr)
        vert_factor = smoothstep(10.0, 14.0, d_vert)
        clearance = edge_factor * vert_factor
        
        # Primary Peak Alpha (-8.0, 8.0) - Matterhorn Horn
        dx1 = x_arr - (-8.0)
        dy1 = y_arr - 8.0
        r1 = np.hypot(dx1, dy1)
        ang1 = np.arctan2(dy1, dx1)
        ar1 = np.maximum(0, np.cos(ang1 - 2.44))**2.5
        ar2 = np.maximum(0, np.cos(ang1 - (-1.40)))**2.5
        ar3 = np.maximum(0, np.cos(ang1 - 0.35))**2.5
        ridges1 = (ar1 + ar2 + ar3) * 0.48
        h_alpha = 58.0 * np.exp(-r1 / 20.0) * (1.0 + ridges1)
        
        # Secondary Peak Beta (12.0, 20.0)
        dx2 = x_arr - 12.0
        dy2 = y_arr - 20.0
        r2 = np.hypot(dx2, dy2)
        ang2 = np.arctan2(dy2, dx2)
        ar_beta = np.maximum(0, np.cos(ang2 - np.pi))**2.0 * 0.38
        h_beta = 39.0 * np.exp(-r2 / 16.0) * (1.0 + ar_beta)
        
        # Connecting Saddle / Col
        pA = np.array([-8.0, 8.0])
        pB = np.array([12.0, 20.0])
        AB = pB - pA
        len_AB2 = np.dot(AB, AB)
        t_proj = np.clip(((x_arr - pA[0]) * AB[0] + (y_arr - pA[1]) * AB[1]) / len_AB2, 0.0, 1.0)
        proj_X = pA[0] + t_proj * AB[0]
        proj_Y = pA[1] + t_proj * AB[1]
        d_ridge = np.hypot(x_arr - proj_X, y_arr - proj_Y)
        h_saddle = 32.0 * np.exp(-d_ridge / 6.0) * np.sin(t_proj * np.pi)**0.5
        
        # Glacial Cirque on North Slope (sculpted depression between Alpha and Beta)
        dx_c = x_arr - 1.0
        dy_c = y_arr - 22.0
        r_cirque = np.hypot(dx_c, dy_c)
        cirque_depression = -6.5 * np.exp(-r_cirque / 8.0) * smoothstep(10.0, 2.0, r_cirque)
        
        # Iron Ore Quarry & Mine Terraces (South-West flank)
        dx_q = x_arr - (-12.0)
        dy_q = y_arr - (-16.0)
        r_q = np.hypot(dx_q, dy_q)
        quarry_benches = 4.2 * np.sin(r_q * 0.75) * np.exp(-r_q / 22.0)
        
        # High-frequency geological rock noise (Granite crags & micro-clivage)
        f1, f2, f3, f4 = 0.22, 0.48, 0.95, 1.85
        rock_noise = (
            2.5 * np.sin(x_arr * f1 + y_arr * 0.15) * np.cos(y_arr * f1 - x_arr * 0.1) +
            1.4 * np.sin(x_arr * f2 * 1.3 - y_arr * f2 * 0.8 + 1.2) +
            0.7 * np.cos(x_arr * f3 * 0.7 + y_arr * f3 * 1.4 + 2.5) +
            0.35 * np.sin(x_arr * f4 + y_arr * f4 * 1.2)
        )
        
        raw_relief = np.maximum(h_alpha, np.maximum(h_beta, h_saddle)) + cirque_depression + quarry_benches + rock_noise
        raw_relief = np.maximum(0.0, raw_relief)
        
        # Normalize to target relief of 58.0 mm
        m_val = np.max(raw_relief)
        if m_val > 0:
            raw_relief = raw_relief * (target_relief / m_val)
            
        sculpted = raw_relief * clearance
        
        # Chit Pocket & Terrace at (16.0, -14.0)
        cx_chit = 16.0
        cy_chit = -14.0
        r_chit_dist = np.hypot(x_arr - cx_chit, y_arr - cy_chit)
        z_plateau_relief = 12.0
        plateau_factor = smoothstep(18.0, 14.5, r_chit_dist)
        sculpted = sculpted * (1.0 - plateau_factor) + z_plateau_relief * plateau_factor
        
        pocket_wall = smoothstep(13.3, 13.0, r_chit_dist)
        sculpted = sculpted * (1.0 - pocket_wall) + (z_plateau_relief - 1.8) * pocket_wall
        
        notch_dist = np.hypot(x_arr - (cx_chit + 13.0), y_arr - cy_chit)
        notch_factor = smoothstep(3.0, 1.8, notch_dist)
        sculpted = sculpted * (1.0 - notch_factor) + (z_plateau_relief - 2.5) * notch_factor
        
        # If Coastal Port Variant: add wooden pier on Edge 0 (Northeast face)
        if is_coastal_port:
            # Edge 0 center: midpoint between (R*cos30, R*sin30) and (R*cos90, R*sin90)
            p_edge0 = 0.5 * (hex_vx[0] + hex_vx[1]), 0.5 * (hex_vy[0] + hex_vy[1])
            d_pier = np.hypot(x_arr - p_edge0[0], y_arr - p_edge0[1])
            pier_mask = smoothstep(10.0, 4.0, d_pier)
            # Pier deck height at Z = 25.0 + 3.0 = 28.0 mm
            sculpted = np.maximum(sculpted, 3.0 * pier_mask)
            
        return z_base + sculpted

    int_x = X[mask_internal]
    int_y = Y[mask_internal]
    int_z = compute_terrain_elevation(int_x, int_y, dist_edge[mask_internal])
    
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
    
    # 3. Bottom Face Construction & Magnet Cavity
    N_mag = 36
    r_mag = 5.1 # radius 5.1mm -> dia 10.2mm
    mag_angles = np.linspace(0, 2*np.pi, N_mag, endpoint=False)
    
    if magnet_mode == "open":
        z_mag_floor = 0.0
        z_mag_ceil = 5.2
    else: # encapsulated
        z_mag_floor = 0.8  # 0.8mm solid bridge on the bottom
        z_mag_ceil = 6.0   # cavity from 0.8mm to 6.0mm (5.2mm height)
        
    mag_rim_pts = np.column_stack([r_mag * np.cos(mag_angles), r_mag * np.sin(mag_angles), np.full(N_mag, z_mag_floor)])
    mag_ceil_pts = np.column_stack([r_mag * np.cos(mag_angles), r_mag * np.sin(mag_angles), np.full(N_mag, z_mag_ceil)])
    mag_ceil_center = np.array([[0.0, 0.0, z_mag_ceil]])
    
    bottom_bound_pts = np.column_stack([bound_x, bound_y, np.zeros(N_b)])
    
    # Internal points for bottom ring Delaunay
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
    
    # Master vertex layout
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
    
    # Extra bottom floor cap if encapsulated
    if magnet_mode == "encapsulated":
        mag_floor_center = np.array([[0.0, 0.0, 0.0]])
        idx_floor_center = len(v_total)
        v_total = np.vstack([v_total, mag_floor_center])
        
    all_faces = list(top_faces)
    
    # Side walls (Z=25 down to Z=0)
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
        
    # If encapsulated: bottom floor disc cap
    if magnet_mode == "encapsulated":
        for i in range(N_mag):
            rim_curr = offset_mag_rim + i
            rim_next = offset_mag_rim + (i + 1) % N_mag
            all_faces.append([rim_curr, rim_next, idx_floor_center])
            
    mesh = trimesh.Trimesh(vertices=v_total, faces=all_faces, process=True)
    
    tag = ("coastal_port" if is_coastal_port else "inland") + f"_mag_{magnet_mode}"
    stl_path = os.path.join(output_dir, f"catan_mountain_{tag}.stl")
    mesh.export(stl_path)
    
    print(f"\n[MODEL READY] {tag.upper()}:")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Triangles: {len(mesh.faces):,}")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Volume: {mesh.volume / 1000.0:.2f} cm3")
    print(f"  Z-Bounds: {mesh.bounds[0][2]:.2f} mm to {mesh.bounds[1][2]:.2f} mm")
    print(f"  File saved: {stl_path}")
    
    return mesh, tag

def generate_full_suite():
    print("===============================================================")
    print("  GENERATING PRODUCTION MOUNTAIN 3D SUITE (CATAN CMYKWOG-B)")
    print("===============================================================")
    
    out_dir = "scratch/production_models"
    
    # 1. Inland Mountain with Open Magnet Pocket
    m_inland_open, tag1 = build_production_mountain(res=181, is_coastal_port=False, magnet_mode="open", output_dir=out_dir)
    
    # 2. Inland Mountain with Encapsulated Magnet Chamber
    m_inland_encap, tag2 = build_production_mountain(res=181, is_coastal_port=False, magnet_mode="encapsulated", output_dir=out_dir)
    
    # 3. Coastal Mountain with Ore Port (2:1) & Open Magnet
    m_port_open, tag3 = build_production_mountain(res=181, is_coastal_port=True, magnet_mode="open", output_dir=out_dir)
    
    # 4. Generate High-Res Photographic CAD Renders using Matplotlib 3D
    generate_renders(m_inland_open, out_dir)
    
    print("\n[COMPLETE] Full production suite successfully generated.")

def generate_renders(mesh, out_dir):
    print("\nGenerating photographic renders...")
    verts = mesh.vertices
    faces = mesh.faces
    
    fig = plt.figure(figsize=(12, 10), facecolor='#090e1a')
    ax = fig.add_subplot(111, projection='3d', facecolor='#090e1a')
    
    # Subsample faces for clean visualization
    step = 3
    plot_faces = faces[::step]
    
    # Calculate lighting / z color shading
    face_z = np.mean(verts[plot_faces, 2], axis=1)
    norm = plt.Normalize(vmin=25, vmax=83)
    colors = plt.cm.copper(norm(face_z))
    # Snow peaks white
    colors[face_z >= 66.0] = [0.95, 0.98, 1.0, 1.0]
    
    poly = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                           triangles=plot_faces,
                           cmap='terrain',
                           edgecolor='none',
                           linewidth=0,
                           antialiased=True,
                           alpha=0.95)
                           
    ax.view_init(elev=35, azim=225)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    render_iso = os.path.join(out_dir, "render_mountain_isometric.png")
    plt.savefig(render_iso, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close()
    print(f"  Render saved: {render_iso}")
    
    # Top orthographic view
    fig2 = plt.figure(figsize=(10, 10), facecolor='#090e1a')
    ax2 = fig2.add_subplot(111, projection='3d', facecolor='#090e1a')
    ax2.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                     triangles=plot_faces,
                     cmap='copper',
                     edgecolor='#334155',
                     linewidth=0.1,
                     alpha=0.95)
    ax2.view_init(elev=90, azim=-90)
    ax2.set_axis_off()
    fig2.subplots_adjust(left=0, right=1, bottom=0, top=1)
    render_top = os.path.join(out_dir, "render_mountain_top_ortho.png")
    plt.savefig(render_top, dpi=200, bbox_inches='tight', facecolor=fig2.get_facecolor(), pad_inches=0)
    plt.close()
    print(f"  Render saved: {render_top}")

if __name__ == '__main__':
    generate_full_suite()
