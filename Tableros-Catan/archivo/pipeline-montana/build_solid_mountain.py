import numpy as np
import trimesh
from scipy.spatial import Delaunay
import os

def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def build_solid_watertight_mountain(res=141, output_path="scratch/catan_mountain_solid.stl", z_interpolator=None):
    R = 64.0  # Outer radius (mm)
    apothem = R * np.cos(np.pi / 6)  # ~55.4256 mm
    z_base = 25.0  # Base zocalo height
    z_peak = 83.0  # Peak height
    target_relief = z_peak - z_base  # 58.0 mm
    
    # 1. Hexagon polygon boundary vertices (6 vertices in CCW order)
    v_angles = (30.0 + 60.0 * np.arange(6)) * np.pi / 180.0
    hex_vx = R * np.cos(v_angles)
    hex_vy = R * np.sin(v_angles)
    
    # Edges sampling (24 points per edge = 144 boundary points)
    pts_per_edge = 24
    boundary_pts_2d = []
    for k in range(6):
        p1 = np.array([hex_vx[k], hex_vy[k]])
        p2 = np.array([hex_vx[(k+1)%6], hex_vy[(k+1)%6]])
        for s in np.linspace(0, 1, pts_per_edge, endpoint=False):
            boundary_pts_2d.append((1.0 - s) * p1 + s * p2)
    boundary_pts_2d = np.array(boundary_pts_2d) # N_b = 144
    N_b = len(boundary_pts_2d)
    
    # 2. Internal grid for top surface
    xs = np.linspace(-apothem + 1.2, apothem - 1.2, res)
    ys = np.linspace(-R + 2.0, R - 2.0, int(res * (2 * R) / (2 * apothem)))
    X, Y = np.meshgrid(xs, ys)
    
    # 6 edges normals
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    d_edges = np.zeros((6, len(ys), len(xs)))
    for k in range(6):
        d_edges[k] = apothem - (X * normals[k, 0] + Y * normals[k, 1])
    dist_edge = np.min(d_edges, axis=0)
    mask_internal = dist_edge >= 1.2
    
    # Topography function
    def compute_elevation(x_arr, y_arr, d_edge_arr=None):
        if d_edge_arr is None:
            d_ed = np.zeros((6, len(x_arr)))
            for k in range(6):
                d_ed[k] = apothem - (x_arr * normals[k, 0] + y_arr * normals[k, 1])
            d_edge_arr = np.min(d_ed, axis=0)
            
        d_vert = np.full(x_arr.shape, 999.0)
        for k in range(6):
            d_vert = np.minimum(d_vert, np.hypot(x_arr - hex_vx[k], y_arr - hex_vy[k]))
            
        edge_factor = smoothstep(3.5, 7.5, d_edge_arr)
        vert_factor = smoothstep(10.0, 14.0, d_vert)
        clearance_factor = edge_factor * vert_factor
        
        # Primary Peak Alpha (-8.0, 8.0)
        dx1 = x_arr - (-8.0)
        dy1 = y_arr - 8.0
        r1 = np.hypot(dx1, dy1)
        ang1 = np.arctan2(dy1, dx1)
        ar1 = np.maximum(0, np.cos(ang1 - 2.44))**2.5
        ar2 = np.maximum(0, np.cos(ang1 - (-1.40)))**2.5
        ar3 = np.maximum(0, np.cos(ang1 - 0.35))**2.5
        ridges1 = (ar1 + ar2 + ar3) * 0.45
        h_alpha = 58.0 * np.exp(-r1 / 20.0) * (1.0 + ridges1)
        
        # Secondary Peak Beta (12.0, 20.0)
        dx2 = x_arr - 12.0
        dy2 = y_arr - 20.0
        r2 = np.hypot(dx2, dy2)
        ang2 = np.arctan2(dy2, dx2)
        ar_beta = np.maximum(0, np.cos(ang2 - np.pi))**2.0 * 0.35
        h_beta = 39.0 * np.exp(-r2 / 16.0) * (1.0 + ar_beta)
        
        # Saddle
        pA = np.array([-8.0, 8.0])
        pB = np.array([12.0, 20.0])
        AB = pB - pA
        len_AB2 = np.dot(AB, AB)
        t_proj = np.clip(((x_arr - pA[0]) * AB[0] + (y_arr - pA[1]) * AB[1]) / len_AB2, 0.0, 1.0)
        proj_X = pA[0] + t_proj * AB[0]
        proj_Y = pA[1] + t_proj * AB[1]
        d_ridge = np.hypot(x_arr - proj_X, y_arr - proj_Y)
        h_saddle = 32.0 * np.exp(-d_ridge / 6.0) * np.sin(t_proj * np.pi)**0.5
        
        # Quarry
        dx_q = x_arr - (-12.0)
        dy_q = y_arr - (-16.0)
        r_q = np.hypot(dx_q, dy_q)
        quarry_steps = 3.5 * np.sin(r_q * 0.8) * np.exp(-r_q / 22.0)
        
        # Rock crags
        f1, f2, f3 = 0.22, 0.45, 0.90
        rock_noise = (
            2.5 * np.sin(x_arr * f1 + y_arr * 0.15) * np.cos(y_arr * f1 - x_arr * 0.1) +
            1.2 * np.sin(x_arr * f2 * 1.3 - y_arr * f2 * 0.8 + 1.2) +
            0.6 * np.cos(x_arr * f3 * 0.7 + y_arr * f3 * 1.4 + 2.5) +
            0.3 * np.sin(x_arr * 1.8 + y_arr * 1.8)
        )
        
        raw_relief = np.maximum(h_alpha, np.maximum(h_beta, h_saddle)) + quarry_steps + rock_noise
        raw_relief = np.maximum(0.0, raw_relief)
        
        # Exact scaling to target_relief = 58.0 mm
        max_val = np.max(raw_relief)
        if max_val > 0:
            raw_relief = raw_relief * (target_relief / max_val)
            
        sculpted_relief = raw_relief * clearance_factor
        
        # Chit pocket at (16.0, -14.0)
        cx_chit = 16.0
        cy_chit = -14.0
        r_chit_dist = np.hypot(x_arr - cx_chit, y_arr - cy_chit)
        z_plateau_relief = 12.0
        plateau_factor = smoothstep(18.0, 14.5, r_chit_dist)
        sculpted_relief = sculpted_relief * (1.0 - plateau_factor) + z_plateau_relief * plateau_factor
        
        pocket_wall = smoothstep(13.3, 13.0, r_chit_dist)
        sculpted_relief = sculpted_relief * (1.0 - pocket_wall) + (z_plateau_relief - 1.8) * pocket_wall
        
        notch_dist = np.hypot(x_arr - (cx_chit + 13.0), y_arr - cy_chit)
        notch_factor = smoothstep(3.0, 1.8, notch_dist)
        sculpted_relief = sculpted_relief * (1.0 - notch_factor) + (z_plateau_relief - 2.5) * notch_factor
        
        return z_base + sculpted_relief
        
    int_x = X[mask_internal]
    int_y = Y[mask_internal]
    if z_interpolator is not None:
        int_z = z_interpolator(np.column_stack((int_y, int_x))).astype(np.float32)
    else:
        int_z = compute_elevation(int_x, int_y, dist_edge[mask_internal])
    
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
    
    # 3. Bottom face ring points with magnet pocket
    N_mag = 32
    r_mag = 5.1 # radius 5.1mm -> dia 10.2mm
    mag_angles = np.linspace(0, 2*np.pi, N_mag, endpoint=False)
    mag_rim_pts = np.column_stack([r_mag * np.cos(mag_angles), r_mag * np.sin(mag_angles), np.zeros(N_mag)])
    mag_ceil_pts = np.column_stack([r_mag * np.cos(mag_angles), r_mag * np.sin(mag_angles), np.full(N_mag, 5.2)])
    mag_ceil_center = np.array([[0.0, 0.0, 5.2]])
    bottom_bound_pts = np.column_stack([bound_x, bound_y, np.zeros(N_b)])
    
    # Bottom ring internal sampling for clean Delaunay
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
    
    # Master vertex table:
    # 0 to len(all_top_pts)-1: Top surface vertices (first N_b are boundary at Z=25)
    # Next N_b: bottom_bound_pts (Z=0)
    # Next N_mag: mag_rim_pts (Z=0)
    # Next len(int_bot_pts): int_bot_pts (Z=0)
    # Next N_mag: mag_ceil_pts (Z=5.2)
    # Last 1: mag_ceil_center (Z=5.2)
    
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
    
    # Side walls (connect top boundary 0..N_b-1 with bottom boundary offset_bot_bound..+N_b-1)
    for i in range(N_b):
        top_curr = i
        top_next = (i + 1) % N_b
        bot_curr = offset_bot_bound + i
        bot_next = offset_bot_bound + (i + 1) % N_b
        all_faces.append([top_curr, bot_curr, bot_next])
        all_faces.append([top_curr, bot_next, top_next])
        
    # Bottom ring faces (map all_bot_2d indices to v_total)
    for tri in bot_ring_tris:
        mapped = []
        for idx in tri:
            if idx < N_b:
                mapped.append(offset_bot_bound + idx)
            elif idx < N_b + N_mag:
                mapped.append(offset_mag_rim + (idx - N_b))
            else:
                mapped.append(offset_int_bot + (idx - N_b - N_mag))
        # Normal pointing downwards (-Z)
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
    
    print("Solid Mesh Statistics:")
    print(f"  Vertices: {len(solid_mesh.vertices)}")
    print(f"  Faces: {len(solid_mesh.faces)}")
    print(f"  Is watertight manifold: {solid_mesh.is_watertight}")
    print(f"  Volume: {solid_mesh.volume / 1000.0:.2f} cm3")
    print(f"  Estimated PLA weight: {solid_mesh.volume * 0.00124:.1f} g")
    print(f"  Bounding box min: {np.round(solid_mesh.bounds[0], 2)}")
    print(f"  Bounding box max: {np.round(solid_mesh.bounds[1], 2)}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    solid_mesh.export(output_path)
    print(f"Successfully exported solid watertight mesh to: {output_path}")
    return solid_mesh

if __name__ == '__main__':
    build_solid_watertight_mountain(res=121)
