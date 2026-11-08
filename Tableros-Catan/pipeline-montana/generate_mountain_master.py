import numpy as np
import trimesh
import os

def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def build_full_3d_mountain_mesh(res=181, output_path="scratch/catan_mountain_high_res.obj"):
    R = 64.0  # Outer radius (mm)
    apothem = R * np.cos(np.pi / 6)  # ~55.4256 mm
    z_base = 25.0  # Base zocalo height
    z_peak = 83.0  # Peak height
    
    # 1. Hexagon polygon boundary vertices (6 vertices in CCW order)
    # Pointy-topped hexagon: angles 30, 90, 150, 210, 270, 330 deg
    v_angles = (30.0 + 60.0 * np.arange(6)) * np.pi / 180.0
    hex_vx = R * np.cos(v_angles)
    hex_vy = R * np.sin(v_angles)
    hex_poly_2d = np.column_stack([hex_vx, hex_vy])
    
    # Grid inside bounding box
    xs = np.linspace(-apothem, apothem, res)
    ys = np.linspace(-R, R, int(res * (2 * R) / (2 * apothem)))
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys)
    
    # Distance to 6 edges
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    d_edges = np.zeros((6, len(ys), len(xs)))
    for k in range(6):
        d_edges[k] = apothem - (X * normals[k, 0] + Y * normals[k, 1])
    dist_edge = np.min(d_edges, axis=0)
    mask_hex = dist_edge >= 0.0
    
    # 6 Vertices distance
    dist_vert = np.full(X.shape, 999.0)
    for k in range(6):
        dist_vert = np.minimum(dist_vert, np.hypot(X - hex_vx[k], Y - hex_vy[k]))
        
    # Clearance factor
    edge_factor = smoothstep(3.5, 7.5, dist_edge)
    vert_factor = smoothstep(10.0, 14.0, dist_vert)
    clearance_factor = edge_factor * vert_factor
    
    # Topography: Primary Peak Alpha
    dx1 = X - (-8.0)
    dy1 = Y - 8.0
    r1 = np.hypot(dx1, dy1)
    ang1 = np.arctan2(dy1, dx1)
    ar1 = np.maximum(0, np.cos(ang1 - 2.44))**2.5
    ar2 = np.maximum(0, np.cos(ang1 - (-1.40)))**2.5
    ar3 = np.maximum(0, np.cos(ang1 - 0.35))**2.5
    ridges1 = (ar1 + ar2 + ar3) * 0.45
    h_alpha = 58.0 * np.exp(-r1 / 20.0) * (1.0 + ridges1)
    
    # Secondary Peak Beta
    dx2 = X - 12.0
    dy2 = Y - 20.0
    r2 = np.hypot(dx2, dy2)
    ang2 = np.arctan2(dy2, dx2)
    ar_beta = np.maximum(0, np.cos(ang2 - np.pi))**2.0 * 0.35
    h_beta = 39.0 * np.exp(-r2 / 16.0) * (1.0 + ar_beta)
    
    # Saddle
    pA = np.array([-8.0, 8.0])
    pB = np.array([12.0, 20.0])
    AB = pB - pA
    len_AB2 = np.dot(AB, AB)
    t_proj = np.clip(((X - pA[0]) * AB[0] + (Y - pA[1]) * AB[1]) / len_AB2, 0.0, 1.0)
    proj_X = pA[0] + t_proj * AB[0]
    proj_Y = pA[1] + t_proj * AB[1]
    d_ridge = np.hypot(X - proj_X, Y - proj_Y)
    h_saddle = 32.0 * np.exp(-d_ridge / 6.0) * np.sin(t_proj * np.pi)**0.5
    
    # Quarry / Iron Ore Benches
    dx_q = X - (-12.0)
    dy_q = Y - (-16.0)
    r_q = np.hypot(dx_q, dy_q)
    quarry_steps = 3.5 * np.sin(r_q * 0.8) * np.exp(-r_q / 22.0)
    
    # Rock crags noise
    f1 = 0.22
    f2 = 0.45
    f3 = 0.90
    rock_noise = (
        2.5 * np.sin(X * f1 + Y * 0.15) * np.cos(Y * f1 - X * 0.1) +
        1.2 * np.sin(X * f2 * 1.3 - Y * f2 * 0.8 + 1.2) +
        0.6 * np.cos(X * f3 * 0.7 + Y * f3 * 1.4 + 2.5) +
        0.3 * np.sin(X * 1.8 + Y * 1.8)
    )
    
    raw_relief = np.maximum(h_alpha, np.maximum(h_beta, h_saddle)) + quarry_steps + rock_noise
    raw_relief = np.maximum(0.0, raw_relief)
    peak_val = np.max(raw_relief)
    if peak_val > 0:
        raw_relief = raw_relief * (58.0 / peak_val)
        
    sculpted_relief = raw_relief * clearance_factor
    
    # Chit Pocket & Terrace
    cx_chit = 16.0
    cy_chit = -14.0
    r_chit_dist = np.hypot(X - cx_chit, Y - cy_chit)
    z_plateau_relief = 12.0
    plateau_factor = smoothstep(18.0, 14.5, r_chit_dist)
    sculpted_relief = sculpted_relief * (1.0 - plateau_factor) + z_plateau_relief * plateau_factor
    
    inside_pocket = r_chit_dist <= 13.0
    pocket_wall = smoothstep(13.3, 13.0, r_chit_dist)
    sculpted_relief = sculpted_relief * (1.0 - pocket_wall) + (z_plateau_relief - 1.8) * pocket_wall
    
    notch_dist = np.hypot(X - (cx_chit + 13.0), Y - cy_chit)
    notch_factor = smoothstep(3.0, 1.8, notch_dist)
    sculpted_relief = sculpted_relief * (1.0 - notch_factor) + (z_plateau_relief - 2.5) * notch_factor
    
    Z_top = z_base + sculpted_relief
    Z_top[~mask_hex] = z_base

    # Triangulate top surface
    # Build vertex indices for points strictly inside hexagon
    vertex_map = np.full(X.shape, -1, dtype=int)
    v_list = []
    
    # Color attribute arrays
    c_list = []
    
    for j in range(len(ys)):
        for i in range(len(xs)):
            if mask_hex[j, i]:
                idx = len(v_list)
                vertex_map[j, i] = idx
                x_val = X[j, i]
                y_val = Y[j, i]
                z_val = Z_top[j, i]
                v_list.append([x_val, y_val, z_val])
                
                # Assign CMYKWOG Color weights
                # Tool W (Snow): Z >= 66mm or high ridge
                if z_val >= 67.0:
                    c = [245, 248, 255, 255] # White
                elif z_val >= 58.0 and y_val > 5.0:
                    c = [230, 240, 250, 255] # Glacial white/cyan
                # Tool O (Iron Ore / Quarry):
                elif -25.0 < x_val < 0.0 and -25.0 < y_val < -5.0 and z_val < 48.0:
                    c = [190, 85, 35, 255] # Rust orange / Terracotta
                # Tool G (Alpine Moss): low elevation near boundary
                elif z_val < 27.5 and dist_edge[j, i] > 4.5:
                    c = [75, 130, 60, 255] # Alpine moss green
                # Chit plateau ring:
                elif r_chit_dist[j, i] < 15.0:
                    c = [120, 115, 110, 255] # Carved grey stone
                # Tool T3 / Key (Granite):
                else:
                    shade = int(np.clip(110 + (z_val - 25.0) * 1.8, 90, 180))
                    c = [shade, shade, int(shade * 1.05), 255] # Granite grey
                c_list.append(c)

    faces = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            idx00 = vertex_map[j, i]
            idx10 = vertex_map[j, i+1]
            idx01 = vertex_map[j+1, i]
            idx11 = vertex_map[j+1, i+1]
            
            # Quad consists of two triangles if all 4 or 3 vertices exist
            if idx00 != -1 and idx10 != -1 and idx01 != -1 and idx11 != -1:
                faces.append([idx00, idx10, idx11])
                faces.append([idx00, idx11, idx01])
            elif idx00 != -1 and idx10 != -1 and idx11 != -1:
                faces.append([idx00, idx10, idx11])
            elif idx00 != -1 and idx11 != -1 and idx01 != -1:
                faces.append([idx00, idx11, idx01])
            elif idx00 != -1 and idx10 != -1 and idx01 != -1:
                faces.append([idx00, idx10, idx01])
            elif idx10 != -1 and idx11 != -1 and idx01 != -1:
                faces.append([idx10, idx11, idx01])
                
    verts = np.array(v_list)
    faces = np.array(faces)
    colors = np.array(c_list, dtype=np.uint8)
    
    print(f"Top surface mesh generated: {len(verts)} vertices, {len(faces)} faces.")
    
    top_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, process=False)
    
    # Save top surface preview
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    top_mesh.export(output_path)
    print(f"Exported preview mesh to {output_path}")
    
    return top_mesh

if __name__ == '__main__':
    build_full_3d_mountain_mesh(res=181)
