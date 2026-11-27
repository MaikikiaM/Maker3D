import numpy as np
import trimesh
import os

def generate_alpine_mountain(res=251):
    R = 64.0
    apothem = R * np.cos(np.pi / 6) # 55.4256 mm
    z_base = 25.0
    z_peak = 83.0
    max_relief = z_peak - z_base # 58.0 mm
    
    xs = np.linspace(-apothem, apothem, res)
    ys = np.linspace(-R, R, int(res * (2 * R) / (2 * apothem)))
    X, Y = np.meshgrid(xs, ys)
    
    # Hexagon boundary distance
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    d_edges = np.zeros((6, len(ys), len(xs)))
    for k in range(6):
        d_edges[k] = apothem - (X * normals[k, 0] + Y * normals[k, 1])
    dist_edge = np.min(d_edges, axis=0)
    mask_hex = dist_edge >= 0.0
    
    # 6 vertices
    v_angles = (30.0 + 60.0 * np.arange(6)) * np.pi / 180.0
    vx = R * np.cos(v_angles)
    vy = R * np.sin(v_angles)
    dist_vert = np.full(X.shape, 999.0)
    for k in range(6):
        dist_vert = np.minimum(dist_vert, np.hypot(X - vx[k], Y - vy[k]))
        
    def smoothstep(e0, e1, v):
        t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
        
    # Road clearance: 3.5mm shelf, smooth transition to 8.0mm
    # Vertex clearance: 10mm radius (dia 20mm), smooth transition to 14.0mm
    clearance = smoothstep(3.5, 8.0, dist_edge) * smoothstep(10.0, 14.0, dist_vert)
    
    # Spines and ridges
    spines = [
        (-4.0, 6.0, -28.0, 32.0, 58.0, 15.0, 10.0),
        (-4.0, 6.0, 14.0, 14.0, 58.0, 47.0, 8.0),
        (14.0, 14.0, 32.0, 22.0, 47.0, 12.0, 9.0),
        (14.0, 14.0, 24.0, -8.0, 47.0, 25.0, 9.0),
        (-4.0, 6.0, -12.0, -18.0, 58.0, 38.0, 8.5),
        (-12.0, -18.0, -32.0, -26.0, 38.0, 10.0, 9.0),
        (-12.0, -18.0, 8.0, -28.0, 38.0, 14.0, 9.0),
    ]
    
    H_spines = np.zeros_like(X)
    for x1, y1, x2, y2, h1, h2, sharp in spines:
        dx, dy = x2 - x1, y2 - y1
        L2 = dx*dx + dy*dy
        t = np.clip(((X - x1)*dx + (Y - y1)*dy) / L2, 0.0, 1.0)
        px = x1 + t * dx
        py = y1 + t * dy
        dist_spine = np.hypot(X - px, Y - py)
        h_line = (1.0 - t) * h1 + t * h2
        h_contrib = h_line * np.exp(-dist_spine / sharp) * (1.0 - 0.25 * np.sin(dist_spine * 0.4))
        H_spines = np.maximum(H_spines, h_contrib)
        
    r_main = np.hypot(X - (-4.0), Y - 6.0)
    ang_main = np.arctan2(Y - 6.0, X - (-4.0))
    facet = np.abs(np.cos(1.5 * ang_main + 0.4))
    h_horn = 58.0 * np.exp(-r_main / 18.0) * (0.85 + 0.35 * facet**1.5)
    
    r_p2 = np.hypot(X - 14.0, Y - 14.0)
    h_p2 = 47.0 * np.exp(-r_p2 / 14.0)
    
    r_p3 = np.hypot(X - (-12.0), Y - (-18.0))
    h_p3 = 38.0 * np.exp(-r_p3 / 15.0)
    
    terrain = np.maximum(H_spines, np.maximum(h_horn, np.maximum(h_p2, h_p3)))
    
    # Multifractal noise
    noise = np.zeros_like(X)
    frequencies = [0.08, 0.16, 0.32, 0.65, 1.30, 2.60]
    amplitudes = [7.5, 4.2, 2.3, 1.2, 0.6, 0.25]
    angles_rot = [0.31, 1.15, 2.47, 0.78, 1.89, 2.92]
    
    for f, a, ang in zip(frequencies, amplitudes, angles_rot):
        Xr = X * np.cos(ang) - Y * np.sin(ang)
        Yr = X * np.sin(ang) + Y * np.cos(ang)
        n1 = 1.0 - np.abs(np.sin(Xr * f + 1.2) * np.cos(Yr * f * 1.1 + 0.7))
        n1 = n1 * n1 * n1
        noise += a * (n1 - 0.5)
        
    terrain = terrain + noise * (terrain / 58.0)**0.5
    terrain = np.maximum(0.0, terrain)
    
    p_max = np.max(terrain)
    if p_max > 0:
        terrain = terrain * (58.0 / p_max)
        
    sculpted = terrain * clearance
    Z_top = z_base + sculpted
    
    print("Alpine terrain generated:")
    print(f"  Min Z (shelf): {np.min(Z_top[mask_hex]):.2f} mm")
    print(f"  Max Z (summit): {np.max(Z_top[mask_hex]):.2f} mm")
    return X, Y, Z_top, mask_hex, apothem, R, z_base

if __name__ == '__main__':
    generate_alpine_mountain(res=101)
