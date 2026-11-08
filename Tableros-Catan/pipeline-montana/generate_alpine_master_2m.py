import os
import math
import time
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

print("=== GENERATING HIGH-FIDELITY ALPINE GEOMORPHOLOGY (NO SINES, REAL FRACTAL) ===")
t0 = time.time()

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF = Z_PEAK - Z_BASE # 58.00 mm

# High-resolution grid for > 2,000,000 triangles
# 1150 x 1328 grid -> 1,144,200 inside hex -> 2,283,450 triangles on top!
nx, ny = 1150, 1328
xs = np.linspace(-APOTHEM, APOTHEM, nx, dtype=np.float32)
ys = np.linspace(-R, R, ny, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)

dx = float(xs[1] - xs[0])
dy = float(ys[1] - ys[0])
print(f"Grid dimensions: {ny}x{nx} ({nx*ny} total cells). Cell size: dx={dx*1000:.1f}um, dy={dy*1000:.1f}um")

# Hexagon distance field
normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

# Corner vertices distance for settlement pad clearance
v_angles = [(30 + 60 * k) * math.pi / 180 for k in range(6)]
hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles])
dist_vert = np.full_like(X, 999.0)
for vx, vy in hex_corners:
    dist_vert = np.minimum(dist_vert, np.hypot(X - vx, Y - vy))

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# -------------------------------------------------------------------------
# 1. PRIMARY ALPINE HORNS & TECTONIC UPLIFT (Matterhorn / Fitz Roy style)
# -------------------------------------------------------------------------
print("1. Sculpting primary alpine massif horns...")
# Main summit peak (offset slightly for natural drama)
p_main_x, p_main_y = -3.5, 4.5
r_main = np.hypot(X - p_main_x, Y - p_main_y)
ang_main = np.arctan2(Y - p_main_y, X - p_main_x)

# Secondary summits / gendarmes
r_p2 = np.hypot(X - 16.0, Y - 15.0)
r_p3 = np.hypot(X - (-14.0), Y - (-16.0))
r_p4 = np.hypot(X - (-25.0), Y - (-5.0))

# Horn profiles (exponential power decay, naturally sharp summits)
h_main = 58.0 * np.exp(-(r_main / 24.0)**1.32)
h_p2 = 47.0 * np.exp(-(r_p2 / 18.0)**1.35)
h_p3 = 41.0 * np.exp(-(r_p3 / 19.0)**1.35)
h_p4 = 28.0 * np.exp(-(r_p4 / 15.0)**1.35)

# -------------------------------------------------------------------------
# 2. RADIATING KNIFE-EDGE ARÊTES (Ridge lines)
# -------------------------------------------------------------------------
print("2. Carving knife-edge arêtes (Alpine ridgelines)...")
ridges = [
    # Main horn to secondary horns
    {"x1": p_main_x, "y1": p_main_y, "x2": 16.0, "y2": 15.0, "h1": 58.0, "h2": 47.0, "w": 6.5},
    {"x1": p_main_x, "y1": p_main_y, "x2": -14.0, "y2": -16.0, "h1": 58.0, "h2": 41.0, "w": 7.0},
    {"x1": p_main_x, "y1": p_main_y, "x2": -25.0, "y2": -5.0, "h1": 58.0, "h2": 28.0, "w": 6.8},
    # Main horn to North-West ridge
    {"x1": p_main_x, "y1": p_main_y, "x2": -32.0, "y2": 32.0, "h1": 58.0, "h2": 10.0, "w": 7.2},
    # Secondary peaks to flanks
    {"x1": 16.0, "y1": 15.0, "x2": 36.0, "y2": 24.0, "h1": 47.0, "h2": 8.0, "w": 7.0},
    {"x1": 16.0, "y1": 15.0, "x2": 30.0, "y2": -12.0, "h1": 47.0, "h2": 12.0, "w": 7.5},
    {"x1": -14.0, "y1": -16.0, "x2": -32.0, "y2": -26.0, "h1": 41.0, "h2": 8.0, "w": 7.2},
    {"x1": -14.0, "y1": -16.0, "x2": 14.0, "y2": -26.0, "h1": 41.0, "h2": 10.0, "w": 7.0},
]

H_ridges = np.zeros_like(X)
for rd in ridges:
    dx_r = rd["x2"] - rd["x1"]
    dy_r = rd["y2"] - rd["y1"]
    L2 = dx_r**2 + dy_r**2
    t_proj = np.clip(((X - rd["x1"])*dx_r + (Y - rd["y1"])*dy_r) / L2, 0.0, 1.0)
    px = rd["x1"] + t_proj * dx_r
    py = rd["y1"] + t_proj * dy_r
    dist_rd = np.hypot(X - px, Y - py)
    h_spine = (1.0 - t_proj) * rd["h1"] + t_proj * rd["h2"]
    # Sharp ridge profile (sharp knife-edge at top, flared base)
    h_contrib = h_spine * np.exp(-(dist_rd / rd["w"])**1.28)
    H_ridges = np.maximum(H_ridges, h_contrib)

massif = np.maximum(H_ridges, np.maximum(h_main, np.maximum(h_p2, np.maximum(h_p3, h_p4))))

# Glacial Cirque Hollows (concave scoops carved between the main ridges)
cirque_NW = np.exp(-((X - (-18.0))**2 + (Y - 18.0)**2) / (18.0**2)) * 12.0
cirque_E  = np.exp(-((X - 8.0)**2 + (Y - (-2.0))**2) / (16.0**2)) * 10.0
cirque_SW = np.exp(-((X - (-22.0))**2 + (Y - (-12.0))**2) / (15.0**2)) * 8.0
massif = np.maximum(0.0, massif - (cirque_NW + cirque_E + cirque_SW) * (massif / 58.0))

print(f"Massif base sculpted in {time.time()-t0:.2f}s. Max height: {np.max(massif):.2f} mm")

# Save checkpoint
np.save('scratch/massif_base.npy', massif)
