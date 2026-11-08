import math
import time
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

print("=== TESTING CHISELED ALPINISM GEOMETRY (CRISP FACETS, KNIFE-EDGES, LEDGES) ===")
t0 = time.time()

# Grid density: 601 x 695 (sub-0.18mm spacing!)
nx, ny = 601, 695
R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
xs = np.linspace(-APOTHEM, APOTHEM, nx, dtype=np.float32)
ys = np.linspace(-R, R, ny, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)

normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

# Base pyramidal skeleton from master alpinism
r_main = np.hypot(X - (-4.0), Y - 6.0)
ang_main = np.arctan2(Y - 6.0, X - (-4.0))

# 4 sharp alpine facets with chiseled ridge crests
# Main horn: Matterhorn pyramidal apex
facet_horn = 0.86 + 0.36 * np.abs(np.cos(2.0 * ang_main + 0.35))**1.6
h_horn = 58.0 * np.exp(-(r_main / 23.0)**1.38) * facet_horn

# East Cathedral Spire: (18.0, 16.0)
r_p2 = np.hypot(X - 18.0, Y - 16.0)
ang_p2 = np.arctan2(Y - 16.0, X - 18.0)
h_p2 = 50.0 * np.exp(-(r_p2 / 17.5)**1.36) * (0.86 + 0.30 * np.abs(np.cos(2.0 * ang_p2))**1.3)

# South Tooth: (-12.0, -18.0)
r_p3 = np.hypot(X - (-12.0), Y - (-18.0))
h_p3 = 44.0 * np.exp(-(r_p3 / 18.5)**1.36) * (0.88 + 0.25 * np.cos(1.8 * ang_main)**2.0)

# West Shoulder: (-26.0, -4.0)
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

# -------------------------------------------------------------
# 2. ADD CRISP FRACTURED ROCK GEOMETRIC RELIEF (NO BLURRY DOME!)
# -------------------------------------------------------------
# A. Sharp Knife-Edge Ridge Noise (Fractured crags)
# Ridge noise: 1.0 - 2.0 * |sin(p)|
ridge_f1 = 1.0 - 2.0 * np.abs(np.sin((X * 0.35 + Y * 0.28) * 1.4))
ridge_f2 = 1.0 - 2.0 * np.abs(np.sin((X * 0.72 - Y * 0.65) * 1.5))
ridge_f3 = 1.0 - 2.0 * np.abs(np.sin((X * 1.6 + Y * 1.4) * 1.6))
chiseled_crags = 1.8 * ridge_f1 + 0.9 * ridge_f2 + 0.4 * ridge_f3

# B. Horizontal Metamorphic Terraces & Ledges (0.75mm steps)
Z_temp = massif + 25.0
strata_angle = (X * 0.18 + Y * 0.10 + Z_temp * 0.32)
stepped_terraces = 0.85 * np.sin(strata_angle * 1.8)

# C. Chiseled Couloirs (Erosion gullies carved into the rock)
couloir_carving = 1.6 * np.abs(np.sin(ang_main * 4.0 + 0.5))**2.2 * np.exp(-(r_main / 30.0)**1.5)

# Modulate geometric texture by elevation
relief_env = (massif / 58.0)**0.85
Z_sculpted = massif + (chiseled_crags + stepped_terraces - couloir_carving) * relief_env
Z_sculpted = np.maximum(0.0, Z_sculpted)
Z_sculpted = Z_sculpted * (58.0 / np.max(Z_sculpted))

# Road shelf & vertex clearance
def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

v_angles = [(30 + 60 * k) * math.pi / 180 for k in range(6)]
hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles])
dist_vert = np.full_like(X, 999.0)
for vx, vy in hex_corners:
    dist_vert = np.minimum(dist_vert, np.hypot(X - vx, Y - vy))

edge_factor = smoothstep(3.5, 12.0, dist_edge)
vert_factor = smoothstep(10.0, 15.0, dist_vert)
clearance = edge_factor * vert_factor

Z_final = 25.0 + Z_sculpted * clearance
Z_final[dist_edge < 0] = 25.0

# Pin exact summit
ix_peak = int(np.round((-4.0 - (-APOTHEM)) / (2 * APOTHEM) * (nx - 1)))
iy_peak = int(np.round((6.0 - (-R)) / (2 * R) * (ny - 1)))
Z_final[iy_peak, ix_peak] = 83.00

print(f"Chiseled Alpinism Geometry generated: {ny}x{nx}, Z in [{np.min(Z_final):.2f}, {np.max(Z_final):.2f}] mm")

# Compute normal hillshade to inspect geometric sharpness
dx = float(xs[1] - xs[0]); dy = float(ys[1] - ys[0])
gy, gx = np.gradient(Z_final, dy, dx)
norm = np.sqrt(gx**2 + gy**2 + 1.0)
sun_x = math.cos(math.radians(48))*math.sin(math.radians(315))
sun_y = math.cos(math.radians(48))*math.cos(math.radians(315))
sun_z = math.sin(math.radians(48))
hs = np.clip((-gx/norm)*sun_x + (-gy/norm)*sun_y + (1.0/norm)*sun_z, 0.0, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#090d16')
axes[0].imshow(hs, cmap='gray', origin='lower')
axes[0].set_title("Geometría Chiseled Alpinism 601x695 (Aristas y Terrazas Nítidas)", color='white', fontsize=12)
axes[0].axis('off')

# Zoom on the summit arête
crop_hs = hs[iy_peak-80:iy_peak+80, ix_peak-80:ix_peak+80]
axes[1].imshow(crop_hs, cmap='gray', origin='lower')
axes[1].set_title("Super-Zoom Cumbre Matterhorn: Aristas Vivas y Fisuras Geométricas", color='white', fontsize=12)
axes[1].axis('off')

plt.tight_layout()
plt.savefig('scratch/chiseled_alpinism_preview.png', dpi=200, facecolor='#090d16')
print(f"Saved scratch/chiseled_alpinism_preview.png in {time.time()-t0:.2f}s")
