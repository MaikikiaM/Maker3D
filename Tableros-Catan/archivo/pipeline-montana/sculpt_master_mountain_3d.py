import math
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

print("=== SCULPTING MASTER 3D MOUNTAIN PAINTING (SOARING ALPINISM, ZERO VEINS, ZERO PIXELS) ===")

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256258 mm
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF_H = Z_PEAK - Z_BASE # 58.0 mm

# High-Resolution Grid: 351 x 405 (~0.31 mm vertex resolution across 110.85 x 128 mm)
nx, ny = 351, 405
xs = np.linspace(-APOTHEM, APOTHEM, nx)
ys = np.linspace(-R, R, ny)
X, Y = np.meshgrid(xs, ys)

# Exact Hexagon Normals & Boundary Distance
normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    d = APOTHEM - (X * nx_h + Y * ny_h)
    dist_edge = np.minimum(dist_edge, d)

v_angles = [(30 + 60 * k) * math.pi / 180 for k in range(6)]
hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles])
dist_vert = np.full_like(X, 999.0)
for vx, vy in hex_corners:
    dist_vert = np.minimum(dist_vert, np.hypot(X - vx, Y - vy))

# -------------------------------------------------------------
# 1. SOARING ALPINE SPINE & HORN ARCHITECTURE
# -------------------------------------------------------------
# Knife-edge Arêtes (The Mountain's Skeleton)
spines = [
    # NW Ridge (Main summit to NW corner)
    {"x1": -4.0, "y1": 6.0, "x2": -32.0, "y2": 34.0, "h1": 58.0, "h2": 12.0, "sharp": 7.5, "curv": 1.1},
    # East Ridge (Connecting Main peak to East Peak)
    {"x1": -4.0, "y1": 6.0, "x2": 16.0, "y2": 16.0, "h1": 58.0, "h2": 48.0, "sharp": 6.8, "curv": 1.0},
    # East Peak outward spurs
    {"x1": 16.0, "y1": 16.0, "x2": 36.0, "y2": 24.0, "h1": 48.0, "h2": 10.0, "sharp": 7.5, "curv": 1.05},
    {"x1": 16.0, "y1": 16.0, "x2": 28.0, "y2": -8.0, "h1": 48.0, "h2": 18.0, "sharp": 7.2, "curv": 1.05},
    # South Ridge (Connecting Main peak to South Peak)
    {"x1": -4.0, "y1": 6.0, "x2": -12.0, "y2": -18.0, "h1": 58.0, "h2": 40.0, "sharp": 7.0, "curv": 1.0},
    # South Peak outward spurs
    {"x1": -12.0, "y1": -18.0, "x2": -32.0, "y2": -28.0, "h1": 40.0, "h2": 8.0, "sharp": 8.0, "curv": 1.1},
    {"x1": -12.0, "y1": -18.0, "x2": 14.0, "y2": -26.0, "h1": 40.0, "h2": 12.0, "sharp": 7.8, "curv": 1.1},
    # West Buttress
    {"x1": -4.0, "y1": 6.0, "x2": -26.0, "y2": -4.0, "h1": 58.0, "h2": 22.0, "sharp": 7.5, "curv": 1.1},
    {"x1": -26.0, "y1": -4.0, "x2": -40.0, "y2": -12.0, "h1": 22.0, "h2": 6.0, "sharp": 8.5, "curv": 1.15}
]

H_spines = np.zeros_like(X)
for sp in spines:
    dx = sp["x2"] - sp["x1"]
    dy = sp["y2"] - sp["y1"]
    L2 = dx*dx + dy*dy
    t = np.clip(((X - sp["x1"])*dx + (Y - sp["y1"])*dy) / L2, 0.0, 1.0)
    px = sp["x1"] + t * dx
    py = sp["y1"] + t * dy
    dist_sp = np.hypot(X - px, Y - py)
    h_line = (1.0 - t) * sp["h1"] + t * sp["h2"]
    h_contrib = h_line * np.exp(-(dist_sp / sp["sharp"])**sp["curv"])
    H_spines = np.maximum(H_spines, h_contrib)

# Primary Pyramidal Horn (Matterhorn Cervino)
r_main = np.hypot(X - (-4.0), Y - 6.0)
ang_main = np.arctan2(Y - 6.0, X - (-4.0))
# Matterhorn 4-faceted pyramidal horn
facet_horn = 0.82 + 0.38 * np.abs(np.cos(1.5 * ang_main + 0.4))**1.4
h_horn = 58.0 * np.exp(-(r_main / 16.5)**1.25) * facet_horn

# East Peak (Grand Teton cathedral spire)
r_p2 = np.hypot(X - 16.0, Y - 16.0)
ang_p2 = np.arctan2(Y - 16.0, X - 16.0)
h_p2 = 48.0 * np.exp(-(r_p2 / 13.0)**1.3) * (0.85 + 0.30 * np.abs(np.cos(2.0 * ang_p2))**1.2)

# South Peak
r_p3 = np.hypot(X - (-12.0), Y - (-18.0))
h_p3 = 40.0 * np.exp(-(r_p3 / 14.0)**1.3) * (0.88 + 0.25 * np.cos(1.8 * ang_main)**2.0)

# West Shoulder
r_p4 = np.hypot(X - (-26.0), Y - (-4.0))
h_p4 = 24.0 * np.exp(-(r_p4 / 12.0)**1.35)

# Combine soaring mountain massifs
massif_core = np.maximum(H_spines, np.maximum(h_horn, np.maximum(h_p2, np.maximum(h_p3, h_p4))))

# Broad Concave Apron (tapers naturally to zero across the base)
r_tile = np.hypot(X, Y)
base_apron = 58.0 * np.clip(1.0 - (r_tile / 54.0), 0.0, 1.0)**1.75
mountain_sculpture = np.maximum(massif_core, base_apron * 0.45)

# -------------------------------------------------------------
# 2. CONTINUOUS FRACTAL ROCK MICRO-FACETING (NO PIXELS)
# -------------------------------------------------------------
# Multi-octave continuous noise using irrational rotations
noise = np.zeros_like(X)
frequencies = [0.07, 0.15, 0.32, 0.68, 1.45, 3.10]
amplitudes = [5.8, 3.2, 1.6, 0.8, 0.35, 0.14]
angles_rot = [0.38, 1.21, 2.34, 0.82, 1.94, 2.81]

for f, a, ang in zip(frequencies, amplitudes, angles_rot):
    Xr = X * math.cos(ang) - Y * math.sin(ang)
    Yr = X * math.sin(ang) + Y * math.cos(ang)
    # Sharp craggy cleavage formula (continuous Voronoi/angular profile)
    n1 = 1.0 - np.abs(np.sin(Xr * f + 1.1) * np.cos(Yr * f * 1.08 + 0.6))
    noise += a * (n1**1.8 - 0.5)

# Metamorphic Bedding / Rock Strata (tilted at 22 degrees)
strata = 0.85 * np.sin((X * 0.24 + Y * 0.14 + mountain_sculpture * 0.4) * 1.5)

# Scale micro-texture by elevation so peaks are craggy, valleys are smooth scree
relief_textured = mountain_sculpture + (noise + strata) * (mountain_sculpture / 58.0)**0.65
relief_textured = np.maximum(0.0, relief_textured)
relief_textured = relief_textured * (RELIEF_H / np.max(relief_textured))

# -------------------------------------------------------------
# 3. ROAD SHELF & VERTEX PAD CLEARANCES
# -------------------------------------------------------------
def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# Flat road shelf: 3.5mm from edge. Gentle talus landing from 3.5mm to 12.0mm
edge_factor = smoothstep(3.5, 12.0, dist_edge)
# Flat vertex pads: 10mm radius (20mm diam). Gentle talus landing from 10.0mm to 15.0mm
vert_factor = smoothstep(10.0, 15.0, dist_vert)

clearance = edge_factor * vert_factor

final_relief = relief_textured * clearance
Z_top = Z_BASE + final_relief
Z_top[dist_edge < 0] = Z_BASE

print(f"Sculpted Topography Complete: Z min = {Z_top.min():.2f} mm, max = {Z_top.max():.2f} mm")

# -------------------------------------------------------------
# 4. FULL-COLOR 3D PAINTING (CMYKWOG-B: ZERO RED VEINS)
# -------------------------------------------------------------
dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
gy, gx = np.gradient(Z_top, dy, dx)
slope = np.hypot(gx, gy)
slope_deg = np.degrees(np.arctan(slope))
laplace = ndi.laplace(Z_top)

# Directional Sun lighting (NW light: 315 deg azimuth, 48 deg altitude)
az = 315.0 * math.pi / 180.0; alt = 48.0 * math.pi / 180.0
sun_x = math.cos(alt)*math.sin(az); sun_y = math.cos(alt)*math.cos(az); sun_z = math.sin(alt)
norm = np.sqrt(gx**2 + gy**2 + 1.0)
hs = np.clip((-gx/norm)*sun_x + (-gy/norm)*sun_y + (1.0/norm)*sun_z, 0.0, 1.0)

# Natural Altitudinal Shading:
# 1. Snow (White T7, 0.2mm nozzle): High elevations (> 48mm), low slopes (< 48 deg), couloirs
snow_score = (final_relief / 58.0)**1.35 * 1.7 - (slope_deg / 45.0)**1.1 * 0.85 + np.clip(-laplace * 0.4, -0.25, 0.45)
snow_mask = (snow_score > 0.42) & (final_relief > 22.0)

# 2. Glacial Blue Ice (Cyan T1): Shaded cirques, couloir troughs
ice_score = np.clip(-laplace * 0.55, 0.0, 1.0) * np.clip((final_relief - 24.0) / 20.0, 0, 1) * (1.0 - np.clip(snow_score, 0, 1))
ice_mask = (ice_score > 0.26) & (final_relief > 20.0) & (~snow_mask)

# 3. Alpine Tundra / Lichen (Green T5): Low valleys (Z 25.0 to 28.5mm), slope < 32 deg
tundra_score = np.clip((6.0 - final_relief) / 5.0, 0, 1) * np.clip((35.0 - slope_deg) / 20.0, 0, 1) * (dist_edge > 3.0)
tundra_mask = (tundra_score > 0.38) & (final_relief > 0.15) & (~snow_mask)

# 4. Dark Slate Shadow Cliffs (Black T3): Steep rock faces (> 46 deg) and crevices
cliff_mask = (slope_deg > 45.0) & (final_relief > 1.5) & (~snow_mask) & (~ice_mask)

# 5. Sunlit Granite Rock Face (Grey T4): Main mountain body
granite_mask = (final_relief > 0.05) & (~snow_mask) & (~ice_mask) & (~tundra_mask) & (~cliff_mask)

# 6. Road Shelf & Vertex Pads: Flat base at Z=25mm (Tool B)
shelf_mask = (final_relief <= 0.05) & (dist_edge >= 0)

# Tool Map Assignment for 3MF:
# 0 = Base Zócalo / Road shelf (Tool B)
# 1 = Glacial Ice (Tool C)
# 3 = Dark Slate Cliffs (Tool K)
# 4 = Granite Rock Face (Tool W / Grey)
# 5 = Alpine Tundra (Tool G / Green)
# 7 = Alpine Snow (Tool W / White 0.2mm)
tool_map = np.zeros_like(Z_top, dtype=np.uint8)
tool_map[shelf_mask] = 0
tool_map[tundra_mask] = 5
tool_map[granite_mask] = 4
tool_map[cliff_mask] = 3
tool_map[ice_mask] = 1
tool_map[snow_mask] = 7
tool_map[dist_edge < 0] = 0

# Colors for 3D Painting:
rgb = np.zeros((ny, nx, 3), dtype=np.float32)
# Sunlit Granite (warm stone grey)
rgb[granite_mask] = [0.58, 0.60, 0.62]
# Dark Slate Cliffs (deep charcoal basalt)
rgb[cliff_mask] = [0.24, 0.26, 0.28]
# Glacial Ice (cyan firn blue)
rgb[ice_mask] = [0.46, 0.74, 0.86]
# Alpine Snow (brilliant crisp white)
rgb[snow_mask] = [0.96, 0.97, 0.99]
# Alpine Tundra (fresh moss lichen)
rgb[tundra_mask] = [0.30, 0.50, 0.26]
# Road Shelf / Zócalo Base
rgb[shelf_mask] = [0.18, 0.20, 0.23]
rgb[dist_edge < 0] = [0.06, 0.08, 0.12]

# Directional lighting blend
lit_rgb = rgb * (0.35 + 0.65 * hs[:, :, None])
lit_rgb[dist_edge < 0] = [0.06, 0.08, 0.12]
lit_rgb = np.clip(lit_rgb, 0.0, 1.0)

fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor='#090d16')
axes[0].imshow(lit_rgb, origin='lower')
axes[0].set_title('Pintura 3D CMYKWOG: Macizo Alpino Tricúspide (Sin Vetas Rojas)', color='white', fontsize=12)
axes[0].axis('off')

axes[1].imshow(hs, cmap='gray', origin='lower')
axes[1].set_title('Sombreado Continuo de Aristas Vivas (Zero Píxeles)', color='white', fontsize=12)
axes[1].axis('off')

mid_y = ny // 2
axes[2].plot(xs, Z_top[mid_y, :], color='#38bdf8', lw=2.5, label='Perfil E-W Transversal')
axes[2].axhline(25.0, color='#94a3b8', linestyle='--', label='Zócalo Base (25mm)')
axes[2].set_ylim(20, 90)
axes[2].set_title('Perfil Transversal (Pendiente Piramidal Cóncava Natural)', color='white', fontsize=12)
axes[2].set_facecolor('#0f172a')
axes[2].tick_params(colors='white')
axes[2].legend(facecolor='#1e293b', labelcolor='white')
axes[2].grid(True, color='#334155', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('scratch/master_sculpture_preview.png', dpi=200, facecolor='#090d16')
print("Saved scratch/master_sculpture_preview.png")

# Save master arrays to scratch
np.save('scratch/Z_top_master_sculpt.npy', Z_top)
np.save('scratch/tool_map_master_sculpt.npy', tool_map)
print("Saved scratch/Z_top_master_sculpt.npy and tool_map_master_sculpt.npy")
