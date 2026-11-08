import math
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

print("=== PROTOTYPING 3D MOUNTAIN PAINTING (PURE CONTINUOUS ALPINISM) ===")

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF_H = Z_PEAK - Z_BASE # 58.0 mm

# Grid resolution: 351 x 405 (aspect ratio 0.866, ~0.31 mm vertex spacing)
nx, ny = 351, 405
xs = np.linspace(-APOTHEM, APOTHEM, nx)
ys = np.linspace(-R, R, ny)
X, Y = np.meshgrid(xs, ys)

normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

v_angles = [(30 + 60 * k) * math.pi / 180 for k in range(6)]
hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles])
dist_vert = np.full_like(X, 999.0)
for vx, vy in hex_corners:
    dist_vert = np.minimum(dist_vert, np.hypot(X - vx, Y - vy))

# ---------------------------------------------------------
# 1. ARTISTIC MACRO-ANATOMY: The Alpine Horn & Massif
# ---------------------------------------------------------
# Summit horn slightly off-center (artistic rule of thirds)
cx, cy = -2.5, 4.0
r = np.hypot(X - cx, Y - cy)

# Continuous Concave Mountain Slope:
# Power 1.65 ensures soaring, steep pyramid pinnacle that smoothly
# widens into a broad mountain base, asymptotically reaching 0 at R=52mm
R_mountain = 52.0
r_norm = np.clip(r / R_mountain, 0.0, 1.0)
macro_slope = (1.0 - r_norm)**1.65

# ---------------------------------------------------------
# 2. KNIFE-EDGE ARÊTES (The Mountain's Backbone)
# ---------------------------------------------------------
# 4 asymmetric, sweeping arêtes radiating outward:
# Arête 1: Heading North-West (Zmutt/Hörnli ridge style)
# Arête 2: Heading North-East towards East Spire
# Arête 3: Heading South-East (Furggen ridge)
# Arête 4: Heading South-West (Lion ridge)

spines = [
    # (x1, y1, x2, y2, h_start, h_end, width, curve)
    (-2.5, 4.0, -30.0, 34.0, 1.0, 0.15, 6.0, 0.9),   # NW Arête
    (-2.5, 4.0, 16.0, 14.0, 1.0, 0.72, 5.0, 1.0),    # Col to East Spire
    (16.0, 14.0, 36.0, 22.0, 0.72, 0.12, 6.5, 0.85), # East Spire descent
    (16.0, 14.0, 28.0, -10.0, 0.72, 0.14, 6.0, 0.85),# East Spur
    (-2.5, 4.0, -12.0, -16.0, 1.0, 0.65, 5.5, 1.0),  # Col to South Peak
    (-12.0, -16.0, -28.0, -32.0, 0.65, 0.10, 7.0, 0.8), # SW descent
    (-12.0, -16.0, 14.0, -24.0, 0.65, 0.15, 6.5, 0.85), # SE ridge
    (-2.5, 4.0, -24.0, 6.0, 1.0, 0.40, 5.8, 0.95),   # West Buttress
    (-24.0, 6.0, -38.0, -8.0, 0.40, 0.10, 7.0, 0.8)  # West spur
]

H_spines = np.zeros_like(X)
for x1, y1, x2, y2, h1, h2, w, curve in spines:
    dx = x2 - x1; dy = y2 - y1
    L2 = dx*dx + dy*dy
    t = np.clip(((X - x1)*dx + (Y - y1)*dy) / L2, 0.0, 1.0)
    px = x1 + t * dx; py = y1 + t * dy
    dist_sp = np.hypot(X - px, Y - py)
    h_line = (1.0 - t) * h1 + t * h2
    h_ridge = h_line * np.exp(-(dist_sp / w)**curve)
    H_spines = np.maximum(H_spines, h_ridge)

# Pyramidal Horn Summit (Matterhorn facet modulation)
theta = np.arctan2(Y - cy, X - cx)
# 4 main rock faces (North, East, South, West faces)
facets = 0.80 + 0.22 * np.cos(2.0 * theta + 0.5)**2.0
horn_peak = np.exp(-(r / 9.5)**1.5) * facets

# Secondary Peaks:
# East Spire (Grand Teton needle)
r_p2 = np.hypot(X - 16.0, Y - 14.0)
spire_east = 0.75 * np.exp(-(r_p2 / 7.5)**1.5)

# South Peak
r_p3 = np.hypot(X - (-12.0), Y - (-16.0))
spire_south = 0.68 * np.exp(-(r_p3 / 8.0)**1.45)

# West Shoulder
r_p4 = np.hypot(X - (-24.0), Y - 6.0)
spire_west = 0.52 * np.exp(-(r_p4 / 7.0)**1.4)

# Combine macro shapes
macro_relief = np.maximum(macro_slope * 0.70, H_spines)
macro_relief = np.maximum(macro_relief, horn_peak)
macro_relief = np.maximum(macro_relief, spire_east)
macro_relief = np.maximum(macro_relief, spire_south)
macro_relief = np.maximum(macro_relief, spire_west)

# ---------------------------------------------------------
# 3. CONTINUOUS FRACTAL ROCK MICRO-FACETING (NO PIXELS)
# ---------------------------------------------------------
# Multi-octave continuous noise using irrational rotations
np.random.seed(42)
rock_noise = np.zeros_like(X)
harmonics = [
    (0.09, 3.8, 0.41),
    (0.20, 2.0, 1.25),
    (0.44, 1.0, 2.18),
    (0.95, 0.45, 0.76),
    (2.10, 0.20, 1.83),
    (4.50, 0.08, 2.92)
]

for freq, amp, rot in harmonics:
    xr = X * math.cos(rot) - Y * math.sin(rot)
    yr = X * math.sin(rot) + Y * math.cos(rot)
    # Sharp angular rock joints
    w1 = np.sin(xr * freq + 0.3) * np.cos(yr * freq * 1.08 + 0.7)
    w2 = 1.0 - np.abs(np.cos(xr * freq * 0.95 - 0.4))
    rock_noise += amp * (w1 * 0.55 + w2 * 0.45 - 0.45)

# Natural Metamorphic Rock Bedding (tilted strata at 20 degrees)
strata = 0.8 * np.sin((X * 0.28 + Y * 0.16 + macro_relief * 18.0) * 1.4)

# Rock micro-detail scales with height (craggy peaks, smoother scree talus)
craggy_relief = macro_relief + (rock_noise + strata) * (macro_relief**0.7) * 0.038
craggy_relief = np.maximum(0.0, craggy_relief)
craggy_relief = craggy_relief / craggy_relief.max()

# ---------------------------------------------------------
# 4. ROAD SHELF & VERTEX PAD CLEARANCE
# ---------------------------------------------------------
def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# 3.5mm road shelf: gentle talus apron landing from 3.5mm to 14.0mm
edge_factor = smoothstep(3.5, 14.0, dist_edge)
# 20mm vertex pads: gentle talus apron landing from 10.0mm to 16.0mm
vert_factor = smoothstep(10.0, 16.0, dist_vert)

clearance = edge_factor * vert_factor

# Scaled relief strictly in [0.0, 58.0] mm
relief_mm = craggy_relief * clearance * RELIEF_H
Z_master = Z_BASE + relief_mm
Z_master[dist_edge < 0] = Z_BASE

print(f"Topography sculpted: Z min = {Z_master.min():.2f} mm, max = {Z_master.max():.2f} mm")

# ---------------------------------------------------------
# 5. FULL-COLOR 3D PAINTING SHADER (CMYKWOG-B)
# ---------------------------------------------------------
# Derivatives & lighting
dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
gy, gx = np.gradient(Z_master, dy, dx)
slope = np.hypot(gx, gy)
slope_deg = np.degrees(np.arctan(slope))
laplace = ndi.laplace(Z_master)

# Directional lighting: Sun from NW (315 deg azimuth, 48 deg altitude)
az = 315.0 * math.pi / 180.0; alt = 48.0 * math.pi / 180.0
sun_x = math.cos(alt)*math.sin(az); sun_y = math.cos(alt)*math.cos(az); sun_z = math.sin(alt)
norm = np.sqrt(gx**2 + gy**2 + 1.0)
hs = np.clip((-gx/norm)*sun_x + (-gy/norm)*sun_y + (1.0/norm)*sun_z, 0.0, 1.0)

# Natural Shading & Geological Layer Allocation:
# 1. Snow: High elevations (> 52mm), flatter slopes (< 46 deg), plus couloirs
snow_score = (relief_mm / 58.0)**1.4 * 1.6 - (slope_deg / 45.0)**1.1 * 0.8 + np.clip(-laplace * 0.35, -0.2, 0.4)
snow_mask = (snow_score > 0.40) & (relief_mm > 24.0)

# 2. Glacial Blue Ice: deep shaded cirques & couloir hollows
ice_score = np.clip(-laplace * 0.5, 0.0, 1.0) * np.clip((relief_mm - 26.0) / 20.0, 0, 1) * (1.0 - np.clip(snow_score, 0, 1))
ice_mask = (ice_score > 0.28) & (relief_mm > 22.0) & (~snow_mask)

# 3. Alpine Tundra: low depressions & scree transitions (Z 25.0 to 28.5mm)
tundra_score = np.clip((6.0 - relief_mm) / 5.0, 0, 1) * np.clip((35.0 - slope_deg) / 20.0, 0, 1) * (dist_edge > 3.0)
tundra_mask = (tundra_score > 0.38) & (relief_mm > 0.2)

# 4. Dark Slate Shadow Cliffs: steep rock faces (> 48 deg)
cliff_mask = (slope_deg > 46.0) & (relief_mm > 2.0) & (~snow_mask) & (~ice_mask)

# 5. Granite Rock Body: everywhere else on the mountain
granite_mask = (relief_mm > 0.1) & (~snow_mask) & (~ice_mask) & (~tundra_mask) & (~cliff_mask)

# 6. Base Shelf: relief < 0.1mm
shelf_mask = (relief_mm <= 0.1) & (dist_edge >= 0)

# Tool Map for Multi-Material 3MF:
# 0 = Base Zócalo (Tool B)
# 1 = Glacial Ice (Tool C)
# 3 = Dark Slate Cliffs (Tool K)
# 4 = Granite Rock Face (Tool W / Grey)
# 5 = Alpine Tundra (Tool G / Green)
# 7 = Alpine Snow (Tool W / White 0.2mm)
tool_map = np.zeros_like(Z_master, dtype=np.uint8)
tool_map[shelf_mask] = 0
tool_map[tundra_mask] = 5
tool_map[granite_mask] = 4
tool_map[cliff_mask] = 3
tool_map[ice_mask] = 1
tool_map[snow_mask] = 7
tool_map[dist_edge < 0] = 0

# Colors for 3D Painting:
rgb = np.zeros((ny, nx, 3), dtype=np.float32)
# Sunlit Granite (warm cool grey)
rgb[granite_mask] = [0.56, 0.59, 0.62]
# Dark Slate Cliffs (deep charcoal shadow)
rgb[cliff_mask] = [0.24, 0.26, 0.29]
# Glacial Ice (soft cyan blue)
rgb[ice_mask] = [0.45, 0.72, 0.85]
# Alpine Snow (brilliant white)
rgb[snow_mask] = [0.96, 0.97, 0.98]
# Alpine Tundra (fresh moss green)
rgb[tundra_mask] = [0.28, 0.48, 0.25]
# Road Shelf / Zócalo Base
rgb[shelf_mask] = [0.18, 0.20, 0.24]
rgb[dist_edge < 0] = [0.06, 0.08, 0.12]

# Directional lighting blend
lit_rgb = rgb * (0.35 + 0.65 * hs[:, :, None])
lit_rgb[dist_edge < 0] = [0.06, 0.08, 0.12]
lit_rgb = np.clip(lit_rgb, 0.0, 1.0)

fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor='#090d16')
axes[0].imshow(lit_rgb, origin='lower')
axes[0].set_title('Pintura 3D CMYKWOG: Macizo Alpino Sin Vetas Artificiales', color='white', fontsize=12)
axes[0].axis('off')

axes[1].imshow(hs, cmap='gray', origin='lower')
axes[1].set_title('Sombreado Continuo de Aristas y Circos (Zero Píxeles)', color='white', fontsize=12)
axes[1].axis('off')

mid_y = ny // 2
axes[2].plot(xs, Z_master[mid_y, :], color='#38bdf8', lw=2.5, label='Perfil E-W Transversal')
axes[2].axhline(25.0, color='#94a3b8', linestyle='--', label='Zócalo Base (25mm)')
axes[2].set_ylim(20, 90)
axes[2].set_title('Perfil Transversal (Pendiente Piramidal Natural Suave)', color='white', fontsize=12)
axes[2].set_facecolor('#0f172a')
axes[2].tick_params(colors='white')
axes[2].legend(facecolor='#1e293b', labelcolor='white')
axes[2].grid(True, color='#334155', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('scratch/mountain_painting_preview.png', dpi=200, facecolor='#090d16')
print("Saved scratch/mountain_painting_preview.png")

# Save master arrays to scratch
np.save('scratch/Z_top_sculpted.npy', Z_master)
np.save('scratch/tool_map_sculpted.npy', tool_map)
print("Saved scratch/Z_top_sculpted.npy and tool_map_sculpted.npy")
