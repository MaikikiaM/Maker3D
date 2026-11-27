import math
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

print("=== GENERATING MASTER ALPINISM 3D MODEL (NO PIXELS, NO PANETTONE, NO RED VEINS) ===")

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF_H = Z_PEAK - Z_BASE # 58.0 mm

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

# 1. Knife-Edge Arêtes with Organic Meandering
spines = [
    # NW Ridge (Main summit to NW corner)
    {"x1": -4.0, "y1": 6.0, "x2": -34.0, "y2": 36.0, "h1": 58.0, "h2": 14.0, "sharp": 9.5, "curv": 1.15, "meander": 1.6},
    # East Ridge (Connecting Main peak to East Peak)
    {"x1": -4.0, "y1": 6.0, "x2": 18.0, "y2": 16.0, "h1": 58.0, "h2": 50.0, "sharp": 8.5, "curv": 1.1, "meander": 1.2},
    # East Peak outward spurs
    {"x1": 18.0, "y1": 16.0, "x2": 38.0, "y2": 26.0, "h1": 50.0, "h2": 12.0, "sharp": 9.0, "curv": 1.15, "meander": 1.4},
    {"x1": 18.0, "y1": 16.0, "x2": 30.0, "y2": -8.0, "h1": 50.0, "h2": 20.0, "sharp": 8.8, "curv": 1.15, "meander": 1.3},
    # South Ridge (Connecting Main peak to South Peak)
    {"x1": -4.0, "y1": 6.0, "x2": -12.0, "y2": -18.0, "h1": 58.0, "h2": 44.0, "sharp": 9.0, "curv": 1.1, "meander": 1.4},
    # South Peak outward spurs
    {"x1": -12.0, "y1": -18.0, "x2": -32.0, "y2": -28.0, "h1": 44.0, "h2": 10.0, "sharp": 9.5, "curv": 1.15, "meander": 1.5},
    {"x1": -12.0, "y1": -18.0, "x2": 16.0, "y2": -26.0, "h1": 44.0, "h2": 14.0, "sharp": 9.2, "curv": 1.15, "meander": 1.4},
    # West Buttress
    {"x1": -4.0, "y1": 6.0, "x2": -26.0, "y2": -4.0, "h1": 58.0, "h2": 26.0, "sharp": 9.0, "curv": 1.1, "meander": 1.3},
    {"x1": -26.0, "y1": -4.0, "x2": -42.0, "y2": -14.0, "h1": 26.0, "h2": 8.0, "sharp": 10.0, "curv": 1.15, "meander": 1.4}
]

H_spines = np.zeros_like(X)
for sp in spines:
    dx = sp["x2"] - sp["x1"]; dy = sp["y2"] - sp["y1"]
    L2 = dx*dx + dy*dy
    t = np.clip(((X - sp["x1"])*dx + (Y - sp["y1"])*dy) / L2, 0.0, 1.0)
    perp_x = -dy / math.sqrt(L2)
    perp_y = dx / math.sqrt(L2)
    meander = sp["meander"] * np.sin(t * 10.0 + 1.2)
    px = sp["x1"] + t * dx + meander * perp_x
    py = sp["y1"] + t * dy + meander * perp_y
    dist_sp = np.hypot(X - px, Y - py)
    h_line = (1.0 - t) * sp["h1"] + t * sp["h2"]
    h_contrib = h_line * np.exp(-(dist_sp / sp["sharp"])**sp["curv"])
    H_spines = np.maximum(H_spines, h_contrib)

# 2. Pyramidal Horns & Peaks (Substantial Alpine Volume)
r_main = np.hypot(X - (-4.0), Y - 6.0)
ang_main = np.arctan2(Y - 6.0, X - (-4.0))
facet_horn = 0.85 + 0.35 * np.abs(np.cos(1.5 * ang_main + 0.4))**1.4
h_horn = 58.0 * np.exp(-(r_main / 24.0)**1.35) * facet_horn

r_p2 = np.hypot(X - 18.0, Y - 16.0)
ang_p2 = np.arctan2(Y - 16.0, X - 18.0)
h_p2 = 50.0 * np.exp(-(r_p2 / 18.0)**1.35) * (0.85 + 0.30 * np.abs(np.cos(2.0 * ang_p2))**1.2)

r_p3 = np.hypot(X - (-12.0), Y - (-18.0))
h_p3 = 44.0 * np.exp(-(r_p3 / 19.0)**1.35) * (0.88 + 0.25 * np.cos(1.8 * ang_main)**2.0)

r_p4 = np.hypot(X - (-26.0), Y - (-4.0))
h_p4 = 30.0 * np.exp(-(r_p4 / 16.0)**1.35)

massif_core = np.maximum(H_spines, np.maximum(h_horn, np.maximum(h_p2, np.maximum(h_p3, h_p4))))

# Broad Concave Talus Apron
r_tile = np.hypot(X, Y)
base_apron = 58.0 * np.clip(1.0 - (r_tile / 54.0), 0.0, 1.0)**1.65
mountain_sculpture = np.maximum(massif_core, base_apron * 0.40)

# 3. Domain-Warped Organic Rock Texture (NO Grid Cross-Hatch)
np.random.seed(42)
warp_x = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=10.0) * 12.0
warp_y = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=10.0) * 12.0

organic_noise = np.zeros_like(X)
for freq, amp in [(0.10, 3.2), (0.22, 1.8), (0.50, 0.9), (1.1, 0.4), (2.4, 0.18)]:
    n_gauss = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=1.0 / freq)
    organic_noise += amp * n_gauss

# Tilted metamorphic rock strata (smoothly modulated)
strata = 0.75 * np.sin((X * 0.22 + Y * 0.12 + mountain_sculpture * 0.35) * 1.4)

relief_textured = mountain_sculpture + (organic_noise + strata) * (mountain_sculpture / 58.0)**0.75
relief_textured = np.maximum(0.0, relief_textured)
relief_textured = relief_textured * (RELIEF_H / np.max(relief_textured))

# 4. Road Shelf & Vertex Clearances
def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

edge_factor = smoothstep(3.5, 12.0, dist_edge)
vert_factor = smoothstep(10.0, 15.0, dist_vert)
clearance = edge_factor * vert_factor

final_relief = relief_textured * clearance
Z_top = Z_BASE + final_relief
Z_top[dist_edge < 0] = Z_BASE

# Slope & Normal derivatives
dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
gy, gx = np.gradient(Z_top, dy, dx)
slope = np.hypot(gx, gy)
slope_deg = np.degrees(np.arctan(slope))
laplace = ndi.laplace(Z_top)

az = 315.0 * math.pi / 180.0; alt = 48.0 * math.pi / 180.0
sun_x = math.cos(alt)*math.sin(az); sun_y = math.cos(alt)*math.cos(az); sun_z = math.sin(alt)
norm = np.sqrt(gx**2 + gy**2 + 1.0)
hs = np.clip((-gx/norm)*sun_x + (-gy/norm)*sun_y + (1.0/norm)*sun_z, 0.0, 1.0)

# 5. Master 3D Painting (CMYKWOG-B: ZERO Red Veins)
# Snow Mantle (White T7, 0.2mm nozzle): High elevations, couloirs
summit_snow = smoothstep(20.0, 35.0, final_relief) * (1.0 - smoothstep(52.0, 70.0, slope_deg) * 0.5)
couloir_snow = smoothstep(12.0, 26.0, final_relief) * np.clip(-laplace * 0.7, 0.0, 1.0) * (1.0 - smoothstep(42.0, 58.0, slope_deg) * 0.5)
snow_prob = np.maximum(summit_snow, couloir_snow)
snow_mask = (snow_prob > 0.35) & (final_relief > 12.0)

# Glacial Ice (Cyan T1)
ice_prob = smoothstep(14.0, 30.0, final_relief) * np.clip(-laplace * 0.65, 0.0, 1.0) * (1.0 - snow_prob * 0.7)
ice_mask = (ice_prob > 0.22) & (final_relief > 12.0) & (~snow_mask)

# Alpine Tundra (Green T5)
np.random.seed(888)
tundra_noise = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=3.5)
tundra_prob = smoothstep(6.5, 0.5, final_relief) * (1.0 - smoothstep(16.0, 28.0, slope_deg)) * (tundra_noise > 0.1)
tundra_mask = (tundra_prob > 0.30) & (final_relief > 0.1) & (dist_edge > 3.2) & (~snow_mask)

# Dark Slate Shadow Cliffs (Black T3)
cliff_mask = (slope_deg > 40.0) & (final_relief > 1.5) & (~snow_mask) & (~ice_mask)

# Sunlit Granite Rock Face (Grey T4)
granite_mask = (final_relief > 0.05) & (~snow_mask) & (~ice_mask) & (~tundra_mask) & (~cliff_mask)

# Road Shelf & Vertex Pads (Tool B - Black)
shelf_mask = (final_relief <= 0.05) & (dist_edge >= 0)

# Tool Map Assignment for 3MF
tool_map = np.zeros_like(Z_top, dtype=np.uint8)
tool_map[shelf_mask] = 0
tool_map[tundra_mask] = 5
tool_map[granite_mask] = 4
tool_map[cliff_mask] = 3
tool_map[ice_mask] = 1
tool_map[snow_mask] = 7
tool_map[dist_edge < 0] = 0

# Colors for 3D Painting
rgb = np.zeros((ny, nx, 3), dtype=np.float32)
rgb[granite_mask] = [0.58, 0.60, 0.63]
rgb[cliff_mask] = [0.24, 0.26, 0.28]
rgb[ice_mask] = [0.44, 0.72, 0.86]
rgb[snow_mask] = [0.96, 0.97, 0.99]
rgb[tundra_mask] = [0.30, 0.52, 0.26]
rgb[shelf_mask] = [0.18, 0.20, 0.23]
rgb[dist_edge < 0] = [0.06, 0.08, 0.12]

lit_rgb = rgb * (0.35 + 0.65 * hs[:, :, None])
lit_rgb[dist_edge < 0] = [0.06, 0.08, 0.12]
lit_rgb = np.clip(lit_rgb, 0.0, 1.0)

fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor='#090d16')
axes[0].imshow(lit_rgb, origin='lower')
axes[0].set_title('Pintura 3D CMYKWOG: Cima Nevada Alpina Integral (Sin Vetas Rojas)', color='white', fontsize=12)
axes[0].axis('off')

axes[1].imshow(hs, cmap='gray', origin='lower')
axes[1].set_title('Sombreado Continuo Sin Cuadrícula (Zero Píxeles)', color='white', fontsize=12)
axes[1].axis('off')

mid_y = ny // 2
axes[2].plot(xs, Z_top[mid_y, :], color='#38bdf8', lw=2.5, label='Perfil E-W Transversal')
axes[2].axhline(25.0, color='#94a3b8', linestyle='--', label='Zócalo Base (25mm)')
axes[2].set_ylim(20, 90)
axes[2].set_title('Perfil Transversal (Volumen Piramidal Alpino Real)', color='white', fontsize=12)
axes[2].set_facecolor('#0f172a')
axes[2].tick_params(colors='white')
axes[2].legend(facecolor='#1e293b', labelcolor='white')
axes[2].grid(True, color='#334155', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('scratch/master_alpinism_preview.png', dpi=200, facecolor='#090d16')
print("Saved scratch/master_alpinism_preview.png")

# Save master arrays to scratch
np.save('scratch/Z_top_master_final.npy', Z_top)
np.save('scratch/tool_map_master_final.npy', tool_map)
print("Saved master final arrays to scratch!")
