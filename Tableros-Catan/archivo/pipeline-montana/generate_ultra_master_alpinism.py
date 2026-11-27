import math
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

print("=== GENERATING ULTRA-RESOLUTION MASTER ALPINISM 3D MODEL ===")

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256258 mm
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF_H = Z_PEAK - Z_BASE # 58.00 mm

# High-Density Sub-Micron Grid: 451 x 521 (dx = 0.246mm, dy = 0.246mm)
nx, ny = 451, 521
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

# -------------------------------------------------------------
# 1. ALPINE ARÊTES & SPINES WITH NATURAL MEANDER & SADDLES
# -------------------------------------------------------------
spines = [
    # NW Ridge (Main summit to NW corner)
    {"x1": -4.0, "y1": 6.0, "x2": -35.0, "y2": 35.0, "h1": 58.0, "h2": 12.0, "sharp": 8.5, "curv": 1.20, "meander": 1.5, "freq": 8.0},
    # East Ridge (Connecting Main peak to East Cathedral Spire)
    {"x1": -4.0, "y1": 6.0, "x2": 18.0, "y2": 16.0, "h1": 58.0, "h2": 50.0, "sharp": 8.0, "curv": 1.15, "meander": 1.2, "freq": 6.0},
    # East Cathedral outward spurs
    {"x1": 18.0, "y1": 16.0, "x2": 38.0, "y2": 26.0, "h1": 50.0, "h2": 10.0, "sharp": 8.5, "curv": 1.20, "meander": 1.4, "freq": 7.0},
    {"x1": 18.0, "y1": 16.0, "x2": 32.0, "y2": -8.0, "h1": 50.0, "h2": 18.0, "sharp": 8.2, "curv": 1.18, "meander": 1.3, "freq": 7.5},
    # South Ridge (Connecting Main peak to South Peak)
    {"x1": -4.0, "y1": 6.0, "x2": -12.0, "y2": -18.0, "h1": 58.0, "h2": 44.0, "sharp": 8.5, "curv": 1.15, "meander": 1.3, "freq": 7.0},
    # South Peak outward spurs
    {"x1": -12.0, "y1": -18.0, "x2": -32.0, "y2": -28.0, "h1": 44.0, "h2": 8.0, "sharp": 9.0, "curv": 1.20, "meander": 1.4, "freq": 8.0},
    {"x1": -12.0, "y1": -18.0, "x2": 16.0, "y2": -26.0, "h1": 44.0, "h2": 12.0, "sharp": 8.8, "curv": 1.18, "meander": 1.3, "freq": 7.5},
    # West Buttress
    {"x1": -4.0, "y1": 6.0, "x2": -26.0, "y2": -4.0, "h1": 58.0, "h2": 28.0, "sharp": 8.5, "curv": 1.15, "meander": 1.2, "freq": 6.5},
    {"x1": -26.0, "y1": -4.0, "x2": -42.0, "y2": -14.0, "h1": 28.0, "h2": 6.0, "sharp": 9.5, "curv": 1.20, "meander": 1.3, "freq": 8.0}
]

H_spines = np.zeros_like(X)
for sp in spines:
    dx = sp["x2"] - sp["x1"]; dy = sp["y2"] - sp["y1"]
    L2 = dx*dx + dy*dy
    t = np.clip(((X - sp["x1"])*dx + (Y - sp["y1"])*dy) / L2, 0.0, 1.0)
    perp_x = -dy / math.sqrt(L2)
    perp_y = dx / math.sqrt(L2)
    meander = sp["meander"] * np.sin(t * sp["freq"] + 1.2)
    px = sp["x1"] + t * dx + meander * perp_x
    py = sp["y1"] + t * dy + meander * perp_y
    dist_sp = np.hypot(X - px, Y - py)
    h_line = (1.0 - t) * sp["h1"] + t * sp["h2"]
    h_contrib = h_line * np.exp(-(dist_sp / sp["sharp"])**sp["curv"])
    H_spines = np.maximum(H_spines, h_contrib)

# -------------------------------------------------------------
# 2. PYRAMIDAL HORNS (MATTERHORN + SECONDARY SPIRES)
# -------------------------------------------------------------
# Main Apex: (-4.0, 6.0)
r_main = np.hypot(X - (-4.0), Y - 6.0)
ang_main = np.arctan2(Y - 6.0, X - (-4.0))
# 4 distinct alpine facets
facet_horn = 0.88 + 0.32 * np.abs(np.cos(2.0 * ang_main + 0.35))**1.4
h_horn = 58.0 * np.exp(-(r_main / 23.0)**1.38) * facet_horn

# East Peak: (18.0, 16.0)
r_p2 = np.hypot(X - 18.0, Y - 16.0)
ang_p2 = np.arctan2(Y - 16.0, X - 18.0)
h_p2 = 50.0 * np.exp(-(r_p2 / 17.5)**1.36) * (0.86 + 0.28 * np.abs(np.cos(2.0 * ang_p2))**1.2)

# South Peak: (-12.0, -18.0)
r_p3 = np.hypot(X - (-12.0), Y - (-18.0))
h_p3 = 44.0 * np.exp(-(r_p3 / 18.5)**1.36) * (0.88 + 0.24 * np.cos(1.8 * ang_main)**2.0)

# West Shoulder: (-26.0, -4.0)
r_p4 = np.hypot(X - (-26.0), Y - (-4.0))
h_p4 = 30.0 * np.exp(-(r_p4 / 15.5)**1.36)

massif_core = np.maximum(H_spines, np.maximum(h_horn, np.maximum(h_p2, np.maximum(h_p3, h_p4))))

# Broad Concave Talus Apron (Natural alpine decay across the hex)
r_tile = np.hypot(X, Y)
base_apron = 58.0 * np.clip(1.0 - (r_tile / 54.0), 0.0, 1.0)**1.65
mountain_sculpture = np.maximum(massif_core, base_apron * 0.40)

# -------------------------------------------------------------
# 3. DOMAIN-WARPED ORGANIC ROCK TEXTURE & COULOIR FLUTING
# -------------------------------------------------------------
np.random.seed(101)
# Domain warp flow field
warp_x = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=12.0) * 14.0
warp_y = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=12.0) * 14.0

# Multi-octave natural rock fractal noise
organic_noise = np.zeros_like(X)
for freq, amp in [(0.08, 3.4), (0.20, 1.9), (0.45, 0.95), (1.0, 0.42), (2.2, 0.18)]:
    n_gauss = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=1.0 / freq)
    organic_noise += amp * n_gauss

# Alpine fluting: vertical ravines and erosion couloirs running downhill
fluting = 0.85 * np.sin((X * 0.32 + Y * 0.16 + warp_x * 0.1) * 1.5)

# Metamorphic folded geological strata
strata = 0.65 * np.cos((X * 0.18 + Y * 0.28 + mountain_sculpture * 0.25) * 1.6)

# Scale texture so it is crisp on the peaks and gentle on the apron
texture_envelope = (mountain_sculpture / 58.0)**0.80
relief_textured = mountain_sculpture + (organic_noise + fluting + strata) * texture_envelope
relief_textured = np.maximum(0.0, relief_textured)

# Enforce exact peak height scaling
relief_textured = relief_textured * (RELIEF_H / np.max(relief_textured))

# -------------------------------------------------------------
# 4. ROAD SHELF (3.5mm) & VERTEX SETTLEMENT PADS (Ø 20mm)
# -------------------------------------------------------------
def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

edge_factor = smoothstep(3.5, 12.0, dist_edge)
vert_factor = smoothstep(10.0, 15.0, dist_vert)
clearance = edge_factor * vert_factor

final_relief = relief_textured * clearance
final_relief = final_relief * (RELIEF_H / np.max(final_relief))

Z_top = Z_BASE + final_relief
Z_top[dist_edge < 0] = Z_BASE

# Pin the exact summit apex to 83.00 mm
ix_peak = int(np.round((-4.0 - (-APOTHEM)) / (2 * APOTHEM) * (nx - 1)))
iy_peak = int(np.round((6.0 - (-R)) / (2 * R) * (ny - 1)))
Z_top[iy_peak, ix_peak] = Z_PEAK

print(f"Topography generated: nx={nx}, ny={ny}, Z in [{np.min(Z_top):.2f}, {np.max(Z_top):.2f}] mm")

# -------------------------------------------------------------
# 5. GEOLOGICAL 3D PAINTING (CMYKWOG-B: ZERO RED VEINS)
# -------------------------------------------------------------
dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
gy, gx = np.gradient(Z_top, dy, dx)
slope = np.hypot(gx, gy)
slope_deg = np.degrees(np.arctan(slope))
laplace = ndi.laplace(Z_top)

# Sun hillshade for preview and artistic contrast
az = 315.0 * math.pi / 180.0; alt = 48.0 * math.pi / 180.0
sun_x = math.cos(alt)*math.sin(az); sun_y = math.cos(alt)*math.cos(az); sun_z = math.sin(alt)
norm = np.sqrt(gx**2 + gy**2 + 1.0)
hs = np.clip((-gx/norm)*sun_x + (-gy/norm)*sun_y + (1.0/norm)*sun_z, 0.0, 1.0)

# A. Alpine Snow (Tool W, 0.2mm Nozzle): High peaks, ridges, couloirs
summit_snow = smoothstep(22.0, 36.0, final_relief) * (1.0 - smoothstep(54.0, 72.0, slope_deg) * 0.5)
# Couloirs: where laplace < 0 (valleys/concavities)
couloir_snow = smoothstep(12.0, 28.0, final_relief) * np.clip(-laplace * 0.75, 0.0, 1.0) * (1.0 - smoothstep(42.0, 60.0, slope_deg) * 0.5)
snow_prob = np.maximum(summit_snow, couloir_snow)
snow_mask = (snow_prob > 0.35) & (final_relief > 12.0)

# B. Glacial Ice (Tool C - Cian): Shaded high cirques
ice_prob = smoothstep(14.0, 32.0, final_relief) * np.clip(-laplace * 0.70, 0.0, 1.0) * (1.0 - snow_prob * 0.75)
ice_mask = (ice_prob > 0.24) & (final_relief > 12.0) & (~snow_mask)

# C. Alpine Tundra (Tool G - Forest Green): Low valley landings
tundra_noise = ndi.gaussian_filter(np.random.randn(ny, nx), sigma=4.0)
tundra_prob = smoothstep(7.0, 0.5, final_relief) * (1.0 - smoothstep(18.0, 30.0, slope_deg)) * (tundra_noise > 0.08)
tundra_mask = (tundra_prob > 0.32) & (final_relief > 0.1) & (dist_edge > 3.2) & (~snow_mask)

# D. Deep Slate Shadow Cliffs (Tool K - Charcoal/Black): Sheer crags
cliff_mask = (slope_deg > 42.0) & (final_relief > 1.5) & (~snow_mask) & (~ice_mask)

# E. Granite Bedrock (Tool W/Gris): Main sunlit rock face
granite_mask = (final_relief > 0.05) & (~snow_mask) & (~ice_mask) & (~tundra_mask) & (~cliff_mask)

# F. Foundation Zócalo, Road Shelf & Vertex Pads (Tool B - Black)
shelf_mask = (final_relief <= 0.05) & (dist_edge >= 0)

# Assembly Tool Map
tool_map = np.zeros_like(Z_top, dtype=np.uint8)
tool_map[shelf_mask] = 0
tool_map[tundra_mask] = 5
tool_map[granite_mask] = 4
tool_map[cliff_mask] = 3
tool_map[ice_mask] = 1
tool_map[snow_mask] = 7
tool_map[dist_edge < 0] = 0

print("Tool Map Distribution (Inside Hexagon):")
in_hex = dist_edge >= 0
for tid, name in [(0, "Tool B (Base/Roads)"), (1, "Tool C (Glacial Ice)"), (3, "Tool K (Slate Cliffs)"),
                  (4, "Tool W/Gris (Granite)"), (5, "Tool G (Alpine Tundra)"), (7, "Tool W (Alpine Snow)")]:
    count = np.sum((tool_map == tid) & in_hex)
    pct = 100.0 * count / np.sum(in_hex)
    print(f"  {name:26s}: {count:7d} px ({pct:5.1f}%)")

# Save master arrays to scratch
np.save('scratch/Z_top_master_final.npy', Z_top)
np.save('scratch/tool_map_master_final.npy', tool_map)
print("Saved Z_top_master_final.npy and tool_map_master_final.npy to scratch!")
