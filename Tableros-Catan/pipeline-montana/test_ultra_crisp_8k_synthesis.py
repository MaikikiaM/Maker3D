import math
import time
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

print("=== TESTING 3D PROCEDURAL CRISP ROCK SYNTHESIS (NO 2000s BLUR, NO UV STRETCH) ===")
t0 = time.time()

# Test on 1024x1024 first to inspect quality and sharpness
dim = 1024
R = 64.0
APOTHEM = R * math.cos(math.radians(30))

xs = np.linspace(-APOTHEM, APOTHEM, dim, dtype=np.float32)
ys = np.linspace(-R, R, dim, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)

Z_master = np.load('scratch/Z_top_master_final.npy')
import scipy.interpolate as interp
ny_m, nx_m = Z_master.shape
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

pts = np.column_stack((Y.ravel(), X.ravel()))
Z = interp_z(pts).reshape((dim, dim))
RELIEF = np.maximum(0.0, Z - 25.0)

dx = float(xs[1] - xs[0]); dy = float(ys[1] - ys[0])
gy, gx = np.gradient(Z, dy, dx)
slope = np.hypot(gx, gy)
slope_deg = np.degrees(np.arctan(slope))
laplace = ndi.laplace(Z)

# -------------------------------------------------------------
# 1. 3D WORLD-SPACE PROCEDURAL ROCK (NO VERTICAL STRETCHING!)
# -------------------------------------------------------------
# Real granite has 3D coordinates: (X, Y, Z)
# By using Z in the coordinates, vertical cliffs get natural horizontal ledges and strata!
# We use multi-octave 3D harmonic noise:
print("Evaluating 3D world-space multi-frequency rock texture...")

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# 3D coordinate space with geological dip angle (30 degrees tilt)
dip = math.radians(28.0)
strike = math.radians(45.0)
# Rotated coordinates for tilted geological bedding planes
X_dip = X * math.cos(strike) + Y * math.sin(strike)
Y_dip = -X * math.sin(strike) + Y * math.cos(strike)
Z_strata = Z * math.cos(dip) - X_dip * math.sin(dip)

# Sharp Strata (Ridge/Cosine noise with crisp transitions)
strata_freq = 0.85 # ~1.2mm layer thickness
strata_fine = np.cos(Z_strata * strata_freq * 2.0 * math.pi)
# Sharp fracture joint planes perpendicular to strata
joint_freq = 0.45
joint_planes = np.abs(np.sin(X_dip * joint_freq + Y_dip * 0.3))**0.5

# High-frequency 3D granite crystal noise (Mica, Quartz, Feldspar)
# Evaluated at pixel level using fast sine-hash lattice:
# hash(x, y, z) = sin(x*12.9898 + y*78.233 + z*37.719) * 43758.5453
p1 = (X * 2.1 + Y * 1.8 + Z * 2.4)
p2 = (X * 5.3 - Y * 4.7 + Z * 6.1)
p3 = (X * 12.7 + Y * 11.2 - Z * 13.5)
p4 = (X * 28.3 - Y * 25.1 + Z * 31.7)

crystal_fBm = (
    0.45 * np.sin(p1) * np.cos(p1 * 0.7 + 1.2) +
    0.28 * np.sin(p2) * np.cos(p2 * 0.8 - 0.5) +
    0.18 * np.sin(p3) * np.cos(p3 * 0.9 + 2.1) +
    0.09 * np.sin(p4)
)

# Chiseled rock relief: combines strata, joints, and crystal fBm
chiseled_rock = 0.40 * strata_fine + 0.35 * joint_planes + 0.25 * crystal_fBm
chiseled_rock = (chiseled_rock - np.min(chiseled_rock)) / (np.max(chiseled_rock) - np.min(chiseled_rock))

# Sharp rock color mapping:
# Deep slate in cracks: [0.16, 0.18, 0.21]
# Shadowed basalt: [0.26, 0.28, 0.32]
# Medium weathered granite: [0.52, 0.54, 0.57]
# Warm feldspar crystals: [0.66, 0.63, 0.58]
# Quartz highlights on edges: [0.78, 0.79, 0.82]

cliff_darkening = smoothstep(25.0, 60.0, slope_deg)
rock_t = np.clip(chiseled_rock * 0.85 + 0.15 * (1.0 - cliff_darkening), 0.0, 1.0)

rock_rgb = np.zeros((dim, dim, 3), dtype=np.float32)
c_slate = np.array([0.16, 0.18, 0.21], dtype=np.float32)
c_granite = np.array([0.52, 0.54, 0.57], dtype=np.float32)
c_quartz = np.array([0.76, 0.77, 0.80], dtype=np.float32)

for c in range(3):
    rock_rgb[:, :, c] = np.where(
        rock_t < 0.45,
        (1.0 - rock_t/0.45) * c_slate[c] + (rock_t/0.45) * c_granite[c],
        (1.0 - (rock_t-0.45)/0.55) * c_granite[c] + ((rock_t-0.45)/0.55) * c_quartz[c]
    )

# -------------------------------------------------------------
# 2. CRISP SNOW ACCUMULATION WITH MICRO-CREVICES
# -------------------------------------------------------------
# Snow settles on ledges (where strata_fine > 0.2 and slope < 55 deg)
snow_alt = smoothstep(16.0, 35.0, RELIEF)
snow_slope = 1.0 - smoothstep(38.0, 65.0, slope_deg)
# Couloirs: where laplace < 0
couloir_flow = smoothstep(-0.8, -0.05, laplace) * smoothstep(8.0, 28.0, RELIEF)
# Micro ledges catch snow:
snow_ledges = smoothstep(0.1, 0.6, strata_fine) * (1.0 - smoothstep(45.0, 70.0, slope_deg))

snow_raw = snow_alt * (0.35 + 0.45 * snow_slope + 0.20 * snow_ledges) + 0.40 * couloir_flow + 0.05 * crystal_fBm
snow_alpha = smoothstep(0.28, 0.65, snow_raw)

# Snow color with subtle sky blue tint
snow_rgb = np.array([0.96, 0.98, 1.00], dtype=np.float32)

# -------------------------------------------------------------
# 3. ORGANIC LICHEN & TUNDRA COLONIES
# -------------------------------------------------------------
tundra_alt = smoothstep(0.1, 1.2, RELIEF) * (1.0 - smoothstep(4.0, 10.0, RELIEF))
tundra_slope = 1.0 - smoothstep(12.0, 25.0, slope_deg)
# Voronoi-like colony patches
tundra_colony = smoothstep(0.25, 0.75, np.abs(np.sin(X * 0.45 + Y * 0.35)) * np.abs(np.cos(X * 0.35 - Y * 0.45)))
tundra_alpha = tundra_alt * tundra_slope * tundra_colony

moss_rgb = np.array([0.24, 0.42, 0.18], dtype=np.float32)
xanthoria_rgb = np.array([0.62, 0.54, 0.22], dtype=np.float32)

# Compose final color
final_rgb = rock_rgb.copy()
for c in range(3):
    final_rgb[:, :, c] = (1.0 - tundra_alpha) * final_rgb[:, :, c] + tundra_alpha * (0.7 * moss_rgb[c] + 0.3 * xanthoria_rgb[c])
for c in range(3):
    final_rgb[:, :, c] = (1.0 - snow_alpha) * final_rgb[:, :, c] + snow_alpha * snow_rgb[c]

# Hexagon margin
normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

road_margin = smoothstep(1.5, 3.5, dist_edge)
for c in range(3):
    final_rgb[:, :, c] = road_margin * final_rgb[:, :, c] + (1.0 - road_margin) * 0.12
final_rgb[dist_edge < 0] = [0.08, 0.09, 0.11]
final_rgb = np.clip(final_rgb, 0.0, 1.0)

# Save test crop
im_test = Image.fromarray((final_rgb[::-1] * 255.0).astype(np.uint8))
im_test.save('scratch/test_crisp_rock_1k.png')

# Extreme zoom on steep cliff side to verify ZERO STRETCHING
ix_mid = dim // 2
iy_mid = dim // 2
crop_zoom = final_rgb[iy_mid-120:iy_mid+120, ix_mid-120:ix_mid+120]
im_crop = Image.fromarray((crop_zoom[::-1] * 255.0).astype(np.uint8))
im_crop.save('scratch/test_crisp_zoom.png')

print(f"Test completed in {time.time()-t0:.2f}s. Saved test_crisp_rock_1k.png and test_crisp_zoom.png")
