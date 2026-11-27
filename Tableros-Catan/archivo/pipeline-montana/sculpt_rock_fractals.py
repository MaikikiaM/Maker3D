import math
import time
import numpy as np
import scipy.ndimage as ndi

print("=== SCULPTING FRACTAL BEDROCK & EROSION SIMULATION ===")
t0 = time.time()

R = 64.0
APOTHEM = R * math.cos(math.radians(30))
nx, ny = 1150, 1328
xs = np.linspace(-APOTHEM, APOTHEM, nx, dtype=np.float32)
ys = np.linspace(-R, R, ny, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)
dx = float(xs[1] - xs[0])
dy = float(ys[1] - ys[0])

massif = np.load('scratch/massif_base.npy')

# -------------------------------------------------------------------------
# 1. SPECTRAL 1/f^beta FRACTAL BEDROCK (Mandelbrot / Musgrave Synthesis)
# -------------------------------------------------------------------------
print("1. Generating scale-invariant fractal rock roughness via FFT...")
def generate_fBm_fft(width, height, beta=2.15, seed=123):
    rng = np.random.RandomState(seed)
    kx = np.fft.rfftfreq(width)
    ky = np.fft.fftfreq(height)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1.0
    amplitude = 1.0 / (K ** (beta / 2.0))
    amplitude[0, 0] = 0.0
    # High-frequency bandpass envelope to suppress DC shift
    envelope = 1.0 - np.exp(-(K / 0.015)**2)
    amplitude *= envelope
    phase = rng.uniform(0, 2 * np.pi, size=amplitude.shape)
    spectrum = amplitude * np.exp(1j * phase)
    noise = np.fft.irfft2(spectrum, s=(height, width)).astype(np.float32)
    return (noise - np.mean(noise)) / np.std(noise)

# Two independent fractal octaves: macro crags (beta=2.2) and micro rock texture (beta=1.85)
fBm_macro = generate_fBm_fft(nx, ny, beta=2.2, seed=77)
fBm_micro = generate_fBm_fft(nx, ny, beta=1.85, seed=333)

fractal_crags = 2.4 * fBm_macro + 0.8 * fBm_micro

# -------------------------------------------------------------------------
# 2. VORONOI GRANITE / BASALT JOINT NETWORK (Angular Block Fractures)
# -------------------------------------------------------------------------
print("2. Generating angular granite joint blocks via Voronoi distance field...")
# Scatter random joint seed points across the domain
rng_seeds = np.random.RandomState(456)
num_seeds = 350
seed_y = rng_seeds.randint(10, ny - 10, num_seeds)
seed_x = rng_seeds.randint(10, nx - 10, num_seeds)

grid_seeds = np.zeros((ny, nx), dtype=bool)
grid_seeds[seed_y, seed_x] = True

# Euclidean distance transform to nearest seed
dist_voronoi = ndi.distance_transform_edt(~grid_seeds).astype(np.float32)
# Normalize to local facet cell relief
dist_voronoi_smooth = ndi.gaussian_filter(dist_voronoi, 2.0)
gy_v, gx_v = np.gradient(dist_voronoi_smooth)
voronoi_ridges = np.hypot(gx_v, gy_v)
# Sharp fracture planes
joint_relief = -1.8 * np.exp(-(dist_voronoi / 12.0)**1.5)

# Combine fractal crags and rock joints modulated by elevation
elev_weight = np.clip(massif / 58.0, 0.0, 1.0)**0.85
Z_sculpted = massif + (fractal_crags + joint_relief) * elev_weight

# Ensure apex reaches 58.0 mm
Z_sculpted = np.maximum(0.0, Z_sculpted)
apex_idx = np.unravel_index(np.argmax(massif), massif.shape)
Z_sculpted = Z_sculpted * (58.0 / Z_sculpted[apex_idx])

# -------------------------------------------------------------------------
# 3. THERMAL EROSION SIMULATION (Angle of Repose & Talus Scree Deposition)
# -------------------------------------------------------------------------
print("3. Simulating physical thermal erosion (talus scree talus fans)...")
# Any slope exceeding 38 degrees sheds material downward to form talus scree cones
talus_crit_deg = 37.0
T_x = np.tan(np.radians(talus_crit_deg)) * dx
T_y = np.tan(np.radians(talus_crit_deg)) * dy

Z_eroded = Z_sculpted.copy()
# Run 12 iterations of thermal relaxation
for it in range(12):
    diff_r = Z_eroded[:, :-1] - Z_eroded[:, 1:]
    diff_l = Z_eroded[:, 1:] - Z_eroded[:, :-1]
    diff_d = Z_eroded[:-1, :] - Z_eroded[1:, :]
    diff_u = Z_eroded[1:, :] - Z_eroded[:-1, :]
    
    excess_r = np.maximum(0.0, diff_r - T_x)
    excess_l = np.maximum(0.0, diff_l - T_x)
    excess_d = np.maximum(0.0, diff_d - T_y)
    excess_u = np.maximum(0.0, diff_u - T_y)
    
    c = 0.22
    Z_eroded[:, :-1] -= c * 0.25 * excess_r
    Z_eroded[:, 1:]  += c * 0.25 * excess_r
    Z_eroded[:, 1:]  -= c * 0.25 * excess_l
    Z_eroded[:, :-1] += c * 0.25 * excess_l
    Z_eroded[:-1, :] -= c * 0.25 * excess_d
    Z_eroded[1:, :]  += c * 0.25 * excess_d
    Z_eroded[1:, :]  -= c * 0.25 * excess_u
    Z_eroded[:-1, :] += c * 0.25 * excess_u

print(f"Thermal erosion converged. Max height: {np.max(Z_eroded):.2f} mm")

# Re-scale to exactly 58.00 mm relief at summit
Z_eroded = Z_eroded * (58.0 / Z_eroded[apex_idx])

# -------------------------------------------------------------------------
# 4. HEXAGON CLEARANCE & BASE INTEGRATION
# -------------------------------------------------------------------------
print("4. Applying perimeter margins and settlement pad clearance...")
normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

v_angles = [(30 + 60 * k) * math.pi / 180 for k in range(6)]
hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles])
dist_vert = np.full_like(X, 999.0)
for vx, vy in hex_corners:
    dist_vert = np.minimum(dist_vert, np.hypot(X - vx, Y - vy))

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# Road margin clearance (1.5mm flat + soft blend)
edge_clear = smoothstep(1.8, 8.5, dist_edge)
# Corner settlement clearance (11mm flat pads)
vert_clear = smoothstep(9.5, 14.5, dist_vert)
total_clearance = edge_clear * vert_clear

# Final relief normalized to exactly 58.00 mm (peak at 83.00 mm)
relief_cleared = Z_eroded * total_clearance
max_r = np.max(relief_cleared)
if max_r > 0:
    relief_cleared = relief_cleared * (58.00 / max_r)

# Final Z top surface
Z_final = 25.0 + relief_cleared
Z_final[dist_edge < 0] = 25.0

print(f"Final topography created! Range: [{np.min(Z_final):.2f}, {np.max(Z_final):.2f}] mm")
print(f"Time elapsed: {time.time()-t0:.2f}s")

np.save('scratch/Z_top_master_2m.npy', Z_final)
print("Saved scratch/Z_top_master_2m.npy")
