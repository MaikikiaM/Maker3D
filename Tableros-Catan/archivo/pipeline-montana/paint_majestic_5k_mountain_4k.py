import os
import math
import time
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

print("=== 1. SYNTHESIZING MAJESTIC 5K GLACIATED PEAK (ALPAMAYO / MONT BLANC INSPIRED) ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D"
brain_dir = r"C:\Users\migue\Proyectos IA varios\Miscelanea\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

R = 64.0
APOTHEM = R * math.cos(math.radians(30))
Z_BASE = 25.0
Z_PEAK = 83.0

dim_tex = 4096
xs_t = np.linspace(-APOTHEM, APOTHEM, dim_tex, dtype=np.float32)
ys_t = np.linspace(-R, R, dim_tex, dtype=np.float32)
dx_t = float(xs_t[1] - xs_t[0])
dy_t = float(ys_t[1] - ys_t[0])

# Load Z master 2M
Z_master = np.load('scratch/Z_top_master_2m.npy')
ny_m, nx_m = Z_master.shape
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)

import scipy.interpolate as interp
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

print(f"Interpolating Z topography on 4K grid ({dim_tex}x{dim_tex})...")
chunk_size = dim_tex // 4
Z_chunks = []
for c_idx in range(4):
    y_c = ys_t[c_idx*chunk_size : (c_idx+1)*chunk_size]
    X_c, Y_c = np.meshgrid(xs_t, y_c)
    pts = np.column_stack((Y_c.ravel(), X_c.ravel()))
    Z_chunks.append(interp_z(pts).reshape((chunk_size, dim_tex)))
Z_4k = np.vstack(Z_chunks)
RELIEF_4k = np.maximum(0.0, Z_4k - Z_BASE)
print(f"Z 4K interpolated in {time.time()-t0:.2f}s. Range: [{np.min(Z_4k):.2f}, {np.max(Z_4k):.2f}] mm")

# Topographic Derivatives & Normals
gy_4k, gx_4k = np.gradient(Z_4k, dy_t, dx_t)
N_norm = np.sqrt(gx_4k**2 + gy_4k**2 + 1.0)
Nx = -gx_4k / N_norm
Ny = -gy_4k / N_norm
Nz = 1.0 / N_norm

slope_mag = np.hypot(gx_4k, gy_4k)
slope_deg = np.degrees(np.arctan(slope_mag))

# Curvature (Laplacian): negative = concave hollows/couloirs; positive = convex ridges/arêtes
laplace_4k = ndi.laplace(Z_4k)

# Soft cavity shading (subtle micro-shadows, NOT black sludge!)
ao_cavity = np.clip(1.0 + laplace_4k * 0.18, 0.65, 1.0)
ao_cavity = ndi.gaussian_filter(ao_cavity, 1.5)

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# Load the 45 Greys and 15 Greens calibrated palettes
greys_45 = np.load('scratch/greys_45.npy')    # shape (45, 3)
greens_15 = np.load('scratch/greens_15.npy')  # shape (15, 3)

# High-frequency rock & ice procedural texture
print("Synthesizing native 4K rock & firn noise...")
def generate_fBm_native_4k(width, height, beta=2.2, seed=999):
    rng = np.random.RandomState(seed)
    kx = np.fft.rfftfreq(width)
    ky = np.fft.fftfreq(height)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1.0
    amplitude = 1.0 / (K ** (beta / 2.0))
    amplitude[0, 0] = 0.0
    amplitude *= (1.0 - np.exp(-(K / 0.005)**2))
    phase = rng.uniform(0, 2*np.pi, size=amplitude.shape)
    spec = amplitude * np.exp(1j * phase)
    noise = np.fft.irfft2(spec, s=(height, width)).astype(np.float32)
    return (noise - np.mean(noise)) / np.std(noise)

fBm_rock_4k = generate_fBm_native_4k(dim_tex, dim_tex, beta=2.3, seed=456)
fBm_grain_4k = generate_fBm_native_4k(dim_tex, dim_tex, beta=1.9, seed=789)

# -------------------------------------------------------------------------
# A. BEDROCK GEOLOGY (45 GREY TONES - RICH ALPINE GRANITE & SCHIST)
# -------------------------------------------------------------------------
print("Synthesizing alpine bedrock from 45 geological greys...")
# Tone indices: 0 = Pure White, 44 = Deep Basalt.
# Realistic bedrock should NOT be pitch black:
# - Sun-facing & high arêtes: Tone 12 - 18 (Silver-grey granite, quartz highlights)
# - Mid-slopes & buttresses: Tone 20 - 28 (Cold mountain granite)
# - Fissures, undercuts & deep crevices: Tone 30 - 38 (Deep wet slate/basalt)

cliff_steepness = smoothstep(30.0, 70.0, slope_deg)
altitude_factor = smoothstep(10.0, 50.0, RELIEF_4k)
concavity_factor = smoothstep(0.05, -0.20, laplace_4k) # hollows are darker

# continuous rock tone in [12.0, 36.0]
rock_t = 0.40 + 0.35 * cliff_steepness - 0.20 * altitude_factor + 0.20 * concavity_factor + 0.08 * fBm_rock_4k
rock_t = np.clip(rock_t, 0.0, 1.0)

grey_idx_f = 12.0 + rock_t * 24.0 # Range: tone 12 to tone 36 (crisp, beautiful granite!)
grey_idx_f = np.clip(grey_idx_f, 0.0, 43.999)

idx_lo = np.floor(grey_idx_f).astype(np.int32)
idx_hi = np.minimum(idx_lo + 1, 44)
frac = (grey_idx_f - idx_lo)[:, :, np.newaxis]

rock_rgb = (1.0 - frac) * greys_45[idx_lo] + frac * greys_45[idx_hi]
for c in range(3):
    rock_rgb[:, :, c] *= ao_cavity

# -------------------------------------------------------------------------
# B. MAJESTIC 5K GLACIATED SNOWPACK (ALPAMAYO / MONT BLANC REGIME)
# -------------------------------------------------------------------------
print("Synthesizing majestic glaciated snowpack & ice flutings...")

# Sun exposure for lighting & thermal aspect
az_sun = 210.0 * math.pi / 180.0
alt_sun = 48.0 * math.pi / 180.0
sun_dir = np.array([math.cos(alt_sun) * math.sin(az_sun), math.cos(alt_sun) * math.cos(az_sun), math.sin(alt_sun)], dtype=np.float32)
solar_exposure = np.clip(Nx * sun_dir[0] + Ny * sun_dir[1] + Nz * sun_dir[2], 0.0, 1.0)

# 1. Base Elevation Snow Probability:
# In iconic 5k peaks, permanent snow covers the upper 70% of relief.
# Summit is at RELIEF = 58 mm (Z = 83 mm).
# Permanent snowfield starts strongly at RELIEF >= 14 mm (Z >= 39 mm).
# Thermal aspect: south-facing faces melt slightly more (2-3 mm offset, not 12 mm!).
effective_relief = RELIEF_4k - 3.5 * solar_exposure
elev_snow = smoothstep(12.0, 28.0, effective_relief)

# 2. Glacial Valleys & Couloirs:
# Snow packs thick in valleys (concave curvature laplace < -0.02) and descends as glacial tongues!
glacial_tongues = smoothstep(0.02, -0.25, laplace_4k) * smoothstep(6.0, 22.0, RELIEF_4k)

# 3. Fluted Ice Couloirs (Alpamayo style):
# Vertical fluting channels on steep slopes that hold snow/ice
fluted_retention = smoothstep(-0.40, -0.05, laplace_4k) * smoothstep(30.0, 65.0, slope_deg)

# 4. Realistic Alpine Slope Retention:
# Snow easily clings to slopes up to 55° in cold 5,000m mountains (firn / rime / ice).
# Between 55° and 68°, snow clings in flutes and couloirs.
# Only sheer vertical cliffs (> 68° - 72°) shed snow completely.
general_slope_retention = 1.0 - smoothstep(52.0, 68.0, slope_deg)
# Couloirs and hollows retain snow even on steeper faces up to 72°!
couloir_slope_retention = 1.0 - smoothstep(62.0, 74.0, slope_deg)
slope_retention = np.maximum(general_slope_retention, fluted_retention * couloir_slope_retention)

# 5. Wind Scouring on Knife-Edge Arêtes:
# Razor arêtes (sharp positive curvature laplace > 0.08) expose bare rock ribs.
arete_scour = smoothstep(0.04, 0.16, laplace_4k)
wind_coverage = 1.0 - 0.70 * arete_scour

# 6. High Summit Dominance:
# Above relief 38 mm (Z >= 63 mm), snow coverage is overwhelming (> 90%), like Mont Blanc / Alpamayo!
summit_dominance = smoothstep(36.0, 50.0, RELIEF_4k)

# Combine snow factors:
snow_raw = (elev_snow * wind_coverage + 0.75 * glacial_tongues + 0.40 * fluted_retention) * slope_retention
snow_raw = np.maximum(snow_raw, summit_dominance * (1.0 - smoothstep(66.0, 75.0, slope_deg)))

# Crisp organic snow threshold
snow_mask = smoothstep(0.32, 0.68, snow_raw + 0.12 * fBm_rock_4k)

# Snow Color Modulation:
# - Sunlit fresh firn: Brilliant White (#ffffff, Greys Tone 0)
# - Shaded alpine snow: Cool pale blue-grey (#dbe7f4, Greys Tone 4)
# - Ancient glacial ice in deep hollows: Subtle glacial cyan (#c8e8f5)
snow_shade_factor = 1.0 - solar_exposure
glacial_ice_factor = smoothstep(0.0, -0.30, laplace_4k) * smoothstep(20.0, 45.0, RELIEF_4k)

snow_rgb = np.zeros((dim_tex, dim_tex, 3), dtype=np.float32)
# Base snow colors
col_sun = greys_45[0]      # [1.0, 1.0, 1.0] (Tone 0: Arctic White)
col_shade = greys_45[4]    # [0.89, 0.90, 0.93] (Tone 4: Alpine Shaded White)
col_glacier = np.array([0.80, 0.92, 0.98], dtype=np.float32) # Pure Glacial Aquamarine

for c in range(3):
    s_col = (1.0 - 0.25 * snow_shade_factor) * col_sun[c] + 0.25 * snow_shade_factor * col_shade[c]
    # Blend subtle glacial blue in ancient ice couloirs
    s_col = (1.0 - 0.35 * glacial_ice_factor) * s_col + 0.35 * glacial_ice_factor * col_glacier[c]
    snow_rgb[:, :, c] = s_col

# -------------------------------------------------------------------------
# C. BOTANICAL SUB-ALPINE APRON (15 CALIBRATED GREENS)
# -------------------------------------------------------------------------
print("Synthesizing sub-alpine botanical apron from 15 greens...")
# Confined to the lowest base skirt: RELIEF in [0.2, 5.0] mm (Z in [25.2, 30.0] mm) on gentle slopes (< 25 deg)
tundra_elev = smoothstep(0.2, 1.0, RELIEF_4k) * (1.0 - smoothstep(2.5, 6.0, RELIEF_4k))
tundra_slope = 1.0 - smoothstep(8.0, 26.0, slope_deg)
tundra_raw = tundra_elev * tundra_slope * (1.0 - snow_mask) * (0.60 + 0.40 * smoothstep(-0.8, 0.8, fBm_rock_4k))
tundra_mask = smoothstep(0.15, 0.60, tundra_raw)

# Green tones index: [0, 14]
# Sunny alluvial turf -> Tone 2 - 5
# Rich mossy hollows -> Tone 6 - 11
green_moisture = smoothstep(-0.25, 0.15, -laplace_4k)
green_t = np.clip(0.30 + 0.45 * green_moisture - 0.20 * solar_exposure + 0.15 * fBm_grain_4k, 0.0, 1.0)
green_idx_f = np.clip(green_t * 14.0, 0.0, 13.999)

g_lo = np.floor(green_idx_f).astype(np.int32)
g_hi = np.minimum(g_lo + 1, 14)
g_frac = (green_idx_f - g_lo)[:, :, np.newaxis]
tundra_rgb = (1.0 - g_frac) * greens_15[g_lo] + g_frac * greens_15[g_hi]

# -------------------------------------------------------------------------
# D. MASTER COMPOSITE
# -------------------------------------------------------------------------
print("Compositing 4K diffuse master...")
diffuse_4k = rock_rgb.copy()
for c in range(3):
    # Layer 1: Alpine botanical greens on lower skirt
    diffuse_4k[:, :, c] = (1.0 - tundra_mask) * diffuse_4k[:, :, c] + tundra_mask * tundra_rgb[:, :, c]
    # Layer 2: Majestic glaciated snowpack
    diffuse_4k[:, :, c] = (1.0 - snow_mask) * diffuse_4k[:, :, c] + snow_mask * snow_rgb[:, :, c]

# Ensure pristine base edge (clear road boundary at Z_BASE = 25.0)
# Road flat border is clean slate granite
road_flat = smoothstep(0.0, 0.25, RELIEF_4k)
road_grey = greys_45[22] # Medium granite road border
for c in range(3):
    diffuse_4k[:, :, c] = (1.0 - road_flat) * road_grey[c] + road_flat * diffuse_4k[:, :, c]

# Clip and save
diffuse_4k = np.clip(diffuse_4k * 255.0, 0.0, 255.0).astype(np.uint8)
img_4k = Image.fromarray(diffuse_4k, mode='RGB')

diffuse_4k_path = os.path.join(out_dir, "catan_mountain_diffuse_4k.png")
img_4k.save(diffuse_4k_path, optimize=True)
print(f"Saved 4K diffuse: {diffuse_4k_path} ({os.path.getsize(diffuse_4k_path):,} bytes)")

# Generate 2K diffuse for WebGL and 3MF
print("Downsampling to 2K for WebGL Viewer and 3MF...")
img_2k = img_4k.resize((2048, 2048), Image.Resampling.LANCZOS)
diffuse_2k_path = os.path.join(out_dir, "catan_mountain_diffuse_2k.png")
img_2k.save(diffuse_2k_path, optimize=True)
print(f"Saved 2K diffuse: {diffuse_2k_path} ({os.path.getsize(diffuse_2k_path):,} bytes)")

print(f"=== COMPLETED IN {time.time()-t0:.2f}s ===")
