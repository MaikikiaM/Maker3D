import os
import math
import time
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

print("=== 3. SYNTHESIZING MASTER 4K DIFFUSE WITH 45 GREYS + 15 GREENS + BARE CLIFFS ===")
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
X_t, Y_t = np.meshgrid(xs_t, ys_t)
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

# Derivatives & Normals
gy_4k, gx_4k = np.gradient(Z_4k, dy_t, dx_t)
N_norm = np.sqrt(gx_4k**2 + gy_4k**2 + 1.0)
Nx = -gx_4k / N_norm
Ny = -gy_4k / N_norm
Nz = 1.0 / N_norm

slope_mag = np.hypot(gx_4k, gy_4k)
slope_deg = np.degrees(np.arctan(slope_mag))

# Curvature (Laplacian)
laplace_4k = ndi.laplace(Z_4k)

# Ambient Crevice Occlusion
ao_cavity = np.clip(1.0 + laplace_4k * 0.35, 0.40, 1.0)
ao_cavity = ndi.gaussian_filter(ao_cavity, 2.0)

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# Load the 45 Greys and 15 Greens calibrated palettes
greys_45 = np.load('scratch/greys_45.npy')    # shape (45, 3)
greens_15 = np.load('scratch/greens_15.npy')  # shape (15, 3)

# Scale-invariant high-frequency rock grain
print("Synthesizing native 4K rock grain...")
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
# A. MAPPING 45 TONES OF GREY TO BEDROCK & CREVICE GEOLOGY
# -------------------------------------------------------------------------
# Grey palette index: 0 is purest white, 44 is deepest black.
# Bedrock spans tones 10 (pale quartz crests) to 43 (dark wet slate/basalt).
# We calculate a continuous tonal coordinate t_grey in [0, 44].
print("Mapping 45 continuous geological grey tones across bedrock...")
cliff_factor = smoothstep(18.0, 52.0, slope_deg) # Steep cliffs -> darker rock
alt_factor = smoothstep(6.0, 52.0, RELIEF_4k)    # High altitude -> paler quartzite crests

# Continuous tone parameter (0.0 to 1.0)
rock_t = np.clip(0.55 + 0.35 * cliff_factor - 0.28 * alt_factor + 0.12 * fBm_rock_4k + 0.04 * fBm_grain_4k, 0.0, 1.0)

# Map rock_t to grey index in [10.0, 43.0]
# Where rock_t=0 (high crests) -> Tone 10 (pale quartz)
# Where rock_t=1 (steep shadowed cliffs) -> Tone 43 (deep slate)
grey_idx_f = 10.0 + rock_t * 33.0
grey_idx_f = np.clip(grey_idx_f, 0.0, 43.999)

# Continuous piecewise linear interpolation between the 45 greys
idx_lo = np.floor(grey_idx_f).astype(np.int32)
idx_hi = np.minimum(idx_lo + 1, 44)
frac = (grey_idx_f - idx_lo)[:, :, np.newaxis]

rock_rgb = (1.0 - frac) * greys_45[idx_lo] + frac * greys_45[idx_hi]
for c in range(3):
    rock_rgb[:, :, c] *= ao_cavity

# -------------------------------------------------------------------------
# B. STRICT PHYSICAL ALPINE SNOWLINE: ZERO SNOW ON STEEP FACES (> 44 deg)
# -------------------------------------------------------------------------
print("Applying physical snow accumulation: ZERO snow on steep cliffs...")
# Solar aspect insolation
az_sun = 205.0 * math.pi / 180.0
alt_sun = 45.0 * math.pi / 180.0
sun_dir = np.array([math.cos(alt_sun) * math.sin(az_sun), math.cos(alt_sun) * math.cos(az_sun), math.sin(alt_sun)], dtype=np.float32)
solar_exposure = np.clip(Nx * sun_dir[0] + Ny * sun_dir[1] + Nz * sun_dir[2], 0.0, 1.0)

# Thermal elevation offset (Sun-baked South faces push snowline up by 12 mm)
effective_relief = RELIEF_4k - 12.0 * solar_exposure

# High-altitude permanent snowpack (upper third: relief 34 to 52 mm, Z = 59 to 77 mm)
summit_snow = smoothstep(33.0, 48.0, effective_relief)

# STRICT SLOPE SHEDDING:
# Slopes < 28 deg: 100% snow retention
# Slopes 28-44 deg: rapid gravity shedding
# Slopes >= 44 deg: EXACTLY 0.00% SNOW! (Bare granite cliff walls)
slope_retention = 1.0 - smoothstep(28.0, 44.0, slope_deg)

# Ridge wind-scouring: convex arêtes (laplace > 0.04) lose snow to wind
ridge_scour = smoothstep(0.02, 0.18, laplace_4k)
wind_factor = 1.0 - 0.85 * ridge_scour

# Couloirs & gullies: concave hollows (laplace < -0.05) catch snow tongues
couloir_funnel = smoothstep(-0.60, -0.05, laplace_4k) * smoothstep(18.0, 36.0, RELIEF_4k)

# Shelves and flat terraces catch snow
shelf_catch = smoothstep(0.18, 0.65, fBm_rock_4k) * smoothstep(24.0, 40.0, RELIEF_4k)

# Combine: NOTICE THAT EVERYTHING IS MULTIPLIED BY slope_retention!
# If slope >= 44 deg, slope_retention == 0 -> SNOW IS STRICTLY ZERO!
snow_raw = (summit_snow * wind_factor + 0.65 * couloir_funnel + 0.25 * shelf_catch) * slope_retention
snow_alpha = smoothstep(0.28, 0.72, snow_raw)
# Zero snow below relief 16 mm (Z < 41 mm)
snow_alpha *= smoothstep(16.0, 24.0, RELIEF_4k)

# Firn and shaded ice use cold grey tones from greys_45 (Tones 5-9)
snow_sun = greys_45[0]      # Tone 1: Arctic White
snow_shade = greys_45[6]    # Tone 7: Shaded snow grey
snow_slope_shade = smoothstep(16.0, 38.0, slope_deg)

snow_rgb = np.zeros((dim_tex, dim_tex, 3), dtype=np.float32)
for c in range(3):
    snow_rgb[:, :, c] = (1.0 - snow_slope_shade) * snow_sun[c] + snow_slope_shade * snow_shade[c]

# -------------------------------------------------------------------------
# C. MAPPING 15 TONES OF GREEN ACROSS SUB-ALPINE BOTANICAL APRON
# -------------------------------------------------------------------------
print("Mapping 15 botanical green tones across sub-alpine skirt...")
# Strictly confined to low alluvial skirt: relief 0.2 to 6.5 mm (Z in [25.2, 31.5]) on gentle slopes (< 22 deg)
tundra_alt = smoothstep(0.2, 1.2, RELIEF_4k) * (1.0 - smoothstep(2.5, 6.5, RELIEF_4k))
tundra_slope = 1.0 - smoothstep(6.0, 22.0, slope_deg)
tundra_alpha = np.clip(tundra_alt * tundra_slope * (0.50 + 0.50 * smoothstep(-0.8, 0.8, fBm_rock_4k)), 0.0, 1.0)
tundra_alpha = smoothstep(0.12, 0.65, tundra_alpha)

# Botanical green index: [0, 14]
# Sunny/dry -> V1 - V4 (pale chartreuse & sulphur lichen)
# Medium turf -> V5 - V8 (alpine meadow fescue & peat moss)
# Wet depressions / deep shadows -> V9 - V14 (damp crevice moss & dark humus)
moisture = smoothstep(-0.35, 0.15, -laplace_4k) # concave hollows retain moisture
green_t = np.clip(0.35 + 0.40 * moisture - 0.25 * solar_exposure + 0.15 * fBm_grain_4k, 0.0, 1.0)
green_idx_f = np.clip(green_t * 14.0, 0.0, 13.999)

g_lo = np.floor(green_idx_f).astype(np.int32)
g_hi = np.minimum(g_lo + 1, 14)
g_frac = (green_idx_f - g_lo)[:, :, np.newaxis]

tundra_rgb = (1.0 - g_frac) * greens_15[g_lo] + g_frac * greens_15[g_hi]

# -------------------------------------------------------------------------
# D. MASTER COMPOSITE (SPECTRUM OF 45 GREYS + 15 GREENS)
# -------------------------------------------------------------------------
print("Compositing continuous layers...")
diffuse_4k = rock_rgb.copy()
for c in range(3):
    diffuse_4k[:, :, c] = (1.0 - tundra_alpha) * diffuse_4k[:, :, c] + tundra_alpha * tundra_rgb[:, :, c]
    diffuse_4k[:, :, c] = (1.0 - snow_alpha) * diffuse_4k[:, :, c] + snow_alpha * snow_rgb[:, :, c]

# Hexagon road border (3.5 mm margin at Z=25 mm): Deep obsidian grey (Tone 44)
normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge_t = np.full_like(X_t, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge_t = np.minimum(dist_edge_t, APOTHEM - (X_t * nx_h + Y_t * ny_h))

road_margin = smoothstep(1.5, 3.5, dist_edge_t)
c_border = greys_45[42] # Tone 43
for c in range(3):
    diffuse_4k[:, :, c] = road_margin * diffuse_4k[:, :, c] + (1.0 - road_margin) * c_border[c]

diffuse_4k[dist_edge_t < 0] = greys_45[44] # Tone 45: deepest black
diffuse_4k = np.clip(diffuse_4k, 0.0, 1.0)

# Save 4K Diffuse PNG
out_4k = os.path.join(out_dir, "catan_mountain_diffuse_4k.png")
im_4k = Image.fromarray((diffuse_4k[::-1] * 255.0).astype(np.uint8))
im_4k.save(out_4k, optimize=True)
print(f"Saved Master 4K Diffuse: {out_4k} ({os.path.getsize(out_4k)/1024/1024:.2f} MB)")

# Save 2K for WebGL viewer
im_2k = im_4k.resize((2048, 2048), Image.Resampling.LANCZOS)
out_2k = os.path.join(out_dir, "catan_mountain_diffuse_2k.png")
im_2k.save(out_2k, optimize=True)
print(f"Saved 2K Diffuse: {out_2k} ({os.path.getsize(out_2k)/1024/1024:.2f} MB)")

# Save diffuse array for PLY
np.save('scratch/diffuse_4k_array.npy', diffuse_4k)

# Compute and save flawless tangent normal map
print("Computing 4K Tangent Normal Map directly from flawless DEM...")
normal_u8 = np.zeros((dim_tex, dim_tex, 3), dtype=np.uint8)
normal_u8[:, :, 0] = ((Nx * 0.5 + 0.5) * 255.0).astype(np.uint8)
normal_u8[:, :, 1] = ((Ny * 0.5 + 0.5) * 255.0).astype(np.uint8)
normal_u8[:, :, 2] = ((Nz * 0.5 + 0.5) * 255.0).astype(np.uint8)

out_norm = os.path.join(out_dir, "catan_mountain_normal_4k.png")
Image.fromarray(normal_u8[::-1]).save(out_norm, optimize=True)
print(f"Saved Flawless 4K Normal Map: {out_norm} ({os.path.getsize(out_norm)/1024/1024:.2f} MB)")

# Verification: Prove that steep faces have 0% snow
steep_mask = (dist_edge_t >= 2.0) & (slope_deg >= 45.0)
print(f"Total steep cliff pixels (slope >= 45 deg): {np.sum(steep_mask):,}")
snow_on_steep = snow_alpha[steep_mask]
print(f"Max snow on steep cliffs (slope >= 45 deg): {np.max(snow_on_steep):.6f}")
assert np.max(snow_on_steep) == 0.0, "FAIL: Snow found on steep cliffs!"
print("VERIFICATION PASSED: 100% of steep cliff faces (>= 45 deg) have EXACTLY 0.00% SNOW!")

print(f"Total synthesis time: {time.time()-t0:.2f}s")
