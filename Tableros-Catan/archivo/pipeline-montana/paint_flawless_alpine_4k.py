import os
import math
import time
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

print("=== 2. SYNTHESIZING FLAWLESS CONTINUOUS TRUECOLOR (NATIVE 4K, NO PIXELATION) ===")
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

# Load new flawless Z master
Z_master = np.load('scratch/Z_top_master_2m.npy')
ny_m, nx_m = Z_master.shape
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)

import scipy.interpolate as interp
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

print(f"Interpolating Z topography on native 4K grid ({dim_tex}x{dim_tex})...")
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

# Surface gradients and normals
print("Computing 4K surface gradients, normals, slope and aspect...")
gy_4k, gx_4k = np.gradient(Z_4k, dy_t, dx_t)
N_norm = np.sqrt(gx_4k**2 + gy_4k**2 + 1.0)
Nx = -gx_4k / N_norm
Ny = -gy_4k / N_norm
Nz = 1.0 / N_norm

slope_mag = np.hypot(gx_4k, gy_4k)
slope_deg = np.degrees(np.arctan(slope_mag))

# Curvature (Laplacian for concave couloirs vs convex ridges)
laplace_4k = ndi.laplace(Z_4k)

# Ambient Crevice Occlusion (smooth, natural cavity shadowing)
ao_cavity = np.clip(1.0 + laplace_4k * 0.35, 0.40, 1.0)
ao_cavity = ndi.gaussian_filter(ao_cavity, 2.0)

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# Native 4K scale-invariant spectral fractal noise (NO downsampling, NO zooming!)
print("Generating native 4K scale-invariant fractal rock fields via FFT...")
def generate_fBm_native_4k(width, height, beta=2.2, seed=999):
    rng = np.random.RandomState(seed)
    kx = np.fft.rfftfreq(width)
    ky = np.fft.fftfreq(height)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1.0
    amplitude = 1.0 / (K ** (beta / 2.0))
    amplitude[0, 0] = 0.0
    # Soft high-pass to remove macro tilt
    amplitude *= (1.0 - np.exp(-(K / 0.005)**2))
    phase = rng.uniform(0, 2*np.pi, size=amplitude.shape)
    spec = amplitude * np.exp(1j * phase)
    noise = np.fft.irfft2(spec, s=(height, width)).astype(np.float32)
    return (noise - np.mean(noise)) / np.std(noise)

fBm_rock_4k = generate_fBm_native_4k(dim_tex, dim_tex, beta=2.3, seed=456)
fBm_grain_4k = generate_fBm_native_4k(dim_tex, dim_tex, beta=1.9, seed=789)

# -------------------------------------------------------------------------
# 1. CONTINUOUS NATURAL BEDROCK LITHOLOGY
# -------------------------------------------------------------------------
print("Painting continuous high-alpine granite & metamorphic rock lithology...")
cliff_factor = smoothstep(22.0, 54.0, slope_deg)
alt_factor = smoothstep(6.0, 50.0, RELIEF_4k)

# Smooth, continuous tonal modulation across the rock face
rock_tone = np.clip(0.48 - 0.32 * cliff_factor + 0.25 * alt_factor + 0.12 * fBm_rock_4k + 0.04 * fBm_grain_4k, 0.0, 1.0)

# Realistic alpine rock palette: slate headwalls, neutral granite, pale quartz crests
c_slate = np.array([0.18, 0.20, 0.23], dtype=np.float32)       # Dark wet slate
c_granite = np.array([0.46, 0.47, 0.49], dtype=np.float32)     # Weathered alpine granite
c_quartzite = np.array([0.66, 0.65, 0.62], dtype=np.float32)   # High-altitude quartz ridge

rock_rgb = np.zeros((dim_tex, dim_tex, 3), dtype=np.float32)
for c in range(3):
    rock_rgb[:, :, c] = np.where(
        rock_tone < 0.5,
        (1.0 - rock_tone*2.0) * c_slate[c] + (rock_tone*2.0) * c_granite[c],
        (1.0 - (rock_tone-0.5)*2.0) * c_granite[c] + ((rock_tone-0.5)*2.0) * c_quartzite[c]
    )
    # Apply cavity ambient occlusion
    rock_rgb[:, :, c] *= ao_cavity

# -------------------------------------------------------------------------
# 2. REALISTIC ALPINE SNOWPACK & GLACIAL ICE DISTRIBUTION
# -------------------------------------------------------------------------
print("Synthesizing natural solar-aspect alpine snowline & couloir funnels...")
# Solar direction from South-Southwest (solar azimuth 205 deg, solar altitude 45 deg)
az_sun = 205.0 * math.pi / 180.0
alt_sun = 45.0 * math.pi / 180.0
sun_dir = np.array([math.cos(alt_sun) * math.sin(az_sun), math.cos(alt_sun) * math.cos(az_sun), math.sin(alt_sun)], dtype=np.float32)
solar_exposure = np.clip(Nx * sun_dir[0] + Ny * sun_dir[1] + Nz * sun_dir[2], 0.0, 1.0)

# Solar thermal offset: Sun-baked slopes push snowline higher by ~10 mm
effective_relief = RELIEF_4k - 10.0 * solar_exposure

# Permanent snowpack on upper mountain (concentrated in upper third: relief 34 to 52 mm, Z = 59 to 77 mm)
summit_snow = smoothstep(33.0, 48.0, effective_relief)

# Slopes steeper than 44 degrees shed snow by gravity (avalanches down to couloirs)
slope_retention = 1.0 - smoothstep(38.0, 52.0, slope_deg)

# Ridge wind-scouring: convex arêtes (laplace > 0.05) lose snow to wind deflation
ridge_scour = smoothstep(0.02, 0.20, laplace_4k)
wind_factor = 1.0 - 0.70 * ridge_scour

# Couloirs and gullies: concave chutes (laplace < -0.05) funnel snow downward
couloir_funnel = smoothstep(-0.60, -0.05, laplace_4k) * smoothstep(18.0, 36.0, RELIEF_4k)

# Gentle ledges catch snow patches
ledge_catch = smoothstep(0.20, 0.70, fBm_rock_4k) * (1.0 - smoothstep(26.0, 46.0, slope_deg)) * smoothstep(24.0, 40.0, RELIEF_4k)

# Total continuous snow probability:
snow_raw = (
    summit_snow * (0.25 + 0.75 * slope_retention) * wind_factor +
    0.65 * couloir_funnel * (1.0 - 0.45 * solar_exposure) +
    0.30 * ledge_catch
)

# Smooth feathered snow alpha (NO thresholding, NO binary clipping, perfectly continuous!)
snow_alpha = smoothstep(0.35, 0.75, snow_raw)
# Zero snow below relief 16 mm (Z < 41 mm)
snow_alpha *= smoothstep(16.0, 24.0, RELIEF_4k)

# Snow lighting and color: sunlit snow vs shaded blue firn
snow_sun = np.array([0.985, 0.990, 1.000], dtype=np.float32)
snow_shade = np.array([0.860, 0.900, 0.965], dtype=np.float32)
snow_slope_shade = smoothstep(18.0, 48.0, slope_deg)

snow_rgb = np.zeros((dim_tex, dim_tex, 3), dtype=np.float32)
for c in range(3):
    snow_rgb[:, :, c] = (1.0 - snow_slope_shade) * snow_sun[c] + snow_slope_shade * snow_shade[c]

# Glacial Ice: concentrated in sheltered North-East cirque hollows
cirque_shadow = 1.0 - solar_exposure
cirque_ice = smoothstep(24.0, 42.0, RELIEF_4k) * smoothstep(-0.9, -0.15, laplace_4k) * cirque_shadow * 0.75
ice_alpha = smoothstep(0.30, 0.70, cirque_ice) * (1.0 - snow_alpha * 0.85)
ice_rgb = np.array([0.38, 0.68, 0.86], dtype=np.float32) # Translucent glacial cyan

# -------------------------------------------------------------------------
# 3. SUB-ALPINE ORGANIC TUNDRA & LICHEN (SOFT BASAL APRON, NO STICKERS!)
# -------------------------------------------------------------------------
print("Painting natural sub-alpine tundra and rock lichen...")
# Strictly confined to base skirt (relief 0.2 to 8.0 mm) on gentle slopes
tundra_elev = smoothstep(0.2, 1.5, RELIEF_4k) * (1.0 - smoothstep(3.0, 8.5, RELIEF_4k))
tundra_slope = 1.0 - smoothstep(10.0, 24.0, slope_deg)
# Smooth, non-thresholded continuous blend
tundra_alpha = np.clip(tundra_elev * tundra_slope * (0.50 + 0.50 * smoothstep(-1.0, 1.0, fBm_rock_4k)), 0.0, 1.0)
tundra_alpha = smoothstep(0.15, 0.70, tundra_alpha) * (1.0 - snow_alpha)

# Subtle, natural peat moss and golden xanthoria rock lichen (NO lime green!)
peat_moss = np.array([0.28, 0.38, 0.22], dtype=np.float32)    # Natural earthy alpine peat moss
gold_lichen = np.array([0.52, 0.48, 0.26], dtype=np.float32)  # High-altitude crustose lichen

tundra_rgb = np.zeros((dim_tex, dim_tex, 3), dtype=np.float32)
for c in range(3):
    tundra_rgb[:, :, c] = tundra_alpha * peat_moss[c] + (1.0 - tundra_alpha) * gold_lichen[c]

# -------------------------------------------------------------------------
# 4. COMPOSITE TRUECOLOR LAYERS (CONTINUOUS SPECTRUM)
# -------------------------------------------------------------------------
print("Compositing continuous layers...")
diffuse_4k = rock_rgb.copy()
for c in range(3):
    diffuse_4k[:, :, c] = (1.0 - tundra_alpha) * diffuse_4k[:, :, c] + tundra_alpha * tundra_rgb[:, :, c]
    diffuse_4k[:, :, c] = (1.0 - ice_alpha) * diffuse_4k[:, :, c] + ice_alpha * ice_rgb[c]
    diffuse_4k[:, :, c] = (1.0 - snow_alpha) * diffuse_4k[:, :, c] + snow_alpha * snow_rgb[:, :, c]

# Hexagon perimeter road border (3.5 mm margin at Z=25 mm)
normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge_t = np.full_like(X_t, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge_t = np.minimum(dist_edge_t, APOTHEM - (X_t * nx_h + Y_t * ny_h))

road_margin = smoothstep(1.5, 3.5, dist_edge_t)
c_border = np.array([0.14, 0.15, 0.17], dtype=np.float32)
for c in range(3):
    diffuse_4k[:, :, c] = road_margin * diffuse_4k[:, :, c] + (1.0 - road_margin) * c_border[c]

diffuse_4k[dist_edge_t < 0] = [0.09, 0.10, 0.12]
diffuse_4k = np.clip(diffuse_4k, 0.0, 1.0)

# Save 4K Diffuse PNG (Image coords: row 0 is top, Y = +R)
out_4k = os.path.join(out_dir, "catan_mountain_diffuse_4k.png")
im_4k = Image.fromarray((diffuse_4k[::-1] * 255.0).astype(np.uint8))
im_4k.save(out_4k, optimize=True)
print(f"Saved 4K Diffuse Texture: {out_4k} ({os.path.getsize(out_4k)/1024/1024:.2f} MB)")

# Save 2K Diffuse for WebGL viewer
im_2k = im_4k.resize((2048, 2048), Image.Resampling.LANCZOS)
out_2k = os.path.join(out_dir, "catan_mountain_diffuse_2k.png")
im_2k.save(out_2k, optimize=True)
print(f"Saved 2K Diffuse Texture: {out_2k} ({os.path.getsize(out_2k)/1024/1024:.2f} MB)")

# Save diffuse array for PLY vertex colors
np.save('scratch/diffuse_4k_array.npy', diffuse_4k)

# -------------------------------------------------------------------------
# 5. COMPUTE & SAVE FLAWLESS 4K TANGENT NORMAL MAP (NO MOIRE, NO RIPPLES!)
# -------------------------------------------------------------------------
print("Computing 4K Tangent Normal Map directly from flawless DEM...")
normal_u8 = np.zeros((dim_tex, dim_tex, 3), dtype=np.uint8)
normal_u8[:, :, 0] = ((Nx * 0.5 + 0.5) * 255.0).astype(np.uint8)
normal_u8[:, :, 1] = ((Ny * 0.5 + 0.5) * 255.0).astype(np.uint8)
normal_u8[:, :, 2] = ((Nz * 0.5 + 0.5) * 255.0).astype(np.uint8)

out_norm = os.path.join(out_dir, "catan_mountain_normal_4k.png")
Image.fromarray(normal_u8[::-1]).save(out_norm, optimize=True)
print(f"Saved Flawless 4K Normal Map: {out_norm} ({os.path.getsize(out_norm)/1024/1024:.2f} MB)")

print(f"All 4K textures synthesized flawlessly in {time.time()-t0:.2f}s!")
