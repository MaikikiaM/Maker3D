import os
import math
import time
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

print("=== RE-SYNTHESIZING CONTINUOUS TRUECOLOR WITH NATURAL ALPINE SNOWLINE ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

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
print(f"Z 4K interpolated in {time.time()-t0:.2f}s. Z in [{np.min(Z_4k):.2f}, {np.max(Z_4k):.2f}] mm")

# Derivatives & surface normals
print("Computing surface gradients, normals, slope and aspect...")
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
ao_cavity = np.clip(1.0 + laplace_4k * 0.45, 0.45, 1.0)
ao_cavity = ndi.gaussian_filter(ao_cavity, 1.5)

def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# Non-periodic fractal rock texture synthesis
print("Generating scale-invariant fractal rock fields via FFT...")
def generate_fBm_4k(width, height, beta=2.2, seed=999):
    rng = np.random.RandomState(seed)
    s_w, s_h = width // 2, height // 2
    kx = np.fft.rfftfreq(s_w)
    ky = np.fft.fftfreq(s_h)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1.0
    amplitude = 1.0 / (K ** (beta / 2.0))
    amplitude[0, 0] = 0.0
    phase = rng.uniform(0, 2*np.pi, size=amplitude.shape)
    spec = amplitude * np.exp(1j * phase)
    noise_half = np.fft.irfft2(spec, s=(s_h, s_w)).astype(np.float32)
    noise_full = ndi.zoom(noise_half, 2.0, order=1)
    return (noise_full - np.mean(noise_full)) / np.std(noise_full)

fBm_rock = generate_fBm_4k(dim_tex, dim_tex, beta=2.2, seed=555)
fBm_grain = generate_fBm_4k(dim_tex, dim_tex, beta=1.8, seed=777)

# -------------------------------------------------------------------------
# 1. CONTINUOUS NATURAL BEDROCK LITHOLOGY
# -------------------------------------------------------------------------
print("Painting natural bedrock lithology...")
cliff_factor = smoothstep(24.0, 56.0, slope_deg)
alt_factor = smoothstep(8.0, 52.0, RELIEF_4k)

rock_val = np.clip(0.48 - 0.35 * cliff_factor + 0.28 * alt_factor + 0.12 * fBm_rock + 0.05 * fBm_grain, 0.0, 1.0)

c_slate = np.array([0.18, 0.20, 0.23], dtype=np.float32)
c_granite = np.array([0.48, 0.49, 0.51], dtype=np.float32)
c_quartzite = np.array([0.68, 0.67, 0.64], dtype=np.float32)

rock_rgb = np.zeros((dim_tex, dim_tex, 3), dtype=np.float32)
for c in range(3):
    rock_rgb[:, :, c] = np.where(
        rock_val < 0.5,
        (1.0 - rock_val*2.0) * c_slate[c] + (rock_val*2.0) * c_granite[c],
        (1.0 - (rock_val-0.5)*2.0) * c_granite[c] + ((rock_val-0.5)*2.0) * c_quartzite[c]
    )

for c in range(3):
    rock_rgb[:, :, c] *= ao_cavity

# -------------------------------------------------------------------------
# 2. REALISTIC ALPINE SNOW DISTRIBUTION (Aspect, Elevation, Couloirs, Ridges)
# -------------------------------------------------------------------------
print("Synthesizing realistic non-uniform alpine snow distribution...")
# Solar aspect: South / Southwest slopes (Y < 0 or X > 0) receive more solar melting
# Solar direction from South-Southwest (azim 205 deg, alt 45 deg)
az_sun = 205.0 * math.pi / 180.0
alt_sun = 45.0 * math.pi / 180.0
sun_dir = np.array([math.cos(alt_sun) * math.sin(az_sun), math.cos(alt_sun) * math.cos(az_sun), math.sin(alt_sun)], dtype=np.float32)
solar_exposure = np.clip(Nx * sun_dir[0] + Ny * sun_dir[1] + Nz * sun_dir[2], 0.0, 1.0)

# Thermal elevation offset: Sun-baked slopes push snowline higher by ~8-10 mm
effective_relief = RELIEF_4k - 9.0 * solar_exposure

# High elevation permanent snowpack (focused on upper third: relief 34 to 52 mm, Z = 59 to 77 mm)
summit_snow = smoothstep(34.0, 50.0, effective_relief)

# Slope retention: slopes steeper than 42 deg shed snow rapidly
slope_retention = 1.0 - smoothstep(38.0, 54.0, slope_deg)

# Ridge wind-scouring: convex arêtes (laplace > 0.05) lose snow to wind deflation
ridge_scour = smoothstep(0.02, 0.25, laplace_4k)
wind_factor = 1.0 - 0.65 * ridge_scour

# Couloirs and hollows: snow funnels and accumulates deeply in concave chutes (laplace < -0.05)
# Couloirs extend snow much lower down (down to relief 20 mm, Z = 45 mm)
couloir_funnel = smoothstep(-0.65, -0.05, laplace_4k) * smoothstep(18.0, 38.0, RELIEF_4k)

# Shelves and ledges: gentle horizontal terraces catch snow patches
ledge_catch = smoothstep(0.15, 0.70, fBm_rock) * (1.0 - smoothstep(28.0, 48.0, slope_deg)) * smoothstep(25.0, 42.0, RELIEF_4k)

# Combine:
# Upper summit gets continuous snow modulated by slope retention and wind scouring
# Mid mountain gets couloir tongues and shelf patches
# Lower mountain has ZERO snow!
snow_raw = (
    summit_snow * (0.25 + 0.75 * slope_retention) * wind_factor +
    0.60 * couloir_funnel * (1.0 - 0.5 * solar_exposure) +
    0.35 * ledge_catch +
    0.06 * fBm_grain
)
# Organic non-uniform feathering without harsh threshold
snow_alpha = smoothstep(0.32, 0.76, snow_raw)
# Zero snow below relief 16 mm (Z < 41 mm)
snow_alpha *= smoothstep(16.0, 24.0, RELIEF_4k)

# Snow color palette: sunlit powder vs shaded snow
snow_sun = np.array([0.985, 0.99, 1.00], dtype=np.float32)
snow_shade = np.array([0.86, 0.90, 0.96], dtype=np.float32)
snow_slope_shade = smoothstep(18.0, 50.0, slope_deg)

snow_rgb = np.zeros((dim_tex, dim_tex, 3), dtype=np.float32)
for c in range(3):
    snow_rgb[:, :, c] = (1.0 - snow_slope_shade) * snow_sun[c] + snow_slope_shade * snow_shade[c]

# Glacial Ice: concentrated in high-altitude sheltered cirques (North-East faces in shadow)
cirque_shadow = (1.0 - solar_exposure)
cirque_ice = smoothstep(26.0, 44.0, RELIEF_4k) * smoothstep(-1.0, -0.15, laplace_4k) * cirque_shadow * 0.70
ice_alpha = smoothstep(0.28, 0.68, cirque_ice + 0.06 * fBm_grain) * (1.0 - snow_alpha * 0.80)
ice_rgb = np.array([0.38, 0.68, 0.85], dtype=np.float32)

# -------------------------------------------------------------
# 3. SUB-ALPINE TUNDRA & LICHEN (SOFT BASAL SKIRT)
# -------------------------------------------------------------
print("Painting sub-alpine tundra and lichen...")
tundra_elev = smoothstep(0.1, 1.2, RELIEF_4k) * (1.0 - smoothstep(3.5, 9.5, RELIEF_4k))
tundra_slope = 1.0 - smoothstep(12.0, 25.0, slope_deg)
tundra_cluster = smoothstep(0.15, 0.75, fBm_rock)
tundra_alpha = np.clip(tundra_elev * tundra_slope * (0.45 + 0.55 * tundra_cluster) * 1.6, 0.0, 1.0)
tundra_alpha = smoothstep(0.18, 0.65, tundra_alpha) * (1.0 - snow_alpha)

moss_deep = np.array([0.22, 0.42, 0.18], dtype=np.float32)
lichen_gold = np.array([0.56, 0.52, 0.24], dtype=np.float32)
tundra_rgb = np.zeros((dim_tex, dim_tex, 3), dtype=np.float32)
for c in range(3):
    tundra_rgb[:, :, c] = tundra_alpha * moss_deep[c] + (1.0 - tundra_alpha) * lichen_gold[c]

# -------------------------------------------------------------
# 4. COMPOSITE TRUECOLOR LAYERS
# -------------------------------------------------------------
print("Compositing continuous layers...")
diffuse_4k = rock_rgb.copy()
for c in range(3):
    diffuse_4k[:, :, c] = (1.0 - tundra_alpha) * diffuse_4k[:, :, c] + tundra_alpha * tundra_rgb[:, :, c]
for c in range(3):
    diffuse_4k[:, :, c] = (1.0 - ice_alpha) * diffuse_4k[:, :, c] + ice_alpha * ice_rgb[c]
for c in range(3):
    diffuse_4k[:, :, c] = (1.0 - snow_alpha) * diffuse_4k[:, :, c] + snow_alpha * snow_rgb[:, :, c]

# Hexagon road border
normals_hex = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge_t = np.full_like(X_t, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge_t = np.minimum(dist_edge_t, APOTHEM - (X_t * nx_h + Y_t * ny_h))

road_margin = smoothstep(1.5, 3.5, dist_edge_t)
c_border = np.array([0.13, 0.14, 0.16], dtype=np.float32)
for c in range(3):
    diffuse_4k[:, :, c] = road_margin * diffuse_4k[:, :, c] + (1.0 - road_margin) * c_border[c]

diffuse_4k[dist_edge_t < 0] = [0.08, 0.09, 0.11]
diffuse_4k = np.clip(diffuse_4k, 0.0, 1.0)

# Save 4K Diffuse PNG
out_4k = os.path.join(out_dir, "catan_mountain_diffuse_4k.png")
im_4k = Image.fromarray((diffuse_4k[::-1] * 255.0).astype(np.uint8))
im_4k.save(out_4k, optimize=True)
print(f"Saved 4K Diffuse Texture: {out_4k} ({os.path.getsize(out_4k)/1024/1024:.2f} MB)")

# Save 2K for WebGL viewer
im_2k = im_4k.resize((2048, 2048), Image.Resampling.LANCZOS)
out_2k = os.path.join(out_dir, "catan_mountain_diffuse_2k.png")
im_2k.save(out_2k, optimize=True)
print(f"Saved 2K Diffuse Texture: {out_2k} ({os.path.getsize(out_2k)/1024/1024:.2f} MB)")

# Save diffuse array for PLY vertex colors
np.save('scratch/diffuse_4k_array.npy', diffuse_4k)
print(f"Total painting time: {time.time()-t0:.2f}s")
