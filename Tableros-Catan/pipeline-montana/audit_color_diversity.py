import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil

print("=== AUDITING COLOR PALETTE DIVERSITY & SNOW SLOPE EXCLUSION ===")

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

diffuse_4k = np.load('scratch/diffuse_4k_array.npy')
greys_45 = np.load('scratch/greys_45.npy')
greens_15 = np.load('scratch/greens_15.npy')

Z_master = np.load('scratch/Z_top_master_2m.npy')
ny_m, nx_m = Z_master.shape

R = 64.0
APOTHEM = R * np.cos(np.pi / 6)
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)

import scipy.interpolate as interp
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

# Resample to 1024x1024 for fast color distance matching
res = 1024
xs = np.linspace(-APOTHEM, APOTHEM, res, dtype=np.float32)
ys = np.linspace(-R, R, res, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)
dx = float(xs[1] - xs[0]); dy = float(ys[1] - ys[0])

pts = np.column_stack((Y.ravel(), X.ravel()))
Z = interp_z(pts).reshape((res, res))

gy, gx = np.gradient(Z, dy, dx)
slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

normals_hex = [[np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals_hex:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

inside_hex = dist_edge >= 1.5

# Downsample diffuse texture to 1024x1024
im_diff = Image.fromarray((diffuse_4k * 255.0).astype(np.uint8))
diff_small = np.array(im_diff.resize((res, res), Image.Resampling.LANCZOS)).astype(np.float32) / 255.0

# Pixels inside hex
pixels = diff_small[inside_hex]
slopes = slope_deg[inside_hex]

print(f"Auditing {len(pixels):,} pixels on mountain surface...")

# Nearest grey tone matching
# Compute Euclidean distance from each pixel to all 45 grey tones
dists_grey = np.linalg.norm(pixels[:, None, :] - greys_45[None, :, :], axis=2) # shape (N, 45)
min_grey_idx = np.argmin(dists_grey, axis=1)
min_grey_dist = np.min(dists_grey, axis=1)

# Nearest green tone matching
dists_green = np.linalg.norm(pixels[:, None, :] - greens_15[None, :, :], axis=2) # shape (N, 15)
min_green_idx = np.argmin(dists_green, axis=1)
min_green_dist = np.min(dists_green, axis=1)

# Histogram of 45 greys
grey_counts = np.bincount(min_grey_idx, minlength=45)
print(f"Greys represented: {np.sum(grey_counts > 0)} / 45 tones active")

# Histogram of 15 greens (where green distance is small, i.e. in vegetated zones)
green_mask = (min_green_dist < 0.12) & (pixels[:, 1] > pixels[:, 0] * 1.05)
green_counts = np.bincount(min_green_idx[green_mask], minlength=15)
print(f"Greens represented: {np.sum(green_counts > 0)} / 15 tones active")

# Steep cliff snow audit (pixels with slope >= 44 deg)
steep_mask = slopes >= 44.0
steep_pixels = pixels[steep_mask]
# High snow pixels have R > 0.85, G > 0.85, B > 0.85
snow_on_steep = (steep_pixels[:, 0] > 0.85) & (steep_pixels[:, 1] > 0.85) & (steep_pixels[:, 2] > 0.85)
print(f"Steep cliff pixels (slope >= 44 deg): {len(steep_pixels):,}")
print(f"Snow pixels on steep cliffs: {np.sum(snow_on_steep):,} ({np.mean(snow_on_steep)*100:.2f}%)")
assert np.sum(snow_on_steep) == 0, "ERROR: Snow found on steep cliffs!"
print("PASSED: Steep cliffs are 100% bare rock!")

# Generate color distribution audit plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), facecolor='#090d16')
ax1.set_facecolor('#0f172a')
ax2.set_facecolor('#0f172a')

# Plot 45 grey distribution
ax1.bar(range(1, 46), grey_counts, color=greys_45, edgecolor='#475569', lw=0.8)
ax1.set_xlim(0, 46)
ax1.set_title("Distribución de los 45 Tonos de Gris en la Montaña (Píxeles Activos)", color='#38bdf8', fontsize=12, fontweight='bold')
ax1.set_xlabel("Índice de Tono de Gris (1 = Nieve Cumbre, 45 = Basalto Fondo)", color='#94a3b8', fontsize=9.5)
ax1.set_ylabel("Recuento de Píxeles", color='#94a3b8', fontsize=9.5)
ax1.tick_params(colors='#cbd5e1')
ax1.grid(True, color='#334155', linestyle=':', alpha=0.5)

# Plot 15 green distribution
ax2.bar(range(1, 16), green_counts, color=greens_15, edgecolor='#475569', lw=0.8)
ax2.set_xlim(0, 16)
ax2.set_title("Distribución de los 15 Tonos de Verde en el Faldón Sub-Alpino (Píxeles Activos)", color='#34d399', fontsize=12, fontweight='bold')
ax2.set_xlabel("Índice de Tono de Verde (V1 = Liquen Claro, V15 = Humus Profundo)", color='#94a3b8', fontsize=9.5)
ax2.set_ylabel("Recuento de Píxeles", color='#94a3b8', fontsize=9.5)
ax2.tick_params(colors='#cbd5e1')
ax2.grid(True, color='#334155', linestyle=':', alpha=0.5)

plt.tight_layout()
out_plot = os.path.join(out_dir, "render_palette_distribution_histogram.png")
plt.savefig(out_plot, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy2(out_plot, os.path.join(brain_dir, "render_palette_distribution_histogram.png"))
print(f"Saved histogram: {out_plot} and copied to brain")
