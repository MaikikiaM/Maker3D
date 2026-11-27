import os
import math
import time
import shutil
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from PIL import Image

print("=== GENERATING FULL-COLOR ULTRA-HD PRESENTATION SUITE ===")
t0 = time.time()

out_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"
os.makedirs(out_dir, exist_ok=True)

# Load 4K diffuse texture
diffuse_4k = np.load('scratch/diffuse_4k_array.npy') # shape (4096, 4096, 3)
dim = diffuse_4k.shape[0]

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
xs = np.linspace(-APOTHEM, APOTHEM, dim, dtype=np.float32)
ys = np.linspace(-R, R, dim, dtype=np.float32)

# Load Z master 2M
Z_master = np.load('scratch/Z_top_master_2m.npy')
import scipy.interpolate as interp
ny_m, nx_m = Z_master.shape
xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

# Resample down to 2048x2048 for fast 3D shading & rendering
res_render = 2048
xs_r = np.linspace(-APOTHEM, APOTHEM, res_render, dtype=np.float32)
ys_r = np.linspace(-R, R, res_render, dtype=np.float32)
X_r, Y_r = np.meshgrid(xs_r, ys_r)
pts_r = np.column_stack((Y_r.ravel(), X_r.ravel()))
Z_r = interp_z(pts_r).reshape((res_render, res_render))

# Downsample diffuse texture to 2048x2048
im_diff = Image.fromarray((diffuse_4k * 255.0).astype(np.uint8))
im_diff_2k = im_diff.resize((res_render, res_render), Image.Resampling.LANCZOS)
diffuse_2k = np.array(im_diff_2k).astype(np.float32) / 255.0

# Distance to hex
normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge_r = np.full_like(X_r, 999.0)
for nx_h, ny_h in normals:
    dist_edge_r = np.minimum(dist_edge_r, APOTHEM - (X_r * nx_h + Y_r * ny_h))

# Shading: Normals & Lighting
dx = float(xs_r[1] - xs_r[0]); dy = float(ys_r[1] - ys_r[0])
gy, gx = np.gradient(Z_r, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm
Ny = -gy / N_norm
Nz = 1.0 / N_norm

laplace = ndi.laplace(Z_r)
ao = 1.0 - np.clip(-laplace * 0.40, 0.0, 0.50)

# Balanced Studio Sky-Dome Lighting (Luminous alpine lighting on all faces)
L1 = np.array([0.5, -0.5, 0.7], dtype=np.float32); L1 /= np.linalg.norm(L1)
d1 = np.clip(Nx * L1[0] + Ny * L1[1] + Nz * L1[2], 0.0, 1.0)
L2 = np.array([-0.5, 0.4, 0.7], dtype=np.float32); L2 /= np.linalg.norm(L2)
d2 = np.clip(Nx * L2[0] + Ny * L2[1] + Nz * L2[2], 0.0, 1.0)

d_dome = np.clip(0.66 + 0.22 * d1 + 0.12 * d2, 0.0, 1.0)
lit = diffuse_2k * d_dome[:, :, None]
lit[dist_edge_r < 0] = [0.06, 0.08, 0.12]
lit = np.clip(lit, 0.0, 1.0)

# -------------------------------------------------------------
# 1. TOP ORTHOGRAPHIC FULL-COLOR MAP (NO STEPPING)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10), facecolor='#090d16')
ax.imshow(lit, origin='lower')
ax.set_title("Full-Color Continuo: Vista Cenital Fotorrealista (Espectro Infinito)", color='#38bdf8', fontsize=13, fontweight='bold', pad=12)
ax.axis('off')
top_path = os.path.join(out_dir, "render_mountain_top_ortho.png")
plt.tight_layout()
plt.savefig(top_path, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()
print(f"Saved: {top_path}")

# -------------------------------------------------------------
# 2. 3D ISOMETRIC FULL-COLOR RENDER
# -------------------------------------------------------------
fig = plt.figure(figsize=(13, 11), facecolor='#090d16')
ax = fig.add_subplot(111, projection='3d', facecolor='#090d16')
step = 4
surf_x = X_r[::step, ::step]; surf_y = Y_r[::step, ::step]; surf_z = Z_r[::step, ::step]
rgba_sub = np.zeros((surf_x.shape[0], surf_x.shape[1], 4))
rgba_sub[:, :, :3] = lit[::step, ::step]
rgba_sub[:, :, 3] = 1.0
dist_sub = dist_edge_r[::step, ::step]
rgba_sub[dist_sub < -0.2, 3] = 0.0

ax.plot_surface(surf_x, surf_y, surf_z, facecolors=rgba_sub, rstride=1, cstride=1, shade=False, antialiased=True)
ax.view_init(elev=33, azim=-52)
ax.set_box_aspect([1.0, 1.15, 0.72])
ax.set_title("Loseta Montaña Catan - Render Isométrico Full-Color Infinito (83.00mm)", fontsize=14, fontweight='bold', color='#38bdf8', pad=14)
ax.axis('off')
iso_path = os.path.join(out_dir, "render_mountain_isometric.png")
plt.tight_layout()
plt.savefig(iso_path, dpi=220, bbox_inches='tight', facecolor='#090d16')
plt.close()
print(f"Saved: {iso_path}")

# -------------------------------------------------------------
# 3. MACRO ZOOM COMPARISON: ZERO PIXELS / ORGANIC TRANSITIONS
# -------------------------------------------------------------
# Crop 1: Summit couloir / snow transition (Direct answer to user image 1)
# Center around main horn: X = -4.0, Y = 6.0
ix_summit = int(np.round((-4.0 - (-APOTHEM)) / (2 * APOTHEM) * (res_render - 1)))
iy_summit = int(np.round((6.0 - (-R)) / (2 * R) * (res_render - 1)))
radius_crop = 160
crop_summit = lit[iy_summit-radius_crop : iy_summit+radius_crop, ix_summit-radius_crop : ix_summit+radius_crop]

# Crop 2: Tundra / lichen transition (Direct answer to user image 2)
# Center around south-east valley meadow: X = 12.0, Y = -22.0
ix_tundra = int(np.round((12.0 - (-APOTHEM)) / (2 * APOTHEM) * (res_render - 1)))
iy_tundra = int(np.round((-22.0 - (-R)) / (2 * R) * (res_render - 1)))
crop_tundra = lit[iy_tundra-radius_crop : iy_tundra+radius_crop, ix_tundra-radius_crop : ix_tundra+radius_crop]

fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#090d16')
axes[0].imshow(crop_summit, origin='lower', interpolation='bicubic')
axes[0].set_title("Super-Zoom Cumbre: Transición Nieve / Hielo / Granito (Cero Píxeles)", color='#38bdf8', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(crop_tundra, origin='lower', interpolation='bicubic')
axes[1].set_title("Super-Zoom Tundra: Difuminado Orgánico de Líquenes (Cero Bordes Poligonales)", color='#38bdf8', fontsize=12, fontweight='bold')
axes[1].axis('off')

macro_path = os.path.join(out_dir, "render_mountain_macro_zoom_proof.png")
plt.tight_layout()
plt.savefig(macro_path, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()
print(f"Saved: {macro_path}")

# Copy to brain artifact directory
for p in [top_path, iso_path, macro_path]:
    shutil.copy(p, os.path.join(brain_dir, os.path.basename(p)))
print("Copied all full-color presentation renders to brain artifact folder!")
print(f"Render suite generated in {time.time()-t0:.2f}s.")
