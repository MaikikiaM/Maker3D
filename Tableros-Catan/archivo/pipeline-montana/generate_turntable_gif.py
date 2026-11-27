import os
import math
import time
import shutil
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from PIL import Image

print("=== GENERATING 360-DEGREE FULL-COLOR TURNTABLE ANIMATION ===")
t0 = time.time()

out_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

diffuse_4k = np.load('scratch/diffuse_4k_array.npy')
Z_master = np.load('scratch/Z_top_master_2m.npy')
import scipy.interpolate as interp
ny_m, nx_m = Z_master.shape

R = 64.0
APOTHEM = R * math.cos(math.radians(30))

# 512x512 grid for super-fast, smooth 36-frame animation rendering
res = 512
xs = np.linspace(-APOTHEM, APOTHEM, res, dtype=np.float32)
ys = np.linspace(-R, R, res, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)

xs_m = np.linspace(-APOTHEM, APOTHEM, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

pts = np.column_stack((Y.ravel(), X.ravel()))
Z = interp_z(pts).reshape((res, res))

im_diff = Image.fromarray((diffuse_4k * 255.0).astype(np.uint8))
im_diff_res = im_diff.resize((res, res), Image.Resampling.LANCZOS)
diffuse_res = np.array(im_diff_res).astype(np.float32) / 255.0

normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

dx = float(xs[1] - xs[0]); dy = float(ys[1] - ys[0])
gy, gx = np.gradient(Z, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm
Ny = -gy / N_norm
Nz = 1.0 / N_norm

laplace = ndi.laplace(Z)
ao = 1.0 - np.clip(-laplace * 0.40, 0.0, 0.50)

# Balanced Studio Sky-Dome Lighting
L1 = np.array([0.5, -0.5, 0.7], dtype=np.float32); L1 /= np.linalg.norm(L1)
d1 = np.clip(Nx * L1[0] + Ny * L1[1] + Nz * L1[2], 0.0, 1.0)
L2 = np.array([-0.5, 0.4, 0.7], dtype=np.float32); L2 /= np.linalg.norm(L2)
d2 = np.clip(Nx * L2[0] + Ny * L2[1] + Nz * L2[2], 0.0, 1.0)

d_dome = np.clip(0.66 + 0.22 * d1 + 0.12 * d2, 0.0, 1.0)
lit = diffuse_res * d_dome[:, :, None]
lit[dist_edge < 0] = [0.06, 0.08, 0.12]
lit = np.clip(lit, 0.0, 1.0)

# Surface downsampled for fast plot_surface (256x256)
step = 2
surf_x = X[::step, ::step]
surf_y = Y[::step, ::step]
surf_z = Z[::step, ::step]
dist_sub = dist_edge[::step, ::step]

rgba = np.zeros((surf_x.shape[0], surf_x.shape[1], 4))
rgba[:, :, :3] = lit[::step, ::step]
rgba[:, :, 3] = 1.0
rgba[dist_sub < -0.2, 3] = 0.0

# Render 36 frames (every 10 degrees)
n_frames = 36
frame_images = []
os.makedirs('scratch/turntable_frames', exist_ok=True)

print(f"Rendering {n_frames} turntable frames...")
for f_idx in range(n_frames):
    azim_deg = f_idx * (360.0 / n_frames)
    fig = plt.figure(figsize=(7, 6), facecolor='#090d16')
    ax = fig.add_subplot(111, projection='3d', facecolor='#090d16')
    ax.plot_surface(surf_x, surf_y, surf_z, facecolors=rgba, rstride=1, cstride=1, shade=False, antialiased=True)
    ax.view_init(elev=30, azim=azim_deg)
    ax.set_box_aspect([1.0, 1.15, 0.70])
    ax.axis('off')
    plt.tight_layout()
    frame_path = f"scratch/turntable_frames/frame_{f_idx:02d}.png"
    plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='#090d16')
    plt.close()
    
    frame_images.append(Image.open(frame_path))
    if f_idx % 6 == 0:
        print(f"  Frame {f_idx+1}/{n_frames} rendered ({azim_deg:.0f}°)")

# Assemble animated GIF
gif_path = os.path.join(out_dir, "catan_mountain_turntable_360.gif")
print("Assembling animated GIF...")
frame_images[0].save(
    gif_path,
    save_all=True,
    append_images=frame_images[1:],
    duration=100, # 100ms per frame = 10 fps (2.4s per full rotation)
    loop=0,
    optimize=True
)
print(f"Saved 360-degree Turntable GIF: {gif_path} ({os.path.getsize(gif_path)/1024/1024:.2f} MB)")

shutil.copy(gif_path, os.path.join(brain_dir, os.path.basename(gif_path)))
print(f"Turntable generation completed in {time.time()-t0:.2f}s")
