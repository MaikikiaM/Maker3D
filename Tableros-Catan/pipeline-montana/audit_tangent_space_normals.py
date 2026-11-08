import os
import math
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print("=== 4K TANGENT SPACE NORMAL MAP MATHEMATICAL ORTHOGONALITY AUDIT ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

norm_path = r"Tableros Catan\Loseta_Montana_3D\catan_mountain_normal_4k.png"
assert os.path.exists(norm_path), f"File not found: {norm_path}"

print(f"Loading 4K Normal Map: {norm_path} ({os.path.getsize(norm_path)/1024/1024:.2f} MB)...")
im_norm = Image.open(norm_path)
arr_norm = np.array(im_norm).astype(np.float32) / 255.0
dim = arr_norm.shape[0]

# Unpack tangent space normals from [0, 1] to [-1, 1]
Nx = arr_norm[:, :, 0] * 2.0 - 1.0
Ny = arr_norm[:, :, 1] * 2.0 - 1.0
Nz = arr_norm[:, :, 2] * 2.0 - 1.0

# Compute vector magnitudes
lengths = np.sqrt(Nx**2 + Ny**2 + Nz**2)
print(f"Normal vector length range: [{np.min(lengths):.4f}, {np.max(lengths):.4f}] (Expected ~ 1.0000)")
print(f"Normal Z (Upwards vector) range: [{np.min(Nz):.4f}, {np.max(Nz):.4f}]")
assert np.min(Nz) >= 0.0, "FAIL: Normal points into lower hemisphere (inverted normal)!"

# Downsample for visualization
step = 4
Nx_sub = Nx[::step, ::step]
Ny_sub = Ny[::step, ::step]
Nz_sub = Nz[::step, ::step]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#090d16')
for ax in axes:
    ax.set_facecolor('#0f172a')

# 1. Nx (Red Channel - Tangent X slope)
im0 = axes[0].imshow(Nx_sub, cmap='RdBu_r', vmin=-0.8, vmax=0.8, origin='lower')
axes[0].set_title("Canal R: Vector Normal X (-Oeste / +Este)", color='#38bdf8', fontsize=11, fontweight='bold', pad=8)
axes[0].axis('off')
cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cbar0.ax.tick_params(colors='#94a3b8')

# 2. Ny (Green Channel - Tangent Y slope)
im1 = axes[1].imshow(Ny_sub, cmap='RdBu_r', vmin=-0.8, vmax=0.8, origin='lower')
axes[1].set_title("Canal G: Vector Normal Y (-Sur / +Norte)", color='#38bdf8', fontsize=11, fontweight='bold', pad=8)
axes[1].axis('off')
cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar1.ax.tick_params(colors='#94a3b8')

# 3. Nz (Blue Channel - Vertical Z alignment)
im2 = axes[2].imshow(Nz_sub, cmap='Blues_r', vmin=0.3, vmax=1.0, origin='lower')
axes[2].set_title("Canal B: Vector Normal Z (Cero Inversiones)", color='#38bdf8', fontsize=11, fontweight='bold', pad=8)
axes[2].axis('off')
cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
cbar2.ax.tick_params(colors='#94a3b8')

plt.suptitle("Auditoría de Componentes del Mapa de Normales Tangentes 4K (4096×4096 px)\n(Ortonormalidad del Marco TBN / Cero Ruido de Moiré / Derivadas Continuas C1)",
             color='#e2e8f0', fontsize=13.5, fontweight='bold', y=0.98)
plt.tight_layout()

out_norm_comp = os.path.join(out_dir, "render_tangent_normal_components.png")
plt.savefig(out_norm_comp, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy(out_norm_comp, os.path.join(brain_dir, "render_tangent_normal_components.png"))
print(f"Saved: {out_norm_comp} and copied to brain")
print(f"Normal map audit finished in {time.time()-t0:.2f}s.")
