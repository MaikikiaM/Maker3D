import math
import os
import shutil
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

print("=== GENERATING MACRO SUMMIT & 4-VIEW PRESENTATION SHEET ===")

out_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

Z_top = np.load('scratch/Z_top_master_final.npy')
tool_map = np.load('scratch/tool_map_master_final.npy')
ny, nx = Z_top.shape

R = 64.0
APOTHEM = R * math.cos(math.radians(30))
xs = np.linspace(-APOTHEM, APOTHEM, nx)
ys = np.linspace(-R, R, ny)
X, Y = np.meshgrid(xs, ys)

normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
gy, gx = np.gradient(Z_top, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm
Ny = -gy / N_norm
Nz = 1.0 / N_norm

laplace = ndi.laplace(Z_top)
ao = 1.0 - np.clip(-laplace * 0.45, 0.0, 0.55)

# Sun & Sky
az_sun = 310.0 * math.pi / 180.0; alt_sun = 46.0 * math.pi / 180.0
Lx_sun = math.cos(alt_sun) * math.sin(az_sun); Ly_sun = math.cos(alt_sun) * math.cos(az_sun); Lz_sun = math.sin(alt_sun)
diff_sun = np.clip(Nx * Lx_sun + Ny * Ly_sun + Nz * Lz_sun, 0.0, 1.0)

az_sky = 130.0 * math.pi / 180.0; alt_sky = 60.0 * math.pi / 180.0
Lx_sky = math.cos(alt_sky) * math.sin(az_sky); Ly_sky = math.cos(alt_sky) * math.cos(az_sky); Lz_sky = math.sin(alt_sky)
diff_sky = np.clip(Nx * Lx_sky + Ny * Ly_sky + Nz * Lz_sky, 0.0, 1.0)

tool_colors = {
    0: np.array([0.14, 0.16, 0.19]),
    1: np.array([0.48, 0.74, 0.88]),
    3: np.array([0.22, 0.24, 0.27]),
    4: np.array([0.62, 0.64, 0.67]),
    5: np.array([0.32, 0.54, 0.28]),
    7: np.array([0.96, 0.97, 0.99])
}

albedo = np.zeros((ny, nx, 3), dtype=np.float32)
for tid, c in tool_colors.items():
    albedo[tool_map == tid] = c

sun_color = np.array([1.05, 1.00, 0.92])
sky_color = np.array([0.68, 0.82, 0.98])
amb_color = np.array([0.25, 0.28, 0.35])

lit = albedo * (
    diff_sun[:, :, None] * sun_color[None, None, :] * 0.75 +
    diff_sky[:, :, None] * sky_color[None, None, :] * 0.35 +
    amb_color[None, None, :] * ao[:, :, None] * 0.40
)
lit[dist_edge < 0] = [0.06, 0.08, 0.12]
lit = np.clip(lit, 0.0, 1.0)

# -------------------------------------------------------------
# 4-VIEW MASTER PRESENTATION SHEET
# -------------------------------------------------------------
fig = plt.figure(figsize=(16, 14), facecolor='#090d16')

# View 1: Isometric CAD
ax1 = fig.add_subplot(221, projection='3d', facecolor='#090d16')
step = 2
surf_x = X[::step, ::step]; surf_y = Y[::step, ::step]; surf_z = Z_top[::step, ::step]
rgba_sub = np.zeros((surf_x.shape[0], surf_x.shape[1], 4))
rgba_sub[:, :, :3] = lit[::step, ::step]
rgba_sub[:, :, 3] = 1.0
dist_sub = dist_edge[::step, ::step]
rgba_sub[dist_sub < -0.2, 3] = 0.0
ax1.plot_surface(surf_x, surf_y, surf_z, facecolors=rgba_sub, rstride=1, cstride=1, shade=False, antialiased=True)
ax1.view_init(elev=32, azim=-52)
ax1.set_box_aspect([1.0, 1.15, 0.72])
ax1.set_title("1. Vista Isométrica Alpina (Matterhorn 83mm)", color='#38bdf8', fontsize=12, fontweight='bold')
ax1.axis('off')

# View 2: North Face Profile
ax2 = fig.add_subplot(222, projection='3d', facecolor='#090d16')
ax2.plot_surface(surf_x, surf_y, surf_z, facecolors=rgba_sub, rstride=1, cstride=1, shade=False, antialiased=True)
ax2.view_init(elev=16, azim=-175)
ax2.set_box_aspect([1.0, 1.15, 0.72])
ax2.set_title("2. Cara Norte (Canales de Nieve / Couloirs)", color='#38bdf8', fontsize=12, fontweight='bold')
ax2.axis('off')

# View 3: Top Orthographic Map
ax3 = fig.add_subplot(223, facecolor='#090d16')
ax3.imshow(lit, origin='lower')
ax3.set_title("3. Mapa Geológico Cenital (CMYKWOG-B)", color='#38bdf8', fontsize=12, fontweight='bold')
ax3.axis('off')

# View 4: Solid Cross Section
ax4 = fig.add_subplot(224, facecolor='#0f172a')
y_peak_idx = int(np.round((6.0 - (-R)) / (2 * R) * (ny - 1)))
x_slice = xs
z_slice = Z_top[y_peak_idx, :]
ax4.fill_between(x_slice, 0, z_slice, color='#334155', alpha=0.95, label='Núcleo 100% Sólido')
ax4.fill_between(x_slice, 0, 25.0, color='#1e293b', alpha=0.98, label='Zócalo Base 25mm')
ax4.plot(x_slice, z_slice, color='#f8fafc', lw=2.2, label='Perfil Cumbre')
ax4.axhline(25.0, color='#38bdf8', linestyle='--', lw=1.2, label='Plano Asentamiento Catan')
ax4.fill_between([-5.1, 5.1], 0, 5.2, color='#020617', hatch='//', edgecolor='#94a3b8', label='Imán Ø10.2x5.2mm')
ax4.scatter([-4.0], [83.0], color='#fbbf24', s=80, zorder=5)
ax4.set_ylim(-2, 90); ax4.set_xlim(-58, 58)
ax4.set_ylabel("Altura Z (mm)", color='#e2e8f0', fontsize=10)
ax4.tick_params(colors='#94a3b8')
ax4.set_title("4. Corte Transversal Z (100% Macizo Watertight)", color='#38bdf8', fontsize=12, fontweight='bold')
ax4.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', loc='upper right', fontsize=8)
ax4.grid(True, color='#334155', linestyle=':', alpha=0.6)

plt.suptitle("LOSETA MONTAÑA CATAN 3D - SUITE MAESTRA MULTI-MATERIAL (INFIMECH MX PRO)", color='#38bdf8', fontsize=14, fontweight='bold', y=0.98)
quad_path = os.path.join(out_dir, "render_mountain_master_suite_4view.png")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(quad_path, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()
print(f"Saved: {quad_path}")

shutil.copy(quad_path, os.path.join(brain_dir, os.path.basename(quad_path)))
print("Copied master quad render to brain artifact folder!")
