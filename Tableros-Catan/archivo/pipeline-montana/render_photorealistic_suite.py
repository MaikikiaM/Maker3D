import math
import os
import shutil
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

print("=== RENDERING ULTRA-HD CAD PRESENTATION SUITE ===")

out_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"
os.makedirs(out_dir, exist_ok=True)

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

# Calculate accurate surface normals
dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
gy, gx = np.gradient(Z_top, dy, dx)
N_norm = np.sqrt(gx**2 + gy**2 + 1.0)
Nx = -gx / N_norm
Ny = -gy / N_norm
Nz = 1.0 / N_norm

# Ambient Occlusion approximation using multiscale curvature / laplacian
laplace = ndi.laplace(Z_top)
ao = 1.0 - np.clip(-laplace * 0.45, 0.0, 0.55)
# Additional soft occlusion in deep valley hollows
ao = ao * (0.82 + 0.18 * np.clip((Z_top - 25.0) / 30.0, 0.0, 1.0))

# Dual Light Setup:
# Sun: Warm golden alpine sunlight (Azimuth 310°, Elevation 46°)
az_sun = 310.0 * math.pi / 180.0
alt_sun = 46.0 * math.pi / 180.0
Lx_sun = math.cos(alt_sun) * math.sin(az_sun)
Ly_sun = math.cos(alt_sun) * math.cos(az_sun)
Lz_sun = math.sin(alt_sun)

# Sky Fill: Cool alpine blue skylight from opposite hemisphere (Azimuth 130°, Elevation 60°)
az_sky = 130.0 * math.pi / 180.0
alt_sky = 60.0 * math.pi / 180.0
Lx_sky = math.cos(alt_sky) * math.sin(az_sky)
Ly_sky = math.cos(alt_sky) * math.cos(az_sky)
Lz_sky = math.sin(alt_sky)

diff_sun = np.clip(Nx * Lx_sun + Ny * Ly_sun + Nz * Lz_sun, 0.0, 1.0)
diff_sky = np.clip(Nx * Lx_sky + Ny * Ly_sky + Nz * Lz_sky, 0.0, 1.0)

# Colors for tools (High dynamic range realistic palette)
tool_colors = {
    0: np.array([0.14, 0.16, 0.19]),  # Base Zócalo / Road shelf
    1: np.array([0.48, 0.74, 0.88]),  # Glacial Firn Cyan
    3: np.array([0.22, 0.24, 0.27]),  # Slate Shadow Cliffs
    4: np.array([0.62, 0.64, 0.67]),  # Sunlit Granite Rock
    5: np.array([0.32, 0.54, 0.28]),  # Alpine Tundra
    7: np.array([0.96, 0.97, 0.99])   # Pure Alpine Snow
}

albedo = np.zeros((ny, nx, 3), dtype=np.float32)
for tid, c in tool_colors.items():
    albedo[tool_map == tid] = c

# Shaded composite with warm sun + cool skylight + ambient occlusion
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
# 1. TOP ORTHOGRAPHIC GEOLOGICAL MAP
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10), facecolor='#090d16')
ax.imshow(lit, origin='lower')
ax.set_title("Pintura 3D Alpina CMYKWOG: Vista Cenital Geológica (Zero Vetas Rojas)", color='#38bdf8', fontsize=13, fontweight='bold', pad=12)
ax.axis('off')
# Add legend annotations
fig.text(0.5, 0.03, "Blanco (T7 Nieve 0.2mm) | Gris (T4 Granito) | Cian (T1 Glaciar) | Carbón (T3 Pizarra) | Verde (T5 Tundra) | Negro (T0 Base)",
         color='#94a3b8', ha='center', fontsize=9.5)
top_path = os.path.join(out_dir, "render_mountain_top_ortho.png")
plt.tight_layout()
plt.savefig(top_path, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()
print(f"Saved: {top_path}")

# -------------------------------------------------------------
# 2. 3D ISOMETRIC PRESENTATION RENDER
# -------------------------------------------------------------
fig = plt.figure(figsize=(13, 11), facecolor='#090d16')
ax = fig.add_subplot(111, projection='3d', facecolor='#090d16')
step = 2
surf_x = X[::step, ::step]
surf_y = Y[::step, ::step]
surf_z = Z_top[::step, ::step]
lit_sub = lit[::step, ::step]
# Add alpha channel
rgba_sub = np.zeros((surf_x.shape[0], surf_x.shape[1], 4))
rgba_sub[:, :, :3] = lit_sub
rgba_sub[:, :, 3] = 1.0
dist_sub = dist_edge[::step, ::step]
rgba_sub[dist_sub < -0.2, 3] = 0.0 # Transparent outside hex

ax.plot_surface(surf_x, surf_y, surf_z, facecolors=rgba_sub, rstride=1, cstride=1, shade=False, antialiased=True)
ax.view_init(elev=33, azim=-52)
ax.set_box_aspect([1.0, 1.15, 0.72])
ax.set_title("Loseta Montaña Catan - Cumbre Matterhorn 83.00mm (Infimech CMYKWOG-B)", fontsize=14, fontweight='bold', color='#38bdf8', pad=14)
ax.axis('off')
iso_path = os.path.join(out_dir, "render_mountain_isometric.png")
plt.tight_layout()
plt.savefig(iso_path, dpi=220, bbox_inches='tight', facecolor='#090d16')
plt.close()
print(f"Saved: {iso_path}")

# -------------------------------------------------------------
# 3. SOLID CROSS SECTION & TOPOGRAPHIC PROFILE
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), facecolor='#090d16', gridspec_kw={'height_ratios': [1.4, 1.0]})

# Cross section along Y=6.0 (Apex)
y_peak_idx = int(np.round((6.0 - (-R)) / (2 * R) * (ny - 1)))
x_slice = xs
z_slice = Z_top[y_peak_idx, :]

ax1.set_facecolor('#0f172a')
ax1.fill_between(x_slice, 0, z_slice, color='#334155', alpha=0.95, label='Núcleo Macizo 100% Sólido (Granito Oscuro)')
ax1.fill_between(x_slice, 0, 25.0, color='#1e293b', alpha=0.98, label='Zócalo Base Hexagonal (Tool B - 25.0mm)')
ax1.plot(x_slice, z_slice, color='#f8fafc', lw=2.5, label='Arista y Cumbre Principal (83.00 mm)')
ax1.axhline(25.0, color='#38bdf8', linestyle='--', lw=1.5, label='Plano Asentamiento Catan (Z=25.0 mm)')
ax1.fill_between([-5.1, 5.1], 0, 5.2, color='#020617', hatch='//', edgecolor='#94a3b8', label='Alojamiento Imán Neodimio Ø10.2 x 5.2mm')
ax1.scatter([-4.0], [83.0], color='#fbbf24', s=90, zorder=5, label='Vértice Cimero Apex (-4.0, 6.0, 83.00mm)')
ax1.set_ylim(-2, 90)
ax1.set_xlim(-58, 58)
ax1.set_ylabel("Altura Z (mm)", color='#e2e8f0', fontsize=11)
ax1.tick_params(colors='#94a3b8')
ax1.set_title("Corte Transversal Z por la Cumbre: Cuerpo Macizo Monolítico Watertight", color='#38bdf8', fontsize=13, fontweight='bold')
ax1.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', loc='upper right', fontsize=8.5)
ax1.grid(True, color='#334155', linestyle=':', alpha=0.6)

# Elevation distribution histogram & tool breakdown
ax2.set_facecolor('#0f172a')
in_hex = dist_edge >= 0
z_relief = Z_top[in_hex] - 25.0
counts, bins, _ = ax2.hist(z_relief, bins=45, color='#38bdf8', alpha=0.75, edgecolor='#0284c7', density=True)
ax2.set_xlabel("Relieve sobre el Zócalo Base (mm)", color='#e2e8f0', fontsize=11)
ax2.set_ylabel("Densidad de Superficie", color='#e2e8f0', fontsize=11)
ax2.tick_params(colors='#94a3b8')
ax2.set_title("Distribución Altimétrica Progresiva (Transición Suave Piramidal - Cero Panettone)", color='#38bdf8', fontsize=11, fontweight='bold')
ax2.grid(True, color='#334155', linestyle=':', alpha=0.6)

cross_path = os.path.join(out_dir, "render_solid_cross_section.png")
plt.tight_layout()
plt.savefig(cross_path, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()
print(f"Saved: {cross_path}")

# Copy all to brain artifact folder
for p in [top_path, iso_path, cross_path]:
    shutil.copy(p, os.path.join(brain_dir, os.path.basename(p)))
print("Copied presentation renders to brain artifact folder!")
print("=== RENDER SUITE COMPLETE ===")
