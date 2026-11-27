import math
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

print("=== TESTING TRUE ALPINE PYRAMIDAL SKELETON (CONTINUOUS SLOPE) ===")

R = 64.0
APOTHEM = R * math.cos(math.radians(30)) # 55.4256 mm
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF_H = Z_PEAK - Z_BASE # 58.0 mm

# High-resolution grid
nx, ny = 401, 463
xs = np.linspace(-APOTHEM, APOTHEM, nx)
ys = np.linspace(-R, R, ny)
X, Y = np.meshgrid(xs, ys)

normals = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
dist_edge = np.full_like(X, 999.0)
for nx_h, ny_h in normals:
    d = APOTHEM - (X * nx_h + Y * ny_h)
    dist_edge = np.minimum(dist_edge, d)

v_angles = [(30 + 60 * k) * math.pi / 180 for k in range(6)]
hex_corners = np.array([[R * math.cos(a), R * math.sin(a)] for a in v_angles])
dist_vert = np.full_like(X, 999.0)
for vx, vy in hex_corners:
    dist_vert = np.minimum(dist_vert, np.hypot(X - vx, Y - vy))

# 1. MACRO-GEOMETRY: The Soaring Pyramidal Massif
r = np.hypot(X, Y)
R_max = 52.0
r_norm = np.clip(r / R_max, 0.0, 1.0)
base_pyramid = (1.0 - r_norm)**1.65

# 2. Asymmetric Pyramidal Ridges
theta = np.arctan2(Y, X)

def ridge_profile(theta, target_angle, sharpness=4.5):
    d_th = np.arctan2(np.sin(theta - target_angle), np.cos(theta - target_angle))
    return np.exp(-0.5 * (d_th * sharpness)**2.0)

r_nw = ridge_profile(theta, 2.35, sharpness=4.5)
r_ne = ridge_profile(theta, 0.78, sharpness=5.0)
r_se = ridge_profile(theta, -1.05, sharpness=4.2)
r_sw = ridge_profile(theta, -2.40, sharpness=4.8)

ridge_field = 0.55 * r_nw + 0.50 * r_ne + 0.45 * r_se + 0.48 * r_sw
profile_power = 1.35 + 0.85 * (1.0 - np.clip(ridge_field, 0, 1))
massif_macro = (1.0 - r_norm)**profile_power

# 3. Secondary Peaks / Spines (Cathedral Spires)
r_p2 = np.hypot(X - 14.0, Y - 12.0)
spire_east = 0.72 * np.exp(-(r_p2 / 10.0)**1.5)

r_p3 = np.hypot(X - (-12.0), Y - (-14.0))
spire_south = 0.62 * np.exp(-(r_p3 / 11.0)**1.4)

r_p4 = np.hypot(X - (-16.0), Y - 6.0)
spire_west = 0.55 * np.exp(-(r_p4 / 9.0)**1.5)

macro_h = np.maximum(massif_macro, spire_east)
macro_h = np.maximum(macro_h, spire_south)
macro_h = np.maximum(macro_h, spire_west)

# 4. Realistic Alpine Cirque Scours
cirque_north = 0.25 * np.exp(-((X - 2.0)**2 + (Y - 22.0)**2) / 120.0)
cirque_east  = 0.22 * np.exp(-((X - 24.0)**2 + (Y - (-2.0))**2) / 110.0)
cirque_south = 0.28 * np.exp(-((X - (-2.0))**2 + (Y - (-24.0))**2) / 140.0)
cirque_west  = 0.24 * np.exp(-((X - (-22.0))**2 + (Y - 0.0)**2) / 110.0)

macro_h = np.clip(macro_h - (cirque_north + cirque_east + cirque_south + cirque_west), 0.0, 1.0)
macro_h = macro_h / macro_h.max()

# 5. Continuous Multi-Scale Rock Micro-Texture
noise = np.zeros_like(X)
harmonics = [
    (0.08, 3.2, 0.35),
    (0.18, 1.8, 1.12),
    (0.38, 0.9, 2.21),
    (0.85, 0.45, 0.74),
    (1.80, 0.22, 1.89),
    (3.60, 0.08, 2.85)
]

for freq, amp, rot in harmonics:
    xr = X * math.cos(rot) - Y * math.sin(rot)
    yr = X * math.sin(rot) + Y * math.cos(rot)
    crag = np.abs(np.sin(xr * freq + 0.4)) * np.cos(yr * freq * 0.9 + 0.8)
    crag_sharp = (1.0 - np.abs(crag))**2.0
    noise += amp * (crag_sharp - 0.5)

strata = 0.65 * np.sin((X * 0.25 + Y * 0.15 + macro_h * 15.0) * 1.6)

rock_texture = (noise + strata) * (macro_h**0.8) * 0.035
total_relief = np.clip(macro_h + rock_texture, 0.0, 1.0)
total_relief = total_relief / total_relief.max() * RELIEF_H

# 6. Road & Vertex Clearance
def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

edge_clearance = smoothstep(3.5, 12.0, dist_edge)
vert_clearance = smoothstep(10.0, 15.0, dist_vert)
clearance = edge_clearance * vert_clearance

final_relief = total_relief * clearance
Z_top = Z_BASE + final_relief
Z_top[dist_edge < 0] = Z_BASE

dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
gy, gx = np.gradient(Z_top, dy, dx)
az = 315.0 * np.pi / 180.0; alt = 48.0 * np.pi / 180.0
sun_x = np.cos(alt)*np.sin(az); sun_y = np.cos(alt)*np.cos(az); sun_z = np.sin(alt)
norm = np.sqrt(gx**2 + gy**2 + 1.0)
hs = np.clip((-gx/norm)*sun_x + (-gy/norm)*sun_y + (1.0/norm)*sun_z, 0.0, 1.0)

fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor='#090d16')

axes[0].imshow(Z_top, cmap='terrain', origin='lower')
axes[0].set_title('Topografía de Macizo Piramidal (Z 25-83mm)', color='white')
axes[0].axis('off')

axes[1].imshow(hs, cmap='gray', origin='lower')
axes[1].set_title('Sombreado Continuo (Zero Píxeles)', color='white')
axes[1].axis('off')

mid_y = ny // 2
axes[2].plot(xs, Z_top[mid_y, :], color='#38bdf8', lw=2.5, label='Perfil E-W')
axes[2].axhline(25.0, color='#94a3b8', linestyle='--', label='Zócalo Base (25mm)')
axes[2].set_ylim(20, 90)
axes[2].set_title('Perfil Transversal (Pendiente Piramidal Cóncava Natural)', color='white')
axes[2].set_facecolor('#0f172a')
axes[2].tick_params(colors='white')
axes[2].legend(facecolor='#1e293b', labelcolor='white')
axes[2].grid(True, color='#334155', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('scratch/true_pyramid_preview.png', dpi=200, facecolor='#090d16')
print("Saved scratch/true_pyramid_preview.png")
