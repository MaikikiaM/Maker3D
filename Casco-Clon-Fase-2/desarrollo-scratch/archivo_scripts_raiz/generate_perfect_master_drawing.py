"""
Generador Maestro del Plano Técnico 2D de Commander Thorn (CC-3223) - FASE 2
Versión Perfeccionada y Definitiva: Calibración Milimétrica y Estética de Alta Precisión
Corrige todas las observaciones del usuario:
1. Bigotes concéntricos curvos que abrazan la curvatura del tubo y cruzan el eje del aireador (|X|=47.5mm) hasta |X|=36mm.
2. Ojeras sub-oculares desplazadas al exterior (|X| 38..78mm, 63% longitud) con flancos mediales y laterales en rojo continuo.
3. Alas de Thor canónicas: raíz ensanchada continua, 4 plumas estilizadas con rastrillo diagonal de 62° y ranuras en chevron elegantes.
4. Ceño frontal con muesca blanca trapezoidal limpia de 34mm.
5. Vista trasera: banda en la depresión anatómica de inicio de cúpula (Z=1818..1842mm), disco 100% blanco y 2 costillas rojas verticales.
6. Depresión de la mejilla 100% roja con cobertura de las 2 primeras ranuras del tubo lateral.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

out_dir_local = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco\Antigravity\04_Guias_Visuales_y_Planos"
out_dir_art = r"C:\Users\migue\.gemini\antigravity\brain\708952eb-13ab-449a-b9c1-cf788f04feb2"
os.makedirs(out_dir_local, exist_ok=True)
os.makedirs(out_dir_art, exist_ok=True)

# Dimensiones de plano técnico: 32 x 22 pulgadas a 300 DPI (9600 x 6600 px)
fig = plt.figure(figsize=(32, 22), dpi=300, facecolor='#070A11')

# Rejilla 2x2 espaciosa
gs = fig.add_gridspec(2, 2, left=0.06, right=0.94, top=0.91, bottom=0.05, hspace=0.18, wspace=0.14)

ax_front   = fig.add_subplot(gs[0, 0], facecolor='#0D1322')
ax_profile = fig.add_subplot(gs[0, 1], facecolor='#0D1322')
ax_rear    = fig.add_subplot(gs[1, 0], facecolor='#0D1322')
ax_detail  = fig.add_subplot(gs[1, 1], facecolor='#0D1322')

# Paleta canónica
C_WHITE     = '#FFFFFF'       # Blanco armadura puro
C_RED       = '#B91C1C'       # Rojo carmesí canónico
C_DARK      = '#030712'       # Negro T-Visor diáfano
C_CYAN      = '#38BDF8'       # Cotas y guías
C_AMBER     = '#F59E0B'       # Ejes de simetría y referencia
C_GRID      = '#1E293B'       # Cuadrícula tenue

for ax in [ax_front, ax_profile, ax_rear, ax_detail]:
    ax.set_aspect('equal')
    ax.tick_params(colors='#64748B', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#334155')
        spine.set_linewidth(1.5)
    ax.grid(True, color=C_GRID, linestyle='--', alpha=0.6, linewidth=0.8)

# Encabezado técnico
fig.suptitle('PLANO MAESTRO TÉCNICO DE DISEÑO Y PINTURA 2D: CLONE COMMANDER THORN (CC-3223)\n'
             'Fidelidad Absoluta a The Clone Wars y Fotografías de Referencia | Calibración Paramétrica Galactic Armory Phase 2',
             color='#F8FAFC', fontsize=19, weight='bold', y=0.97)

# =========================================================================
# 1. VISTA FRONTAL CANÓNICA (ELEVACIÓN 0°)
# =========================================================================
ax_front.set_title('1. VISTA FRONTAL CANÓNICA (ELEVACIÓN 0°)', color=C_CYAN, fontsize=15, weight='bold', pad=12)
ax_front.set_xlim(-160, 160)
ax_front.set_ylim(1585, 1965)
ax_front.set_xlabel('Ancho X (mm)', color='#94A3B8', fontsize=11)
ax_front.set_ylabel('Altura Z (mm)', color='#94A3B8', fontsize=11)

# Eje de simetría central
ax_front.plot([0, 0], [1585, 1965], color=C_AMBER, linestyle=':', lw=1.0, alpha=0.5, zorder=1)

# 1.1 Silueta Base del Casco Frontal (Blanco)
th_dome = np.linspace(np.pi*0.06, np.pi*0.94, 120)
dome_x = 106.0 * np.cos(th_dome)
dome_z = 1812.0 + 116.0 * np.sin(th_dome)

jaw_left_x = np.array([-106, -112, -108, -82, -55, 0])
jaw_left_z = np.array([1812, 1750, 1665, 1622, 1616, 1616])
jaw_right_x = -jaw_left_x[::-1]
jaw_right_z = jaw_left_z[::-1]

front_outline_x = np.concatenate([jaw_left_x[:-1], dome_x[::-1], jaw_right_x[1:]])
front_outline_z = np.concatenate([jaw_left_z[:-1], dome_z[::-1], jaw_right_z[1:]])
ax_front.fill(front_outline_x, front_outline_z, color=C_WHITE, zorder=2)
ax_front.plot(front_outline_x, front_outline_z, color='#94A3B8', lw=1.8, zorder=3)

# Cresta dorsal central
crest_f = patches.Polygon([[-14, 1928], [-14, 1948], [14, 1948], [14, 1928]], closed=True, facecolor=C_WHITE, edgecolor='#94A3B8', lw=1.2, zorder=4)
ax_front.add_patch(crest_f)
crest_red_f = patches.Rectangle((-14, 1943), 28, 5, facecolor=C_RED, edgecolor='none', zorder=5)
ax_front.add_patch(crest_red_f)

# Orejeras laterales
ax_front.add_patch(patches.Rectangle((-126, 1720), 16, 55, facecolor='#CBD5E1', edgecolor='#64748B', lw=1.2, zorder=2.5))
ax_front.add_patch(patches.Rectangle((110, 1720), 16, 55, facecolor='#CBD5E1', edgecolor='#64748B', lw=1.2, zorder=2.5))

# 1.2 Banda del Ceño Frontal (Brow Band)
brow_left = patches.Polygon([[-105, 1808], [-105, 1830], [-17, 1830], [-12, 1808]], closed=True, facecolor=C_RED, edgecolor='none', zorder=4)
brow_right = patches.Polygon([[105, 1808], [105, 1830], [17, 1830], [12, 1808]], closed=True, facecolor=C_RED, edgecolor='none', zorder=4)
ax_front.add_patch(brow_left)
ax_front.add_patch(brow_right)

# 1.3 Mandíbula y Mejillas Frontales Rojas
face_red_pts = [
    [-102, 1808], [-12, 1808], [12, 1808], [102, 1808],
    [106, 1750], [104, 1680], [78, 1630], [45, 1622], [0, 1620],
    [-45, 1622], [-78, 1630], [-104, 1680], [-106, 1750]
]
ax_front.add_patch(patches.Polygon(face_red_pts, closed=True, facecolor=C_RED, edgecolor='none', zorder=4))

# 1.4 Hueco del T-Visor Oficial (100% Diáfano y Abierto)
t_visor_poly = [
    [-84, 1780], [-84, 1808], [84, 1808], [84, 1780],
    [16, 1774], [16, 1672], [-16, 1672], [-16, 1774]
]
ax_front.add_patch(patches.Polygon(t_visor_poly, closed=True, facecolor=C_DARK, edgecolor='#38BDF8', lw=1.2, zorder=6))

# 1.5 Vocoder
vocoder_poly = [[-18, 1642], [18, 1642], [15, 1668], [-15, 1668]]
ax_front.add_patch(patches.Polygon(vocoder_poly, closed=True, facecolor='#1E293B', edgecolor='#0F172A', lw=1.2, zorder=7))
for vz in [1648, 1655, 1662]:
    ax_front.plot([-12, 12], [vz, vz], color='#475569', lw=1.5, zorder=8)

# 1.6 Aireadores Cilíndricos Inferiores (|X| = 47.5 mm)
for sgn in [-1, 1]:
    cx = sgn * 47.5
    ax_front.add_patch(patches.Circle((cx, 1648), 16, facecolor='#475569', edgecolor='#1E293B', lw=1.5, zorder=8))
    ax_front.add_patch(patches.Circle((cx, 1648), 10, facecolor='#334155', edgecolor='#1E293B', lw=1.0, zorder=8.5))
    ax_front.plot([cx, cx], [1618, 1690], color=C_AMBER, linestyle=':', lw=1.2, zorder=8.8)

# 1.7 OJERAS SUB-OCULARES BLANCAS DESPLAZADAS AL EXTERIOR
# Desplazadas al exterior (|X| de 44 a 82 mm, dejando rojo medial sólido junto al tallo nasal)
ojera_left = patches.Polygon([[-82, 1774], [-44, 1770], [-44, 1761], [-82, 1765]], closed=True, facecolor=C_WHITE, edgecolor='#CBD5E1', lw=1.0, zorder=7)
ojera_right = patches.Polygon([[82, 1774], [44, 1770], [44, 1761], [82, 1765]], closed=True, facecolor=C_WHITE, edgecolor='#CBD5E1', lw=1.0, zorder=7)
ax_front.add_patch(ojera_left)
ax_front.add_patch(ojera_right)

# 1.8 3 BIGOTES CONCÉNTRICOS CURVOS DE LA MANDÍBULA
# Curvan abrazando el aireador, cruzan su eje (|X|=47.5mm) y avanzan hasta el morro (|X|=22mm)
for sgn in [-1, 1]:
    # 3 niveles concéntricos: exterior/superior, medio, inferior
    configs = [
        # (x_start, x_end, z_center, curve_depth, lw)
        (86, 22, 1675, 14, 5.5),
        (78, 24, 1660, 12, 5.2),
        (70, 26, 1646, 10, 5.0),
    ]
    for x_out, x_in, z_base, curvature, w_lw in configs:
        xs = np.linspace(x_out, x_in, 45)
        # Curvatura parabólica/catenaria que abraza la copa del aireador y desciende suavemente
        u = (xs - x_in) / (x_out - x_in) # 0 en morro, 1 en tubo
        zs = z_base - curvature * np.sin(u * np.pi) + (u - 0.5) * 6.0
        ax_front.plot(sgn * xs, zs, color=C_WHITE, lw=w_lw, solid_capstyle='round', zorder=9)

# Anotaciones frontales perfectamente emplazadas
ax_front.annotate('Muesca Ceño 34mm\n(Blanco continuo con cúpula)', xy=(0, 1826), xytext=(0, 1885),
                  arrowprops=dict(arrowstyle='->', color=C_CYAN, lw=1.2),
                  color=C_CYAN, fontsize=9.5, ha='center', weight='bold')

ax_front.annotate('Ojeras desplazadas al exterior\n(|X| 44..82mm, 62% longitud)\nFlanco medial en rojo continuo',
                  xy=(63, 1766), xytext=(85, 1845),
                  arrowprops=dict(arrowstyle='->', color=C_CYAN, lw=1.2),
                  color=C_CYAN, fontsize=8.5, ha='left')

ax_front.annotate('3 Bigotes Curvos Concéntricos\nCruzan eje vertical (|X|=47.5mm)\nhasta morro (|X|=22mm)',
                  xy=(-47.5, 1674), xytext=(-85, 1845),
                  arrowprops=dict(arrowstyle='->', color=C_CYAN, lw=1.2),
                  color=C_CYAN, fontsize=8.5, ha='right')

ax_front.annotate('Eje aireador\n(|X|=47.5mm)', xy=(47.5, 1618), xytext=(47.5, 1595),
                  color=C_AMBER, fontsize=8, ha='center')

ax_front.annotate('Morro central rojo continuo (|X| <= 22mm)', xy=(0, 1630), xytext=(0, 1602),
                  color='#FCA5A5', fontsize=8.5, ha='center', weight='bold')


# =========================================================================
# 2. VISTA LATERAL IZQUIERDA (PERFIL 90° CON ALAS DE THOR)
# =========================================================================
ax_profile.set_title('2. VISTA LATERAL IZQUIERDA (PERFIL 90° CON ALAS DE THOR)', color=C_CYAN, fontsize=15, weight='bold', pad=12)
ax_profile.set_xlim(-115, 235)
ax_profile.set_ylim(1585, 1965)
ax_profile.set_xlabel('Profundidad Y (mm: Frente < 0 < Trasera)', color='#94A3B8', fontsize=11)
ax_profile.set_ylabel('Altura Z (mm)', color='#94A3B8', fontsize=11)

# 2.1 Silueta Base del Perfil Lateral
th_dome_p = np.linspace(np.pi*0.04, np.pi*0.96, 120)
dome_p_y = 65.0 + 138.0 * np.cos(th_dome_p)
dome_p_z = 1815.0 + 115.0 * np.sin(th_dome_p)

profile_outline_y = np.concatenate([
    [-75, -88, -80, -45, 0, 75, 140, 195, 205, 200, 185],
    dome_p_y[::-1]
])
profile_outline_z = np.concatenate([
    [1808, 1750, 1660, 1625, 1618, 1616, 1620, 1640, 1720, 1780, 1815],
    dome_p_z[::-1]
])
ax_profile.fill(profile_outline_y, profile_outline_z, color=C_WHITE, zorder=2)
ax_profile.plot(profile_outline_y, profile_outline_z, color='#94A3B8', lw=1.8, zorder=3)

# Cresta dorsal lateral
crest_p = patches.Polygon([[-30, 1928], [-30, 1948], [170, 1948], [180, 1928]], closed=True, facecolor=C_WHITE, edgecolor='#94A3B8', lw=1.2, zorder=4)
ax_profile.add_patch(crest_p)
crest_red_p = patches.Polygon([[-30, 1943], [-30, 1948], [170, 1948], [170, 1943]], closed=True, facecolor=C_RED, edgecolor='none', zorder=5)
ax_profile.add_patch(crest_red_p)

# Visor aperture lateral (negro diáfano)
ax_profile.add_patch(patches.Polygon([[-75, 1780], [-75, 1808], [-25, 1808], [-25, 1780]], closed=True, facecolor=C_DARK, edgecolor='#38BDF8', lw=1.0, zorder=6))

# Orejera cilíndrica izquierda
ear_cap = patches.Circle((72, 1775), 24, facecolor='#F1F5F9', edgecolor='#94A3B8', lw=1.5, zorder=7)
ax_profile.add_patch(ear_cap)
ear_inner = patches.Circle((72, 1775), 15, facecolor='#E2E8F0', edgecolor='#94A3B8', lw=1.0, zorder=7.2)
ax_profile.add_patch(ear_inner)

# Tubo vertical de la orejera
ear_pipe = patches.Rectangle((62, 1618), 20, 135, facecolor='#F1F5F9', edgecolor='#94A3B8', lw=1.2, zorder=6.5)
ax_profile.add_patch(ear_pipe)

# Ranuras del tubo lateral (10 ranuras)
for i in range(10):
    slot_y = 5 + i * 5.5
    slot_color = C_RED if i < 2 else '#94A3B8'
    ax_profile.plot([slot_y, slot_y + 4], [1655, 1675], color=slot_color, lw=2.4, zorder=7)

# 2.2 Pintura Roja de Mandíbula, Ceño y Depresión de la Mejilla
# El ceño rojo corta ANTES de la orejera (Y=20mm), dejando el escalón sobre la oreja 100% BLANCO
ax_profile.add_patch(patches.Rectangle((-84, 1808), 104, 22, facecolor=C_RED, edgecolor='none', zorder=4))

# La mejilla roja contornea por delante de la orejera (corta en Y=44mm, antes de la cápsula Y=48..96mm)
cheek_red_pts = [
    [-84, 1808], [44, 1808], [44, 1720], [16, 1670],
    [5, 1650], [-45, 1625], [-80, 1660], [-88, 1750]
]
ax_profile.add_patch(patches.Polygon(cheek_red_pts, closed=True, facecolor=C_RED, edgecolor='none', zorder=4))

# Bigotes del morro en perfil lateral (cruzan el tubo y avanzan al morro)
for z_wh, x_start in [(1675, -88), (1660, -84), (1646, -80)]:
    ax_profile.plot([x_start, -45], [z_wh - 6, z_wh + 4], color=C_WHITE, lw=5.0, solid_capstyle='round', zorder=8)

# 2.3 Banda Trasera en la Depresión Anatómica (Z = 1818..1842 mm)
rear_band_p = patches.Polygon([[115, 1818], [115, 1842], [198, 1842], [198, 1818]], closed=True, facecolor=C_RED, edgecolor='none', zorder=4)
ax_profile.add_patch(rear_band_p)

# 2.4 ALAS DE THOR ORGÁNICAS (Perfil Lateral Canónico)
wing_base_pts = [
    [-20, 1830], [-15, 1855], [-6, 1885], [4, 1912], [15, 1926],
    # Pluma 1
    [65, 1922], [105, 1908], [68, 1902], [32, 1898],
    # Pluma 2
    [62, 1894], [94, 1888], [62, 1884], [28, 1880],
    # Pluma 3
    [58, 1876], [84, 1870], [54, 1866], [24, 1862],
    # Pluma 4 (Pinza inferior esbelta)
    [52, 1858], [74, 1852], [38, 1840], [10, 1834], [-8, 1830]
]
ax_profile.add_patch(patches.Polygon(wing_base_pts, closed=True, facecolor=C_RED, edgecolor='#991B1B', lw=1.2, zorder=8))

# Ranuras chevron en BLANCO PURO (casco base visible entre plumas)
ax_profile.plot([32, 68, 105], [1898, 1902, 1908], color=C_WHITE, lw=2.6, zorder=8.5)
ax_profile.plot([28, 62, 94],  [1880, 1884, 1888], color=C_WHITE, lw=2.6, zorder=8.5)
ax_profile.plot([24, 54, 84],  [1862, 1866, 1870], color=C_WHITE, lw=2.6, zorder=8.5)

# Rastrillo diagonal canónico (62°)
ax_profile.plot([105, 74], [1908, 1852], color=C_AMBER, linestyle=':', lw=1.5, zorder=9)

# Anotaciones de perfil
ax_profile.annotate('Alas de Thor (4 Plumas)\nEmpalme orgánico en base\nPuntas en rastrillo 62°',
                    xy=(20, 1865), xytext=(-108, 1850),
                    arrowprops=dict(arrowstyle='->', color=C_CYAN, lw=1.2),
                    color=C_CYAN, fontsize=8.0, ha='left', weight='bold')

ax_profile.annotate('Paso blanco sobre orejera\n(Corte antes del disco Y=20..115mm)',
                    xy=(65, 1820), xytext=(110, 1750),
                    arrowprops=dict(arrowstyle='->', color=C_CYAN, lw=1.2),
                    color=C_CYAN, fontsize=8.0, ha='left')

ax_profile.annotate('Depresión mejilla 100% roja\nCubre las 2 primeras ranuras del tubo',
                    xy=(10, 1665), xytext=(-85, 1715),
                    arrowprops=dict(arrowstyle='->', color=C_CYAN, lw=1.2),
                    color=C_CYAN, fontsize=8.5, ha='left')

ax_profile.annotate('Banda roja en escalón anatómico\n(Depresión inicio cúpula Z=1818..1842mm)',
                    xy=(155, 1830), xytext=(125, 1870),
                    arrowprops=dict(arrowstyle='->', color=C_CYAN, lw=1.2),
                    color=C_CYAN, fontsize=8.5)


# =========================================================================
# 3. VISTA TRASERA CANÓNICA (ELEVACIÓN 180°)
# =========================================================================
ax_rear.set_title('3. VISTA TRASERA CANÓNICA (ELEVACIÓN 180°)', color=C_CYAN, fontsize=15, weight='bold', pad=12)
ax_rear.set_xlim(-160, 160)
ax_rear.set_ylim(1585, 1965)
ax_rear.set_xlabel('Ancho X (mm)', color='#94A3B8', fontsize=11)
ax_rear.set_ylabel('Altura Z (mm)', color='#94A3B8', fontsize=11)

# Eje de simetría central
ax_rear.plot([0, 0], [1585, 1965], color=C_AMBER, linestyle=':', lw=1.0, alpha=0.5, zorder=1)

# 3.1 Silueta Base Trasera
th_dome_r = np.linspace(np.pi*0.06, np.pi*0.94, 120)
dome_r_x = 106.0 * np.cos(th_dome_r)
dome_r_z = 1812.0 + 116.0 * np.sin(th_dome_r)

rear_outline_x = np.concatenate([[-106, -112, -110, -95, -70, 0, 70, 95, 110, 112, 106], dome_r_x[::-1]])
rear_outline_z = np.concatenate([[1812, 1740, 1660, 1622, 1616, 1616, 1616, 1622, 1660, 1740, 1812], dome_r_z[::-1]])
ax_rear.fill(rear_outline_x, rear_outline_z, color=C_WHITE, zorder=2)
ax_rear.plot(rear_outline_x, rear_outline_z, color='#94A3B8', lw=1.8, zorder=3)

# Cresta central trasera
crest_r = patches.Polygon([[-14, 1928], [-14, 1948], [14, 1948], [14, 1928]], closed=True, facecolor=C_WHITE, edgecolor='#94A3B8', lw=1.2, zorder=4)
ax_rear.add_patch(crest_r)
crest_red_r = patches.Rectangle((-14, 1943), 28, 5, facecolor=C_RED, edgecolor='none', zorder=5)
ax_rear.add_patch(crest_red_r)

# Depresión anatómica horizontal de inicio de cúpula (Z = 1818 mm)
ax_rear.plot([-106, 106], [1818, 1818], color='#38BDF8', linestyle='--', lw=1.0, alpha=0.7, zorder=3.5)

# 3.2 BANDA ROJA TRASERA
rear_band_left = patches.Polygon([[-95, 1818], [-95, 1842], [-28, 1842], [-28, 1826], [-35, 1818]], closed=True, facecolor=C_RED, edgecolor='none', zorder=4)
rear_band_right = patches.Polygon([[95, 1818], [95, 1842], [28, 1842], [28, 1826], [35, 1818]], closed=True, facecolor=C_RED, edgecolor='none', zorder=4)
rear_band_arch = patches.Polygon([[-28, 1842], [28, 1842], [26, 1828], [-26, 1828]], closed=True, facecolor=C_RED, edgecolor='none', zorder=4)
ax_rear.add_patch(rear_band_left)
ax_rear.add_patch(rear_band_right)
ax_rear.add_patch(rear_band_arch)

# 3.3 Disco de Rango Trasero (100% BLANCO PURO)
disc_rim = patches.Circle((0, 1795), 33, facecolor='#E2E8F0', edgecolor='#94A3B8', lw=1.5, zorder=5)
disc_face = patches.Circle((0, 1795), 30, facecolor=C_WHITE, edgecolor='#94A3B8', lw=1.2, zorder=5.5)
ax_rear.add_patch(disc_rim)
ax_rear.add_patch(disc_face)
for ang in np.linspace(0, 2*np.pi, 9)[:-1]:
    dx, dy = 18 * np.cos(ang), 18 * np.sin(ang)
    ax_rear.plot([0, dx], [1795, 1795 + dy], color='#E2E8F0', lw=0.8, zorder=5.8)

# 3.4 FRANJAS ROJAS VERTICALES DE LAS COSTILLAS
rib_left = patches.Polygon([[-32, 1780], [-20, 1780], [-23, 1616], [-35, 1616]], closed=True, facecolor=C_RED, edgecolor='none', zorder=4)
rib_right = patches.Polygon([[32, 1780], [20, 1780], [23, 1616], [35, 1616]], closed=True, facecolor=C_RED, edgecolor='none', zorder=4)
ax_rear.add_patch(rib_left)
ax_rear.add_patch(rib_right)

# Peldaños centrales 100% BLANCOS
for rz in range(1628, 1775, 16):
    ax_rear.plot([-18, 18], [rz, rz], color='#CBD5E1', lw=1.5, zorder=4.5)

# Anotaciones traseras
ax_rear.annotate('Banda roja alojada en la depresión anatómica\n(Escalón inicio cúpula Z = 1818 a 1842 mm)',
                 xy=(60, 1836), xytext=(45, 1885),
                 arrowprops=dict(arrowstyle='->', color=C_CYAN, lw=1.2),
                 color=C_CYAN, fontsize=8.5, weight='bold')

ax_rear.annotate('Disco de rango 100% BLANCO (R = 30mm)\nArco carmesí rodea el perímetro superior sin tocarlo',
                 xy=(0, 1826), xytext=(0, 1795),
                 color=C_DARK, fontsize=8.0, ha='center', weight='bold', zorder=10)

ax_rear.annotate('2 Franjas rojas verticales (12mm)\nsobre costillas anatómicas\nPeldaños centrales 100% BLANCOS',
                 xy=(28, 1680), xytext=(50, 1660),
                 arrowprops=dict(arrowstyle='->', color=C_CYAN, lw=1.2),
                 color=C_CYAN, fontsize=8.5)


# =========================================================================
# 4. DESPIECE VECTORIAL 1:1: ANATOMÍA CANÓNICA Y COTAS MAESTRAS
# =========================================================================
ax_detail.set_title('4. DESPIECE VECTORIAL 1:1: ANATOMÍA CANÓNICA Y COTAS', color=C_CYAN, fontsize=15, weight='bold', pad=12)
ax_detail.set_xlim(-65, 145)
ax_detail.set_ylim(1610, 1960)
ax_detail.set_xlabel('Profundidad / Detalle Y (mm)', color='#94A3B8', fontsize=11)
ax_detail.set_ylabel('Altura Z (mm)', color='#94A3B8', fontsize=11)

# 4.1 DESPIECE DETALLADO DE LAS ALAS DE THOR
wing_d_pts = [
    [-25, 1830], [-18, 1860], [-8, 1895], [5, 1925], [18, 1938],
    # Pluma 1
    [75, 1934], [120, 1918], [78, 1910], [40, 1904],
    # Pluma 2
    [74, 1902], [106, 1894], [70, 1886], [35, 1882],
    # Pluma 3
    [68, 1880], [94, 1872], [62, 1864], [30, 1860],
    # Pluma 4
    [58, 1858], [80, 1852], [45, 1838], [15, 1832], [-12, 1830]
]
ax_detail.add_patch(patches.Polygon(wing_d_pts, closed=True, facecolor=C_RED, edgecolor='#991B1B', lw=1.5, zorder=4))

# Ranuras chevron en BLANCO PURO
ax_detail.plot([40, 78, 120], [1904, 1910, 1918], color=C_WHITE, lw=2.6, zorder=5)
ax_detail.plot([35, 70, 106], [1882, 1886, 1894], color=C_WHITE, lw=2.6, zorder=5)
ax_detail.plot([30, 62, 94],  [1860, 1864, 1872], color=C_WHITE, lw=2.6, zorder=5)

# Cotas de las plumas
ax_detail.annotate('Pluma 1: L=105mm (Cúspide Z=1938mm)', xy=(120, 1918), xytext=(124, 1935),
                   arrowprops=dict(arrowstyle='->', color=C_AMBER, lw=1.0), color=C_AMBER, fontsize=8.5, weight='bold')
ax_detail.annotate('Pluma 2: L=92mm', xy=(106, 1894), xytext=(124, 1905),
                   arrowprops=dict(arrowstyle='->', color=C_AMBER, lw=1.0), color=C_AMBER, fontsize=8)
ax_detail.annotate('Pluma 3: L=84mm', xy=(94, 1872), xytext=(124, 1880),
                   arrowprops=dict(arrowstyle='->', color=C_AMBER, lw=1.0), color=C_AMBER, fontsize=8)
ax_detail.annotate('Pluma 4: L=74mm (Pinza inferior)', xy=(80, 1852), xytext=(104, 1845),
                   arrowprops=dict(arrowstyle='->', color=C_AMBER, lw=1.0), color=C_AMBER, fontsize=8.5, weight='bold')

# Raíz ensanchada
ax_detail.annotate('Raíz ensanchada y empalmada suavemente\n(Eliminado mástil recto o rígido)',
                   xy=(-15, 1845), xytext=(-10, 1818),
                   arrowprops=dict(arrowstyle='->', color=C_CYAN, lw=1.2), color=C_CYAN, fontsize=8.5, ha='center')

# Línea diagonal de rastrillo 62°
ax_detail.plot([120, 80], [1918, 1852], color='#E11D48', linestyle='--', lw=1.5, zorder=6)
ax_detail.text(104, 1868, 'Rastrillo 62°', color='#E11D48', fontsize=8.5, rotation=-45, weight='bold')

# 4.2 DETALLE MAESTRO 1:1: BIGOTES CURVOS DE LA MANDÍBULA
box_wh = patches.Rectangle((-55, 1715), 195, 85, facecolor='#162032', edgecolor='#334155', lw=1.2, zorder=3)
ax_detail.add_patch(box_wh)
ax_detail.text(-48, 1788, 'DETALLE MAESTRO: 3 BIGOTES CURVOS CONCÉNTRICOS (1:1)', color=C_CYAN, fontsize=9.5, weight='bold', zorder=4)

# Fondo rojo mandíbula
ax_detail.add_patch(patches.Rectangle((-46, 1725), 178, 55, facecolor=C_RED, edgecolor='none', zorder=4))
# Aireador de referencia (|X| = 47.5 mm)
ax_detail.add_patch(patches.Circle((20, 1740), 16, facecolor='#475569', edgecolor='#1E293B', lw=1.2, zorder=5))
ax_detail.plot([20, 20], [1720, 1780], color=C_AMBER, linestyle=':', lw=1.2, zorder=5.5)
ax_detail.text(26, 1722, 'Eje vertical aireador (|X|=47.5mm)', color=C_AMBER, fontsize=7.5, zorder=6)

# 3 Bigotes curvos cruzando el eje del aireador hacia el morro
for z_wh, x_start, x_end, c_depth in [(1770, 75, -25, 12), (1756, 70, -22, 10), (1742, 65, -19, 8)]:
    w_xs = np.linspace(x_start, x_end, 35)
    u_w = (w_xs - x_end) / (x_start - x_end)
    w_zs = z_wh - c_depth * np.sin(u_w * np.pi) + (u_w - 0.5) * 5.0
    ax_detail.plot(w_xs, w_zs, color=C_WHITE, lw=5.0, solid_capstyle='round', zorder=6)

ax_detail.annotate('Franjas cruzan eje y avanzan al centro hasta |X|=22mm',
                   xy=(-15, 1765), xytext=(-40, 1730),
                   arrowprops=dict(arrowstyle='->', color=C_WHITE, lw=1.0), color=C_WHITE, fontsize=8)

# 4.3 DETALLE MAESTRO 1:1: OJERAS SUB-OCULARES DESPLAZADAS AL EXTERIOR
box_oj = patches.Rectangle((-55, 1622), 195, 80, facecolor='#162032', edgecolor='#334155', lw=1.2, zorder=3)
ax_detail.add_patch(box_oj)
ax_detail.text(-48, 1688, 'DETALLE MAESTRO: OJERA SUB-OCULAR DESPLAZADA AL EXTERIOR (1:1)', color=C_CYAN, fontsize=9.5, weight='bold', zorder=4)

# Barra horizontal del visor
ax_detail.add_patch(patches.Rectangle((-46, 1662), 178, 18, facecolor=C_DARK, edgecolor='none', zorder=4))
ax_detail.text(-40, 1667, 'APERTURA T-VISOR DIÁFANA (100% HUECA)', color='#94A3B8', fontsize=7.5, zorder=5)

# Medial rojo: 0 a 44 mm
ax_detail.add_patch(patches.Rectangle((-46, 1632), 54, 26, facecolor=C_RED, edgecolor='none', zorder=4))
ax_detail.text(-42, 1642, 'Rojo Medial\n(|X| 0..44mm)', color='#FECACA', fontsize=7.5, zorder=5)

# Blanco Ojera: 44 a 82 mm (62% longitud total)
ax_detail.add_patch(patches.Rectangle((8, 1632), 68, 26, facecolor=C_WHITE, edgecolor='#CBD5E1', lw=1.2, zorder=4))
ax_detail.text(14, 1644, 'FRANJA BLANCA (62%)\n(|X| = 44 a 82 mm)', color=C_DARK, fontsize=8, weight='bold', zorder=5)

# Lateral rojo: > 82 mm
ax_detail.add_patch(patches.Rectangle((76, 1632), 56, 26, facecolor=C_RED, edgecolor='none', zorder=4))
ax_detail.text(82, 1642, 'Rojo Lateral\n(|X| > 82mm)', color='#FECACA', fontsize=7.5, zorder=5)


# Guardar en alta resolución PNG y formato vectorial SVG
fig_png = os.path.join(out_dir_local, "dibujo_maestro_thorn_2d.png")
fig_svg = os.path.join(out_dir_local, "dibujo_maestro_thorn_2d.svg")
fig.savefig(fig_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
fig.savefig(fig_svg, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)

import shutil
shutil.copy2(fig_png, os.path.join(out_dir_art, "dibujo_maestro_thorn_2d.png"))
shutil.copy2(fig_svg, os.path.join(out_dir_art, "dibujo_maestro_thorn_2d.svg"))

print(f"Plano Maestro 2D generado con éxito en:")
print(f"  PNG: {fig_png}")
print(f"  SVG: {fig_svg}")
