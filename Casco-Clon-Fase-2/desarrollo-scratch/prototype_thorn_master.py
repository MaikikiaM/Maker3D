"""
Prototipo parametrico de alta precision V4 para el Comandante Thorn (TCW S6 / Galactic Armory Phase 2).
Incorpora las 5 mejoras de calibracion fina detectadas en la auditoria visual de V3:
1. Mastil frontal diagonal suave y organico en coordenadas desplegadas (u, Z) sin codo horizontal en L en la frente.
2. Bigotes elipsoidales (escala Y=1.32) centrados en [+-60, -96, 1608] que barren todo el flanco lateral con pendiente dY/dZ=-0.75
   estrictamente paralela a las ranuras y al corte trasero, y recorte interior limpio sin asomar bajo el labio interno del aireador.
3. Extremos delantero y trasero de la cresta dorsal recortados en Y=[-30.5, +182.0] con margen Z_bot+0.75 mm (0 puntos rojos en vista cenital).
4. Esquina inferior de la ranura vertical del T-visor 100% limpia ([+-6.2, 1743.8]).
5. Escalon blanco en Z bajo el alfeizar del pomulo con banda superior solida (6 mm) sin astilla fina en Y=-3.5.
"""
import os
import sys
import time
from io import BytesIO
import numpy as np
import trimesh
import trimesh.viewer.windowed
import manifold3d as m3d
from PIL import Image
from scipy.ndimage import uniform_filter1d

_orig_update_perspective = trimesh.viewer.windowed.SceneViewer._update_perspective
def _safe_update_perspective(self, width, height):
    return _orig_update_perspective(self, max(1, width), max(1, height))
trimesh.viewer.windowed.SceneViewer._update_perspective = _safe_update_perspective

_orig_on_resize = trimesh.viewer.windowed.SceneViewer.on_resize
def _safe_on_resize(self, width, height):
    try:
        return _orig_on_resize(self, max(1, width), max(1, height))
    except Exception:
        return
trimesh.viewer.windowed.SceneViewer.on_resize = _safe_on_resize

sys.path.insert(0, r"c:\Users\migue\Proyectos IA varios\Miscelanea\scratch")
import geo_tools as gt

t0 = time.time()
BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
SCR_DIR = os.path.join(BASE_DIR, "scratch")

helmet_mesh = gt.load_helmet()
greeb = gt.load_greeblies()
v_base = helmet_mesh.vertices.astype(np.float32)
f_base = helmet_mesh.faces.astype(np.uint32)
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))
base_vol = helmet_mani.volume()

sam_p, sam_n, _ = gt.get_helmet_samples()


def catmull_rom_chain(pts, samples_per_seg=24, closed=False):
    P = np.asarray(pts, dtype=np.float64)
    if closed:
        P_ext = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P_ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    n_seg = len(P) if closed else len(P) - 1
    for i in range(n_seg):
        p0, p1, p2, p3 = P_ext[i], P_ext[i + 1], P_ext[i + 2], P_ext[i + 3]
        if np.linalg.norm(p2 - p1) < 1e-6:
            continue

        def tj(ti, pa, pb):
            return ti + max(np.linalg.norm(pb - pa) ** 0.5, 1e-4)

        t0_ = 0.0
        t1_ = tj(t0_, p0, p1)
        t2_ = tj(t1_, p1, p2)
        t3_ = tj(t2_, p2, p3)
        ts = np.linspace(t1_, t2_, samples_per_seg, endpoint=False)
        for t in ts:
            a1 = (t1_ - t) / (t1_ - t0_) * p0 + (t - t0_) / (t1_ - t0_) * p1
            a2 = (t2_ - t) / (t2_ - t1_) * p1 + (t - t1_) / (t2_ - t1_) * p2
            a3 = (t3_ - t) / (t3_ - t2_) * p2 + (t - t2_) / (t3_ - t2_) * p3
            b1 = (t2_ - t) / (t2_ - t0_) * a1 + (t - t0_) / (t2_ - t0_) * a2
            b2 = (t3_ - t) / (t3_ - t1_) * a2 + (t - t1_) / (t3_ - t1_) * a3
            c = (t2_ - t) / (t2_ - t1_) * b1 + (t - t1_) / (t2_ - t1_) * b2
            out.append(c)
    if not closed:
        out.append(P[-1])
    return np.array(out, dtype=np.float32)


def make_ccw(pts):
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) > 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area < 0:
        pts = pts[::-1]
    return m3d.CrossSection([pts])


def make_manifold_from_vf(verts, faces):
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32).copy()
    m = m3d.Manifold(m3d.Mesh(vert_properties=verts, tri_verts=faces))
    if m.volume() <= 0:
        faces[:, [0, 1]] = faces[:, [1, 0]]
        m = m3d.Manifold(m3d.Mesh(vert_properties=verts, tri_verts=faces))
    return m


def loft_quad_sections(sections):
    K = len(sections)
    verts = np.vstack(sections).astype(np.float32)
    faces = [[0, 2, 1], [0, 3, 2]]
    for i in range(K - 1):
        b0, b1 = 4 * i, 4 * (i + 1)
        for j in range(4):
            jn = (j + 1) % 4
            faces.append([b0 + j, b0 + jn, b1 + jn])
            faces.append([b0 + j, b1 + jn, b1 + j])
    bf = 4 * (K - 1)
    faces.extend([[bf + 0, bf + 1, bf + 2], [bf + 0, bf + 2, bf + 3]])
    return make_manifold_from_vf(verts, faces)


def stroke_polyline_2d(pts, half_w, round_caps=True):
    P = np.asarray(pts, dtype=np.float32)
    n = len(P)
    tangents = np.zeros_like(P)
    for i in range(n):
        if i == 0:
            d = P[1] - P[0]
        elif i == n - 1:
            d = P[-1] - P[-2]
        else:
            d1 = P[i] - P[i - 1]
            d2 = P[i + 1] - P[i]
            d1 /= max(np.linalg.norm(d1), 1e-6)
            d2 /= max(np.linalg.norm(d2), 1e-6)
            d = d1 + d2
        tangents[i] = d / max(np.linalg.norm(d), 1e-6)

    left_pts, right_pts = [], []
    for i in range(n):
        t = tangents[i]
        nrm = np.array([-t[1], t[0]], dtype=np.float32)
        scale = 1.0
        if 0 < i < n - 1:
            d1 = P[i] - P[i - 1]
            d1 /= max(np.linalg.norm(d1), 1e-6)
            n1 = np.array([-d1[1], d1[0]], dtype=np.float32)
            dot = max(float(np.dot(nrm, n1)), 0.40)
            scale = min(1.0 / dot, 2.0)
        left_pts.append(P[i] + nrm * (half_w * scale))
        right_pts.append(P[i] - nrm * (half_w * scale))

    poly = []
    if round_caps:
        t0_ = tangents[0]
        n0_ = np.array([-t0_[1], t0_[0]], dtype=np.float32)
        for a in np.linspace(np.pi / 2, 3 * np.pi / 2, 12):
            poly.append(P[0] + half_w * (np.cos(a) * t0_ + np.sin(a) * n0_))
    poly.extend(right_pts)
    if round_caps:
        t1_ = tangents[-1]
        n1_ = np.array([-t1_[1], t1_[0]], dtype=np.float32)
        for a in np.linspace(-np.pi / 2, np.pi / 2, 12):
            poly.append(P[-1] + half_w * (np.cos(a) * t1_ + np.sin(a) * n1_))
    poly.extend(left_pts[::-1])
    return make_ccw(poly)


# =========================================================================
# ZONA 1: CRESTA DORSAL (FLANCOS ROJOS PUROS, CERO PUNTOS EN EXTREMOS)
# =========================================================================
print("Zona 1: Flancos exteriores de la cresta dorsal...")
fw_mask = (sam_p[:, 0] > 12.2) & (sam_p[:, 0] < 19.5) & (sam_p[:, 2] > 1875.0) & (sam_n[:, 0] > 0.82)
fw_pts = sam_p[fw_mask]

ys_fin = np.linspace(-26.5, 178.5, 52)
x_in_arr, x_out_arr, z_bot_arr, z_top_arr, y_valid = [], [], [], [], []
for yv in ys_fin:
    s = fw_pts[np.abs(fw_pts[:, 1] - yv) < 2.8]
    if len(s) < 12:
        continue
    y_valid.append(yv)
    x_in_arr.append(float(np.percentile(s[:, 0], 10)) - 0.42)
    x_out_arr.append(float(np.percentile(s[:, 0], 98)) + 3.0)
    z_bot_arr.append(float(np.percentile(s[:, 2], 6)) + 0.72)
    z_top_arr.append(float(np.percentile(s[:, 2], 95)) - 0.22)

x_in_s = uniform_filter1d(x_in_arr, size=3, mode="nearest")
z_bot_s = uniform_filter1d(z_bot_arr, size=3, mode="nearest")
z_top_s = uniform_filter1d(z_top_arr, size=3, mode="nearest")

quads_r, quads_l = [], []
for yv, xi, xo, zb, zt in zip(y_valid, x_in_s, x_out_arr, z_bot_s, z_top_s):
    quads_r.append(np.array([[xi, yv, zb], [xo, yv, zb], [xo, yv, zt], [xi, yv, zt]], dtype=np.float32))
    quads_l.append(np.array([[-xo, yv, zb], [-xi, yv, zb], [-xi, yv, zt], [-xo, yv, zt]], dtype=np.float32))

fin_cutter = loft_quad_sections(quads_r) + loft_quad_sections(quads_l)

# =========================================================================
# ZONA 2: MASTIL DIAGONAL ORGANICO + BAHIA EN C + HOMBRO DE HALCON + 4 PLUMAS
# =========================================================================
print("Zona 2: Mastil diagonal fluido + Alas de Thor...")
Y_DOME_C = 35.0
R_REF = 90.0
outer_dome = sam_p[(sam_p[:, 0] > 5.0) & (sam_p[:, 2] > 1804.0) & ((sam_n[:, 0] > 0.15) | (sam_n[:, 1] < -0.15))]


def dome_xz_to_uv(x_target, z_target):
    s = outer_dome[(np.abs(outer_dome[:, 0] - x_target) < 2.2) & (np.abs(outer_dome[:, 2] - z_target) < 2.2) & (outer_dome[:, 1] < 20.0)]
    y_surf = float(np.percentile(s[:, 1], 5)) if len(s) else -45.0
    phi = np.arctan2(y_surf - Y_DOME_C, x_target)
    return [R_REF * phi, z_target]


def dome_yz_to_uv(y_target, z_target):
    s = outer_dome[(np.abs(outer_dome[:, 1] - y_target) < 2.5) & (np.abs(outer_dome[:, 2] - z_target) < 2.5)]
    x_surf = float(np.percentile(s[:, 0], 95)) if len(s) else 65.0
    phi = np.arctan2(y_target - Y_DOME_C, x_surf)
    return [R_REF * phi, z_target]


# Progresion 100% suave en (u, Z) sin codo horizontal:
# u_inner sube diagonalmente de -122.7 (X=18.5, Z=1807.5) -> -110.5 (Z=1821) -> -93.5 (Z=1834)
# -> -73.0 (Z=1844.5) -> -54.5 (bahia concava en Y=-13.0, Z=1854.0) -> avanza al hombro convexo
# de halcon en Y=-20.5 (Z=1875.5) y barre hasta Y=115.5!
front_top_uv = [
    dome_xz_to_uv(18.5, 1807.5),
    dome_xz_to_uv(28.5, 1820.5),
    dome_xz_to_uv(42.5, 1833.5),
    [-68.0, 1844.5],
    [-55.5, 1853.5],               # Bahia concava circular suave sin pico agudo
    [-57.5, 1863.0],
    dome_yz_to_uv(-20.5, 1875.0),  # Hombro convexo redondeado de halcon
    dome_yz_to_uv(-11.0, 1887.5),
    dome_yz_to_uv(  8.0, 1896.2),
    dome_yz_to_uv( 34.0, 1899.5),  # Cima achatada 25%
    dome_yz_to_uv( 64.0, 1895.8),
    dome_yz_to_uv( 91.0, 1888.2),
    dome_yz_to_uv(115.5, 1877.0),  # Punta Pluma 1 (+15% hacia atras)
]
curve_top = catmull_rom_chain(front_top_uv, samples_per_seg=24, closed=False)

# Borde exterior/inferior: paralelo al mastil interior (ancho constante ~28-33 mm)
# y ciñendose a 2.5 mm por encima del canal esculpido de la cupula (Z=1851):
lower_bot_uv = [
    dome_yz_to_uv( 77.0, 1853.8),  # Punta Pluma 4
    dome_yz_to_uv( 48.0, 1853.3),
    dome_yz_to_uv( 22.0, 1852.8),  # Esquina superior del canal esculpido
    [-31.5, 1846.0],               # Curva suave sin estrangular la bahia C
    [-60.5, 1833.5],               # Ancho solido frente a Z=1833.5
    dome_xz_to_uv( 54.5, 1820.0),
    dome_xz_to_uv( 45.0, 1807.5),  # Alineado 1:1 con el escalon sub-ocular |X|=45.0!
]
curve_bot = catmull_rom_chain(lower_bot_uv, samples_per_seg=20, closed=False)

p_tip2 = dome_yz_to_uv(103.5, 1868.8)
p_tip3 = dome_yz_to_uv( 90.5, 1860.8)
p_tip4 = dome_yz_to_uv( 77.0, 1853.8)
rear_rake_uv = catmull_rom_chain([front_top_uv[-1], p_tip2, p_tip3, p_tip4], samples_per_seg=12, closed=False)

wing_silhouette = np.vstack([curve_top, rear_rake_uv[1:-1], curve_bot])
cs_wing = make_ccw(wing_silhouette)

# Espina blanca interior + 3 ranuras (inclinacion canonica TCW):
apex1 = dome_yz_to_uv(  4.0, 1880.5)
apex2 = dome_yz_to_uv( 15.0, 1869.5)
apex3 = dome_yz_to_uv( 23.0, 1859.8)
apex4 = dome_yz_to_uv( 29.0, 1849.5)

slot1_uv = catmull_rom_chain([apex1, dome_yz_to_uv(42.0, 1882.2), dome_yz_to_uv(80.0, 1878.5), dome_yz_to_uv(117.5, 1871.8)], 16)
slot2_uv = catmull_rom_chain([apex2, dome_yz_to_uv(48.0, 1870.2), dome_yz_to_uv(78.0, 1867.2), dome_yz_to_uv(105.5, 1863.2)], 16)
slot3_uv = catmull_rom_chain([apex3, dome_yz_to_uv(50.0, 1860.0), dome_yz_to_uv(72.0, 1858.2), dome_yz_to_uv( 92.5, 1856.2)], 16)
spine_uv = [apex1, apex2, apex3, apex4]

cs_grooves = (
    stroke_polyline_2d(slot1_uv, half_w=1.75, round_caps=True)
    + stroke_polyline_2d(slot2_uv, half_w=1.75, round_caps=True)
    + stroke_polyline_2d(slot3_uv, half_w=1.70, round_caps=True)
    + stroke_polyline_2d(spine_uv, half_w=1.65, round_caps=True)
)
cs_wing_final = cs_wing - cs_grooves


def cylindrical_uv_extrude_both_sides(cs_2d, r_in=15.0, r_out=145.0):
    raw = cs_2d.extrude(1.0)
    m = raw.to_mesh()
    verts_list = m.vert_properties[:, :3].tolist()
    tris = m.tri_verts.copy()
    for _ in range(2):
        edge_mid = {}
        new_tris = []
        for a, b, c in tris:
            def mid(i, j):
                key = (min(i, j), max(i, j))
                if key not in edge_mid:
                    p_m = 0.5 * (np.array(verts_list[i]) + np.array(verts_list[j]))
                    edge_mid[key] = len(verts_list)
                    verts_list.append(p_m.tolist())
                return edge_mid[key]
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_tris.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        tris = np.array(new_tris, dtype=np.uint32)

    V2 = np.array(verts_list, dtype=np.float64)
    phi = V2[:, 0] / R_REF
    z = V2[:, 1]
    rad = r_in + V2[:, 2] * (r_out - r_in)

    vr = np.column_stack([rad * np.cos(phi), Y_DOME_C + rad * np.sin(phi), z])
    vl = np.column_stack([-rad * np.cos(phi), Y_DOME_C + rad * np.sin(phi), z])
    return make_manifold_from_vf(vr, tris) + make_manifold_from_vf(vl, tris)


wings_and_stems_cutter = cylindrical_uv_extrude_both_sides(cs_wing_final)
visor_slot_guard = m3d.Manifold.cube([240.0, 85.0, 50.0], center=False).translate([-120.0, -95.0, 1758.55])
wings_and_stems_cutter = wings_and_stems_cutter - visor_slot_guard

# =========================================================================
# ZONA 3 Y 5: MEJILLAS + DEPRESION 100% ROJA + CORTE TUBO LIMPIO CON ESCALON Z SOLIDO
# =========================================================================
print("Zona 3 y 5: Mejillas + depresion + corte tubo con escalon Z limpio...")
# En Z=1705..1712.5 mantenemos una pestaña roja solida hasta Y=+0.5 mm (justo encima de la 1a ranura Y=+1.0, Z=1700),
# y el trazo blanco en Z arranca en Y=+4.0 (en zona blanca) cortando limpiamente sin dejar astillas!
cheek_tube_yz = [
    [-105.0, 1615.0],
    [  43.5, 1615.0],
    [  41.1, 1640.0],  # Recta exacta dY/dZ = -0.72 hasta Z=1705!
    [   5.1, 1690.0],
    [  -5.7, 1705.0],
    [   0.5, 1705.0],  # Pestaña superior solida (~7.5 mm de alto) bajo el alfeizar del pomulo
    [   0.5, 1712.8],
    [  47.5, 1713.5],  # Suelo completo de la depresion del pomulo
    [  47.5, 1782.8],  # Pared trasera biselada completa
    [ -11.0, 1766.5],  # Arista superior esculpida de la depresion
    [ -14.0, 1803.0],
    [-105.0, 1803.0],
]
cs_cheek_yz = make_ccw(cheek_tube_yz)
raw_cheek = cs_cheek_yz.extrude(135.0)
mat_ch_r = np.array([[0.0, 0.0, 1.0, 4.20], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_ch_l = np.array([[0.0, 0.0, -1.0, -4.20], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
cheeks_mani = raw_cheek.transform(mat_ch_r) + raw_cheek.transform(mat_ch_l)

# Motivos ortogonales blancos en el borde trasero del tubo (Punto 3):
e_par = np.array([0.5812, -0.8137], dtype=np.float32)
e_perp = np.array([-0.8137, -0.5812], dtype=np.float32)

# 1) Escalon blanco en Z (arranca en Y=+3.5 detras de Y=+0.5, dejando una pestaña roja solida de ~6.5 mm arriba!):
p_z0 = np.array([3.5, 1706.5], dtype=np.float32)
p_z1 = p_z0 + 16.0 * e_perp
p_z2 = p_z1 + 9.2 * e_par
p_z3 = p_z2 - 14.0 * e_perp  # Vuelve hacia atras a la zona blanca!
# Espera: en TCW el escalon Z baja y conecta con el blanco trasero! Si p_z3 vuelve hacia -e_perp,
# corta un bloque escalonado blanco; o si va hacia +e_perp hace un rayo en Z!
# Veamos compare_v3.png fila 2 izq (Zoom Canon TCW): en TCW es un TRAZO BLANCO EN Z (p_z2 + 11.5 * e_perp)!
p_z3 = p_z2 + 11.5 * e_perp
cs_tube_zstep = stroke_polyline_2d([p_z0, p_z1, p_z2, p_z3], half_w=2.30, round_caps=False)

# 2) Muesca rectangular blanca inferior en Z~1663:
p_n0 = np.array([26.5, 1664.2], dtype=np.float32)
p_n1 = p_n0 + 14.5 * e_perp
cs_tube_notch = stroke_polyline_2d([p_n0, p_n1], half_w=2.35, round_caps=False)

cs_tube_motifs = cs_tube_zstep + cs_tube_notch
raw_tube_motifs = cs_tube_motifs.extrude(65.0)
mat_tm_r = np.array([[0.0, 0.0, 1.0, 62.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_tm_l = np.array([[0.0, 0.0, -1.0, -62.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
tube_motifs_cutter = raw_tube_motifs.transform(mat_tm_r) + raw_tube_motifs.transform(mat_tm_l)

# Banda sub-ocular blanca continua desde |X|=45.0 mm hasta la sien:
suboc_cut_pts_xz = [
    [45.0, 1810.0],
    [45.0, 1785.0],
    [46.2, 1783.6],
    [52.0, 1781.6],
    [60.0, 1778.2],
    [68.0, 1774.4],
    [76.0, 1770.3],
    [82.0, 1767.5],
    [86.5, 1766.2],
    [125.0, 1766.2],
    [125.0, 1810.0],
]
raw_suboc = make_ccw(suboc_cut_pts_xz).extrude(90.0)
mat_xz_r = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -98.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_xz_l = np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -98.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
suboc_cutter = raw_suboc.transform(mat_xz_r) + raw_suboc.transform(mat_xz_l)

# =========================================================================
# ZONA 4: RAYO BLANCO EN LA DEPRESION DEL POMULO (CALIBRADO FINO 5.4 MM)
# =========================================================================
print("Zona 4: Rayo blanco en la depresion del pomulo...")
bolt_centerline_yz = [
    [-7.5, 1719.5],
    [ 9.0, 1731.5],
    [ 3.0, 1741.8],
    [22.5, 1755.5],
]
cs_bolt_yz = stroke_polyline_2d(bolt_centerline_yz, half_w=2.70, round_caps=False)
cs_bolt_clip = make_ccw([[-25.0, 1714.5], [46.0, 1714.5], [46.0, 1775.0], [-25.0, 1775.0]])
cs_bolt_yz = cs_bolt_yz ^ cs_bolt_clip
raw_bolt = cs_bolt_yz.extrude(56.0)
mat_bolt_r = np.array([[0.0, 0.0, 1.0, 66.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_bolt_l = np.array([[0.0, 0.0, -1.0, -66.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
cheek_bolt_cutter = raw_bolt.transform(mat_bolt_r) + raw_bolt.transform(mat_bolt_l)

# =========================================================================
# ZONA 7: MORRO ROJO + ARCO EXACTO DE LA CUENCA DE LOS DIENTES + ESQUINA T-VISOR LIMPIA
# =========================================================================
print("Zona 7: Morro piramidal rojo + cuenca exacta de los dientes...")
snout_pts_xz = [
    [-46.0, 1616.0],
    [ 46.0, 1616.0],
    [ 44.5, 1709.5],
    [ 38.0, 1714.3],
    [ 30.0, 1718.3],
    [ 20.0, 1722.3],
    [ 10.0, 1726.5],
    [  0.0, 1728.1],
    [-10.0, 1726.5],
    [-20.0, 1722.3],
    [-30.0, 1718.3],
    [-38.0, 1714.3],
    [-44.5, 1709.5],
]
snout_mani = make_ccw(snout_pts_xz).extrude(62.0).transform(
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -98.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
)

frown_arch_xz = [
    [  0.0, 1728.0],
    [ 10.0, 1726.4],
    [ 20.0, 1722.2],
    [ 30.0, 1718.2],
    [ 38.0, 1714.2],
    [ 43.5, 1710.2],
    [ 38.0, 1720.2],
    [ 30.0, 1726.2],
    [ 20.0, 1732.2],
    [ 10.0, 1736.8],
    [  6.2, 1743.8],  # Limpia al 100% la esquina inferior redondeada de la ranura T!
    [ -6.2, 1743.8],
    [-10.0, 1736.8],
    [-20.0, 1732.2],
    [-30.0, 1726.2],
    [-38.0, 1720.2],
    [-43.5, 1710.2],
    [-38.0, 1714.2],
    [-30.0, 1718.2],
    [-20.0, 1722.2],
    [-10.0, 1726.4],
]
frown_cutter = make_ccw(frown_arch_xz).extrude(50.0).transform(
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -98.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
)

# =========================================================================
# ZONA 6: 3 BIGOTES ELIPSOIDALES ENVOLVENTES EN C=[±60, -96, 1608] (PUNTO 4)
# =========================================================================
print("Zona 6: 3 bigotes blancos elipsoidales envolventes de lado a lado...")


def make_ellipsoidal_shell(center, r_in, r_out, scale_xyz=(1.0, 1.32, 1.0)):
    sx, sy, sz = scale_xyz
    outer = m3d.Manifold.sphere(1.0, 96).scale([r_out * sx, r_out * sy, r_out * sz]).translate(center)
    inner = m3d.Manifold.sphere(1.0, 96).scale([r_in * sx, r_in * sy, r_in * sz]).translate(center)
    return outer - inner


shell_specs = [(50.2, 55.4), (64.2, 69.4), (78.2, 83.4)]
whiskers_cutter = m3d.Manifold()
for side in (+1.0, -1.0):
    c_cone = [side * 60.0, -96.0, 1606.0]
    shells_side = m3d.Manifold()
    for r_in, r_out in shell_specs:
        shells_side = shells_side + make_ellipsoidal_shell(c_cone, r_in, r_out, scale_xyz=(1.0, 1.46, 1.0))

    # Evita al 100% que el Bigote 1 asome por el labio inferior interno del aireador (Z<1642, |X|<52),
    # mientras arriba (Z=1655..1705) llega limpio al pliegue del valle anatomico morro-tubo (|X|=36..44)!
    pts_r = [
        [52.5, 1615.0],
        [135.0, 1615.0],
        [135.0, 1710.0],
        [43.8, 1710.0],
        [35.2, 1662.0],
        [51.5, 1642.0],
    ]
    valley_keep_xz = pts_r if side > 0 else [[-x, z] for x, z in pts_r[::-1]]
    keep_box = make_ccw(valley_keep_xz).extrude(135.0).transform(
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -100.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    )
    whiskers_cutter = whiskers_cutter + (shells_side ^ keep_box)

# =========================================================================
# ZONA 8: BANDA EN DEPRESION TRASERA + DISCO BLANCO + 2 COSTILLAS CONICAS
# =========================================================================
print("Zona 8: Banda trasera en depresion exacta, disco blanco y costillas conicas...")
rear_band_yz = [
    [134.0, 1824.25 + 0.0145 * 134.0],
    [230.0, 1824.25 + 0.0145 * 230.0],
    [230.0, 1850.00 + 0.0165 * 230.0],
    [134.0, 1850.00 + 0.0165 * 134.0],
]
rear_band_mani = make_ccw(rear_band_yz).extrude(210.0).transform(
    np.array([[0.0, 0.0, 1.0, -105.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
)

rib_right_xz = [
    [12.75, 1615.0],
    [16.20, 1615.0],
    [16.30, 1650.0],
    [19.95, 1675.0],
    [22.85, 1700.0],
    [25.85, 1730.0],
    [28.85, 1760.0],
    [31.65, 1790.0],
    [33.85, 1816.5],
    [12.75, 1816.5],
]
rib_left_xz = [[-x, z] for x, z in rib_right_xz[::-1]]
mat_rib = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 190.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
ribs_mani = (
    make_ccw(rib_right_xz).extrude(45.0).transform(mat_rib)
    + make_ccw(rib_left_xz).extrude(45.0).transform(mat_rib)
)
disc_excl = m3d.Manifold.cylinder(60.0, 26.6, 26.6, 128).rotate([90, 0, 0]).translate([0.0, 240.0, 1817.5])
rear_cutter = (rear_band_mani + ribs_mani) - disc_excl

# =========================================================================
# BOOLEANO FINAL MANIFOLD
# =========================================================================
ear_excl = (
    m3d.Manifold.cylinder(80.0, 32.5, 32.5, 96).rotate([0, 90, 0]).translate([85.0, 77.7, 1789.1])
    + m3d.Manifold.cylinder(80.0, 32.5, 32.5, 96).rotate([0, -90, 0]).translate([-85.0, 77.7, 1789.1])
)

face_cutter = (
    ((cheeks_mani + snout_mani) - suboc_cutter)
    - frown_cutter
    - cheek_bolt_cutter
    - tube_motifs_cutter
    - whiskers_cutter
)

total_red_cutter = (fin_cutter + wings_and_stems_cutter + rear_cutter + face_cutter) - ear_excl

print("Ejecutando interseccion booleana manifold3d...")
red_mani = helmet_mani ^ total_red_cutter
white_mani = helmet_mani - total_red_cutter

vol_r = red_mani.volume()
vol_w = white_mani.volume()
dev = abs((vol_r + vol_w) - base_vol)
print(f"Volumen Rojo = {vol_r:.1f} | Blanco = {vol_w:.1f} | Suma = {vol_r+vol_w:.1f} | Base = {base_vol:.1f} | dV = {dev:.5f} mm3")

mr = red_mani.to_mesh()
mw = white_mani.to_mesh()
mesh_red = trimesh.Trimesh(vertices=mr.vert_properties[:, :3], faces=mr.tri_verts, process=False)
mesh_white = trimesh.Trimesh(vertices=mw.vert_properties[:, :3], faces=mw.tri_verts, process=False)

mesh_red.visual.vertex_colors = [185, 28, 28, 255]
mesh_white.visual.vertex_colors = [244, 244, 246, 255]
mouth_m = greeb["mouth"].copy()
mouth_m.visual.vertex_colors = [20, 20, 22, 255]
ears_m = trimesh.util.concatenate([greeb["ear_l"], greeb["ear_r"]])
ears_m.visual.vertex_colors = [238, 239, 243, 255]
aer_m = trimesh.util.concatenate([greeb["aer_l"], greeb["aer_r"]])
aer_m.visual.vertex_colors = [68, 70, 75, 255]

print("Renderizando 8 vistas OpenGL solidas V4...")
scene = trimesh.Scene([mesh_white, mesh_red, mouth_m, ears_m, aer_m])
center = scene.centroid
scene.camera.z_near = 10.0
scene.camera.z_far = 2500.0


def render_cam(cam_pos, look_target, res=1400, out_size=750):
    view_vec = np.asarray(look_target, float) - np.asarray(cam_pos, float)
    forward = view_vec / np.linalg.norm(view_vec)
    up = np.array([0.0, 0.0, 1.0])
    if abs(forward[2]) > 0.99:
        up = np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    actual_up = np.cross(right, forward)
    actual_up /= np.linalg.norm(actual_up)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = actual_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = cam_pos
    scene.camera_transform = c2w
    for attempt in range(4):
        try:
            time.sleep(0.06)
            raw = scene.save_image(resolution=(res, res))
            break
        except Exception as e:
            if attempt == 3:
                raise e
            time.sleep(0.2)
    im = Image.open(BytesIO(raw)).convert("RGB")
    return im.resize((out_size, out_size), Image.Resampling.LANCZOS)


def orbit_pos(yaw_deg, pitch_deg, dist=530.0):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return center + np.array([dist * np.sin(y) * np.cos(p), -dist * np.cos(y) * np.cos(p), dist * np.sin(p)])


shots = [
    ("front", orbit_pos(0, 0), center),
    ("tcw_34", orbit_pos(36, 7), center),
    ("tcw_ref_side", orbit_pos(-36, 7), center),
    ("left", orbit_pos(90, 0), center),
    ("back", orbit_pos(180, 0), center),
    ("top", orbit_pos(0, 88), center),
    ("macro_side_jaw", [-195.0, -185.0, 1705.0], [-45.0, -20.0, 1705.0]),
    ("macro_wing", [-340.0, 35.0, 1880.0], [-65.0, 35.0, 1870.0]),
]

for key, cpos, tgt in shots:
    im = render_cam(cpos, tgt)
    out_p = os.path.join(SCR_DIR, f"proto_v6_{key}.png")
    im.save(out_p)
    print(f"  Guardado: {out_p}")

print(f"V6 completado en {time.time() - t0:.1f} s!")
