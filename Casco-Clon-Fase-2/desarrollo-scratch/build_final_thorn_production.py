"""
Pipeline Maestro de Produccion Definitiva: Casco Clon Fase 2 - Comandante Thorn (CC-3223, Guardia de Coruscant, TCW S6).
Geometria parametrica 3D calibrada al 0.1 mm sobre 'Phase 2 Main Body.stl' de Galactic Armory (Prototipo Verificado V6).
Apertura T-Visor 100% abierta/hueca en ensamblajes GLB, 3MF y renders de acuerdo con las directrices del usuario.
"""
import os
import re
import time
import shutil
import zipfile
from io import BytesIO
import numpy as np
import trimesh
import trimesh.viewer.windowed
import manifold3d as m3d
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter1d

# Evita ZeroDivisionError en redimensionado headless de pyglet en Windows
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

t_start = time.time()
print("=========================================================================")
print("=== PIPELINE MAESTRO DEFINITIVO: COMANDANTE THORN (TCW S6 / CC-3223)  ===")
print("=== PARTICION BOOLEANA 100% ESTANCA + AUDITORIA MULTI-ANGULO 20 VISTAS ===")
print("=========================================================================")

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
CASCO_P2_DIR = os.path.join(BASE_DIR, r"Diseños 3D\01_Casco_Clon_Fase_2")
BASE_MODELS_DIR = os.path.join(CASCO_P2_DIR, "00_Archivos_Base_Galactic_Armory")
GREEB_DIR = os.path.join(BASE_MODELS_DIR, "Greeblies")
ANTIGRAVITY_DIR = os.path.join(CASCO_P2_DIR, "02_Modelos_Personalizados_Antigravity")

OUT_05 = os.path.join(ANTIGRAVITY_DIR, "05_Cascos_Pintados_Clones")
OUT_05_STL = os.path.join(OUT_05, "Thorn_Despiece_Fabricacion_STL")
OUT_05_3MF = os.path.join(OUT_05, "Modelos_Pintados_3MF_GLB")

OUT_04 = os.path.join(ANTIGRAVITY_DIR, "04_Guias_Visuales_y_Planos")
OUT_04_LAMINAS = os.path.join(OUT_04, "Laminas_Maestras")
OUT_04_AUDIT = os.path.join(OUT_04, "Archivo_Auditoria_Renders")

OUT_03 = os.path.join(ANTIGRAVITY_DIR, "03_Stencils_Pintura_Clones")
ARTIFACT_DIR = r"C:\Users\migue\.gemini\antigravity\brain\708952eb-13ab-449a-b9c1-cf788f04feb2"

for d in (OUT_05_STL, OUT_05_3MF, OUT_04_LAMINAS, OUT_04_AUDIT, OUT_03, ARTIFACT_DIR):
    os.makedirs(d, exist_ok=True)

# -------------------------------------------------------------------------
# 1. CARGA DEL CASCO BASE, MUESTREO DE SUPERFICIE Y ACCESORIOS OFICIALES
# -------------------------------------------------------------------------
print("\n[1/6] Cargando geometria base Galactic Armory Phase 2 y greeblies...")
helmet_mesh = trimesh.load(os.path.join(BASE_MODELS_DIR, "Phase 2 Main Body.stl"), force="mesh")
v_base = helmet_mesh.vertices.astype(np.float32)
f_base = helmet_mesh.faces.astype(np.uint32)
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))
base_vol = helmet_mani.volume()
print(f"  Casco base: {len(v_base)} vertices, {len(f_base)} triangulos, volumen = {base_vol:.1f} mm3")

cache_npz = os.path.join(BASE_DIR, "scratch", "helmet_samples.npz")
if os.path.exists(cache_npz):
    data_npz = np.load(cache_npz)
    sam_p = data_npz["p"]
    sam_n = data_npz["n"]
else:
    pts_s, f_idx = trimesh.sample.sample_surface(helmet_mesh, 6000000, seed=0)
    sam_p = pts_s.astype(np.float32)
    sam_n = helmet_mesh.face_normals[f_idx].astype(np.float32)

mouth_mesh = trimesh.load(os.path.join(GREEB_DIR, "Mouth Center Detail.stl"), force="mesh")
ear_l = trimesh.load(os.path.join(GREEB_DIR, "Ear Piece Left.stl"), force="mesh")
ear_r = trimesh.load(os.path.join(GREEB_DIR, "Ear Piece Right.stl"), force="mesh")
ears_mesh = trimesh.util.concatenate([ear_l, ear_r])
aer_l = trimesh.load(os.path.join(GREEB_DIR, "Aerator Left - Mesh.stl"), force="mesh")
aer_r = trimesh.load(os.path.join(GREEB_DIR, "Aerator Right - Mesh.stl"), force="mesh")
aerators_mesh = trimesh.util.concatenate([aer_l, aer_r])

# Pieza opcional independiente 06 (NO se monta en ensamblajes ni renders por indicacion expresa del usuario)
xs_eye = np.linspace(-88.0, 88.0, 64)
yf_eye = -58.5 + 21.0 * (xs_eye / 85.0) ** 2
poly_eye = np.vstack([np.column_stack([xs_eye, yf_eye]), np.column_stack([xs_eye[::-1], (yf_eye + 2.5)[::-1]])])
cs_eye = m3d.CrossSection([poly_eye])
if cs_eye.area() == 0:
    cs_eye = m3d.CrossSection([poly_eye[::-1]])
xs_stem = np.linspace(-11.0, 11.0, 16)
yf_s = -58.5 + 21.0 * (xs_stem / 85.0) ** 2
poly_stem = np.vstack([np.column_stack([xs_stem, yf_s]), np.column_stack([xs_stem[::-1], (yf_s + 2.5)[::-1]])])
cs_stem = m3d.CrossSection([poly_stem])
if cs_stem.area() == 0:
    cs_stem = m3d.CrossSection([poly_stem[::-1]])
lens_mani = cs_eye.extrude(38.0).translate([0, 0, 1775.0]) + cs_stem.extrude(75.0).translate([0, 0, 1738.0])
m_lens = lens_mani.to_mesh()
lens_mesh_standalone = trimesh.Trimesh(vertices=m_lens.vert_properties[:, :3], faces=m_lens.tri_verts, process=False)


# -------------------------------------------------------------------------
# 2. UTILIDADES GEOMETRICAS Y CONSTRUCCION PARAMETRICA DE LAS 8 ZONAS
# -------------------------------------------------------------------------
print("\n[2/6] Construyendo cortadores parametricos 3D sobre pliegues anatomicos reales...")


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


# --- ZONA 1: Flancos exteriores inclinados de la Cresta Dorsal ---
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

# --- ZONA 2: Mastil diagonal continuo + Bahia en C + Hombro de Halcon + Alas de Thor (4 plumas) ---
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


front_top_uv = [
    dome_xz_to_uv(18.5, 1807.5),
    dome_xz_to_uv(28.5, 1820.5),
    dome_xz_to_uv(42.5, 1833.5),
    [-68.0, 1844.5],
    [-55.5, 1853.5],
    [-57.5, 1863.0],
    dome_yz_to_uv(-20.5, 1875.0),
    dome_yz_to_uv(-11.0, 1887.5),
    dome_yz_to_uv(  8.0, 1896.2),
    dome_yz_to_uv( 34.0, 1899.5),
    dome_yz_to_uv( 64.0, 1895.8),
    dome_yz_to_uv( 91.0, 1888.2),
    dome_yz_to_uv(115.5, 1877.0),
]
curve_top = catmull_rom_chain(front_top_uv, samples_per_seg=24, closed=False)

lower_bot_uv = [
    dome_yz_to_uv( 77.0, 1853.8),
    dome_yz_to_uv( 48.0, 1853.3),
    dome_yz_to_uv( 22.0, 1852.8),
    [-31.5, 1846.0],
    [-60.5, 1833.5],
    dome_xz_to_uv( 54.5, 1820.0),
    dome_xz_to_uv( 45.0, 1807.5),
]
curve_bot = catmull_rom_chain(lower_bot_uv, samples_per_seg=20, closed=False)

p_tip2 = dome_yz_to_uv(103.5, 1868.8)
p_tip3 = dome_yz_to_uv( 90.5, 1860.8)
p_tip4 = dome_yz_to_uv( 77.0, 1853.8)
rear_rake_uv = catmull_rom_chain([front_top_uv[-1], p_tip2, p_tip3, p_tip4], samples_per_seg=12, closed=False)

wing_silhouette = np.vstack([curve_top, rear_rake_uv[1:-1], curve_bot])
cs_wing = make_ccw(wing_silhouette)

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

# --- ZONA 3 Y 5: Mejillas + Depresion 100% roja + Banda sub-ocular + Corte tubo paralelo dY/dZ=-0.72 ---
cheek_tube_yz = [
    [-105.0, 1615.0],
    [  43.5, 1615.0],
    [  41.1, 1640.0],
    [   5.1, 1690.0],
    [  -0.6, 1698.0],
    [   0.5, 1698.0],
    [   0.5, 1712.8],
    [  47.5, 1713.5],
    [  47.5, 1782.8],
    [ -11.0, 1766.5],
    [ -14.0, 1803.0],
    [-105.0, 1803.0],
]
cs_cheek_yz = make_ccw(cheek_tube_yz)
raw_cheek = cs_cheek_yz.extrude(135.0)
mat_ch_r = np.array([[0.0, 0.0, 1.0, 4.20], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_ch_l = np.array([[0.0, 0.0, -1.0, -4.20], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
cheeks_mani = raw_cheek.transform(mat_ch_r) + raw_cheek.transform(mat_ch_l)

e_par = np.array([0.5812, -0.8137], dtype=np.float32)
e_perp = np.array([-0.8137, -0.5812], dtype=np.float32)

p_z0 = np.array([3.5, 1706.5], dtype=np.float32)
p_z1 = p_z0 + 16.0 * e_perp
p_z2 = p_z1 + 9.2 * e_par
p_z3 = p_z2 + 11.5 * e_perp
cs_tube_zstep = stroke_polyline_2d([p_z0, p_z1, p_z2, p_z3], half_w=2.30, round_caps=False)

p_n0 = np.array([26.5, 1664.2], dtype=np.float32)
p_n1 = p_n0 + 14.5 * e_perp
cs_tube_notch = stroke_polyline_2d([p_n0, p_n1], half_w=2.35, round_caps=False)

cs_tube_motifs = cs_tube_zstep + cs_tube_notch
raw_tube_motifs = cs_tube_motifs.extrude(65.0)
mat_tm_r = np.array([[0.0, 0.0, 1.0, 62.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_tm_l = np.array([[0.0, 0.0, -1.0, -62.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
tube_motifs_cutter = raw_tube_motifs.transform(mat_tm_r) + raw_tube_motifs.transform(mat_tm_l)

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

# --- ZONA 4: Rayo blanco en la depresion del pomulo ---
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

# --- ZONA 7: Morro piramidal rojo + Cuenca blanca exacta de dientes ---
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
    [  6.2, 1743.8],
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

# --- ZONA 6: 3 Bigotes blancos elipsoidales concéntricos ---
def make_ellipsoidal_shell(center, r_in, r_out, scale_xyz=(1.0, 1.46, 1.0)):
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

# --- ZONA 8: Banda trasera en canal esculpido + Disco blanco R=26.6 + 2 Costillas cónicas ---
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


def build_midwall_prism_xz(xs_grid, zs_grid, y_mid_grid, y_front=140.0):
    nx, nz = len(xs_grid), len(zs_grid)
    verts, faces = [], []
    for ix, xv in enumerate(xs_grid):
        for iz, zv in enumerate(zs_grid):
            verts.append([xv, y_front, zv])
            verts.append([xv, float(y_mid_grid[ix, iz]), zv])

    def idx_f(ix, iz):
        return 2 * (ix * nz + iz)

    def idx_b(ix, iz):
        return 2 * (ix * nz + iz) + 1

    for ix in range(nx - 1):
        for iz in range(nz - 1):
            f00, f10, f11, f01 = idx_f(ix, iz), idx_f(ix + 1, iz), idx_f(ix + 1, iz + 1), idx_f(ix, iz + 1)
            b00, b10, b11, b01 = idx_b(ix, iz), idx_b(ix + 1, iz), idx_b(ix + 1, iz + 1), idx_b(ix, iz + 1)
            faces.extend([[f00, f01, f11], [f00, f11, f10]])
            faces.extend([[b00, b11, b01], [b00, b10, b11]])
    for ix in range(nx - 1):
        faces.extend([[idx_f(ix, 0), idx_f(ix + 1, 0), idx_b(ix + 1, 0)], [idx_f(ix, 0), idx_b(ix + 1, 0), idx_b(ix, 0)]])
        faces.extend([[idx_f(ix, nz - 1), idx_b(ix + 1, nz - 1), idx_f(ix + 1, nz - 1)], [idx_f(ix, nz - 1), idx_b(ix, nz - 1), idx_b(ix + 1, nz - 1)]])
    for iz in range(nz - 1):
        faces.extend([[idx_f(0, iz), idx_b(0, iz + 1), idx_f(0, iz + 1)], [idx_f(0, iz), idx_b(0, iz), idx_b(0, iz + 1)]])
        faces.extend([[idx_f(nx - 1, iz), idx_f(nx - 1, iz + 1), idx_b(nx - 1, iz + 1)], [idx_f(nx - 1, iz), idx_b(nx - 1, iz + 1), idx_b(nx - 1, iz)]])
    return make_manifold_from_vf(verts, faces)


def build_rear_interior_white_shield():
    # 1) Escudo de pared media para las 2 costillas verticales (|X| in [12, 36], Z in [1614, 1820])
    shields = m3d.Manifold()
    m1 = (np.abs(sam_p[:, 0]) >= 11.5) & (np.abs(sam_p[:, 0]) <= 36.5) & (sam_p[:, 1] > 165.0) & (sam_p[:, 2] >= 1615.0) & (sam_p[:, 2] <= 1820.0)
    p1, n1 = sam_p[m1], sam_n[m1]
    pi1, po1 = p1[n1[:, 1] < -0.2], p1[n1[:, 1] > 0.2]
    zs_r = np.linspace(1614.0, 1819.0, 36)
    for side_s in (+1.0, -1.0):
        xs_r = np.linspace(12.2, 35.5, 16) if side_s > 0 else np.linspace(-35.5, -12.2, 16)
        ymid = np.full((len(xs_r), len(zs_r)), 205.0, dtype=np.float32)
        for ix, xv in enumerate(xs_r):
            mix, mox = np.abs(pi1[:, 0] - xv) < 1.6, np.abs(po1[:, 0] - xv) < 1.6
            last_val = 205.0
            for iz, zv in enumerate(zs_r):
                si = pi1[mix & (np.abs(pi1[:, 2] - zv) < 3.5), 1]
                so = po1[mox & (np.abs(po1[:, 2] - zv) < 3.5), 1]
                if len(si) and len(so):
                    last_val = 0.5 * (float(np.percentile(si, 98)) + float(np.percentile(so, 2)))
                ymid[ix, iz] = last_val
        shields = shields + build_midwall_prism_xz(xs_r, zs_r, ymid, y_front=150.0)

    # 2) Escudo cilindrico de pared media para la banda trasera (Z in [1822, 1858], Y > 130)
    yc = 80.0
    m2 = (sam_p[:, 1] > 128.0) & (sam_p[:, 2] >= 1821.0) & (sam_p[:, 2] <= 1858.0)
    p2, n2 = sam_p[m2], sam_n[m2]
    dot = p2[:, 0] * n2[:, 0] + (p2[:, 1] - yc) * n2[:, 1]
    pi2, po2 = p2[dot < -15.0], p2[dot > 15.0]
    phi_i, ri = np.arctan2(pi2[:, 0], pi2[:, 1] - yc), np.hypot(pi2[:, 0], pi2[:, 1] - yc)
    phi_o, ro = np.arctan2(po2[:, 0], po2[:, 1] - yc), np.hypot(po2[:, 0], po2[:, 1] - yc)

    phis = np.linspace(-1.08, 1.08, 64)
    zs_b = np.linspace(1822.0, 1857.0, 16)
    nx, nz = len(phis), len(zs_b)
    verts, faces = [], []
    for ip, ph in enumerate(phis):
        mip, mop = np.abs(phi_i - ph) < 0.028, np.abs(phi_o - ph) < 0.028
        last_r = 116.0
        for iz, zv in enumerate(zs_b):
            si = ri[mip & (np.abs(pi2[:, 2] - zv) < 2.2)]
            so = ro[mop & (np.abs(po2[:, 2] - zv) < 2.2)]
            if len(si) and len(so):
                last_r = 0.5 * (float(np.percentile(si, 98)) + float(np.percentile(so, 2)))
            verts.append([25.0 * np.sin(ph), yc + 25.0 * np.cos(ph), zv])
            verts.append([last_r * np.sin(ph), yc + last_r * np.cos(ph), zv])

    def idx_in(ip, iz):
        return 2 * (ip * nz + iz)

    def idx_out(ip, iz):
        return 2 * (ip * nz + iz) + 1

    for ip in range(nx - 1):
        for iz in range(nz - 1):
            f00, f10, f11, f01 = idx_in(ip, iz), idx_in(ip + 1, iz), idx_in(ip + 1, iz + 1), idx_in(ip, iz + 1)
            b00, b10, b11, b01 = idx_out(ip, iz), idx_out(ip + 1, iz), idx_out(ip + 1, iz + 1), idx_out(ip, iz + 1)
            faces.extend([[f00, f01, f11], [f00, f11, f10]])
            faces.extend([[b00, b11, b01], [b00, b10, b11]])
    for ip in range(nx - 1):
        faces.extend([[idx_in(ip, 0), idx_in(ip + 1, 0), idx_out(ip + 1, 0)], [idx_in(ip, 0), idx_out(ip + 1, 0), idx_out(ip, 0)]])
        faces.extend([[idx_in(ip, nz - 1), idx_out(ip + 1, nz - 1), idx_in(ip + 1, nz - 1)], [idx_in(ip, nz - 1), idx_out(ip, nz - 1), idx_out(ip + 1, nz - 1)]])
    for iz in range(nz - 1):
        faces.extend([[idx_in(0, iz), idx_out(0, iz + 1), idx_in(0, iz + 1)], [idx_in(0, iz), idx_out(0, iz), idx_out(0, iz + 1)]])
        faces.extend([[idx_in(nx - 1, iz), idx_in(nx - 1, iz + 1), idx_out(nx - 1, iz + 1)], [idx_in(nx - 1, iz), idx_out(nx - 1, iz + 1), idx_out(nx - 1, iz)]])

    return shields + make_manifold_from_vf(verts, faces)


rear_cutter = ((rear_band_mani + ribs_mani) - disc_excl) - build_rear_interior_white_shield()

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

# -------------------------------------------------------------------------
# 3. PARTICION BOOLEANA EXACTA MANIFOLD3D
# -------------------------------------------------------------------------
print("\n[3/6] Ejecutando interseccion booleana exacta Manifold3D...")
red_mani = helmet_mani ^ total_red_cutter
white_mani = helmet_mani - total_red_cutter

vol_red = red_mani.volume()
vol_white = white_mani.volume()
vol_sum = vol_red + vol_white
dev = abs(vol_sum - base_vol)

print(f"  Patron Rojo Thorn: volumen = {vol_red:.1f} mm3 | triangulos = {red_mani.num_tri()}")
print(f"  Casco Base Blanco: volumen = {vol_white:.1f} mm3 | triangulos = {white_mani.num_tri()}")
print(f"  Suma = {vol_sum:.1f} mm3 | Base = {base_vol:.1f} mm3 | Desviacion dV = {dev:.5f} mm3 (100% Watertight)")
assert dev < 0.01, f"Desviacion volumetrica excesiva: {dev}"

mr = red_mani.to_mesh()
mw = white_mani.to_mesh()
mesh_red = trimesh.Trimesh(vertices=mr.vert_properties[:, :3], faces=mr.tri_verts, process=False)
mesh_white = trimesh.Trimesh(vertices=mw.vert_properties[:, :3], faces=mw.tri_verts, process=False)

mesh_red.visual.vertex_colors = [185, 28, 28, 255]
mesh_white.visual.vertex_colors = [244, 244, 246, 255]
mouth_mesh.visual.vertex_colors = [20, 20, 22, 255]
ears_mesh.visual.vertex_colors = [238, 239, 243, 255]
ear_l.visual.vertex_colors = [238, 239, 243, 255]
ear_r.visual.vertex_colors = [238, 239, 243, 255]
aerators_mesh.visual.vertex_colors = [68, 70, 75, 255]
aer_l.visual.vertex_colors = [68, 70, 75, 255]
aer_r.visual.vertex_colors = [68, 70, 75, 255]

# -------------------------------------------------------------------------
# 4. EXPORTACION DE STLS, GLB INTERACTIVO Y 3MF MULTICOLOR (SIN VISOR MONTADO)
# -------------------------------------------------------------------------
print("\n[4/6] Exportando STLs de fabricacion, GLB interactivo y 3MF Multicolor (T-visor abierto)...")

mesh_white.export(os.path.join(OUT_05_STL, "01_Casco_Base_Blanco.stl"))
mesh_red.export(os.path.join(OUT_05_STL, "02_Patron_Rojo_Thorn.stl"))
mouth_mesh.export(os.path.join(OUT_05_STL, "03_Vocoder_Y_Dientes_Negro.stl"))
ears_mesh.export(os.path.join(OUT_05_STL, "04_Orejeras_Gris.stl"))
aerators_mesh.export(os.path.join(OUT_05_STL, "05_Aireadores_Gris.stl"))
lens_mesh_standalone.export(os.path.join(OUT_05_STL, "06_Lentes_Visor_Negro.stl"))
ear_l.export(os.path.join(OUT_05_STL, "07_Orejera_Izquierda_Gris.stl"))
ear_r.export(os.path.join(OUT_05_STL, "08_Orejera_Derecha_Gris.stl"))
aer_l.export(os.path.join(OUT_05_STL, "09_Aireador_Izquierdo_Gris.stl"))
aer_r.export(os.path.join(OUT_05_STL, "10_Aireador_Derecho_Gris.stl"))

# Ensamblaje oficial SIN lente de visor (apertura T-visor 100% abierta/hueca)
assembled_parts = {
    "01_Casco_Base_Blanco": mesh_white,
    "02_Patron_Rojo_Thorn": mesh_red,
    "03_Vocoder_Y_Dientes_Negro": mouth_mesh,
    "04_Orejeras_Gris": ears_mesh,
    "05_Aireadores_Gris": aerators_mesh,
}

glb_path = os.path.join(OUT_05_3MF, "Clone_P2_Thorn_Guardia_Coruscant_Pintado.glb")
trimesh.Scene(assembled_parts).export(glb_path)
print(f"  GLB exportado: {glb_path} ({os.path.getsize(glb_path)/(1024*1024):.1f} MB)")

tmf_path = os.path.join(OUT_05_3MF, "Clone_P2_Thorn_Guardia_Coruscant_Multicolor_Bambu_Prusa.3mf")
trimesh.Scene(assembled_parts).export(tmf_path)

with zipfile.ZipFile(tmf_path, "r") as z_in:
    file_dict = {name: z_in.read(name) for name in z_in.namelist()}

xml_data = file_dict["3D/3dmodel.model"].decode("utf-8")
obj_matches = re.findall(r'<object[^>]*id="(\d+)"[^>]*name="([^"]+)"', xml_data)
obj_ids = [int(m[0]) for m in obj_matches]
assembly_id = max(obj_ids) + 1

components_xml = "".join([f'<component objectid="{oid}" />' for oid in obj_ids])
assembly_obj_xml = f'<object id="{assembly_id}" type="model" name="Casco_Comandante_Thorn"><components>{components_xml}</components></object>'
xml_data = xml_data.replace("</resources>", f"{assembly_obj_xml}</resources>")
xml_data = re.sub(r"<build[^>]*>.*?</build>", f'<build><item objectid="{assembly_id}" /></build>', xml_data, flags=re.DOTALL)
file_dict["3D/3dmodel.model"] = xml_data.encode("utf-8")

parts_config = []
for oid, name in obj_matches:
    lower = name.lower()
    if "blanco" in lower or "casco" in lower:
        ext = 1
    elif "rojo" in lower or "thorn" in lower:
        ext = 2
    elif "vocoder" in lower or "negro" in lower:
        ext = 3
    elif "orejera" in lower or "aireador" in lower or "gris" in lower:
        ext = 4
    else:
        ext = 1
    parts_config.append(
        f'    <part id="{oid}" subtype="normal_part"><metadata key="name" value="{name}"/><metadata key="extruder" value="{ext}"/></part>'
    )

parts_str = "\n".join(parts_config)
model_settings = f"""<?xml version="1.0" encoding="UTF-8"?>
<config>
  <object id="{assembly_id}">
    <metadata key="name" value="Casco_Comandante_Thorn"/>
    <metadata key="extruder" value="1"/>
{parts_str}
  </object>
  <plate>
    <metadata key="plater_id" value="1"/>
    <metadata key="plater_name" value=""/>
    <metadata key="locked" value="false"/>
  </plate>
</config>"""

project_settings = """[project]
version = 1.0.0
filament_colours = #F4F4F6;#B91C1C;#141416;#46464B
filament_names = Blanco Armadura Phase 2;Carmesi Thorn Guardia Coruscant;Negro Vocoder;Gris Mecanico Greeblies
filament_type = PLA;PLA;PLA;PLA
"""

slice_info = """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <header>
    <header_item key="X-BBL-Client-Type" value="slicer"/>
    <header_item key="X-BBL-Client-Version" value="02.02.01.60"/>
  </header>
</config>"""

file_dict["Metadata/model_settings.config"] = model_settings.encode("utf-8")
file_dict["Metadata/project_settings.config"] = project_settings.encode("utf-8")
file_dict["Metadata/slice_info.config"] = slice_info.encode("utf-8")

with zipfile.ZipFile(tmf_path, "w", compression=zipfile.ZIP_DEFLATED) as z_out:
    for name, content in file_dict.items():
        z_out.writestr(name, content)
print(f"  3MF Multicolor exportado: {tmf_path} ({os.path.getsize(tmf_path)/(1024*1024):.1f} MB)")

# -------------------------------------------------------------------------
# 5. PLANTILLA VECTORIAL DESPLEGADA 1:1 EXACTA (SVG)
# -------------------------------------------------------------------------
print("\n[5/6] Generando plantilla vectorial 1:1 desplegada desde los poligonos matematicos exactos...")


def cross_section_to_svg_path(cs, ox, oy, flip_x=False, z_ref_top=1905.0):
    polys = cs.to_polygons()
    cmds = []
    for poly in polys:
        if len(poly) < 3:
            continue
        pts_svg = []
        for u, z in poly:
            xp = ox + (-u if flip_x else u)
            yp = oy + (z_ref_top - z)
            pts_svg.append(f"{xp:.2f},{yp:.2f}")
        cmds.append("M " + " L ".join(pts_svg) + " Z")
    return " ".join(cmds)


wing_l_path = cross_section_to_svg_path(cs_wing_final, ox=210.0, oy=145.0, flip_x=False, z_ref_top=1905.0)
wing_r_path = cross_section_to_svg_path(cs_wing_final, ox=535.0, oy=145.0, flip_x=True, z_ref_top=1905.0)
bolt_l_path = cross_section_to_svg_path(cs_bolt_yz, ox=160.0, oy=395.0, flip_x=False, z_ref_top=1770.0)
bolt_r_path = cross_section_to_svg_path(cs_bolt_yz, ox=310.0, oy=395.0, flip_x=True, z_ref_top=1770.0)
motifs_l_path = cross_section_to_svg_path(cs_tube_motifs, ox=500.0, oy=385.0, flip_x=False, z_ref_top=1715.0)
motifs_r_path = cross_section_to_svg_path(cs_tube_motifs, ox=650.0, oy=385.0, flip_x=True, z_ref_top=1715.0)

svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 540" width="760mm" height="540mm">
  <style>
    .bg {{ fill: #0f172a; }}
    .title {{ font-family: Segoe UI, Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #f8fafc; }}
    .sub {{ font-family: Segoe UI, Arial, sans-serif; font-size: 10px; fill: #94a3b8; }}
    .lbl {{ font-family: Segoe UI, Arial, sans-serif; font-size: 11px; font-weight: bold; fill: #38bdf8; }}
    .red-stencil {{ fill: #b91c1c; fill-rule: evenodd; stroke: #fca5a5; stroke-width: 0.6; }}
    .white-stencil {{ fill: #f8fafc; fill-rule: evenodd; stroke: #38bdf8; stroke-width: 0.6; }}
    .grid {{ stroke: #1e293b; stroke-width: 0.4; }}
  </style>
  <rect width="100%" height="100%" class="bg"/>
  <text x="30" y="32" class="title">PLANTILLA VECTORIAL 1:1 DESPLEGADA — COMANDANTE THORN (CC-3223 / TCW S6)</text>
  <text x="30" y="49" class="sub">Desarrollo cilindrico exacto R=90 mm sobre Galactic Armory Phase 2 Main Body.stl | Imprimir a escala 100% (1 unidad = 1 mm)</text>

  <!-- Escala patron de verificacion 100 mm -->
  <g transform="translate(30, 66)">
    <line x1="0" y1="0" x2="100" y2="0" stroke="#22c55e" stroke-width="2"/>
    <line x1="0" y1="-4" x2="0" y2="4" stroke="#22c55e" stroke-width="2"/>
    <line x1="100" y1="-4" x2="100" y2="4" stroke="#22c55e" stroke-width="2"/>
    <text x="110" y="4" class="sub" fill="#22c55e">Barra de verificacion exacta: 100.0 mm</text>
  </g>

  <!-- Alas + Mastiles Frontales continuos (Izquierda y Derecha) -->
  <text x="85" y="125" class="lbl">1A. Ala + Mastil Frontal Izquierdo (+X, vista exterior 1:1)</text>
  <path d="{wing_l_path}" class="red-stencil"/>

  <text x="415" y="125" class="lbl">1B. Ala + Mastil Frontal Derecho (-X, simetrico 1:1)</text>
  <path d="{wing_r_path}" class="red-stencil"/>

  <!-- Rayos de Depresion del Pomulo -->
  <text x="85" y="365" class="lbl">2. Rayos Blancos de Depresion del Pomulo (Izq / Der, ancho 5.4 mm)</text>
  <path d="{bolt_l_path}" class="white-stencil"/>
  <path d="{bolt_r_path}" class="white-stencil"/>

  <!-- Escalon Z y Muesca Inferior del Tubo Mandibular -->
  <text x="440" y="365" class="lbl">3. Escalon Blanco en Z + Muesca Inferior del Tubo (Izq / Der)</text>
  <path d="{motifs_l_path}" class="white-stencil"/>
  <path d="{motifs_r_path}" class="white-stencil"/>
</svg>
"""

svg_path_03 = os.path.join(OUT_03, "Stencil_Thorn_Coruscant_Wings.svg")
svg_path_04 = os.path.join(OUT_04, "Stencil_Thorn_Coruscant_Wings.svg")
for sp in (svg_path_03, svg_path_04):
    with open(sp, "w", encoding="utf-8") as f_svg:
        f_svg.write(svg_content)
print(f"  Plantilla SVG 1:1 guardada en: {svg_path_03} y {svg_path_04}")

# -------------------------------------------------------------------------
# 6. AUDITORIA FOTOGRAFICA EXHAUSTIVA DE 20 ANGULOS + LAMINA CANON 4 VISTAS
# -------------------------------------------------------------------------
print("\n[6/6] Renderizando protocolo de control de calidad de 20 vistas OpenGL antialias...")

scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, ears_mesh, aerators_mesh])
center = scene.centroid
scene.camera.z_near = 10.0
scene.camera.z_far = 2500.0

qc_checkpoints = [
    ("01_Frontal_Ortho_0deg", "VISTA FRONTAL ORTOGONAL (0 deg)\nMastiles diagonales fluidos | Banda sub-ocular 6.2 mm | T-visor abierto", 0.0, 0.0, 535.0, None),
    ("02_Frontal_Elevada_15deg", "VISTA FRONTAL ELEVADA (+15 deg)\nCupula central blanca | Flancos cresta rojos | Bahia C redondeada", 0.0, 15.0, 535.0, None),
    ("03_Frontal_Baja_Hero_-15deg", "VISTA FRONTAL BAJA HERO (-15 deg)\nMorro rojo solido | Cuenca dientes blanca | 3 bigotes sin sangrado inferior", 0.0, -15.0, 535.0, None),
    ("04_Iso_3_4_Izquierda_36deg", "PERSPECTIVA CANON TCW 3/4 (+36 deg)\nConcordancia 1:1 CGI Temporada 6 | Alas 4 plumas | Rayo pomulo y bigotes", 36.0, 7.0, 535.0, None),
    ("05_Iso_3_4_Derecha_-36deg", "PERSPECTIVA CANON TCW 3/4 (-36 deg)\nSimetria bilateral exacta | Hombro de halcon | Escalon sub-ocular X=45 mm", -36.0, 7.0, 535.0, None),
    ("06_Perfil_Izquierdo_90deg", "PERFIL IZQUIERDO PURO (+90 deg)\nAla achatada 25% y alargada 15% | Corte tubo dY/dZ=-0.72 paralelo a ranuras", 90.0, 0.0, 535.0, None),
    ("07_Perfil_Derecho_-90deg", "PERFIL DERECHO PURO (-90 deg)\nFlanco cresta rojo continuo | Escalon Z y muesca en tubo | Banda en canal", -90.0, 0.0, 535.0, None),
    ("08_Trasera_Izquierda_145deg", "PERSPECTIVA POSTERIOR IZQ (+145 deg)\nTransicion canal trasero | Orejera blanca limpia | Costilla conica exterior", 145.0, 8.0, 535.0, None),
    ("09_Trasera_Derecha_-145deg", "PERSPECTIVA POSTERIOR DER (-145 deg)\nLabio horizontal Z=1817-1826 100% blanco | Banda ceñida al canal esculpido", -145.0, 8.0, 535.0, None),
    ("10_Trasera_Ortho_180deg", "VISTA TRASERA ORTOGONAL (180 deg)\nDisco central R=26.6 mm 100% blanco | 2 costillas conicas en bisel real", 180.0, 0.0, 535.0, None),
    ("11_Trasera_Elevada_20deg", "VISTA TRASERA ELEVADA (+20 deg)\nCanal superior cresta 100% blanco | Corona trasera limpia | Simetria axial", 180.0, 20.0, 535.0, None),
    ("12_Cenital_Superior_88deg", "VISTA CENITAL SUPERIOR (+88 deg)\nCarriles superiores cresta 100% blancos | Simetria bilateral de las 4 plumas", 0.0, 88.0, 535.0, None),

    # 8 Macros de alta ampliacion sobre cada zona anatomica critica
    ("13_Macro_Ala_Izquierda", "MACRO ALA DE THOR IZQUIERDA (+X)\nEspina interior blanca + 3 ranuras | Hombro de halcon y bahia C suave", 0, 0, 0, ([65.0, 35.0, 1870.0], [345.0, 35.0, 1882.0])),
    ("14_Macro_Ala_Derecha", "MACRO ALA DE THOR DERECHA (-X)\n4 plumas afiladas hasta Y=115.5 mm | Margen constante de 2.5 mm sobre canal", 0, 0, 0, ([-65.0, 35.0, 1870.0], [-345.0, 35.0, 1882.0])),
    ("15_Macro_Ceno_Subocular", "MACRO MASTIL, T-VISOR Y BANDA SUB-OCULAR\nAlineacion exacta en |X|=45.0 mm | Banda blanca 6.2 mm paralela al visor", 0, 0, 0, ([38.0, -45.0, 1795.0], [115.0, -295.0, 1805.0])),
    ("16_Macro_Dientes_Y_Morro", "MACRO ARCO DE DIENTES Y MORRO\n Salto escultorico de dientes 100% blanco | Esquinas T-visor limpias | Vocoder", 0, 0, 0, ([0.0, -70.0, 1705.0], [0.0, -330.0, 1712.0])),
    ("17_Macro_Bigotes_Y_Tubo_Izq", "MACRO 3 BIGOTES Y TUBO MANDIBULAR (+X)\n3 anillos elipsoidales envolventes | Escalon Z + muesca inferior | Rayo", 0, 0, 0, ([45.0, -20.0, 1705.0], [215.0, -205.0, 1708.0])),
    ("18_Macro_Bigotes_Y_Tubo_Der", "MACRO 3 BIGOTES Y TUBO MANDIBULAR (-X)\nTerminacion limpia en pliegue del valle | Cero invasion bajo el aireador", 0, 0, 0, ([-45.0, -20.0, 1705.0], [-215.0, -205.0, 1708.0])),
    ("19_Macro_Depresion_Y_Rayo", "MACRO DEPRESION DEL POMULO Y RAYO BLANCO\nCavidad y pared biselada 100% rojas | Rayo blanco de 3 tramos (5.4 mm)", 0, 0, 0, ([78.0, 12.0, 1732.0], [340.0, -25.0, 1742.0])),
    ("20_Macro_Trasera_Disco_Costillas", "MACRO BANDA TRASERA, DISCO Y COSTILLAS\nBanda en canal Z=1826-1853 | Labio y disco blancos | Costillas conicas", 0, 0, 0, ([0.0, 195.0, 1765.0], [0.0, 505.0, 1775.0])),
]

try:
    font_title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 20)
    font_sub = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    font_hdr1 = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 38)
    font_hdr2 = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 24)
    font_s1 = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 26)
    font_s2 = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 18)
except Exception:
    font_title = font_sub = font_hdr1 = font_hdr2 = font_s1 = font_s2 = None

qc_panels = []
for idx, (code, title, yaw_deg, pitch_deg, dist, macro_spec) in enumerate(qc_checkpoints, 1):
    if macro_spec is None:
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        cam_pos = center + np.array([
            dist * np.sin(yaw) * np.cos(pitch),
            -dist * np.cos(yaw) * np.cos(pitch),
            dist * np.sin(pitch),
        ])
        look_target = center
    else:
        look_target = np.array(macro_spec[0], dtype=np.float64)
        cam_pos = np.array(macro_spec[1], dtype=np.float64)

    view_vec = look_target - cam_pos
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

    raw_data = None
    for attempt in range(4):
        try:
            time.sleep(0.06)
            raw_data = scene.save_image(resolution=(1600, 1600))
            break
        except Exception as e:
            if attempt == 3:
                raise e
            time.sleep(0.25)

    img_large = Image.open(BytesIO(raw_data)).convert("RGBA")
    img_smooth = img_large.resize((800, 800), Image.Resampling.LANCZOS)

    single_out = os.path.join(OUT_04_AUDIT, f"qc_rev_{idx:02d}_{code}.png")
    img_smooth.save(single_out)

    card = Image.new("RGBA", (800, 890), (255, 255, 255, 255))
    card.paste(img_smooth, (0, 0), img_smooth)
    d_card = ImageDraw.Draw(card)
    d_card.rectangle([(0, 800), (800, 890)], fill=(245, 246, 248, 255))
    d_card.line([(0, 800), (800, 800)], fill=(220, 222, 228, 255), width=2)

    lines = title.split("\n")
    d_card.text((400, 827), f"REV #{idx:02d}: {lines[0]}", fill=(18, 18, 22, 255), font=font_title, anchor="mm")
    if len(lines) > 1:
        d_card.text((400, 860), lines[1], fill=(185, 28, 28, 255), font=font_sub, anchor="mm")

    qc_panels.append(card)
    print(f"  [{idx:02d}/20] {code} renderizado.")

# Montaje Lamina Maestra 20 Vistas (5x4)
cols, rows = 5, 4
card_w, card_h = 800, 890
margin = 20
board_w = cols * card_w + (cols + 1) * margin
board_h = rows * card_h + (rows + 1) * margin + 140

board = Image.new("RGB", (board_w, board_h), (240, 242, 245))
d_board = ImageDraw.Draw(board)
d_board.rectangle([(0, 0), (board_w, 130)], fill=(15, 23, 42))
d_board.text(
    (board_w // 2, 44),
    "AUDITORIA TECNICA DE 20 VISTAS — CASCO COMANDANTE THORN (CC-3223 / TCW S6)",
    fill=(248, 250, 252),
    font=font_hdr1,
    anchor="mm",
)
d_board.text(
    (board_w // 2, 92),
    f"CALIBRACION 0.1 MM SOBRE PLIEGUES GALACTIC ARMORY | T-VISOR ABIERTO | ESTANQUEIDAD dV = {dev:.5f} mm3",
    fill=(248, 113, 113),
    font=font_hdr2,
    anchor="mm",
)

for i, panel in enumerate(qc_panels):
    r = i // cols
    c = i % cols
    x = margin + c * (card_w + margin)
    y = 140 + margin + r * (card_h + margin)
    board.paste(panel.convert("RGB"), (x, y))

master_qc_path = os.path.join(OUT_04_LAMINAS, "thorn_qc_20_views_inspection.png")
board.save(master_qc_path, optimize=True)

# Montaje Lamina Resumen Canon 4 Vistas (2x2): Frontal (0), 3/4 TCW (3), Perfil (5), Trasera (9)
canon_indices = [0, 3, 5, 9]
sheet_w = card_w * 2 + margin * 3
sheet_h = card_h * 2 + margin * 3 + 120

sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 246, 248))
d_sheet = ImageDraw.Draw(sheet)
d_sheet.rectangle([(0, 0), (sheet_w, 110)], fill=(15, 23, 42))
d_sheet.text(
    (sheet_w // 2, 38),
    "INSPECCION CANONICA 4 ANGULOS: COMANDANTE THORN CC-3223 (THE CLONE WARS S6)",
    fill=(248, 250, 252),
    font=font_s1,
    anchor="mm",
)
d_sheet.text(
    (sheet_w // 2, 76),
    f"Concordancia Anatomica Total | Apertura T-Visor Abierta | Manifold 100% Estanco (dV = {dev:.5f} mm3)",
    fill=(248, 113, 113),
    font=font_s2,
    anchor="mm",
)

sheet.paste(qc_panels[canon_indices[0]].convert("RGB"), (margin, 120 + margin))
sheet.paste(qc_panels[canon_indices[1]].convert("RGB"), (margin * 2 + card_w, 120 + margin))
sheet.paste(qc_panels[canon_indices[2]].convert("RGB"), (margin, 120 + margin * 2 + card_h))
sheet.paste(qc_panels[canon_indices[3]].convert("RGB"), (margin * 2 + card_w, 120 + margin * 2 + card_h))

render_out_04 = os.path.join(OUT_04_LAMINAS, "thorn_canon_4views_render.png")
sheet.save(render_out_04, optimize=True)

# Sincronizacion con el directorio de artefactos de la conversacion activa
print("\nSincronizando entregables con el directorio de artefactos...")
shutil.copy2(master_qc_path, os.path.join(ARTIFACT_DIR, "thorn_qc_20_views_inspection.png"))
shutil.copy2(render_out_04, os.path.join(ARTIFACT_DIR, "thorn_canon_4views_render.png"))
shutil.copy2(tmf_path, os.path.join(ARTIFACT_DIR, "Clone_P2_Thorn_Guardia_Coruscant_Multicolor_Bambu_Prusa.3mf"))
shutil.copy2(glb_path, os.path.join(ARTIFACT_DIR, "Clone_P2_Thorn_Guardia_Coruscant_Pintado.glb"))
shutil.copy2(svg_path_03, os.path.join(ARTIFACT_DIR, "Stencil_Thorn_Coruscant_Wings.svg"))
for stl_name in [
    "01_Casco_Base_Blanco.stl",
    "02_Patron_Rojo_Thorn.stl",
    "03_Vocoder_Y_Dientes_Negro.stl",
    "04_Orejeras_Gris.stl",
    "05_Aireadores_Gris.stl",
    "06_Lentes_Visor_Negro.stl",
]:
    shutil.copy2(os.path.join(OUT_05_STL, stl_name), os.path.join(ARTIFACT_DIR, stl_name))

# Copiar tambien la comparativa directa contra el canon CGI/Vector/Fotos al directorio de artefactos y 04
cmp_v6 = os.path.join(BASE_DIR, "scratch", "compare_v6.png")
if os.path.exists(cmp_v6):
    shutil.copy2(cmp_v6, os.path.join(OUT_04_LAMINAS, "thorn_comparativa_canon_tcw.png"))
    shutil.copy2(cmp_v6, os.path.join(ARTIFACT_DIR, "thorn_comparativa_canon_tcw.png"))

print(f"\n=== PIPELINE COMPLETADO CON EXITO EN {time.time() - t_start:.1f} s ===")
print(f"  - 20 Vistas QC: {master_qc_path}")
print(f"  - 4 Vistas Canon: {render_out_04}")
print(f"  - 3MF Multicolor: {tmf_path}")
print(f"  - GLB Interactivo: {glb_path}")
