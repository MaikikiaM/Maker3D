import os
import time
import numpy as np
import trimesh
import manifold3d as m3d
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import trimesh.viewer.windowed

# Window resize patch for Windows pyglet
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

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
casco_dir = os.path.join(BASE_DIR, r"Diseños 3D\Casco")
greeb_dir = os.path.join(casco_dir, "Greeblies")

print("[1] Cargando geometria base Galactic Armory...")
mb = trimesh.load(os.path.join(casco_dir, "Phase 2 Main Body.stl"))
v_base = mb.vertices.astype('float32')
f_base = mb.faces.astype('uint32')
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))
base_vol = helmet_mani.volume()
print(f"Casco base cargado: {len(v_base)} verts, {len(f_base)} tris, vol = {base_vol:.1f} mm3")

mouth_mesh = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
ear_l = trimesh.load(os.path.join(greeb_dir, "Ear Piece Left.stl"))
ear_r = trimesh.load(os.path.join(greeb_dir, "Ear Piece Right.stl"))
ears_mesh = trimesh.util.concatenate([ear_l, ear_r])
aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))
aerators_mesh = trimesh.util.concatenate([aer_l, aer_r])

lens_path = os.path.join(casco_dir, r"Antigravity\05_Cascos_Pintados_Clones\06_Lentes_Visor_Negro.stl")
if os.path.exists(lens_path):
    lens_mesh = trimesh.load(lens_path)
else:
    theta = np.linspace(-np.radians(60), np.radians(60), 30)
    lx = 92.0 * np.sin(theta)
    ly = 10.0 - 58.0 * np.cos(theta)
    pts = []
    for i in range(len(theta)):
        pts.append([lx[i], ly[i], 1750.0])
        pts.append([lx[i], ly[i], 1805.0])
    lh = trimesh.convex.convex_hull(np.vstack([pts, np.array(pts) + [0, 4, 0]]))
    lens_mesh = trimesh.Trimesh(vertices=lh.vertices, faces=lh.faces)

print("[2] Construyendo cortadores de alta precision V12...")

# =========================================================================
# A. CRESTA DORSAL (Fin)
# =========================================================================
fin_poly_yz = np.array([
    [-12.0, 1918.0],
    [-2.0,  1885.0],
    [205.0, 1870.0],
    [205.0, 1960.0],
    [-12.0, 1960.0]
], dtype=np.float32)
cs_fin = m3d.CrossSection([fin_poly_yz])
raw_fin = cs_fin.extrude(36.0)
m_rf = raw_fin.to_mesh()
v_rf = m_rf.vert_properties
v_fin_3d = np.column_stack([v_rf[:, 2] - 18.0, v_rf[:, 0], v_rf[:, 1]]).astype('float32')
f_fin_3d = m_rf.tri_verts.copy()
fin_cutter = m3d.Manifold(m3d.Mesh(vert_properties=v_fin_3d, tri_verts=f_fin_3d))
if fin_cutter.volume() <= 0:
    f_fin_3d[:, [0, 1]] = f_fin_3d[:, [1, 0]]
    fin_cutter = m3d.Manifold(m3d.Mesh(vert_properties=v_fin_3d, tri_verts=f_fin_3d))

# =========================================================================
# B. CEÑO Y FRENTE CON V BLANCA CENTRAL
# =========================================================================
# brow_lower: comienza estrictamente en Z = 1808.6 mm (borde superior de la abertura del visor).
# Se extiende hasta Y = +60 mm sobre la mejilla, tocando la base de la orejera.
brow_lower = m3d.Manifold.cube([270.0, 160.0, 16.5], center=False).translate([-135.0, -100.0, 1808.6])
# brow_upper: frente en la cúpula (Z = 1825.02 a 1874.0 mm). Termina en Y = -16.0 mm.
# Deja un margen blanco limpio de ~28 mm en la cúpula antes del inicio del ala (Y = +12.0 mm).
brow_upper = m3d.Manifold.cube([270.0, 84.0, 49.0], center=False).translate([-135.0, -100.0, 1825.02])
brow_solid = brow_lower + brow_upper

# Cuña central en V
wedge_verts = np.array([
    [-14.0, -105.0, 1806.0], [ 14.0, -105.0, 1806.0], [ 14.0,  30.0, 1806.0], [-14.0,  30.0, 1806.0],
    [-23.0, -105.0, 1876.0], [ 23.0, -105.0, 1876.0], [ 23.0,  30.0, 1876.0], [-23.0,  30.0, 1876.0],
], dtype=np.float32)
wedge_faces = np.array([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]
], dtype=np.uint32)
wedge_mani = m3d.Manifold(m3d.Mesh(vert_properties=wedge_verts, tri_verts=wedge_faces))
brow_cutter = brow_solid - wedge_mani

# =========================================================================
# C. ALAS DE THOR EN LA CÚPULA (Elegantes, dinámicas, 4 plumas + 3 ranuras profundas)
# =========================================================================
# Mástil curvado orgánico: base ensanchada que empalma con la costura de la cúpula (Z = 1825.0 mm),
# luego sube con un cuello esbelto y curvado hacia atrás siguiendo la cúpula.
# Pluma 1 (superior): llega hasta Y ≈ 142 mm, Z ≈ 1895 mm
# Pluma 2: punta en Y ≈ 126 mm, Z ≈ 1878 mm
# Pluma 3: punta en Y ≈ 110 mm, Z ≈ 1860 mm
# Pluma 4 (inferior): punta en Y ≈ 94 mm, Z ≈ 1840 mm, base corriendo a lo largo de Z = 1825 mm
y_mast_outer = [12.0, 13.5, 16.0, 20.0, 26.0, 35.0, 48.0, 65.0, 85.0, 108.0, 126.0, 142.0]
z_mast_outer = [1825.0, 1832.0, 1845.0, 1860.0, 1875.0, 1890.0, 1904.0, 1913.0, 1915.0, 1912.0, 1905.0, 1895.0]

# Puntas de plumas con corte inclinado forward-raked (~45°)
# Pluma 1 tip: (142, 1895) -> ranura 1
# Pluma 2 tip: (126, 1878)
# Pluma 3 tip: (110, 1860)
# Pluma 4 tip: (94, 1840)
y_feathers_trail = [
    142.0,  # Pluma 1 tip
    126.0,  # Pluma 2 tip
    110.0,  # Pluma 3 tip
    94.0,   # Pluma 4 tip
    80.0, 65.0, 48.0, 32.0, 20.0, 12.0  # Borde inferior de la pluma 4 retornando a la base
]
z_feathers_trail = [
    1895.0,
    1878.0,
    1860.0,
    1840.0,
    1834.0, 1829.0, 1826.0, 1825.0, 1825.0, 1825.0
]

pts_wing = []
for y, z in zip(y_mast_outer, z_mast_outer):
    pts_wing.append([y, z])
# Añadir escalones del trailing edge
pts_wing.append([126.0, 1878.0])
pts_wing.append([110.0, 1860.0])
for y, z in zip(y_feathers_trail[3:], z_feathers_trail[3:]):
    pts_wing.append([y, z])

cs_wing_base = m3d.CrossSection([pts_wing[::-1]])

def make_curved_slot(pts_spine, thickness=3.5):
    pts_spine = np.array(pts_spine, dtype=np.float32)
    diffs = np.diff(pts_spine, axis=0)
    normals = np.column_stack([-diffs[:, 1], diffs[:, 0]])
    lens = np.hypot(normals[:, 0], normals[:, 1])[:, None]
    normals = normals / np.maximum(lens, 1e-6)
    
    v_norms = np.zeros_like(pts_spine)
    v_norms[0] = normals[0]
    v_norms[-1] = normals[-1]
    for i in range(1, len(pts_spine)-1):
        v_norms[i] = (normals[i-1] + normals[i]) / 2.0
        v_norms[i] /= np.maximum(np.hypot(v_norms[i, 0], v_norms[i, 1]), 1e-6)
    
    half_th = thickness / 2.0
    side_top = pts_spine + v_norms * half_th
    side_bot = pts_spine - v_norms * half_th
    poly = np.vstack([side_top, side_bot[::-1]])
    cs = m3d.CrossSection([poly])
    if cs.area() == 0:
        cs = m3d.CrossSection([poly[::-1]])
    return cs

# 3 ranuras curvadas profundas: comienzan cerca del mástil delantero y cortan hasta más allá de las puntas
# Ranura 1: entre Pluma 1 y Pluma 2
s1 = make_curved_slot([[35.0, 1891.0], [60.0, 1897.0], [95.0, 1894.0], [145.0, 1883.0]], thickness=3.5)
# Ranura 2: entre Pluma 2 y Pluma 3
s2 = make_curved_slot([[27.0, 1871.0], [50.0, 1877.0], [82.0, 1874.0], [130.0, 1865.0]], thickness=3.5)
# Ranura 3: entre Pluma 3 y Pluma 4
s3 = make_curved_slot([[21.0, 1851.0], [42.0, 1856.0], [70.0, 1853.0], [115.0, 1845.0]], thickness=3.2)

cs_wing = cs_wing_base - (s1 + s2 + s3)
raw_w = cs_wing.extrude(115.0)
m_rw = raw_w.to_mesh()
vw = m_rw.vert_properties
fw = m_rw.tri_verts

# Ala Derecha (+X)
vw_r = np.column_stack([vw[:, 2] + 28.0, vw[:, 0], vw[:, 1]]).astype('float32')
tri_r = fw.copy()
mani_w_r = m3d.Manifold(m3d.Mesh(vert_properties=vw_r, tri_verts=tri_r))
if mani_w_r.volume() <= 0:
    tri_r[:, [0, 1]] = tri_r[:, [1, 0]]
    mani_w_r = m3d.Manifold(m3d.Mesh(vert_properties=vw_r, tri_verts=tri_r))

# Ala Izquierda (-X)
vw_l = np.column_stack([-(vw[:, 2] + 28.0), vw[:, 0], vw[:, 1]]).astype('float32')
tri_l = tri_r.copy()
tri_l[:, [0, 1]] = tri_l[:, [1, 0]]
mani_w_l = m3d.Manifold(m3d.Mesh(vert_properties=vw_l, tri_verts=tri_l))

wings_cutter = mani_w_r + mani_w_l

# =========================================================================
# D. ZONA TRASERA (Banda arqueada sobre cúpula Z=1832-1868 + Costillas continuas)
# =========================================================================
band_box = m3d.Manifold.cube([150.0, 55.0, 36.0], center=False).translate([-75.0, 172.0, 1832.0])
disc_cyl = m3d.Manifold.cylinder(80.0, 25.0, 25.0, 128).rotate([90, 0, 0]).translate([0, 235.0, 1819.8])

rib_pts = np.array([
    [13.0, 1616.0],
    [22.5, 1616.0],
    [24.5, 1750.0],
    [28.5, 1800.0],
    [32.5, 1820.0],
    [30.5, 1844.0],
    [22.0, 1844.0],
    [24.5, 1820.0],
    [19.0, 1800.0],
    [14.0, 1750.0],
    [13.0, 1616.0]
], dtype=np.float32)

cs_rib_r = m3d.CrossSection([rib_pts])
raw_rib_r = cs_rib_r.extrude(60.0)
m_rrr = raw_rib_r.to_mesh()
v_rrr = m_rrr.vert_properties
v_rib_r_3d = np.column_stack([v_rrr[:, 0], v_rrr[:, 2] + 175.0, v_rrr[:, 1]]).astype('float32')
f_rib_r_3d = m_rrr.tri_verts.copy()
rib_r_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_rib_r_3d, tri_verts=f_rib_r_3d))
if rib_r_mani.volume() <= 0:
    f_rib_r_3d[:, [0, 1]] = f_rib_r_3d[:, [1, 0]]
    rib_r_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_rib_r_3d, tri_verts=f_rib_r_3d))

v_rib_l_3d = v_rib_r_3d.copy()
v_rib_l_3d[:, 0] = -v_rib_l_3d[:, 0]
f_rib_l_3d = f_rib_r_3d.copy()
f_rib_l_3d[:, [0, 1]] = f_rib_l_3d[:, [1, 0]]
rib_l_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_rib_l_3d, tri_verts=f_rib_l_3d))

rear_markings = (band_box + rib_r_mani + rib_l_mani) - disc_cyl

# =========================================================================
# E. MORRO, MENTÓN Y MEJILLAS (GEOMETRÍA LIMPIA Y RIGUROSA)
# =========================================================================
# 1. Morro central: cubre completamente el mentón y morro hasta la base exacta de la abertura del visor (Z = 1741.8 mm).
# NUNCA sobrepasa Z = 1741.8 mm en el centro, por lo que NO invade el visor y NO requiere recortar el morro.
snout_center = m3d.Manifold.cube([88.0, 65.0, 125.8], center=False).translate([-44.0, -100.0, 1616.0])

# 2. Mejillas y solapas:
# - Comienzan estrictamente a partir de |X| >= 8.85 mm para NO entrar en la hendidura vertical del visor (que termina en X = ±8.85 mm).
# - En Z suben hasta Z = 1808.6 mm (uniéndose al ceño).
# - Cubren las 2 primeras ranuras de ventilación (Y <= -19.5 mm) en el tubo inferior.
# - Escalón en Z = 1658 mm hasta Y = +52.0 mm cubriendo el suelo y la pared trasera vertical de la depresión de mejilla.
# - Repisa horizontal sub-ocular entre Z = 1789.5 y 1798.0 mm.
cheek_poly_yz = np.array([
    [-100.0, 1616.0],
    [ -19.5, 1616.0],  # Tubo inferior: cubre ranuras 1 y 2
    [ -19.5, 1658.0],  # Escalón al fondo de la depresión
    [  52.0, 1658.0],  # Pared trasera de la depresión
    [  52.0, 1808.6],  # Sube vertical por la pared trasera hasta el ceño
    [-25.0,  1808.6],  # Ceño
    [-25.0,  1798.0],  # Repisa sub-ocular
    [-100.0, 1798.0]   # Solapa frontal
], dtype=np.float32)

cs_cheek = m3d.CrossSection([cheek_poly_yz])
if cs_cheek.area() == 0:
    cs_cheek = m3d.CrossSection([cheek_poly_yz[::-1]])

raw_cheek = cs_cheek.extrude(110.0)
m_rc = raw_cheek.to_mesh()
v_rc = m_rc.vert_properties
f_rc = m_rc.tri_verts

# Mejilla Derecha: comienza en X = +8.85 mm
v_cr = np.column_stack([v_rc[:, 2] + 8.85, v_rc[:, 0], v_rc[:, 1]]).astype('float32')
f_cr = f_rc.copy()
cheek_r = m3d.Manifold(m3d.Mesh(vert_properties=v_cr, tri_verts=f_cr))
if cheek_r.volume() <= 0:
    f_cr[:, [0, 1]] = f_cr[:, [1, 0]]
    cheek_r = m3d.Manifold(m3d.Mesh(vert_properties=v_cr, tri_verts=f_cr))

# Mejilla Izquierda: comienza en X = -8.85 mm
v_cl = np.column_stack([-(v_rc[:, 2] + 8.85), v_rc[:, 0], v_rc[:, 1]]).astype('float32')
f_cl = f_cr.copy()
f_cl[:, [0, 1]] = f_cl[:, [1, 0]]
cheek_l = m3d.Manifold(m3d.Mesh(vert_properties=v_cl, tri_verts=f_cl))

cheeks_raw = cheek_r + cheek_l

# 3. Franja blanca sub-ocular parcial (58.3% del vano de la visera horizontal):
# Delimitada estrictamente entre |X| = 24.5 mm y |X| = 42.0 mm (span 17.5 mm sobre vano de 35 mm).
# Altura de corte: de Z = 1789.5 a 1796.5 mm (solo la repisa/borde superior bajo el visor).
# Hacia el centro (|X| < 24.5 mm): solapa de mejilla ROJA.
# En el extremo lateral (|X| > 42.0 mm): esquina ROJA.
suboc_box_r = m3d.Manifold.cube([17.5, 30.0, 7.0], center=False).translate([24.5, -65.0, 1789.5])
suboc_box_l = m3d.Manifold.cube([17.5, 30.0, 7.0], center=False).translate([-42.0, -65.0, 1789.5])
suboc_cutter = suboc_box_r + suboc_box_l

# 4. 3 Garras blancas sobre los tubos de los aireadores y el mentón:
# Colocadas visiblemente sobre el montículo curvado del mentón/tubo (Z en [1655, 1705] mm):
# Inclinadas ligeramente hacia arriba hacia las branquias (siguiendo el tubo).
# Cruzan ampliamente el eje vertical del aireador (|X| = 47.44 mm) extendiéndose hacia el mentón hasta X ≈ 28-32 mm.
# Mantienen un margen de seguridad de >12 mm respecto al vocoder negro (|X| <= 15 mm).
claws_r = m3d.Manifold()
claws_l = m3d.Manifold()
claw_configs = [
    # (zc, yc, xc, length_x, thick_z, rot_x, rot_y, rot_z)
    (1688.0, -66.0, 48.0, 42.0, 7.0, 12.0, -4.0, -10.0), # Garra 1 (superior): prominente en el bulbo
    (1673.0, -62.0, 50.0, 42.0, 7.0, 12.0, -4.0, -10.0), # Garra 2 (media): centro del tubo
    (1658.0, -58.0, 52.0, 42.0, 6.5, 12.0, -4.0, -10.0), # Garra 3 (inferior): borde superior del aireador
]
for zc, yc, xc, length_x, thick_z, rot_x, rot_y, rot_z in claw_configs:
    cr = m3d.Manifold.cube([length_x, 60.0, thick_z], center=True).rotate([rot_x, rot_y, rot_z]).translate([xc, yc, zc])
    cl = m3d.Manifold.cube([length_x, 60.0, thick_z], center=True).rotate([rot_x, -rot_y, -rot_z]).translate([-xc, yc, zc])
    claws_r = claws_r + cr
    claws_l = claws_l + cl

claws_cutter = claws_r + claws_l

face_cutter = (snout_center + cheeks_raw) - suboc_cutter - claws_cutter

# Cortador de protección estricta del visor (Hollow Visor Keep-Out Volume):
# Se ubica estrictamente dentro de la abertura hueca:
# Z de 1742.0 a 1808.5 mm (nunca por debajo de 1742 mm para no tocar el morro, nunca por encima de 1808.5 mm para no tocar el ceño).
visor_clear_h = m3d.Manifold.cube([180.0, 60.0, 13.0], center=False).translate([-90.0, -80.0, 1795.5])
visor_clear_v = m3d.Manifold.cube([  17.5, 50.0, 53.5], center=False).translate([ -8.75, -80.0, 1742.0])
visor_clearance = visor_clear_h + visor_clear_v

total_red_cutter = (fin_cutter + brow_cutter + wings_cutter + rear_markings + face_cutter) - visor_clearance

print("[3] Ejecutando corte booleano exacto manifold...")
red_mani = helmet_mani ^ total_red_cutter
white_mani = helmet_mani - total_red_cutter

vol_red = red_mani.volume()
vol_white = white_mani.volume()
vol_sum = vol_red + vol_white
dev = abs(vol_sum - base_vol)

print(f"Patron Rojo: volumen = {vol_red:.1f} mm3, triangulos = {red_mani.num_tri()}")
print(f"Base Blanca: volumen = {vol_white:.1f} mm3, triangulos = {white_mani.num_tri()}")
print(f"Conservacion volumetrica: Suma = {vol_sum:.1f} mm3 | Desviacion = {dev:.4f} mm3")

m_red = red_mani.to_mesh()
mesh_red = trimesh.Trimesh(vertices=m_red.vert_properties[:, :3], faces=m_red.tri_verts, process=False)
mesh_red.visual.vertex_colors = [185, 28, 28, 255]

m_white = white_mani.to_mesh()
mesh_white = trimesh.Trimesh(vertices=m_white.vert_properties[:, :3], faces=m_white.tri_verts, process=False)
mesh_white.visual.vertex_colors = [244, 244, 246, 255]

mouth_mesh.visual.vertex_colors = [20, 20, 22, 255]
ears_mesh.visual.vertex_colors = [240, 240, 244, 255]
aerators_mesh.visual.vertex_colors = [70, 70, 75, 255]
lens_mesh.visual.vertex_colors = [15, 15, 18, 255]

scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, ears_mesh, aerators_mesh, lens_mesh])
center = scene.centroid

scene.camera.z_near = 10.0
scene.camera.z_far = 2500.0

test_views = [
    ("v12_01_front_ortho.png", 0.0, 0.0, 820.0, None),
    ("v12_02_front_elevated.png", 0.0, 15.0, 820.0, None),
    ("v12_03_iso_3_4_left.png", 35.0, 8.0, 820.0, None),
    ("v12_04_profile_left.png", 90.0, 0.0, 820.0, None),
    ("v12_05_rear_ortho.png", 180.0, 0.0, 820.0, None),
    ("v12_06_macro_visor_shelf.png", 0, 0, 0, ([0.0, -75.0, 1785.0], [0.0, -75.0 - 320.0, 1785.0 + 10.0])),
    ("v12_07_macro_claws.png", 0, 0, 0, ([40.0, -60.0, 1665.0], [40.0 + 160.0, -60.0 - 240.0, 1665.0 + 30.0])),
    ("v12_08_macro_wing.png", 0, 0, 0, ([75.0, 70.0, 1865.0], [75.0 + 320.0, 70.0, 1865.0 + 15.0])),
    ("v12_09_macro_rear_disc.png", 0, 0, 0, ([0.0, 205.0, 1825.0], [0.0, 205.0 + 340.0, 1825.0 + 10.0]))
]

print("[4] Renderizando vistas de inspeccion V12...")
for filename, yaw_deg, pitch_deg, dist, macro_spec in test_views:
    if macro_spec is not None:
        target, eye = macro_spec
        fwd = np.array(target) - np.array(eye)
        fwd /= np.linalg.norm(fwd)
        up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        actual_up = np.cross(right, fwd)
        actual_up /= np.linalg.norm(actual_up)
        
        R = np.eye(4)
        R[:3, 0] = right
        R[:3, 1] = actual_up
        R[:3, 2] = -fwd
        R[:3, 3] = eye
        scene.camera_transform = R
    else:
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        eye = center + np.array([
            dist * np.sin(yaw) * np.cos(pitch),
            -dist * np.cos(yaw) * np.cos(pitch),
            dist * np.sin(pitch)
        ])
        fwd = center - eye
        fwd /= np.linalg.norm(fwd)
        up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(fwd, up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-4:
            right = np.array([1, 0, 0], dtype=np.float32)
        else:
            right /= r_norm
        actual_up = np.cross(right, fwd)
        actual_up /= np.linalg.norm(actual_up)
        
        R = np.eye(4)
        R[:3, 0] = right
        R[:3, 1] = actual_up
        R[:3, 2] = -fwd
        R[:3, 3] = eye
        scene.camera_transform = R
    
    png_bytes = scene.save_image(resolution=(1024, 1024))
    with open(filename, "wb") as f:
        f.write(png_bytes)
    print(f"   Renderizado: {filename}")

print("\n[OK] V12 renders completados con exito.")
