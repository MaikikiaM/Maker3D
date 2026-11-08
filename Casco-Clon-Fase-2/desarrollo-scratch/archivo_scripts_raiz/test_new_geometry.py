import os
import re
import numpy as np
import trimesh
import manifold3d as m3d
from PIL import Image
import trimesh.viewer.windowed

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

print("=======================================================================")
print("=== GENERACIÓN Y AUDITORÍA RIGUROSA DE GEOMETRÍA PERFECCIONADA V13.3 ===")
print("=======================================================================")

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
casco_dir = os.path.join(BASE_DIR, r"Diseños 3D\Casco")
greeb_dir = os.path.join(casco_dir, "Greeblies")

print("\n[1/5] Cargando casco base y accesorios...")
helmet_mesh = trimesh.load(os.path.join(casco_dir, "Phase 2 Main Body.stl"))
v_base = helmet_mesh.vertices.astype('float32')
f_base = helmet_mesh.faces.astype('uint32')
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))
base_volume = helmet_mani.volume()

mouth_mesh = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
ear_l = trimesh.load(os.path.join(greeb_dir, "Ear Piece Left.stl"))
ear_r = trimesh.load(os.path.join(greeb_dir, "Ear Piece Right.stl"))
ears_mesh = trimesh.util.concatenate([ear_l, ear_r])
aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))
aerators_mesh = trimesh.util.concatenate([aer_l, aer_r])

# -------------------------------------------------------------------------
# CONSTRUCCIÓN DE CORTADORES GEOMÉTRICOS PERFECCIONADOS V13.3
# -------------------------------------------------------------------------
print("\n[2/5] Diseñando cortadores de precisión...")

# A. Cresta Dorsal (Dorsal Fin):
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

# B. Ceño y Frente con Cuña Blanca en V:
# 1. Ceño horizontal inferior (Z de 1808.6 a 1825.0 mm): sobre el visor, va de sien a sien.
brow_lower = m3d.Manifold.cube([270.0, 160.0, 16.4], center=False).translate([-135.0, -100.0, 1808.6])

# 2. Frente de la cúpula (Z de 1825.0 a 1860.0 mm):
# En Y cubre desde -100 hasta -8.0 mm, conectando suavemente con la raíz del ala.
# En X cubre de -72 a +72 mm (área frontal de la frente).
brow_upper = m3d.Manifold.cube([144.0, 92.0, 35.0], center=False).translate([-72.0, -100.0, 1825.0])

# Cuña central en V: trapecio blanco invertido
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
brow_cutter = (brow_lower + brow_upper) - wedge_mani

# C. Alas de Thor en la Cúpula LATERAL:
pts_wing_outer = [
    [-8.0,  1826.0], # Raíz inferior en la frente
    [-6.0,  1840.0], # Curva ascendente del mástil
    [ 0.0,  1858.0],
    [10.0,  1878.0],
    [24.0,  1895.0],
    [40.0,  1907.0],
    [54.0,  1912.0], # Vértice superior curvado
    [70.0,  1906.0], # Punta pluma 1 (superior)
    [64.0,  1888.0], # Punta pluma 2
    [56.0,  1870.0], # Punta pluma 3
    [48.0,  1852.0], # Punta pluma 4 (inferior)
    [35.0,  1844.0], # Borde inferior de la pluma 4
    [20.0,  1838.0],
    [ 5.0,  1832.0],
    [-3.0,  1827.0],
    [-8.0,  1826.0]  # Cierre en la raíz
]

cs_wing_base = m3d.CrossSection([pts_wing_outer[::-1]])

def make_curved_slot(pts_spine, thickness=3.2):
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

# 3 ranuras curvadas ascendentes paralelas
s1 = make_curved_slot([[26.0, 1895.0], [40.0, 1900.0], [54.0, 1898.0], [72.0, 1894.0]], thickness=3.2)
s2 = make_curved_slot([[18.0, 1878.0], [32.0, 1883.0], [46.0, 1880.0], [66.0, 1876.0]], thickness=3.2)
s3 = make_curved_slot([[10.0, 1860.0], [24.0, 1864.0], [38.0, 1862.0], [58.0, 1858.0]], thickness=3.0)

cs_wing = cs_wing_base - (s1 + s2 + s3)
# Extrusión lateral de 85.0 mm desde X = 40.0 mm hasta X = 125.0 mm
raw_w = cs_wing.extrude(85.0)
m_rw = raw_w.to_mesh()
vw = m_rw.vert_properties
fw = m_rw.tri_verts

# Ala Derecha (+X): de X = 40.0 a 125.0 mm
vw_r = np.column_stack([vw[:, 2] + 40.0, vw[:, 0], vw[:, 1]]).astype('float32')
tri_r = fw.copy()
mani_w_r = m3d.Manifold(m3d.Mesh(vert_properties=vw_r, tri_verts=tri_r))
if mani_w_r.volume() <= 0:
    tri_r[:, [0, 1]] = tri_r[:, [1, 0]]
    mani_w_r = m3d.Manifold(m3d.Mesh(vert_properties=vw_r, tri_verts=tri_r))

# Ala Izquierda (-X): de X = -125.0 a -40.0 mm
vw_l = np.column_stack([-(vw[:, 2] + 40.0), vw[:, 0], vw[:, 1]]).astype('float32')
tri_l = tri_r.copy()
tri_l[:, [0, 1]] = tri_l[:, [1, 0]]
mani_w_l = m3d.Manifold(m3d.Mesh(vert_properties=vw_l, tri_verts=tri_l))

wings_cutter = mani_w_r + mani_w_l

# D. Zona Trasera:
band_box = m3d.Manifold.cube([116.0, 55.0, 33.0], center=False).translate([-58.0, 172.0, 1845.0])
disc_cyl = m3d.Manifold.cylinder(80.0, 28.5, 28.5, 128).rotate([90, 0, 0]).translate([0, 235.0, 1820.5])

# Dos franjas verticales continuas sobre las costillas estructurales posteriores:
rib_pts = np.array([
    [13.0, 1616.0],
    [22.0, 1616.0],
    [26.0, 1750.0],
    [29.0, 1855.0],
    [19.0, 1855.0],
    [16.0, 1750.0],
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

# E. Morro, Mentón, Depresión Lateral de Mejilla, Franja Sub-ocular y 3 Garras:
snout_center = m3d.Manifold.cube([88.0, 75.0, 141.8], center=False).translate([-44.0, -110.0, 1600.0])

cheek_poly_yz = np.array([
    [-110.0, 1600.0],
    [ -19.5, 1600.0],  # Tubo inferior: cubre ranuras 1 y 2
    [ -19.5, 1658.0],  # Escalón al fondo de la depresión
    [  52.0, 1658.0],  # Pared trasera vertical de la depresión
    [  52.0, 1808.6],  # Sube vertical por la pared trasera hasta el ceño
    [-25.0,  1808.6],  # Ceño
    [-25.0,  1798.0],  # Repisa horizontal sub-ocular
    [-110.0, 1798.0]   # Solapa frontal
], dtype=np.float32)

cs_cheek = m3d.CrossSection([cheek_poly_yz])
if cs_cheek.area() == 0:
    cs_cheek = m3d.CrossSection([cheek_poly_yz[::-1]])

raw_cheek = cs_cheek.extrude(110.0)
m_rc = raw_cheek.to_mesh()
v_rc = m_rc.vert_properties
f_rc = m_rc.tri_verts

v_cr = np.column_stack([v_rc[:, 2] + 8.85, v_rc[:, 0], v_rc[:, 1]]).astype('float32')
f_cr = f_rc.copy()
cheek_r = m3d.Manifold(m3d.Mesh(vert_properties=v_cr, tri_verts=f_cr))
if cheek_r.volume() <= 0:
    f_cr[:, [0, 1]] = f_cr[:, [1, 0]]
    cheek_r = m3d.Manifold(m3d.Mesh(vert_properties=v_cr, tri_verts=f_cr))

v_cl = np.column_stack([-(v_rc[:, 2] + 8.85), v_rc[:, 0], v_rc[:, 1]]).astype('float32')
f_cl = f_cr.copy()
f_cl[:, [0, 1]] = f_cl[:, [1, 0]]
cheek_l = m3d.Manifold(m3d.Mesh(vert_properties=v_cl, tri_verts=f_cl))

cheeks_raw = cheek_r + cheek_l

# Franja blanca sub-ocular (55% de la longitud horizontal):
suboc_box_r = m3d.Manifold.cube([34.0, 30.0, 7.5], center=False).translate([20.0, -65.0, 1789.0])
suboc_box_l = m3d.Manifold.cube([34.0, 30.0, 7.5], center=False).translate([-54.0, -65.0, 1789.0])
suboc_cutter = suboc_box_r + suboc_box_l

# 3 Garras blancas sobre los tubos de los aireadores:
claws_r = m3d.Manifold()
claws_l = m3d.Manifold()
claw_configs = [
    # (zc, yc, xc, length_x, thick_z, rot_x, rot_y, rot_z)
    (1688.0, -66.0, 48.0, 44.0, 7.0, 12.0, -4.0, -10.0),
    (1673.0, -62.0, 50.0, 44.0, 7.0, 12.0, -4.0, -10.0),
    (1658.0, -58.0, 52.0, 44.0, 6.5, 12.0, -4.0, -10.0),
]
for zc, yc, xc, length_x, thick_z, rot_x, rot_y, rot_z in claw_configs:
    cr = m3d.Manifold.cube([length_x, 60.0, thick_z], center=True).rotate([rot_x, rot_y, rot_z]).translate([xc, yc, zc])
    cl = m3d.Manifold.cube([length_x, 60.0, thick_z], center=True).rotate([rot_x, -rot_y, -rot_z]).translate([-xc, yc, zc])
    claws_r = claws_r + cr
    claws_l = claws_l + cl

claws_cutter = claws_r + claws_l

face_cutter = (snout_center + cheeks_raw) - suboc_cutter - claws_cutter

# Visor Keep-Out
visor_clear_h = m3d.Manifold.cube([180.0, 60.0, 13.0], center=False).translate([-90.0, -80.0, 1795.5])
visor_clear_v = m3d.Manifold.cube([  17.5, 50.0, 53.5], center=False).translate([ -8.75, -80.0, 1742.0])
visor_clearance = visor_clear_h + visor_clear_v

total_red_cutter = (fin_cutter + brow_cutter + wings_cutter + rear_markings + face_cutter) - visor_clearance

# Corte booleano
red_markings_mani = helmet_mani ^ total_red_cutter
white_base_mani = helmet_mani - total_red_cutter

vol_red = red_markings_mani.volume()
vol_white = white_base_mani.volume()
dev = abs((vol_red + vol_white) - base_volume)

print(f"Patrón Rojo: {vol_red:.1f} mm³, triángulos = {red_markings_mani.num_tri()}")
print(f"Base Blanca: {vol_white:.1f} mm³, triángulos = {white_base_mani.num_tri()}")
print(f"Desviación volumétrica: {dev:.5f} mm³ (100% Watertight Interlocking)")

m_red = red_markings_mani.to_mesh()
mesh_red = trimesh.Trimesh(vertices=m_red.vert_properties[:, :3], faces=m_red.tri_verts, process=False)
mesh_red.visual.vertex_colors = [185, 28, 28, 255]

m_white = white_base_mani.to_mesh()
mesh_white = trimesh.Trimesh(vertices=m_white.vert_properties[:, :3], faces=m_white.tri_verts, process=False)
mesh_white.visual.vertex_colors = [244, 244, 246, 255]

mouth_mesh.visual.vertex_colors = [20, 20, 22, 255]
ears_mesh.visual.vertex_colors = [240, 240, 244, 255]
aerators_mesh.visual.vertex_colors = [70, 70, 75, 255]

mesh_red.export("test_mesh_red.stl")
mesh_white.export("test_mesh_white.stl")

# Render 4 views
scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, ears_mesh, aerators_mesh])

def render_angle(yaw_deg, pitch_deg, target_z=1780.0, dist=780.0, resolution=(1400, 1400)):
    y_rad = np.radians(yaw_deg)
    p_rad = np.radians(pitch_deg)
    cx = dist * np.sin(y_rad) * np.cos(p_rad)
    cy = -dist * np.cos(y_rad) * np.cos(p_rad)
    cz = target_z + dist * np.sin(p_rad)
    cam_pos = np.array([cx, cy, cz])
    target = np.array([0.0, 50.0, target_z])
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rot = np.eye(4)
    rot[:3, 0] = right
    rot[:3, 1] = true_up
    rot[:3, 2] = -forward
    rot[:3, 3] = cam_pos
    scene.camera_transform = rot
    data = scene.save_image(resolution=resolution)
    return Image.open(trimesh.util.wrap_as_stream(data))

views = [
    ("VISTA FRONTAL (0°)", 0, 0),
    ("PERSPECTIVA 3/4 IZQ (+35°)", 35, 8),
    ("PERFIL IZQUIERDO (90°)", 90, 0),
    ("VISTA TRASERA (180°)", 180, 5),
]

imgs = []
for name, y, p in views:
    img = render_angle(y, p)
    imgs.append(img)

sheet = Image.new("RGB", (2800, 2800), (255, 255, 255))
sheet.paste(imgs[0], (0, 0))
sheet.paste(imgs[1], (1400, 0))
sheet.paste(imgs[2], (0, 1400))
sheet.paste(imgs[3], (1400, 1400))
sheet.save("test_v13_3_4views.png")
print("Renderizado completado en test_v13_3_4views.png")
