import os
import numpy as np
import trimesh
import manifold3d as m3d
from PIL import Image, ImageDraw
from io import BytesIO
import time
import trimesh.viewer.windowed

# Monkey patch trimesh resize
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

print("=== PROTOTIPO V15: AJUSTES EXACTOS BIGOTES, OJERAS, TRASERA Y ALAS ===")

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
casco_dir = os.path.join(BASE_DIR, r"Diseños 3D\Casco")
greeb_dir = os.path.join(casco_dir, "Greeblies")

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

# -------------------------------------------------------------
# 1. CRESTA DORSAL
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# 2. CEÑO Y FRENTE CON CUÑA EN V
# -------------------------------------------------------------
# Ceño inferior sobre el visor (Z=1808.6 a 1824.8 mm, la repisa del ceño de la pieza Front)
brow_lower = m3d.Manifold.cube([270.0, 160.0, 16.2], center=False).translate([-135.0, -100.0, 1808.6])

# Frente de la cúpula (Z=1824.8 a 1860.0 mm): termina en Y=-42.0 mm para dejar espacio blanco antes del ala!
brow_upper = m3d.Manifold.cube([130.0, 58.0, 36.0], center=False).translate([-65.0, -100.0, 1824.8])

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
brow_cutter = (brow_lower + brow_upper) - wedge_mani

# -------------------------------------------------------------
# 3. ALAS DE THOR EN LA CÚPULA LATERAL (CON NOTCHES CHEVRON)
# -------------------------------------------------------------
# Contorno exterior en Y-Z
pts_wing_outer = [
    [-10.0, 1832.0], # Raíz frontal suave sobre la frente
    [-8.0,  1845.0],
    [ 0.0,  1868.0],
    [12.0,  1890.0],
    [26.0,  1904.0],
    [42.0,  1913.0],
    [58.0,  1915.0], # Cima curvada
    [76.0,  1907.0], # Punta pluma 1 (superior)
    [68.0,  1887.0], # Punta pluma 2
    [60.0,  1868.0], # Punta pluma 3
    [52.0,  1848.0], # Punta pluma 4 (inferior)
    [38.0,  1841.0], # Borde inferior de la pluma 4
    [20.0,  1836.0],
    [ 5.0,  1833.0],
    [-5.0,  1832.0],
    [-10.0, 1832.0]  # Cierre en la raíz
]
cs_wing_base = m3d.CrossSection([pts_wing_outer[::-1]])

def make_chevron_slot(spine_pts, notch_pt, thickness=6.0):
    # Spine path from notch backwards to tips
    pts = np.vstack([[notch_pt], spine_pts])
    diffs = np.diff(pts, axis=0)
    normals = np.column_stack([-diffs[:, 1], diffs[:, 0]])
    lens = np.hypot(normals[:, 0], normals[:, 1])[:, None]
    normals = normals / np.maximum(lens, 1e-6)
    v_norms = np.zeros_like(pts)
    v_norms[0] = normals[0]
    v_norms[-1] = normals[-1]
    for i in range(1, len(pts)-1):
        v_norms[i] = (normals[i-1] + normals[i]) / 2.0
        v_norms[i] /= np.maximum(np.hypot(v_norms[i, 0], v_norms[i, 1]), 1e-6)
    
    half_th = thickness / 2.0
    side_top = pts[1:] + v_norms[1:] * half_th
    side_bot = pts[1:] - v_norms[1:] * half_th
    
    # Notch point creates a sharp chevron pointing forward
    poly = np.vstack([
        [notch_pt],
        side_top,
        side_bot[::-1]
    ])
    cs = m3d.CrossSection([poly])
    if cs.area() == 0:
        cs = m3d.CrossSection([poly[::-1]])
    return cs

# 3 ranuras curvadas con notch chevron frontal
s1 = make_chevron_slot([[32.0, 1898.0], [48.0, 1896.0], [78.0, 1892.0]], notch_pt=[22.0, 1898.0], thickness=6.0)
s2 = make_chevron_slot([[24.0, 1880.0], [40.0, 1878.0], [70.0, 1873.0]], notch_pt=[14.0, 1880.0], thickness=6.0)
s3 = make_chevron_slot([[16.0, 1860.0], [32.0, 1858.0], [62.0, 1854.0]], notch_pt=[ 6.0, 1860.0], thickness=5.5)

cs_wing = cs_wing_base - (s1 + s2 + s3)
raw_w = cs_wing.extrude(85.0)
m_rw = raw_w.to_mesh()
vw = m_rw.vert_properties
fw = m_rw.tri_verts

# Ala Derecha (+X)
vw_r = np.column_stack([vw[:, 2] + 38.0, vw[:, 0], vw[:, 1]]).astype('float32')
tri_r = fw.copy()
mani_w_r = m3d.Manifold(m3d.Mesh(vert_properties=vw_r, tri_verts=tri_r))
if mani_w_r.volume() <= 0:
    tri_r[:, [0, 1]] = tri_r[:, [1, 0]]
    mani_w_r = m3d.Manifold(m3d.Mesh(vert_properties=vw_r, tri_verts=tri_r))

# Ala Izquierda (-X)
vw_l = np.column_stack([-(vw[:, 2] + 38.0), vw[:, 0], vw[:, 1]]).astype('float32')
tri_l = tri_r.copy()
tri_l[:, [0, 1]] = tri_l[:, [1, 0]]
mani_w_l = m3d.Manifold(m3d.Mesh(vert_properties=vw_l, tri_verts=tri_l))

wings_cutter = mani_w_r + mani_w_l

# -------------------------------------------------------------
# 4. PARTE TRASERA: BANDA EN LA DEPRESIÓN DEL INICIO DE LA CÚPULA
# -------------------------------------------------------------
# La depresión del inicio de la cúpula empieza exactamente en Z = 1824.8 mm y sube a Z = 1858.0 mm!
# En X va de -60.0 a +60.0 mm (deteniéndose limpiamente detrás de las orejeras)
band_box = m3d.Manifold.cube([120.0, 60.0, 33.2], center=False).translate([-60.0, 168.0, 1824.8])

# Disco blanco cilíndrico de rango (R = 28.5 mm, centrado en Z = 1820.5 mm)
disc_cyl = m3d.Manifold.cylinder(80.0, 28.5, 28.5, 128).rotate([90, 0, 0]).translate([0, 235.0, 1820.5])

# Dos franjas verticales continuas sobre las costillas estructurales
rib_pts = np.array([
    [13.0, 1616.0],
    [22.0, 1616.0],
    [26.0, 1750.0],
    [29.0, 1835.0],
    [19.0, 1835.0],
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

# -------------------------------------------------------------
# 5. MEJILLAS, DEPRESIÓN LATERAL Y MORRO
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# 6. OJERAS BLANCAS DESPLAZADAS HACIA LOS LADOS
# -------------------------------------------------------------
# El blanco se desplaza hacia los lados: X de 28.0 a 68.0 mm!
suboc_box_r = m3d.Manifold.cube([40.0, 30.0, 7.5], center=False).translate([28.0, -65.0, 1789.0])
suboc_box_l = m3d.Manifold.cube([40.0, 30.0, 7.5], center=False).translate([-68.0, -65.0, 1789.0])
suboc_cutter = suboc_box_r + suboc_box_l

# -------------------------------------------------------------
# 7. BIGOTES BLANCOS (3 BANDAS CURVAS CONCÉNTRICAS MÁS LARGAS)
# -------------------------------------------------------------
# Como al principio: cortes planos rotados en X perpendicularmente al tubo cilíndrico,
# extendidos en X desde X = 18.0 mm (snout) hasta X = 86.0 mm (mejilla)
whiskers_r = m3d.Manifold()
whiskers_l = m3d.Manifold()

# 3 franjas con grosor 5.5 mm rotadas 26.5° en X (exactamente perpendiculares al tubo)
# y con recorte en X para longitud graduada (la superior llega más al centro)
for y_c, x_in, x_out in [
    (-55.0, 18.0, 86.0),  # Bigote 1 (superior): llega más al centro
    (-41.0, 22.0, 84.0),  # Bigote 2 (medio)
    (-27.0, 26.0, 82.0)   # Bigote 3 (inferior)
]:
    len_x = x_out - x_in
    mid_x = (x_out + x_in) / 2.0
    
    # Caja rotada perpendicular al tubo:
    slab_r = m3d.Manifold.cube([len_x, 5.5, 90.0], center=True).rotate([26.5, 0, 0]).translate([mid_x, y_c, 1641.0])
    slab_l = m3d.Manifold.cube([len_x, 5.5, 90.0], center=True).rotate([26.5, 0, 0]).translate([-mid_x, y_c, 1641.0])
    
    whiskers_r = whiskers_r + slab_r
    whiskers_l = whiskers_l + slab_l

# Límite vertical del tubo
tube_limit = m3d.Manifold.cube([240.0, 120.0, 56.0], center=False).translate([-120.0, -98.0, 1618.0])
whiskers_cutter = (whiskers_r + whiskers_l) ^ tube_limit

face_cutter = (snout_center + cheeks_raw) - suboc_cutter - whiskers_cutter

# Visor Keep-Out
visor_clear_h = m3d.Manifold.cube([180.0, 60.0, 13.0], center=False).translate([-90.0, -80.0, 1795.5])
visor_clear_v = m3d.Manifold.cube([  17.5, 50.0, 53.5], center=False).translate([ -8.75, -80.0, 1742.0])
visor_clearance = visor_clear_h + visor_clear_v

total_red_cutter = (fin_cutter + brow_cutter + wings_cutter + rear_markings + face_cutter) - visor_clearance

# Corte Manifold
print("Ejecutando corte booleano manifold...")
red_markings_mani = helmet_mani ^ total_red_cutter
white_base_mani = helmet_mani - total_red_cutter

vol_red = red_markings_mani.volume()
vol_white = white_base_mani.volume()
vol_sum = vol_red + vol_white
dev = abs(vol_sum - base_volume)
print(f"Volumen Rojo: {vol_red:.1f} mm³ | Blanco: {vol_white:.1f} mm³ | Desviación: {dev:.5f} mm³")

m_red = red_markings_mani.to_mesh()
mesh_red = trimesh.Trimesh(vertices=m_red.vert_properties[:, :3], faces=m_red.tri_verts, process=False)
mesh_red.visual.vertex_colors = [185, 28, 28, 255]

m_white = white_base_mani.to_mesh()
mesh_white = trimesh.Trimesh(vertices=m_white.vert_properties[:, :3], faces=m_white.tri_verts, process=False)
mesh_white.visual.vertex_colors = [244, 244, 246, 255]

mouth_mesh.visual.vertex_colors = [20, 20, 22, 255]
ears_mesh.visual.vertex_colors = [240, 240, 244, 255]
aerators_mesh.visual.vertex_colors = [70, 70, 75, 255]

scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, ears_mesh, aerators_mesh])
center = scene.centroid
scene.camera.z_near = 10.0
scene.camera.z_far = 2500.0

test_views = [
    ("test_v15_front.png", 0.0, 0.0, 820.0, None),
    ("test_v15_iso.png", 35.0, 8.0, 820.0, None),
    ("test_v15_profile.png", 90.0, 0.0, 820.0, None),
    ("test_v15_rear.png", 180.0, 0.0, 820.0, None),
    ("test_v15_macro_whiskers.png", 0, 0, 0, ([35.0, -60.0, 1650.0], [35.0 + 160.0, -60.0 - 240.0, 1650.0 + 30.0])),
    ("test_v15_macro_rear_depression.png", 0, 0, 0, ([0.0, 205.0, 1835.0], [0.0, 205.0 + 340.0, 1835.0 + 10.0])),
    ("test_v15_macro_wing.png", 0, 0, 0, ([75.0, 60.0, 1870.0], [75.0 + 320.0, 60.0, 1870.0 + 15.0]))
]

for filename, yaw_deg, pitch_deg, dist, macro_spec in test_views:
    if macro_spec is None:
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        cam_x = dist * np.sin(yaw) * np.cos(pitch)
        cam_y = -dist * np.cos(yaw) * np.cos(pitch)
        cam_z = dist * np.sin(pitch)
        cam_pos = np.array([cam_x, cam_y, cam_z]) + center
        look_target = center
    else:
        target_pt, cam_pos_pt = macro_spec
        look_target = np.array(target_pt, dtype=np.float32)
        cam_pos = np.array(cam_pos_pt, dtype=np.float32)
    
    view_vec = look_target - cam_pos
    forward = view_vec / np.linalg.norm(view_vec)
    up = np.array([0, 0, 1.0])
    if abs(forward[2]) > 0.99:
        up = np.array([0, -1.0, 0])
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
            time.sleep(0.08)
            raw_data = scene.save_image(resolution=(1600, 1600))
            break
        except Exception:
            time.sleep(0.2)
            
    img = Image.open(BytesIO(raw_data)).convert("RGBA")
    img_smooth = img.resize((800, 800), Image.Resampling.LANCZOS)
    img_smooth.save(filename)
    print(f"Renderizado: {filename}")

print("V15 Test completado con éxito!")
