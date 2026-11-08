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

print("=== PROTOTIPO V17: PERFECCIÓN CANÓNICA ALAS, BIGOTES, OJERAS Y TRASERA ===")

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
brow_lower = m3d.Manifold.cube([270.0, 160.0, 16.2], center=False).translate([-135.0, -100.0, 1808.6])
brow_upper = m3d.Manifold.cube([124.0, 56.0, 36.0], center=False).translate([-62.0, -100.0, 1824.8])

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
# 3. ALAS DE THOR ORGÁNICAS ESTILIZADAS (4 PLUMAS DINÁMICAS)
# -------------------------------------------------------------
# Mástil curvado suave en la frente:
spine_pts = [
    [-14.0, 1832.0],
    [-10.0, 1848.0],
    [  0.0, 1874.0],
    [ 14.0, 1898.0],
    [ 30.0, 1912.0],
    [ 50.0, 1920.0],
    [ 66.0, 1920.0], # Cima
    [ 52.0, 1912.0],
    [ 38.0, 1896.0],
    [ 24.0, 1876.0],
    [ 10.0, 1856.0],
    [  0.0, 1840.0],
    [ -8.0, 1832.0],
    [-14.0, 1832.0]
]
cs_spine = m3d.CrossSection([spine_pts[::-1]])

# Pluma 1 (Superior, larga, curvada y estilizada hacia atrás):
f1_pts = [
    [30.0, 1912.0],
    [50.0, 1920.0],
    [66.0, 1920.0],
    [86.0, 1912.0],
    [98.0, 1902.0], # Punta 1 afilada
    [88.0, 1896.0],
    [62.0, 1906.0],
    [36.0, 1902.0]
]
cs_f1 = m3d.CrossSection([f1_pts[::-1]])

# Pluma 2:
f2_pts = [
    [22.0, 1894.0],
    [48.0, 1902.0],
    [74.0, 1894.0],
    [86.0, 1884.0], # Punta 2
    [76.0, 1878.0],
    [52.0, 1886.0],
    [24.0, 1882.0]
]
cs_f2 = m3d.CrossSection([f2_pts[::-1]])

# Pluma 3:
f3_pts = [
    [12.0, 1874.0],
    [38.0, 1882.0],
    [62.0, 1874.0],
    [74.0, 1864.0], # Punta 3
    [64.0, 1858.0],
    [40.0, 1866.0],
    [14.0, 1862.0]
]
cs_f3 = m3d.CrossSection([f3_pts[::-1]])

# Pluma 4 (Inferior):
f4_pts = [
    [ 2.0, 1854.0],
    [26.0, 1862.0],
    [50.0, 1854.0],
    [62.0, 1844.0], # Punta 4
    [52.0, 1838.0],
    [28.0, 1844.0],
    [ 0.0, 1840.0]
]
cs_f4 = m3d.CrossSection([f4_pts[::-1]])

cs_wing = cs_spine + cs_f1 + cs_f2 + cs_f3 + cs_f4

# Extrusión lateral de 85 mm desde X = 38.0 a 123.0 mm
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
# La depresión del inicio de la cúpula empieza exactamente en Z = 1824.8 mm y sube a Z = 1858.0 mm
band_box = m3d.Manifold.cube([136.0, 60.0, 33.2], center=False).translate([-68.0, 168.0, 1824.8])
disc_cyl = m3d.Manifold.cylinder(80.0, 28.5, 28.5, 128).rotate([90, 0, 0]).translate([0, 235.0, 1820.5])

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
    [ -19.5, 1600.0],
    [ -19.5, 1658.0],
    [  52.0, 1658.0],
    [  52.0, 1808.6],
    [-25.0,  1808.6],
    [-25.0,  1798.0],
    [-110.0, 1798.0]
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
# 6. OJERAS BLANCAS DESPLAZADAS A LOS LADOS
# -------------------------------------------------------------
# Blanco desplazado a |X| en [28.0, 70.0] mm
suboc_box_r = m3d.Manifold.cube([42.0, 30.0, 7.5], center=False).translate([28.0, -65.0, 1789.0])
suboc_box_l = m3d.Manifold.cube([42.0, 30.0, 7.5], center=False).translate([-70.0, -65.0, 1789.0])
suboc_cutter = suboc_box_r + suboc_box_l

# -------------------------------------------------------------
# 7. BIGOTES BLANCOS: 3 FRANJAS ORGÁNICAS PARALELAS Y EXTENDIDAS
# -------------------------------------------------------------
whiskers_r = m3d.Manifold()
whiskers_l = m3d.Manifold()

# Ángulo anatómico suave (rot_x=16.0, rot_z=-8.0) para que sigan la silueta de la Foto 1
whisker_specs = [
    # (zc, yc, xc, length_x, thick, rx, ry, rz)
    (1680.0, -78.0, 48.0, 62.0, 6.5, 16.0, -4.0, -8.0),
    (1666.0, -75.0, 50.0, 58.0, 6.5, 16.0, -4.0, -8.0),
    (1652.0, -72.0, 52.0, 54.0, 6.0, 16.0, -4.0, -8.0)
]

for zc, yc, xc, length_x, thick, rx, ry, rz in whisker_specs:
    wr = m3d.Manifold.cube([length_x, 60.0, thick], center=True).rotate([rx, ry, rz]).translate([xc, yc, zc])
    wl = m3d.Manifold.cube([length_x, 60.0, thick], center=True).rotate([rx, -ry, -rz]).translate([-xc, yc, zc])
    whiskers_r = whiskers_r + wr
    whiskers_l = whiskers_l + wl

whiskers_cutter = whiskers_r + whiskers_l

face_cutter = (snout_center + cheeks_raw) - suboc_cutter - whiskers_cutter

# Visor Keep-Out
visor_clear_h = m3d.Manifold.cube([180.0, 60.0, 13.0], center=False).translate([-90.0, -80.0, 1795.5])
visor_clear_v = m3d.Manifold.cube([  17.5, 50.0, 53.5], center=False).translate([ -8.75, -80.0, 1742.0])
visor_clearance = visor_clear_h + visor_clear_v

total_red_cutter = (fin_cutter + brow_cutter + wings_cutter + rear_markings + face_cutter) - visor_clearance

print("Ejecutando corte booleano manifold V17...")
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
    ("test_v17_front.png", 0.0, 0.0, 820.0, None),
    ("test_v17_iso.png", 35.0, 8.0, 820.0, None),
    ("test_v17_profile.png", 90.0, 0.0, 820.0, None),
    ("test_v17_rear.png", 180.0, 0.0, 820.0, None),
    ("test_v17_macro_whiskers.png", 0, 0, 0, ([35.0, -75.0, 1665.0], [35.0 + 160.0, -75.0 - 240.0, 1665.0 + 30.0])),
    ("test_v17_macro_wing.png", 0, 0, 0, ([75.0, 60.0, 1870.0], [75.0 + 320.0, 60.0, 1870.0 + 15.0])),
    ("test_v17_macro_rear.png", 0, 0, 0, ([0.0, 205.0, 1835.0], [0.0, 205.0 + 340.0, 1835.0 + 10.0]))
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

print("V17 completado con éxito!")
