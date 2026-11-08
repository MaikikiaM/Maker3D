import os
import sys
import numpy as np
import trimesh
import manifold3d as m3d
from PIL import Image

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
casco_dir = os.path.join(BASE_DIR, r"Diseños 3D\Casco")
helmet_mesh = trimesh.load(os.path.join(casco_dir, "Phase 2 Main Body.stl"))
v_base = helmet_mesh.vertices.astype('float32')
f_base = helmet_mesh.faces.astype('uint32')
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))

greeb_dir = os.path.join(casco_dir, "Greeblies")
mouth_mesh = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
ear_l = trimesh.load(os.path.join(greeb_dir, "Ear Piece Left.stl"))
ear_r = trimesh.load(os.path.join(greeb_dir, "Ear Piece Right.stl"))
ears_mesh = trimesh.util.concatenate([ear_l, ear_r])
aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))
aerators_mesh = trimesh.util.concatenate([aer_l, aer_r])

mouth_mesh.visual.vertex_colors = [25, 25, 25, 255]
ears_mesh.visual.vertex_colors = [180, 180, 180, 255]
aerators_mesh.visual.vertex_colors = [120, 120, 120, 255]

def make_ccw(pts):
    pts = np.array(pts, dtype=np.float32)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area < 0:
        pts = pts[::-1]
    return m3d.CrossSection([pts])

def make_rounded_slot(y_in, z_in, y_out, z_out, th=4.0, r_tip=1.8):
    p_in = np.array([y_in, z_in], dtype=np.float32)
    p_out = np.array([y_out, z_out], dtype=np.float32)
    v_dir = p_out - p_in
    length = np.linalg.norm(v_dir)
    u_dir = v_dir / length
    n_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float32)
    pts = []
    c_tip = p_in + u_dir * r_tip
    for a in np.linspace(np.pi/2, 3*np.pi/2, 12):
        pt = c_tip + r_tip * (np.cos(a) * u_dir + np.sin(a) * n_dir)
        pts.append(pt.tolist())
    pts.append((p_out - n_dir * (th/2.0 + 1.0)).tolist())
    pts.append((p_out + n_dir * (th/2.0 + 1.0)).tolist())
    return make_ccw(pts)

# 1. CRESTA DORSAL: Franja central roja de 20 mm, flancos blancos
# El rojo termina antes de la banda trasera para dejar el espacio blanco
fin_poly_yz = np.array([
    [-18.0, 1918.0],
    [-2.0,  1885.0],
    [175.0, 1880.0],  # Termina antes del escalón trasero
    [175.0, 1960.0],
    [-18.0, 1960.0]
], dtype=np.float32)
cs_fin = make_ccw(fin_poly_yz)
raw_fin = cs_fin.extrude(20.0)
m_rf = raw_fin.to_mesh()
v_rf = m_rf.vert_properties
v_fin_3d = np.column_stack([v_rf[:, 2] - 10.0, v_rf[:, 0], v_rf[:, 1]]).astype('float32')
f_fin_3d = m_rf.tri_verts.copy()
fin_cutter = m3d.Manifold(m3d.Mesh(vert_properties=v_fin_3d, tri_verts=f_fin_3d))
if fin_cutter.volume() <= 0:
    f_fin_3d[:, [0, 1]] = f_fin_3d[:, [1, 0]]
    fin_cutter = m3d.Manifold(m3d.Mesh(vert_properties=v_fin_3d, tri_verts=f_fin_3d))

# 2. FRENTE ROJO ALTO Y CONCORDANCIA CON ALAS:
# El rojo del ceño cubre desde el visor Z=1808.6 hasta Z=1855.0 mm en la cúpula!
# Se detiene antes de la oreja en Y = -18.0 mm (NO pasa por encima de la oreja).
# Muesca central blanca trapezoidal ensanchándose hacia arriba (26 mm abajo, 36 mm arriba).
forehead_box = m3d.Manifold.cube([220.0, 65.0, 50.0], center=False).translate([-110.0, -80.0, 1808.6])
# Cortar en Y para que no pase de Y = -18 mm (la zona sobre la oreja debe ser 100% blanca):
forehead_limiter = m3d.Manifold.cube([260.0, 100.0, 60.0], center=False).translate([-130.0, -18.0, 1805.0])
forehead_red = forehead_box - forehead_limiter

notch_poly = [
    [-13.0, 1807.0],  # 26 mm ancho en visor
    [ 13.0, 1807.0],
    [ 18.0, 1860.0],  # 36 mm ancho arriba bajo cresta
    [-18.0, 1860.0]
]
cs_notch = make_ccw(notch_poly)
raw_notch = cs_notch.extrude(120.0)
mat_n = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -105.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
center_notch = raw_notch.transform(mat_n)
forehead_cutter = forehead_red - center_notch

# 3. ALAS DE THOR: Emergen directamente de la frente roja alta y corren por la cúpula
# La base empalma con la frente roja en Y = -22 a -18 mm, Z = 1808.6 a 1855 mm
wing_pts = [
    [-22.0, 1808.6],
    [-22.0, 1855.0],  # Empalme continuo con la frente alta
    [-15.0, 1868.0],
    [ -7.0, 1882.0],
    [  5.0, 1894.0],
    [ 25.0, 1897.0],
    [ 55.0, 1893.0],
    [ 85.0, 1885.0],
    [113.0, 1874.0],  # Punta Pluma 1 (+15% hacia atrás)
    [ 95.0, 1858.0],  # Punta Pluma 2
    [ 79.0, 1844.0],  # Punta Pluma 3
    [ 63.0, 1831.0],  # Punta Pluma 4 (pinza)
    [ -2.0, 1835.0],  # Destalonado: deja blanco debajo de las plumas sobre la oreja
    [-15.0, 1830.0],
    [-18.0, 1808.6]
]
cs_w = make_ccw(wing_pts)
slot1 = make_rounded_slot( 12.0, 1881.0, 118.0, 1866.0, th=4.0, r_tip=1.8)
slot2 = make_rounded_slot(  2.0, 1865.0, 100.0, 1851.0, th=4.0, r_tip=1.8)
slot3 = make_rounded_slot( -6.0, 1849.0,  84.0, 1837.0, th=4.0, r_tip=1.8)
cs_w = ((cs_w - slot1) - slot2) - slot3
raw_w = cs_w.extrude(100.0)
mat_wr = np.array([[0.0, 0.0, 1.0, 25.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_wl = np.array([[0.0, 0.0, -1.0, -25.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
wings_cutter = raw_w.transform(mat_wr) + raw_w.transform(mat_wl)

# 4. ZONA TRASERA: BANDA ELEVADA EN LA DEPRESIÓN DE LA CÚPULA (Z = 1840 a 1874 mm)
# Por encima del disco blanco (que queda 100% blanco abajo),
# y 2 costillas verticales que bajan desde la banda hasta la base del casco.
# La banda trasera termina lateralmente en X in [-78, 78], dejando blanco sobre el hueco de la oreja.
rear_band = m3d.Manifold.cube([156.0, 95.0, 34.0], center=False).translate([-78.0, 135.0, 1840.0])
disc_cutter = m3d.Manifold.cylinder(80.0, 22.0, 22.0, 128).rotate([90, 0, 0]).translate([0, 235.0, 1815.0])
rib_r = m3d.Manifold.cube([14.0, 55.0, 230.0], center=False).translate([ 13.0, 170.0, 1616.0])
rib_l = m3d.Manifold.cube([14.0, 55.0, 230.0], center=False).translate([-27.0, 170.0, 1616.0])
rear_markings = (rear_band + rib_r + rib_l) - disc_cutter

# 5. MEJILLAS, DEPRESIÓN Y TUBO MANDIBULAR:
cheek_pts = [
    [-110.0, 1600.0],
    [ -50.0, 1600.0],  # Paralelo a los 3 bigotes
    [   3.0, 1710.0],  # Conecta al 35% de la depresión
    [  72.0, 1710.0],  # Suelo 100% rojo
    [  72.0, 1780.0],  # Pared posterior 100% roja
    [  55.0, 1812.0],
    [-110.0, 1812.0]
]
cs_ch = make_ccw(cheek_pts)
raw_ch = cs_ch.extrude(120.0)
mat_ch_r = np.array([[0.0, 0.0, 1.0, 8.85], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_ch_l = np.array([[0.0, 0.0, -1.0, -8.85], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
cheeks_raw = raw_ch.transform(mat_ch_r) + raw_ch.transform(mat_ch_l)

snout_box = m3d.Manifold.cube([92.0, 55.0, 122.0], center=False).translate([-46.0, -97.0, 1620.0])
nose_teeth_bridge = m3d.Manifold.cube([32.0, 50.0, 125.0], center=False).translate([-16.0, -98.0, 1620.0])
ear_clear_r = m3d.Manifold.cylinder(80.0, 33.0, 33.0, 96).rotate([0, 90, 0]).translate([85.0, 77.7, 1789.1])
ear_clear_l = m3d.Manifold.cylinder(80.0, 33.0, 33.0, 96).rotate([0, -90, 0]).translate([-85.0, 77.7, 1789.1])
ear_exclusion = ear_clear_r + ear_clear_l

# 6. 3 BIGOTES CURVOS:
claws_raw = m3d.Manifold()
for y_c, x_in, length in [(-54.0, 26.0, 68.0), (-41.0, 22.0, 72.0), (-28.0, 20.0, 76.0)]:
    th = 5.2
    r_c = th / 2.0
    b_pts = []
    for a in np.linspace(np.pi/2, 3*np.pi/2, 10):
        b_pts.append([r_c + r_c * np.cos(a), r_c * np.sin(a)])
    b_pts.append([length, -r_c])
    b_pts.append([length,  r_c])
    cs_b = make_ccw(b_pts)
    raw_b = cs_b.extrude(100.0).translate([0, 0, -50.0])
    mat_br = np.array([
        [1.0, 0.0, 0.0, x_in],
        [0.0, np.cos(np.radians(26.5)), -np.sin(np.radians(26.5)), y_c],
        [0.0, np.sin(np.radians(26.5)),  np.cos(np.radians(26.5)), 1641.0]
    ], dtype=np.float32)
    mat_bl = np.array([
        [-1.0, 0.0, 0.0, -x_in],
        [0.0, np.cos(np.radians(26.5)), -np.sin(np.radians(26.5)), y_c],
        [0.0, np.sin(np.radians(26.5)),  np.cos(np.radians(26.5)), 1641.0]
    ], dtype=np.float32)
    claws_raw = claws_raw + raw_b.transform(mat_br) + raw_b.transform(mat_bl)

# 7. FRANJA SUB-OCULAR (4.5 mm ciñéndose al visor)
p1_r = np.array([32.0, -50.5, 1795.0], dtype=np.float32)
p2_r = np.array([64.0, -36.5, 1783.0], dtype=np.float32)
p0_r = (p1_r + p2_r) / 2.0
u_r = (p2_r - p1_r) / np.linalg.norm(p2_r - p1_r)
u_len = np.linalg.norm(p2_r - p1_r)
w_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
v_r = np.cross(w_ref, u_r); v_r /= np.linalg.norm(v_r)
if v_r[2] < 0: v_r = -v_r
w_r = np.cross(u_r, v_r); w_r /= np.linalg.norm(w_r)

r_round = 1.5
oj_pts = [
    [-u_len/2.0,  25.0],
    [-u_len/2.0,  -4.5 + r_round],
    [-u_len/2.0 + r_round, -4.5],
    [ u_len/2.0 - r_round, -4.5],
    [ u_len/2.0,  -4.5 + r_round],
    [ u_len/2.0,  25.0]
]
cs_oj = make_ccw(oj_pts)
raw_oj = cs_oj.extrude(60.0).translate([0, 0, -30.0])
mat_oj_r = np.eye(4, dtype=np.float32)
mat_oj_r[:3, 0] = u_r; mat_oj_r[:3, 1] = v_r; mat_oj_r[:3, 2] = w_r; mat_oj_r[:3, 3] = p0_r

p1_l = np.array([-32.0, -50.5, 1795.0], dtype=np.float32)
p2_l = np.array([-64.0, -36.5, 1783.0], dtype=np.float32)
p0_l = (p1_l + p2_l) / 2.0
u_l = (p2_l - p1_l) / np.linalg.norm(p2_l - p1_l)
v_l = np.cross(w_ref, u_l); v_l /= np.linalg.norm(v_l)
if v_l[2] < 0: v_l = -v_l
w_l = np.cross(u_l, v_l); w_l /= np.linalg.norm(w_l)
mat_oj_l = np.eye(4, dtype=np.float32)
mat_oj_l[:3, 0] = u_l; mat_oj_l[:3, 1] = v_l; mat_oj_l[:3, 2] = w_l; mat_oj_l[:3, 3] = p0_l

suboc_cutter = raw_oj.transform(mat_oj_r[:3]) + raw_oj.transform(mat_oj_l[:3])

face_solid = cheeks_raw + snout_box + nose_teeth_bridge
face_cutter = (face_solid - claws_raw) - suboc_cutter
total_red_cutter = (fin_cutter + forehead_cutter + wings_cutter + rear_markings + face_cutter) - ear_exclusion

red_markings_mani = helmet_mani ^ total_red_cutter
white_base_mani = helmet_mani - total_red_cutter

print(f"Patron rojo vol: {red_markings_mani.volume():.1f}")
print(f"Base blanca vol: {white_base_mani.volume():.1f}")
print(f"Dev: {abs(red_markings_mani.volume() + white_base_mani.volume() - helmet_mani.volume()):.5f}")

mesh_red = trimesh.Trimesh(vertices=red_markings_mani.to_mesh().vert_properties[:, :3],
                           faces=red_markings_mani.to_mesh().tri_verts, process=False)
mesh_red.visual.vertex_colors = [185, 28, 28, 255]

mesh_white = trimesh.Trimesh(vertices=white_base_mani.to_mesh().vert_properties[:, :3],
                             faces=white_base_mani.to_mesh().tri_verts, process=False)
mesh_white.visual.vertex_colors = [245, 245, 245, 255]

scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, ears_mesh, aerators_mesh])
center = scene.centroid
scene.camera.z_near = 10.0
scene.camera.z_far = 2500.0

views = [
    ("front", 0.0, 0.0),
    ("iso", 35.0, 8.0),
    ("profile", 90.0, 0.0),
    ("rear", 180.0, 0.0)
]
dist = 540.0
panel_imgs = []
for name, yaw_deg, pitch_deg in views:
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    cam_pos = np.array([dist * np.sin(yaw) * np.cos(pitch), -dist * np.cos(yaw) * np.cos(pitch), dist * np.sin(pitch)]) + center
    view_vec = center - cam_pos
    forward = view_vec / np.linalg.norm(view_vec)
    up = np.array([0, 0, 1.0])
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    actual_up = np.cross(right, forward); actual_up /= np.linalg.norm(actual_up)
    c2w = np.eye(4)
    c2w[:3, 0] = right; c2w[:3, 1] = actual_up; c2w[:3, 2] = -forward; c2w[:3, 3] = cam_pos
    scene.camera_transform = c2w
    data = scene.save_image(resolution=(1024, 1024))
    p_path = f"test_v30_{name}.png"
    with open(p_path, 'wb') as f:
        f.write(data)
    panel_imgs.append(Image.open(p_path))

grid = Image.new('RGB', (2048, 2048), (255, 255, 255))
grid.paste(panel_imgs[0], (0, 0))
grid.paste(panel_imgs[1], (1024, 0))
grid.paste(panel_imgs[2], (0, 1024))
grid.paste(panel_imgs[3], (1024, 1024))
grid.save("test_v30_4views.png")
print("Saved test_v30_4views.png successfully!")
