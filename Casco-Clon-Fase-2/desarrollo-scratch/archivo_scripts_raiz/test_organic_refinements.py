import os
import numpy as np
import trimesh
import manifold3d as m3d
from PIL import Image
from io import BytesIO
import time

casco_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco"
greeb_dir = os.path.join(casco_dir, "Greeblies")
out_dir = os.path.join(casco_dir, r"Antigravity\04_Guias_Visuales_y_Planos")

print("[1] Cargando mallas base...")
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

print("[2] Construyendo cortadores hiper-orgánicos y afinados...")

# A. Cresta Dorsal (Lateral rojo, parte superior y centro blanco):
fin_poly_yz = np.array([
    [-12.0, 1918.0],
    [-2.0,  1885.0],
    [205.0, 1870.0],
    [205.0, 1960.0],
    [-12.0, 1960.0]
], dtype=np.float32)
cs_fin = m3d.CrossSection([fin_poly_yz])
raw_fin = cs_fin.extrude(42.0)
m_rf = raw_fin.to_mesh()
v_rf = m_rf.vert_properties
v_fin_3d = np.column_stack([v_rf[:, 2] - 21.0, v_rf[:, 0], v_rf[:, 1]]).astype('float32')
f_fin_3d = m_rf.tri_verts.copy()
fin_full = m3d.Manifold(m3d.Mesh(vert_properties=v_fin_3d, tri_verts=f_fin_3d))
if fin_full.volume() <= 0:
    f_fin_3d[:, [0, 1]] = f_fin_3d[:, [1, 0]]
    fin_full = m3d.Manifold(m3d.Mesh(vert_properties=v_fin_3d, tri_verts=f_fin_3d))

crest_white_center = m3d.Manifold.cube([16.0, 240.0, 100.0], center=False).translate([-8.0, -20.0, 1870.0])
fin_cutter = fin_full - crest_white_center

# B. Ceño Frontal con Muesca Acampanada Orgánica:
brow_box = m3d.Manifold.cube([240.0, 85.0, 23.4], center=False).translate([-120.0, -100.0, 1808.6])
# Muesca central acampanada (28 mm en base, 36 mm arriba, con chaflán elegante):
notch_poly = [
    [-14.0, 1807.0],
    [ 14.0, 1807.0],
    [ 18.0, 1835.0],
    [-18.0, 1835.0]
]
cs_notch = make_ccw(notch_poly)
raw_notch = cs_notch.extrude(120.0)
mat_n = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -105.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
center_notch = raw_notch.transform(mat_n)
brow_cutter = brow_box - center_notch

# C. Alas de Thor con 4 Plumas Dinámicas, Ranuras Chevron en Cuña y Puntas Afiladas:
wing_pts = [
    [-18.0, 1808.6], # Base orgánica que empalma con el ceño
    [-16.5, 1832.0],
    [-14.5, 1855.0],
    [-10.5, 1878.0],
    [ -4.5, 1900.0],
    [  3.5, 1918.0],
    [ 15.0, 1926.0], # Corona suave
    [ 45.0, 1926.0],
    [ 72.0, 1922.0],
    [ 95.0, 1912.0], # Punta Pluma 1
    [ 80.0, 1891.0], # Punta Pluma 2
    [ 66.0, 1870.0], # Punta Pluma 3
    [ 54.0, 1849.0], # Punta Pluma 4 (pinza inferior)
    [  6.0, 1842.0], # Muesca blanca bajo Pluma 4
    [  6.0, 1808.6], # Empalme posterior al ceño
    [-18.0, 1808.6]
]
cs_w = make_ccw(wing_pts)

# Ranuras en cuña (más anchas atrás, estrechándose hacia el mástil con vértice suave):
slot1 = make_ccw([[  6.0, 1900.0], [ 93.0, 1903.0], [ 93.0, 1907.5], [  6.0, 1901.8]])
slot2 = make_ccw([[ -4.0, 1878.5], [ 79.0, 1882.0], [ 79.0, 1886.5], [ -4.0, 1880.3]])
slot3 = make_ccw([[ -9.0, 1857.0], [ 65.0, 1860.5], [ 65.0, 1865.0], [ -9.0, 1858.8]])

cs_w = ((cs_w - slot1) - slot2) - slot3

raw_w = cs_w.extrude(100.0)
mat_wr = np.array([[0.0, 0.0, 1.0, 25.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_wl = np.array([[0.0, 0.0, -1.0, -25.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
wings_cutter = raw_w.transform(mat_wr) + raw_w.transform(mat_wl)

# D. Zona Trasera (Banda en escalón/depresión Z=1808.6 a 1844.0, disco blanco y costillas):
rear_band = m3d.Manifold.cube([192.0, 95.0, 35.4], center=False).translate([-96.0, 135.0, 1808.6])
disc_cutter = m3d.Manifold.cylinder(80.0, 21.5, 21.5, 128).rotate([90, 0, 0]).translate([0, 235.0, 1815.6])
rib_r = m3d.Manifold.cube([14.0, 55.0, 202.0], center=False).translate([ 13.0, 170.0, 1616.0])
rib_l = m3d.Manifold.cube([14.0, 55.0, 202.0], center=False).translate([-27.0, 170.0, 1616.0])
rear_markings = (rear_band + rib_r + rib_l) - disc_cutter

# E. Morro, Dientes continuos, Ojeras PARALELAS AL VISOR y 3 Bigotes con puntas redondeadas:
cheek_pts = [
    [-110.0, 1600.0],
    [ -22.0, 1600.0],  # Detiene el rojo antes de las rejillas
    [ -22.0, 1680.0],
    [  20.0, 1680.0],  # Depresión frente a la orejera
    [  20.0, 1775.0],
    [ -15.0, 1808.6],  # Conecta con el ceño frontal
    [-25.0,  1808.6],
    [-25.0,  1798.0],
    [-110.0, 1798.0]
]
cs_ch = make_ccw(cheek_pts)
raw_ch = cs_ch.extrude(120.0)
mat_ch_r = np.array([[0.0, 0.0, 1.0, 8.85], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_ch_l = np.array([[0.0, 0.0, -1.0, -8.85], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
cheeks_raw = raw_ch.transform(mat_ch_r) + raw_ch.transform(mat_ch_l)

snout_box = m3d.Manifold.cube([92.0, 55.0, 122.0], center=False).translate([-46.0, -97.0, 1620.0])
nose_teeth_bridge = m3d.Manifold.cube([32.0, 50.0, 125.0], center=False).translate([-16.0, -98.0, 1620.0])
tubes_box = m3d.Manifold.cube([230.0, 85.0, 50.0], center=False).translate([-115.0, -65.0, 1618.0])

# Bigotes Curvos con PUNTAS REDONDEADAS/AFINADAS:
claws_raw = m3d.Manifold()
for y_c, x_in, length in [(-54.0, 26.0, 68.0), (-41.0, 22.0, 72.0), (-28.0, 20.0, 76.0)]:
    # Creamos cada bigote como una cápsula 2D redondeada extruida
    th = 5.2
    r_c = th / 2.0
    b_pts = []
    # Punta interna redondeada (estilo pincelada/garra)
    for a in np.linspace(np.pi/2, 3*np.pi/2, 10):
        b_pts.append([r_c + r_c * np.cos(a), r_c * np.sin(a)])
    # Extremo externo
    b_pts.append([length, -r_c])
    b_pts.append([length,  r_c])
    cs_b = make_ccw(b_pts)
    raw_b = cs_b.extrude(100.0).translate([0, 0, -50.0])
    
    # Rotar e inclinar para seguir la curvatura del tubo (26.5°)
    # Local X es a lo largo de la franja (X global)
    # Local Y es el espesor (rotado 26.5°)
    # Local Z es la profundidad en la mejilla
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

# OJERAS ORGÁNICAS PARALELAS AL VISOR CON PUNTAS REDONDEADAS:
# Longitud 40 mm, altura 8.0 mm (sleek), extremos redondeados
u_len = 40.0
v_h = 8.0
r_oj = v_h / 2.0
oj_pts = []
for a in np.linspace(np.pi/2, 3*np.pi/2, 12):
    oj_pts.append([-u_len/2.0 + r_oj + r_oj * np.cos(a), r_oj * np.sin(a)])
for a in np.linspace(-np.pi/2, np.pi/2, 12):
    oj_pts.append([ u_len/2.0 - r_oj + r_oj * np.cos(a), r_oj * np.sin(a)])

cs_oj = make_ccw(oj_pts)
raw_oj = cs_oj.extrude(60.0).translate([0, 0, -30.0])

# Transformación de la ojera derecha:
# Vector a lo largo del visor: P1=(32, -47, 1794) a P2=(72, -30, 1779)
p1_r = np.array([32.0, -47.0, 1794.0], dtype=np.float32)
p2_r = np.array([72.0, -30.0, 1779.0], dtype=np.float32)
p0_r = (p1_r + p2_r) / 2.0
u_r = (p2_r - p1_r) / np.linalg.norm(p2_r - p1_r)

w_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
v_r = np.cross(w_ref, u_r)
v_r /= np.linalg.norm(v_r)
if v_r[2] > 0:
    v_r = -v_r
w_r = np.cross(u_r, v_r)
w_r /= np.linalg.norm(w_r)

mat_oj_r = np.eye(4, dtype=np.float32)
mat_oj_r[:3, 0] = u_r
mat_oj_r[:3, 1] = v_r
mat_oj_r[:3, 2] = w_r
mat_oj_r[:3, 3] = p0_r

# Transformación de la ojera izquierda (simetría estricta):
p1_l = np.array([-32.0, -47.0, 1794.0], dtype=np.float32)
p2_l = np.array([-72.0, -30.0, 1779.0], dtype=np.float32)
p0_l = (p1_l + p2_l) / 2.0
u_l = (p2_l - p1_l) / np.linalg.norm(p2_l - p1_l)

v_l = np.cross(w_ref, u_l)
v_l /= np.linalg.norm(v_l)
if v_l[2] > 0:
    v_l = -v_l
w_l = np.cross(u_l, v_l)
w_l /= np.linalg.norm(w_l)

mat_oj_l = np.eye(4, dtype=np.float32)
mat_oj_l[:3, 0] = u_l
mat_oj_l[:3, 1] = v_l
mat_oj_l[:3, 2] = w_l
mat_oj_l[:3, 3] = p0_l

suboc_cutter = raw_oj.transform(mat_oj_r[:3]) + raw_oj.transform(mat_oj_l[:3])

face_solid = cheeks_raw + snout_box + nose_teeth_bridge + tubes_box
face_cutter = (face_solid - claws_raw) - suboc_cutter

visor_clear_h = m3d.Manifold.cube([180.0, 60.0, 13.0], center=False).translate([-90.0, -80.0, 1795.5])
visor_clear_v = m3d.Manifold.cube([  17.5, 50.0, 53.5], center=False).translate([ -8.75, -80.0, 1742.0])
visor_clearance = visor_clear_h + visor_clear_v

total_red_cutter = (fin_cutter + brow_cutter + wings_cutter + rear_markings + face_cutter) - visor_clearance

print("[3] Ejecutando corte booleano exacto manifold...")
red_markings_mani = helmet_mani ^ total_red_cutter
white_base_mani = helmet_mani - total_red_cutter

vol_red = red_markings_mani.volume()
vol_white = white_base_mani.volume()
vol_sum = vol_red + vol_white
dev = abs(vol_sum - base_volume)
print(f"Volumen Rojo: {vol_red:.1f} mm³ | Base Blanca: {vol_white:.1f} mm³ | Desviación: {dev:.4f} mm³")

m_red = red_markings_mani.to_mesh()
mesh_red = trimesh.Trimesh(vertices=m_red.vert_properties[:, :3], faces=m_red.tri_verts, process=False)
mesh_red.visual.vertex_colors = [185, 28, 28, 255]

m_white = white_base_mani.to_mesh()
mesh_white = trimesh.Trimesh(vertices=m_white.vert_properties[:, :3], faces=m_white.tri_verts, process=False)
mesh_white.visual.vertex_colors = [244, 244, 246, 255]

mouth_mesh.visual.vertex_colors = [20, 20, 22, 255]
ears_mesh.visual.vertex_colors = [240, 240, 244, 255]
aerators_mesh.visual.vertex_colors = [70, 70, 75, 255]

print("[4] Renderizando las 4 vistas canónicas para control visual...")
scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, ears_mesh, aerators_mesh])
center = scene.centroid
scene.camera.z_near = 10.0
scene.camera.z_far = 2500.0

test_views = [
    ("01_Frontal", 0.0, 0.0, 540.0),
    ("02_Iso_3_4", 35.0, 8.0, 540.0),
    ("03_Perfil_Izq", 90.0, 0.0, 540.0),
    ("04_Trasera", 180.0, 0.0, 540.0)
]

cards = []
for name, yaw_deg, pitch_deg, dist in test_views:
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    cam_x = dist * np.sin(yaw) * np.cos(pitch)
    cam_y = -dist * np.cos(yaw) * np.cos(pitch)
    cam_z = dist * np.sin(pitch)
    cam_pos = np.array([cam_x, cam_y, cam_z]) + center
    look_target = center

    view_vec = look_target - cam_pos
    forward = view_vec / np.linalg.norm(view_vec)
    up = np.array([0, 0, 1.0])
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

    time.sleep(0.08)
    raw_data = scene.save_image(resolution=(1600, 1600))
    img_large = Image.open(BytesIO(raw_data)).convert("RGBA")
    img_smooth = img_large.resize((800, 800), Image.Resampling.LANCZOS)
    cards.append(img_smooth)
    print(f" - Vista {name} renderizada.")

sheet_w = 800 * 2 + 60
sheet_h = 800 * 2 + 60
sheet = Image.new("RGBA", (sheet_w, sheet_h), (245, 246, 248, 255))
sheet.paste(cards[0], (20, 20))
sheet.paste(cards[1], (840, 20))
sheet.paste(cards[2], (20, 840))
sheet.paste(cards[3], (840, 840))

test_img_path = os.path.join(out_dir, "test_v21_organic_check.png")
sheet.save(test_img_path)
print(f"Imagen de prueba guardada en: {test_img_path}")
