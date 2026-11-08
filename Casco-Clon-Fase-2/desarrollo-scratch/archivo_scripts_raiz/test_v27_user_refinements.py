import os
import numpy as np
import trimesh
import manifold3d as m3d
from PIL import Image
from io import BytesIO
import time

casco_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco"
greeb_dir = os.path.join(casco_dir, "Greeblies")

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

print("[2] Construyendo cortadores calibrados con las correcciones del usuario...")

# 1. Cresta dorsal
fin_poly_yz = np.array([
    [-12.0, 1918.0],
    [-2.0,  1885.0],
    [205.0, 1870.0],
    [205.0, 1960.0],
    [-12.0, 1960.0]
], dtype=np.float32)
cs_fin = make_ccw(fin_poly_yz)
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

# 2. Ceño Frontal continuo que corre hasta enmarcar la orejera en Y=55:
brow_box = m3d.Manifold.cube([250.0, 155.0, 23.4], center=False).translate([-125.0, -100.0, 1808.6])
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

# 3. Alas de Thor (ACHATADAS 25% EN ALTURA, ALARGADAS 15% HACIA ATRÁS):
# Altura máxima achatada a Z ≈ 1897 mm (antes 1926 mm).
# Longitud hacia atrás alargada a Y ≈ 113 mm (antes 98 mm).
# Ángulo de ataque mucho más estilizado y agresivo:
wing_pts = [
    [-22.0, 1808.6],  # Empalme orgánico frontal con el ceño
    [-18.0, 1826.0],
    [-13.0, 1846.0],
    [ -7.0, 1866.0],
    [  0.0, 1882.0],
    [ 12.0, 1894.0],  # Coronación superior achatada suave
    [ 30.0, 1897.0],
    [ 60.0, 1894.0],
    [ 88.0, 1886.0],
    [113.0, 1874.0],  # Punta Pluma 1 (+15% hacia atrás, Y=113)
    [ 95.0, 1858.0],  # Punta Pluma 2 (+15% hacia atrás, Y=95)
    [ 79.0, 1844.0],  # Punta Pluma 3 (+15% hacia atrás, Y=79)
    [ 63.0, 1831.0],  # Punta Pluma 4 (pinza inferior, Y=63)
    [ -2.0, 1835.0],  # Destalonado bajo Pluma 4 (separa la pluma del ceño)
    [-10.0, 1821.0],
    [-18.0, 1808.6]   # Retorno a la base
]
cs_w = make_ccw(wing_pts)

# 3 ranuras chevron descendentes adaptadas a la nueva esbeltez y ángulo de ataque:
slot1 = make_rounded_slot( 12.0, 1881.0, 118.0, 1866.0, th=4.0, r_tip=1.8)
slot2 = make_rounded_slot(  2.0, 1865.0, 100.0, 1851.0, th=4.0, r_tip=1.8)
slot3 = make_rounded_slot( -6.0, 1849.0,  84.0, 1837.0, th=4.0, r_tip=1.8)
cs_w = ((cs_w - slot1) - slot2) - slot3

raw_w = cs_w.extrude(100.0)
mat_wr = np.array([[0.0, 0.0, 1.0, 25.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_wl = np.array([[0.0, 0.0, -1.0, -25.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
wings_cutter = raw_w.transform(mat_wr) + raw_w.transform(mat_wl)

# 4. Zona Trasera (Banda en la depresión de la cúpula Z=1808.6 a 1842.0, disco blanco y 2 costillas):
rear_band = m3d.Manifold.cube([196.0, 95.0, 33.4], center=False).translate([-98.0, 135.0, 1808.6])
disc_cutter = m3d.Manifold.cylinder(80.0, 21.5, 21.5, 128).rotate([90, 0, 0]).translate([0, 235.0, 1815.6])
rib_r = m3d.Manifold.cube([14.0, 55.0, 202.0], center=False).translate([ 13.0, 170.0, 1616.0])
rib_l = m3d.Manifold.cube([14.0, 55.0, 202.0], center=False).translate([-27.0, 170.0, 1616.0])
rear_markings = (rear_band + rib_r + rib_l) - disc_cutter

# 5. Mejillas y Depresión del Pómulo:
# REQUERIMIENTO DEL USUARIO:
# a) "que todo lo de la depresión sea rojo incluidas las paredes. Hablo de la depresión del pómulo":
#    La depresión entera llega hasta Y=72.0 mm y sube hasta Z=1812.0 mm, cubriendo todas las paredes.
# b) "ese espacio que hay que es un puto ángulo recto desde el lado es horrible...
#    mejor pon de forma paralela a los bigotes el acabado de rojo final y échalo más hacia detrás.
#    Si te parece, hasta la mitad de la depresión":
#    En lugar del corte en ángulo recto en Y=-14 y Z=1715:
#    El remate rojo asciende con una línea DIAGONAL PARALELA A LOS BIGOTES:
#    Desde (Y = -5.0 mm, Z = 1600.0 mm) hasta la mitad de la depresión en (Y = 18.0 mm, Z = 1715.0 mm)!
#    Elimina completamente el ángulo recto y fluye paralelo a la inclinación de los bigotes.
#    A partir de (Y = 18.0 mm, Z = 1715.0 mm), el rojo cubre toda la depresión hasta Y = 72.0 mm.
cheek_pts = [
    [-110.0, 1600.0],
    [  -5.0, 1600.0],  # Línea diagonal paralela a los bigotes
    [  18.0, 1715.0],  # Llega hasta la mitad de la depresión sin ángulos rectos
    [  72.0, 1715.0],  # Cubre toda la base de la depresión del pómulo
    [  72.0, 1780.0],  # Cubre la pared posterior completa de la depresión
    [  55.0, 1812.0],  # Conecta suavemente con la pared superior y el ceño
    [-110.0, 1812.0]
]
cs_ch = make_ccw(cheek_pts)
raw_ch = cs_ch.extrude(120.0)
mat_ch_r = np.array([[0.0, 0.0, 1.0, 8.85], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_ch_l = np.array([[0.0, 0.0, -1.0, -8.85], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
cheeks_raw = raw_ch.transform(mat_ch_r) + raw_ch.transform(mat_ch_l)

snout_box = m3d.Manifold.cube([92.0, 55.0, 122.0], center=False).translate([-46.0, -97.0, 1620.0])
nose_teeth_bridge = m3d.Manifold.cube([32.0, 50.0, 125.0], center=False).translate([-16.0, -98.0, 1620.0])

# Exclusión cilíndrica de la orejera (enmarca el disco de la orejera con R=33.0 mm en Y=77.7, Z=1789.1):
ear_clear_r = m3d.Manifold.cylinder(80.0, 33.0, 33.0, 96).rotate([0, 90, 0]).translate([85.0, 77.7, 1789.1])
ear_clear_l = m3d.Manifold.cylinder(80.0, 33.0, 33.0, 96).rotate([0, -90, 0]).translate([-85.0, 77.7, 1789.1])
ear_exclusion = ear_clear_r + ear_clear_l

# Bigotes Curvos con Puntas Redondeadas:
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

# Ojeras esbeltas paralelas al visor:
p1_r = np.array([28.0, -48.0, 1795.0], dtype=np.float32)
p2_r = np.array([62.0, -33.0, 1782.0], dtype=np.float32)
p0_r = (p1_r + p2_r) / 2.0
u_r = (p2_r - p1_r) / np.linalg.norm(p2_r - p1_r)
u_len = np.linalg.norm(p2_r - p1_r)

w_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
v_r = np.cross(w_ref, u_r)
v_r /= np.linalg.norm(v_r)
if v_r[2] > 0:
    v_r = -v_r
w_r = np.cross(u_r, v_r)
w_r /= np.linalg.norm(w_r)

v_down = -3.5
v_up = 6.0
r_c = 2.0
oj_pts = [
    [-u_len/2.0, v_up],
    [-u_len/2.0, v_down + r_c]
]
for a in np.linspace(np.pi, 1.5*np.pi, 8):
    oj_pts.append([-u_len/2.0 + r_c + r_c * np.cos(a), v_down + r_c + r_c * np.sin(a)])
for a in np.linspace(1.5*np.pi, 2.0*np.pi, 8):
    oj_pts.append([ u_len/2.0 - r_c + r_c * np.cos(a), v_down + r_c + r_c * np.sin(a)])
oj_pts.append([ u_len/2.0, v_up])

cs_oj = make_ccw(oj_pts)
raw_oj = cs_oj.extrude(60.0).translate([0, 0, -30.0])

mat_oj_r = np.eye(4, dtype=np.float32)
mat_oj_r[:3, 0] = u_r
mat_oj_r[:3, 1] = v_r
mat_oj_r[:3, 2] = w_r
mat_oj_r[:3, 3] = p0_r

p1_l = np.array([-28.0, -48.0, 1795.0], dtype=np.float32)
p2_l = np.array([-62.0, -33.0, 1782.0], dtype=np.float32)
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

face_solid = cheeks_raw + snout_box + nose_teeth_bridge
face_cutter = (face_solid - claws_raw) - suboc_cutter

# Ensamblado total con exclusión cilíndrica de la orejera
total_red_cutter = (fin_cutter + brow_cutter + wings_cutter + rear_markings + face_cutter) - ear_exclusion

print("[3] Operaciones booleanas exactas...")
red_mani = helmet_mani ^ total_red_cutter
white_mani = helmet_mani - total_red_cutter

vol_r = red_mani.volume()
vol_w = white_mani.volume()
print(f"Volumen Rojo: {vol_r:.1f} mm³ | Base Blanca: {vol_w:.1f} mm³ | Delta: {abs(vol_r + vol_w - base_volume):.5f} mm³")

m_red = red_mani.to_mesh()
mesh_red = trimesh.Trimesh(vertices=m_red.vert_properties[:, :3], faces=m_red.tri_verts, process=False)
mesh_red.visual.vertex_colors = [185, 28, 28, 255]

m_white = white_mani.to_mesh()
mesh_white = trimesh.Trimesh(vertices=m_white.vert_properties[:, :3], faces=m_white.tri_verts, process=False)
mesh_white.visual.vertex_colors = [244, 244, 246, 255]

mouth_mesh.visual.vertex_colors = [20, 20, 22, 255]
ears_mesh.visual.vertex_colors = [240, 240, 244, 255]
aerators_mesh.visual.vertex_colors = [70, 70, 75, 255]

print("[4] Renderizando 4 vistas canónicas + perfil derecho...")
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

sheet.save("test_v27_allviews.png")
print("Listo! 4 vistas guardadas en: test_v27_allviews.png")
