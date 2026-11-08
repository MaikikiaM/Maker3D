import os
import re
import numpy as np
import trimesh
import manifold3d as m3d
from PIL import Image, ImageDraw
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

print("[2] Construyendo cortadores hiper-calibrados con ranuras paralelas dinámicas...")

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

# B. Ceño Frontal (Z = 1808.6 a 1832.0 mm, Y = -100 a -15 mm, muesca blanca central 36 mm):
brow_box = m3d.Manifold.cube([240.0, 85.0, 23.4], center=False).translate([-120.0, -100.0, 1808.6])
center_notch = m3d.Manifold.cube([36.0, 95.0, 30.0], center=False).translate([-18.0, -105.0, 1807.0])
brow_cutter = brow_box - center_notch

# C. Alas de Thor (4 Plumas Dinámicas Orgánicas con Ranuras Inclinadas Aerodinámicas):
wing_pts = [
    [-18.0, 1808.6], # Base orgánica que empalma con el ceño
    [-16.0, 1832.0],
    [-14.0, 1855.0],
    [-10.0, 1878.0],
    [ -4.0, 1900.0],
    [  4.0, 1918.0],
    [ 16.0, 1926.0], # Corona curvada continua que fluye hacia la pluma 1
    [ 45.0, 1926.0],
    [ 72.0, 1922.0],
    [ 94.0, 1912.0], # Punta Pluma 1 (superior larga)
    [ 80.0, 1891.0], # Punta Pluma 2
    [ 66.0, 1870.0], # Punta Pluma 3
    [ 56.0, 1849.0], # Punta Pluma 4 (inferior esbelta)
    [  4.0, 1843.0], # Parte inferior de Pluma 4 (espacio blanco debajo)
    [  4.0, 1808.6], # Borde posterior del tallo empalmando con ceño
    [-18.0, 1808.6]
]
cs_w = make_ccw(wing_pts)

# 3 ranuras chevron inclinadas dinámicamente (~4° de elevación hacia atrás):
slots = [
    # Ranura 1: entre Pluma 1 y 2
    [[  6.0, 1898.0], [ 92.0, 1903.5], [ 92.0, 1906.8], [  6.0, 1901.3]],
    # Ranura 2: entre Pluma 2 y 3
    [[ -4.0, 1877.0], [ 78.0, 1882.5], [ 78.0, 1885.8], [ -4.0, 1880.3]],
    # Ranura 3: entre Pluma 3 y 4
    [[ -9.0, 1856.0], [ 64.0, 1861.5], [ 64.0, 1864.8], [ -9.0, 1859.3]],
]
for s in slots:
    cs_w = cs_w - make_ccw(s)

raw_w = cs_w.extrude(100.0)
mat_wr = np.array([[0.0, 0.0, 1.0, 25.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_wl = np.array([[0.0, 0.0, -1.0, -25.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
wings_cutter = raw_w.transform(mat_wr) + raw_w.transform(mat_wl)

# D. Zona Trasera (Banda en la depresión de la cúpula Z=1808.6 a 1844.0, disco 100% blanco y 2 costillas):
rear_band = m3d.Manifold.cube([192.0, 95.0, 35.4], center=False).translate([-96.0, 135.0, 1808.6])
disc_cutter = m3d.Manifold.cylinder(80.0, 21.5, 21.5, 128).rotate([90, 0, 0]).translate([0, 235.0, 1815.6])
rib_r = m3d.Manifold.cube([14.0, 55.0, 202.0], center=False).translate([ 13.0, 170.0, 1616.0])
rib_l = m3d.Manifold.cube([14.0, 55.0, 202.0], center=False).translate([-27.0, 170.0, 1616.0])
rear_markings = (rear_band + rib_r + rib_l) - disc_cutter

# E. Morro, Dientes continuos, Ojeras elegantes y 3 Bigotes:
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

claws_raw = m3d.Manifold()
for y_c, x_in in [(-54.0, 26.0), (-41.0, 22.0), (-28.0, 20.0)]:
    sr = m3d.Manifold.cube([100.0, 5.5, 95.0], center=True).rotate([26.5, 0, 0]).translate([70.0, y_c, 1641.0])
    sl = m3d.Manifold.cube([100.0, 5.5, 95.0], center=True).rotate([26.5, 0, 0]).translate([-70.0, y_c, 1641.0])
    mr = m3d.Manifold.cube([90.0, 120.0, 70.0], center=False).translate([x_in, -110.0, 1618.0])
    ml = m3d.Manifold.cube([90.0, 120.0, 70.0], center=False).translate([-x_in - 90.0, -110.0, 1618.0])
    claws_raw = claws_raw + (sr ^ mr) + (sl ^ ml)

suboc_box_r = m3d.Manifold.cube([46.0, 60.0, 23.0], center=False).translate([ 30.0, -85.0, 1778.0])
suboc_box_l = m3d.Manifold.cube([46.0, 60.0, 23.0], center=False).translate([-76.0, -85.0, 1778.0])
suboc_cutter = suboc_box_r + suboc_box_l

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

test_img_path = os.path.join(out_dir, "test_v20_4views_check.png")
sheet.save(test_img_path)
print(f"Imagen de prueba guardada en: {test_img_path}")
