import os
import trimesh
import manifold3d as m3d
import numpy as np
from PIL import Image

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
casco_dir = os.path.join(BASE_DIR, "Diseños 3D", "Casco")
greeb_dir = os.path.join(casco_dir, "Greeblies")

print("[1/5] Cargando casco Galactic Armory Phase 2 y accesorios...")
helmet_mesh = trimesh.load(os.path.join(casco_dir, "Phase 2 Main Body.stl"))
v_base = helmet_mesh.vertices.astype('float32')
f_base = helmet_mesh.faces.astype('uint32')
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))

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

print("[2/5] Construyendo cortadores geométricos de alta fidelidad...")

# 1. Cresta Dorsal
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

# 2. Ceño Frontal (corta en Y=18mm antes de la oreja) con muesca central de 34mm
brow_box = m3d.Manifold.cube([270.0, 118.0, 23.4], center=False).translate([-135.0, -100.0, 1808.6])
center_notch = m3d.Manifold.cube([34.0, 100.0, 30.0], center=False).translate([-17.0, -105.0, 1807.0])
brow_cutter = brow_box - center_notch

# 3. Alas de Thor Orgánicas (4 Plumas con empalme suave en la base y rastrillo diagonal canónico)
wing_poly = [
    # Raíz orgánica que nace del ceño (Z=1808.6mm, Y=-18..18)
    [-18.0, 1808.6],
    [ 18.0, 1808.6],
    [ 25.0, 1832.0],
    # Pluma 4 (Pinza inferior envolvente)
    [ 84.0, 1856.0],
    [ 62.0, 1860.0],
    [ 35.0, 1862.0],
    # Pluma 3
    [ 96.0, 1874.0],
    [ 72.0, 1878.0],
    [ 40.0, 1880.0],
    # Pluma 2
    [108.0, 1892.0],
    [ 82.0, 1896.0],
    [ 45.0, 1898.0],
    # Pluma 1 (Superior alargada siguiendo curvatura de cúpula)
    [122.0, 1910.0],
    [ 90.0, 1918.0],
    [ 50.0, 1922.0],
    [ 15.0, 1920.0],
    [-10.0, 1895.0],
    [-20.0, 1855.0]
]
cs_wing = make_ccw(wing_poly)

# 3 ranuras chevron blancas que separan limpiamente las 4 plumas (espesor 3.2mm)
for d_z in [1904.0, 1886.0, 1868.0]:
    slot = np.array([
        [ 32.0, d_z - 0.12 * 32.0 - 1.6],
        [140.0, d_z - 0.12 * 140.0 - 1.6],
        [140.0, d_z - 0.12 * 140.0 + 1.6],
        [ 32.0, d_z - 0.12 * 32.0 + 1.6]
    ], dtype=np.float32)
    cs_wing = cs_wing - m3d.CrossSection([slot])

raw_w = cs_wing.extrude(95.0)
mat_wr = np.array([[0.0, 0.0, 1.0, 32.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_wl = np.array([[0.0, 0.0, -1.0, -32.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mw_r = raw_w.transform(mat_wr)
mw_l = raw_w.transform(mat_wl)
wings_cutter = mw_r + mw_l

# 4. Zona Trasera: Depresión Anatómica (Z=1818..1842) + Disco 100% Blanco + 2 Costillas
band_box = m3d.Manifold.cube([140.0, 60.0, 24.0], center=False).translate([-70.0, 168.0, 1818.0])
disc_cyl = m3d.Manifold.cylinder(80.0, 33.0, 33.0, 128).rotate([90, 0, 0]).translate([0, 235.0, 1801.5])

rib_pts = [[17.0, 1616.0], [29.0, 1616.0], [29.0, 1822.0], [17.0, 1822.0]]
cs_rib = make_ccw(rib_pts)
raw_rib = cs_rib.extrude(60.0)
mat_rib_r = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 175.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_rib_l = np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 175.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
rib_r = raw_rib.transform(mat_rib_r)
rib_l = raw_rib.transform(mat_rib_l)
rear_markings = (band_box + rib_r + rib_l) - disc_cyl

# 5. Mejillas, Morro y Bigotes Curvos
cheek_pts = [
    [-110.0, 1600.0],
    [ -15.0, 1600.0], # Cubre las 2 primeras ranuras diagonales del tubo
    [ -15.0, 1665.0],
    [  40.0, 1665.0], # Fondo de la mejilla frente al tubo de la orejera
    [  40.0, 1775.0], # Frente al disco de la orejera
    [  18.0, 1808.6], # Conecta con el corte del ceño (Y=18mm)
    [-25.0,  1808.6],
    [-25.0,  1798.0],
    [-110.0, 1798.0]
]
cs_ch = make_ccw(cheek_pts)
raw_ch = cs_ch.extrude(120.0)
mat_ch_r = np.array([[0.0, 0.0, 1.0, 8.85], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
mat_ch_l = np.array([[0.0, 0.0, -1.0, -8.85], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
ch_r = raw_ch.transform(mat_ch_r)
ch_l = raw_ch.transform(mat_ch_l)
cheeks_raw = ch_r + ch_l

# Morro central y tubos
snout_box = m3d.Manifold.cube([92.0, 55.0, 95.0], center=False).translate([-46.0, -97.0, 1620.0])
tubes_box = m3d.Manifold.cube([230.0, 85.0, 50.0], center=False).translate([-115.0, -65.0, 1618.0])

# Bigotes Curvos (3 franjas anguladas a 25° que abrazan la curvatura del tubo y avanzan al morro)
rings_cutter = m3d.Manifold()
for y_center in [-68.0, -52.0, -36.0]:
    rc = m3d.Manifold.cube([250.0, 6.5, 65.0], center=True).rotate([25, 0, 0]).translate([0, y_center, 1640.0])
    rings_cutter = rings_cutter + rc

# Máscara lateral para bigotes: activa desde |X|=22mm hasta |X|=125mm (reserva centro rojo |X| <= 22mm)
mask_wh_r = m3d.Manifold.cube([105.0, 200.0, 200.0], center=False).translate([ 22.0, -150.0, 1600.0])
mask_wh_l = m3d.Manifold.cube([105.0, 200.0, 200.0], center=False).translate([-127.0, -150.0, 1600.0])
whiskers_active = rings_cutter ^ (mask_wh_r + mask_wh_l)

# Ojeras blancas desplazadas al exterior (|X| de 44 a 84 mm)
suboc_box_r = m3d.Manifold.cube([40.0, 45.0, 12.0], center=False).translate([ 44.0, -75.0, 1787.0])
suboc_box_l = m3d.Manifold.cube([40.0, 45.0, 12.0], center=False).translate([-84.0, -75.0, 1787.0])
suboc_cutter = suboc_box_r + suboc_box_l

# Ensamblado total de la cara: mejillas + morro + tubos, con corte de bigotes y ojeras
face_solid = cheeks_raw + snout_box + tubes_box
face_cutter = (face_solid - whiskers_active) - suboc_cutter

# Visor 100% diáfano y hueco (Keep-out)
visor_clear_h = m3d.Manifold.cube([180.0, 60.0, 13.0], center=False).translate([-90.0, -80.0, 1795.5])
visor_clear_v = m3d.Manifold.cube([  17.5, 50.0, 53.5], center=False).translate([ -8.75, -80.0, 1742.0])
visor_clearance = visor_clear_h + visor_clear_v

total_red_cutter = (fin_cutter + brow_cutter + wings_cutter + rear_markings + face_cutter) - visor_clearance

print("[3/5] Ejecutando corte booleano manifold...")
red_markings_mani = helmet_mani ^ total_red_cutter
white_base_mani = helmet_mani - total_red_cutter

m_red = red_markings_mani.to_mesh()
mesh_red = trimesh.Trimesh(vertices=m_red.vert_properties[:, :3], faces=m_red.tri_verts, process=False)
mesh_red.visual.vertex_colors = [185, 28, 28, 255]

m_white = white_base_mani.to_mesh()
mesh_white = trimesh.Trimesh(vertices=m_white.vert_properties[:, :3], faces=m_white.tri_verts, process=False)
mesh_white.visual.vertex_colors = [244, 244, 246, 255]

mouth_mesh.visual.vertex_colors = [20, 20, 22, 255]
ears_mesh.visual.vertex_colors = [240, 240, 244, 255]
aerators_mesh.visual.vertex_colors = [70, 70, 75, 255]

print("[4/5] Renderizando 4 vistas canónicas con Trimesh...")
scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, ears_mesh, aerators_mesh])
center = scene.centroid
cam_dist = 680.0
elev = np.radians(6.0)

angles = [
    ("front", 0.0),
    ("iso", 35.0),
    ("side", 90.0),
    ("rear", 180.0)
]

imgs = {}
for name, yaw_deg in angles:
    yaw = np.radians(yaw_deg)
    cam_x = cam_dist * np.sin(yaw) * np.cos(elev)
    cam_y = -cam_dist * np.cos(yaw) * np.cos(elev)
    cam_z = cam_dist * np.sin(elev)
    cam_pos = np.array([cam_x, cam_y, cam_z])
    
    forward = -cam_pos / np.linalg.norm(cam_pos)
    up = np.array([0, 0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    actual_up = np.cross(right, forward)
    actual_up /= np.linalg.norm(actual_up)
    
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = actual_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = cam_pos + center
    
    scene.camera_transform = c2w
    data = scene.save_image(resolution=(1200, 1200))
    from io import BytesIO
    img = Image.open(BytesIO(data))
    imgs[name] = img
    img.save(f"test_flawless_3d_{name}.png")
    print(f"  Renderizado: test_flawless_3d_{name}.png")

# Combinar en panel 2x2
w, h = 1200, 1200
canvas = Image.new("RGBA", (w*2, h*2), (255, 255, 255, 255))
canvas.paste(imgs["front"], (0, 0))
canvas.paste(imgs["iso"], (w, 0))
canvas.paste(imgs["side"], (0, h))
canvas.paste(imgs["rear"], (w, h))

canvas.save("test_flawless_3d_turnaround.png")
print("[5/5] ¡Render 4 vistas completado con éxito en test_flawless_3d_turnaround.png!")
