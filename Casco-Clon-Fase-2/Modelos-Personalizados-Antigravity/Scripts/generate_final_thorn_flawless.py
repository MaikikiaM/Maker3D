import os
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import manifold3d as m3d
from PIL import Image

print("=== INICIANDO PIPELINE MAESTRO DE COMMANDER THORN (THE CLONE WARS S6) ===")

# Directorios de salida
script_dir = os.path.dirname(os.path.abspath(__file__))
casco_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
antigravity_dir = os.path.join(casco_root, "Modelos-Personalizados-Antigravity")
casco_base_dir = os.path.join(casco_root, "Archivos-Base-Galactic-Armory")

OUT_05 = os.path.join(antigravity_dir, "Cascos-Pintados-Clones")
OUT_04 = os.path.join(antigravity_dir, "Guias-Visuales-y-Planos")
OUT_03 = os.path.join(antigravity_dir, "Stencils-Pintura-Clones")

os.makedirs(OUT_05, exist_ok=True)
os.makedirs(OUT_04, exist_ok=True)
os.makedirs(OUT_03, exist_ok=True)

# 1. CARGA DEL CASCO BASE FASE 2
print("[1/6] Cargando modelo base Galactic Armory Phase 2...")
base_mesh = trimesh.load(os.path.join(casco_base_dir, "Phase 2 Main Body.stl"))
v_base = base_mesh.vertices.astype('float32')
f_base = base_mesh.faces.astype('uint32')
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))
print(f"Casco base cargado: {len(v_base)} vértices, volumen = {helmet_mani.volume():.1f} mm³")

# 2. CONSTRUCCIÓN DE LAS ALAS LATERALES DE THOR (CÚPULA)
print("[2/6] Generando alas laterales de Thor con 4 plumas escalonadas y 3 ranuras diagonales...")
wing_pts = np.array([
    # Pluma 4 (inferior-frontal sobre la ceja)
    [-46.0, 1845.0],
    [-20.0, 1845.0],
    [15.0,  1850.0],
    [36.0,  1858.0],
    # Festón cóncavo inferior-trasero
    [52.0,  1874.0],
    [64.0,  1868.0],
    [72.0,  1872.0],
    [80.0,  1880.0],
    # Lóbulo posterior y arco superior
    [76.0,  1898.0],
    [54.0,  1914.0],
    [32.0,  1922.0],
    [10.0,  1918.0],
    # Punta frontal Pluma 1 (superior)
    [-18.0, 1905.0],
    # Salto a punta frontal Pluma 2
    [-10.0, 1895.0],
    [-28.0, 1885.0],
    # Salto a punta frontal Pluma 3
    [-20.0, 1875.0],
    [-38.0, 1865.0],
    # Salto a punta frontal Pluma 4
    [-30.0, 1855.0],
    [-46.0, 1845.0]
], dtype=np.float32)

cs_wing = m3d.CrossSection([wing_pts[:-1]])

# 3 ranuras oblicuas que cortan las plumas
for d_val in [1890.0, 1868.0, 1846.0]:
    slot_pts = np.array([
        [-60.0, d_val - 0.85*(-60.0) - 2.8],
        [90.0,  d_val - 0.85*(90.0)  - 2.8],
        [90.0,  d_val - 0.85*(90.0)  + 2.8],
        [-60.0, d_val - 0.85*(-60.0) + 2.8]
    ], dtype=np.float32)
    cs_wing = cs_wing - m3d.CrossSection([slot_pts])

raw_wing_solid = cs_wing.extrude(115.0)
m_rw = raw_wing_solid.to_mesh()
vw = m_rw.vert_properties
fw = m_rw.tri_verts

# Ala derecha: X en [28, 143] (cresta central y separación blanca protegida)
vw_right = np.column_stack([vw[:, 2] + 28.0, vw[:, 0], vw[:, 1]]).astype('float32')
triw_right = fw.copy()
mani_w_right = m3d.Manifold(m3d.Mesh(vert_properties=vw_right, tri_verts=triw_right))
if mani_w_right.volume() <= 0:
    triw_right[:, [0, 1]] = triw_right[:, [1, 0]]
    mani_w_right = m3d.Manifold(m3d.Mesh(vert_properties=vw_right, tri_verts=triw_right))

# Ala izquierda: simétrica
vw_left = np.column_stack([-(vw[:, 2] + 28.0), vw[:, 0], vw[:, 1]]).astype('float32')
triw_left = triw_right.copy()
triw_left[:, [0, 1]] = triw_left[:, [1, 0]]
mani_w_left = m3d.Manifold(m3d.Mesh(vert_properties=vw_left, tri_verts=triw_left))
wings_cutter = mani_w_right + mani_w_left

# 3. CONSTRUCCIÓN DE MEJILLAS, MORRO Y TUBOS INFERIORES CON 3 ANILLOS BLANCOS
print("[3/6] Generando morro, mejillas y tubos aeradores con 3 anillos blancos envolventes...")
# Hueco de mejillas siguiendo la pendiente real del casco
cheek_profile = np.array([
    [-75.0, 1665.0],
    [12.0,  1695.0],
    [12.0,  1754.0],
    [-75.0, 1754.0]
], dtype=np.float32)
cs_cheek = m3d.CrossSection([cheek_profile])
raw_cheek = cs_cheek.extrude(170.0).translate([0, 0, -85.0])
m_rc = raw_cheek.to_mesh()
vc = m_rc.vert_properties
fc = m_rc.tri_verts
vc_global = np.column_stack([vc[:, 2], vc[:, 0], vc[:, 1]]).astype('float32')
tric = fc.copy()
cheeks_mani = m3d.Manifold(m3d.Mesh(vert_properties=vc_global, tri_verts=tric))
if cheeks_mani.volume() <= 0:
    tric[:, [0, 1]] = tric[:, [1, 0]]
    cheeks_mani = m3d.Manifold(m3d.Mesh(vert_properties=vc_global, tri_verts=tric))

# Tubos inferiores: abarcan X en [-115, 115], Y en [-65, 24], Z en [1620, 1675]
tubes_box = m3d.Manifold.cube([230.0, 89.0, 55.0], center=False).translate([-115.0, -65.0, 1620.0])

# Los 3 anillos blancos que envuelven cada tubo
ring_cutter = m3d.Manifold()
for y_start, y_end in [(-40.0, -33.0), (-18.0, -11.0), (4.0, 11.0)]:
    rc = m3d.Manifold.cube([250.0, y_end - y_start, 70.0]).translate([-125.0, y_start, 1615.0])
    ring_cutter = ring_cutter + rc
tubes_with_rings = tubes_box - ring_cutter

# Mentón y vocoder
chin_box = m3d.Manifold.cube([56.0, 35.0, 50.0], center=False).translate([-28.0, -95.0, 1620.0])
face_cutter = cheeks_mani + tubes_with_rings + chin_box

# Cortador total rojo
total_red_cutter = wings_cutter + face_cutter

# Operaciones booleanas exactas
red_markings_mani = helmet_mani ^ total_red_cutter
white_base_mani = helmet_mani - red_markings_mani
print(f"Patrón rojo generado: {red_markings_mani.volume():.1f} mm³")
print(f"Base blanca generada: {white_base_mani.volume():.1f} mm³")

# 4. CONSTRUCCIÓN DE LA VISERA DE COMANDANTE (AERODINÁMICA Y FACETADA) Y ANTENA
print("[4/6] Modelando visera de comandante aerodinámica con protectores laterales y antena...")
theta = np.linspace(-np.radians(60), np.radians(60), 24)
brow_x = 108.0 * np.sin(theta)
brow_y = 10.0 - 66.0 * np.cos(theta)
brow_z = 1806.0 + 6.0 * np.cos(theta)

lip_proj = 38.0 * np.cos(theta * 0.7)
lip_x = 86.0 * np.sin(theta)
lip_y = brow_y - lip_proj
lip_z = brow_z - 17.0

# Alero de la visera
awning_mani = m3d.Manifold()
for i in range(len(theta) - 1):
    p0 = np.array([brow_x[i], brow_y[i], brow_z[i]])
    p1 = np.array([brow_x[i+1], brow_y[i+1], brow_z[i+1]])
    p2 = np.array([lip_x[i+1], lip_y[i+1], lip_z[i+1]])
    p3 = np.array([lip_x[i], lip_y[i], lip_z[i]])
    
    seg_pts = np.vstack([
        [p0, p1, p2, p3],
        [p0 + [0, 0, 7], p1 + [0, 0, 7], p2 + [0, 0, 5], p3 + [0, 0, 5]]
    ]).astype('float32')
    
    thull = trimesh.convex.convex_hull(seg_pts)
    seg_mani = m3d.Manifold(m3d.Mesh(vert_properties=thull.vertices.astype('float32'), tri_verts=thull.faces.astype('uint32')))
    awning_mani = awning_mani + seg_mani

# Protectores laterales que caen a los lados de la lente
shield_r_pts = np.array([
    [lip_x[-1], lip_y[-1], lip_z[-1]],
    [brow_x[-1], brow_y[-1], brow_z[-1]],
    [brow_x[-1], brow_y[-1], 1768.0],
    [lip_x[-1], lip_y[-1], 1768.0]
], dtype=np.float32)
shield_r_hull = trimesh.convex.convex_hull(np.vstack([shield_r_pts, shield_r_pts - [5, 0, 0]]))
shield_r_mani = m3d.Manifold(m3d.Mesh(vert_properties=shield_r_hull.vertices.astype('float32'), tri_verts=shield_r_hull.faces.astype('uint32')))

shield_l_pts = np.array([
    [lip_x[0], lip_y[0], lip_z[0]],
    [brow_x[0], brow_y[0], brow_z[0]],
    [brow_x[0], brow_y[0], 1768.0],
    [lip_x[0], lip_y[0], 1768.0]
], dtype=np.float32)
shield_l_hull = trimesh.convex.convex_hull(np.vstack([shield_l_pts, shield_l_pts + [5, 0, 0]]))
shield_l_mani = m3d.Manifold(m3d.Mesh(vert_properties=shield_l_hull.vertices.astype('float32'), tri_verts=shield_l_hull.faces.astype('uint32')))

# Discos de pivote en las orejas
disc_r = m3d.Manifold.cylinder(8.0, 18.0, 18.0, circular_segments=32).rotate([0, 90, 0]).translate([120.0, 68.0, 1785.0])
disc_l = m3d.Manifold.cylinder(8.0, 18.0, 18.0, circular_segments=32).rotate([0, -90, 0]).translate([-120.0, 68.0, 1785.0])

# Brazos laterales hacia los pivotes
arm_pts_r = np.array([
    [brow_y[-1], brow_z[-1] + 7.0],
    [68.0, 1795.0],
    [68.0, 1775.0],
    [brow_y[-1], brow_z[-1] - 3.0]
], dtype=np.float32)
cs_arm_r = m3d.CrossSection([arm_pts_r])
raw_arm_r = cs_arm_r.extrude(16.0)
mar = raw_arm_r.to_mesh()
var = mar.vert_properties
far = mar.tri_verts
var_global = np.column_stack([var[:, 2] + 112.0, var[:, 0], var[:, 1]]).astype('float32')
triar = far.copy()
arm_r_mani = m3d.Manifold(m3d.Mesh(vert_properties=var_global, tri_verts=triar))
if arm_r_mani.volume() <= 0:
    triar[:, [0, 1]] = triar[:, [1, 0]]
    arm_r_mani = m3d.Manifold(m3d.Mesh(vert_properties=var_global, tri_verts=triar))

var_l_global = np.column_stack([-(var[:, 2] + 112.0), var[:, 0], var[:, 1]]).astype('float32')
trial = triar.copy()
trial[:, [0, 1]] = trial[:, [1, 0]]
arm_l_mani = m3d.Manifold(m3d.Mesh(vert_properties=var_l_global, tri_verts=trial))

complete_visor_mani = awning_mani + shield_r_mani + shield_l_mani + arm_r_mani + arm_l_mani + disc_r + disc_l

# Antena en oreja derecha
ant_rod = m3d.Manifold.cylinder(190.0, 3.2, 3.2, circular_segments=24).translate([125.0, 78.0, 1785.0])
ant_base = m3d.Manifold.cube([12.0, 16.0, 22.0], center=True).translate([124.0, 78.0, 1795.0])
antenna_mani = ant_rod + ant_base

# Detalle geométrico superior de la visera (corchetes gris claro `[   ]`)
decal_l = trimesh.creation.box([26.0, 3.5, 1.5]).apply_translation([-26.0, -84.0, 1801.0])
decal_r = trimesh.creation.box([26.0, 3.5, 1.5]).apply_translation([26.0, -84.0, 1801.0])
decal_mesh = decal_l + decal_r

# Lente de visor oscura interna
lens_x = 94.0 * np.sin(theta)
lens_y = 10.0 - 58.0 * np.cos(theta)
lens_pts = []
for i in range(len(theta)):
    lens_pts.append([lens_x[i], lens_y[i], 1752.0])
    lens_pts.append([lens_x[i], lens_y[i], 1794.0])
lens_pts = np.array(lens_pts, dtype=np.float32)
lens_hull = trimesh.convex.convex_hull(np.vstack([lens_pts, lens_pts + [0, 4, 0]]))
lens_mesh = trimesh.Trimesh(vertices=lens_hull.vertices, faces=lens_hull.faces)

# Greeblies originales
ears_l = trimesh.load(os.path.join(BASE_DIR, r"Diseños 3D\Casco\Greeblies\Ear Piece Left.stl"))
ears_r = trimesh.load(os.path.join(BASE_DIR, r"Diseños 3D\Casco\Greeblies\Ear Piece Right.stl"))
aero_l = trimesh.load(os.path.join(BASE_DIR, r"Diseños 3D\Casco\Greeblies\Aerator Left - Mesh.stl"))
aero_r = trimesh.load(os.path.join(BASE_DIR, r"Diseños 3D\Casco\Greeblies\Aerator Right - Mesh.stl"))
mouth = trimesh.load(os.path.join(BASE_DIR, r"Diseños 3D\Casco\Greeblies\Mouth Center Detail.stl"))

greeblies_mesh = ears_l + ears_r + aero_l + aero_r

# Conversión de Manifolds a Trimesh
m_white = white_base_mani.to_mesh()
mesh_white = trimesh.Trimesh(vertices=m_white.vert_properties[:, :3], faces=m_white.tri_verts)

m_red = red_markings_mani.to_mesh()
mesh_red = trimesh.Trimesh(vertices=m_red.vert_properties[:, :3], faces=m_red.tri_verts)

m_vis = complete_visor_mani.to_mesh()
mesh_visor = trimesh.Trimesh(vertices=m_vis.vert_properties[:, :3], faces=m_vis.tri_verts)

m_ant = antenna_mani.to_mesh()
mesh_antenna = trimesh.Trimesh(vertices=m_ant.vert_properties[:, :3], faces=m_ant.tri_verts)

# 5. EXPORTACIÓN DE STLS INDIVIDUALES
print("[5/6] Exportando piezas STL y ensamblaje 3MF para Bambu Studio / PrusaSlicer...")
mesh_white.export(os.path.join(OUT_05, "01_Casco_Base_Blanco.stl"))
mesh_red.export(os.path.join(OUT_05, "02_Patron_Rojo_Thorn.stl"))
mesh_visor.export(os.path.join(OUT_05, "03_Visera_Comandante_Gris.stl"))
decal_mesh.export(os.path.join(OUT_05, "04_Detalle_Visera_GrisClaro.stl"))
mesh_antenna.export(os.path.join(OUT_05, "05_Antena_Comandante.stl"))
lens_mesh.export(os.path.join(OUT_05, "06_Lentes_Visor_Negro.stl"))
mouth.export(os.path.join(OUT_05, "07_Vocoder_Y_Dientes_Negro.stl"))
greeblies_mesh.export(os.path.join(OUT_05, "08_Accesorios_Greeblies_Gris.stl"))

# 6. EMPAQUETADO 3MF MULTICOLOR NATIVO PARA BAMBU STUDIO
target_3mf = os.path.join(OUT_05, "Clone_P2_Thorn_Guardia_Coruscant_Multicolor_Bambu_Prusa.3mf")

# Construir XML 3D model
NS_3MF = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_BAMBU = "http://schemas.bambulab.com/package/2021"

ET.register_namespace('', NS_3MF)
ET.register_namespace('p', NS_BAMBU)

model_elem = ET.Element(f'{{{NS_3MF}}}model', attrib={'unit': 'millimeter', 'xml:lang': 'en-US'})
res_elem = ET.SubElement(model_elem, f'{{{NS_3MF}}}resources')

components_list = [
    ("01_Casco_Base_Blanco", mesh_white, 1),
    ("02_Patron_Rojo_Thorn", mesh_red, 2),
    ("03_Visera_Comandante_Gris", mesh_visor, 3),
    ("04_Detalle_Visera_GrisClaro", decal_mesh, 1),
    ("05_Antena_Comandante", mesh_antenna, 3),
    ("06_Lentes_Visor_Negro", lens_mesh, 4),
    ("07_Vocoder_Y_Dientes_Negro", mouth, 4),
    ("08_Accesorios_Greeblies_Gris", greeblies_mesh, 3)
]

for idx, (name, m_part, extr_id) in enumerate(components_list, start=1):
    obj_elem = ET.SubElement(res_elem, f'{{{NS_3MF}}}object', attrib={
        'id': str(idx),
        'type': 'model',
        'name': name
    })
    mesh_elem = ET.SubElement(obj_elem, f'{{{NS_3MF}}}mesh')
    verts_elem = ET.SubElement(mesh_elem, f'{{{NS_3MF}}}vertices')
    for v in m_part.vertices:
        ET.SubElement(verts_elem, f'{{{NS_3MF}}}vertex', attrib={
            'x': f"{v[0]:.4f}", 'y': f"{v[1]:.4f}", 'z': f"{v[2]:.4f}"
        })
    tris_elem = ET.SubElement(mesh_elem, f'{{{NS_3MF}}}triangles')
    for tri in m_part.faces:
        ET.SubElement(tris_elem, f'{{{NS_3MF}}}triangle', attrib={
            'v1': str(tri[0]), 'v2': str(tri[1]), 'v3': str(tri[2])
        })

# Objeto ensamble que agrupa todas las piezas
assembly_id = len(components_list) + 1
assembly_obj = ET.SubElement(res_elem, f'{{{NS_3MF}}}object', attrib={
    'id': str(assembly_id),
    'type': 'model',
    'name': 'Casco_Comandante_Thorn_Clone_Wars'
})
comps_elem = ET.SubElement(assembly_obj, f'{{{NS_3MF}}}components')
for idx in range(1, len(components_list) + 1):
    ET.SubElement(comps_elem, f'{{{NS_3MF}}}component', attrib={
        'objectid': str(idx),
        'transform': '1 0 0 0 1 0 0 0 1 0 0 0'
    })

# Build element
build_elem = ET.SubElement(model_elem, f'{{{NS_3MF}}}build')
ET.SubElement(build_elem, f'{{{NS_3MF}}}item', attrib={
    'objectid': str(assembly_id),
    'transform': '1 0 0 0 1 0 0 0 1 0 0 0'
})

xml_3dmodel = ET.tostring(model_elem, encoding='utf-8', xml_declaration=True)

# Configs de Bambu Studio
model_settings = "; model settings\n"
for idx, (name, _, extr_id) in enumerate(components_list, start=1):
    model_settings += f"[object_{idx}]\nname = {name}\nextruder = {extr_id}\n\n"

project_settings = """[project]
version = 1.0.0
filament_colours = #F2F2F5;#B91216;#32363A;#141416
filament_names = Blanco Base;Rojo Thorn TCW;Gris Visera/Antena;Negro Visor/Vocoder
filament_type = PLA;PLA;PLA;PLA
"""

slice_info = """[slice]
layer_height = 0.16
initial_layer_print_height = 0.2
"""

content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="config" ContentType="text/plain"/>
</Types>
"""

rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""

with zipfile.ZipFile(target_3mf, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    z.writestr('3D/3dmodel.model', xml_3dmodel)
    z.writestr('Metadata/model_settings.config', model_settings)
    z.writestr('Metadata/project_settings.config', project_settings)
    z.writestr('Metadata/slice_info.config', slice_info)
    z.writestr('[Content_Types].xml', content_types)
    z.writestr('_rels/.rels', rels)

print(f"Proyecto 3MF multicolor guardado en: {target_3mf}")

# 7. EXPORTAR VISTA INTERACTIVA GLB CON MATERIALES PBR
print("[6/6] Exportando modelo interactivo GLB y renderizando guía visual 4 vistas...")
# Asignar colores a Trimesh visual
mesh_white.visual.vertex_colors = [242, 242, 245, 255]
# Nudge visual de 0.25mm para el modelo GLB y el render libre de z-fighting
mesh_red_vis = mesh_red.copy()
mesh_red_vis.vertices += mesh_red_vis.vertex_normals * 0.25
mesh_red_vis.visual.vertex_colors = [185, 18, 22, 255]

mesh_visor.visual.vertex_colors = [50, 54, 58, 255]
decal_mesh.visual.vertex_colors = [205, 210, 215, 255]
mesh_antenna.visual.vertex_colors = [50, 54, 58, 255]
lens_mesh.visual.vertex_colors = [20, 20, 22, 255]
mouth.visual.vertex_colors = [25, 25, 28, 255]
greeblies_mesh.visual.vertex_colors = [95, 100, 105, 255]

scene = trimesh.Scene([
    mesh_white, mesh_red_vis, mesh_visor, decal_mesh,
    mesh_antenna, lens_mesh, mouth, greeblies_mesh
])

glb_path = os.path.join(OUT_05, "Clone_P2_Thorn_Guardia_Coruscant_Pintado.glb")
scene.export(glb_path)
print(f"Modelo GLB interactivo exportado en: {glb_path}")

# Render 4 vistas de auditoría de precisión
def make_camera_matrix(eye, target, up=np.array([0, 0, 1])):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = true_up
    mat[:3, 2] = -forward
    mat[:3, 3] = eye
    return mat

centroid = np.array([0.0, 50.0, 1780.0])
dist = 570.0

views = {
    'front': make_camera_matrix(centroid + np.array([0, -dist, 10]), centroid),
    'iso': make_camera_matrix(centroid + np.array([dist*0.65, -dist*0.65, dist*0.35]), centroid),
    'side': make_camera_matrix(centroid + np.array([dist, 0, 15]), centroid),
    'top': make_camera_matrix(centroid + np.array([0, 1, dist]), centroid, up=np.array([0, 1, 0]))
}

imgs = []
for name, mat in views.items():
    scene.camera_transform = mat
    png = scene.save_image(resolution=(700, 700))
    with open(f'view_final_{name}.png', 'wb') as f:
        f.write(png)
    imgs.append(Image.open(f'view_final_{name}.png'))

grid = Image.new('RGB', (1400, 1400), (25, 25, 30))
grid.paste(imgs[0], (0, 0))
grid.paste(imgs[1], (700, 0))
grid.paste(imgs[2], (0, 700))
grid.paste(imgs[3], (700, 700))

render_out1 = os.path.join(OUT_04, "thorn_accurate_render.png")
render_out2 = os.path.join(OUT_04, "thorn_canon_4views_render.png")
grid.save(render_out1)
grid.save(render_out2)
print(f"Renders guardados en:\n- {render_out1}\n- {render_out2}")

# 8. GENERAR STENCIL VECTORIAL SVG
stencil_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" width="1000" height="1000">
  <rect width="1000" height="1000" fill="#1e1e24"/>
  <text x="50" y="60" fill="#ffffff" font-family="Arial" font-size="28" font-weight="bold">PLANTILLA DE PINTURA / STENCILS: COMANDANTE THORN (CORUSCANT GUARD)</text>
  <text x="50" y="95" fill="#a0a5b0" font-family="Arial" font-size="16">Fase 2 Clon - Diseño Fiel The Clone Wars (Temporada 6, Scipio) | Escala 1:1 Cosplay &amp; Miniaturas</text>
  
  <!-- Ala Izquierda -->
  <g transform="translate(150, 200)">
    <text x="0" y="-30" fill="#ff4d4d" font-family="Arial" font-size="18" font-weight="bold">Ala Cúpula Izquierda (4 Plumas + 3 Ranuras)</text>
    <path d="M 50 180 L 120 180 L 220 160 L 280 130 C 310 100 330 60 320 20 C 300 -40 220 -80 120 -85 L 0 -45 L 80 -15 L -20 0 L 70 20 L -30 40 L 50 60 Z" fill="#b91216" stroke="#ffffff" stroke-width="3"/>
  </g>

  <!-- Ala Derecha -->
  <g transform="translate(850, 200) scale(-1, 1)">
    <text x="0" y="-30" fill="#ff4d4d" font-family="Arial" font-size="18" font-weight="bold" transform="scale(-1, 1)" text-anchor="end">Ala Cúpula Derecha</text>
    <path d="M 50 180 L 120 180 L 220 160 L 280 130 C 310 100 330 60 320 20 C 300 -40 220 -80 120 -85 L 0 -45 L 80 -15 L -20 0 L 70 20 L -30 40 L 50 60 Z" fill="#b91216" stroke="#ffffff" stroke-width="3"/>
  </g>

  <!-- 3 Anillos Tubos Aeradores -->
  <g transform="translate(150, 650)">
    <text x="0" y="-30" fill="#ff4d4d" font-family="Arial" font-size="18" font-weight="bold">3 Anillos Blancos por Tubo Inferior (Aeradores)</text>
    <rect x="0" y="0" width="80" height="200" rx="10" fill="#b91216"/>
    <rect x="110" y="0" width="80" height="200" rx="10" fill="#b91216"/>
    <rect x="220" y="0" width="80" height="200" rx="10" fill="#b91216"/>
    <!-- Franjas blancas envolventes -->
    <rect x="15" y="30" width="50" height="140" fill="#ffffff"/>
    <rect x="125" y="30" width="50" height="140" fill="#ffffff"/>
    <rect x="235" y="30" width="50" height="140" fill="#ffffff"/>
    <text x="0" y="240" fill="#a0a5b0" font-family="Arial" font-size="14">Envolver a 70° perpendicular al eje del tubo mandibular</text>
  </g>

  <!-- Visera y Antena -->
  <g transform="translate(600, 650)">
    <text x="0" y="-30" fill="#a0a5b0" font-family="Arial" font-size="18" font-weight="bold">Visera de Comandante y Antena de Oreja</text>
    <path d="M 0 50 L 250 50 L 220 120 L 30 120 Z" fill="#32363a" stroke="#ffffff" stroke-width="2"/>
    <rect x="80" y="70" width="30" height="10" fill="#cdd2d7"/>
    <rect x="140" y="70" width="30" height="10" fill="#cdd2d7"/>
    <line x1="280" y1="20" x2="280" y2="200" stroke="#32363a" stroke-width="8" stroke-linecap="round"/>
    <text x="0" y="240" fill="#a0a5b0" font-family="Arial" font-size="14">Visera carbón con corchetes claros y antena vertical derecha</text>
  </g>
</svg>
"""

svg_path = os.path.join(OUT_03, "Stencil_Thorn_Coruscant_Wings.svg")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write(stencil_svg)
print(f"Stencil vectorial SVG actualizado en: {svg_path}")

print("=== PIPELINE COMPLETADO CON ÉXITO ABSOLUTO ===")
