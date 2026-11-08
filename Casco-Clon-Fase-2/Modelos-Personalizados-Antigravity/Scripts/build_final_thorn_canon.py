import os
import zipfile
import json
import trimesh
from manifold3d import Manifold, Mesh, CrossSection, OpType
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
casco_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
casco_dir = os.path.join(casco_root, "Archivos-Base-Galactic-Armory")
greeb_dir = os.path.join(casco_dir, "Greeblies")
antigravity_dir = os.path.join(casco_root, "Modelos-Personalizados-Antigravity")
out_clones = os.path.join(antigravity_dir, "Cascos-Pintados-Clones")
out_stencils = os.path.join(antigravity_dir, "Stencils-Pintura-Clones")
out_visuals = os.path.join(antigravity_dir, "Guias-Visuales-y-Planos")

print("1. Loading geometry...")
mb_teeth = trimesh.load(os.path.join(casco_dir, "Phase 2 Main Body.stl"))
mouth = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
ear_l = trimesh.load(os.path.join(greeb_dir, "Ear Piece Left.stl"))
ear_r = trimesh.load(os.path.join(greeb_dir, "Ear Piece Right.stl"))
aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))

ears = trimesh.util.concatenate([ear_l, ear_r])
aerators = trimesh.util.concatenate([aer_l, aer_r])

mf_mb = Manifold(Mesh(vert_properties=mb_teeth.vertices.astype('float32'), tri_verts=mb_teeth.faces.astype('uint32')))

print("2. Generating smooth CAD cutters for Canon Commander Thorn...")

# Feature 1: Dorsal Fin / Crest stripe (22mm wide, runs along top ridge)
box_crest = Manifold.cube([22.0, 320.0, 160.0], center=True).translate([0.0, 95.0, 1885.0])
part_crest = mf_mb ^ box_crest

# Feature 2: Thor Wings on lateral dome (3 distinct feathers sweeping backward over each ear)
p1 = [
    [-15.0, 1845.0], [0.0, 1865.0], [20.0, 1890.0], [50.0, 1912.0], [80.0, 1922.0], [105.0, 1918.0], 
    [118.0, 1902.0], [108.0, 1888.0], [80.0, 1895.0], [50.0, 1885.0], [20.0, 1870.0], [0.0, 1850.0]
][::-1]

p2 = [
    [-10.0, 1835.0], [15.0, 1848.0], [45.0, 1862.0], [75.0, 1875.0], [100.0, 1880.0], [110.0, 1870.0],
    [98.0, 1858.0], [70.0, 1855.0], [40.0, 1845.0], [10.0, 1835.0]
][::-1]

p3 = [
    [-5.0, 1826.0], [20.0, 1832.0], [50.0, 1838.0], [75.0, 1845.0], [92.0, 1848.0], [85.0, 1838.0],
    [60.0, 1834.0], [30.0, 1828.0], [0.0, 1825.0]
][::-1]

cs_wings = CrossSection([p1, p2, p3])
mf_wing_ext = Manifold.extrude(cs_wings, height=160.0)
mat_right = [
    [0.0, 0.0, 1.0, 25.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]
]
mf_wing_cutter_r = mf_wing_ext.transform(mat_right)
wing_r = mf_mb ^ mf_wing_cutter_r
wing_l = wing_r.mirror([1.0, 0.0, 0.0])
part_wings = wing_r + wing_l

# Feature 3: Nose / Snout / Frown (Triangular lower central face between visor and mouth)
p_snout = [
    [-16.0, 1650.0], [16.0, 1650.0], [40.0, 1745.0], [-40.0, 1745.0]
]
cs_snout = CrossSection([p_snout])
mf_snout_ext = Manifold.extrude(cs_snout, height=120.0)
mat_snout = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, -50.0],
    [0.0, 1.0, 0.0, 0.0]
]
mf_snout_cutter = mf_snout_ext.transform(mat_snout)
part_snout = mf_mb ^ mf_snout_cutter

# Feature 4: Cheek scoops (Hendiduras cóncavas de las mejillas)
box_cheek_r = Manifold.cube([53.0, 65.0, 65.0], center=True).translate([71.5, -22.5, 1702.5])
cheek_r = mf_mb ^ box_cheek_r
cheek_l = cheek_r.mirror([1.0, 0.0, 0.0])
part_cheeks = cheek_r + cheek_l

# Combine all red Thorn markings
thorn_red_all = part_crest + part_wings + part_snout + part_cheeks
print(f"  Thorn Red markings volume: {thorn_red_all.volume():.1f} mm3, tris: {thorn_red_all.num_tri()}")

# Subtract to produce exact interlocking White Helmet base
helmet_white_base = mf_mb - thorn_red_all
print(f"  Helmet White base volume: {helmet_white_base.volume():.1f} mm3, tris: {helmet_white_base.num_tri()}")

# Convert to trimesh
mesh_thorn_red = trimesh.Trimesh(
    vertices=thorn_red_all.to_mesh().vert_properties,
    faces=thorn_red_all.to_mesh().tri_verts,
    process=True
)
mesh_helmet_white = trimesh.Trimesh(
    vertices=helmet_white_base.to_mesh().vert_properties,
    faces=helmet_white_base.to_mesh().tri_verts,
    process=True
)

# Colors
COLOR_WHITE = np.array([244, 244, 246, 255], dtype=np.uint8)
COLOR_RED = np.array([185, 28, 28, 255], dtype=np.uint8)
COLOR_BLACK = np.array([20, 20, 24, 255], dtype=np.uint8)
COLOR_GREY = np.array([120, 120, 125, 255], dtype=np.uint8)
COLOR_DARK_GREY = np.array([60, 60, 65, 255], dtype=np.uint8)

mesh_helmet_white.visual.vertex_colors = np.tile(COLOR_WHITE, (len(mesh_helmet_white.vertices), 1))
mesh_thorn_red.visual.vertex_colors = np.tile(COLOR_RED, (len(mesh_thorn_red.vertices), 1))
mouth.visual.vertex_colors = np.tile(COLOR_BLACK, (len(mouth.vertices), 1))
ears.visual.vertex_colors = np.tile(COLOR_GREY, (len(ears.vertices), 1))
aerators.visual.vertex_colors = np.tile(COLOR_DARK_GREY, (len(aerators.vertices), 1))

# 3. Export Full-Color GLB
print("3. Exporting Full-Color GLB...")
scene_glb = trimesh.Scene({
    "01_Casco_Base_Blanco": mesh_helmet_white,
    "02_Thorn_Patron_Rojo_Carmesí": mesh_thorn_red,
    "03_Vocoder_Menton_Negro": mouth,
    "04_Orejeras_Gris": ears,
    "05_Aireadores_Gris": aerators
})
glb_path = os.path.join(out_clones, "Clone_P2_Thorn_Guardia_Coruscant_Pintado.glb")
scene_glb.export(glb_path)
print(f"  Saved GLB: {glb_path} ({os.path.getsize(glb_path)/(1024*1024):.1f} MB)")

# 4. Export Bambu Studio 3MF Project
print("4. Exporting Bambu Studio 3MF Project...")
tmf_path = os.path.join(out_clones, "Clone_P2_Thorn_Guardia_Coruscant_Multicolor_Bambu_Prusa.3mf")

parts_dict = {
    "01_Casco_Base_Blanco": mesh_helmet_white,
    "02_Thorn_Patron_Rojo_Carmesí": mesh_thorn_red,
    "03_Vocoder_Menton_Negro": mouth,
    "04_Orejeras_Gris_Metalico": ears,
    "05_Aireadores_Gris_Oscuro": aerators
}
scene_3mf = trimesh.Scene(parts_dict)
scene_3mf.export(tmf_path)

# Inject Bambu Studio XML & config
with zipfile.ZipFile(tmf_path, "r") as z_in:
    file_dict = {name: z_in.read(name) for name in z_in.namelist()}

xml_data = file_dict["3D/3dmodel.model"].decode("utf-8")
import re
obj_matches = re.findall(r'<object[^>]*id="(\d+)"[^>]*name="([^"]+)"', xml_data)
obj_ids = [int(m[0]) for m in obj_matches]
assembly_id = max(obj_ids) + 1

components_xml = "".join([f'<component objectid="{oid}" />' for oid in obj_ids])
assembly_obj_xml = f'<object id="{assembly_id}" type="model" name="Casco_Comandante_Thorn"><components>{components_xml}</components></object>'
xml_data = xml_data.replace("</resources>", f"{assembly_obj_xml}</resources>")

build_pattern = r'<build[^>]*>.*?</build>'
new_build = f'<build><item objectid="{assembly_id}" /></build>'
xml_data = re.sub(build_pattern, new_build, xml_data, flags=re.DOTALL)
file_dict["3D/3dmodel.model"] = xml_data.encode("utf-8")

# Map parts to Extruders
# 1 = White (#F4F4F6)
# 2 = Red (#B91C1C)
# 3 = Black (#18181B)
# 4 = Grey (#71717A)
parts_config = []
for oid, name in obj_matches:
    lower = name.lower()
    if "blanco" in lower or "casco" in lower:
        ext = 1
    elif "rojo" in lower or "thorn" in lower:
        ext = 2
    elif "vocoder" in lower or "negro" in lower:
        ext = 3
    elif "orejera" in lower or "aireador" in lower or "gris" in lower:
        ext = 4
    else:
        ext = 1
    parts_config.append(f'    <part id="{oid}" subtype="normal_part"><metadata key="name" value="{name}"/><metadata key="extruder" value="{ext}"/></part>')

parts_str = "\n".join(parts_config)
model_settings = f"""<?xml version="1.0" encoding="UTF-8"?>
<config>
  <object id="{assembly_id}">
    <metadata key="name" value="Casco_Comandante_Thorn"/>
    <metadata key="extruder" value="1"/>
{parts_str}
  </object>
  <plate>
    <metadata key="plater_id" value="1"/>
    <metadata key="plater_name" value=""/>
    <metadata key="locked" value="false"/>
  </plate>
</config>"""

slice_info = """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <header>
    <header_item key="X-BBL-Client-Type" value="slicer"/>
    <header_item key="X-BBL-Client-Version" value="02.02.01.60"/>
  </header>
</config>"""

# Bambu Studio project settings with exact filament colors
project_settings = {
    "filament_colour": ["#F4F4F6", "#B91C1C", "#18181B", "#71717A"],
    "filament_type": ["PLA", "PLA", "PLA", "PLA"],
    "filament_vendor": ["Bambu", "Bambu", "Bambu", "Bambu"],
    "filament_names": ["1. Blanco Armadura", "2. Rojo Thorn", "3. Negro Vocoder", "4. Gris Mecánico"],
    "default_print_profile": "0.20mm Standard @BBL A1"
}

file_dict["Metadata/model_settings.config"] = model_settings.encode("utf-8")
file_dict["Metadata/slice_info.config"] = slice_info.encode("utf-8")
file_dict["Metadata/project_settings.config"] = json.dumps(project_settings, indent=2).encode("utf-8")

if "[Content_Types].xml" in file_dict:
    ct_xml = file_dict["[Content_Types].xml"].decode("utf-8")
    if 'Extension="config"' not in ct_xml:
        ct_xml = ct_xml.replace("</Types>", '<Default Extension="config" ContentType="application/xml"/></Types>')
        file_dict["[Content_Types].xml"] = ct_xml.encode("utf-8")

with zipfile.ZipFile(tmf_path, "w", compression=zipfile.ZIP_DEFLATED) as z_out:
    for name, data in file_dict.items():
        z_out.writestr(name, data)

print(f"  Saved 3MF: {tmf_path} ({os.path.getsize(tmf_path)/(1024*1024):.1f} MB)")

# 5. Render High-Resolution Multi-Angle Inspection Image
print("5. Rendering High-Resolution Inspection Sheet...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

vw = mesh_helmet_white.vertices[::3]
vr = mesh_thorn_red.vertices[::2]
vm = mouth.vertices[::2]
ve = ears.vertices[::2]
va = aerators.vertices[::2]

pts = np.vstack([vw, vr, vm, ve, va])
cols = np.vstack([
    np.tile([0.94, 0.94, 0.96, 1.0], (len(vw), 1)), # White
    np.tile([0.82, 0.10, 0.14, 1.0], (len(vr), 1)), # Crimson Red
    np.tile([0.10, 0.10, 0.12, 1.0], (len(vm), 1)), # Black
    np.tile([0.55, 0.55, 0.58, 1.0], (len(ve), 1)), # Grey
    np.tile([0.30, 0.30, 0.32, 1.0], (len(va), 1))  # Dark Grey
])

fig, axs = plt.subplots(2, 2, figsize=(16, 16), dpi=150)
fig.patch.set_facecolor('#0b0f19')

views = [
    ('Vista Frontal (0°)', 0, axs[0, 0]),
    ('Vista Perspectiva 3/4 (35°)', 35, axs[0, 1]),
    ('Vista Perfil Lateral (90°)', 90, axs[1, 0]),
    ('Vista Superior / Planta (Cúpula y Alas)', 'top', axs[1, 1])
]

for title, ang, ax in views:
    ax.set_facecolor('#030712')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, color='#f87171', fontsize=14, fontweight='bold', pad=12)
    
    if ang == 'top':
        depth = pts[:, 2]
        order = np.argsort(depth)
        ax.scatter(pts[order, 0], pts[order, 1], c=cols[order], s=0.9, edgecolors='none')
    else:
        rad = np.radians(ang)
        rot_x = pts[:, 0] * np.cos(rad) + pts[:, 1] * np.sin(rad)
        rot_depth = -pts[:, 0] * np.sin(rad) + pts[:, 1] * np.cos(rad)
        rot_y = pts[:, 2]
        order = np.argsort(rot_depth)
        ax.scatter(rot_x[order], rot_y[order], c=cols[order], s=0.9, edgecolors='none')

plt.tight_layout()
render_file = os.path.join(out_visuals, 'thorn_canon_4views_render.png')
plt.savefig(render_file, facecolor=fig.get_facecolor(), edgecolor='none')
print(f"  Saved render to: {render_file}")

# 6. Update SVG Stencil
print("6. Updating SVG Stencil...")
svg_thorn = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" width="1000" height="1000">
  <rect width="100%" height="100%" fill="#0b0f19"/>
  <text x="500" y="45" fill="#f87171" font-family="sans-serif" font-size="24" font-weight="bold" text-anchor="middle">COMMANDER THORN (CORUSCANT GUARD) - CANON CRL STENCIL</text>
  <text x="500" y="75" fill="#94a3b8" font-family="sans-serif" font-size="14" text-anchor="middle">Trazado canónico: 3 Plumas Thor en cúpula lateral, aleta dorsal, morro triangular y pómulos</text>
  
  <g fill="#b91c1c" stroke="#ef4444" stroke-width="2">
    <!-- Dorsal Fin Stripe -->
    <rect x="480" y="100" width="40" height="340" rx="6" />
    
    <!-- Thor Dome Wings (Left & Right) -->
    <!-- Right Wing (Viewer Right) -->
    <path d="M 600 240 C 650 240, 750 260, 840 310 L 820 330 C 740 290, 650 270, 600 270 Z" />
    <path d="M 590 285 C 640 285, 740 310, 810 350 L 790 370 C 730 335, 640 315, 590 315 Z" />
    <path d="M 580 330 C 630 330, 710 350, 770 385 L 750 405 C 700 375, 630 360, 580 360 Z" />
    
    <!-- Left Wing (Viewer Left - Mirrored) -->
    <path d="M 400 240 C 350 240, 250 260, 160 310 L 180 330 C 260 290, 350 270, 400 270 Z" />
    <path d="M 410 285 C 360 285, 260 310, 190 350 L 210 370 C 270 335, 360 315, 410 315 Z" />
    <path d="M 420 330 C 370 330, 290 350, 230 385 L 250 405 C 300 375, 370 360, 420 360 Z" />
    
    <!-- Nose / Snout (Triangular snout between visor and mouth) -->
    <polygon points="440,520 560,520 525,670 475,670" />
    
    <!-- Cheek Scoops -->
    <path d="M 390 540 C 330 550, 310 630, 330 690 C 365 675, 395 620, 405 560 Z" />
    <path d="M 610 540 C 670 550, 690 630, 670 690 C 635 675, 605 620, 595 560 Z" />
  </g>
</svg>'''

svg_path = os.path.join(out_stencils, "Stencil_Thorn_Coruscant_Wings.svg")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg_thorn)

print("Finished building all canon Commander Thorn files successfully!")
