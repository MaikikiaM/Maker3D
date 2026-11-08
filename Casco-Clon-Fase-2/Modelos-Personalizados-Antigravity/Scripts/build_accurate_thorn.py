import os
import zipfile
import json
import trimesh
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
casco_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
casco_dir = os.path.join(casco_root, "Archivos-Base-Galactic-Armory")
greeb_dir = os.path.join(casco_dir, "Greeblies")
antigravity_dir = os.path.join(casco_root, "Modelos-Personalizados-Antigravity")
out_clones = os.path.join(antigravity_dir, "Cascos-Pintados-Clones")
out_stencils = os.path.join(antigravity_dir, "Stencils-Pintura-Clones")

print("Loading geometry...")
mb_teeth = trimesh.load(os.path.join(casco_dir, "Phase 2 Main Body.stl"))
mouth = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
ear_l = trimesh.load(os.path.join(greeb_dir, "Ear Piece Left.stl"))
ear_r = trimesh.load(os.path.join(greeb_dir, "Ear Piece Right.stl"))
aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))

ears = trimesh.util.concatenate([ear_l, ear_r])
aerators = trimesh.util.concatenate([aer_l, aer_r])

v = mb_teeth.vertices
fc = mb_teeth.triangles_center
fn = mb_teeth.face_normals

fx, fy, fz = fc[:, 0], fc[:, 1], fc[:, 2]
vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

# --- Precise Face Masks for Commander Thorn ---
# 1. Dorsal fin / crest: central stripe along top ridge
mask_crest = (np.abs(fx) < 13) & (fz > 1825) & (fy > -35) & (fn[:, 2] > 0.05)

# 2. Thor Forehead Wings (Winged chevron inspired by Thor)
# Starts above the visor bridge (z=1805) and sweeps up to z=1875
dz_wing = fz - 1812
wing_outer = 12.0 + np.maximum(0.0, dz_wing) * 1.25
wing_inner = np.maximum(0.0, (dz_wing - 18) * 0.95)
mask_wings = (
    (fz >= 1810) & (fz <= 1875) &
    (np.abs(fx) >= wing_inner) & (np.abs(fx) <= wing_outer) &
    (fy < -35) & (fn[:, 1] < -0.15)
)
mask_bridge = (np.abs(fx) < 13) & (fz >= 1795) & (fz <= 1822) & (fy < -35)

# 3. Nose / Snout / Frown area (Central lower face between visor and chin)
snout_x_width = 16.0 + (fz - 1650) * 0.38
mask_snout = (
    (fz >= 1650) & (fz <= 1745) &
    (np.abs(fx) <= snout_x_width) &
    (fy < -55)
)

# 4. Concave Cheek Recesses (Pómulos)
mask_cheeks = (
    (fz >= 1665) & (fz <= 1738) &
    (np.abs(fx) >= 40) & (np.abs(fx) <= 96) &
    (fy >= -58) & (fy <= 15) &
    (fn[:, 1] < 0.25)
)

# 5. Aerator tubes / Jaw cylinders
mask_aerators = (
    (fz >= 1625) & (fz <= 1665) &
    (np.abs(fx) >= 26) & (np.abs(fx) <= 68) &
    (fy < -35)
)

thorn_red_mask = mask_crest | mask_wings | mask_bridge | mask_snout | mask_cheeks | mask_aerators
print(f"Thorn pattern triangles: {np.sum(thorn_red_mask):,}")

# --- Vertex Masks for Full-Color GLB ---
vz_wing = vz - 1812
v_wing_outer = 12.0 + np.maximum(0.0, vz_wing) * 1.25
v_wing_inner = np.maximum(0.0, (vz_wing - 18) * 0.95)
v_wings = (
    (vz >= 1810) & (vz <= 1875) &
    (np.abs(vx) >= v_wing_inner) & (np.abs(vx) <= v_wing_outer) &
    (vy < -35)
)
v_bridge = (np.abs(vx) < 13) & (vz >= 1795) & (vz <= 1822) & (vy < -35)
v_crest = (np.abs(vx) < 13) & (vz > 1825) & (vy > -35)
v_snout = (
    (vz >= 1650) & (vz <= 1745) &
    (np.abs(vx) <= (16.0 + (vz - 1650) * 0.38)) &
    (vy < -55)
)
v_cheeks = (
    (vz >= 1665) & (vz <= 1738) &
    (np.abs(vx) >= 40) & (np.abs(vx) <= 96) &
    (vy >= -58) & (vy <= 15)
)
v_aerators = (
    (vz >= 1625) & (vz <= 1665) &
    (np.abs(vx) >= 26) & (np.abs(vx) <= 68) &
    (vy < -35)
)

v_thorn_red = v_crest | v_wings | v_bridge | v_snout | v_cheeks | v_aerators

# 1. Export Accurate GLB
COLOR_WHITE = np.array([244, 244, 246, 255], dtype=np.uint8)
COLOR_BLACK = np.array([24, 24, 28, 255], dtype=np.uint8)
COLOR_GREY = np.array([120, 120, 125, 255], dtype=np.uint8)
COLOR_RED_THORN = np.array([185, 28, 28, 255], dtype=np.uint8)

v_colors = np.tile(COLOR_WHITE, (len(v), 1))
# Brow band black
brow_v = (vz >= 1802) & (vz <= 1822) & (vy < -25) & (np.abs(vx) < 95) & (~v_bridge)
v_colors[brow_v] = COLOR_BLACK
v_colors[v_thorn_red] = COLOR_RED_THORN

mb_thorn_glb = mb_teeth.copy()
mb_thorn_glb.visual.vertex_colors = v_colors

# Vocoder colored red with black center detail
mouth_thorn = mouth.copy()
mouth_thorn.visual.vertex_colors = np.tile(COLOR_BLACK, (len(mouth.vertices), 1))

scene_glb = trimesh.Scene({
    "Casco_Base_Thorn": mb_thorn_glb,
    "Vocoder_Menton": mouth_thorn,
    "Orejera_Izquierda": ear_l.copy(),
    "Orejera_Derecha": ear_r.copy(),
    "Aireador_Izquierdo": aer_l.copy(),
    "Aireador_Derecho": aer_r.copy()
})

glb_path = os.path.join(out_clones, "Clone_P2_Thorn_Guardia_Coruscant_Pintado.glb")
scene_glb.export(glb_path)
print(f"Exported Accurate Thorn GLB: {glb_path} ({os.path.getsize(glb_path)/(1024*1024):.1f} MB)")

# 2. Export Accurate 3MF with 3D Decal Bodies for Bambu Studio
sub_thorn = mb_teeth.submesh([thorn_red_mask], append=True)
v_sub = sub_thorn.vertices.copy()
vn_sub = sub_thorn.vertex_normals
v_out = v_sub + vn_sub * 0.35
v_in = v_sub - vn_sub * 0.20
n_v = len(v_sub)
all_v = np.vstack([v_out, v_in])
all_f = np.vstack([sub_thorn.faces, sub_thorn.faces[:, [0, 2, 1]] + n_v])
thorn_decal_3d = trimesh.Trimesh(vertices=all_v, faces=all_f, process=True)

# Brow band decal
sub_brow = mb_teeth.submesh([brow_v[:len(fc)] if len(brow_v) == len(fc) else (fz >= 1802) & (fz <= 1820) & (fy < -25) & (np.abs(fx) < 95) & (~mask_bridge)], append=True)
v_b = sub_brow.vertices.copy()
vn_b = sub_brow.vertex_normals
brow_decal_3d = trimesh.Trimesh(
    vertices=np.vstack([v_b + vn_b*0.25, v_b - vn_b*0.15]),
    faces=np.vstack([sub_brow.faces, sub_brow.faces[:, [0, 2, 1]] + len(v_b)]),
    process=True
)

tmf_path = os.path.join(out_clones, "Clone_P2_Thorn_Guardia_Coruscant_Multicolor_Bambu_Prusa.3mf")

# Scene for 3MF
parts_dict = {
    "01_Casco_Base_Blanco": mb_teeth.copy(),
    "02_Banda_Ceja_Negra": brow_decal_3d,
    "02_Vocoder_Menton_Negro": mouth.copy(),
    "03_Orejeras_Gris": ears.copy(),
    "03_Aireadores_Gris": aerators.copy(),
    "04_Thorn_Patron_Rojo_Carmesí": thorn_decal_3d
}

scene_3mf = trimesh.Scene(parts_dict)
scene_3mf.export(tmf_path)

# Inject Bambu Studio metadata
with zipfile.ZipFile(tmf_path, "r") as z_in:
    file_dict = {name: z_in.read(name) for name in z_in.namelist()}

xml_data = file_dict["3D/3dmodel.model"].decode("utf-8")
import re
obj_matches = re.findall(r'<object[^>]*id="(\d+)"[^>]*name="([^"]+)"', xml_data)
obj_ids = [int(m[0]) for m in obj_matches]
assembly_id = max(obj_ids) + 1

components_xml = "".join([f'<component objectid="{oid}" />' for oid in obj_ids])
assembly_obj_xml = f'<object id="{assembly_id}" type="model" name="Casco_Clon_Fase_2"><components>{components_xml}</components></object>'
xml_data = xml_data.replace("</resources>", f"{assembly_obj_xml}</resources>")

build_pattern = r'<build[^>]*>.*?</build>'
new_build = f'<build><item objectid="{assembly_id}" /></build>'
xml_data = re.sub(build_pattern, new_build, xml_data, flags=re.DOTALL)
file_dict["3D/3dmodel.model"] = xml_data.encode("utf-8")

parts_config = []
for oid, name in obj_matches:
    lower = name.lower()
    if "casco" in lower:
        ext = 1 # White
    elif "vocoder" in lower or "ceja" in lower or "negro" in lower:
        ext = 2 # Black
    elif "orejera" in lower or "aireador" in lower or "gris" in lower:
        ext = 3 # Grey
    elif "thorn" in lower or "rojo" in lower or "04_" in lower:
        ext = 4 # Red
    else:
        ext = 1
    parts_config.append(f'    <part id="{oid}" subtype="normal_part"><metadata key="name" value="{name}"/><metadata key="extruder" value="{ext}"/></part>')

parts_str = "\n".join(parts_config)
model_settings = f"""<?xml version="1.0" encoding="UTF-8"?>
<config>
  <object id="{assembly_id}">
    <metadata key="name" value="Casco_Clon_Fase_2"/>
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

project_settings = {
    "filament_colour": ["#FFFFFF", "#18181B", "#71717A", "#B91C1C"],
    "filament_type": ["PLA", "PLA", "PLA", "PLA"],
    "filament_vendor": ["Bambu", "Bambu", "Bambu", "Bambu"],
    "filament_names": ["Blanco Armadura", "Negro Vocoder", "Gris Mecánico", "Rojo Coruscant Guard"],
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

print(f"Exported Accurate Thorn 3MF: {tmf_path} ({os.path.getsize(tmf_path)/(1024*1024):.1f} MB)")

# 3. Update Stencil SVG for Thorn
svg_thorn_accurate = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" width="1000" height="1000">
  <rect width="100%" height="100%" fill="#0f172a"/>
  <text x="500" y="50" fill="#f87171" font-family="sans-serif" font-size="26" font-weight="bold" text-anchor="middle">COMMANDER THORN (CORUSCANT GUARD) - ACCURATE CANON STENCIL</text>
  <text x="500" y="80" fill="#94a3b8" font-family="sans-serif" font-size="14" text-anchor="middle">Inspirado en el motivo de alas de Thor (Dave Filoni - The Clone Wars Season 6)</text>
  
  <g fill="#b91c1c" stroke="#ef4444" stroke-width="2">
    <!-- Dorsal Fin Stripe -->
    <rect x="480" y="110" width="40" height="300" rx="8" />
    
    <!-- Thor Forehead Wings -->
    <!-- Left Wing -->
    <path d="M 480 380 L 460 360 L 320 280 L 340 330 L 430 385 L 350 360 L 370 400 L 450 415 L 480 415 Z" />
    <!-- Right Wing -->
    <path d="M 520 380 L 540 360 L 680 280 L 660 330 L 570 385 L 650 360 L 630 400 L 550 415 L 520 415 Z" />
    <!-- Center Nose / Bridge connector -->
    <polygon points="485,410 515,410 510,480 490,480" />
    
    <!-- Triangular Nose / Snout (Between visor and mouth) -->
    <polygon points="430,520 570,520 540,680 460,680" />
    
    <!-- Left Cheek Scoop -->
    <path d="M 380 540 C 330 550, 310 620, 330 680 C 360 670, 390 620, 400 560 Z" />
    <!-- Right Cheek Scoop -->
    <path d="M 620 540 C 670 550, 690 620, 670 680 C 640 670, 610 620, 600 560 Z" />
    
    <!-- Aerator Lower Cylinders -->
    <rect x="420" y="700" width="60" height="50" rx="6" />
    <rect x="520" y="700" width="60" height="50" rx="6" />
  </g>
  
  <text x="500" y="850" fill="#cbd5e1" font-family="sans-serif" font-size="16" text-anchor="middle">Esquema canónico: Alas en frente, cresta, morro triangular, pómulos hundidos y tubos de aeradores</text>
</svg>'''

with open(os.path.join(out_stencils, "Stencil_Thorn_Coruscant_Wings.svg"), "w", encoding="utf-8") as f:
    f.write(svg_thorn_accurate)

print("Accurate Thorn Stencil SVG updated!")
