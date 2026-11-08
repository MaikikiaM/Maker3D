import os
import zipfile
import json
import trimesh
import numpy as np

casco_dir = r"C:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco"
greeb_dir = os.path.join(casco_dir, "Greeblies")
antigravity_dir = os.path.join(casco_dir, "Antigravity")
out_clones = os.path.join(antigravity_dir, "05_Cascos_Pintados_Clones")
out_editable = os.path.join(antigravity_dir, "02_Proyecto_Editable_MultiColor")

print("Loading base geometry...")
mb_teeth = trimesh.load(os.path.join(casco_dir, "Phase 2 Main Body.stl"))
mouth = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
ear_l = trimesh.load(os.path.join(greeb_dir, "Ear Piece Left.stl"))
ear_r = trimesh.load(os.path.join(greeb_dir, "Ear Piece Right.stl"))
aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))

ears = trimesh.util.concatenate([ear_l, ear_r])
aerators = trimesh.util.concatenate([aer_l, aer_r])

fc = mb_teeth.triangles_center
fn = mb_teeth.face_normals
fx, fy, fz = fc[:, 0], fc[:, 1], fc[:, 2]

# Common brow band (black rubber strip)
brow_mask = (fz >= 1802) & (fz <= 1822) & (fy < -25) & (np.abs(fx) < 95)

def make_decal_body(mesh, mask, offset_out=0.35, offset_in=0.25):
    sub = mesh.submesh([mask], append=True)
    v = sub.vertices.copy()
    vn = sub.vertex_normals
    v_out = v + vn * offset_out
    v_in = v - vn * offset_in
    n_v = len(v)
    all_v = np.vstack([v_out, v_in])
    all_f = np.vstack([sub.faces, sub.faces[:, [0, 2, 1]] + n_v])
    return trimesh.Trimesh(vertices=all_v, faces=all_f, process=True)

# Brow band decal body
brow_decal = make_decal_body(mb_teeth, brow_mask, offset_out=0.25, offset_in=0.2)

def build_bambu_3mf(filepath, parts_dict, filament_colors, filament_names):
    """
    Exports a scene to 3MF and injects Bambu Studio native project_settings and model_settings
    """
    scene = trimesh.Scene(parts_dict)
    scene.export(filepath)
    
    # Read back zip
    with zipfile.ZipFile(filepath, "r") as z_in:
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
    
    # Map parts to extruders
    parts_config = []
    for oid, name in obj_matches:
        lower = name.lower()
        if "casco" in lower:
            ext = 1 # White
        elif "vocoder" in lower or "ceja" in lower or "negro" in lower:
            ext = 2 # Black
        elif "orejera" in lower or "aireador" in lower or "gris" in lower:
            ext = 3 # Grey
        elif "color" in lower or "patron" in lower or "alas" in lower or "cresta" in lower or "jaig" in lower or "mejilla" in lower or "anguila" in lower or "chevron" in lower or "llamas" in lower:
            ext = 4 # Color (Red/Blue)
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

    # Project settings with custom filament colours
    project_settings = {
        "filament_colour": filament_colors,
        "filament_type": ["PLA"] * len(filament_colors),
        "filament_vendor": ["Bambu"] * len(filament_colors),
        "filament_names": filament_names,
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

    with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as z_out:
        for name, data in file_dict.items():
            z_out.writestr(name, data)
    print(f"  Exported Bambu 3MF: {os.path.basename(filepath)} ({os.path.getsize(filepath)/(1024*1024):.1f} MB)")

# 1. Base Editable Project
print("\nRebuilding Base Editable 3MF...")
build_bambu_3mf(
    os.path.join(out_editable, "Clone_P2_Editable_Bambu_Prusa.3mf"),
    {
        "01_Casco_Base_Blanco": mb_teeth.copy(),
        "02_Banda_Ceja_Negra": brow_decal.copy(),
        "02_Vocoder_Menton_Negro": mouth.copy(),
        "03_Orejeras_Gris": ears.copy(),
        "03_Aireadores_Gris": aerators.copy()
    },
    ["#FFFFFF", "#18181B", "#71717A", "#3B82F6"],
    ["Blanco Armadura", "Negro Vocoder", "Gris Mecánico", "Color Acento"]
)

# 2. Commander Thorn
print("\nRebuilding Thorn 3MF...")
thorn_crest_mask = (np.abs(fx) < 14) & (fz > 1815) & (fy > -40) & (fn[:, 2] > 0.05)
thorn_wings_mask = (fz >= 1820) & (fz <= 1895) & (np.abs(fx) >= 35) & (np.abs(fx) <= 105) & (fy >= -50) & (fy <= 60) & (fn[:, 2] > 0.1)
thorn_brow_mask = (fz >= 1802) & (fz <= 1822) & (np.abs(fx) < 85) & (fy < -25)

thorn_patron = make_decal_body(mb_teeth, thorn_crest_mask | thorn_wings_mask | thorn_brow_mask, offset_out=0.35)

build_bambu_3mf(
    os.path.join(out_clones, "Clone_P2_Thorn_Guardia_Coruscant_Multicolor_Bambu_Prusa.3mf"),
    {
        "01_Casco_Base_Blanco": mb_teeth.copy(),
        "02_Vocoder_Menton_Negro": mouth.copy(),
        "03_Orejeras_Gris": ears.copy(),
        "03_Aireadores_Gris": aerators.copy(),
        "04_Patron_Thorn_Rojo_Carmesí": thorn_patron
    },
    ["#FFFFFF", "#18181B", "#71717A", "#B91C1C"],
    ["Blanco Armadura", "Negro Vocoder", "Gris Mecánico", "Rojo Coruscant Guard"]
)

# 3. Captain Rex
print("\nRebuilding Rex 3MF...")
rex_jaig = (fz >= 1822) & (fz <= 1865) & (np.abs(fx) >= 20) & (np.abs(fx) <= 65) & (fy < -40)
rex_cheeks = (fz >= 1675) & (fz <= 1735) & (np.abs(fx) >= 45) & (np.abs(fx) <= 95) & (fy >= -55) & (fy <= 20)
rex_patron = make_decal_body(mb_teeth, rex_jaig | rex_cheeks, offset_out=0.35)

build_bambu_3mf(
    os.path.join(out_clones, "Clone_P2_Capitan_Rex_501_Multicolor_Bambu_Prusa.3mf"),
    {
        "01_Casco_Base_Blanco": mb_teeth.copy(),
        "02_Banda_Ceja_Negra": brow_decal.copy(),
        "02_Vocoder_Menton_Negro": mouth.copy(),
        "03_Orejeras_Gris": ears.copy(),
        "03_Aireadores_Gris": aerators.copy(),
        "04_Patron_Rex_Azul_501": rex_patron
    },
    ["#FFFFFF", "#18181B", "#71717A", "#0284C7"],
    ["Blanco Armadura", "Negro Vocoder", "Gris Mecánico", "Azul Legión 501"]
)

# 4. Fives
print("\nRebuilding Fives 3MF...")
fives_eel = (fz >= 1770) & (fz <= 1940) & (np.abs(fx) <= (14 + (fz - 1770)*0.06)) & (fy >= -55) & (fy <= 120) & (fn[:, 2] > 0.05)
fives_chevrons = (fz >= 1680) & (fz <= 1750) & (np.abs(fx) >= 40) & (np.abs(fx) <= 100) & (fy >= -50) & (fy <= 30) & ((fz - np.abs(fx)*0.6) % 25 < 12)
fives_patron = make_decal_body(mb_teeth, fives_eel | fives_chevrons, offset_out=0.35)

build_bambu_3mf(
    os.path.join(out_clones, "Clone_P2_Fives_ARC_5555_Multicolor_Bambu_Prusa.3mf"),
    {
        "01_Casco_Base_Blanco": mb_teeth.copy(),
        "02_Banda_Ceja_Negra": brow_decal.copy(),
        "02_Vocoder_Menton_Negro": mouth.copy(),
        "03_Orejeras_Gris": ears.copy(),
        "03_Aireadores_Gris": aerators.copy(),
        "04_Patron_Fives_Anguila_Rishi": fives_patron
    },
    ["#FFFFFF", "#18181B", "#71717A", "#1D4ED8"],
    ["Blanco Armadura", "Negro Vocoder", "Gris Mecánico", "Azul ARC Fives"]
)

# 5. Hardcase
print("\nRebuilding Hardcase 3MF...")
hardcase_flames = (fz >= 1640) & (fz <= 1790) & (np.abs(fx) >= 20) & (np.abs(fx) <= 110) & (fy < 30) & ((np.sin(fz*0.08)*22 + np.abs(fx)) > 52)
hardcase_patron = make_decal_body(mb_teeth, hardcase_flames, offset_out=0.35)

build_bambu_3mf(
    os.path.join(out_clones, "Clone_P2_Hardcase_501_Multicolor_Bambu_Prusa.3mf"),
    {
        "01_Casco_Base_Blanco": mb_teeth.copy(),
        "02_Banda_Ceja_Negra": brow_decal.copy(),
        "02_Vocoder_Menton_Negro": mouth.copy(),
        "03_Orejeras_Gris": ears.copy(),
        "03_Aireadores_Gris": aerators.copy(),
        "04_Patron_Hardcase_Llamas_501": hardcase_patron
    },
    ["#FFFFFF", "#18181B", "#71717A", "#0284C7"],
    ["Blanco Armadura", "Negro Vocoder", "Gris Mecánico", "Azul Llamas 501"]
)

print("\nAll 3MF files have been rebuilt with native Bambu filament palettes and 3D visible decal bodies!")
