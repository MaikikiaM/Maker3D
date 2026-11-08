import os
import math
import zipfile
import numpy as np
import trimesh

print("=== BUILDING OFFICIAL SPEC-COMPLIANT TEXTURED 3MF ARCHIVE ===")

out_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D"
mf3_path = os.path.join(out_dir, "catan_mountain_fullcolor.3mf")
tex_path = os.path.join(out_dir, "catan_mountain_diffuse_2k.png")

# Load solid mesh
solid_mesh = trimesh.load(os.path.join(out_dir, "catan_mountain_solid.stl"))
verts = solid_mesh.vertices
faces = solid_mesh.faces

R = 64.0
APOTHEM = R * math.cos(math.radians(30))

# UVs for vertices
u = np.clip((verts[:, 0] + APOTHEM) / (2 * APOTHEM), 0.0, 1.0)
v = np.clip((verts[:, 1] + R) / (2 * R), 0.0, 1.0)
# Side walls & bottom (z < 24.95) mapped to charcoal corner (0.01, 0.01)
is_base = verts[:, 2] < 24.95
u[is_base] = 0.01
v[is_base] = 0.01

print(f"Mesh: {len(verts):,} vertices, {len(faces):,} faces")

# 1. Generate 3D/3dmodel.model XML
print("Generating 3dmodel.model XML...")
lines = []
lines.append('<?xml version="1.0" encoding="UTF-8"?>')
lines.append('<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">')
lines.append('  <resources>')
lines.append('    <m:texture2d id="1" path="/3D/Textures/diffuse.png" contenttype="image/png" />')
lines.append('    <m:texture2dgroup id="2" texid="1">')
for i in range(len(verts)):
    lines.append(f'      <m:tex2coord u="{u[i]:.6f}" v="{v[i]:.6f}" />')
lines.append('    </m:texture2dgroup>')
lines.append('    <object id="3" type="model" pid="2">')
lines.append('      <mesh>')
lines.append('        <vertices>')
for pt in verts:
    lines.append(f'          <vertex x="{pt[0]:.4f}" y="{pt[1]:.4f}" z="{pt[2]:.4f}" />')
lines.append('        </vertices>')
lines.append('        <triangles>')
for tri in faces:
    v1, v2, v3 = tri[0], tri[1], tri[2]
    lines.append(f'          <triangle v1="{v1}" v2="{v2}" v3="{v3}" p1="{v1}" p2="{v2}" p3="{v3}" />')
lines.append('        </triangles>')
lines.append('      </mesh>')
lines.append('    </object>')
lines.append('  </resources>')
lines.append('  <build>')
lines.append('    <item objectid="3" />')
lines.append('  </build>')
lines.append('</model>')

xml_model = "\n".join(lines)
print(f"Model XML generated: {len(xml_model)/1024/1024:.2f} MB")

# 2. Relationship file 3D/_rels/3dmodel.model.rels
rels_model = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/Textures/diffuse.png" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/texture" />
</Relationships>'''

# 3. Root relationship file _rels/.rels
rels_root = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />
</Relationships>'''

# 4. [Content_Types].xml
content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />
  <Default Extension="png" ContentType="image/png" />
</Types>'''

# 5. Pack into 3MF Zip
print(f"Packing into 3MF Archive: {mf3_path}...")
with zipfile.ZipFile(mf3_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.writestr('[Content_Types].xml', content_types)
    z.writestr('_rels/.rels', rels_root)
    z.writestr('3D/_rels/3dmodel.model.rels', rels_model)
    z.writestr('3D/3dmodel.model', xml_model)
    # Add texture image
    with open(tex_path, 'rb') as f_tex:
        z.writestr('3D/Textures/diffuse.png', f_tex.read())

print(f"SUCCESS: Generated Textured 3MF ({os.path.getsize(mf3_path)/1024/1024:.2f} MB)")
