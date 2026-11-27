import os
import math
import time
import zipfile
import shutil
import numpy as np
import trimesh

print("==================================================================")
print("  EXPORTING ALL DELIVERABLES FOR 5.11M WATERTIGHT MASTER SOLID    ")
print("==================================================================")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

R = 64.0
apothem = R * math.cos(math.radians(30))

# 1. Load the 5.11M watertight solid mesh
mesh_path = 'scratch/test_5m_bsm.stl'
solid_mesh = trimesh.load(mesh_path)
print(f"Loaded 5.11M Master Solid Mesh:")
print(f"  Vertices: {len(solid_mesh.vertices):,}")
print(f"  Faces: {len(solid_mesh.faces):,}")
print(f"  Watertight: {solid_mesh.is_watertight}")
print(f"  Euler: {solid_mesh.euler_number}")
print(f"  Volume: {solid_mesh.volume / 1000.0:.2f} cm3")

# Copy to final production path
out_stl = os.path.join(out_dir, "catan_mountain_solid.stl")
shutil.copy2(mesh_path, out_stl)
print(f"Exported Final STL: {out_stl} ({os.path.getsize(out_stl)/1024/1024:.2f} MB)")

# 2. Export 24-bit TrueColor PLY with continuous vertex colors
print("\n--- 2. Exporting 5.11M Full-Color PLY ---")
t_ply = time.time()
diffuse_4k = np.load('scratch/diffuse_4k_array.npy')
dim_tex = diffuse_4k.shape[0]

vx = solid_mesh.vertices[:, 0]
vy = solid_mesh.vertices[:, 1]
vz = solid_mesh.vertices[:, 2]

u_tex = np.clip((vx - (-apothem)) / (2.0 * apothem), 0.0, 1.0)
v_tex = np.clip((vy - (-R)) / (2.0 * R), 0.0, 1.0)

px = np.clip((u_tex * (dim_tex - 1)).astype(np.int32), 0, dim_tex - 1)
py = np.clip((v_tex * (dim_tex - 1)).astype(np.int32), 0, dim_tex - 1)

vert_rgb = (diffuse_4k[py, px] * 255.0).astype(np.uint8)
vert_rgb[vz < 24.5] = [35, 38, 42]

color_mesh = solid_mesh.copy()
color_mesh.visual.vertex_colors = np.column_stack([vert_rgb, np.full(len(vert_rgb), 255, dtype=np.uint8)])

out_ply = os.path.join(out_dir, "catan_mountain_fullcolor.ply")
color_mesh.export(out_ply)
print(f"Exported Final PLY: {out_ply} ({os.path.getsize(out_ply)/1024/1024:.2f} MB in {time.time()-t_ply:.2f}s)")

# 3. Export Wavefront OBJ + MTL
print("\n--- 3. Exporting 5.11M Wavefront OBJ + MTL ---")
out_obj = os.path.join(out_dir, "catan_mountain_fullcolor.obj")
out_mtl = os.path.join(out_dir, "catan_mountain.mtl")

with open(out_mtl, 'w') as f_mtl:
    f_mtl.write("# Catan Mountain Full-Color Material\n")
    f_mtl.write("newmtl MountainMaterial\n")
    f_mtl.write("Ka 1.000 1.000 1.000\n")
    f_mtl.write("Kd 1.000 1.000 1.000\n")
    f_mtl.write("Ks 0.080 0.080 0.080\n")
    f_mtl.write("Ns 15.0\n")
    f_mtl.write("map_Kd catan_mountain_diffuse_4k.png\n")
    f_mtl.write("norm catan_mountain_normal_4k.png\n")

print(f"Writing OBJ file with {len(solid_mesh.vertices):,} vertices and {len(solid_mesh.faces):,} faces...")
t_obj = time.time()
with open(out_obj, 'w') as f_obj:
    f_obj.write("# Catan 3D Mountain Full-Color Tile (5.11M Triangles)\n")
    f_obj.write("mtllib catan_mountain.mtl\n")
    f_obj.write("o CatanMountain\n")
    for v in solid_mesh.vertices:
        f_obj.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
    for u, v in zip(u_tex, v_tex):
        f_obj.write(f"vt {u:.6f} {v:.6f}\n")
    f_obj.write("usemtl MountainMaterial\n")
    f_obj.write("s 1\n")
    for face in solid_mesh.faces:
        f0, f1, f2 = face[0] + 1, face[1] + 1, face[2] + 1
        f_obj.write(f"f {f0}/{f0} {f1}/{f1} {f2}/{f2}\n")
print(f"Exported Final OBJ: {out_obj} ({os.path.getsize(out_obj)/1024/1024:.2f} MB in {time.time()-t_obj:.2f}s)")

# 4. Export Spec-Compliant 3MF Archive
print("\n--- 4. Packaging 5.11M Full-Color 3MF Archive ---")
t_3mf = time.time()
out_3mf = os.path.join(out_dir, "catan_mountain_fullcolor.3mf")
with open(os.path.join(out_dir, "catan_mountain_diffuse_4k.png"), "rb") as f_png:
    png_bytes = f_png.read()

model_xml_path = "scratch/3dmodel.model"
with open(model_xml_path, 'w', encoding='utf-8') as f_m:
    f_m.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f_m.write('<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">\n')
    f_m.write('  <resources>\n')
    f_m.write('    <m:texture2d id="1" path="/3D/Textures/catan_mountain_diffuse_4k.png" contenttype="image/png" />\n')
    f_m.write('    <m:texture2dgroup id="2" texid="1">\n')
    for u, v in zip(u_tex, v_tex):
        f_m.write(f'      <m:tex2coord u="{u:.5f}" v="{v:.5f}" />\n')
    f_m.write('    </m:texture2dgroup>\n')
    f_m.write('    <object id="3" type="model">\n')
    f_m.write('      <mesh>\n')
    f_m.write('        <vertices>\n')
    for v in solid_mesh.vertices:
        f_m.write(f'          <vertex x="{v[0]:.3f}" y="{v[1]:.3f}" z="{v[2]:.3f}" />\n')
    f_m.write('        </vertices>\n')
    f_m.write('        <triangles>\n')
    for face in solid_mesh.faces:
        f0, f1, f2 = face[0], face[1], face[2]
        f_m.write(f'          <triangle v1="{f0}" v2="{f1}" v3="{f2}" pid="2" p1="{f0}" p2="{f1}" p3="{f2}" />\n')
    f_m.write('        </triangles>\n')
    f_m.write('      </mesh>\n')
    f_m.write('    </object>\n')
    f_m.write('  </resources>\n')
    f_m.write('  <build>\n')
    f_m.write('    <item objectid="3" />\n')
    f_m.write('  </build>\n')
    f_m.write('</model>\n')

with zipfile.ZipFile(out_3mf, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    z.write(model_xml_path, arcname="3D/3dmodel.model")
    z.writestr("3D/Textures/catan_mountain_diffuse_4k.png", png_bytes)
    z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/><Default Extension="png" ContentType="image/png"/></Types>')
    z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')

print(f"Exported Final 3MF: {out_3mf} ({os.path.getsize(out_3mf)/1024/1024:.2f} MB in {time.time()-t_3mf:.2f}s)")
os.remove(model_xml_path)

print(f"\nAll 5.11M Deliverables Exported in {time.time()-t0:.2f}s!")
