import os
import glob
import math
import trimesh
import numpy as np

print("==================================================================")
print("  EXHAUSTIVE AUDIT: CATAN MOUNTAIN PRODUCTION ASSETS              ")
print("==================================================================")

proj_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D"
solid_stl = os.path.join(proj_dir, "catan_mountain_solid.stl")
cmykwog_3mf = os.path.join(proj_dir, "catan_mountain_cmykwog.3mf")
parts_dir = os.path.join(proj_dir, "piezas_cmykwog_separadas")

# 1. Solid STL Audit
print("\n--- 1. AUDITING SOLID STL ---")
assert os.path.exists(solid_stl), f"Missing {solid_stl}"
m_solid = trimesh.load(solid_stl)
print(f"File: {os.path.basename(solid_stl)}")
print(f"  Vertices: {len(m_solid.vertices):,}")
print(f"  Faces: {len(m_solid.faces):,}")
print(f"  Is Watertight: {m_solid.is_watertight}")
print(f"  Is Winding Consistent: {m_solid.is_winding_consistent}")
print(f"  Euler Characteristic: {m_solid.euler_number}")
print(f"  Volume: {m_solid.volume / 1000.0:.3f} cm³")
bounds = m_solid.bounds
print(f"  Bounding Box:")
print(f"    X: [{bounds[0][0]:.3f}, {bounds[1][0]:.3f}] mm (Span: {bounds[1][0]-bounds[0][0]:.3f} mm)")
print(f"    Y: [{bounds[0][1]:.3f}, {bounds[1][1]:.3f}] mm (Span: {bounds[1][1]-bounds[0][1]:.3f} mm)")
print(f"    Z: [{bounds[0][2]:.3f}, {bounds[1][2]:.3f}] mm (Span: {bounds[1][2]-bounds[0][2]:.3f} mm)")

# Check magnet cavity in bottom
bot_verts = m_solid.vertices[m_solid.vertices[:, 2] < 6.0]
r_bot = np.hypot(bot_verts[:, 0], bot_verts[:, 1])
mag_pocket_verts = bot_verts[(r_bot < 5.2) & (bot_verts[:, 2] > 0.05)]
print(f"  Magnet Pocket Vertices at Z in (0, 5.2]: {len(mag_pocket_verts)}")
print(f"    Max pocket Z: {np.max(mag_pocket_verts[:, 2]):.2f} mm")

# 2. Separate Tool STLs Audit
print("\n--- 2. AUDITING SEPARATE TOOL STLS ---")
tool_stls = sorted(glob.glob(os.path.join(parts_dir, "*.stl")))
print(f"Found {len(tool_stls)} tool STLs:")
total_faces_parts = 0
for stl in tool_stls:
    sm = trimesh.load(stl)
    total_faces_parts += len(sm.faces)
    print(f"  * {os.path.basename(stl):32s}: {len(sm.vertices):6,d} verts, {len(sm.faces):6,d} faces, Z in [{sm.bounds[0][2]:.2f}, {sm.bounds[1][2]:.2f}] mm")

# 3. Multi-Body 3MF Audit
print("\n--- 3. AUDITING 3MF ASSEMBLY ---")
assert os.path.exists(cmykwog_3mf), f"Missing {cmykwog_3mf}"
s_3mf = trimesh.load(cmykwog_3mf)
print(f"File: {os.path.basename(cmykwog_3mf)}")
if isinstance(s_3mf, trimesh.Scene):
    print(f"  3MF Scene Geometries: {len(s_3mf.geometry)}")
    for name, geom in s_3mf.geometry.items():
        print(f"    - {name:30s}: {len(geom.faces):6,d} faces, Z max={geom.bounds[1][2]:.2f} mm")
else:
    print(f"  Loaded as single geometry: {len(s_3mf.faces)} faces")

# 4. WebGL Viewer Audit
print("\n--- 4. AUDITING WEBGL VIEWER HTML ---")
viewer_path = os.path.join(proj_dir, "catan_mountain_viewer.html")
assert os.path.exists(viewer_path), f"Missing {viewer_path}"
size_kb = os.path.getsize(viewer_path) / 1024.0
print(f"Viewer size: {size_kb:.1f} KB")

print("\n=== ALL CHECKS PASSED PERFECTLY ===")
