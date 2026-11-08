import os
import trimesh
import numpy as np
from PIL import Image

print("=== FINAL PRODUCTION AUDIT: 2M+ TRIANGLE ALPINE MASTERPIECE ===")

out_dir = r"Tableros Catan\Loseta_Montana_3D"
stl_file = os.path.join(out_dir, "catan_mountain_solid.stl")
ply_file = os.path.join(out_dir, "catan_mountain_fullcolor.ply")
obj_file = os.path.join(out_dir, "catan_mountain_fullcolor.obj")
mf_file  = os.path.join(out_dir, "catan_mountain_fullcolor.3mf")
png_4k   = os.path.join(out_dir, "catan_mountain_diffuse_4k.png")
norm_4k  = os.path.join(out_dir, "catan_mountain_normal_4k.png")
html_file = os.path.join(out_dir, "catan_mountain_viewer.html")

m = trimesh.load(stl_file)

print(f"STL File: {stl_file} ({os.path.getsize(stl_file)/1024/1024:.2f} MB)")
print(f"  Total Vertices: {len(m.vertices):,}")
print(f"  Total Faces: {len(m.faces):,} (Criteria >= 2,000,000: {'PASSED' if len(m.faces)>=2000000 else 'FAILED'})")
print(f"  Watertight Manifold: {m.is_watertight} (Criteria True: {'PASSED' if m.is_watertight else 'FAILED'})")
print(f"  Euler Characteristic: {m.euler_number} (Criteria 2: {'PASSED' if m.euler_number==2 else 'FAILED'})")
print(f"  Volume: {m.volume / 1000.0:.2f} cm3")
print(f"  Bounding Box Min: {np.round(m.bounds[0], 2)}")
print(f"  Bounding Box Max: {np.round(m.bounds[1], 2)}")

print(f"\nDeliverable Sizes:")
print(f"  PLY: {os.path.getsize(ply_file)/1024/1024:.2f} MB")
print(f"  OBJ: {os.path.getsize(obj_file)/1024/1024:.2f} MB")
print(f"  3MF: {os.path.getsize(mf_file)/1024/1024:.2f} MB")
print(f"  4K Diffuse: {os.path.getsize(png_4k)/1024/1024:.2f} MB")
print(f"  4K Normal: {os.path.getsize(norm_4k)/1024/1024:.2f} MB")
print(f"  HTML Viewer: {os.path.getsize(html_file)/1024/1024:.2f} MB")
