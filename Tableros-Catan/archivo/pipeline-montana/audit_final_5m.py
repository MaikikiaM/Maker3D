import os
import trimesh
from PIL import Image

out_dir = r"Tableros Catan\Loseta_Montana_3D"

print("==================================================================")
print("             MASTER PRODUCTION DELIVERABLES AUDIT                 ")
print("==================================================================")

stl_path = os.path.join(out_dir, "catan_mountain_solid.stl")
ply_path = os.path.join(out_dir, "catan_mountain_fullcolor.ply")
obj_path = os.path.join(out_dir, "catan_mountain_fullcolor.obj")
mf_path  = os.path.join(out_dir, "catan_mountain_fullcolor.3mf")
html_path = os.path.join(out_dir, "catan_mountain_viewer.html")
diff_4k  = os.path.join(out_dir, "catan_mountain_diffuse_4k.png")
diff_2k  = os.path.join(out_dir, "catan_mountain_diffuse_2k.png")
norm_4k  = os.path.join(out_dir, "catan_mountain_normal_4k.png")

print(f"1. STL Solid:        {os.path.getsize(stl_path)/1024/1024:.2f} MB")
print(f"2. PLY TrueColor:    {os.path.getsize(ply_path)/1024/1024:.2f} MB")
print(f"3. OBJ Wavefront:    {os.path.getsize(obj_path)/1024/1024:.2f} MB")
print(f"4. 3MF Archive:      {os.path.getsize(mf_path)/1024/1024:.2f} MB")
print(f"5. WebGL Viewer:     {os.path.getsize(html_path)/1024/1024:.2f} MB")
print(f"6. 4K Diffuse PNG:   {os.path.getsize(diff_4k)/1024/1024:.2f} MB (Dims: {Image.open(diff_4k).size})")
print(f"7. 2K Diffuse PNG:   {os.path.getsize(diff_2k)/1024/1024:.2f} MB (Dims: {Image.open(diff_2k).size})")
print(f"8. 4K Normal PNG:    {os.path.getsize(norm_4k)/1024/1024:.2f} MB (Dims: {Image.open(norm_4k).size})")

mesh = trimesh.load(stl_path)
print(f"\nMesh Audit:")
print(f"  Watertight Manifold: {mesh.is_watertight}")
print(f"  Euler Characteristic: {mesh.euler_number}")
print(f"  Vertices: {len(mesh.vertices):,}")
print(f"  Faces: {len(mesh.faces):,}")
print(f"  Volume: {mesh.volume / 1000.0:.2f} cm3")
print(f"  Bounds: min={mesh.bounds[0].round(2)}, max={mesh.bounds[1].round(2)}")

assert mesh.is_watertight, "FAIL: Mesh is not watertight!"
assert mesh.euler_number == 2, "FAIL: Euler number != 2!"
assert len(mesh.faces) >= 5000000, "FAIL: Faces < 5,000,000!"
print("\n>>> ALL CRITERIA STRICTLY PASSED! <<<")
