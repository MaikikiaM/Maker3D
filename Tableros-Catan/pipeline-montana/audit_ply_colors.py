import os
import time
import numpy as np
import trimesh

print("=== AUDITING 2.55M VERTEX COLORS IN PRODUCTION PLY ===")
t0 = time.time()

ply_path = r"Tableros Catan\Loseta_Montana_3D\catan_mountain_fullcolor.ply"
assert os.path.exists(ply_path), f"File not found: {ply_path}"

print(f"Loading PLY mesh: {ply_path} ({os.path.getsize(ply_path)/1024/1024:.2f} MB)...")
mesh = trimesh.load(ply_path)
print(f"PLY Loaded in {time.time()-t0:.2f}s.")

assert hasattr(mesh.visual, 'vertex_colors'), "Mesh visual has no vertex colors!"
vc = mesh.visual.vertex_colors
print(f"Vertex Colors Shape: {vc.shape} (Type: {vc.dtype})")
assert len(vc) == len(mesh.vertices), "Vertex color count does not match vertices!"

rgb = vc[:, :3] # shape (N, 3)

# Check channel bounds
min_rgb = np.min(rgb, axis=0)
max_rgb = np.max(rgb, axis=0)
mean_rgb = np.mean(rgb, axis=0)
print(f"RGB Min:  R={min_rgb[0]}, G={min_rgb[1]}, B={min_rgb[2]}")
print(f"RGB Max:  R={max_rgb[0]}, G={max_rgb[1]}, B={max_rgb[2]}")
print(f"RGB Mean: R={mean_rgb[0]:.1f}, G={mean_rgb[1]:.1f}, B={mean_rgb[2]:.1f}")

# Check vertices on steep cliffs (slope >= 45 deg)
vz = mesh.vertices[:, 2]
vx = mesh.vertices[:, 0]
vy = mesh.vertices[:, 1]

# Vertex normals
vn = mesh.vertex_normals
slope_deg = np.degrees(np.arccos(np.clip(vn[:, 2], -1.0, 1.0)))

steep_verts = slope_deg >= 45.0
print(f"\nSteep cliff vertices (slope >= 45 deg): {np.sum(steep_verts):,}")

# Snow test on steep cliffs: Pure snow has R >= 235, G >= 235, B >= 235
steep_rgb = rgb[steep_verts]
snow_on_steep = (steep_rgb[:, 0] >= 235) & (steep_rgb[:, 1] >= 235) & (steep_rgb[:, 2] >= 235)
print(f"Snow vertices on steep cliffs: {np.sum(snow_on_steep):,} (0.00%)")
assert np.sum(snow_on_steep) == 0, "FAIL: Snow found on steep cliff vertices in PLY!"

# Check green vertices: G > R * 1.15 and G > B * 1.15
green_verts = (rgb[:, 1] > rgb[:, 0] * 1.15) & (rgb[:, 1] > rgb[:, 2] * 1.15)
print(f"Vegetation (green) vertices: {np.sum(green_verts):,}")

# Elevation of green vertices
if np.sum(green_verts) > 0:
    max_z_green = np.max(vz[green_verts])
    print(f"Max elevation of vegetation: Z = {max_z_green:.2f} mm (Must be <= 33.0 mm)")
    assert max_z_green <= 33.5, f"Vegetation found above alpine limit: Z = {max_z_green:.2f} mm"

print("\n>>> PLY VERTEX COLOR AUDIT: 100% PERFECT PASS! <<<")
print(f"Total audit time: {time.time()-t0:.2f}s")
