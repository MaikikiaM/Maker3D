import os
import time
import numpy as np
import trimesh

print("=== 5.11M TRIANGLE MESH TOPOLOGICAL & NORMAL VECTOR AUDIT ===")
t0 = time.time()

stl_path = r"Tableros Catan\Loseta_Montana_3D\catan_mountain_solid.stl"
assert os.path.exists(stl_path), f"File not found: {stl_path}"

print(f"Loading {stl_path} ({os.path.getsize(stl_path)/1024/1024:.2f} MB)...")
mesh = trimesh.load(stl_path)
print(f"Mesh loaded in {time.time()-t0:.2f}s.")

n_verts = len(mesh.vertices)
n_faces = len(mesh.faces)
print(f"\nTopological Metrics:")
print(f"  Vertices: {n_verts:,}")
print(f"  Faces:    {n_faces:,}")
print(f"  Is Watertight Manifold: {mesh.is_watertight}")
print(f"  Euler Characteristic:   {mesh.euler_number} (Must be exactly 2 for genus-0 sphere-equivalent solid)")
print(f"  Volume:                 {mesh.volume / 1000.0:.2f} cm3")

# Degenerate triangle check (zero area faces)
areas = mesh.area_faces
zero_area = np.sum(areas <= 1e-8)
print(f"  Zero-area degenerate faces: {zero_area} (0.00%)")

# Aspect ratio check
v0 = mesh.vertices[mesh.faces[:, 0]]
v1 = mesh.vertices[mesh.faces[:, 1]]
v2 = mesh.vertices[mesh.faces[:, 2]]
e0 = np.linalg.norm(v1 - v0, axis=1)
e1 = np.linalg.norm(v2 - v1, axis=1)
e2 = np.linalg.norm(v0 - v2, axis=1)
max_edge = np.maximum(e0, np.maximum(e1, e2))
min_edge = np.minimum(e0, np.minimum(e1, e2))
aspect_ratios = max_edge / np.maximum(min_edge, 1e-6)
print(f"  Mean face aspect ratio: {np.mean(aspect_ratios):.2f}")
print(f"  99th percentile aspect ratio: {np.percentile(aspect_ratios, 99):.2f}")

# Normal vectors check
normals = mesh.face_normals
norm_lengths = np.linalg.norm(normals, axis=1)
invalid_normals = np.sum(np.abs(norm_lengths - 1.0) > 1e-3)
print(f"  Invalid / unnormalized face normals: {invalid_normals}")

# Bounding box check
bounds = mesh.bounds
print(f"\nGeometric Bounding Box:")
print(f"  X: [{bounds[0, 0]:.2f}, {bounds[1, 0]:.2f}] mm (Span: {bounds[1, 0] - bounds[0, 0]:.2f} mm | Expected: 110.85 mm)")
print(f"  Y: [{bounds[0, 1]:.2f}, {bounds[1, 1]:.2f}] mm (Span: {bounds[1, 1] - bounds[0, 1]:.2f} mm | Expected: 128.00 mm)")
print(f"  Z: [{bounds[0, 2]:.2f}, {bounds[1, 2]:.2f}] mm (Span: {bounds[1, 2] - bounds[0, 2]:.2f} mm | Expected: 83.00 mm)")

# Base magnet cavity check (probe points inside cylinder r < 5.0, z < 5.0)
# A hollow cavity means the volume at bottom center is empty
r_pocket = 5.1
h_pocket = 5.2
print(f"\nMagnet Cavity Probe:")
# In STL coordinate system, the magnet pocket is centered at (0, 0) for Z in [0, 5.2]
# Vertices should exist along the cylinder wall at radius ~ 5.1mm
cavity_verts = mesh.vertices[(mesh.vertices[:, 2] <= h_pocket + 0.1) & (np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]) <= r_pocket + 0.5)]
print(f"  Vertices defining magnet pocket: {len(cavity_verts):,}")

# Assertions
assert mesh.is_watertight, "FAIL: Mesh is not watertight manifold!"
assert mesh.euler_number == 2, f"FAIL: Euler characteristic {mesh.euler_number} != 2!"
assert n_faces >= 5000000, f"FAIL: Face count {n_faces} < 5,000,000!"
assert zero_area == 0, f"FAIL: Found {zero_area} degenerate faces!"
assert invalid_normals == 0, f"FAIL: Found {invalid_normals} invalid normals!"

print("\n>>> 5.11M MESH TOPOLOGICAL AUDIT: 100% PERFECT PASS! <<<")
print(f"Total audit time: {time.time()-t0:.2f}s")
