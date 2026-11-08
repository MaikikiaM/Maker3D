import os
import zipfile
import re
import time

print("=== STREAM-PARSING & VALIDATING ALL 5.11M TRIANGLES IN 3MF MODEL XML ===")
t0 = time.time()

mf_path = r"Tableros Catan\Loseta_Montana_3D\catan_mountain_fullcolor.3mf"
assert os.path.exists(mf_path), f"File not found: {mf_path}"

with zipfile.ZipFile(mf_path, 'r') as z:
    with z.open("3D/3dmodel.model") as f:
        # Stream read line by line or buffer by buffer to avoid loading 773 MB into RAM at once
        vertex_count = 0
        triangle_count = 0
        texcoord_count = 0
        
        # Stream read line-by-line using io.TextIOWrapper
        import io
        f_text = io.TextIOWrapper(f, encoding='utf-8')
        for line in f_text:
            if '<vertex ' in line:
                vertex_count += 1
            elif '<triangle ' in line:
                triangle_count += 1
            elif '<m:tex2coord ' in line:
                texcoord_count += 1
                
print(f"Total Vertices in 3MF:     {vertex_count:,} (Expected: 2,555,530)")
print(f"Total UV Coordinates:      {texcoord_count:,} (Expected: 2,555,530)")
print(f"Total Triangles in 3MF:    {triangle_count:,} (Expected: 5,111,056)")

assert vertex_count == 2555530, f"Vertex count mismatch: {vertex_count}"
assert triangle_count == 5111056, f"Triangle count mismatch: {triangle_count}"
assert texcoord_count == 2555530, f"Texcoord count mismatch: {texcoord_count}"

print("\n>>> 3MF MODEL XML STREAM AUDIT: 100% PERFECT MATCH! <<<")
print(f"Execution time: {time.time()-t0:.2f}s")
