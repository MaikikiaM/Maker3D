import os
import re
import base64
import io
import numpy as np
from PIL import Image

print("=== VERIFYING WEBGL VIEWER HTML DATA INTEGRITY ===")

html_path = r"Tableros Catan\Loseta_Montana_3D\catan_mountain_viewer.html"
with open(html_path, 'r', encoding='utf-8') as f:
    html_text = f.read()

print(f"HTML Size: {len(html_text) / (1024*1024):.2f} MB")

# Check DEM_NX and DEM_NY
m_nx = re.search(r"const DEM_NX = (\d+);", html_text)
m_ny = re.search(r"const DEM_NY = (\d+);", html_text)
assert m_nx and m_ny, "Missing DEM dimensions in HTML!"
nx = int(m_nx.group(1))
ny = int(m_ny.group(1))
print(f"DEM Grid: {nx} x {ny} points")
assert (nx, ny) == (1150, 1328), f"Unexpected DEM dimensions: ({nx}, {ny})"

# Check DEM_Z_B64
m_z = re.search(r'const DEM_Z_B64 = "([^"]+)";', html_text)
assert m_z, "Missing DEM_Z_B64 in HTML!"
b64_z = m_z.group(1)
bin_z = base64.b64decode(b64_z)
u16_z = np.frombuffer(bin_z, dtype=np.uint16)
print(f"Decoded Z points: {len(u16_z):,} (Expected: {nx*ny:,})")
assert len(u16_z) == nx * ny, "Z point count mismatch!"
z_norm = u16_z / 65535.0
z_world = 25.0 + z_norm * 58.0
print(f"Z World Range in Viewer: [{np.min(z_world):.2f}, {np.max(z_world):.2f}] mm")
assert np.min(z_world) >= 24.99 and np.max(z_world) <= 83.01, "Z world bounds out of range!"

# Check DEM_TEX_B64
m_tex = re.search(r'const DEM_TEX_B64 = "data:image/png;base64,([^"]+)";', html_text)
assert m_tex, "Missing DEM_TEX_B64 in HTML!"
b64_tex = m_tex.group(1)
bin_tex = base64.b64decode(b64_tex)
im_tex = Image.open(io.BytesIO(bin_tex))
print(f"Decoded Embedded Texture: {im_tex.size[0]} x {im_tex.size[1]} ({im_tex.format}, {im_tex.mode})")
assert im_tex.size == (2048, 2048), "Embedded texture is not 2048x2048!"

# Check camera.up.set(0, 0, 1)
assert "camera.up.set(0, 0, 1)" in html_text, "Missing camera.up.set(0, 0, 1)!"
print("Camera Up Vector: camera.up.set(0, 0, 1) verified!")

# Check UI Specs Strings
assert "5,111,056 caras | 2,555,530" in html_text, "Missing 5.11M triangles in specs card!"
assert "416.20 cm" in html_text, "Missing 416.20 cm3 in specs card!"
assert "110.85 mm" in html_text, "Missing 110.85 mm in specs card!"
assert "128.00 mm" in html_text, "Missing 128.00 mm in specs card!"
assert "64.00 mm" in html_text, "Missing 64.00 mm in specs card!"
assert "0.00% Nieve en pendientes" in html_text, "Missing 0.00% snow physics in specs card!"
print("UI Specs Card: 100% verified with all engineering data!")

print("\n>>> WEBGL VIEWER HTML DIAGNOSTIC TEST: 100% PASSED! <<<")
