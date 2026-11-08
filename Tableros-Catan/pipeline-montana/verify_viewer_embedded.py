import re
import base64
import numpy as np

with open('Tableros Catan/Loseta_Montana_3D/catan_mountain_viewer.html', 'r', encoding='utf-8') as f:
    text = f.read()

m_z = re.search(r'const DEM_Z_B64 = "([^"]+)";', text)
m_t = re.search(r'const DEM_TOOLS_B64 = "([^"]+)";', text)
print('Found DEM_Z_B64 length:', len(m_z.group(1)) if m_z else None)
print('Found DEM_TOOLS_B64 length:', len(m_t.group(1)) if m_t else None)

if m_t:
    t_bytes = base64.b64decode(m_t.group(1))
    arr_t = np.frombuffer(t_bytes, dtype=np.uint8)
    vals, counts = np.unique(arr_t, return_counts=True)
    for v, c in zip(vals, counts):
        pct = 100.0 * c / len(arr_t)
        print(f"  Tool {v}: {c:6d} points ({pct:5.1f}%)")

if m_z:
    z_bytes = base64.b64decode(m_z.group(1))
    arr_z = np.frombuffer(z_bytes, dtype=np.uint16)
    z_mm = 25.0 + (arr_z / 65535.0) * 58.0
    print(f"Z range in viewer: min={np.min(z_mm):.2f} mm, max={np.max(z_mm):.2f} mm")
