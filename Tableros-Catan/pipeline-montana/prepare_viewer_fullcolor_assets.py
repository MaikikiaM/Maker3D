import os
import base64
import numpy as np

print("--- Preparing Flawless Viewer Assets ---")
tex_2k_path = r"Tableros Catan\Loseta_Montana_3D\catan_mountain_diffuse_2k.png"
with open(tex_2k_path, 'rb') as f:
    tex_bytes = f.read()
b64_tex = base64.b64encode(tex_bytes).decode('ascii')
print(f"2K Texture B64 length: {len(b64_tex):,} chars ({len(b64_tex)/1024/1024:.2f} MB)")

Z_master = np.load('scratch/Z_top_master_2m.npy')
ny_m, nx_m = Z_master.shape
Z_BASE = 25.0
Z_PEAK = 83.0
RELIEF_H = 58.0

z_norm = np.clip((Z_master - Z_BASE) / RELIEF_H, 0.0, 1.0)
z_u16 = (z_norm * 65535.0).astype(np.uint16)
b64_z = base64.b64encode(z_u16.tobytes()).decode('ascii')
print(f"Z Topography B64 length: {len(b64_z):,} chars ({len(b64_z)/1024:.1f} KB)")

with open('scratch/fullcolor_viewer_data.js', 'w', encoding='utf-8') as f:
    f.write(f"// Master Full-Color Continuous Topography & 2K Diffuse Map\n")
    f.write(f"const DEM_NX = {nx_m};\n")
    f.write(f"const DEM_NY = {ny_m};\n")
    f.write(f"const DEM_Z_B64 = \"{b64_z}\";\n")
    f.write(f"const DEM_TEX_B64 = \"data:image/png;base64,{b64_tex}\";\n")

print("Generated scratch/fullcolor_viewer_data.js successfully.")
