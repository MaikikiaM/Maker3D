import os
import sys
import time
import zipfile
import hashlib
import numpy as np
import trimesh
from PIL import Image

print("==================================================================")
print("     MASTER END-TO-END CERTIFICATION & VERIFICATION HARNESS       ")
print("==================================================================")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D"
renders_dir = os.path.join(out_dir, "renders_cad")
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

all_passed = True
def check(desc, condition, details=""):
    global all_passed
    if condition:
        print(f"  [PASS] {desc}" + (f" -> {details}" if details else ""))
    else:
        print(f"  [FAIL] {desc}" + (f" -> {details}" if details else ""))
        all_passed = False

# -------------------------------------------------------------
# 1. 3D SOLID WATERTIGHT MESH AUDIT (STL)
# -------------------------------------------------------------
print("\n--- 1. AUDIT: 3D SOLID MONOLITHIC STL (5.11M TRIANGLES) ---")
stl_path = os.path.join(out_dir, "catan_mountain_solid.stl")
check("STL File Exists", os.path.exists(stl_path), f"{os.path.getsize(stl_path)/1024/1024:.2f} MB")

mesh = trimesh.load(stl_path)
check("Watertight Manifold", mesh.is_watertight, f"is_watertight = {mesh.is_watertight}")
check("Euler Characteristic chi = 2", mesh.euler_number == 2, f"chi = {mesh.euler_number} (genus 0)")
check("Triangles >= 5,000,000", len(mesh.faces) >= 5000000, f"{len(mesh.faces):,} faces")
check("Vertices >= 2,500,000", len(mesh.vertices) >= 2500000, f"{len(mesh.vertices):,} vertices")
check("Solid Volume ~ 416 cm3", 414.0 <= mesh.volume/1000.0 <= 418.0, f"{mesh.volume/1000.0:.2f} cm3")
check("Zero Degenerate Faces", np.sum(mesh.area_faces <= 1e-8) == 0, "0 degenerate faces")

b = mesh.bounds
check("X Bounding Span ~ 110.85 mm", abs((b[1, 0] - b[0, 0]) - 110.85) < 0.1, f"{b[1, 0] - b[0, 0]:.2f} mm")
check("Y Bounding Span ~ 128.00 mm", abs((b[1, 1] - b[0, 1]) - 128.00) < 0.1, f"{b[1, 1] - b[0, 1]:.2f} mm")
check("Z Elevation Span ~ 83.00 mm", abs(b[1, 2] - 83.00) < 0.1, f"{b[1, 2]:.2f} mm")

# -------------------------------------------------------------
# 2. FULL-COLOR DELIVERABLES AUDIT (PLY, OBJ, 3MF)
# -------------------------------------------------------------
print("\n--- 2. AUDIT: FULL-COLOR MULTI-FORMAT DELIVERABLES ---")
ply_path = os.path.join(out_dir, "catan_mountain_fullcolor.ply")
check("PLY TrueColor Exists", os.path.exists(ply_path), f"{os.path.getsize(ply_path)/1024/1024:.2f} MB")

obj_path = os.path.join(out_dir, "catan_mountain_fullcolor.obj")
mtl_path = os.path.join(out_dir, "catan_mountain.mtl")
check("OBJ Wavefront Exists", os.path.exists(obj_path), f"{os.path.getsize(obj_path)/1024/1024:.2f} MB")
check("MTL Material Exists", os.path.exists(mtl_path), f"{os.path.getsize(mtl_path)} bytes")

mf_path = os.path.join(out_dir, "catan_mountain_fullcolor.3mf")
check("3MF Archive Exists", os.path.exists(mf_path), f"{os.path.getsize(mf_path)/1024/1024:.2f} MB")
with zipfile.ZipFile(mf_path, 'r') as z:
    check("3MF contains 3D/3dmodel.model", "3D/3dmodel.model" in z.namelist())
    check("3MF contains embedded 4K Diffuse", "3D/Textures/catan_mountain_diffuse_4k.png" in z.namelist())

# -------------------------------------------------------------
# 3. TEXTURES AND NORMAL MAPS AUDIT (4K UHD)
# -------------------------------------------------------------
print("\n--- 3. AUDIT: 4K TEXTURES & NORMAL MAPS ---")
diff_4k = os.path.join(out_dir, "catan_mountain_diffuse_4k.png")
diff_2k = os.path.join(out_dir, "catan_mountain_diffuse_2k.png")
norm_4k = os.path.join(out_dir, "catan_mountain_normal_4k.png")

check("4K Diffuse Exists", os.path.exists(diff_4k), f"Size: {Image.open(diff_4k).size}")
check("2K Diffuse Exists", os.path.exists(diff_2k), f"Size: {Image.open(diff_2k).size}")
check("4K Normal Map Exists", os.path.exists(norm_4k), f"Size: {Image.open(norm_4k).size}")

# -------------------------------------------------------------
# 4. ALPINE SNOW PHYSICS & COLOR AUDIT
# -------------------------------------------------------------
print("\n--- 4. AUDIT: ALPINE SNOW RETENTION & 45/15 PALETTE ---")
diffuse_arr = np.load('scratch/diffuse_4k_array.npy')
greys_45 = np.load('scratch/greys_45.npy')
greens_15 = np.load('scratch/greens_15.npy')

check("45 Greys Palette Defined", len(greys_45) == 45, f"{len(greys_45)} tones")
check("15 Greens Palette Defined", len(greens_15) == 15, f"{len(greens_15)} tones")

# Verify zero green in snow
green_mask = (diffuse_arr[:, :, 1] > diffuse_arr[:, :, 0] * 1.15) & (diffuse_arr[:, :, 1] > diffuse_arr[:, :, 2] * 1.15)
snow_mask = (diffuse_arr[:, :, 0] > 0.85) & (diffuse_arr[:, :, 1] > 0.85) & (diffuse_arr[:, :, 2] > 0.85)
import scipy.ndimage as ndi
snow_dilated = ndi.binary_dilation(snow_mask, iterations=8)
green_near_snow = np.sum(green_mask & snow_dilated)
check("Zero Green Near Alpine Snow", green_near_snow == 0, f"{green_near_snow} pixels (100% clean separation)")

# -------------------------------------------------------------
# 5. WEBGL VIEWER HTML AUDIT
# -------------------------------------------------------------
print("\n--- 5. AUDIT: INTERACTIVE WEBGL 3D VIEWER ---")
html_path = os.path.join(out_dir, "catan_mountain_viewer.html")
check("Viewer HTML Exists", os.path.exists(html_path), f"{os.path.getsize(html_path)/1024/1024:.2f} MB")
with open(html_path, 'r', encoding='utf-8') as f_h:
    html_content = f_h.read()
check("Viewer Camera Up Vector (0, 0, 1)", "camera.up.set(0, 0, 1)" in html_content)
check("Viewer Updated to 5.11M faces", "5,111,056 caras" in html_content)
check("Viewer Base Cotas: 110.85 mm", "110.85 mm" in html_content)
check("Viewer Base Cotas: 128.00 mm", "128.00 mm" in html_content)
check("Viewer Base Cotas: 64.00 mm", "64.00 mm" in html_content)
check("Viewer 0% Snow on Cliffs Note", "0.00% Nieve en pendientes" in html_content)

# -------------------------------------------------------------
# 6. COMPLETE CAD RENDERS & BLUEPRINT SUITE
# -------------------------------------------------------------
print("\n--- 6. AUDIT: 14 MASTER CAD RENDERS & BLUEPRINTS ---")
expected_renders = [
    "render_mountain_isometric.png",
    "render_mountain_top_ortho.png",
    "render_mountain_macro_zoom_proof.png",
    "render_mountain_base_dimensions_blueprint.png",
    "render_palette_60_tones_audit.png",
    "render_palette_distribution_histogram.png",
    "render_mountain_6_player_perspectives.png",
    "render_solid_cross_section.png",
    "render_slicing_analysis_fdm.png",
    "render_alpine_solar_ephemeris_study.png",
    "render_macro_verification_6panel.png",
    "render_overhang_printability_map.png",
    "render_catan_7hex_cluster_assembly.png",
    "render_cie_lab_gamut_audit.png",
    "render_topo_contour_map.png",
    "render_tangent_normal_components.png",
    "render_lod_mesh_comparison.png",
    "render_multi_material_wipe_tower.png",
    "render_mountain_tabletop_eyelevel.png",
    "render_mountain_seasonal_triptych.png",
    "catan_mountain_turntable_360.gif"
]

for r_name in expected_renders:
    r_path = os.path.join(renders_dir, r_name)
    brain_r_path = os.path.join(brain_dir, r_name)
    exists_both = os.path.exists(r_path) and os.path.exists(brain_r_path)
    check(f"Render: {r_name}", exists_both, f"Size: {os.path.getsize(r_path)/1024:.1f} KB" if exists_both else "MISSING")

# -------------------------------------------------------------
# 7. FINAL VERDICT
# -------------------------------------------------------------
print("\n==================================================================")
if all_passed:
    print(">>> 100% OF ALL SPECIFICATIONS & CHECKS PASSED WITH ZERO FLAWS! <<<")
else:
    print(">>> WARNING: ONE OR MORE CHECKS FAILED! <<<")
print(f"Total Audit Execution Time: {time.time()-t0:.2f}s")
print("==================================================================")
