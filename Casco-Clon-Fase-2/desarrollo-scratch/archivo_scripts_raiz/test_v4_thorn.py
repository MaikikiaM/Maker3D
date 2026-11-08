import os
import numpy as np
import trimesh
import manifold3d as m3d
from PIL import Image, ImageDraw

print("=== TESTING PROTOTYPE V4 THORN GEOMETRY ===")

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
casco_dir = os.path.join(BASE_DIR, r"Diseños 3D\Casco")
greeb_dir = os.path.join(casco_dir, "Greeblies")

# Load base helmet
helmet_mesh = trimesh.load(os.path.join(casco_dir, "Phase 2 Main Body.stl"))
v_base = helmet_mesh.vertices.astype('float32')
f_base = helmet_mesh.faces.astype('uint32')
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))
base_volume = helmet_mani.volume()
print(f"Helmet base loaded: {base_volume:.1f} mm3")

mouth_mesh = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
ear_l = trimesh.load(os.path.join(greeb_dir, "Ear Piece Left.stl"))
ear_r = trimesh.load(os.path.join(greeb_dir, "Ear Piece Right.stl"))
ears_mesh = trimesh.util.concatenate([ear_l, ear_r])
aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))
aerators_mesh = trimesh.util.concatenate([aer_l, aer_r])

# -------------------------------------------------------------
# 1. DORSAL FIN CREST (White front face, red top and slope)
# -------------------------------------------------------------
print("1. Dorsal fin cutter...")
fin_poly_yz = np.array([
    [-12.0, 1918.0],
    [-2.0,  1885.0],
    [205.0, 1870.0],
    [205.0, 1960.0],
    [-12.0, 1960.0]
], dtype=np.float32)
cs_fin = m3d.CrossSection([fin_poly_yz])
raw_fin = cs_fin.extrude(31.6)
m_rf = raw_fin.to_mesh()
v_rf = m_rf.vert_properties
v_fin_3d = np.column_stack([v_rf[:, 2] - 15.8, v_rf[:, 0], v_rf[:, 1]]).astype('float32')
f_fin_3d = m_rf.tri_verts.copy()
fin_cutter = m3d.Manifold(m3d.Mesh(vert_properties=v_fin_3d, tri_verts=f_fin_3d))
if fin_cutter.volume() <= 0:
    f_fin_3d[:, [0, 1]] = f_fin_3d[:, [1, 0]]
    fin_cutter = m3d.Manifold(m3d.Mesh(vert_properties=v_fin_3d, tri_verts=f_fin_3d))

# -------------------------------------------------------------
# 2. FRONT BROW BAND & CENTRAL WHITE V-WEDGE
# Height: Z = 1808.6 (visor rim) to 1845.0 (dome groove).
# Y: from -95.0 to ear front (Y = 58.0).
# Wedge: X from [-15, 15] at Z=1808.6 to [-25, 25] at Z=1845.0.
# -------------------------------------------------------------
print("2. Front brow band with central V-wedge...")
brow_outer = m3d.Manifold.cube([270.0, 153.0, 36.4], center=False).translate([-135.0, -95.0, 1808.6])
# V-wedge
wedge_verts = np.array([
    [-15.0, -98.0, 1806.0], [ 15.0, -98.0, 1806.0], [ 15.0,  15.0, 1806.0], [-15.0,  15.0, 1806.0],
    [-25.0, -98.0, 1848.0], [ 25.0, -98.0, 1848.0], [ 25.0,  15.0, 1848.0], [-25.0,  15.0, 1848.0],
], dtype=np.float32)
wedge_faces = np.array([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]
], dtype=np.uint32)
wedge_mani = m3d.Manifold(m3d.Mesh(vert_properties=wedge_verts, tri_verts=wedge_faces))
brow_cutter = brow_outer - wedge_mani

# -------------------------------------------------------------
# 3. THOR DOME WINGS (Organic Clamp / Pinza with 4 feathers & 3 slits)
# Smoothly emerges from brow at Z=1844, widens at root base,
# top curves with dome, bottom feather is narrower, clean white gap below.
# Slits end before front root.
# -------------------------------------------------------------
print("3. Thor dome wings...")
# Wing outline in Y-Z plane:
# Root base at Y in [14.0, 32.0], Z = 1844.0 (emerges from brow band)
# Front contour: curves smoothly up from (14.0, 1844.0) to top corner (18.0, 1898.0)
# Top edge: curves with dome from (18.0, 1898.0) -> (45.0, 1908.0) -> (75.0, 1910.0) -> (105.0, 1904.0) -> (118.0, 1896.0)
# Rear slanted cut: from (118.0, 1896.0) to (76.0, 1856.0)
# Under-edge of bottom feather (Feather 4): from (76.0, 1856.0) to (32.0, 1850.0)
# White gap below Feather 4: between Z=1844 and Z=1850, behind root (Y > 32.0).
# Front edge of root widening:
wing_poly_yz = np.array([
    [ 12.0, 1844.0],  # Bottom front of root (wide base at brow)
    [ 16.0, 1870.0],  # Front edge rising and leaning back slightly
    [ 18.0, 1898.0],  # Front top curve of feather 1
    [ 35.0, 1906.0],  # Top curve of feather 1 following dome
    [ 65.0, 1910.0],
    [ 95.0, 1905.0],
    [118.0, 1896.0],  # Rear tip of feather 1
    # Rear diagonal cut plane:
    [104.0, 1883.0],  # Rear tip of feather 2
    [ 90.0, 1870.0],  # Rear tip of feather 3
    [ 76.0, 1857.0],  # Rear tip of feather 4 (bottom feather)
    # Bottom edge of feather 4:
    [ 50.0, 1853.0],
    [ 32.0, 1850.0],  # Under-junction with root
    # Gap boundary to root:
    [ 32.0, 1844.0],  # Root back edge at brow
    [ 12.0, 1844.0]   # Close loop
], dtype=np.float32)

cs_wing_base = m3d.CrossSection([wing_poly_yz])

# 3 Parallel white slits:
# They terminate at Y = 26.0 (leaving a solid 10-14mm red root at the front)
# Width of white slit = 3.2 mm.
# Slit 1 (between feather 1 and 2):
slit1 = np.array([
    [ 26.0, 1887.0],
    [130.0, 1887.0],
    [130.0, 1890.2],
    [ 26.0, 1890.2]
], dtype=np.float32)

# Slit 2 (between feather 2 and 3):
slit2 = np.array([
    [ 28.0, 1874.0],
    [120.0, 1874.0],
    [120.0, 1877.2],
    [ 28.0, 1877.2]
], dtype=np.float32)

# Slit 3 (between feather 3 and 4):
slit3 = np.array([
    [ 30.0, 1861.0],
    [110.0, 1861.0],
    [110.0, 1864.2],
    [ 30.0, 1864.2]
], dtype=np.float32)

cs_slits = m3d.CrossSection([slit1]) + m3d.CrossSection([slit2]) + m3d.CrossSection([slit3])
cs_wing = cs_wing_base - cs_slits

# Extrude wing in X: starts at |X| = 28.0 mm (leaving space for fin/forehead) and extends out to 130.0 mm
raw_w = cs_wing.extrude(110.0)
m_rw = raw_w.to_mesh()
vw = m_rw.vert_properties
fw = m_rw.tri_verts

# Right wing (+X)
vw_right = np.column_stack([vw[:, 2] + 28.0, vw[:, 0], vw[:, 1]]).astype('float32')
triw_r = fw.copy()
mani_w_r = m3d.Manifold(m3d.Mesh(vert_properties=vw_right, tri_verts=triw_r))
if mani_w_r.volume() <= 0:
    triw_r[:, [0, 1]] = triw_r[:, [1, 0]]
    mani_w_r = m3d.Manifold(m3d.Mesh(vert_properties=vw_right, tri_verts=triw_r))

# Left wing (-X)
vw_left = np.column_stack([-(vw[:, 2] + 28.0), vw[:, 0], vw[:, 1]]).astype('float32')
triw_l = triw_r.copy()
triw_l[:, [0, 1]] = triw_l[:, [1, 0]]
mani_w_l = m3d.Manifold(m3d.Mesh(vert_properties=vw_left, tri_verts=triw_l))

wings_cutter = mani_w_r + mani_w_l

# -------------------------------------------------------------
# 4. REAR BAND ON DOME & 2 FULL-LENGTH VERTICAL STRIPES
# Horizontal band sits ON DOME at Z = 1840.0 to 1872.0 mm!
# Arches over circular disc (disc center Z=1819, top Z=1840).
# Wraps around back but stops at Y = 135.0 mm (generous white gap to wings).
# 2 Vertical stripes rise from Z = 1616.0 ALL THE WAY to Z = 1840.0 mm!
# -------------------------------------------------------------
print("4. Rear markings on dome and full-length vertical stripes...")
rear_band_box = m3d.Manifold.cube([270.0, 95.0, 32.0], center=False).translate([-135.0, 135.0, 1840.0])
# Rear circular disc cutter (radius 22 mm, centered at Y=230, Z=1819)
rear_disc_c = m3d.Manifold.cylinder(60.0, 22.0, 22.0, 128).rotate([90, 0, 0]).translate([0, 230.0, 1819.0])
rear_band = rear_band_box - rear_disc_c

# 2 Vertical stripes running up the ribs from Z = 1616 to Z = 1840.0 (meeting rear band!):
rib_r_xz = np.array([
    [13.3, 1616.0],
    [14.5, 1616.0],
    [26.5, 1840.0],
    [13.3, 1840.0]
], dtype=np.float32)
cs_rib_r = m3d.CrossSection([rib_r_xz])
raw_rib_r = cs_rib_r.extrude(55.0)
m_rrr = raw_rib_r.to_mesh()
v_rrr = m_rrr.vert_properties
v_rib_r_3d = np.column_stack([v_rrr[:, 0], v_rrr[:, 2] + 175.0, v_rrr[:, 1]]).astype('float32')
f_rib_r_3d = m_rrr.tri_verts.copy()
rib_r_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_rib_r_3d, tri_verts=f_rib_r_3d))
if rib_r_mani.volume() <= 0:
    f_rib_r_3d[:, [0, 1]] = f_rib_r_3d[:, [1, 0]]
    rib_r_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_rib_r_3d, tri_verts=f_rib_r_3d))

v_rib_l_3d = v_rib_r_3d.copy()
v_rib_l_3d[:, 0] = -v_rib_l_3d[:, 0]
f_rib_l_3d = f_rib_r_3d.copy()
f_rib_l_3d[:, [0, 1]] = f_rib_l_3d[:, [1, 0]]
rib_l_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_rib_l_3d, tri_verts=f_rib_l_3d))

rear_markings = rear_band + rib_r_mani + rib_l_mani

# -------------------------------------------------------------
# 5. FACE, SNOUT, CHEEKS & DEPRESSION, WITH CLEAN VISOR NOTCH
# -------------------------------------------------------------
print("5. Snout, cheeks, lateral depression, claws, and sub-ocular...")

# A. Snout & Chin (Z in [1618, 1680], X in [-45, 45], Y in [-98, -40])
snout_cutter = m3d.Manifold.cube([92.0, 60.0, 64.0], center=False).translate([-46.0, -98.0, 1618.0])

# B. Cheeks with anatomical lateral depression:
# In Y-Z, the cheek boundary sweeps backward:
# - Z = 1618 to 1660: Y up to +4.0 (covers first 2 tube slits)
# - Z = 1660 to 1750: Y curves back to +42.0 (covers entire depression floor and vertical back wall)
# - Z = 1750 to 1808.6: Y extends to +58.0 (brow/ledge meeting ear cap)
cheek_poly_yz = np.array([
    [-98.0, 1618.0],
    [  4.0, 1618.0],
    [  4.0, 1660.0],
    [ 25.0, 1700.0],
    [ 42.0, 1735.0],
    [ 42.0, 1755.0],
    [ 58.0, 1775.0],
    [ 58.0, 1808.6],
    [-98.0, 1808.6],
    [-98.0, 1618.0]
], dtype=np.float32)
cs_cheek = m3d.CrossSection([cheek_poly_yz])
# Extrude in X: from |X| = 14.5 to 125.0 mm (covers both cheeks, leaving 29mm central visor slot clean!)
raw_cheek = cs_cheek.extrude(110.0)
m_rc = raw_cheek.to_mesh()
v_rc = m_rc.vert_properties
f_rc = m_rc.tri_verts

# Right cheek (+X)
v_cr = np.column_stack([v_rc[:, 2] + 14.5, v_rc[:, 0], v_rc[:, 1]]).astype('float32')
f_cr = f_rc.copy()
cheek_r = m3d.Manifold(m3d.Mesh(vert_properties=v_cr, tri_verts=f_cr))
if cheek_r.volume() <= 0:
    f_cr[:, [0, 1]] = f_cr[:, [1, 0]]
    cheek_r = m3d.Manifold(m3d.Mesh(vert_properties=v_cr, tri_verts=f_cr))

# Left cheek (-X)
v_cl = np.column_stack([-(v_rc[:, 2] + 14.5), v_rc[:, 0], v_rc[:, 1]]).astype('float32')
f_cl = f_cr.copy()
f_cl[:, [0, 1]] = f_cl[:, [1, 0]]
cheek_l = m3d.Manifold(m3d.Mesh(vert_properties=v_cl, tri_verts=f_cl))

cheeks_raw = cheek_r + cheek_l

# C. Sub-ocular white stripe (strictly 55-65% length, between |X| = 38 and 72 mm):
# Regions |X| < 38 and |X| > 72 stay completely RED!
pts_suboc_r = np.array([
    [ 38.0, -85.0, 1782.0], [ 72.0, -35.0, 1768.0], [ 72.0, 10.0, 1768.0], [ 38.0, 10.0, 1782.0],
    [ 38.0, -85.0, 1810.0], [ 72.0, -35.0, 1810.0], [ 72.0, 10.0, 1810.0], [ 38.0, 10.0, 1810.0]
], dtype=np.float32)
faces_suboc = np.array([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]
], dtype=np.uint32)
mani_suboc_r = m3d.Manifold(m3d.Mesh(vert_properties=pts_suboc_r, tri_verts=faces_suboc))

pts_suboc_l = pts_suboc_r.copy()
pts_suboc_l[:, 0] = -pts_suboc_l[:, 0]
faces_suboc_l = faces_suboc.copy()
faces_suboc_l[:, [0, 1]] = faces_suboc_l[:, [1, 0]]
mani_suboc_l = m3d.Manifold(m3d.Mesh(vert_properties=pts_suboc_l, tri_verts=faces_suboc_l))

suboc_cutter = mani_suboc_r + mani_suboc_l

# D. 3 White Claw stripes on tubes extending across aerator centerline (X = +/- 47.4):
# Span in X: from |X| = 40.0 to 98.0 mm!
claws_raw = m3d.Manifold()
for y_c in [-54.0, -41.0, -28.0]:
    s_r = m3d.Manifold.cube([58.0, 5.2, 90.0], center=True).rotate([26.5, 0, 0]).translate([69.0, y_c, 1641.0])
    s_l = m3d.Manifold.cube([58.0, 5.2, 90.0], center=True).rotate([26.5, 0, 0]).translate([-69.0, y_c, 1641.0])
    claws_raw = claws_raw + s_r + s_l

tube_limit = m3d.Manifold.cube([240.0, 100.0, 48.0], center=False).translate([-120.0, -98.0, 1618.0])
claws_cutter = claws_raw ^ tube_limit

face_cutter = (snout_cutter + cheeks_raw) - suboc_cutter - claws_cutter

# -------------------------------------------------------------
# 6. TOTAL BOOLEAN ASSEMBLY & VISOR APERTURE PROTECTION
# -------------------------------------------------------------
print("6. Total assembly and visor aperture protection...")
# Subtract an explicit volume for the T-visor aperture so ZERO red material
# can ever cross, bleed, or penetrate inside the visor opening!
# Horizontal visor aperture: |X| in [0, 85], Y in [-90, -10], Z in [1745, 1808.6]
# Vertical visor slit: |X| in [0, 14.5], Y in [-95, -10], Z in [1680, 1808.6]
visor_protect = m3d.Manifold.cube([170.0, 90.0, 70.0], center=False).translate([-85.0, -90.0, 1740.0]) + \
                m3d.Manifold.cube([29.0, 90.0, 130.0], center=False).translate([-14.5, -95.0, 1680.0])

total_red_cutter = (fin_cutter + brow_cutter + wings_cutter + rear_markings + face_cutter) - visor_protect

print("Performing boolean operations...")
red_markings_mani = helmet_mani ^ total_red_cutter
white_base_mani = helmet_mani - total_red_cutter

vol_red = red_markings_mani.volume()
vol_white = white_base_mani.volume()
vol_sum = vol_red + vol_white
dev = abs(vol_sum - base_volume)

print(f"Red markings volume: {vol_red:.1f} mm3")
print(f"White base volume: {vol_white:.1f} mm3")
print(f"Volumetric conservation: {vol_sum:.1f} mm3 (Deviation: {dev:.4f} mm3)")

# Extract meshes
m_red = red_markings_mani.to_mesh()
mesh_red = trimesh.Trimesh(vertices=m_red.vert_properties[:, :3], faces=m_red.tri_verts, process=False)
mesh_red.visual.vertex_colors = [185, 28, 28, 255]

m_white = white_base_mani.to_mesh()
mesh_white = trimesh.Trimesh(vertices=m_white.vert_properties[:, :3], faces=m_white.tri_verts, process=False)
mesh_white.visual.vertex_colors = [244, 244, 246, 255]

mouth_mesh.visual.vertex_colors = [20, 20, 22, 255]
ears_mesh.visual.vertex_colors = [240, 240, 244, 255]
aerators_mesh.visual.vertex_colors = [70, 70, 75, 255]

# Render 4-angle turnaround
print("Rendering 4-angle test turnaround...")
angles = [
    ("Front (0 deg)", 0.0, 0.0),
    ("3/4 Iso (35 deg)", 35.0, 8.0),
    ("Side Profile (90 deg)", 90.0, 0.0),
    ("Rear View (180 deg)", 180.0, 0.0)
]

scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, ears_mesh, aerators_mesh])
center = scene.centroid
cam_dist = 820.0

imgs = []
for label, yaw_deg, pitch_deg in angles:
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    cam_x = cam_dist * np.sin(yaw) * np.cos(pitch)
    cam_y = -cam_dist * np.cos(yaw) * np.cos(pitch)
    cam_z = cam_dist * np.sin(pitch)
    cam_pos = np.array([cam_x, cam_y, cam_z]) + center
    
    view_vec = center - cam_pos
    forward = view_vec / np.linalg.norm(view_vec)
    up = np.array([0, 0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    actual_up = np.cross(right, forward)
    actual_up /= np.linalg.norm(actual_up)
    
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = actual_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = cam_pos
    scene.camera_transform = c2w
    
    data = scene.save_image(resolution=(1200, 1200))
    from io import BytesIO
    img = Image.open(BytesIO(data)).convert("RGBA").resize((800, 800), Image.Resampling.LANCZOS)
    imgs.append(img)

# Combine 2x2
canvas = Image.new("RGBA", (1600, 1600), (255, 255, 255, 255))
canvas.paste(imgs[0], (0, 0))
canvas.paste(imgs[1], (800, 0))
canvas.paste(imgs[2], (0, 800))
canvas.paste(imgs[3], (800, 800))

out_test = os.path.join(BASE_DIR, "test_v4_turnaround.png")
canvas.save(out_test)
print(f"Test turnaround saved to: {out_test}")
