import os
import numpy as np
import trimesh
import manifold3d as m3d
from PIL import Image

print("=== RUNNING V6 PERFECT THORN VALIDATION ===")

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
casco_dir = os.path.join(BASE_DIR, r"Diseños 3D\Casco")
greeb_dir = os.path.join(casco_dir, "Greeblies")

# Load base helmet
helmet_mesh = trimesh.load(os.path.join(casco_dir, "Phase 2 Main Body.stl"))
v_base = helmet_mesh.vertices.astype('float32')
f_base = helmet_mesh.faces.astype('uint32')
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))
base_volume = helmet_mani.volume()
print(f"Base helmet volume: {base_volume:.1f} mm3")

mouth_mesh = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
ear_l = trimesh.load(os.path.join(greeb_dir, "Ear Piece Left.stl"))
ear_r = trimesh.load(os.path.join(greeb_dir, "Ear Piece Right.stl"))
ears_mesh = trimesh.util.concatenate([ear_l, ear_r])
aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))
aerators_mesh = trimesh.util.concatenate([aer_l, aer_r])

# 1. DORSAL FIN CREST
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

# 2. FRONT BROW BAND & CENTRAL WHITE V-WEDGE
brow_outer = m3d.Manifold.cube([270.0, 153.0, 36.4], center=False).translate([-135.0, -95.0, 1808.6])
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

# 3. THOR DOME WINGS (4 FEATHERS, 3 SLITS, PINZA ANATOMY)
pts_wing = [
    [ 10.0, 1845.0], [ 13.0, 1865.0], [ 16.0, 1885.0], [ 19.0, 1900.0],
    [ 24.0, 1906.0], [ 45.0, 1910.0], [ 70.0, 1911.0], [ 95.0, 1906.0],
    [118.0, 1896.0], [104.0, 1883.0], [ 90.0, 1870.0], [ 76.0, 1856.0],
    [ 55.0, 1852.0], [ 32.0, 1849.0], [ 32.0, 1845.0]
]
cs_wing_base = m3d.CrossSection([pts_wing])
if cs_wing_base.area() == 0:
    cs_wing_base = m3d.CrossSection([pts_wing[::-1]])

def make_slot(y_start, z_start, y_end, z_end, thickness=3.2):
    dx = y_end - y_start
    dy = z_end - z_start
    L = np.hypot(dx, dy)
    ux, uy = -dy/L * (thickness/2), dx/L * (thickness/2)
    p = [
        [y_start - ux, z_start - uy],
        [y_end   - ux, z_end   - uy],
        [y_end   + ux, z_end   + uy],
        [y_start + ux, z_start + uy]
    ]
    cs = m3d.CrossSection([p])
    if cs.area() == 0:
        cs = m3d.CrossSection([p[::-1]])
    return cs

s1 = make_slot(28.0, 1889.0, 125.0, 1890.0, thickness=3.2)
s2 = make_slot(30.0, 1876.0, 115.0, 1877.0, thickness=3.2)
s3 = make_slot(32.0, 1863.0, 105.0, 1864.0, thickness=3.2)

cs_wing = cs_wing_base - (s1 + s2 + s3)
raw_w = cs_wing.extrude(110.0)

m_rw = raw_w.to_mesh()
vw = m_rw.vert_properties
fw = m_rw.tri_verts

vw_r = np.column_stack([vw[:, 2] + 28.0, vw[:, 0], vw[:, 1]]).astype('float32')
tri_r = fw.copy()
mani_w_r = m3d.Manifold(m3d.Mesh(vert_properties=vw_r, tri_verts=tri_r))
if mani_w_r.volume() <= 0:
    tri_r[:, [0, 1]] = tri_r[:, [1, 0]]
    mani_w_r = m3d.Manifold(m3d.Mesh(vert_properties=vw_r, tri_verts=tri_r))

vw_l = np.column_stack([-(vw[:, 2] + 28.0), vw[:, 0], vw[:, 1]]).astype('float32')
tri_l = tri_r.copy()
tri_l[:, [0, 1]] = tri_l[:, [1, 0]]
mani_w_l = m3d.Manifold(m3d.Mesh(vert_properties=vw_l, tri_verts=tri_l))

wings_cutter = mani_w_r + mani_w_l

# 4. REAR MARKINGS ON DOME & 2 FULL-LENGTH VERTICAL STRIPES
rear_band_box = m3d.Manifold.cube([270.0, 95.0, 32.0], center=False).translate([-135.0, 135.0, 1840.0])
rear_disc_c = m3d.Manifold.cylinder(60.0, 22.0, 22.0, 128).rotate([90, 0, 0]).translate([0, 230.0, 1819.0])
rear_band = rear_band_box - rear_disc_c

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

# 5. SNOUT, CHIN, CHEEKS, LATERAL DEPRESSION, CLAWS & SUB-OCULAR STRIPE
# Snout & Chin: covers from Z = 1618.0 up to Z = 1741.8 (bottom of vertical visor opening)
# Full width X in [-46, 46], Y in [-98, -40]
snout_cutter = m3d.Manifold.cube([92.0, 60.0, 123.8], center=False).translate([-46.0, -98.0, 1618.0])

# Cheeks with lateral depression:
# In Y-Z: sweeps back to cover first 2 tube slits at bottom (Y=+4), entire depression floor and back wall (Y=+42),
# and cheek upper ledge meeting ear cap (Y=+58).
cheek_poly_yz = np.array([
    [-98.0, 1618.0],
    [  4.0, 1618.0],
    [  4.0, 1660.0],
    [ 25.0, 1700.0],
    [ 42.0, 1735.0],
    [ 42.0, 1755.0],
    [ 58.0, 1775.0],
    [ 58.0, 1808.6],
    [-98.0, 1808.6]
], dtype=np.float32)

cs_cheek = m3d.CrossSection([cheek_poly_yz])
if cs_cheek.area() == 0:
    cs_cheek = m3d.CrossSection([cheek_poly_yz[::-1]])

raw_cheek = cs_cheek.extrude(117.0)
m_rc = raw_cheek.to_mesh()
v_rc = m_rc.vert_properties
f_rc = m_rc.tri_verts

# Right cheek (+X): starts at X = +5.0 mm (covers entire cheek flap right up to visor edge)
v_cr = np.column_stack([v_rc[:, 2] + 5.0, v_rc[:, 0], v_rc[:, 1]]).astype('float32')
f_cr = f_rc.copy()
cheek_r = m3d.Manifold(m3d.Mesh(vert_properties=v_cr, tri_verts=f_cr))
if cheek_r.volume() <= 0:
    f_cr[:, [0, 1]] = f_cr[:, [1, 0]]
    cheek_r = m3d.Manifold(m3d.Mesh(vert_properties=v_cr, tri_verts=f_cr))

# Left cheek (-X): starts at X = -5.0 mm
v_cl = np.column_stack([-(v_rc[:, 2] + 5.0), v_rc[:, 0], v_rc[:, 1]]).astype('float32')
f_cl = f_cr.copy()
f_cl[:, [0, 1]] = f_cl[:, [1, 0]]
cheek_l = m3d.Manifold(m3d.Mesh(vert_properties=v_cl, tri_verts=f_cl))

cheeks_raw = cheek_r + cheek_l

# Sub-ocular white stripe (strictly between |X| = 32.0 and 68.0 mm)
pts_suboc_r = np.array([
    [ 32.0, -75.0, 1780.0], [ 68.0, -35.0, 1770.0], [ 68.0, 0.0, 1770.0], [ 32.0, 0.0, 1780.0],
    [ 32.0, -75.0, 1805.0], [ 68.0, -35.0, 1805.0], [ 68.0, 0.0, 1805.0], [ 32.0, 0.0, 1805.0]
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

# 3 White Claw stripes on tubes extending across aerator centerline (X = +/- 47.4)
claws_r = m3d.Manifold()
claws_l = m3d.Manifold()
for offset in [-14.0, 0.0, 14.0]:
    cr = m3d.Manifold.cube([45.0, 5.5, 60.0], center=True)
    cr = cr.rotate([26.0, 0.0, -18.0])
    cr = cr.translate([65.0 + offset * 0.4, -42.0 + offset * 0.9, 1640.0 + offset * 0.4])
    claws_r = claws_r + cr
    
    cl = m3d.Manifold.cube([45.0, 5.5, 60.0], center=True)
    cl = cl.rotate([26.0, 0.0, 18.0])
    cl = cl.translate([- (65.0 + offset * 0.4), -42.0 + offset * 0.9, 1640.0 + offset * 0.4])
    claws_l = claws_l + cl

tube_limit = m3d.Manifold.cube([240.0, 100.0, 56.0], center=False).translate([-120.0, -98.0, 1618.0])
claws_cutter = (claws_r + claws_l) ^ tube_limit

face_cutter = (snout_cutter + cheeks_raw) - suboc_cutter - claws_cutter

# 6. TOTAL ASSEMBLY
total_red_cutter = fin_cutter + brow_cutter + wings_cutter + rear_markings + face_cutter

print("Performing exact booleans...")
red_markings_mani = helmet_mani ^ total_red_cutter
white_base_mani = helmet_mani - total_red_cutter

vol_red = red_markings_mani.volume()
vol_white = white_base_mani.volume()
vol_sum = vol_red + vol_white
dev = abs(vol_sum - base_volume)

print(f"Red volume: {vol_red:.1f} mm3")
print(f"White volume: {vol_white:.1f} mm3")
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

canvas = Image.new("RGBA", (1600, 1600), (255, 255, 255, 255))
canvas.paste(imgs[0], (0, 0))
canvas.paste(imgs[1], (800, 0))
canvas.paste(imgs[2], (0, 800))
canvas.paste(imgs[3], (800, 800))

out_test = os.path.join(BASE_DIR, "test_v6_turnaround.png")
canvas.save(out_test)
print(f"V6 test turnaround saved to: {out_test}")
