import os
import time
import numpy as np
import trimesh
import manifold3d as m3d
from io import BytesIO
from PIL import Image, ImageDraw
import trimesh.viewer.windowed

_orig_update_perspective = trimesh.viewer.windowed.SceneViewer._update_perspective
def _safe_update_perspective(self, width, height):
    return _orig_update_perspective(self, max(1, width), max(1, height))
trimesh.viewer.windowed.SceneViewer._update_perspective = _safe_update_perspective

_orig_on_resize = trimesh.viewer.windowed.SceneViewer.on_resize
def _safe_on_resize(self, width, height):
    try:
        return _orig_on_resize(self, max(1, width), max(1, height))
    except Exception:
        return
trimesh.viewer.windowed.SceneViewer.on_resize = _safe_on_resize

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
casco_dir = os.path.join(BASE_DIR, r"Diseños 3D\Casco")
greeb_dir = os.path.join(casco_dir, "Greeblies")

print("1. Loading base helmet and components...")
mb = trimesh.load(os.path.join(casco_dir, "Phase 2 Main Body.stl"))
v_base = mb.vertices.astype('float32')
f_base = mb.faces.astype('uint32')
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))

mouth_mesh = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
ear_l = trimesh.load(os.path.join(greeb_dir, "Ear Piece Left.stl"))
ear_r = trimesh.load(os.path.join(greeb_dir, "Ear Piece Right.stl"))
ears_mesh = trimesh.util.concatenate([ear_l, ear_r])
aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))
aerators_mesh = trimesh.util.concatenate([aer_l, aer_r])

# Visor Lens (black curved visor filling the aperture)
lens_path = os.path.join(casco_dir, r"Antigravity\05_Cascos_Pintados_Clones\06_Lentes_Visor_Negro.stl")
if os.path.exists(lens_path):
    lens_mesh = trimesh.load(lens_path)
else:
    theta = np.linspace(-np.radians(60), np.radians(60), 30)
    lx = 92.0 * np.sin(theta)
    ly = 10.0 - 58.0 * np.cos(theta)
    pts = []
    for i in range(len(theta)):
        pts.append([lx[i], ly[i], 1750.0])
        pts.append([lx[i], ly[i], 1805.0])
    lh = trimesh.convex.convex_hull(np.vstack([pts, np.array(pts) + [0, 4, 0]]))
    lens_mesh = trimesh.Trimesh(vertices=lh.vertices, faces=lh.faces)

# Construct cutters
# A. Dorsal Fin (Cresta dorsal)
fin_poly_yz = np.array([
    [-12.0, 1918.0],
    [-2.0,  1885.0],
    [205.0, 1870.0],
    [205.0, 1960.0],
    [-12.0, 1960.0]
], dtype=np.float32)
cs_fin = m3d.CrossSection([fin_poly_yz])
raw_fin = cs_fin.extrude(36.0)
m_rf = raw_fin.to_mesh()
v_rf = m_rf.vert_properties
v_fin_3d = np.column_stack([v_rf[:, 2] - 18.0, v_rf[:, 0], v_rf[:, 1]]).astype('float32')
f_fin_3d = m_rf.tri_verts.copy()
fin_cutter = m3d.Manifold(m3d.Mesh(vert_properties=v_fin_3d, tri_verts=f_fin_3d))
if fin_cutter.volume() <= 0:
    f_fin_3d[:, [0, 1]] = f_fin_3d[:, [1, 0]]
    fin_cutter = m3d.Manifold(m3d.Mesh(vert_properties=v_fin_3d, tri_verts=f_fin_3d))

# B. Brow & Forehead with Central V Wedge
# Lower brow band (Z from 1808.6 to 1826.0): wraps along brow above cheek depression to Y = +44 mm
brow_lower = m3d.Manifold.cube([270.0, 144.0, 17.4], center=False).translate([-135.0, -100.0, 1808.6])
# Upper forehead on dome (Z from 1826.0 to 1862.0): stops at Y = -12.0 mm, leaving clean white gap of 18 mm before wing
brow_upper = m3d.Manifold.cube([270.0, 88.0, 36.0], center=False).translate([-135.0, -100.0, 1826.0])
brow_solid = brow_lower + brow_upper

wedge_verts = np.array([
    [-14.0, -105.0, 1806.0], [ 14.0, -105.0, 1806.0], [ 14.0,  30.0, 1806.0], [-14.0,  30.0, 1806.0],
    [-22.0, -105.0, 1865.0], [ 22.0, -105.0, 1865.0], [ 22.0,  30.0, 1865.0], [-22.0,  30.0, 1865.0],
], dtype=np.float32)
wedge_faces = np.array([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]
], dtype=np.uint32)
wedge_mani = m3d.Manifold(m3d.Mesh(vert_properties=wedge_verts, tri_verts=wedge_faces))
brow_cutter = brow_solid - wedge_mani

# C. Thor Wings on Dome (Organic pincher starting right above horizontal seam)
pts_wing = [
    [ 6.0, 1824.0],   # Seamless empalme with brow_lower (Z=1826.0) across the seam line
    [ 8.0, 1842.0],   # Smooth forward-leaning curve (no sharp 90 deg corner)
    [12.0, 1858.0],
    [18.0, 1874.0],
    [26.0, 1888.0],
    [38.0, 1898.0],   # Smooth arch to top feather
    [60.0, 1902.0],
    [85.0, 1898.0],
    [110.0, 1886.0],  # Tip Feather 1 (top)
    [ 98.0, 1874.0],  # Tip Feather 2
    [ 86.0, 1862.0],  # Tip Feather 3
    [ 74.0, 1848.0],  # Tip Feather 4 (bottom)
    [ 55.0, 1836.0],  # Lower contour of Feather 4
    [ 38.0, 1828.0],  # Lower contour along dome base
    [ 24.0, 1824.0],  # Lower root transition empalming with brow_lower
    [  6.0, 1824.0]
]
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

# Slits angled slightly following dome curve upward
s1 = make_slot(34.0, 1886.0, 115.0, 1888.0, thickness=3.2)
s2 = make_slot(26.0, 1868.0, 102.0, 1873.0, thickness=3.2)
s3 = make_slot(18.0, 1850.0,  90.0, 1857.0, thickness=3.2)

cs_wing = cs_wing_base - (s1 + s2 + s3)
raw_w = cs_wing.extrude(115.0)
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

# D. Rear Band & Anatomical Tapered Ribs
band_box = m3d.Manifold.cube([150.0, 50.0, 22.0], center=False).translate([-75.0, 175.0, 1848.0])
disc_cyl = m3d.Manifold.cylinder(80.0, 26.5, 26.5, 128).rotate([90, 0, 0]).scale([1.0, 1.0, 31.0/26.5]).translate([0, 235.0, 1820.0])

# Anatomical tapered ribs (without [::-1] so area is positive 2054 mm2!)
rib_pts = np.array([
    [13.0, 1616.0],
    [21.0, 1616.0],
    [23.0, 1750.0],
    [28.0, 1800.0],
    [34.5, 1820.0],
    [34.0, 1850.0],
    [24.0, 1850.0],
    [25.5, 1820.0],
    [19.0, 1800.0],
    [14.0, 1750.0],
    [13.0, 1616.0]
], dtype=np.float32)

cs_rib_r = m3d.CrossSection([rib_pts])
raw_rib_r = cs_rib_r.extrude(60.0)
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

rear_markings = (band_box + rib_r_mani + rib_l_mani) - disc_cyl

# E. Snout & Central Chin
snout_box = m3d.Manifold.cube([92.0, 65.0, 168.0], center=False).translate([-46.0, -100.0, 1618.0])

# Cheeks:
# - Mandible tube: covers exactly slot 1 and slot 2 (Y <= +10.5 mm)
# - At Z = 1662: steps back to Y = +45.0 mm covering the ENTIRE depression floor and vertical back wall!
# - Rises vertically along rear wall up to brow band!
cheek_poly_yz = np.array([
    [-100.0, 1618.0],
    [  10.5, 1618.0],  # Bottom tube: covers slot 1 and 2
    [  10.5, 1662.0],  # Step up to bottom of depression
    [  45.0, 1662.0],  # Step back to rear wall of depression!
    [  45.0, 1808.6],  # Rear vertical wall up to brow
    [-25.0,  1808.6],  # Brow band
    [-25.0,  1798.0],  # Sub-ocular horizontal shelf
    [-100.0, 1798.0]   # Front cheek flap
], dtype=np.float32)

cs_cheek = m3d.CrossSection([cheek_poly_yz])
if cs_cheek.area() == 0:
    cs_cheek = m3d.CrossSection([cheek_poly_yz[::-1]])

raw_cheek = cs_cheek.extrude(115.0)
m_rc = raw_cheek.to_mesh()
v_rc = m_rc.vert_properties
f_rc = m_rc.tri_verts

v_cr = np.column_stack([v_rc[:, 2] + 5.0, v_rc[:, 0], v_rc[:, 1]]).astype('float32')
f_cr = f_rc.copy()
cheek_r = m3d.Manifold(m3d.Mesh(vert_properties=v_cr, tri_verts=f_cr))
if cheek_r.volume() <= 0:
    f_cr[:, [0, 1]] = f_cr[:, [1, 0]]
    cheek_r = m3d.Manifold(m3d.Mesh(vert_properties=v_cr, tri_verts=f_cr))

v_cl = np.column_stack([-(v_rc[:, 2] + 5.0), v_rc[:, 0], v_rc[:, 1]]).astype('float32')
f_cl = f_cr.copy()
f_cl[:, [0, 1]] = f_cl[:, [1, 0]]
cheek_l = m3d.Manifold(m3d.Mesh(vert_properties=v_cl, tri_verts=f_cl))

cheeks_raw = cheek_r + cheek_l

# Sub-ocular partial white stripe:
# Bounded strictly between |X| = 25.0 mm and |X| = 54.0 mm!
# For |X| < 25.0 mm: SOLID RED cheek flap!
# For |X| > 54.0 mm: SOLID RED cheek corner!
# In the middle (~60% of span): WHITE stripe on the shelf!
suboc_box_r = m3d.Manifold.cube([29.0, 45.0, 16.0], center=False).translate([25.0, -80.0, 1783.0])
suboc_box_l = m3d.Manifold.cube([29.0, 45.0, 16.0], center=False).translate([-54.0, -80.0, 1783.0])
suboc_cutter = suboc_box_r + suboc_box_l

# 3 White Claws on aerator tubes:
# Sits directly on the rounded cheek/mandible bulge above and around the aerator socket!
# Sweeps from the side inwards to X ≈ 20 mm, crossing the aerator center (X = +/- 47.4 mm) by 27 mm!
claws_r = m3d.Manifold()
claws_l = m3d.Manifold()
for zc, yc in [(1666.0, -68.0), (1652.0, -65.0), (1638.0, -62.0)]:
    cr = m3d.Manifold.cube([65.0, 60.0, 7.5], center=True).rotate([15.0, 5.0, -12.0]).translate([48.0, yc, zc])
    cl = m3d.Manifold.cube([65.0, 60.0, 7.5], center=True).rotate([15.0, -5.0, 12.0]).translate([-48.0, yc, zc])
    claws_r = claws_r + cr
    claws_l = claws_l + cl

claws_cutter = claws_r + claws_l

face_cutter = (snout_box + cheeks_raw) - suboc_cutter - claws_cutter

# Hollow Visor Aperture Keep-Out:
visor_clear_h = m3d.Manifold.cube([130.0, 60.0, 13.0], center=False).translate([-65.0, -90.0, 1796.0])
visor_clear_v = m3d.Manifold.cube([   9.0, 50.0, 70.0], center=False).translate([ -4.5, -90.0, 1726.0])
visor_clearance = visor_clear_h + visor_clear_v

total_red_cutter = (fin_cutter + brow_cutter + wings_cutter + rear_markings + face_cutter) - visor_clearance

red_mani = helmet_mani ^ total_red_cutter
white_mani = helmet_mani - total_red_cutter

m_red = red_mani.to_mesh()
mesh_red = trimesh.Trimesh(vertices=m_red.vert_properties[:, :3], faces=m_red.tri_verts, process=False)
mesh_red.visual.vertex_colors = [185, 28, 28, 255]

m_white = white_mani.to_mesh()
mesh_white = trimesh.Trimesh(vertices=m_white.vert_properties[:, :3], faces=m_white.tri_verts, process=False)
mesh_white.visual.vertex_colors = [244, 244, 246, 255]

mouth_mesh.visual.vertex_colors = [20, 20, 22, 255]
ears_mesh.visual.vertex_colors = [240, 240, 244, 255]
aerators_mesh.visual.vertex_colors = [70, 70, 75, 255]
lens_mesh.visual.vertex_colors = [15, 15, 18, 255]

scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, ears_mesh, aerators_mesh, lens_mesh])
center = scene.centroid
scene.camera.z_near = 300.0
scene.camera.z_far = 1500.0

test_views = [
    ("test_01_front_ortho.png", 0.0, 0.0, 820.0, None),
    ("test_02_front_elevated.png", 0.0, 15.0, 820.0, None),
    ("test_03_iso_3_4_left.png", 35.0, 8.0, 820.0, None),
    ("test_04_profile_left.png", 90.0, 0.0, 820.0, None),
    ("test_05_rear_ortho.png", 180.0, 0.0, 820.0, None),
    ("test_06_macro_visor_shelf.png", 0, 0, 0, ([0.0, -75.0, 1785.0], [0.0, -75.0 - 320.0, 1785.0 + 10.0])),
    ("test_07_macro_claws.png", 0, 0, 0, ([40.0, -60.0, 1650.0], [40.0 + 160.0, -60.0 - 240.0, 1650.0 + 30.0])),
    ("test_08_macro_wing.png", 0, 0, 0, ([75.0, 50.0, 1865.0], [75.0 + 320.0, 50.0, 1865.0 + 15.0])),
    ("test_09_macro_rear_disc.png", 0, 0, 0, ([0.0, 205.0, 1825.0], [0.0, 205.0 + 340.0, 1825.0 + 10.0]))
]

for filename, yaw_deg, pitch_deg, dist, macro_spec in test_views:
    if macro_spec is None:
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        cam_x = dist * np.sin(yaw) * np.cos(pitch)
        cam_y = -dist * np.cos(yaw) * np.cos(pitch)
        cam_z = dist * np.sin(pitch)
        cam_pos = np.array([cam_x, cam_y, cam_z]) + center
        look_target = center
    else:
        target_pt, cam_pos_pt = macro_spec
        look_target = np.array(target_pt, dtype=np.float32)
        cam_pos = np.array(cam_pos_pt, dtype=np.float32)
    
    view_vec = look_target - cam_pos
    forward = view_vec / np.linalg.norm(view_vec)
    up = np.array([0, 0, 1.0])
    if abs(forward[2]) > 0.99:
        up = np.array([0, -1.0, 0])
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
    
    raw_data = scene.save_image(resolution=(1200, 1200))
    img = Image.open(BytesIO(raw_data)).convert("RGBA")
    out_path = os.path.join(BASE_DIR, filename)
    img.save(out_path)
    print(f"Rendered: {filename}")

print("All test renders complete!")
