import os
import numpy as np
import trimesh
import manifold3d as m3d
from PIL import Image

print("=== BUILDING ABSOLUTELY FLAWLESS USER THORN HELMET ===")

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
helmet_path = os.path.join(BASE_DIR, r"Diseños 3D\Casco\Phase 2 Main Body.stl")
greeb_dir = os.path.join(BASE_DIR, r"Diseños 3D\Casco\Greeblies")
clones_dir = os.path.join(BASE_DIR, r"Diseños 3D\Casco\Antigravity\05_Cascos_Pintados_Clones")

# 1. Load Base Helmet
print("Loading base helmet...")
base_mesh = trimesh.load(helmet_path)
v_base = base_mesh.vertices.astype('float32')
f_base = base_mesh.faces.astype('uint32')
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))

mouth_mesh = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))
aerators_mesh = trimesh.util.concatenate([aer_l, aer_r])
visor_lens = trimesh.load(os.path.join(clones_dir, "06_Lentes_Visor_Negro.stl"))

# 1. DORSAL FIN CREST (Exact Y-Z Profile, zero stray specks!)
print("1. Dorsal fin with exact profile...")
fin_profile = np.array([
    [-44.0, 1868.0], # Front tip of fin
    [-20.0, 1888.0],
    [ 10.0, 1920.0],
    [ 50.0, 1932.0],
    [ 90.0, 1932.0],
    [136.0, 1924.0], # Rear base of fin
    [136.0, 1944.0], # Rear top of fin
    [ 90.0, 1952.0],
    [ 50.0, 1952.0],
    [ 10.0, 1940.0],
    [-20.0, 1915.0],
    [-44.0, 1885.0], # Front top of fin
    [-44.0, 1868.0]  # Close loop
], dtype=np.float32)
cs_fin = m3d.CrossSection([fin_profile[:-1]])
# Extrude 21mm in X (centered from -10.5 to +10.5)
raw_fin = cs_fin.extrude(21.0).translate([0, 0, -10.5])
m_rf = raw_fin.to_mesh()
vf = m_rf.vert_properties
ff = m_rf.tri_verts
vf_rot = np.column_stack([vf[:, 2], vf[:, 0], vf[:, 1]]).astype('float32')
trif = ff.copy()
fin_box = m3d.Manifold(m3d.Mesh(vert_properties=vf_rot, tri_verts=trif))
if fin_box.volume() <= 0:
    trif[:, [0, 1]] = trif[:, [1, 0]]
    fin_box = m3d.Manifold(m3d.Mesh(vert_properties=vf_rot, tri_verts=trif))

# 2. FOREHEAD BROW BAND & CENTRAL WHITE V-WEDGE
print("2. Forehead brow band with central white V-flare...")
brow_outer = m3d.Manifold.cube([270.0, 110.0, 52.0], center=False).translate([-135.0, -82.0, 1782.0])
wedge_verts = np.array([
    [-14.0, -90.0, 1775.0], [ 14.0, -90.0, 1775.0], [ 14.0,  15.0, 1775.0], [-14.0,  15.0, 1775.0],
    [-31.0, -90.0, 1860.0], [ 31.0, -90.0, 1860.0], [ 31.0,  15.0, 1860.0], [-31.0,  15.0, 1860.0],
], dtype=np.float32)
wedge_faces = np.array([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]
], dtype=np.uint32)
wedge_mani = m3d.Manifold(m3d.Mesh(vert_properties=wedge_verts, tri_verts=wedge_faces))
brow_cutter = brow_outer - wedge_mani

# 3. THOR DOME WINGS (4 BARS OF EQUAL ~11-12mm THICKNESS!)
print("3. Thor dome wings with uniform sleek feather thickness...")
wing_outline = np.array([
    [ 18.0, 1834.0], # Front connection to brow
    [ 34.0, 1834.0], # Bottom of vertical root
    [ 34.0, 1846.0], # Under-cut below Bar 4
    [ 80.0, 1844.0], # Rear tip of Bar 4
    [ 84.0, 1856.0], # Step between 4 and 3
    [ 96.0, 1862.0], # Rear tip of Bar 3
    [101.0, 1874.0], # Step between 3 and 2
    [114.0, 1880.0], # Rear tip of Bar 2
    [119.0, 1892.0], # Step between 2 and 1
    [128.0, 1898.0], # Rear tip of Bar 1 (top feather)
    # Top edge of Bar 1: uniform 11-12mm thickness, parallel to slits!
    [120.0, 1905.0],
    [ 80.0, 1909.0],
    [ 40.0, 1910.0],
    [ 18.0, 1906.0], # Front top corner of Bar 1
    [ 18.0, 1834.0]  # Close loop at front post
], dtype=np.float32)

cs_wing = m3d.CrossSection([wing_outline[:-1]])

# 3 parallel slanted slits:
for d_z in [1894.0, 1878.0, 1862.0]:
    slot_poly = np.array([
        [ 26.0, d_z - 0.10 * ( 26.0) - 2.0],
        [140.0, d_z - 0.10 * (140.0) - 2.0],
        [140.0, d_z - 0.10 * (140.0) + 2.0],
        [ 26.0, d_z - 0.10 * ( 26.0) + 2.0]
    ], dtype=np.float32)
    cs_wing = cs_wing - m3d.CrossSection([slot_poly])

# White gap separating Bar 4 from Brow band behind front post
under_slot = np.array([
    [ 32.0, 1832.0],
    [ 90.0, 1832.0],
    [ 90.0, 1846.0],
    [ 32.0, 1846.0]
], dtype=np.float32)
cs_wing = cs_wing - m3d.CrossSection([under_slot])

raw_wing = cs_wing.extrude(120.0)
m_rw = raw_wing.to_mesh()
vw = m_rw.vert_properties
fw = m_rw.tri_verts

vw_right = np.column_stack([vw[:, 2] + 32.0, vw[:, 0], vw[:, 1]]).astype('float32')
triw_right = fw.copy()
mani_w_right = m3d.Manifold(m3d.Mesh(vert_properties=vw_right, tri_verts=triw_right))
if mani_w_right.volume() <= 0:
    triw_right[:, [0, 1]] = triw_right[:, [1, 0]]
    mani_w_right = m3d.Manifold(m3d.Mesh(vert_properties=vw_right, tri_verts=triw_right))

vw_left = np.column_stack([-(vw[:, 2] + 32.0), vw[:, 0], vw[:, 1]]).astype('float32')
triw_left = triw_right.copy()
triw_left[:, [0, 1]] = triw_left[:, [1, 0]]
mani_w_left = m3d.Manifold(m3d.Mesh(vert_properties=vw_left, tri_verts=triw_left))
wings_cutter = mani_w_right + mani_w_left

# 4. REAR HORIZONTAL BAND & VERTICAL VENT STRIPES
print("4. Rear markings...")
rear_band_box = m3d.Manifold.cube([270.0, 110.0, 26.0], center=False).translate([-135.0, 118.0, 1802.0])
rear_disc_cutter = m3d.Manifold.cylinder(60.0, 24.0, 24.0, 32).rotate([90, 0, 0]).translate([0, 225.0, 1805.0])
rear_band = rear_band_box - rear_disc_cutter

left_stripe = m3d.Manifold.cube([12.0, 65.0, 190.0], center=False).translate([-29.0, 165.0, 1616.0])
right_stripe = m3d.Manifold.cube([12.0, 65.0, 190.0], center=False).translate([17.0, 165.0, 1616.0])
rear_stripes = left_stripe + right_stripe
rear_markings = rear_band + rear_stripes

# 5. NOSE FLAPS, CHEEKS, SNOUT & AERATOR TUBES
print("5. Snout, cheeks, and claw stripes...")
nose_pts = np.array([
    [-15.0, 1705.0], [ 15.0, 1705.0], [ 22.0, 1785.0], [-22.0, 1785.0]
], dtype=np.float32)
cs_nose = m3d.CrossSection([nose_pts])
raw_nose = cs_nose.extrude(60.0).translate([0, 0, -80.0])
m_rn = raw_nose.to_mesh()
vn = m_rn.vert_properties
fn = m_rn.tri_verts
vn_rot = np.column_stack([vn[:, 0], vn[:, 2], vn[:, 1]]).astype('float32')
trin = fn.copy()
nose_flaps_mani = m3d.Manifold(m3d.Mesh(vert_properties=vn_rot, tri_verts=trin))
if nose_flaps_mani.volume() <= 0:
    trin[:, [0, 1]] = trin[:, [1, 0]]
    nose_flaps_mani = m3d.Manifold(m3d.Mesh(vert_properties=vn_rot, tri_verts=trin))

snout_box = m3d.Manifold.cube([92.0, 55.0, 95.0], center=False).translate([-46.0, -97.0, 1620.0])
cheeks_box = m3d.Manifold.cube([200.0, 95.0, 90.0], center=False).translate([-100.0, -75.0, 1655.0])
tubes_box = m3d.Manifold.cube([230.0, 85.0, 50.0], center=False).translate([-115.0, -65.0, 1618.0])

rings_cutter = m3d.Manifold()
for y_center in [-42.0, -20.0, 3.0]:
    rc = m3d.Manifold.cube([250.0, 7.0, 65.0], center=True).rotate([25, 0, 0]).translate([0, y_center, 1640.0])
    rings_cutter = rings_cutter + rc

tubes_with_rings = tubes_box - rings_cutter
face_cutter = nose_flaps_mani + snout_box + cheeks_box + tubes_with_rings

# 6. TOTAL BOOLEAN ASSEMBLY
print("Combining all red cutters...")
total_red_cutter = fin_box + brow_cutter + wings_cutter + rear_markings + face_cutter

red_markings_mani = helmet_mani ^ total_red_cutter
white_base_mani = helmet_mani - red_markings_mani

print(f"Red markings volume: {red_markings_mani.volume():.1f} mm3")
print(f"White base volume: {white_base_mani.volume():.1f} mm3")

# Extract meshes
m_red = red_markings_mani.to_mesh()
mesh_red = trimesh.Trimesh(vertices=m_red.vert_properties, faces=m_red.tri_verts, process=False)
mesh_red.visual.vertex_colors = [190, 15, 25, 255]

m_white = white_base_mani.to_mesh()
mesh_white = trimesh.Trimesh(vertices=m_white.vert_properties, faces=m_white.tri_verts, process=False)
mesh_white.visual.vertex_colors = [245, 245, 248, 255]

mouth_mesh.visual.vertex_colors = [20, 20, 22, 255]
aerators_mesh.visual.vertex_colors = [80, 80, 85, 255]
# Adjust visor lens slightly inward so zero edges peek out
v_vl = visor_lens.vertices.copy()
v_vl[:, 1] += 2.0 # shift 2mm deeper into helmet
visor_lens.vertices = v_vl
visor_lens.visual.vertex_colors = [15, 15, 18, 255]

# Render 4-angle turnaround
print("Rendering flawless 4-angle turnaround...")
angles = [
    ("Front (0 deg)", 0),
    ("3/4 Iso (35 deg)", 35),
    ("Side Profile (90 deg)", 90),
    ("Rear View (180 deg)", 180)
]

scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, aerators_mesh, visor_lens])
center = scene.centroid
cam_dist = 680.0
elev = np.radians(8.0)

imgs = []
for label, yaw_deg in angles:
    yaw = np.radians(yaw_deg)
    cam_x = cam_dist * np.sin(yaw) * np.cos(elev)
    cam_y = -cam_dist * np.cos(yaw) * np.cos(elev)
    cam_z = cam_dist * np.sin(elev)
    cam_pos = np.array([cam_x, cam_y, cam_z])
    
    forward = -cam_pos / np.linalg.norm(cam_pos)
    up = np.array([0, 0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    actual_up = np.cross(right, forward)
    actual_up /= np.linalg.norm(actual_up)
    
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = actual_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = cam_pos + center
    
    scene.camera_transform = c2w
    
    data = scene.save_image(resolution=(800, 800))
    from io import BytesIO
    img = Image.open(BytesIO(data))
    imgs.append(img)

# Combine into 2x2 grid
w, h = 800, 800
canvas = Image.new("RGBA", (w*2, h*2), (255, 255, 255, 255))
canvas.paste(imgs[0], (0, 0))
canvas.paste(imgs[1], (w, 0))
canvas.paste(imgs[2], (0, h))
canvas.paste(imgs[3], (w, h))

out_img_path = os.path.join(BASE_DIR, "perfect_user_thorn_turnaround.png")
canvas.save(out_img_path)
print(f"Turnaround saved to: {out_img_path}")
