import os
import numpy as np
import trimesh
import manifold3d as m3d
from PIL import Image

print("=== REFINING CHEEK SCOOP BOUNDARY TO EXACT Y=5.0 ===")

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
helmet_path = os.path.join(BASE_DIR, r"Diseños 3D\Casco\Phase 2 Main Body.stl")
greeb_dir = os.path.join(BASE_DIR, r"Diseños 3D\Casco\Greeblies")

# 1. Load Base Helmet
base_mesh = trimesh.load(helmet_path)
v_base = base_mesh.vertices.astype('float32')
f_base = base_mesh.faces.astype('uint32')
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))

mouth_mesh = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))
aerators_mesh = trimesh.util.concatenate([aer_l, aer_r])

# 1. Dorsal fin
fin_profile = np.array([
    [-46.0, 1865.0], [-20.0, 1888.0], [ 10.0, 1920.0], [ 50.0, 1932.0],
    [ 90.0, 1932.0], [136.0, 1924.0], [136.0, 1950.0], [ 90.0, 1952.0],
    [ 50.0, 1952.0], [ 10.0, 1940.0], [-20.0, 1915.0], [-46.0, 1885.0],
    [-46.0, 1865.0]
], dtype=np.float32)
cs_fin = m3d.CrossSection([fin_profile[:-1]])
raw_fin = cs_fin.extrude(20.0).translate([0, 0, -10.0])
m_rf = raw_fin.to_mesh()
vf = m_rf.vert_properties
ff = m_rf.tri_verts
vf_rot = np.column_stack([vf[:, 2], vf[:, 0], vf[:, 1]]).astype('float32')
trif = ff.copy()
fin_box = m3d.Manifold(m3d.Mesh(vert_properties=vf_rot, tri_verts=trif))
if fin_box.volume() <= 0:
    trif[:, [0, 1]] = trif[:, [1, 0]]
    fin_box = m3d.Manifold(m3d.Mesh(vert_properties=vf_rot, tri_verts=trif))

# 2. Forehead Brow Band with Central White V-Flare
brow_outer = m3d.Manifold.cube([270.0, 106.0, 52.0], center=False).translate([-135.0, -82.0, 1782.0])
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

# 3. Thor Dome Wings
slope = -0.10
wing_outline = np.array([
    [20.0, 1834.0],
    [26.0, 1834.0],
    [26.0, 1857.0 + slope*26.0],
    [82.0, 1857.0 + slope*82.0],
    [82.0, 1867.0 + slope*82.0],
    [98.0, 1869.0 + slope*98.0],
    [98.0, 1879.0 + slope*98.0],
    [112.0, 1881.0 + slope*112.0],
    [112.0, 1891.0 + slope*112.0],
    [125.0, 1893.0 + slope*125.0],
    [125.0, 1903.0 + slope*125.0],
    [22.0, 1903.0 + slope*22.0],
    [20.0, 1834.0]
], dtype=np.float32)

cs_wing = m3d.CrossSection([wing_outline[:-1]])
for d_z in [1892.0, 1880.0, 1868.0]:
    slot_poly = np.array([
        [ 25.0, d_z + slope*( 25.0) - 1.1],
        [135.0, d_z + slope*(135.0) - 1.1],
        [135.0, d_z + slope*(135.0) + 1.1],
        [ 25.0, d_z + slope*( 25.0) + 1.1]
    ], dtype=np.float32)
    cs_wing = cs_wing - m3d.CrossSection([slot_poly])

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

# 4. Rear Markings
rear_band_box = m3d.Manifold.cube([270.0, 110.0, 26.0], center=False).translate([-135.0, 118.0, 1802.0])
rear_disc_cutter = m3d.Manifold.cylinder(60.0, 24.0, 24.0, 32).rotate([90, 0, 0]).translate([0, 225.0, 1805.0])
rear_band = rear_band_box - rear_disc_cutter

left_stripe = m3d.Manifold.cube([12.0, 65.0, 190.0], center=False).translate([-29.0, 165.0, 1616.0])
right_stripe = m3d.Manifold.cube([12.0, 65.0, 190.0], center=False).translate([17.0, 165.0, 1616.0])
rear_stripes = left_stripe + right_stripe
rear_markings = rear_band + rear_stripes

# 5. Continuous Solid Red Face (cheeks, snout, nose flaps, aerator tubes)
# Y from -97 to 5.0 (EXACTLY at rear wall of cheek scoop, 100% protecting ribbed slats!)
# Z from 1616 to 1786 (covering full height of nose flaps and cheeks to visor aperture)
face_solid = m3d.Manifold.cube([196.0, 102.0, 170.0], center=False).translate([-98.0, -97.0, 1616.0])

# 3 slanted claw rings on lower aerator tubes
claw_rings = m3d.Manifold()
for y_center in [-44.0, -26.0, -8.0]:
    rc = m3d.Manifold.cube([250.0, 6.5, 55.0], center=True).rotate([25, 0, 0]).translate([0, y_center, 1640.0])
    claw_rings = claw_rings + rc

face_cutter = face_solid - claw_rings

# 6. TOTAL BOOLEAN
total_red_cutter = fin_box + brow_cutter + wings_cutter + rear_markings + face_cutter

red_markings_mani = helmet_mani ^ total_red_cutter
white_base_mani = helmet_mani - red_markings_mani

print(f"Red markings volume: {red_markings_mani.volume():.1f} mm3")
print(f"White base volume: {white_base_mani.volume():.1f} mm3")

m_red = red_markings_mani.to_mesh()
mesh_red = trimesh.Trimesh(vertices=m_red.vert_properties[:, :3], faces=m_red.tri_verts, process=False)
mesh_red.visual.vertex_colors = [190, 15, 25, 255]

m_white = white_base_mani.to_mesh()
mesh_white = trimesh.Trimesh(vertices=m_white.vert_properties[:, :3], faces=m_white.tri_verts, process=False)
mesh_white.visual.vertex_colors = [245, 245, 248, 255]

mouth_mesh.visual.vertex_colors = [20, 20, 22, 255]
aerators_mesh.visual.vertex_colors = [80, 80, 85, 255]

# Render perspective view WITHOUT VISOR
scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, aerators_mesh])
center = scene.centroid
cam_dist = 680.0
elev = np.radians(8.0)
yaw = np.radians(35.0)

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
out_test = os.path.join(BASE_DIR, "test_clean_face_novisor.png")
img.save(out_test)
print(f"Saved refined clean face image to {out_test}")
