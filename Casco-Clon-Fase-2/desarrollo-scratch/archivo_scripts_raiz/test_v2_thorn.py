import os
import numpy as np
import trimesh
import manifold3d as m3d
from PIL import Image

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
helmet_path = os.path.join(BASE_DIR, r"Diseños 3D\Casco\Phase 2 Main Body.stl")
greeb_dir = os.path.join(BASE_DIR, r"Diseños 3D\Casco\Greeblies")

base_mesh = trimesh.load(helmet_path)
v_base = base_mesh.vertices.astype('float32')
f_base = base_mesh.faces.astype('uint32')
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))

ear_l = trimesh.load(os.path.join(greeb_dir, "Ear Piece Left.stl"))
ear_r = trimesh.load(os.path.join(greeb_dir, "Ear Piece Right.stl"))
ears_mesh = trimesh.util.concatenate([ear_l, ear_r])
mouth_mesh = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))
aerators_mesh = trimesh.util.concatenate([aer_l, aer_r])

# Wing test:
slope = 0.24  # Positive upward slope!
z_top_root = 1898.0
fw = 8.5   # feather width
sw = 2.8   # slit width
pitch = fw + sw

# Tips Y coordinates:
y_tips = [112.0, 98.0, 84.0, 70.0]

# Construct 4 separate feather polygons:
feathers_cs = m3d.CrossSection()

for i in range(4):
    z_f_top = z_top_root - i * pitch
    z_f_bot = z_f_top - fw
    y_end = y_tips[i]
    y_start = 20.0
    
    # Polygon for feather i:
    poly = np.array([
        [y_start, z_f_top + slope * (y_start - 20.0)],
        [y_end,   z_f_top + slope * (y_end - 20.0)],
        [y_end - 5.0, z_f_bot + slope * (y_end - 5.0 - 20.0)],
        [y_start, z_f_bot + slope * (y_start - 20.0)],
        [y_start, z_f_top + slope * (y_start - 20.0)]
    ], dtype=np.float32)
    feathers_cs = feathers_cs + m3d.CrossSection([poly[:-1]])

# Front root post connecting down to brow band (Z = 1836):
root_poly = np.array([
    [16.0, 1836.0],
    [26.0, 1836.0],
    [26.0, z_top_root],
    [16.0, z_top_root],
    [16.0, 1836.0]
], dtype=np.float32)
cs_wing = feathers_cs + m3d.CrossSection([root_poly[:-1]])

# Extrude wing in X:
raw_wing = cs_wing.extrude(120.0)
m_rw = raw_wing.to_mesh()
vw = m_rw.vert_properties
fw_tri = m_rw.tri_verts

# Right wing:
vw_right = np.column_stack([vw[:, 2] + 25.0, vw[:, 0], vw[:, 1]]).astype('float32')
triw_right = fw_tri.copy()
mani_w_right = m3d.Manifold(m3d.Mesh(vert_properties=vw_right, tri_verts=triw_right))
if mani_w_right.volume() <= 0:
    triw_right[:, [0, 1]] = triw_right[:, [1, 0]]
    mani_w_right = m3d.Manifold(m3d.Mesh(vert_properties=vw_right, tri_verts=triw_right))

# Left wing:
vw_left = np.column_stack([-(vw[:, 2] + 25.0), vw[:, 0], vw[:, 1]]).astype('float32')
triw_left = triw_right.copy()
triw_left[:, [0, 1]] = triw_left[:, [1, 0]]
mani_w_left = m3d.Manifold(m3d.Mesh(vert_properties=vw_left, tri_verts=triw_left))
wings_cutter = mani_w_right + mani_w_left

# Brow cutter:
brow_outer = m3d.Manifold.cube([270.0, 150.0, 54.0], center=False).translate([-135.0, -90.0, 1782.0])
wedge_verts = np.array([
    [-14.0, -95.0, 1780.0], [ 14.0, -95.0, 1780.0], [ 14.0,  15.0, 1780.0], [-14.0,  15.0, 1780.0],
    [-26.0, -95.0, 1840.0], [ 26.0, -95.0, 1840.0], [ 26.0,  15.0, 1840.0], [-26.0,  15.0, 1840.0],
], dtype=np.float32)
wedge_faces = np.array([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]
], dtype=np.uint32)
wedge_mani = m3d.Manifold(m3d.Mesh(vert_properties=wedge_verts, tri_verts=wedge_faces))
# Limit brow Y up to 60.0
brow_limit = m3d.Manifold.cube([280.0, 155.0, 60.0], center=False).translate([-140.0, -95.0, 1780.0]) # Y up to 60.0
brow_cutter = (brow_outer - wedge_mani) ^ brow_limit

# Fin cutter:
fin_cutter = m3d.Manifold.cube([33.6, 193.0, 75.0], center=False).translate([-16.8, -25.0, 1880.0])

# Rear cutter:
rear_band_box = m3d.Manifold.cube([270.0, 133.0, 34.0], center=False).translate([-135.0, 92.0, 1802.0])
rear_disk_cutter = m3d.Manifold.cylinder(60.0, 23.0, 23.0, 64).rotate([90, 0, 0]).translate([0, 230.0, 1815.0])
rear_band = rear_band_box - rear_disk_cutter
left_stripe = m3d.Manifold.cube([13.0, 65.0, 186.0], center=False).translate([-28.0, 165.0, 1616.0])
right_stripe = m3d.Manifold.cube([13.0, 65.0, 186.0], center=False).translate([15.0, 165.0, 1616.0])
rear_markings = rear_band + left_stripe + right_stripe

# Face cutter:
# Entire front face from Y = -97 to Y = -22, Z in [1618, 1785], X in [-110, 110]
face_solid = m3d.Manifold.cube([220.0, 75.0, 168.0], center=False).translate([-110.0, -97.0, 1618.0])

# 3 Claw stripes on each tube ONLY (|X| in [35, 110], Z in [1618, 1665]):
claw_box_r = m3d.Manifold()
claw_box_l = m3d.Manifold()
for y_c in [-54.0, -41.0, -28.0]:
    # Stripe rotated across tube:
    c_r = m3d.Manifold.cube([65.0, 5.5, 55.0], center=True).rotate([22, 0, 0]).translate([65.0, y_c, 1638.0])
    c_l = m3d.Manifold.cube([65.0, 5.5, 55.0], center=True).rotate([22, 0, 0]).translate([-65.0, y_c, 1638.0])
    claw_box_r = claw_box_r + c_r
    claw_box_l = claw_box_l + c_l

face_cutter = face_solid - claw_box_r - claw_box_l

# Assemble:
total_red_cutter = fin_cutter + brow_cutter + wings_cutter + rear_markings + face_cutter

print("Cutting meshes...")
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

scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, ears_mesh, aerators_mesh])
center = scene.centroid
cam_dist = 680.0
elev = np.radians(8.0)

for name, yaw_deg in [("Side_Profile", 90), ("Front_View", 0), ("Iso_35", 35), ("Rear_View", 180)]:
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
    
    raw = scene.save_image(resolution=(1600, 1600))
    from io import BytesIO
    im = Image.open(BytesIO(raw)).convert("RGBA")
    im_smooth = im.resize((800, 800), Image.Resampling.LANCZOS)
    im_smooth.save(f"v2_{name}.png")
    print(f"Saved v2_{name}.png")
