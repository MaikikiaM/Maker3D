import os
import numpy as np
import trimesh
import manifold3d as m3d
from PIL import Image

print("=== PROTOTYPING PERFECT COMMANDER THORN CUTTERS ===")

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
helmet_path = os.path.join(BASE_DIR, r"Diseños 3D\Casco\Phase 2 Main Body.stl")
greeb_dir = os.path.join(BASE_DIR, r"Diseños 3D\Casco\Greeblies")

# 1. Load Base Helmet
print("Loading base helmet...")
base_mesh = trimesh.load(helmet_path)
v_base = base_mesh.vertices.astype('float32')
f_base = base_mesh.faces.astype('uint32')
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))

mouth_mesh = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
ear_l = trimesh.load(os.path.join(greeb_dir, "Ear Piece Left.stl"))
ear_r = trimesh.load(os.path.join(greeb_dir, "Ear Piece Right.stl"))
ears_mesh = trimesh.util.concatenate([ear_l, ear_r])
aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))
aerators_mesh = trimesh.util.concatenate([aer_l, aer_r])

# =========================================================================
# 1. DORSAL FIN CREST
# =========================================================================
# The fin starts at Y ~ -55 (front) to Y ~ 170 (rear).
# Fin width is within X in [-17.5, 17.5].
# Front face of fin (Y < -28) is WHITE.
# From Y = -28 to Y = 170, the crest and sides are RED.
print("Generating Dorsal Fin cutter...")
fin_pts = np.array([
    [-28.0, 1895.0],  # Start of red on crest top
    [ 10.0, 1920.0],
    [ 50.0, 1930.0],
    [ 90.0, 1930.0],
    [136.0, 1922.0],
    [168.0, 1905.0],  # Rear base of fin
    [168.0, 1955.0],  # Top rear box corner
    [ 90.0, 1955.0],
    [ 50.0, 1955.0],
    [ 10.0, 1945.0],
    [-28.0, 1925.0],  # Top front corner
    [-28.0, 1895.0]   # Close
], dtype=np.float32)

cs_fin = m3d.CrossSection([fin_pts[:-1]])
# Extrude in X: exactly 35.0 mm centered from -17.5 to +17.5 mm
# This guarantees it NEVER cuts the dome surface beside the fin (outer dome is at |X| > 18.2)
raw_fin = cs_fin.extrude(35.0).translate([0, 0, -17.5])
m_rf = raw_fin.to_mesh()
vf = m_rf.vert_properties
ff = m_rf.tri_verts
vf_rot = np.column_stack([vf[:, 2], vf[:, 0], vf[:, 1]]).astype('float32')
trif = ff.copy()
fin_box = m3d.Manifold(m3d.Mesh(vert_properties=vf_rot, tri_verts=trif))
if fin_box.volume() <= 0:
    trif[:, [0, 1]] = trif[:, [1, 0]]
    fin_box = m3d.Manifold(m3d.Mesh(vert_properties=vf_rot, tri_verts=trif))

# =========================================================================
# 2. FOREHEAD BROW BAND & CENTRAL WHITE V-WEDGE
# =========================================================================
# Brow band goes from Z = 1782 to Z = 1836.
# Y goes from front (Y ~ -90) all the way to Y = 60.0 (stopping right in front of ear at Y=62.5)!
print("Generating Brow Band and Central V-Wedge...")
brow_outer = m3d.Manifold.cube([270.0, 150.0, 54.0], center=False).translate([-135.0, -90.0, 1782.0])

# Central white V-wedge:
# Bottom at Z=1780: X in [-14, 14] (width 28mm)
# Top at Z=1840: X in [-26, 26] (width 52mm)
# Deep in Y to cleanly pierce the brow: Y from -95 to 10
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
brow_cutter = brow_outer - wedge_mani

# =========================================================================
# 3. THOR DOME WINGS (4 SLEEK FEATHERS, 3 PARALLEL SLITS)
# =========================================================================
print("Generating Thor Wings...")
# Wing outline in (Y, Z):
# Root pillar: Y from 20 to 32, Z from 1836 to 1905
slope = -0.10
# Feathers extending back:
# Feather 1: Y from 22 to 126, Z ~ 1905 + slope*Y
# Feather 2: Y from 25 to 112, Z ~ 1891 + slope*Y
# Feather 3: Y from 25 to 98,  Z ~ 1877 + slope*Y
# Feather 4: Y from 25 to 84,  Z ~ 1863 + slope*Y
wing_outline = np.array([
    # Front root pillar anchor to brow (Z = 1836)
    [20.0, 1836.0],
    [32.0, 1836.0],
    # White gap separating Feather 4 from brow band
    [32.0, 1851.0 + slope * 32.0],
    [84.0, 1851.0 + slope * 84.0],  # Rear tip Feather 4
    [88.0, 1863.0 + slope * 88.0],  # Step to Feather 3
    [98.0, 1865.0 + slope * 98.0],  # Rear tip Feather 3
    [102.0, 1877.0 + slope * 102.0], # Step to Feather 2
    [112.0, 1879.0 + slope * 112.0], # Rear tip Feather 2
    [116.0, 1891.0 + slope * 116.0], # Step to Feather 1
    [126.0, 1893.0 + slope * 126.0], # Rear tip Feather 1
    # Top edge of Feather 1
    [120.0, 1905.0 + slope * 120.0],
    [22.0,  1905.0 + slope * 22.0],
    [20.0,  1836.0]
], dtype=np.float32)

cs_wing = m3d.CrossSection([wing_outline[:-1]])

# 3 straight, parallel, uniform white slits of 2.8 mm thickness
for z_base in [1892.0, 1878.0, 1864.0]:
    slit = np.array([
        [ 26.0, z_base + slope *  26.0 - 1.4],
        [135.0, z_base + slope * 135.0 - 1.4],
        [135.0, z_base + slope * 135.0 + 1.4],
        [ 26.0, z_base + slope *  26.0 + 1.4]
    ], dtype=np.float32)
    cs_wing = cs_wing - m3d.CrossSection([slit])

# Extrude wing in X:
# The dome lateral wall at Z ~ 1850-1905 has X between 70 and 105 mm.
# We extrude from X = 25.0 to X = 145.0 (depth 120.0 mm).
raw_wing = cs_wing.extrude(120.0)
m_rw = raw_wing.to_mesh()
vw = m_rw.vert_properties
fw = m_rw.tri_verts

# Right Wing (positive X):
vw_right = np.column_stack([vw[:, 2] + 25.0, vw[:, 0], vw[:, 1]]).astype('float32')
triw_right = fw.copy()
mani_w_right = m3d.Manifold(m3d.Mesh(vert_properties=vw_right, tri_verts=triw_right))
if mani_w_right.volume() <= 0:
    triw_right[:, [0, 1]] = triw_right[:, [1, 0]]
    mani_w_right = m3d.Manifold(m3d.Mesh(vert_properties=vw_right, tri_verts=triw_right))

# Left Wing (negative X, symmetric):
vw_left = np.column_stack([-(vw[:, 2] + 25.0), vw[:, 0], vw[:, 1]]).astype('float32')
triw_left = triw_right.copy()
triw_left[:, [0, 1]] = triw_left[:, [1, 0]]
mani_w_left = m3d.Manifold(m3d.Mesh(vert_properties=vw_left, tri_verts=triw_left))
wings_cutter = mani_w_right + mani_w_left

# =========================================================================
# 4. REAR HORIZONTAL BAND & VERTICAL VENT STRIPES
# =========================================================================
print("Generating Rear Band, Disk Cutout, and Vertical Stripes...")
# Rear band: starts at Y = 92.0 (immediately behind ear at Y=92.1) and wraps to Y = 225.0.
# Height: Z from 1802.0 to 1836.0 (matching the brow band height).
rear_band_box = m3d.Manifold.cube([270.0, 133.0, 34.0], center=False).translate([-135.0, 92.0, 1802.0])

# Circular cutout around the rear disk:
# Disk center: X = 0, Y ~ 212, Z ~ 1815. Radius = 23.0 mm.
rear_disk_cutter = m3d.Manifold.cylinder(60.0, 23.0, 23.0, 64).rotate([90, 0, 0]).translate([0, 230.0, 1815.0])
rear_band = rear_band_box - rear_disk_cutter

# Flanking vertical red stripes along the rear vent ladder:
# Ladder rungs are in X in [-15, 15].
# Raised ribs are in X in [-28, -15] and [15, 28].
# Stripes run from bottom of rear band (Z = 1802) down to bottom rim (Z = 1616).
left_stripe = m3d.Manifold.cube([13.0, 65.0, 186.0], center=False).translate([-28.0, 165.0, 1616.0])
right_stripe = m3d.Manifold.cube([13.0, 65.0, 186.0], center=False).translate([15.0, 165.0, 1616.0])
rear_markings = rear_band + left_stripe + right_stripe

# =========================================================================
# 5. NOSE, CHEEKS, SNOUT & AERATOR TUBES WITH 3 WHITE CLAW STRIPES
# =========================================================================
print("Generating Snout, Cheek Recesses, and Tube Claws...")
# A. Nose / Snout / Chin:
# Central nose flaps and snout: X in [-45, 45], Y in [-97, -60], Z in [1620, 1785]
snout_box = m3d.Manifold.cube([90.0, 37.0, 165.0], center=False).translate([-45.0, -97.0, 1620.0])

# B. Cheek Recesses (under eyes):
# From visor bottom corner down to top of tube, and back to Y = -22 (BEFORE cheek vents).
# Louvers start at Y = -29.9 on the inner part, and outer grille is at Y > -20.
# So Y in [-65, -22], Z in [1700, 1782], X in [-105, 105].
cheeks_box = m3d.Manifold.cube([210.0, 43.0, 82.0], center=False).translate([-105.0, -65.0, 1700.0])

# C. Lower Aerator Tubes:
# Front portion of tubes ONLY: Y from -70.0 to -22.0, Z in [1618, 1675], X in [-115, 115].
# Behind Y = -22.0, the tube under the cheek vents is 100% WHITE!
tubes_box = m3d.Manifold.cube([230.0, 48.0, 57.0], center=False).translate([-115.0, -70.0, 1618.0])

# 3 White Claw Stripes across each tube:
claw_slits = m3d.Manifold()
for y_c in [-54.0, -41.0, -28.0]:
    # Each claw slit is 5.5 mm thick, angled by 22 degrees
    cs = m3d.Manifold.cube([250.0, 5.5, 60.0], center=True).rotate([22, 0, 0]).translate([0, y_c, 1640.0])
    claw_slits = claw_slits + cs

tubes_with_claws = tubes_box - claw_slits

face_cutter = snout_box + cheeks_box + tubes_with_claws

# =========================================================================
# 6. TOTAL BOOLEAN ASSEMBLY
# =========================================================================
print("Assembling total red cutter...")
total_red_cutter = fin_box + brow_cutter + wings_cutter + rear_markings + face_cutter

print("Executing Manifold boolean intersection and difference...")
red_markings_mani = helmet_mani ^ total_red_cutter
white_base_mani = helmet_mani - total_red_cutter

print(f"Red volume: {red_markings_mani.volume():.1f} mm3, tri: {red_markings_mani.num_tri()}")
print(f"White volume: {white_base_mani.volume():.1f} mm3, tri: {white_base_mani.num_tri()}")

# Meshes
m_red = red_markings_mani.to_mesh()
mesh_red = trimesh.Trimesh(vertices=m_red.vert_properties[:, :3], faces=m_red.tri_verts, process=False)
mesh_red.visual.vertex_colors = [185, 28, 28, 255]

m_white = white_base_mani.to_mesh()
mesh_white = trimesh.Trimesh(vertices=m_white.vert_properties[:, :3], faces=m_white.tri_verts, process=False)
mesh_white.visual.vertex_colors = [244, 244, 246, 255]

mouth_mesh.visual.vertex_colors = [20, 20, 22, 255]
ears_mesh.visual.vertex_colors = [240, 240, 244, 255]
aerators_mesh.visual.vertex_colors = [70, 70, 75, 255]

# Render 4 views with supersampling
print("Rendering 4 review angles...")
scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, ears_mesh, aerators_mesh])
center = scene.centroid
cam_dist = 680.0
elev = np.radians(8.0)

angles = [
    ("Front", 0),
    ("Iso_35", 35),
    ("Side_90", 90),
    ("Rear_180", 180)
]

for name, yaw_deg in angles:
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
    
    # Supersampled render at 1600x1600, then downsampled to 800x800 with Lanczos
    raw_bytes = scene.save_image(resolution=(1600, 1600))
    from io import BytesIO
    im = Image.open(BytesIO(raw_bytes)).convert("RGBA")
    im_smooth = im.resize((800, 800), Image.Resampling.LANCZOS)
    im_smooth.save(f"review_{name}.png")
    print(f"Saved review_{name}.png")

print("=== PROTOTYPE COMPLETE ===")
