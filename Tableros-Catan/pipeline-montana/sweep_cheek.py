import os
import sys
import numpy as np
import trimesh
import manifold3d as m3d
from PIL import Image

sys.path.append(r"C:\Users\migue\.gemini\antigravity\brain\708952eb-13ab-449a-b9c1-cf788f04feb2\scratch")
from test_cheek_setup import *

# Let's inspect the cheek depression span in Y around Z=1710:
# The depression floor/sill begins around Y = -22 to -20 and ends at Y = 55 to 72.
# Let's test a range of y_top around 35%:
# If Y_front = -20, Y_rear = 55 (span 75 mm): 35% is Y = -20 + 26.25 = +6.25 mm.
# If Y_front = -25, Y_rear = 55 (span 80 mm): 35% is Y = -25 + 28.0 = +3.0 mm.
# Let's test y_top in [0.0, 3.0, 6.0, 9.0] mm.

# And for each y_top, let's test delta_Y = y_bottom - y_top in [35.0, 45.0, 55.0, 65.0]:
results = []
test_configs = [
    # (tag, y_bottom, y_top)
    ("sweep_1", 45.0, 5.0),
    ("sweep_2", 50.0, 5.0),
    ("sweep_3", 55.0, 5.0),
    ("sweep_4", 60.0, 5.0),
    ("sweep_5", 48.0, 3.0),
    ("sweep_6", 52.0, 3.0),
    ("sweep_7", 40.0, 0.0),
    ("sweep_8", 45.0, 0.0),
]

for tag, y_b, y_t in test_configs:
    cheek_pts = [
        [-110.0, 1600.0],
        [y_b, 1600.0],
        [y_t, 1710.0],
        [72.0, 1710.0],
        [72.0, 1780.0],
        [55.0, 1812.0],
        [-110.0, 1812.0]
    ]
    cs_ch = make_ccw(cheek_pts)
    raw_ch = cs_ch.extrude(120.0)
    mat_ch_r = np.array([[0.0, 0.0, 1.0, 8.85], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    mat_ch_l = np.array([[0.0, 0.0, -1.0, -8.85], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    cheeks_raw = raw_ch.transform(mat_ch_r) + raw_ch.transform(mat_ch_l)

    face_solid = cheeks_raw + snout_box + nose_teeth_bridge
    face_cutter = (face_solid - claws_raw) - suboc_cutter
    total_red_cutter = (fin_cutter + brow_cutter + wings_cutter + rear_markings + face_cutter) - ear_exclusion

    red_markings_mani = helmet_mani ^ total_red_cutter
    white_base_mani = helmet_mani - total_red_cutter

    mesh_red = trimesh.Trimesh(vertices=red_markings_mani.to_mesh().vert_properties[:, :3],
                               faces=red_markings_mani.to_mesh().tri_verts, process=False)
    mesh_red.visual.vertex_colors = [185, 28, 28, 255]

    mesh_white = trimesh.Trimesh(vertices=white_base_mani.to_mesh().vert_properties[:, :3],
                                 faces=white_base_mani.to_mesh().tri_verts, process=False)
    mesh_white.visual.vertex_colors = [245, 245, 245, 255]

    scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, ears_mesh, aerators_mesh])
    center = scene.centroid
    scene.camera.z_near = 10.0
    scene.camera.z_far = 2500.0

    yaw = np.radians(90.0)
    pitch = np.radians(0.0)
    dist = 540.0
    cam_pos = np.array([dist * np.sin(yaw) * np.cos(pitch), -dist * np.cos(yaw) * np.cos(pitch), dist * np.sin(pitch)]) + center
    view_vec = center - cam_pos
    forward = view_vec / np.linalg.norm(view_vec)
    up = np.array([0, 0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    actual_up = np.cross(right, forward)
    actual_up /= np.linalg.norm(actual_up)

    c2w = np.eye(4)
    c2w[:3, 0] = right; c2w[:3, 1] = actual_up; c2w[:3, 2] = -forward; c2w[:3, 3] = cam_pos
    scene.camera_transform = c2w

    data = scene.save_image(resolution=(1024, 1024))
    out_path = f"sweep_{tag}.png"
    with open(out_path, 'wb') as f:
        f.write(data)
    
    img = Image.open(out_path)
    w, h = img.size
    crop = img.crop((0, int(h * 0.4), int(w * 0.7), h))
    crop_path = f"crop_{tag}.png"
    crop.save(crop_path)

    # Analyze parallelism: measure distance between Whisker 3 and Red border at y=360, 390, 420
    crop_arr = np.array(crop)
    distances = []
    for test_y in [360, 390, 420]:
        xs = [x for x in range(100, 300) if crop_arr[test_y, x, 0] > 220 and crop_arr[test_y, x, 1] > 220 and crop_arr[test_y, x, 2] > 220]
        clusters = []
        if xs:
            c = [xs[0]]
            for x in xs[1:]:
                if x == c[-1] + 1: c.append(x)
                else: clusters.append(c); c = [x]
            clusters.append(c)
            w3_x = np.mean(clusters[-1])
        else:
            w3_x = np.nan
        red_xs = [x for x in range(150, 450) if crop_arr[test_y, x, 0] > 180 and crop_arr[test_y, x, 1] < 60 and crop_arr[test_y, x, 2] < 60]
        border_x = max(red_xs) if red_xs else np.nan
        distances.append(border_x - w3_x)
    
    d_range = max(distances) - min(distances) if not any(np.isnan(distances)) else 999
    print(f"Config {tag} (yb={y_b}, yt={y_t}): distances at [360, 390, 420]={distances}, variation={d_range:.1f}px")
    results.append((d_range, tag, y_b, y_t, distances))

results.sort(key=lambda x: x[0])
print("\n=== TOP PARALLEL CONFIGURATIONS ===")
for r in results[:4]:
    print(f"Rank: var={r[0]:.1f}px | tag={r[1]} | y_bottom={r[2]} | y_top={r[3]} | dists={[round(d,1) for d in r[4]]}")
