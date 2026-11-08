import os
import numpy as np
import trimesh
from PIL import Image, ImageDraw

base_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
out_05 = os.path.join(base_dir, r"Diseños 3D\Casco\Antigravity\05_Cascos_Pintados_Clones")
greeb_dir = os.path.join(base_dir, r"Diseños 3D\Casco\Greeblies")

mesh_white = trimesh.load(os.path.join(out_05, "01_Casco_Base_Blanco.stl"))
mesh_red = trimesh.load(os.path.join(out_05, "02_Patron_Rojo_Thorn.stl"))
mouth_mesh = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
ear_l = trimesh.load(os.path.join(greeb_dir, "Ear Piece Left.stl"))
ear_r = trimesh.load(os.path.join(greeb_dir, "Ear Piece Right.stl"))
ears_mesh = trimesh.util.concatenate([ear_l, ear_r])
aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))
aerators_mesh = trimesh.util.concatenate([aer_l, aer_r])

mesh_red.visual.vertex_colors = [185, 28, 28, 255]
mesh_white.visual.vertex_colors = [244, 244, 246, 255]
mouth_mesh.visual.vertex_colors = [20, 20, 22, 255]
ears_mesh.visual.vertex_colors = [240, 240, 244, 255]
aerators_mesh.visual.vertex_colors = [70, 70, 75, 255]

# Render without lens_mesh!
scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, ears_mesh, aerators_mesh])
center = scene.centroid
scene.camera.z_near = 10.0
scene.camera.z_far = 2500.0

angles = [
    ("VISTA FRONTAL (0°)", 0.0, 0.0),
    ("PERSPECTIVA 3/4 (+35°)", 35.0, 8.0),
    ("PERFIL LATERAL (90°)", 90.0, 0.0),
    ("VISTA TRASERA (180°)", 180.0, 0.0)
]

panel_imgs = []
dist = 820.0

for title, yaw_deg, pitch_deg in angles:
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    cam_x = dist * np.sin(yaw) * np.cos(pitch)
    cam_y = -dist * np.cos(yaw) * np.cos(pitch)
    cam_z = dist * np.sin(pitch)
    cam_pos = np.array([cam_x, cam_y, cam_z]) + center
    look_target = center
    
    view_vec = look_target - cam_pos
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
    
    data = scene.save_image(resolution=(800, 800))
    from io import BytesIO
    p_img = Image.open(BytesIO(data)).convert("RGBA")
    
    panel = Image.new("RGBA", (800, 880), (255, 255, 255, 255))
    panel.paste(p_img, (0, 10), p_img)
    draw = ImageDraw.Draw(panel)
    draw.text((400, 835), title, fill=(20, 20, 25, 255), anchor="mm")
    panel_imgs.append(panel)

sheet = Image.new("RGBA", (1640, 1890), (245, 246, 248, 255))
sheet.paste(panel_imgs[0], (20, 130))
sheet.paste(panel_imgs[1], (840, 130))
sheet.paste(panel_imgs[2], (20, 1010))
sheet.paste(panel_imgs[3], (840, 1010))
sheet.save(os.path.join(base_dir, "test_render_novisor.png"))
print("Saved test_render_novisor.png successfully!")
