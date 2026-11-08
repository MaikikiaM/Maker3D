import os
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
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

casco_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco"
greeb_dir = os.path.join(casco_dir, "Greeblies")

mesh_red = trimesh.load("test_mesh_red.stl")
mesh_red.visual.vertex_colors = [185, 28, 28, 255]

mesh_white = trimesh.load("test_mesh_white.stl")
mesh_white.visual.vertex_colors = [244, 244, 246, 255]

mouth_mesh = trimesh.load(os.path.join(greeb_dir, "Mouth Center Detail.stl"))
mouth_mesh.visual.vertex_colors = [20, 20, 22, 255]

ear_l = trimesh.load(os.path.join(greeb_dir, "Ear Piece Left.stl"))
ear_r = trimesh.load(os.path.join(greeb_dir, "Ear Piece Right.stl"))
ears_mesh = trimesh.util.concatenate([ear_l, ear_r])
ears_mesh.visual.vertex_colors = [240, 240, 244, 255]

aer_l = trimesh.load(os.path.join(greeb_dir, "Aerator Left - Mesh.stl"))
aer_r = trimesh.load(os.path.join(greeb_dir, "Aerator Right - Mesh.stl"))
aerators_mesh = trimesh.util.concatenate([aer_l, aer_r])
aerators_mesh.visual.vertex_colors = [70, 70, 75, 255]

scene = trimesh.Scene([mesh_white, mesh_red, mouth_mesh, ears_mesh, aerators_mesh])

# Off-screen render helper
def render_angle(yaw_deg, pitch_deg, roll_deg=0, target_z=1780.0, dist=780.0, resolution=(1400, 1400)):
    y_rad = np.radians(yaw_deg)
    p_rad = np.radians(pitch_deg)
    
    # Camera position spherical
    cx = dist * np.sin(y_rad) * np.cos(p_rad)
    cy = -dist * np.cos(y_rad) * np.cos(p_rad)
    cz = target_z + dist * np.sin(p_rad)
    cam_pos = np.array([cx, cy, cz])
    target = np.array([0.0, 50.0, target_z])
    
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    
    rot = np.eye(4)
    rot[:3, 0] = right
    rot[:3, 1] = true_up
    rot[:3, 2] = -forward
    rot[:3, 3] = cam_pos
    
    scene.camera_transform = rot
    data = scene.save_image(resolution=resolution)
    return Image.open(trimesh.util.wrap_as_stream(data))

views = [
    ("VISTA FRONTAL (0°)", 0, 0),
    ("PERSPECTIVA 3/4 IZQ (+35°)", 35, 8),
    ("PERFIL IZQUIERDO (90°)", 90, 0),
    ("VISTA TRASERA (180°)", 180, 5),
]

imgs = []
for name, y, p in views:
    print(f"Renderizando {name}...")
    img = render_angle(y, p)
    imgs.append(img)

# Combine into 2x2 sheet
sheet = Image.new("RGB", (2800, 2800), (255, 255, 255))
sheet.paste(imgs[0], (0, 0))
sheet.paste(imgs[1], (1400, 0))
sheet.paste(imgs[2], (0, 1400))
sheet.paste(imgs[3], (1400, 1400))

sheet.save("test_v13_4views.png")
print("Lámina de prueba guardada en test_v13_4views.png")
