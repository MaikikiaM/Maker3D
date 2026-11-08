import numpy as np
import trimesh
import os

def export_multibody_3mf_from_mesh(mesh_path, output_3mf, is_port=False):
    mesh = trimesh.load(mesh_path)
    verts = mesh.vertices
    faces = mesh.faces
    
    face_centers = np.mean(verts[faces], axis=1)
    fx = face_centers[:, 0]
    fy = face_centers[:, 1]
    fz = face_centers[:, 2]
    
    r_chit = np.hypot(fx - 16.0, fy - (-14.0))
    r_quarry = np.hypot(fx - (-12.0), fy - (-16.0))
    
    # 6 Bodies
    # 0: Base Zocalo (Z <= 25.0)
    # 1: Granite Rock (Main body)
    # 2: Iron Ore Quarry (South flank, rust/orange)
    # 3: Glacial Snow Caps (Z >= 65 or couloirs)
    # 4: Alpine Moss (Low elevation)
    # 5: Chit Plateau & Pocket
    # 6: (If port) Wooden Pier Dock
    
    face_labels = np.zeros(len(faces), dtype=int)
    
    for idx in range(len(faces)):
        z = fz[idx]
        x = fx[idx]
        y = fy[idx]
        rc = r_chit[idx]
        rq = r_quarry[idx]
        
        if z < 24.9:
            face_labels[idx] = 0 # Base Zocalo (Tool B)
        elif is_port and y > 25.0 and x > 20.0 and z <= 28.5:
            face_labels[idx] = 6 # Wooden Pier (Tool O / Brown)
        elif rc <= 15.5:
            face_labels[idx] = 5 # Chit Plateau
        elif z >= 66.0:
            face_labels[idx] = 3 # Snow Caps (Tool W - 0.2mm)
        elif z >= 56.0 and y > 6.0:
            face_labels[idx] = 3 # Glacial couloir
        elif rq <= 20.0 and z < 46.0 and y < -6.0:
            face_labels[idx] = 2 # Iron Ore Quarry (Tool O / Y)
        elif 25.0 <= z <= 27.5 and np.hypot(x, y) > 35.0:
            face_labels[idx] = 4 # Alpine Moss (Tool G)
        else:
            face_labels[idx] = 1 # Granite Rock (Tool T3 / K)
            
    body_names = [
        "Body_0_Base_Zocalo_Tool_B",
        "Body_1_Granite_Rock_Tool_T3_K",
        "Body_2_Iron_Ore_Quarry_Tool_O_Y",
        "Body_3_Glacial_Snow_Caps_Tool_W",
        "Body_4_Alpine_Moss_Tool_G",
        "Body_5_Chit_Plateau_Tool_Neutral"
    ]
    if is_port:
        body_names.append("Body_6_Coastal_Port_Dock_Tool_O")
        
    body_colors = [
        [40, 42, 45, 255],     # Base PLA
        [135, 138, 142, 255],  # Granite Grey
        [205, 95, 40, 255],    # Iron Ore Orange / Rust
        [250, 252, 255, 255],  # Alpine Snow White
        [65, 125, 55, 255],    # Alpine Moss Green
        [160, 155, 150, 255],  # Carved Stone Grey
        [140, 75, 30, 255]     # Wooden Pier Oak Brown
    ]
    
    scene = trimesh.Scene()
    for b_id, name in enumerate(body_names):
        sub_faces = faces[face_labels == b_id]
        if len(sub_faces) == 0:
            continue
        u_v_idx, inv = np.unique(sub_faces, return_inverse=True)
        sub_verts = verts[u_v_idx]
        sub_f = inv.reshape(sub_faces.shape)
        
        sub_mesh = trimesh.Trimesh(vertices=sub_verts, faces=sub_f, process=False)
        sub_mesh.visual.face_colors = np.tile(body_colors[b_id], (len(sub_f), 1))
        scene.add_geometry(sub_mesh, node_name=name)
        
    scene.export(output_3mf)
    print(f"Generated 3MF assembly: {output_3mf}")

if __name__ == '__main__':
    export_multibody_3mf_from_mesh(
        "scratch/production_models/catan_mountain_inland_mag_open.stl",
        "scratch/production_models/catan_mountain_inland_cmykwog.3mf",
        is_port=False
    )
    export_multibody_3mf_from_mesh(
        "scratch/production_models/catan_mountain_coastal_port_mag_open.stl",
        "scratch/production_models/catan_mountain_coastal_cmykwog.3mf",
        is_port=True
    )
