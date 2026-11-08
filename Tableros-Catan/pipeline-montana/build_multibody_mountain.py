import numpy as np
import trimesh
from scipy.spatial import Delaunay
import os

def build_multibody_mountain(output_3mf="scratch/catan_mountain_cmykwog.3mf"):
    # Load the solid master mesh
    solid_mesh = trimesh.load("scratch/catan_mountain_solid.stl")
    
    # Let's extract vertex coordinates and faces
    verts = solid_mesh.vertices
    faces = solid_mesh.faces
    
    # Compute face centers and face heights
    face_centers = np.mean(verts[faces], axis=1)
    fx = face_centers[:, 0]
    fy = face_centers[:, 1]
    fz = face_centers[:, 2]
    
    # Face classification into 6 bodies for CMYKWOG:
    # 0: Base Zocalo (Z <= 25.0 mm and side/bottom walls)
    # 1: Granite Mountain (Main rock body)
    # 2: Iron Ore & Quarry (South-West flank, stepped benches, rust/orange)
    # 3: Glacial Snow Caps (Z >= 65 mm or high ridges)
    # 4: Alpine Moss (Z between 25.0 and 28.0 mm near edges)
    # 5: Number Chit Plateau & Pocket
    
    # Chit pocket center (16.0, -14.0)
    r_chit = np.hypot(fx - 16.0, fy - (-14.0))
    
    # Quarry center (-12.0, -16.0)
    r_quarry = np.hypot(fx - (-12.0), fy - (-16.0))
    
    face_labels = np.zeros(len(faces), dtype=int)
    
    for idx in range(len(faces)):
        z = fz[idx]
        x = fx[idx]
        y = fy[idx]
        rc = r_chit[idx]
        rq = r_quarry[idx]
        
        # Base zocalo: bottom face or vertical wall below 25mm
        if z < 24.9:
            face_labels[idx] = 0 # Base Zocalo (Tool B)
        elif rc <= 15.5:
            face_labels[idx] = 5 # Chit Plateau (Tool T3/Neutral)
        elif z >= 66.0:
            face_labels[idx] = 3 # Snow Caps (Tool W - 0.2mm nozzle)
        elif z >= 56.0 and y > 6.0:
            face_labels[idx] = 3 # Glacial couloir
        elif rq <= 20.0 and z < 46.0 and y < -6.0:
            face_labels[idx] = 2 # Iron Ore Quarry (Tool O / Y)
        elif 25.0 <= z <= 27.5 and np.hypot(x, y) > 35.0:
            face_labels[idx] = 4 # Alpine Moss (Tool G)
        else:
            face_labels[idx] = 1 # Granite Rock (Tool T3 / K)
            
    # Create sub-meshes for each body
    body_names = [
        "Body_0_Base_Zocalo_Tool_B",
        "Body_1_Granite_Rock_Tool_T3_K",
        "Body_2_Iron_Ore_Quarry_Tool_O_Y",
        "Body_3_Glacial_Snow_Caps_Tool_W",
        "Body_4_Alpine_Moss_Tool_G",
        "Body_5_Chit_Plateau_Tool_Neutral"
    ]
    
    body_colors = [
        [40, 42, 45, 255],     # Dark charcoal / Base PLA
        [135, 138, 142, 255],  # Granite Grey
        [205, 95, 40, 255],    # Iron Ore Orange / Rust
        [250, 252, 255, 255],  # Alpine Snow White
        [65, 125, 55, 255],    # Alpine Moss Green
        [160, 155, 150, 255]   # Carved Stone Grey
    ]
    
    scene = trimesh.Scene()
    parts_dir = "scratch/catan_mountain_parts"
    os.makedirs(parts_dir, exist_ok=True)
    
    print("Multi-body classification breakdown:")
    for b_id, name in enumerate(body_names):
        sub_faces = faces[face_labels == b_id]
        if len(sub_faces) == 0:
            continue
        # Extract unique vertices for sub-mesh
        u_v_idx, inv = np.unique(sub_faces, return_inverse=True)
        sub_verts = verts[u_v_idx]
        sub_f = inv.reshape(sub_faces.shape)
        
        sub_mesh = trimesh.Trimesh(vertices=sub_verts, faces=sub_f, process=False)
        sub_mesh.visual.face_colors = np.tile(body_colors[b_id], (len(sub_f), 1))
        
        # Save individual STL
        stl_path = os.path.join(parts_dir, f"{name}.stl")
        sub_mesh.export(stl_path)
        
        scene.add_geometry(sub_mesh, node_name=name)
        print(f"  {name}: {len(sub_f)} faces -> {stl_path}")
        
    scene.export(output_3mf)
    print(f"\nSuccessfully generated full-color 3MF assembly: {output_3mf}")

if __name__ == '__main__':
    build_multibody_mountain()
