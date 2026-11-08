import numpy as np
import scipy.interpolate as interp
import trimesh
from build_solid_mountain import build_solid_watertight_mountain

R = 64.0
apothem = R * np.cos(np.pi / 6)
Z_master = np.load('scratch/Z_top_master_2m.npy')
ny_m, nx_m = Z_master.shape
xs_m = np.linspace(-apothem, apothem, nx_m, dtype=np.float32)
ys_m = np.linspace(-R, R, ny_m, dtype=np.float32)
interp_z = interp.RegularGridInterpolator((ys_m, xs_m), Z_master, bounds_error=False, fill_value=25.0)

mesh = build_solid_watertight_mountain(res=1710, output_path='scratch/test_flawless_5m.stl', z_interpolator=interp_z)
print("Mesh is_watertight:", mesh.is_watertight)
print("Faces:", len(mesh.faces))
