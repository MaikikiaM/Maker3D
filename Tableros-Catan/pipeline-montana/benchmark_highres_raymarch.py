import os
import math
import time
import numpy as np

print("=== RUNNING ADVANCED RAY-MARCHING AMBIENT OCCLUSION BENCHMARK ===")
t0 = time.time()

Z_master = np.load('scratch/Z_top_master_2m.npy')
ny, nx = Z_master.shape
print(f"Topography grid: {nx} x {ny} ({nx*ny:,} points)")

# Multi-directional ambient horizon occlusion benchmark across 16 sample directions
# This simulates hemispherical ambient sky visibility (AO) by ray-marching in 16 azimuthal directions
angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
R = 64.0; APOTHEM = R * math.cos(math.radians(30))
dx = 2 * APOTHEM / (nx - 1)

ao_acc = np.zeros((ny, nx), dtype=np.float32)

for a_idx, ang in enumerate(angles):
    t_dir = time.time()
    cos_a = math.sin(ang)
    sin_a = math.cos(ang)
    
    # Max horizon angle search over 32 steps
    max_tan = np.zeros((ny, nx), dtype=np.float32)
    for step in range(1, 33):
        ox = int(round(step * 2.0 * cos_a))
        oy = int(round(step * 2.0 * sin_a))
        
        ys_src = np.clip(np.arange(ny) + oy, 0, ny - 1)
        xs_src = np.clip(np.arange(nx) + ox, 0, nx - 1)
        
        sample_z = Z_master[np.ix_(ys_src, xs_src)]
        dist_world = step * 2.0 * dx
        
        tan_elev = (sample_z - Z_master) / dist_world
        max_tan = np.maximum(max_tan, tan_elev)
        
    horizon_angle = np.arctan(np.maximum(0.0, max_tan))
    sky_vis = 1.0 - horizon_angle / (math.pi / 2.0)
    ao_acc += sky_vis
    print(f"  Direction {a_idx+1}/16 (Azimuth {math.degrees(ang):.1f}°) computed in {time.time()-t_dir:.2f}s")

ao_final = ao_acc / 16.0
print(f"Ray-marching benchmark finished in {time.time()-t0:.2f}s.")
print(f"Mean sky visibility: {np.mean(ao_final):.3f} | Min: {np.min(ao_final):.3f} | Max: {np.max(ao_final):.3f}")

np.save('scratch/ao_raymarched_16dir.npy', ao_final)
print("Saved scratch/ao_raymarched_16dir.npy")
