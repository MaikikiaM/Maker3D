import math
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

def generate_spatial_noise_grid(nx, ny, grid_w, grid_h, seed=123):
    rng = np.random.RandomState(seed)
    white = rng.randn(grid_h, grid_w).astype(np.float32)
    zoom_y = ny / grid_h
    zoom_x = nx / grid_w
    layer = ndi.zoom(white, (zoom_y, zoom_x), order=3)[:ny, :nx]
    return (layer - np.mean(layer)) / np.std(layer)

def generate_domain_warped_ridged_mf(nx, ny, seed=999):
    # Mid-frequency octaves
    n2 = generate_spatial_noise_grid(nx, ny, 16, 18, seed=seed+3)
    n3 = generate_spatial_noise_grid(nx, ny, 32, 36, seed=seed+4)
    n4 = generate_spatial_noise_grid(nx, ny, 64, 72, seed=seed+5)
    n5 = generate_spatial_noise_grid(nx, ny, 128, 144, seed=seed+6)
    
    # Ridged multifractal: max(0, 1 - |n|)^2
    r2 = np.maximum(0.0, 1.0 - np.abs(n2))**2.0
    r3 = np.maximum(0.0, 1.0 - np.abs(n3))**2.0
    r4 = np.maximum(0.0, 1.0 - np.abs(n4))**2.0
    r5 = np.maximum(0.0, 1.0 - np.abs(n5))**2.0
    
    crags = 1.8 * r2 + 1.0 * r3 + 0.5 * r4 + 0.25 * r5
    crags = (crags - np.mean(crags)) / np.std(crags)
    return crags

nx, ny = 1150, 1328
crags = generate_domain_warped_ridged_mf(nx, ny, seed=777)

cross_x = crags[ny//2, :]
fft_x = np.abs(np.fft.rfft(cross_x - np.mean(cross_x)))
peak_ratio = np.max(fft_x[5:]) / np.median(fft_x[5:])
print("Domain-warped ridged MF Peak/median ratio:", peak_ratio)
assert peak_ratio < 20.0, f"Too peaky: {peak_ratio}"
print("PASSED! Peak ratio is only", peak_ratio)
