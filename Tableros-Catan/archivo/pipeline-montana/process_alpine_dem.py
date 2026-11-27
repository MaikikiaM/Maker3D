import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# Load DEM
dem = np.load('scratch/matterhorn_dem_crop.npy')
H, W = dem.shape

# Resample to 301x301
from scipy.ndimage import zoom
dem_resampled = zoom(dem, 301.0 / H, order=3)
dem_min = 2300.0
dem_norm = np.clip((dem_resampled - dem_min) / (dem_resampled.max() - dem_min), 0.0, 1.0)

# Alpine Ridge Sharpening (Unsharp Masking + Power contrast)
# High-pass filter to accentuate granite arêtes and crags
blurred = ndi.gaussian_filter(dem_norm, sigma=2.0)
high_pass = dem_norm - blurred
sharp_dem = dem_norm + 1.6 * high_pass
sharp_dem = np.clip(sharp_dem, 0.0, 1.0)

# Non-linear alpine profile (steepens peaks, carves glacial valleys)
sharp_dem = np.power(sharp_dem, 1.25)
sharp_dem = sharp_dem / sharp_dem.max()

# Radial falloff to fit hexagon cleanly
y_idx, x_idx = np.ogrid[:301, :301]
cy, cx = 150.0, 150.0
r_dist = np.hypot(x_idx - cx, y_idx - cy)
falloff = np.clip(1.0 - (r_dist / 142.0)**3.5, 0.0, 1.0)
alpine_relief = sharp_dem * falloff * 58.0 # 58mm target relief

# Slopes & Curvature
gy, gx = np.gradient(alpine_relief)
slope = np.hypot(gx, gy)
laplacian = ndi.laplace(alpine_relief)

print("Alpine processing complete:")
print("  Relief min:", alpine_relief.min(), "max:", alpine_relief.max())
print("  Max slope:", slope.max())
print("  Sharp crags count:", np.sum(laplacian < -0.5))

np.save('scratch/matterhorn_alpine_relief.npy', alpine_relief)
