import numpy as np
from PIL import Image

Z = np.load('scratch/Z_top_master_2m.npy')
print('Z shape:', Z.shape, 'min:', Z.min(), 'max:', Z.max())

# Compute hillshade of Z
ny, nx = Z.shape
gy, gx = np.gradient(Z)
slope = np.hypot(gx, gy)
aspect = np.arctan2(gy, -gx)

az = 315.0 * np.pi / 180.0
alt = 45.0 * np.pi / 180.0
shaded = np.sin(alt) + np.cos(alt) * slope * np.cos(az - aspect)
shaded = (shaded - shaded.min()) / (shaded.max() - shaded.min())

im = Image.fromarray((shaded * 255.0).astype(np.uint8))
im.save('scratch/inspect_Z_hillshade.png')
print('Saved scratch/inspect_Z_hillshade.png')
