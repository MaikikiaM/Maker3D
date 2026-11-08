import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from test_alpine_gen import generate_alpine_mountain

X, Y, Z_top, mask_hex, apothem, R, z_base = generate_alpine_mountain(res=181)

# Mask outside hex with NaN
Z_plot = np.copy(Z_top)
Z_plot[~mask_hex] = np.nan

fig = plt.figure(figsize=(10, 8), facecolor='#090e1a')
ax = fig.add_subplot(111, projection='3d', facecolor='#090e1a')

# Plot wireframe/surface
surf = ax.plot_surface(X, Y, Z_plot, cmap='terrain', vmin=25, vmax=83,
                       linewidth=0, antialiased=True, rcount=140, ccount=140)

ax.view_init(elev=38, azim=215)
ax.set_axis_off()
plt.tight_layout()
plt.savefig('scratch/test_alpine_preview.png', dpi=150, facecolor='#090e1a')
print('Preview saved to scratch/test_alpine_preview.png')
