import numpy as np
from PIL import Image

Z = np.load('scratch/Z_top_master_2m.npy')
print("Shape:", Z.shape)
print("Min:", Z.min(), "Max:", Z.max())

# Check for any NaNs or Infs
assert not np.isnan(Z).any(), "NaN found!"
assert not np.isinf(Z).any(), "Inf found!"

# Check spectrum of cross sections to ensure 1/f falloff without sharp spike frequencies (no sines!)
cross_x = Z[Z.shape[0]//2, :]
fft_x = np.abs(np.fft.rfft(cross_x - np.mean(cross_x)))
print("Max FFT peak / median ratio:", np.max(fft_x[5:]) / np.median(fft_x[5:]))
# A pure sine wave would have a peak ratio of 100-1000. Fractal noise is typically 3-8.
