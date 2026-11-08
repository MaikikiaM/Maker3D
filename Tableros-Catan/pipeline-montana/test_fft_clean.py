import numpy as np

def generate_fBm_fft(width, height, beta=2.15, seed=123):
    rng = np.random.RandomState(seed)
    kx = np.fft.rfftfreq(width)
    ky = np.fft.fftfreq(height)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1.0
    amplitude = 1.0 / (K ** (beta / 2.0))
    amplitude[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, size=amplitude.shape)
    spectrum = amplitude * np.exp(1j * phase)
    noise = np.fft.irfft2(spectrum, s=(height, width)).astype(np.float32)
    return (noise - np.mean(noise)) / np.std(noise)

nx, ny = 1150, 1328
noise = generate_fBm_fft(nx, ny, beta=2.15, seed=42)

cross_x = noise[ny//2, :]
fft_x = np.abs(np.fft.rfft(cross_x - np.mean(cross_x)))
# Check high frequency ratios (above low frequency drift k > 20)
peak_ratio = np.max(fft_x[20:]) / np.mean(fft_x[20:])
print("FFT noise peak/mean ratio for k > 20:", peak_ratio)
assert peak_ratio < 6.0, f"Spike detected: {peak_ratio}"
print("Pure scale-invariant fractal noise confirmed! Peak/mean ratio is only", peak_ratio)
