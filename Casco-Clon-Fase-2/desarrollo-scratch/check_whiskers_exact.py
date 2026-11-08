import sys
import numpy as np

sys.path.insert(0, r"c:\Users\migue\Proyectos IA varios\Miscelanea\scratch")
import geo_tools as gt

sam_p, sam_n, _ = gt.get_helmet_samples()
P = sam_p[(sam_p[:, 0] > 35.0) & (sam_p[:, 2] < 1705.0) & (sam_p[:, 1] < 45.0) & ((sam_n[:, 0] > 0.3) | (sam_n[:, 1] < -0.3))]

c = np.array([60.0, -96.0, 1616.0])
scale = np.array([1.0, 1.32, 1.0])
r_eff = np.linalg.norm((P - c) / scale, axis=1)

for r_mid in [50.0, 56.0, 62.0, 68.0, 74.0, 80.0, 86.0]:
    band = P[np.abs(r_eff - r_mid) < 1.5]
    zmax = band[:, 2].max() if len(band) else -1
    b_mid = band[(np.abs(band[:, 2] - 1655.0) < 3.0) & (band[:, 0] > 72.0)]
    b_bot = band[(np.abs(band[:, 2] - 1635.0) < 3.0) & (band[:, 0] > 60.0)]
    ym = np.median(b_mid[:, 1]) if len(b_mid) else np.nan
    yb = np.median(b_bot[:, 1]) if len(b_bot) else np.nan
    slope = (yb - ym) / (-20.0)
    print(f"r={r_mid:4.1f}: Z_max={zmax:6.1f} | Y(Z=1655)={ym:5.1f} | Y(Z=1635)={yb:5.1f} | dY/dZ={slope:.2f}")
