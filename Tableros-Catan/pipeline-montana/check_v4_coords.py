import sys
import numpy as np

sys.path.insert(0, r"c:\Users\migue\Proyectos IA varios\Miscelanea\scratch")
import geo_tools as gt

sam_p, sam_n, _ = gt.get_helmet_samples()

# 1. Check jaw tube outer flank (X > 45, Z in 1616..1710)
print("=== 1. JAW TUBE SURFACE EXTENTS ===")
for zv in [1620, 1630, 1645, 1660, 1675, 1690, 1705]:
    m = (sam_p[:, 0] > 45.0) & (np.abs(sam_p[:, 2] - zv) < 1.5) & (sam_p[:, 1] < 80.0) & (sam_n[:, 0] > 0.2)
    pts = sam_p[m]
    if len(pts):
        print(f"Z={zv}: Y in [{pts[:,1].min():.1f}, {pts[:,1].max():.1f}], X_max={pts[:,0].max():.1f}")

# Check where the raised vertical ear-rib / collar starts on the tube (Y around 45..70)
print("\n=== 2. JAW TUBE REAR COLLAR / RIBS ===")
fe_data = np.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\scratch\helmet_feature_edges.npz")
print("fe keys:", list(fe_data.keys()))
fe_pts = fe_data[list(fe_data.keys())[0]]
if fe_pts.ndim == 3:
    fe_mid = fe_pts.mean(axis=1)
else:
    fe_mid = fe_pts
m_fe = (fe_mid[:, 0] > 70.0) & (fe_mid[:, 2] < 1715.0) & (fe_mid[:, 1] > -10.0) & (fe_mid[:, 1] < 75.0)
for zv in [1625, 1640, 1655, 1670, 1685, 1700]:
    sub = fe_mid[m_fe & (np.abs(fe_mid[:, 2] - zv) < 4.0)]
    if len(sub):
        ys = np.sort(np.round(sub[:, 1], 1))
        print(f"Z={zv}: feature Ys = {np.percentile(ys, [5, 25, 50, 75, 95]).round(1)}")

# 3. Check whisker sphere center candidates on outer jaw tube points
print("\n=== 3. WHISKER SHELL PROJECTION ON OUTER TUBE ===")
tube_outer = sam_p[(sam_p[:, 0] > 48.0) & (sam_p[:, 2] < 1705.0) & (sam_p[:, 1] < 45.0) & ((sam_n[:, 0] > 0.35) | (sam_n[:, 1] < -0.35))]
for cx, cy, cz in [(66.0, -112.0, 1602.0), (68.0, -118.0, 1595.0), (65.0, -108.0, 1608.0)]:
    c = np.array([cx, cy, cz])
    d = np.linalg.norm(tube_outer - c, axis=1)
    # Check Y position at Z=1655 (mid-flank) for several radii
    mid_m = (np.abs(tube_outer[:, 2] - 1655.0) < 2.0) & (tube_outer[:, 0] > 70.0)
    pts_mid = tube_outer[mid_m]
    d_mid = np.linalg.norm(pts_mid - c, axis=1)
    order = np.argsort(pts_mid[:, 1])
    print(f"\nCenter ({cx}, {cy}, {cz}) at Z=1655 (outer flank X>70):")
    for idx in np.linspace(0, len(order)-1, 7, dtype=int):
        p = pts_mid[order[idx]]
        r = d_mid[order[idx]]
        # also slope dY/dZ on cylinder X~85: (Z - cz) / (Y - cy) -> dY/dZ = -(Z - cz)/(Y - cy)
        dydz = -(p[2] - cz) / (p[1] - cy)
        print(f"  Y={p[1]:6.1f}, X={p[0]:5.1f} -> R={r:5.1f}, local dY/dZ={dydz:.2f}")

# 4. Check dome UV mapping smoothness
print("\n=== 4. DOME FOREHEAD UV MAPPING ===")
Y_DOME_C = 35.0
R_REF = 90.0
outer_dome = sam_p[(sam_p[:, 0] > 5.0) & (sam_p[:, 2] > 1804.0) & ((sam_n[:, 0] > 0.15) | (sam_n[:, 1] < -0.15))]
for zv in [1808, 1815, 1822, 1830, 1838, 1846, 1853, 1865, 1875]:
    s = outer_dome[np.abs(outer_dome[:, 2] - zv) < 1.5]
    # print u = R_REF * atan2(y - 35, x) for various x or y
    phis = np.arctan2(s[:, 1] - Y_DOME_C, s[:, 0])
    us = R_REF * phis
    order = np.argsort(us)
    picks = np.linspace(0, len(order)-1, 6, dtype=int)
    row = " | ".join([f"u={us[order[i]]:6.1f}(X={s[order[i],0]:4.1f},Y={s[order[i],1]:5.1f})" for i in picks])
    print(f"Z={zv}: {row}")
