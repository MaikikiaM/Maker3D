import numpy as np

d = np.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\scratch\helmet_samples.npz")
sam_p, sam_n = d["p"], d["n"]

# 1. Ribs: |X| <= 36, Z in [1660, 1820], Y > 175
m1 = (np.abs(sam_p[:, 0]) <= 36.0) & (sam_p[:, 1] > 175.0) & (sam_p[:, 2] >= 1660.0) & (sam_p[:, 2] <= 1820.0)
p1, n1 = sam_p[m1], sam_n[m1]
pi1, po1 = p1[n1[:, 1] < -0.2], p1[n1[:, 1] > 0.2]
g1 = []
for xv in np.linspace(-35, 35, 36):
    mix, mox = np.abs(pi1[:, 0] - xv) < 1.2, np.abs(po1[:, 0] - xv) < 1.2
    for zv in np.linspace(1665, 1818, 32):
        si = pi1[mix & (np.abs(pi1[:, 2] - zv) < 2.8), 1]
        so = po1[mox & (np.abs(po1[:, 2] - zv) < 2.8), 1]
        if len(si) and len(so):
            g1.append(np.percentile(so, 2) - np.percentile(si, 98))
print("Ribs 2D grid min_gap:", min(g1), "median:", np.median(g1))

# 2. Band: Z in [1822, 1856], Y > 130, cylindrical around (0, 80)
Y_C = 80.0
m2 = (sam_p[:, 1] > 130.0) & (sam_p[:, 2] >= 1822.0) & (sam_p[:, 2] <= 1856.0)
p2, n2 = sam_p[m2], sam_n[m2]
dot = p2[:, 0] * n2[:, 0] + (p2[:, 1] - Y_C) * n2[:, 1]
pi2, po2 = p2[dot < -15.0], p2[dot > 15.0]
phi_i, ri = np.arctan2(pi2[:, 0], pi2[:, 1] - Y_C), np.hypot(pi2[:, 0], pi2[:, 1] - Y_C)
phi_o, ro = np.arctan2(po2[:, 0], po2[:, 1] - Y_C), np.hypot(po2[:, 0], po2[:, 1] - Y_C)
g2 = []
for ph in np.linspace(-1.02, 1.02, 60):
    mip, mop = np.abs(phi_i - ph) < 0.02, np.abs(phi_o - ph) < 0.02
    for zv in np.linspace(1824, 1854, 14):
        si = ri[mip & (np.abs(pi2[:, 2] - zv) < 1.8)]
        so = ro[mop & (np.abs(po2[:, 2] - zv) < 1.8)]
        if len(si) and len(so):
            g2.append(np.percentile(so, 2) - np.percentile(si, 98))
print("Band 2D grid min_gap:", min(g2), "median:", np.median(g2))
