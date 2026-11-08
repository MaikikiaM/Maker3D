import numpy as np

d = np.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\scratch\helmet_feature_edges.npz")
P = 0.5 * (d["a"] + d["b"])

sam = np.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\scratch\helmet_samples.npz")
sp, sn = sam["p"], sam["n"]

print("--- 1. FIN OUTER SIDE WALL (sp with X in [12, 22], Z > 1870, nx > 0.7) ---")
fw = sp[(sp[:, 0] > 12) & (sp[:, 0] < 22) & (sp[:, 2] > 1868) & (sn[:, 0] > 0.75)]
print(f"Fin right wall sample count: {len(fw)}, Y_range=[{fw[:,1].min():.2f}, {fw[:,1].max():.2f}]")
for y0 in range(-40, 195, 15):
    s = fw[(fw[:, 1] >= y0) & (fw[:, 1] < y0 + 15)]
    if len(s):
        print(f"Y=[{y0:3d},{y0+15:3d}]: X=[{s[:,0].min():5.2f}, {s[:,0].max():5.2f}], Z=[{s[:,2].min():7.2f}, {s[:,2].max():7.2f}]")

print("\n--- 2. DOME RECESSED BAND FEATURE EDGES (Z in [1815, 1862], Y > 0) ---")
b = P[(P[:, 0] > 0) & (P[:, 1] > 0) & (P[:, 2] >= 1815) & (P[:, 2] <= 1862)]
for y0 in [10, 40, 80, 120, 150, 180, 200]:
    s = b[(b[:, 1] >= y0 - 8) & (b[:, 1] <= y0 + 8)]
    if len(s):
        vals = []
        for zc in [1819, 1827, 1854]:
            sub = s[(s[:, 2] >= zc - 3.5) & (s[:, 2] <= zc + 3.5)]
            if len(sub):
                vals.append(f"~{zc}: Z={np.median(sub[:,2]):7.2f} (X_max={sub[:,0].max():5.1f})")
        print(f"Y~{y0:3d}: " + " | ".join(vals))

print("\n--- 3. CHEEK DEPRESSION FEATURE EDGES (+X > 65, Y in [-20, 55], Z in [1705, 1790]) ---")
c = P[(P[:, 0] > 65) & (P[:, 1] >= -20) & (P[:, 1] <= 55) & (P[:, 2] >= 1705) & (P[:, 2] <= 1790)]
for y0 in range(-16, 52, 8):
    s = c[(c[:, 1] >= y0) & (c[:, 1] < y0 + 8)]
    if len(s):
        zs = np.sort(np.unique(np.round(s[:, 2], 1)))
        print(f"Y=[{y0:3d},{y0+8:3d}]: Z_min={s[:,2].min():6.1f}, Z_max={s[:,2].max():6.1f} | Y_max={s[:,1].max():5.2f}")

# Exact rear wall Y of cheek depression:
cw_back = sp[(sp[:, 0] > 68) & (sp[:, 1] > 42) & (sp[:, 1] < 53) & (sp[:, 2] > 1715) & (sp[:, 2] < 1780) & (sn[:, 1] < -0.7)]
print(f"Cheek depression vertical back wall (ny < -0.7): Y_median={np.median(cw_back[:,1]):.2f}, Y_range=[{cw_back[:,1].min():.2f}, {cw_back[:,1].max():.2f}], Z_range=[{cw_back[:,2].min():.2f}, {cw_back[:,2].max():.2f}], X_range=[{cw_back[:,0].min():.2f}, {cw_back[:,0].max():.2f}]")

print("\n--- 4. REAR RIBS & DISC FEATURE EDGES (Y > 180, X in [0, 35], Z in [1645, 1848]) ---")
r = P[(P[:, 0] >= 0) & (P[:, 0] <= 35) & (P[:, 1] > 185) & (P[:, 2] >= 1645) & (P[:, 2] <= 1848)]
for z0 in [1652, 1680, 1720, 1760, 1800, 1816]:
    s = r[(r[:, 2] >= z0 - 4) & (r[:, 2] <= z0 + 4)]
    if len(s):
        # Outer skin only (Y > 194)
        s_out = s[s[:, 1] > 194]
        print(f"Z~{z0:4d}: X feature clusters = {np.unique(np.round(s_out[:, 0], 1))}")

disc = P[(np.abs(P[:, 0]) <= 30) & (P[:, 1] > 196) & (P[:, 2] >= 1790) & (P[:, 2] <= 1848)]
d_base = disc[(disc[:, 1] < 206) & (disc[:, 2] > 1792) & (disc[:, 2] < 1847)]
if len(d_base):
    zc_est = (d_base[:, 2].max() + d_base[:, 2].min()) * 0.5
    rad = np.sqrt(d_base[:, 0]**2 + (d_base[:, 2] - zc_est)**2)
    print(f"Rear Disc base crease: Z_center={zc_est:.2f}, R_median={np.median(rad):.2f}, R_max={rad.max():.2f}, Z_range=[{d_base[:,2].min():.2f}, {d_base[:,2].max():.2f}]")

print("\n--- 5. FROWN UPPER & LOWER CREASES (Y < -45, Z in [1698, 1745], X in [0, 46]) ---")
fr = P[(P[:, 0] >= 0) & (P[:, 0] <= 46) & (P[:, 1] < -45) & (P[:, 2] >= 1698) & (P[:, 2] <= 1745)]
for x0 in range(0, 44, 6):
    s = fr[(fr[:, 0] >= x0) & (fr[:, 0] < x0 + 6)]
    if len(s):
        print(f"X=[{x0:2d},{x0+6:2d}]: Z_min={s[:,2].min():6.2f}, Z_max={s[:,2].max():6.2f}, Y_min={s[:,1].min():6.2f}")
