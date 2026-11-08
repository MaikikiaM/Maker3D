import numpy as np

sam = np.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\scratch\helmet_samples.npz")
sp, sn = sam["p"], sam["n"]

print("=== 1. REAR RIBS OUTER SKIN (ny > 0.5, Y > 192, X in [0, 38]) ===")
rb = sp[(sp[:, 0] >= 0) & (sp[:, 0] <= 38) & (sp[:, 1] > 192) & (sn[:, 1] > 0.3)]
for zv in [1652, 1675, 1700, 1730, 1760, 1790, 1812]:
    s = rb[(rb[:, 2] >= zv - 2.5) & (rb[:, 2] <= zv + 2.5)]
    # Print Y_max in 1 mm X bins from X=8 to X=35
    line = []
    for x0 in range(10, 34, 1):
        b = s[(s[:, 0] >= x0) & (s[:, 0] < x0 + 1)]
        if len(b):
            # Look at normal nx to find side walls of rib!
            nx_med = np.median(sn[(sp[:, 0] >= x0) & (sp[:, 0] < x0 + 1) & (sp[:, 1] > 192) & (np.abs(sp[:, 2] - zv) <= 2.5), 0])
            line.append(f"{x0}:{b[:,1].max():.1f}({nx_med:+.1f})")
    print(f"Z={zv:4d}: " + " ".join(line))

print("\n=== 2. CHEEK DEPRESSION OUTER SKIN X(Y, Z) (nx > 0.2 or ny != 0, X > 65) ===")
ch = sp[(sp[:, 0] > 65) & (sp[:, 1] >= -18) & (sp[:, 1] <= 50) & (sp[:, 2] >= 1710) & (sp[:, 2] <= 1785)]
# Outer skin has normal pointing generally outward (nx > 0 or ny < -0.2, not inner helmet cavity nx < -0.3)
ch_n = sn[(sp[:, 0] > 65) & (sp[:, 1] >= -18) & (sp[:, 1] <= 50) & (sp[:, 2] >= 1710) & (sp[:, 2] <= 1785)]
ch_out = ch[ch_n[:, 0] > -0.1]
print(f"Cheek hollow outer skin X_min={ch_out[:,0].min():.2f}, X_max={ch_out[:,0].max():.2f}")
for zv in [1718, 1730, 1742, 1755, 1768]:
    row = []
    for yv in [-10, 0, 10, 20, 30, 40]:
        b = ch_out[(np.abs(ch_out[:, 1] - yv) < 3) & (np.abs(ch_out[:, 2] - zv) < 3)]
        if len(b):
            row.append(f"Y={yv:3d}:X={b[:,0].max():5.1f}")
    print(f"Z={zv:4d}: " + " | ".join(row))

print("\n=== 3. ALL 11 TUBE SLITS BOUNDING BOX (+X > 85) ===")
slits = sp[(sp[:, 0] > 85) & (sp[:, 1] >= -2) & (sp[:, 1] <= 65) & (sp[:, 2] >= 1680) & (sp[:, 2] <= 1710) & (np.abs(sn[:, 1]) > 0.75)]
print(f"Slits Y_range=[{slits[:,1].min():.2f}, {slits[:,1].max():.2f}], Z_range=[{slits[:,2].min():.2f}, {slits[:,2].max():.2f}]")
