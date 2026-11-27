import os
import numpy as np
import trimesh

CASCO = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco"
main = trimesh.load(os.path.join(CASCO, "Phase 2 Main Body.stl"))
V = main.vertices
N = main.vertex_normals

print("=== 1. DORSAL FIN EXACT PROFILE ALONG Y ===")
for yv in range(-40, 196, 10):
    sec = main.section(plane_origin=[0, yv, 0], plane_normal=[0, 1, 0])
    if sec is None:
        continue
    pts = sec.vertices
    # Outer surface on +X side near top of dome (X in [0, 30], Z > 1860)
    s = pts[(pts[:, 0] >= 0) & (pts[:, 0] <= 30) & (pts[:, 2] > 1860)]
    if len(s) == 0:
        continue
    # Take upper envelope Z(X) in 0.5 mm bins
    xbins = np.arange(0, 28, 0.5)
    zx = []
    for x0 in xbins:
        b = s[(s[:, 0] >= x0) & (s[:, 0] < x0 + 0.5)]
        if len(b):
            zx.append((x0 + 0.25, b[:, 2].max()))
    zx = np.array(zx)
    if len(zx) < 10:
        continue
    z_peak = zx[:, 1].max()
    x_peak = zx[np.argmax(zx[:, 1]), 0]
    # Find steepest drop dZ/dX (the vertical outer wall of the fin!)
    dz = np.diff(zx[:, 1]) / np.diff(zx[:, 0])
    steep = np.where(dz < -1.2)[0]
    if len(steep):
        x_rim_outer = zx[steep[0], 0]
        x_dome_junc = zx[steep[-1] + 1, 0]
        z_dome_junc = zx[steep[-1] + 1, 1]
        print(f"Y={yv:4d}: Z_peak={z_peak:7.2f} (at X={x_peak:4.1f}) | wall X=[{x_rim_outer:5.2f} -> {x_dome_junc:5.2f}], Z_dome={z_dome_junc:7.2f}")
    else:
        print(f"Y={yv:4d}: NO FIN WALL (Z_max={z_peak:7.2f})")

print("\n=== 2. EYE VISOR UPPER & LOWER RIM (RIGHT SIDE +X) ===")
# Aperture edge vertices around visor
for xv in [10, 18, 26, 34, 42, 46, 50, 58, 66, 74, 80, 84, 86]:
    sec = main.section(plane_origin=[xv, 0, 0], plane_normal=[1, 0, 0])
    if sec is None:
        continue
    pts = sec.vertices
    s = pts[(pts[:, 1] < 5) & (pts[:, 2] > 1760) & (pts[:, 2] < 1820)]
    # Check gap in Z where visor opening is hollow!
    z_sorted = np.sort(s[:, 2])
    gaps = np.diff(z_sorted)
    big = np.where(gaps > 4.0)[0]
    if len(big):
        idx = big[np.argmax(gaps[big])]
        z_bot = z_sorted[idx]
        z_top = z_sorted[idx + 1]
        y_bot = s[np.argmin(np.abs(s[:, 2] - z_bot)), 1]
        y_top = s[np.argmin(np.abs(s[:, 2] - z_top)), 1]
        print(f"X={xv:2d}: Z_bot={z_bot:7.2f} (Y={y_bot:6.2f}) | Z_top={z_top:7.2f} (Y={y_top:6.2f})")

print("\n=== 3. SNOUT-TO-TUBE VALLEY CREASE (+X SIDE) ===")
for zv in range(1635, 1706, 5):
    sec = main.section(plane_origin=[0, 0, zv], plane_normal=[0, 0, 1])
    pts = sec.vertices
    # Front outer surface X in [10, 85], Y < -20
    s = pts[(pts[:, 0] >= 10) & (pts[:, 0] <= 85) & (pts[:, 1] < -20)]
    # Frontmost Y(X) in 1.5 mm bins
    xbins = np.arange(12, 82, 1.5)
    yx = []
    for x0 in xbins:
        b = s[(s[:, 0] >= x0) & (s[:, 0] < x0 + 1.5)]
        if len(b):
            yx.append((x0 + 0.75, b[:, 1].min()))
    yx = np.array(yx)
    if len(yx) > 5:
        # Valley crease is where Y(X) has a concave inflection / local max (furthest back +Y between snout & tube!)
        # Between X=18 and X=55
        mid = yx[(yx[:, 0] >= 18) & (yx[:, 0] <= 55)]
        if len(mid):
            # Look at second derivative or local max of Y (since front is -Y, recessed valley is LESS negative Y = max Y!)
            imax = np.argmax(mid[:, 1])
            print(f"Z={zv:4d}: max-Y valley at X={mid[imax,0]:5.1f}, Y={mid[imax,1]:6.2f} | X=25 Y={yx[np.argmin(np.abs(yx[:,0]-25)),1]:6.1f} | X=45 Y={yx[np.argmin(np.abs(yx[:,0]-45)),1]:6.1f}")

print("\n=== 4. REAR RECESSED BAND & RIBS EXACT CREASES ===")
for xv in [0, 30, 55, 75]:
    sec = main.section(plane_origin=[xv, 0, 0], plane_normal=[1, 0, 0]).vertices
    s = sec[(sec[:, 1] > 130) & (sec[:, 2] >= 1812) & (sec[:, 2] <= 1865)]
    prof = []
    for z0 in np.arange(1814, 1862, 2.0):
        b = s[(s[:, 2] >= z0) & (s[:, 2] < z0 + 2.0)]
        if len(b):
            prof.append(f"{z0+1:.0f}:{b[:,1].max():.1f}")
    print(f"Rear X={xv:2d}: " + " ".join(prof))
