import os
import numpy as np
import trimesh

CASCO = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco"
main = trimesh.load(os.path.join(CASCO, "Phase 2 Main Body.stl"))
no_teeth = trimesh.load(os.path.join(CASCO, "Phase 2 Main Body - No Teeth.stl"))
V = main.vertices
N = main.vertex_normals

# 1. Fin cross-section at Y=40, Z > 1915
print("=== 1. FIN CROSS SECTION AT Y in [38, 42], Z > 1915 ===")
m = (V[:,1] >= 38) & (V[:,1] <= 42) & (V[:,2] > 1915) & (np.abs(V[:,0]) < 25)
pts = V[m]
nrm = N[m]
for x0 in np.arange(-20, 20, 2):
    s = pts[(pts[:,0] >= x0) & (pts[:,0] < x0+2)]
    sn = nrm[(pts[:,0] >= x0) & (pts[:,0] < x0+2)]
    if len(s):
        print(f"X=[{x0:5.1f},{x0+2:5.1f}]: Z_max={s[:,2].max():7.2f}, Z_min={s[:,2].min():7.2f}, mean_nx={sn[:,0].mean():+.2f}, mean_nz={sn[:,2].mean():+.2f}")

# 2. Trace exact Dome Recessed Band step (where normal has strong +Z or -Z or step in radius)
print("\n=== 2. DOME RECESSED BAND UPPER & LOWER EDGES ALONG Y (X > 40) ===")
for y0 in range(-10, 205, 10):
    m = (V[:,0] > 30) & (np.abs(V[:,1] - y0) < 2.5) & (V[:,2] >= 1815) & (V[:,2] <= 1865)
    s = V[m]
    sn = N[m]
    # outer surface only: nx > 0.2 or ny > 0.2
    out = (s[:,0]**2 + (s[:,1]-60)**2) > 60**2
    s, sn = s[out], sn[out]
    # Look for downward-facing overhang normal (nz < -0.25) -> upper edge of recess!
    up_edge = s[sn[:,2] < -0.25]
    # Look for upward-facing shelf normal (nz > 0.35) in [1820, 1835] -> lower edge of recess!
    lo_edge = s[(sn[:,2] > 0.35) & (s[:,2] < 1840) & (s[:,2] > 1820)]
    u_str = f"{np.median(up_edge[:,2]):.2f} (n={len(up_edge)})" if len(up_edge) else "None"
    l_str = f"{np.median(lo_edge[:,2]):.2f} (n={len(lo_edge)})" if len(lo_edge) else "None"
    print(f"Y={y0:4d}: lower_step Z={l_str:16s} | upper_step Z={u_str}")

# 3. Forward curved edge of recessed band (Y in [0, 35], Z in [1824, 1855], X > 60)
print("\n=== 3. DOME RECESSED BAND FRONT CURVE (ny < -0.15 and nx > 0.3) ===")
m = (V[:,0] > 60) & (V[:,1] >= 0) & (V[:,1] <= 35) & (V[:,2] >= 1824) & (V[:,2] <= 1855)
s, sn = V[m], N[m]
# Step faces forward-outward or check X jump
for z0 in range(1825, 1854, 2):
    sub = s[(s[:,2] >= z0) & (s[:,2] < z0+2)]
    subn = sn[(s[:,2] >= z0) & (s[:,2] < z0+2)]
    # find where ny is most negative or curvature creases
    if len(sub):
        idx = np.argsort(sub[:,1])
        sub = sub[idx]
        # compute dx/dy
        print(f"Z={z0+1:4d}: Y range [{sub[:,1].min():5.1f}, {sub[:,1].max():5.1f}], min_ny={subn[:,1].min():+.2f} at Y={sub[np.argmin(subn[:,1]),1]:5.2f}, X={sub[np.argmin(subn[:,1]),0]:6.2f}")

# 4. Visor aperture exact contour (inner rim of eye & T-stem)
print("\n=== 4. VISOR APERTURE CONTOUR (X >= 0) ===")
# Faces on visor rim have normals pointing into the visor hole (+Z on bottom rim, -Z on top rim, -X on outer corner, -X/+X on stem)
m_vis = (V[:,0] >= 0) & (V[:,0] <= 102) & (V[:,1] < -10) & (V[:,2] >= 1670) & (V[:,2] <= 1815)
sv, sn = V[m_vis], N[m_vis]
for x0 in range(0, 96, 6):
    sub = sv[(sv[:,0] >= x0) & (sv[:,0] < x0+6)]
    subn = sn[(sv[:,0] >= x0) & (sv[:,0] < x0+6)]
    top_rim = sub[(sub[:,2] > 1805) & (sub[:,2] < 1813) & (subn[:,2] < -0.5)]
    bot_rim = sub[(sub[:,2] > 1765) & (sub[:,2] < 1802) & (subn[:,2] > 0.4)]
    t_s = f"Z={np.median(top_rim[:,2]):.2f}, Y={np.median(top_rim[:,1]):.2f}" if len(top_rim) else "-"
    b_s = f"Z={np.median(bot_rim[:,2]):.2f}, Y={np.median(bot_rim[:,1]):.2f}" if len(bot_rim) else "-"
    print(f"X=[{x0:2d},{x0+6:2d}]: TOP({t_s}) | BOT({b_s})")

# 5. Cheek depression (hollow) exact bounds & walls (X > 55, Y in [-30, 80], Z in [1700, 1790])
print("\n=== 5. CHEEK DEPRESSION (PÓMULO) TOPOLOGY ===")
m_ch = (V[:,0] > 55) & (V[:,1] >= -25) & (V[:,1] <= 75) & (V[:,2] >= 1705) & (V[:,2] <= 1790)
sc, snc = V[m_ch], N[m_ch]
# Back wall of cheek depression faces forward (ny < -0.6)
back_wall = sc[(snc[:,1] < -0.6) & (sc[:,1] > 40)]
print(f"Back wall of cheek depression: Y=[{back_wall[:,1].min():.2f}, {back_wall[:,1].max():.2f}], Z=[{back_wall[:,2].min():.2f}, {back_wall[:,2].max():.2f}], X=[{back_wall[:,0].min():.2f}, {back_wall[:,0].max():.2f}]")
# Bottom shelf of cheek depression faces upward (nz > 0.5)
bot_shelf = sc[(snc[:,2] > 0.5) & (sc[:,2] < 1735)]
print(f"Bottom shelf of cheek depression: Y=[{bot_shelf[:,1].min():.2f}, {bot_shelf[:,1].max():.2f}], Z=[{bot_shelf[:,2].min():.2f}, {bot_shelf[:,2].max():.2f}], X=[{bot_shelf[:,0].min():.2f}, {bot_shelf[:,0].max():.2f}]")
for y0 in range(-20, 72, 8):
    b_sub = bot_shelf[(bot_shelf[:,1] >= y0) & (bot_shelf[:,1] < y0+8)]
    if len(b_sub):
        print(f"  Shelf Y=[{y0:3d},{y0+8:3d}]: Z_med={np.median(b_sub[:,2]):.2f}, X_range=[{b_sub[:,0].min():.1f},{b_sub[:,0].max():.1f}]")

# 6. Tube stripe recessed slots on jaw tube!
print("\n=== 6. JAW TUBE RECESSED SLOTS (RAYITAS LATERALES) ===")
# Recessed slots have inward X normal or sideways Y normals inside X > 80, Y in [-15, 70], Z in [1680, 1720]
m_tb = (V[:,0] > 75) & (V[:,1] >= -15) & (V[:,1] <= 70) & (V[:,2] >= 1680) & (V[:,2] <= 1720)
st, snt = V[m_tb], N[m_tb]
slot_walls = st[np.abs(snt[:,1]) > 0.75]
print(f"Tube slot walls: Y=[{slot_walls[:,1].min():.2f}, {slot_walls[:,1].max():.2f}], Z=[{slot_walls[:,2].min():.2f}, {slot_walls[:,2].max():.2f}], X=[{slot_walls[:,0].min():.2f}, {slot_walls[:,0].max():.2f}]")
# Estimate slope dY/dZ of the tube slots!
# Cluster or check top vs bottom of first slot
front_slot = slot_walls[slot_walls[:,1] < 10]
for z0 in range(1686, 1716, 4):
    fs = front_slot[(front_slot[:,2] >= z0) & (front_slot[:,2] < z0+4)]
    if len(fs):
        print(f"  First slot Z=[{z0},{z0+4}]: Y_min={fs[:,1].min():.2f}")

# 7. Teeth exact difference between Main Body and Main Body - No Teeth
print("\n=== 7. EXACT TEETH GEOMETRY ===")
from scipy.spatial import cKDTree
tree = cKDTree(no_teeth.vertices)
dists, _ = tree.query(V, k=1)
teeth_pts = V[dists > 0.5]
print(f"Teeth vertices (dist > 0.5mm from No-Teeth): count={len(teeth_pts)}")
if len(teeth_pts):
    print(f"  X=[{teeth_pts[:,0].min():.2f}, {teeth_pts[:,0].max():.2f}], Y=[{teeth_pts[:,1].min():.2f}, {teeth_pts[:,1].max():.2f}], Z=[{teeth_pts[:,2].min():.2f}, {teeth_pts[:,2].max():.2f}]")

# 8. Rear disc & rear vertical ribs exact coordinates
print("\n=== 8. REAR DISC & VERTICAL RIBS ===")
m_rear = (V[:,1] > 175) & (np.abs(V[:,0]) < 45) & (V[:,2] >= 1616) & (V[:,2] <= 1845)
sr, snr = V[m_rear], N[m_rear]
# Rear disc center & radius (Z in [1790, 1842], Y > 210)
disc_pts = sr[(sr[:,2] > 1788) & (sr[:,2] < 1845) & (sr[:,1] > 210)]
print(f"Rear disc protruding face: X=[{disc_pts[:,0].min():.2f}, {disc_pts[:,0].max():.2f}], Z=[{disc_pts[:,2].min():.2f}, {disc_pts[:,2].max():.2f}], Z_center={(disc_pts[:,2].min()+disc_pts[:,2].max())/2:.2f}")
# Rear vertical ribs flanking the center ladder (Z=1700)
rib_slice = sr[(sr[:,2] > 1698) & (sr[:,2] < 1702) & (sr[:,0] > 0)]
for x0 in range(0, 40, 2):
    rs = rib_slice[(rib_slice[:,0] >= x0) & (rib_slice[:,0] < x0+2)]
    rsn = snr[(sr[:,2] > 1698) & (sr[:,2] < 1702) & (sr[:,0] > 0)][(rib_slice[:,0] >= x0) & (rib_slice[:,0] < x0+2)]
    if len(rs):
        print(f"  Rear Z=1700 X=[{x0:2d},{x0+2:2d}]: Y_max={rs[:,1].max():.2f}, mean_nx={rsn[:,0].mean():+.2f}")
