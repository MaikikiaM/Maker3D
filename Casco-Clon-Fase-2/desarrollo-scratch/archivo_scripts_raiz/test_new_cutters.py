import os
import numpy as np
import trimesh
import manifold3d as m3d

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
casco_dir = os.path.join(BASE_DIR, r"Diseños 3D\Casco")

print("1. Loading base helmet...")
mb = trimesh.load(os.path.join(casco_dir, "Phase 2 Main Body.stl"))
v_base = mb.vertices.astype('float32')
f_base = mb.faces.astype('uint32')
helmet_mani = m3d.Manifold(m3d.Mesh(vert_properties=v_base, tri_verts=f_base))
base_vol = helmet_mani.volume()
print(f"Base volume: {base_vol:.1f} mm3")

# A. Dorsal Fin (Cresta dorsal)
print("2. Constructing Dorsal Fin cutter...")
fin_poly_yz = np.array([
    [-12.0, 1918.0],
    [-2.0,  1885.0],
    [205.0, 1870.0],
    [205.0, 1960.0],
    [-12.0, 1960.0]
], dtype=np.float32)
cs_fin = m3d.CrossSection([fin_poly_yz])
raw_fin = cs_fin.extrude(36.0)
m_rf = raw_fin.to_mesh()
v_rf = m_rf.vert_properties
v_fin_3d = np.column_stack([v_rf[:, 2] - 18.0, v_rf[:, 0], v_rf[:, 1]]).astype('float32')
f_fin_3d = m_rf.tri_verts.copy()
fin_cutter = m3d.Manifold(m3d.Mesh(vert_properties=v_fin_3d, tri_verts=f_fin_3d))
if fin_cutter.volume() <= 0:
    f_fin_3d[:, [0, 1]] = f_fin_3d[:, [1, 0]]
    fin_cutter = m3d.Manifold(m3d.Mesh(vert_properties=v_fin_3d, tri_verts=f_fin_3d))

# B. Brow and Forehead with Central V Wedge
print("3. Constructing Brow & Forehead cutter...")
# Lower brow band (Z from 1808.6 to 1826.0): wraps back above cheek depression towards ear to Y = +42 mm
brow_lower = m3d.Manifold.cube([270.0, 142.0, 17.4], center=False).translate([-135.0, -100.0, 1808.6])
# Upper forehead on dome (Z from 1826.0 to 1862.0): stops at Y = -18.0 mm, leaving clean white dome gap before wing (starts at Y=+14)
brow_upper = m3d.Manifold.cube([270.0, 82.0, 36.0], center=False).translate([-135.0, -100.0, 1826.0])
brow_solid = brow_lower + brow_upper

# Central V-shaped white wedge:
# At Z = 1808.6: width 28 mm (X in [-14, 14])
# At Z = 1862.0: width 44 mm (X in [-22, 22])
wedge_verts = np.array([
    [-14.0, -105.0, 1806.0], [ 14.0, -105.0, 1806.0], [ 14.0,  30.0, 1806.0], [-14.0,  30.0, 1806.0],
    [-22.0, -105.0, 1865.0], [ 22.0, -105.0, 1865.0], [ 22.0,  30.0, 1865.0], [-22.0,  30.0, 1865.0],
], dtype=np.float32)
wedge_faces = np.array([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]
], dtype=np.uint32)
wedge_mani = m3d.Manifold(m3d.Mesh(vert_properties=wedge_verts, tri_verts=wedge_faces))
brow_cutter = brow_solid - wedge_mani

# C. Thor Wings on Dome (Organic pincher enclosing 4 feathers and 3 angled slits)
print("4. Constructing Thor Wings...")
pts_wing = [
    [14.0, 1845.0],   # Thickened organic root at base (sits right on dome depression)
    [16.5, 1858.0],   # Smooth forward-leaning curve (no sharp 90 deg corner)
    [21.0, 1874.0],
    [27.0, 1890.0],
    [35.0, 1904.0],
    [48.0, 1913.0],   # Smooth arch to top feather
    [72.0, 1912.0],
    [94.0, 1905.0],
    [114.0, 1892.0],  # Tip Feather 1 (top)
    [102.0, 1880.0],  # Tip Feather 2
    [ 90.0, 1868.0],  # Tip Feather 3
    [ 78.0, 1856.0],  # Tip Feather 4 (bottom)
    [ 62.0, 1850.0],  # Lower contour of Feather 4
    [ 45.0, 1846.0],  # Lower contour along dome base
    [ 32.0, 1845.0],  # Lower root transition
    [ 14.0, 1845.0]
]
cs_wing_base = m3d.CrossSection([pts_wing[::-1]])

def make_slot(y_start, z_start, y_end, z_end, thickness=3.2):
    dx = y_end - y_start
    dy = z_end - z_start
    L = np.hypot(dx, dy)
    ux, uy = -dy/L * (thickness/2), dx/L * (thickness/2)
    p = [
        [y_start - ux, z_start - uy],
        [y_end   - ux, z_end   - uy],
        [y_end   + ux, z_end   + uy],
        [y_start + ux, z_start + uy]
    ]
    cs = m3d.CrossSection([p])
    if cs.area() == 0:
        cs = m3d.CrossSection([p[::-1]])
    return cs

s1 = make_slot(36.0, 1901.0, 120.0, 1885.0, thickness=3.2)
s2 = make_slot(30.0, 1885.0, 108.0, 1873.0, thickness=3.2)
s3 = make_slot(26.0, 1869.0,  96.0, 1861.0, thickness=3.2)

cs_wing = cs_wing_base - (s1 + s2 + s3)
raw_w = cs_wing.extrude(115.0)
m_rw = raw_w.to_mesh()
vw = m_rw.vert_properties
fw = m_rw.tri_verts

# Right wing (+X: from X = 28 to 143 mm)
vw_r = np.column_stack([vw[:, 2] + 28.0, vw[:, 0], vw[:, 1]]).astype('float32')
tri_r = fw.copy()
mani_w_r = m3d.Manifold(m3d.Mesh(vert_properties=vw_r, tri_verts=tri_r))
if mani_w_r.volume() <= 0:
    tri_r[:, [0, 1]] = tri_r[:, [1, 0]]
    mani_w_r = m3d.Manifold(m3d.Mesh(vert_properties=vw_r, tri_verts=tri_r))

# Left wing (-X: mirrored)
vw_l = np.column_stack([-(vw[:, 2] + 28.0), vw[:, 0], vw[:, 1]]).astype('float32')
tri_l = tri_r.copy()
tri_l[:, [0, 1]] = tri_l[:, [1, 0]]
mani_w_l = m3d.Manifold(m3d.Mesh(vert_properties=vw_l, tri_verts=tri_l))

wings_cutter = mani_w_r + mani_w_l

# D. Rear Band & Vertical Stripes
print("5. Constructing Rear Band & Vertical Stripes...")
# Rear horizontal band: placed strictly at dome depression Z in [1845.0, 1868.0] mm!
# Arches over the white circular disc (center Z = 1818.7, radius 26 mm).
# Stops at Y = 175 mm, without wrapping to the ears!
band_box = m3d.Manifold.cube([270.0, 60.0, 23.0], center=False).translate([-135.0, 175.0, 1845.0])
disc_clear = m3d.Manifold.cylinder(65.0, 26.5, 26.5, 128).rotate([90, 0, 0]).translate([0, 230.0, 1818.7])
rear_band = band_box - disc_clear

# Two vertical stripes running down the structural ribs:
# From base Z = 1616.0 to Z = 1848.0 (joining seamlessly with the rear band)
# Right stripe: X in [13.5, 25.5]
# Left stripe: X in [-25.5, -13.5]
rib_box_r = m3d.Manifold.cube([12.0, 50.0, 232.0], center=False).translate([13.5, 175.0, 1616.0])
rib_box_l = m3d.Manifold.cube([12.0, 50.0, 232.0], center=False).translate([-25.5, 175.0, 1616.0])

# Keep central column (|X| < 13.5) pure white
rear_markings = rear_band + rib_box_r + rib_box_l

# E. Cheeks, Snout, Mandible, Sub-ocular partial stripe, and 3 Claws
print("6. Constructing Cheeks, Snout, Sub-ocular shelf & 3 Claws...")

# Snout and central chin (solid red up to Z = 1785 mm)
snout_box = m3d.Manifold.cube([92.0, 65.0, 168.0], center=False).translate([-46.0, -100.0, 1618.0])

# Cheeks:
# - Covers bottom tube covering exactly the first TWO slots (up to Y = +10.5 mm)
# - Dips into cheek depression, covering floor and back vertical wall (up to Y = +44.0 mm)
# - Upper cheek behind eye goes up to Z = 1808.6 mm, stops before ear at Y = +44.0 mm
# - Front under eye goes up to Z = 1798.0 mm (reaching top of cheek flap and sub-ocular ledge!)
cheek_poly_yz = np.array([
    [-100.0, 1618.0],
    [  10.5, 1618.0],  # Reaches Y = +10.5 mm covering exactly the first 2 slots!
    [  10.5, 1660.0],
    [  25.0, 1695.0],
    [  44.0, 1735.0],  # Back vertical wall of cheek depression
    [  44.0, 1808.6],  # Upper cheek stopping before ear
    [-25.0,  1808.6],
    [-25.0,  1798.0],  # Sub-ocular horizontal rim
    [-100.0, 1798.0]
], dtype=np.float32)

cs_cheek = m3d.CrossSection([cheek_poly_yz])
if cs_cheek.area() == 0:
    cs_cheek = m3d.CrossSection([cheek_poly_yz[::-1]])

raw_cheek = cs_cheek.extrude(115.0)
m_rc = raw_cheek.to_mesh()
v_rc = m_rc.vert_properties
f_rc = m_rc.tri_verts

# Right cheek (+X: from X = 5 to 120 mm)
v_cr = np.column_stack([v_rc[:, 2] + 5.0, v_rc[:, 0], v_rc[:, 1]]).astype('float32')
f_cr = f_rc.copy()
cheek_r = m3d.Manifold(m3d.Mesh(vert_properties=v_cr, tri_verts=f_cr))
if cheek_r.volume() <= 0:
    f_cr[:, [0, 1]] = f_cr[:, [1, 0]]
    cheek_r = m3d.Manifold(m3d.Mesh(vert_properties=v_cr, tri_verts=f_cr))

# Left cheek (-X: mirrored)
v_cl = np.column_stack([-(v_rc[:, 2] + 5.0), v_rc[:, 0], v_rc[:, 1]]).astype('float32')
f_cl = f_cr.copy()
f_cl[:, [0, 1]] = f_cl[:, [1, 0]]
cheek_l = m3d.Manifold(m3d.Mesh(vert_properties=v_cl, tri_verts=f_cl))

cheeks_raw = cheek_r + cheek_l

# Sub-ocular partial white stripe:
# Bounded strictly between |X| = 27.0 mm and |X| = 53.0 mm!
# For |X| < 27.0 mm (towards central T-stem): cheek flap is SOLID RED!
# For |X| > 53.0 mm (towards outer eye corner): cheek wall is SOLID RED!
# In the middle (55-60% of span): WHITE stripe on the shelf!
suboc_box_r = m3d.Manifold.cube([26.0, 45.0, 16.0], center=False).translate([27.0, -80.0, 1783.0])
suboc_box_l = m3d.Manifold.cube([26.0, 45.0, 16.0], center=False).translate([-53.0, -80.0, 1783.0])
suboc_cutter = suboc_box_r + suboc_box_l

# 3 White Claws on aerator tube curves:
# Extending inward past aerator center (X = +/- 47.4 mm) to X = +/- 26 mm!
# Bold thickness (7.5 mm), clearly visible
claws_r = m3d.Manifold()
claws_l = m3d.Manifold()
for zc, yc in [(1656.0, -57.0), (1642.0, -54.0), (1628.0, -51.0)]:
    cr = m3d.Manifold.cube([56.0, 7.5, 45.0], center=True).rotate([18.0, 5.0, -12.0]).translate([54.0, yc, zc])
    cl = m3d.Manifold.cube([56.0, 7.5, 45.0], center=True).rotate([18.0, -5.0, 12.0]).translate([-54.0, yc, zc])
    claws_r = claws_r + cr
    claws_l = claws_l + cl

claws_cutter = claws_r + claws_l

face_cutter = (snout_box + cheeks_raw) - suboc_cutter - claws_cutter

# Hollow Visor Aperture Keep-Out:
# Horizontal slit: X in [-65, 65], Y in [-90, -35], Z in [1796.0, 1808.5]
# Vertical T-stem: X in [-5.0, 5.0], Y in [-90, -45], Z in [1720.0, 1796.0]
visor_clear_h = m3d.Manifold.cube([130.0, 60.0, 13.0], center=False).translate([-65.0, -90.0, 1796.0])
visor_clear_v = m3d.Manifold.cube([ 10.0, 50.0, 76.0], center=False).translate([-5.0, -90.0, 1720.0])
visor_clearance = visor_clear_h + visor_clear_v

total_red_cutter = (fin_cutter + brow_cutter + wings_cutter + rear_markings + face_cutter) - visor_clearance

print("7. Performing Boolean Interlocking Cut...")
red_mani = helmet_mani ^ total_red_cutter
white_mani = helmet_mani - total_red_cutter

vol_red = red_mani.volume()
vol_white = white_mani.volume()
vol_sum = vol_red + vol_white
dev = abs(vol_sum - base_vol)

print(f"Red markings volume: {vol_red:.1f} mm3, tris: {red_mani.num_tri()}")
print(f"White base volume: {vol_white:.1f} mm3, tris: {white_mani.num_tri()}")
print(f"Volume conservation: Sum = {vol_sum:.1f} mm3 | Diff = {dev:.4f} mm3")
print("SUCCESS!")
