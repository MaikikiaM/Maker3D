import trimesh
import numpy as np

m = trimesh.load(r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco\Phase 2 Main Body.stl")
v = m.vertices

# Dome center in Y-Z: roughly Y ≈ 50, Z ≈ 1830
mask_dome = (v[:, 2] > 1825) & (v[:, 1] > -30) & (v[:, 1] < 120) & (v[:, 0] > 0)
pts = v[mask_dome]

# Spherical / cylindrical coordinates from dome center (0, 50, 1835)
x = pts[:, 0]
y = pts[:, 1]
z = pts[:, 2]
r_sagittal = np.hypot(x, z - 1835) # distance from Y axis in X-Z plane
angle_xz = np.degrees(np.arctan2(x, z - 1835)) # 0 = straight up (crest), 90 = lateral (ear)

print(f"Dome points: {len(pts)}")
print(f"Angle X-Z range: min = {angle_xz.min():.1f}°, max = {angle_xz.max():.1f}°")
print(f"Radius in X-Z: min = {r_sagittal.min():.1f}, max = {r_sagittal.max():.1f}")

# The crest (fin) is at |X| <= 18 mm.
# At Z ≈ 1930, X=18 corresponds to angle = arctan(18 / (1930 - 1835)) = arctan(18/95) ≈ 10.7°!
# Where does the white space end and the wing start?
# In photos 2 and 3, the wing is on the side of the dome:
# angle_xz is between roughly 30° and 85°!
print("\nChecking angles for different X at apex (Z ≈ 1900, dz = 65):")
for xi in [20, 30, 40, 50, 60, 70, 80, 90]:
    ang = np.degrees(np.arctan2(xi, 65))
    print(f"X = {xi:2d} mm -> angle = {ang:4.1f}°")
