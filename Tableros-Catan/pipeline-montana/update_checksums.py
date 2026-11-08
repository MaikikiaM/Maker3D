import os
import hashlib

proj_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D"
manifest_path = os.path.join(proj_dir, "checksums_sha256.txt")

files_to_hash = [
    "catan_mountain_solid.stl",
    "catan_mountain_fullcolor.ply",
    "catan_mountain_fullcolor.obj",
    "catan_mountain.mtl",
    "catan_mountain_fullcolor.3mf",
    "catan_mountain_diffuse_4k.png",
    "catan_mountain_diffuse_2k.png",
    "catan_mountain_normal_4k.png",
    "catan_mountain_viewer.html",
    "catan_mountain_engineering_dossier.md",
    "bambu_orcaslicer_mountain_profile.json",
    "prusaslicer_mmu3_mountain_profile.ini"
]

lines = [
    "# SHA-256 INTEGRITY MANIFEST - CATAN 3D MOUNTAIN FULL-COLOR PRODUCTION SUITE",
    "# Certified: 2026-11-04 (5K Glaciated Peak Alpamayo / Mont Blanc Edition)\n"
]

for fname in files_to_hash:
    fpath = os.path.join(proj_dir, fname)
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        digest = h.hexdigest()
        lines.append(f"{digest}  {size_mb:8.2f} MB  {fname}")

content = "\n".join(lines) + "\n"
with open(manifest_path, "w", encoding="utf-8") as f:
    f.write(content)

print(content)
print("Updated checksums_sha256.txt successfully.")
