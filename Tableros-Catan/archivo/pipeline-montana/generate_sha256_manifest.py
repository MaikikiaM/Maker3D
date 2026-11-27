import os
import hashlib

print("=== GENERATING CRYPTOGRAPHIC SHA-256 INTEGRITY MANIFEST ===")

out_dir = r"Tableros Catan\Loseta_Montana_3D"
deliverables = [
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

manifest_lines = []
manifest_lines.append("# SHA-256 INTEGRITY MANIFEST - CATAN 3D MOUNTAIN FULL-COLOR PRODUCTION SUITE")
manifest_lines.append("# Certified: 2026-11-04\n")

for fn in deliverables:
    p = os.path.join(out_dir, fn)
    if os.path.exists(p):
        sha = hashlib.sha256()
        with open(p, 'rb') as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        digest = sha.hexdigest()
        size_mb = os.path.getsize(p) / (1024 * 1024)
        line = f"{digest}  {size_mb:8.2f} MB  {fn}"
        print(line)
        manifest_lines.append(line)

out_manifest = os.path.join(out_dir, "checksums_sha256.txt")
with open(out_manifest, 'w', encoding='utf-8') as f:
    f.write("\n".join(manifest_lines) + "\n")
print(f"\nSaved manifest to {out_manifest}")
