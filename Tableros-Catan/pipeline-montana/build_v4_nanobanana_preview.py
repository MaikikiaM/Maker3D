"""
Non-destructive V4 Nano Banana 3D Prototype & WebGL PBR Preview Builder.
Reuses nanobanana_to_catan_hex.py (pure NumPy/SciPy Delaunay CAD bridge) and
assemble_viewer_v2.py (Three.js HDR PMREM + PCFSoftShadowMap viewer) without
touching production V3 assets.
"""
import argparse
import json
import math
import os
import sys
import numpy as np
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SKILL_SCRIPTS = os.path.join(REPO_ROOT, ".agents", "skills", "nanobanana-3d-sculpt", "scripts")
sys.path.insert(0, SKILL_SCRIPTS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nanobanana_to_catan_hex as nb
import assemble_viewer_v2 as av2


def build_v4_preview(heightmap_path, albedo_path=None, out_dir="Tableros-Catan/pipeline-montana/scratch_v4_nanobanana",
                     viewer_html="Visores-HTML/preview-montana-v4-nanobanana.html", relief_mm=64.0, nx=350, tex_size=2048,
                     micro_detail_mm=0.25):
    os.makedirs(out_dir, exist_ok=True)
    hm_arr = np.asarray(Image.open(heightmap_path).convert("L"))
    albedo_arr = np.asarray(Image.open(albedo_path).convert("RGB")) if albedo_path else None

    Xt, Yt, Zt, dist_t, _, dxt, dyt = nb.synthesize_dem_from_heightmap(
        hm_arr, tex_size, tex_size, relief_mm=relief_mm, micro_detail_mm=micro_detail_mm, x_lim=nb.R, y_lim=nb.R
    )
    clay_rgb = nb.render_shaded_clay_rgb(Zt, dist_t, dxt, dyt)
    clay_path = os.path.join(out_dir, "v4_nanobanana_clay_overhead.png")
    nb.save_uv_aligned_png(clay_rgb, clay_path)

    if albedo_arr is None:
        print(f"[Clay Ready] Inspect & pass to image-generator for painterly albedo: {clay_path}")
        return clay_path

    diffuse_rgb = np.clip(nb.sample_square_domain(albedo_arr, Xt, Yt), 0, 255).astype(np.uint8)
    normal_u8, rough_u8 = nb.compute_pbr_maps(Zt, diffuse_rgb, dxt, dyt)

    nb.save_uv_aligned_png(diffuse_rgb, os.path.join(out_dir, "catan_mountain_diffuse_2k.png"))
    nb.save_uv_aligned_png(normal_u8, os.path.join(out_dir, "catan_mountain_normal_2k.png"))
    nb.save_uv_aligned_png(rough_u8, os.path.join(out_dir, "catan_mountain_roughness_2k.png"))

    ny = int(round(nx * (nb.R / nb.APOTHEM)))
    Xm, Ym, Zm, dist_m, _, _, _ = nb.synthesize_dem_from_heightmap(
        hm_arr, nx, ny, relief_mm=relief_mm, micro_detail_mm=micro_detail_mm, x_lim=nb.APOTHEM, y_lim=nb.R
    )
    verts, faces = nb.build_solid_hex_mesh(Xm, Ym, Zm, dist_m, include_magnet=True)
    stats = nb.verify_watertight(verts, faces)
    if not stats["watertight"]:
        raise RuntimeError(f"Mesh failed watertight check: {stats}")

    stl_path = os.path.join(out_dir, "catan_mountain_v4_nanobanana.stl")
    nb.write_binary_stl(verts, faces, stl_path, label="Catan Mountain V4 NanoBanana")
    np.savez_compressed(os.path.join(out_dir, "viewer_mesh_v2.npz"), verts=verts, faces=faces)

    z_peak = nb.Z_BASE + relief_mm
    metrics = {
        "master": {
            "V": stats["V"], "E": stats["E"], "F": stats["F"],
            "chi": stats["chi"], "watertight": stats["watertight"],
            "volume_cm3": stats["vol_cm3"],
        },
        "coverage": {"max_snow_steep": 0.0},
        "viewer": {"V": stats["V"], "F": stats["F"]},
    }
    with open(os.path.join(out_dir, "master_metrics_v2.json"), "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2)

    av2.OUT_DIR = out_dir
    av2.SCRATCH_DIR = out_dir
    av2.TARGET_PATHS = [viewer_html]
    av2.build_viewer()

    # Keep title/height label truthful to actual Z_PEAK
    with open(viewer_html, "r", encoding="utf-8") as hf:
        html = hf.read()
    html = html.replace("Cordillera Montaña Catan 3D (V3)", "Montaña Catan V4 (Nano Banana 360°)")
    html = html.replace("Z=83mm", f"Z={z_peak:.0f}mm")
    with open(viewer_html, "w", encoding="utf-8") as hf:
        hf.write(html)

    print(f"[V4 NanoBanana Ready] V={stats['V']} F={stats['F']} chi={stats['chi']} vol={stats['vol_cm3']:.2f}cm3 -> {viewer_html}")
    return viewer_html


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heightmap", required=True)
    p.add_argument("--albedo", default=None)
    p.add_argument("--relief-mm", type=float, default=64.0)
    p.add_argument("--nx", type=int, default=350)
    p.add_argument("--micro-detail", type=float, default=0.25)
    args = p.parse_args()
    build_v4_preview(args.heightmap, args.albedo, relief_mm=args.relief_mm, nx=args.nx, micro_detail_mm=args.micro_detail)


if __name__ == "__main__":
    main()
