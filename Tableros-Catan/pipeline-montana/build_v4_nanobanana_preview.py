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
import scipy.ndimage as ndi


def refine_heightmap_array(hm_u8, gamma=0.72):
    """Open steep canyon walls (gamma < 1.0) and smooth micro-spikes so AI slopes hold crisp UV texels."""
    norm = hm_u8.astype(np.float32) / 255.0
    # Slight smoothing first eliminates JPEG step ringing before opening mid-slopes
    norm = ndi.gaussian_filter(norm, sigma=1.1)
    opened = np.power(np.clip(norm, 0.0, 1.0), gamma)
    return np.clip(opened * 255.0, 0.0, 255.0).astype(np.uint8)


def refine_albedo_physics(diffuse_rgb, Z, dist_edge, dist_vert, dx, dy, snowline_mm=18.0):
    """
    1) Remove Nano Banana snow painted onto steep cliffs (>=44 deg) or low gullies (< snowline_mm)
       by smoothly pulling in surrounding Nano Banana mineral granite/scree colors.
    2) Lift crushed black shadows on sunlit granite faces & unify architectural dark slate plinth collar.
    """
    rgb = diffuse_rgb.astype(np.float32) / 255.0
    gy, gx = np.gradient(ndi.gaussian_filter(Z, 0.35 / dx), dy, dx)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    relief = Z - nb.Z_BASE

    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    sat = cmax - cmin

    # Continuous soft snow probability (avoids sharp white halo rings on snow borders!)
    snow_prob = nb.smoothstep(0.46, 0.72, lum) * (1.0 - nb.smoothstep(0.07, 0.20, sat))
    snow_allowed = (1.0 - nb.smoothstep(38.0, 44.0, slope_deg)) * nb.smoothstep(snowline_mm - 3.0, snowline_mm + 4.0, relief)
    forbidden_snow = ndi.gaussian_filter(snow_prob * (1.0 - snow_allowed), sigma=2.5)
    forbidden_snow = np.clip(forbidden_snow * 1.15, 0.0, 1.0)

    if np.any(forbidden_snow > 0.01):
        # Extract surrounding rock/scree color via normalized Gaussian inpainting from low-snow terrain
        in_terrain = ((dist_edge > nb.ROAD_MARGIN + 1.0) & (dist_vert > nb.PAD_RADIUS + 1.0)).astype(np.float32)
        rock_weight = in_terrain * np.maximum(0.0, 1.0 - snow_prob * 1.4)
        sig = 26.0
        w_sum = ndi.gaussian_filter(rock_weight, sigma=sig) + 1e-4
        rock_fill = np.stack([
            ndi.gaussian_filter(rgb[:, :, c] * rock_weight, sigma=sig) / w_sum for c in range(3)
        ], axis=-1)
        # Synthesize crisp alpine granite striation + micro-grain from geometry + multi-scale noise
        H, W = Z.shape
        g1 = nb.fft_fbm(W, H, 1.4, 5.5, 2.0 * nb.R, 2.0 * nb.R, beta=2.0, seed=71)
        g2 = nb.fft_fbm(W, H, 0.45, 1.4, 2.0 * nb.R, 2.0 * nb.R, beta=1.8, seed=72)
        cracks = 0.65 * (1.0 - np.abs(g1)) + 0.35 * g2
        slope_shade = 0.85 + 0.22 * (1.0 - nb.smoothstep(20.0, 70.0, slope_deg))
        clean_rock = np.clip(rock_fill * (0.94 + 0.18 * cracks)[..., None] * slope_shade[..., None], 0.14, 0.62)
        alpha_clean = forbidden_snow[..., None]
        rgb = rgb * (1.0 - alpha_clean) + clean_rock * alpha_clean

    # Subtle shadow lift on steep sunlit granite slabs so dark South crags remain readable
    dark_rock = (snow_prob < 0.25) & (dist_edge > nb.ROAD_MARGIN + 0.8) & (lum < 0.24)
    lift = np.where(dark_rock, 1.18, 1.0)[..., None]
    rgb = np.clip(rgb * lift, 0.0, 1.0)

    # Clean architectural dark basalt-slate rim on 3.5mm road lane, Ø20mm pads & outer UV corner (0.015,0.015)
    collar_mask = np.maximum(
        1.0 - nb.smoothstep(nb.ROAD_MARGIN - 0.3, nb.ROAD_MARGIN + 1.2, dist_edge),
        1.0 - nb.smoothstep(nb.PAD_RADIUS - 0.3, nb.PAD_RADIUS + 1.2, dist_vert),
    )[..., None]
    slate_col = np.array([0.19, 0.21, 0.24], dtype=np.float32)
    rgb = rgb * (1.0 - collar_mask * 0.72) + slate_col * (collar_mask * 0.72)

    # Telemetry check: max snow fraction remaining >= 44 deg
    cmax2 = np.max(rgb, axis=-1)
    lum2 = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    rem_snow = (lum2 > 0.68) & ((cmax2 - np.min(rgb, axis=-1)) < 0.12) & (slope_deg >= 44.0)
    max_snow_steep = float(np.mean(rem_snow)) * 100.0
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8), max_snow_steep


def build_v4_preview(heightmap_path, albedo_path=None, out_dir="Tableros-Catan/pipeline-montana/scratch_v4_nanobanana",
                     viewer_html="Visores-HTML/preview-montana-v4-nanobanana.html", relief_mm=60.0, nx=360, tex_size=2048,
                     micro_detail_mm=0.20, slope_gamma=0.72):
    os.makedirs(out_dir, exist_ok=True)
    hm_raw = np.asarray(Image.open(heightmap_path).convert("L"))
    hm_arr = refine_heightmap_array(hm_raw, gamma=slope_gamma)
    albedo_arr = np.asarray(Image.open(albedo_path).convert("RGB")) if albedo_path else None

    Xt, Yt, Zt, dist_t, dist_vt, dxt, dyt = nb.synthesize_dem_from_heightmap(
        hm_arr, tex_size, tex_size, relief_mm=relief_mm, micro_detail_mm=micro_detail_mm, x_lim=nb.R, y_lim=nb.R
    )
    clay_rgb = nb.render_shaded_clay_rgb(Zt, dist_t, dxt, dyt)
    clay_path = os.path.join(out_dir, "v4_nanobanana_clay_overhead.png")
    nb.save_uv_aligned_png(clay_rgb, clay_path)

    if albedo_arr is None:
        print(f"[Clay Ready] Inspect & pass to image-generator for painterly albedo: {clay_path}")
        return clay_path

    diffuse_raw = np.clip(nb.sample_square_domain(albedo_arr, Xt, Yt), 0, 255).astype(np.uint8)
    diffuse_rgb, max_snow_steep = refine_albedo_physics(diffuse_raw, Zt, dist_t, dist_vt, dxt, dyt)
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
        "coverage": {"max_snow_steep": round(max_snow_steep, 6)},
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
    html = html.replace("Cordillera Montaña Catan 3D (V3)", "Montaña Catan V4 Refinada (Nano Banana 360°)")
    html = html.replace("Z=83mm", f"Z={z_peak:.0f}mm")
    with open(viewer_html, "w", encoding="utf-8") as hf:
        hf.write(html)

    print(f"[V4 NanoBanana Ready] V={stats['V']} F={stats['F']} chi={stats['chi']} vol={stats['vol_cm3']:.2f}cm3 max_snow_ge44={max_snow_steep:.6f}% -> {viewer_html}")
    return viewer_html


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heightmap", required=True)
    p.add_argument("--albedo", default=None)
    p.add_argument("--relief-mm", type=float, default=60.0)
    p.add_argument("--nx", type=int, default=360)
    p.add_argument("--micro-detail", type=float, default=0.20)
    p.add_argument("--slope-gamma", type=float, default=0.72)
    args = p.parse_args()
    build_v4_preview(args.heightmap, args.albedo, relief_mm=args.relief_mm, nx=args.nx,
                     micro_detail_mm=args.micro_detail, slope_gamma=args.slope_gamma)


if __name__ == "__main__":
    main()
