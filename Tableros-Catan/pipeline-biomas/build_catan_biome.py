"""
Unified Pure NumPy/SciPy Catan Biome Pipeline (Nano Banana 6-View 360° -> Watertight CAD + WebGL PBR).
Reuses nanobanana_to_catan_hex.py and assemble_viewer_v2.py with zero HTML duplication.
"""
import argparse
import json
import math
import os
import sys
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SKILL_SCRIPTS = os.path.join(REPO_ROOT, ".agents", "skills", "nanobanana-3d-sculpt", "scripts")
MONTANA_PIPE = os.path.join(REPO_ROOT, "Tableros-Catan", "pipeline-montana")
sys.path.insert(0, SKILL_SCRIPTS)
sys.path.insert(0, MONTANA_PIPE)

import nanobanana_to_catan_hex as nb
import assemble_viewer_v2 as av2
import prototype_v3_cordillera as pv3


def refine_heightmap(hm_u8, gamma=0.80, blur_px=1.0):
    """Smooth JPEG step quantization and open steep slopes cleanly via power curve."""
    norm = hm_u8.astype(np.float32) / 255.0
    if blur_px > 0.0:
        norm = ndi.gaussian_filter(norm, sigma=blur_px)
    opened = np.power(np.clip(norm, 0.0, 1.0), gamma)
    return np.clip(opened * 255.0, 0.0, 255.0).astype(np.uint8)


def refine_biome_albedo(diffuse_rgb, Z, dist_edge, dist_vert, dx, dy, snow_filter=False, snowline_mm=16.0):
    """Enforce shadow readability, clean dark slate road collar, and optional physical snow clamp."""
    rgb = diffuse_rgb.astype(np.float32) / 255.0
    gy, gx = np.gradient(ndi.gaussian_filter(Z, 0.35 / dx), dy, dx)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    relief = Z - nb.Z_BASE

    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    sat = cmax - cmin
    max_snow_steep = 0.0

    if snow_filter:
        snow_prob = nb.smoothstep(0.46, 0.72, lum) * (1.0 - nb.smoothstep(0.07, 0.20, sat))
        snow_allowed = (1.0 - nb.smoothstep(38.0, 44.0, slope_deg)) * nb.smoothstep(snowline_mm - 3.0, snowline_mm + 4.0, relief)
        forbidden = np.clip(ndi.gaussian_filter(snow_prob * (1.0 - snow_allowed), sigma=2.5) * 1.15, 0.0, 1.0)
        if np.any(forbidden > 0.01):
            in_terrain = ((dist_edge > nb.ROAD_MARGIN + 1.0) & (dist_vert > nb.PAD_RADIUS + 1.0)).astype(np.float32)
            rock_weight = in_terrain * np.maximum(0.0, 1.0 - snow_prob * 1.4)
            w_sum = ndi.gaussian_filter(rock_weight, sigma=26.0) + 1e-4
            rock_fill = np.stack([ndi.gaussian_filter(rgb[:, :, c] * rock_weight, sigma=26.0) / w_sum for c in range(3)], axis=-1)
            H, W = Z.shape
            g1 = nb.fft_fbm(W, H, 1.4, 5.5, 2.0 * nb.R, 2.0 * nb.R, beta=2.0, seed=71)
            g2 = nb.fft_fbm(W, H, 0.45, 1.4, 2.0 * nb.R, 2.0 * nb.R, beta=1.8, seed=72)
            cracks = 0.65 * (1.0 - np.abs(g1)) + 0.35 * g2
            slope_shade = 0.86 + 0.20 * (1.0 - nb.smoothstep(20.0, 70.0, slope_deg))
            clean_rock = np.clip(rock_fill * (0.94 + 0.18 * cracks)[..., None] * slope_shade[..., None], 0.14, 0.64)
            rgb = rgb * (1.0 - forbidden[..., None]) + clean_rock * forbidden[..., None]

        cmax2 = np.max(rgb, axis=-1)
        lum2 = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        rem_snow = (lum2 > 0.68) & ((cmax2 - np.min(rgb, axis=-1)) < 0.12) & (slope_deg >= 44.0)
        max_snow_steep = float(np.mean(rem_snow)) * 100.0
    else:
        lum2 = lum

    # Gentle shadow recovery on crushed dark terrain pixels without touching outer border
    dark_t = (dist_edge > nb.ROAD_MARGIN + 0.8) & (lum2 < 0.22)
    rgb = np.clip(rgb * np.where(dark_t, 1.14, 1.0)[..., None], 0.0, 1.0)

    # Clean matte charcoal slate architectural plinth rim on 3.5mm road lane & Ø20mm pads
    collar = np.maximum(
        1.0 - nb.smoothstep(nb.ROAD_MARGIN - 0.3, nb.ROAD_MARGIN + 1.1, dist_edge),
        1.0 - nb.smoothstep(nb.PAD_RADIUS - 0.3, nb.PAD_RADIUS + 1.1, dist_vert),
    )[..., None]
    slate = np.array([0.19, 0.21, 0.24], dtype=np.float32)
    rgb = rgb * (1.0 - collar * 0.78) + slate * (collar * 0.78)
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8), max_snow_steep


def render_6_equidistant_views(hm_arr, diffuse_flip_u8, relief_mm, out_path, title):
    """Render gapless 3x2 studio turnaround every 60 deg (0, 60, 120, 180, 240, 300 deg)."""
    N = 900
    X, Y, Z, de, dv, dx, dy = nb.synthesize_dem_from_heightmap(
        hm_arr, N, N, relief_mm=relief_mm, micro_detail_mm=0.0, x_lim=nb.R, y_lim=nb.R
    )
    Z = apply_smooth_hex_apron(X, Y, Z, de, dv, relief_mm, apron_mm=9.0)
    diff = np.flipud(diffuse_flip_u8)
    col = np.clip((X + nb.R) / (2.0 * nb.R) * (diff.shape[1] - 1), 0, diff.shape[1] - 1).astype(np.int32)
    row = np.clip((Y + nb.R) / (2.0 * nb.R) * (diff.shape[0] - 1), 0, diff.shape[0] - 1).astype(np.int32)
    rgb = diff[row, col].astype(np.float32) / 255.0

    gy, gx = np.gradient(Z, dy, dx)
    nz = 1.0 / np.sqrt(1.0 + gx * gx + gy * gy)
    nx_n, ny_n = -gx * nz, -gy * nz
    ndotl = np.clip(nx_n * (-0.42) + ny_n * 0.50 + nz * 0.75, 0.0, 1.0)
    shaded = np.clip(rgb * (0.48 + 0.64 * ndotl)[..., None], 0.0, 1.0)

    # Extra intermediate cliff-skirt fill on steep drops so zero vertical splat holes appear
    z_min_nb = ndi.minimum_filter(Z, size=3)
    in_hex = de >= 0.0
    xs, ys, zs, cs = X[in_hex], Y[in_hex], Z[in_hex], shaded[in_hex]
    dz = zs - z_min_nb[in_hex]
    steep = dz > 0.7
    pts_list = [np.column_stack([xs, ys, zs])]
    cols_list = [cs]
    for f in (0.33, 0.66):
        pts_list.append(np.column_stack([xs[steep], ys[steep], zs[steep] - f * dz[steep]]))
        cols_list.append(cs[steep] * 0.92)

    ang = np.radians(30.0 + 60.0 * np.arange(6))
    cx, cyh = nb.R * np.cos(ang), nb.R * np.sin(ang)
    for e in range(6):
        t = np.linspace(0, 1, 420)
        wx, wy = cx[e] + t * (cx[(e + 1) % 6] - cx[e]), cyh[e] + t * (cyh[(e + 1) % 6] - cyh[e])
        enx, eny = (cyh[(e + 1) % 6] - cyh[e]), -(cx[(e + 1) % 6] - cx[e])
        nl = math.hypot(enx, eny)
        shade_w = max(0.25, 0.38 + 0.25 * ((enx / nl) * (-0.42) + (eny / nl) * 0.50))
        for wz in np.linspace(0, nb.Z_BASE, 130):
            pts_list.append(np.column_stack([wx, wy, np.full_like(wx, wz)]))
            cols_list.append(np.tile([0.20 * shade_w * 1.65, 0.22 * shade_w * 1.65, 0.25 * shade_w * 1.65], (len(wx), 1)))

    P_all = np.vstack(pts_list)
    C_all = np.vstack(cols_list)

    def render_cam(yaw_deg, pitch_deg=34.0, W=520, H=430):
        th, ph = math.radians(yaw_deg), math.radians(pitch_deg)
        cy, sy = math.cos(th), math.sin(th)
        cp, sp = math.cos(ph), math.sin(ph)
        xr = P_all[:, 0] * cy - P_all[:, 1] * sy
        yr = P_all[:, 0] * sy + P_all[:, 1] * cy
        zr = P_all[:, 2] - (18.0 + 0.25 * relief_mm)
        yp = yr * cp - zr * sp
        zp = yr * sp + zr * cp
        sc = 3.25
        px = np.round(W / 2 + xr * sc).astype(np.int32)
        py = np.round(H * 0.55 - zp * sc).astype(np.int32)
        valid = (px >= 1) & (px < W - 1) & (py >= 1) & (py < H - 1)
        px, py, yp, C = px[valid], py[valid], yp[valid], C_all[valid]
        order = np.argsort(-yp)
        img = np.full((H, W, 3), [0.90, 0.91, 0.93], dtype=np.float32)
        for dxo in (-1, 0, 1):
            for dyo in (-1, 0, 1):
                img[py[order] + dyo, px[order] + dxo] = C[order]
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        try:
            fnt = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
        except Exception:
            fnt = ImageFont.load_default()
        draw.text((16, 12), f"Vista {int(yaw_deg)} deg", fill=(35, 42, 54), font=fnt)
        return np.asarray(pil)

    angles = [0, 60, 120, 180, 240, 300]
    row1 = np.hstack([render_cam(a) for a in angles[:3]])
    row2 = np.hstack([render_cam(a) for a in angles[3:]])
    body = np.vstack([row1, row2])
    footer = np.full((38, body.shape[1], 3), (230, 233, 238), dtype=np.uint8)
    grid = Image.fromarray(np.vstack([body, footer]))
    draw = ImageDraw.Draw(grid)
    try:
        fnt_b = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 20)
    except Exception:
        fnt_b = ImageFont.load_default()
    draw.text((18, body.shape[0] + 7), f"{title} — 6 Vistas Equidistantes 60 deg (Z_PEAK={nb.Z_BASE+relief_mm:.1f}mm)", fill=(30, 41, 59), font=fnt_b)
    grid.save(out_path)
    return out_path


def apply_smooth_hex_apron(X, Y, Z, dist_edge, dist_vert, relief_mm, apron_mm=16.0):
    """Feather steep AI edge drops gradually over apron_mm so terrain flows naturally out of flat road margins."""
    d_clear = np.minimum(dist_edge - nb.ROAD_MARGIN, dist_vert - nb.PAD_RADIUS)
    taper = nb.smoothstep(0.0, apron_mm, d_clear)
    rel = np.maximum(0.0, Z - nb.Z_BASE) * taper
    rmax = float(np.max(rel))
    if rmax > 1e-5:
        rel *= (relief_mm / rmax)
    return (nb.Z_BASE + rel).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slug", required=True, help="Biome slug (e.g. bosque, colina, pasto, campo, desierto, oro, oceano, montana)")
    p.add_argument("--title", required=True, help="Viewer display title")
    p.add_argument("--mode", choices=["clay", "build"], required=True)
    p.add_argument("--heightmap", required=True)
    p.add_argument("--albedo", default=None)
    p.add_argument("--relief-mm", type=float, required=True)
    p.add_argument("--slope-gamma", type=float, default=0.80)
    p.add_argument("--blur-px", type=float, default=1.2)
    p.add_argument("--micro-detail", type=float, default=0.18)
    p.add_argument("--nx", type=int, default=340)
    p.add_argument("--tex-size", type=int, default=2048)
    p.add_argument("--snow-filter", action="store_true")
    p.add_argument("--auto-mineral-albedo", action="store_true", help="Fallback: paint Nano Banana DEM with V3 hyperreal mineral-runoff PBR shader.")
    p.add_argument("--viewer-html", default=None, help="Output HTML path relative to repo root")
    args = p.parse_args()

    out_dir = os.path.join("Tableros-Catan", "pipeline-biomas", args.slug)
    os.makedirs(out_dir, exist_ok=True)
    viewer_html = args.viewer_html or os.path.join("Visores-HTML", f"catan-bioma-{args.slug}.html")

    hm_raw = np.asarray(Image.open(args.heightmap).convert("L"))
    hm_arr = refine_heightmap(hm_raw, gamma=args.slope_gamma, blur_px=args.blur_px)

    Xt, Yt, Zt, dist_t, dist_vt, dxt, dyt = nb.synthesize_dem_from_heightmap(
        hm_arr, args.tex_size, args.tex_size, relief_mm=args.relief_mm, micro_detail_mm=args.micro_detail, x_lim=nb.R, y_lim=nb.R
    )
    Zt = apply_smooth_hex_apron(Xt, Yt, Zt, dist_t, dist_vt, args.relief_mm, apron_mm=9.0)
    clay_rgb = nb.render_shaded_clay_rgb(Zt, dist_t, dxt, dyt)
    clay_path = os.path.join(out_dir, f"{args.slug}_clay_overhead.png")
    nb.save_uv_aligned_png(clay_rgb, clay_path)

    if args.mode == "clay":
        print(f"[Clay Ready] {args.slug} (Z_PEAK={nb.Z_BASE + args.relief_mm:.1f}mm) -> {clay_path}")
        return

    if not args.albedo and not args.auto_mineral_albedo:
        p.error("--albedo or --auto-mineral-albedo is required in --mode build")

    if args.auto_mineral_albedo:
        rmf_m = pv3.ridged_multifractal_xy(Xt, Yt, seed=401)
        rmf_f = pv3.ridged_multifractal_xy(Xt, Yt, seed=402)
        broad = pv3.fft_fbm_canonical(Xt, Yt, 7.0, 24.0, beta=2.15, seed=404)
        z_scaled = nb.Z_BASE + (Zt - nb.Z_BASE) * (58.0 / max(args.relief_mm, 1.0))
        d_f32, _, _, slope_s, _, _, shader_stats = pv3.synthesize_hyperreal_textures(
            Xt, Yt, z_scaled, dist_t, dist_vt, dxt, dyt, rmf_m, rmf_f, broad
        )
        # Mild warm alpine sunlit boost + natural moss valley tint without flat opaque paint-over
        rel = np.maximum(0.0, Zt - nb.Z_BASE)
        in_area = nb.smoothstep(nb.ROAD_MARGIN + 0.4, nb.ROAD_MARGIN + 3.2, dist_t) * nb.smoothstep(nb.PAD_RADIUS + 0.4, nb.PAD_RADIUS + 3.2, dist_vt)
        meadow_tint = np.clip((1.0 - nb.smoothstep(3.0, args.relief_mm * 0.28, rel)) * (1.0 - nb.smoothstep(26.0, 44.0, slope_s)) * in_area, 0.0, 0.55)[..., None]
        warm_boost = np.array([1.08, 1.04, 0.96], dtype=np.float32)
        moss_col = np.array([0.28, 0.37, 0.17], dtype=np.float32)
        d_mix = np.clip(d_f32 * warm_boost, 0.0, 1.0)
        d_mix = np.clip(d_mix * (1.0 - meadow_tint * 0.48) + moss_col * (meadow_tint * 0.48), 0.0, 1.0)
        d_u8 = np.clip(d_mix * 255.0, 0, 255).astype(np.uint8)
        diff_rgb, max_snow_phys = refine_biome_albedo(d_u8, Zt, dist_t, dist_vt, dxt, dyt, snow_filter=True, snowline_mm=args.relief_mm * 0.38)
        max_snow = max(float(shader_stats.get("max_snow_steep", 0.0)), max_snow_phys)
        norm_u8, rough_u8 = nb.compute_pbr_maps(Zt, diff_rgb, dxt, dyt)
    else:
        alb_arr = np.asarray(Image.open(args.albedo).convert("RGB"))
        diff_raw = np.clip(nb.sample_square_domain(alb_arr, Xt, Yt), 0, 255).astype(np.uint8)
        diff_rgb, max_snow = refine_biome_albedo(diff_raw, Zt, dist_t, dist_vt, dxt, dyt, snow_filter=args.snow_filter)
        norm_u8, rough_u8 = nb.compute_pbr_maps(Zt, diff_rgb, dxt, dyt)

    nb.save_uv_aligned_png(diff_rgb, os.path.join(out_dir, "catan_mountain_diffuse_2k.png"))
    nb.save_uv_aligned_png(norm_u8, os.path.join(out_dir, "catan_mountain_normal_2k.png"))
    nb.save_uv_aligned_png(rough_u8, os.path.join(out_dir, "catan_mountain_roughness_2k.png"))

    ny = int(round(args.nx * (nb.R / nb.APOTHEM)))
    Xm, Ym, Zm, dist_m, dist_vm, _, _ = nb.synthesize_dem_from_heightmap(
        hm_arr, args.nx, ny, relief_mm=args.relief_mm, micro_detail_mm=args.micro_detail, x_lim=nb.APOTHEM, y_lim=nb.R
    )
    Zm = apply_smooth_hex_apron(Xm, Ym, Zm, dist_m, dist_vm, args.relief_mm, apron_mm=9.0)
    verts, faces = nb.build_solid_hex_mesh(Xm, Ym, Zm, dist_m, include_magnet=True)
    stats = nb.verify_watertight(verts, faces)
    if not stats["watertight"]:
        raise RuntimeError(f"{args.slug} mesh failed watertight check: {stats}")

    stl_path = os.path.join(out_dir, f"catan_{args.slug}.stl")
    nb.write_binary_stl(verts, faces, stl_path, label=f"Catan {args.slug}")
    np.savez_compressed(os.path.join(out_dir, "viewer_mesh_v2.npz"), verts=verts, faces=faces)

    audit_path = os.path.join(out_dir, f"{args.slug}_6views_audit.png")
    render_6_equidistant_views(hm_arr, np.flipud(diff_rgb), args.relief_mm, audit_path, args.title)

    z_peak = nb.Z_BASE + args.relief_mm
    metrics = {
        "master": {"V": stats["V"], "E": stats["E"], "F": stats["F"], "chi": stats["chi"], "watertight": True, "volume_cm3": stats["vol_cm3"]},
        "coverage": {"max_snow_steep": round(max_snow, 6)},
        "viewer": {"V": stats["V"], "F": stats["F"]},
    }
    with open(os.path.join(out_dir, "master_metrics_v2.json"), "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2)

    av2.OUT_DIR = out_dir
    av2.SCRATCH_DIR = out_dir
    av2.TARGET_PATHS = [viewer_html]
    av2.build_viewer()

    with open(viewer_html, "r", encoding="utf-8") as hf:
        html = hf.read()
    html = html.replace("Cordillera Montaña Catan 3D (V3)", args.title)
    html = html.replace("Z=83mm", f"Z={z_peak:.0f}mm")
    if not args.snow_filter:
        html = html.replace("<span>Nieve en ≥44°</span><strong>0.000000 (0%)</strong>", f"<span>Relieve Máx</span><strong>+{args.relief_mm:.1f} mm</strong>")
    with open(viewer_html, "w", encoding="utf-8") as hf:
        hf.write(html)

    print(f"[BIOME OK] {args.slug}: V={stats['V']} F={stats['F']} chi={stats['chi']} vol={stats['vol_cm3']:.2f}cm3 Z={z_peak:.1f}mm audit={audit_path} viewer={viewer_html}")


if __name__ == "__main__":
    main()
