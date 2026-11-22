"""
Prepara geometria Hero V4 enriquecida solo en cara oculta norte + guias Clay+PartialTexture
para completado 360° Nano Banana (montana_hero360 -> Blender Cycles Smart UV 3D).
"""
import argparse
import json
import math
import os
import sys
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import montana_sculpt_v4 as v4
import montana_granito_v5_sculpt as v5

V4_REFS = os.path.join(REPO_ROOT, "Tableros-Catan", "pipeline-biomas", "montana", "refs")
V4_WORK = os.path.join(REPO_ROOT, "Tableros-Catan", "pipeline-biomas", "montana", "work_v4")
OUT_DIR = os.path.join(REPO_ROOT, "Tableros-Catan", "pipeline-biomas", "montana_hero360")
GUIDES_DIR = os.path.join(OUT_DIR, "guides")
VIEWS_DIR = os.path.join(OUT_DIR, "views")

R = v4.R
Z_BASE = v4.Z_BASE


def build_v4_enhanced_north(Z_v4: np.ndarray, cam_p: np.ndarray) -> np.ndarray:
    """
    Conserva 100% intacta la zona frontal/lateral vista por camara 0° (Y <= 12 o vis_000)
    y enriquece solo la cara trasera norte oculta (Y > 14) para eliminar bandas planas en 180°.
    """
    N = Z_v4.shape[0]
    lin = np.linspace(-R, R, N)
    X, Y = np.meshgrid(lin, lin)
    dist_h = v4.hex_dist(X, Y)
    dist_v = v4.vert_dist(X, Y)
    in_area = v4.smoothstep(v4.ROAD_MARGIN + 0.3, v4.ROAD_MARGIN + 2.8, dist_h) * v4.smoothstep(
        v4.PAD_RADIUS + 0.2, v4.PAD_RADIUS + 2.5, dist_v
    )

    # Mascarar estrictamente lo invisible desde camara 0° en la mitad norte (Y > 11)
    _, _, _, hx, hy, hz, cam_xyz, nx_g, ny_g, nz_g = v4.ray_march_heightfield(Z_v4, cam_p, out_w=512, out_h=512)
    u_all, v_all, _ = v4.project_pts_vec(cam_p, X, Y, Z_v4, 512.0, 512.0)
    ui = np.clip(np.round(u_all), 0, 511).astype(np.int32)
    vi = np.clip(np.round(v_all), 0, 511).astype(np.int32)
    cam_d = np.sqrt((hx - cam_xyz[0]) ** 2 + (hy - cam_xyz[1]) ** 2 + (hz - cam_xyz[2]) ** 2)
    grid_d = np.sqrt((X - cam_xyz[0]) ** 2 + (Y - cam_xyz[1]) ** 2 + (Z_v4 - cam_xyz[2]) ** 2)
    is_front_vis = grid_d <= cam_d[vi, ui] + 0.6
    # Calcular tambien altura maxima del rayo visual 0° sobre cada celda (garantiza 0 cambios en skyline frontal!)
    dx_v = X - cam_xyz[0]
    dy_v = Y - cam_xyz[1]
    dz_h = np.maximum(hz[vi, ui] - cam_xyz[2], -500.0)
    t_ratio = np.hypot(dx_v, dy_v) / np.maximum(np.hypot(hx[vi, ui] - cam_xyz[0], hy[vi, ui] - cam_xyz[1]), 1e-4)
    z_ray_horizon = np.where(is_front_vis, Z_v4, cam_xyz[2] + dz_h * t_ratio)

    blind_north = (~is_front_vis) & (Y > 10.5)
    blind_w = ndi.gaussian_filter(blind_north.astype(np.float64), sigma=7.0) * v4.smoothstep(9.5, 16.5, Y) * in_area

    # Contrafuertes norte escalonados (36mm -> 6mm) con 2 valles nevados concavos para eliminar pared lisa de 80° en 180°
    spurs_n = np.zeros_like(Z_v4)
    north_ridges = [
        (0.5,   14.2, 36.5, -2.5,  32.0, 14.0, 13.5, 8.8, 0.92, 12.6, 0.16),
        (-2.5,  32.0, 14.0, -5.0,  42.0, 4.5,  8.8,  5.5, 0.94, 13.8, 0.14),
        (-13.5, 13.0, 32.0, -23.0, 29.0, 12.5, 12.0, 8.0, 0.92, 11.2, 0.16),
        (-23.0, 29.0, 12.5, -31.5, 36.5, 4.0,  8.0,  5.0, 0.95, 12.0, 0.14),
        (14.5,  12.5, 31.5, 22.5,  29.5, 11.5, 12.0, 8.0, 0.92, 14.1, 0.16),
        (22.5,  29.5, 11.5, 30.5,  37.0, 4.0,  8.0,  5.0, 0.95, 15.2, 0.14),
        (-24.0, 10.0, 21.0, -37.0, 21.5, 4.8,  9.2,  5.8, 0.96, 15.5, 0.14),
        (23.5,  10.0, 21.5, 35.5,  21.5, 4.8,  9.2,  5.8, 0.96, 16.9, 0.14),
    ]
    for r in north_ridges:
        spurs_n = v4.smax(spurs_n, v4.faceted_ridge(X, Y, *r), k=2.8)

    # Pequeñas crestas/afloramientos en pie de monte norte + depresion suave en las 2 vaguadas norte
    foot_n = v4.ridge_fbm(X, Y, [0.22, 0.56], [0.6, 0.4], seed=919)
    n_hills = v4.smooth_fbm(X, Y, [0.12, 0.30], [0.6, 0.4], seed=920)
    spurs_n = np.maximum(spurs_n, (1.6 + 2.8 * n_hills + 2.2 * foot_n) * v4.smoothstep(17.0, 31.0, Y))
    bowl_nw = 5.0 * np.exp(-0.5 * (((X + 11.5) / 5.5) ** 2 + ((Y - 29.0) / 6.8) ** 2))
    bowl_ne = 5.0 * np.exp(-0.5 * (((X - 9.5) / 5.5) ** 2 + ((Y - 29.5) / 6.8) ** 2))
    spurs_n = np.maximum(1.2, spurs_n - (bowl_nw + bowl_ne)) * in_area

    Z_n_cand = np.maximum(Z_v4, Z_BASE + spurs_n)
    Z_cap = np.minimum(float(Z_v4.max()) - 0.8, z_ray_horizon - 0.6)
    Z_n_cand = np.minimum(Z_n_cand, Z_cap)

    Z_out = Z_v4 * (1.0 - blind_w) + np.maximum(Z_v4, Z_n_cand) * blind_w
    return Z_out.astype(np.float64)


def composite_partial_view(Z_2k, cam_p, proj_list, out_path, yaw_deg=180.0, size=1024):
    """
    Renderiza vista Clay a yaw_deg y pinta sobre los pixeles visibles cualquiera de las
    vistas ya completadas en proj_list = [(delit_2k, weight_2k), ...].
    """
    view_p = v5.compute_cam_p_angle(cam_p, yaw_offset_deg=-yaw_deg)
    clay_img, any_hit, hit_ter, hx, hy, hz, _, _, _, _ = v4.ray_march_heightfield(
        Z_2k, view_p, out_w=size, out_h=size, tex_rgb=None, yaw_offset_deg=0.0, sun_rel=True
    )
    tex_acc = np.zeros((size, size, 3), dtype=np.float32)
    w_acc = np.zeros((size, size), dtype=np.float32)
    N2 = Z_2k.shape[0]
    inv_span = (N2 - 1) / (2.0 * R)

    if np.any(hit_ter):
        tx = np.clip((hx[hit_ter] + R) * inv_span, 0, N2 - 1.001)
        ty = np.clip((hy[hit_ter] + R) * inv_span, 0, N2 - 1.001)
        ix = tx.astype(np.int32)
        iy = ty.astype(np.int32)
        for col_2k, wt_2k in proj_list:
            w_p = wt_2k[iy, ix]
            c_p = col_2k[iy, ix]
            tex_acc[hit_ter] += c_p * w_p[:, None]
            w_acc[hit_ter] += w_p

    out = np.full((size, size, 3), np.array([0.895, 0.902, 0.910], dtype=np.float32), dtype=np.float32)
    if np.any(hit_ter):
        w_n = np.clip(w_acc, 0.0, 1.0)
        shaded_tex = np.where(w_acc[..., None] > 1e-4, tex_acc / np.maximum(w_acc[..., None], 1e-4), clay_img)
        # Iluminar suave para que empaste con el clay neutro
        ter_col = clay_img * (1.0 - w_n[..., None]) + np.clip(shaded_tex * (0.62 + 0.45 * clay_img), 0.0, 1.0) * w_n[..., None]
        out[hit_ter] = ter_col[hit_ter]
    plinth = any_hit & ~hit_ter
    out[plinth] = np.array([0.16, 0.175, 0.195], dtype=np.float32)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(out_path)
    return view_p, any_hit, hit_ter


def run_prep180():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(GUIDES_DIR, exist_ok=True)
    os.makedirs(VIEWS_DIR, exist_ok=True)

    cam_p = np.load(os.path.join(V4_REFS, "hero_cam_params.npy"))
    fg_hero = np.load(os.path.join(V4_REFS, "hero_fg_clean.npy"))
    hero_rgb = np.asarray(Image.open(os.path.join(V4_REFS, "hero_nanobanana.jpg")).convert("RGB"), dtype=np.float32) / 255.0
    Z_v4 = np.load(os.path.join(V4_WORK, "Z_geom_v4.npy")).astype(np.float64)

    Z_enh = build_v4_enhanced_north(Z_v4, cam_p)
    np.save(os.path.join(OUT_DIR, "Z_hero360.npy"), Z_enh.astype(np.float32))

    # Copiar hero_nanobanana como vista 000 canonica
    Image.fromarray((hero_rgb * 255).astype(np.uint8)).save(os.path.join(VIEWS_DIR, "view_000.png"))

    # Auditar que 0° sigue identico al Hero y revisar 6 vistas Clay
    clay_1024, _, _, _, _, _, _, _, _, _ = v4.ray_march_heightfield(Z_enh, cam_p, out_w=1024, out_h=1024, sun_rel=False)
    pair = np.hstack([
        (np.clip(clay_1024, 0, 1) * 255).astype(np.uint8),
        (np.clip(hero_rgb, 0, 1) * 255).astype(np.uint8),
    ])
    Image.fromarray(pair).resize((1400, 700), Image.LANCZOS).save(os.path.join(OUT_DIR, "clay_vs_hero.png"))

    v6 = []
    for ang in [0, -60, -120, -180, -240, -300]:
        ci, _, _, _, _, _, _, _, _, _ = v4.ray_march_heightfield(Z_enh, cam_p, out_w=460, out_h=460, yaw_offset_deg=ang, sun_rel=True)
        pil = Image.fromarray((np.clip(ci, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(pil)
        try:
            fnt = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
        except Exception:
            fnt = ImageFont.load_default()
        dr.text((14, 12), f"{(-ang) % 360} deg", fill=(30, 40, 52), font=fnt)
        v6.append(np.asarray(pil))
    Image.fromarray(np.vstack([np.hstack(v6[:3]), np.hstack(v6[3:])])).save(os.path.join(OUT_DIR, "clay_6views.png"))

    # Proyectar vista 000 sobre 2K para construir la guia partial_view_180.png
    N2 = 2048
    Z_2k = ndi.zoom(Z_enh, N2 / Z_enh.shape[0], order=3)
    lin = np.linspace(-R, R, N2)
    X, Y = np.meshgrid(lin, lin)
    dist_h = v4.hex_dist(X, Y)
    dist_v = v4.vert_dist(X, Y)
    dx = 2.0 * R / (N2 - 1)

    delit_0, vis_w0, _, _ = v5.project_view_clean(
        Z_2k, X, Y, dist_h, dist_v, dx, hero_rgb, cam_p, min_ndotv=0.22, erode_px=4, is_hero=True
    )
    # Bloquear manchas rasantes del horizonte en Y > 18 al renderizar partial_view_180
    vis_w0 = vis_w0 * v4.smoothstep(22.0, 8.0, Y) * v4.smoothstep(4.8, 6.2, dist_h)
    np.savez_compressed(os.path.join(OUT_DIR, "proj_000.npz"), delit=delit_0.astype(np.float32), vis_w=vis_w0.astype(np.float32))

    # Renderizar partial_view_180.png (con margenes ya pintados por 0° + arcilla neutra detras)
    composite_partial_view(Z_2k, cam_p, [(delit_0, vis_w0)], os.path.join(GUIDES_DIR, "partial_view_180.png"), yaw_deg=180.0, size=1024)
    v5.render_clay_turnaround(Z_2k, cam_p, os.path.join(GUIDES_DIR, "clay_180.png"), yaw_offset_deg=-180.0, size=1024)
    print(f"[Prep180 Done] front_cov={(vis_w0 > 0.2).mean():.3f} -> {GUIDES_DIR}")


def run_prep_sides(view_180_path: str):
    cam_p = np.load(os.path.join(V4_REFS, "hero_cam_params.npy"))
    Z_enh = np.load(os.path.join(OUT_DIR, "Z_hero360.npy")).astype(np.float64)
    N2 = 2048
    Z_2k = ndi.zoom(Z_enh, N2 / Z_enh.shape[0], order=3)
    lin = np.linspace(-R, R, N2)
    X, Y = np.meshgrid(lin, lin)
    dist_h = v4.hex_dist(X, Y)
    dist_v = v4.vert_dist(X, Y)
    dx = 2.0 * R / (N2 - 1)

    p0 = np.load(os.path.join(OUT_DIR, "proj_000.npz"))
    delit_0, vis_w0 = p0["delit"], p0["vis_w"]

    v180_rgb = np.asarray(Image.open(view_180_path).convert("RGB"), dtype=np.float32) / 255.0
    p180_cam = v5.compute_cam_p_angle(cam_p, yaw_offset_deg=-180.0)
    delit_180, vis_w180, _, _ = v5.project_view_clean(
        Z_2k, X, Y, dist_h, dist_v, dx, v180_rgb, p180_cam, min_ndotv=0.16, erode_px=4, is_hero=False
    )
    np.savez_compressed(os.path.join(OUT_DIR, "proj_180.npz"), delit=delit_180.astype(np.float32), vis_w=vis_w180.astype(np.float32))

    # Guardar tambien view_180 alineada a silueta
    _, any_hit_180, _, _, _, _, _, _, _, _ = v4.ray_march_heightfield(Z_2k, p180_cam, out_w=1024, out_h=1024)
    aligned_180, iou_b, iou_a = v5.align_nb_view_to_clay(v180_rgb, any_hit_180)
    print(f"[View 180 Aligned] IoU {iou_b:.4f} -> {iou_a:.4f}")
    Image.fromarray((np.clip(aligned_180, 0, 1) * 255).astype(np.uint8)).save(os.path.join(VIEWS_DIR, "view_180.png"))

    for ang in [60, 120, 240, 300]:
        composite_partial_view(
            Z_2k, cam_p, [(delit_0, vis_w0), (delit_180, vis_w180)],
            os.path.join(GUIDES_DIR, f"partial_view_{ang:03d}.png"), yaw_deg=float(ang), size=1024
        )
        v5.render_clay_turnaround(Z_2k, cam_p, os.path.join(GUIDES_DIR, f"clay_{ang:03d}.png"), yaw_offset_deg=-float(ang), size=1024)
    print(f"[PrepSides Done] Saved 060/120/240/300 partial guides in {GUIDES_DIR}")


def _cam_basis_dict(cam_p: np.ndarray, size: int = 1024) -> dict:
    az_deg, el_deg, dist, tx, ty, tz, fov_deg, roll_deg = [float(x) for x in cam_p]
    az, el = math.radians(az_deg), math.radians(el_deg)
    cam = np.array([tx + dist * math.cos(el) * math.sin(az), ty - dist * math.cos(el) * math.cos(az), tz + dist * math.sin(el)], dtype=np.float64)
    fwd = np.array([tx, ty, tz], dtype=np.float64) - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    right_r = cr * right + sr * up
    up_r = -sr * right + cr * up
    return {
        "cam_xyz": cam.tolist(),
        "fwd": fwd.tolist(),
        "right_r": right_r.tolist(),
        "up_r": up_r.tolist(),
        "fov_deg": float(fov_deg),
        "W": int(size),
        "H": int(size),
    }


def run_export_3d(nx_mesh: int = 768):
    """
    Alinea las 6 vistas 360°, genera mascaras erosionadas (cero halos/zocalo), exporta cameras.json
    y construye tile_base.stl estanco unido al troquel B-Rep exacto (build123d + manifold3d).
    """
    import manifold3d as m3d
    import trimesh
    from catan_kit.build_catan_base_brep import build_catan_base, build_catan_inner_cutter, export_stl
    from blender.catan_3d_factory_csg import build_terrain_bed_manifold, _trimesh_to_manifold, _manifold_to_trimesh

    cam_p = np.load(os.path.join(V4_REFS, "hero_cam_params.npy"))
    Z_enh = np.load(os.path.join(OUT_DIR, "Z_hero360.npy")).astype(np.float64)
    N2 = 2048
    Z_2k = ndi.zoom(Z_enh, N2 / Z_enh.shape[0], order=3)

    views_2k_dir = os.path.join(OUT_DIR, "views_2k")
    os.makedirs(views_2k_dir, exist_ok=True)
    rng = np.random.default_rng(2026)

    cams_out = {}
    ref_mean_luma = None
    for ang in [0, 60, 120, 180, 240, 300]:
        is_hero = (ang == 0)
        p_ang = cam_p.copy() if is_hero else v5.compute_cam_p_angle(cam_p, yaw_offset_deg=-float(ang))
        _, any_hit, hit_ter, hx, hy, hz, _, _, _, _ = v4.ray_march_heightfield(Z_2k, p_ang, out_w=1024, out_h=1024)

        if is_hero:
            src_path = os.path.join(VIEWS_DIR, "view_000.png")
            rgb = np.asarray(Image.open(src_path).convert("RGB"), dtype=np.float32) / 255.0
            iou_a = 1.0
        else:
            cand_jpg = os.path.join(VIEWS_DIR, f"view_{ang:03d}_nb.jpg")
            cand_png = os.path.join(VIEWS_DIR, f"view_{ang:03d}_nb.png")
            src_path = cand_jpg if os.path.exists(cand_jpg) else cand_png
            raw_rgb = np.asarray(Image.open(src_path).convert("RGB"), dtype=np.float32) / 255.0
            rgb, iou_b, iou_a = v5.align_nb_view_to_clay(raw_rgb, any_hit)
            print(f"  [Align {ang:03d}] IoU {iou_b:.4f} -> {iou_a:.4f}")

        # Mascara estricta del terreno interior sin el bisel metalico gris de las vistas NB
        hit_safe = ndi.binary_erosion(hit_ter, iterations=4 if not is_hero else 3)
        lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        sat = np.max(rgb, axis=-1) - np.min(rgb, axis=-1)
        rel_z = np.maximum(0.0, hz - Z_BASE)
        warm = rgb[..., 0] + rgb[..., 1] - 2.0 * rgb[..., 2]
        dh = v4.hex_dist(hx, hy)
        dv = v4.vert_dist(hx, hy)
        is_slate = (warm < 0.05) & (lum < 0.48) & (sat < 0.11) & (rel_z < 3.8)
        if not is_hero:
            is_inner_metal = ((dh < v4.ROAD_MARGIN + 2.45) | (dv < v4.PAD_RADIUS + 2.0)) & (sat < 0.16) & (lum < 0.54)
            is_slate = is_slate | is_inner_metal | (dh < v4.ROAD_MARGIN + 0.65)
        is_sky_bleed = (lum > 0.70) & (sat < 0.06) & (rel_z < 4.5)
        mask_ter = hit_safe & (~is_slate) & (~is_sky_bleed)

        # Armonizacion global suave de exposicion respecto a hero 000 dentro del terreno valido
        if np.any(mask_ter):
            ml = float(np.mean(lum[mask_ter]))
            if ref_mean_luma is None:
                ref_mean_luma = ml
            else:
                gain = float(np.clip(ref_mean_luma / max(ml, 1e-4), 0.88, 1.14))
                rgb = np.clip(rgb * gain, 0.0, 1.0)

        out_rgb = os.path.join(VIEWS_DIR, f"view_{ang:03d}.png")
        out_msk = os.path.join(VIEWS_DIR, f"mask_{ang:03d}.png")
        Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).save(out_rgb)
        msk_soft = ndi.gaussian_filter(mask_ter.astype(np.float32), sigma=1.2)
        Image.fromarray((np.clip(msk_soft, 0.0, 1.0) * 255.0).astype(np.uint8)).save(out_msk)

        # Super-resolucion 2048x2048 + Unsharp multiescala + micro-grano mineral (sin ensuciar nieve glaciar)
        rgb_2k_pil = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).resize((2048, 2048), Image.Resampling.LANCZOS)
        arr_2k = np.asarray(rgb_2k_pil, dtype=np.float32) / 255.0
        lum_2k = 0.2126 * arr_2k[..., 0] + 0.7152 * arr_2k[..., 1] + 0.0722 * arr_2k[..., 2]
        sat_2k = np.max(arr_2k, axis=-1) - np.min(arr_2k, axis=-1)
        rel_z_2k = ndi.zoom(rel_z, 2.0, order=1)
        snow_2k = v4.smoothstep(0.68, 0.84, lum_2k) * (1.0 - v4.smoothstep(0.08, 0.18, sat_2k)) * v4.smoothstep(3.0, 7.5, rel_z_2k)
        blur_f = ndi.gaussian_filter(arr_2k, (1.3, 1.3, 0.0))
        blur_m = ndi.gaussian_filter(arr_2k, (3.2, 3.2, 0.0))
        sharp_w = (1.0 - 0.85 * snow_2k)[..., None]
        arr_2k = np.clip(arr_2k + sharp_w * (0.44 * (arr_2k - blur_f) + 0.22 * (arr_2k - blur_m)), 0.0, 1.0)

        w_noise = rng.normal(0.0, 1.0, (2048, 2048)).astype(np.float32)
        grain_f = ndi.gaussian_filter(w_noise, 0.85) - ndi.gaussian_filter(w_noise, 2.4)
        grain_f /= max(float(np.std(grain_f)), 1e-4)
        msk_2k_arr = np.asarray(
            Image.fromarray((np.clip(msk_soft, 0.0, 1.0) * 255.0).astype(np.uint8)).resize((2048, 2048), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        rock_grain = (1.0 - snow_2k) * msk_2k_arr * 0.036 * grain_f
        arr_2k = np.clip(arr_2k + rock_grain[..., None], 0.0, 1.0)

        out_rgb_2k = os.path.join(views_2k_dir, f"view_{ang:03d}.png")
        out_msk_2k = os.path.join(views_2k_dir, f"mask_{ang:03d}.png")
        Image.fromarray((arr_2k * 255.0).astype(np.uint8)).save(out_rgb_2k)
        Image.fromarray((np.clip(msk_2k_arr, 0.0, 1.0) * 255.0).astype(np.uint8)).save(out_msk_2k)

        cdict = _cam_basis_dict(p_ang, size=2048)
        cdict["angle_deg"] = ang
        cdict["image"] = out_rgb_2k
        cdict["mask"] = out_msk_2k
        cdict["iou"] = round(float(iou_a), 4)
        cams_out[str(ang)] = cdict

    cam_json = os.path.join(OUT_DIR, "cameras.json")
    with open(cam_json, "w", encoding="utf-8") as f:
        json.dump(cams_out, f, indent=2)
    print(f"[Export3D] Saved {cam_json}")

    # Construir tile_base.stl estanco unido al zocalo B-Rep canon (build123d + manifold3d)
    work_brep = os.path.join(OUT_DIR, "work_brep")
    os.makedirs(work_brep, exist_ok=True)
    base_solid, _ = build_catan_base("circle")
    cutter_solid = build_catan_inner_cutter(road_margin=v4.ROAD_MARGIN, pad_radius=v4.PAD_RADIUS, z_min=20.0, z_max=120.0)
    base_stl = os.path.join(work_brep, "catan_base_brep.stl")
    cutter_stl = os.path.join(work_brep, "catan_inner_cutter.stl")
    export_stl(base_solid, base_stl)
    export_stl(cutter_solid, cutter_stl)

    base_mf = _trimesh_to_manifold(trimesh.load(base_stl, force="mesh"))
    cutter_mf = _trimesh_to_manifold(trimesh.load(cutter_stl, force="mesh"))

    Z_mesh = ndi.zoom(Z_enh, nx_mesh / Z_enh.shape[0], order=3)
    lin = np.linspace(-R, R, nx_mesh, dtype=np.float64)
    Xm, Ym = np.meshgrid(lin, lin)
    in_border = v4.smoothstep(v4.ROAD_MARGIN + 0.10, v4.ROAD_MARGIN + 2.0, v4.hex_dist(Xm, Ym)) * v4.smoothstep(
        v4.PAD_RADIUS + 0.10, v4.PAD_RADIUS + 1.8, v4.vert_dist(Xm, Ym)
    )
    Z_mesh = Z_BASE + np.maximum(0.0, Z_mesh - Z_BASE) * in_border
    bed_mf = build_terrain_bed_manifold(nx_mesh, Z_mesh, z_bot=Z_BASE - 2.6)
    full_solid = base_mf + (bed_mf ^ cutter_mf)
    tm_out = _manifold_to_trimesh(full_solid)
    comps = tm_out.split(only_watertight=False)
    if len(comps) > 1:
        comps.sort(key=lambda c: abs(float(c.volume)), reverse=True)
        tm_out = comps[0]
    assert tm_out.is_watertight, "tile_base.stl is not watertight!"
    out_stl = os.path.join(OUT_DIR, "tile_base.stl")
    tm_out.export(out_stl)
    print(f"[Export3D Done] {out_stl}: {len(tm_out.vertices)} V / {len(tm_out.faces)} F, chi={tm_out.euler_number}, Z=[{tm_out.bounds[0][2]:.2f}, {tm_out.bounds[1][2]:.2f}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["prep180", "prep-sides", "export-3d"], required=True)
    p.add_argument("--view-180", default=os.path.join(VIEWS_DIR, "view_180_nb.png"))
    p.add_argument("--nx-mesh", type=int, default=768)
    args = p.parse_args()
    if args.stage == "prep180":
        run_prep180()
    elif args.stage == "prep-sides":
        run_prep_sides(args.view_180)
    elif args.stage == "export-3d":
        run_export_3d(nx_mesh=args.nx_mesh)


if __name__ == "__main__":
    main()
