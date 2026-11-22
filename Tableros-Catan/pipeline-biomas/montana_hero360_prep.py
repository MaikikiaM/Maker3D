"""
Prepara geometria Hero V4 enriquecida solo en cara oculta norte + guias Clay+PartialTexture
para completado 360° Nano Banana (montana_hero360 -> Blender Cycles Smart UV 3D).
"""
import argparse
import math
import os
import sys
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["prep180", "prep-sides"], required=True)
    p.add_argument("--view-180", default=os.path.join(VIEWS_DIR, "view_180_nb.png"))
    args = p.parse_args()
    if args.stage == "prep180":
        run_prep180()
    elif args.stage == "prep-sides":
        run_prep_sides(args.view_180)


if __name__ == "__main__":
    main()
