"""
Montaña Centrada — Reconstrucción 3D fotorrealista 100% sincronizada 2D↔3D desde
la lámina dual-view de alta resolución (`Gemini_Generated_Image_ytaogxytaogxytao.jpg`, 2752x1536):
  * Panel izquierdo: FRONT 3/4 VIEW (0°)  -> az ≈ 30°
  * Panel derecho:   BACK 3/4 VIEW (180°) -> az ≈ 210°

Etapas:
  --stage calib : Calibración subpíxel de ambas cámaras contra el zócalo hexagonal + pads.
  --stage prep  : Construcción del campo de alturas 3D alineado a ambas siluetas, extracción
                  dual del albedo cenital 4K sin sombras y mapa de esculpido físico sincronizado.
"""
import argparse
import math
import os
import sys

import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import minimize
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import montana_sculpt_v4 as m4  # noqa: E402

R = m4.R
APOTHEM = m4.APOTHEM
Z_BASE = m4.Z_BASE
ROAD_MARGIN = m4.ROAD_MARGIN
PAD_RADIUS = m4.PAD_RADIUS
smoothstep = m4.smoothstep
smax = m4.smax

CENTRADA_DIR = os.path.join(HERE, "montana_centrada")
REFS_DIR = os.path.join(CENTRADA_DIR, "refs")
WORK_DIR = os.path.join(CENTRADA_DIR, "work")
DUAL_REF = os.path.join(REFS_DIR, "concept_dual_fullres.jpg")

PAN_W = 1368
PAN_H = 1536


def load_panels():
    im = np.asarray(Image.open(DUAL_REF).convert("RGB"), dtype=np.float32) / 255.0
    front = im[:, :PAN_W]
    back = im[:, 2752 - PAN_W :]
    return front, back


def find_panel_landmarks(panel_rgb):
    """Detecta los puntos de referencia del zócalo hexagonal en un panel 1368x1536."""
    lum = 0.2126 * panel_rgb[..., 0] + 0.7152 * panel_rgb[..., 1] + 0.0722 * panel_rgb[..., 2]
    H, W = lum.shape

    # Borde exterior oscuro del zócalo vs fondo gris claro (~0.83..0.90)
    plinth_or_terrain = ndi.binary_fill_holes(lum < 0.72)
    # Limpiar texto inferior
    plinth_or_terrain[int(H * 0.87) :, :] = False
    lbl, n_l = ndi.label(plinth_or_terrain)
    if n_l > 0:
        cnt = np.bincount(lbl.ravel())
        cnt[0] = 0
        main_mask = lbl == int(np.argmax(cnt))
    else:
        main_mask = plinth_or_terrain

    # Detectar borde inferior del plinto en X central [420..940] (Z=0) y borde del bisel superior (Z=Z_BASE)
    xs_bot = np.arange(450, 920)
    y_bot_base = np.array([np.where(main_mask[:, x])[0].max() for x in xs_bot], dtype=np.float64)

    # Línea exacta del borde exterior superior frontal del plinto (caída brusca 0.41 -> 0.18 entre cara superior y pared frontal)
    y_bevel = []
    for x in xs_bot:
        yb = int(y_bot_base[x - 450])
        col = ndi.gaussian_filter1d(lum[yb - 160 : yb - 65, x], 1.5)
        idx_drop = int(np.argmin(np.gradient(col)))
        y_bevel.append(yb - 160 + idx_drop)
    y_bevel = np.asarray(y_bevel, dtype=np.float64)

    # Detectar los 4 centros de los cajeados circulares en las esquinas FL, FR, BL, BR
    def detect_socket(x0, x1, y0, y1):
        sub = lum[y0:y1, x0:x1]
        dark = sub < 0.18
        dark = ndi.binary_opening(dark, iterations=2)
        lb, nl = ndi.label(dark)
        if nl > 0:
            areas = np.bincount(lb.ravel())
            areas[0] = 0
            best = lb == int(np.argmax(areas))
            ys, xs = np.where(best)
            return float(xs.mean() + x0), float(ys.mean() + y0)
        return float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)

    fl_pad = detect_socket(240, 420, 1060, 1190)
    fr_pad = detect_socket(950, 1130, 1060, 1190)
    bl_pad = detect_socket(300, 450, 345, 450)
    br_pad = detect_socket(920, 1070, 345, 450)

    def find_back_edge_y(x0, x1):
        ys_edge = []
        for x in range(x0, x1):
            col = ndi.gaussian_filter1d(lum[330:420, x], 1.2)
            ys_edge.append(330.0 + float(np.argmin(np.gradient(col))))
        return float(np.median(ys_edge)) if ys_edge else 374.0

    y_back_l = find_back_edge_y(430, 500)
    y_back_r = find_back_edge_y(870, 940)

    # Encontrar las 4 esquinas exteriores superiores exactas del plinto (FL, FR, BL, BR)
    # Tomar la anchura máxima del plinto oscuro justo dentro del bisel trasero (y ≈ yb_i + 4) y delantero (y ≈ yf_i)
    yb_i = int(round(0.5 * (y_back_l + y_back_r))) + 3
    yf_i = int(round(np.median(y_bevel)))
    row_top = lum[yb_i - 1 : yb_i + 5, :].mean(axis=0)
    xs_top = np.where((row_top < 0.56) & (np.arange(W) > 250) & (np.arange(W) < 1120))[0]
    row_bot = lum[yf_i - 2 : yf_i + 3, :].mean(axis=0)
    xs_bot_c = np.where((row_bot < 0.52) & (np.arange(W) > 120) & (np.arange(W) < 1260))[0]

    bl_outer = (float(xs_top[0]), float(yb_i)) if len(xs_top) else (328.0, 378.0)
    br_outer = (float(xs_top[-1]), float(yb_i)) if len(xs_top) else (1052.0, 378.0)
    fl_outer = (float(xs_bot_c[0]), float(yf_i)) if len(xs_bot_c) else (184.0, 1190.0)
    fr_outer = (float(xs_bot_c[-1]), float(yf_i)) if len(xs_bot_c) else (1198.0, 1190.0)

    # Localizar los 2 vértices interiores laterales del terreno (donde empieza la hierba/musgo por la izquierda y derecha)
    # En FRONT y BACK, el borde interior del hexágono del terreno en los extremos izquierdo/derecho cae alrededor de x=28..70 y x=1298..1340, y≈740
    in_l_row = np.where((panel_rgb[720:760, :140, 1] > 0.24) & (panel_rgb[720:760, :140, 1] > panel_rgb[720:760, :140, 2] * 1.15))
    in_r_row = np.where((panel_rgb[720:760, W - 140 :, 1] > 0.24) & (panel_rgb[720:760, W - 140 :, 1] > panel_rgb[720:760, W - 140 :, 2] * 1.15))
    ml_in = (float(in_l_row[1].min()), float(720 + np.median(in_l_row[0]))) if len(in_l_row[0]) else (46.0, 738.0)
    mr_in = (float(W - 140 + in_r_row[1].max()), float(720 + np.median(in_r_row[0]))) if len(in_r_row[0]) else (1322.0, 738.0)

    return {
        "main_mask": main_mask,
        "y_front_bot_med": float(np.median(y_bot_base)),
        "y_front_top_med": float(yf_i),
        "y_back_top_med": float(yb_i),
        "fl_pad": fl_pad,
        "fr_pad": fr_pad,
        "bl_pad": bl_pad,
        "br_pad": br_pad,
        "fl_outer": fl_outer,
        "fr_outer": fr_outer,
        "bl_outer": bl_outer,
        "br_outer": br_outer,
        "ml_in": ml_in,
        "mr_in": mr_in,
    }


def fit_camera_to_panel(panel_rgb, lm, init_az_deg=30.0):
    """Ajusta con precisión subpíxel la cámara perspectiva contra las 4 esquinas exteriores (R=64, Z=Z_BASE) y los bordes interiores."""
    is_front = abs((init_az_deg % 360) - 30.0) < 45.0
    c_angles = np.radians([270.0, 330.0, 150.0, 90.0] if is_front else [90.0, 150.0, 330.0, 270.0])
    s_angles = np.radians([210.0, 30.0] if is_front else [30.0, 210.0])

    outer_3d = np.column_stack([R * np.cos(c_angles), R * np.sin(c_angles), np.full(4, Z_BASE)])
    outer_2d = np.array([lm["fl_outer"], lm["fr_outer"], lm["bl_outer"], lm["br_outer"]], dtype=np.float64)
    in_2d = np.array([lm["ml_in"], lm["mr_in"]], dtype=np.float64)
    pad_2d_target = np.array([lm["fl_pad"], lm["fr_pad"], lm["bl_pad"], lm["br_pad"]], dtype=np.float64)

    bounds = [
        (init_az_deg - 3.5, init_az_deg + 3.5),
        (28.0, 48.0),
        (210.0, 460.0),
        (-5.0, 5.0),
        (-5.0, 5.0),
        (20.0, 36.0),
        (16.0, 38.0),
        (-1.8, 1.8),
        (46.0, 55.5),
    ]

    # También comprobar los 2 bordes laterales exteriores (x ≈ 0 y x ≈ 1368 a altura y_mid ≈ 740)
    def loss(x_vec):
        az, el, dist, tx, ty, tz, fov, roll, ter_r = x_vec
        params = np.array([az, el, dist, tx, ty, tz, fov, roll], dtype=np.float64)
        ou, ov, _ = m4.project_pts_vec(params, outer_3d[:, 0], outer_3d[:, 1], outer_3d[:, 2], PAN_W, PAN_H)
        err_outer = np.sum((ou - outer_2d[:, 0]) ** 2 + (ov - outer_2d[:, 1]) ** 2) * 4.0

        in_3d = np.column_stack([ter_r * np.cos(s_angles), ter_r * np.sin(s_angles), np.full(2, Z_BASE)])
        iu, iv, _ = m4.project_pts_vec(params, in_3d[:, 0], in_3d[:, 1], in_3d[:, 2], PAN_W, PAN_H)
        err_in = np.sum((iu - in_2d[:, 0]) ** 2 + (iv - in_2d[:, 1]) ** 2) * 1.5
        return float(err_outer + err_in)

    x0 = np.array([init_az_deg, 38.0, 290.0, 0.0, 0.0, 26.0, 25.0, 0.0, 50.5], dtype=np.float64)
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
    res2 = minimize(loss, res.x, method="Powell", bounds=bounds, options={"maxiter": 3000})
    bx = res2.x if res2.fun <= res.fun else res.x
    params, ter_r_fit = bx[:8], float(bx[8])
    ou, ov, _ = m4.project_pts_vec(params, outer_3d[:, 0], outer_3d[:, 1], outer_3d[:, 2], PAN_W, PAN_H)
    rmse_o = float(np.sqrt(np.mean((ou - outer_2d[:, 0]) ** 2 + (ov - outer_2d[:, 1]) ** 2)))
    print(f"  [fit ter_r={ter_r_fit:.2f}mm] outer_rmse={rmse_o:.2f}px")
    return params, rmse_o, ter_r_fit


def run_calib():
    os.makedirs(WORK_DIR, exist_ok=True)
    front, back = load_panels()
    lm_f = find_panel_landmarks(front)
    lm_b = find_panel_landmarks(back)

    cam_f, rmse_f, padr_f = fit_camera_to_panel(front, lm_f, init_az_deg=30.0)
    cam_b, rmse_b, padr_b = fit_camera_to_panel(back, lm_b, init_az_deg=210.0)

    np.save(os.path.join(WORK_DIR, "cam_front.npy"), cam_f)
    np.save(os.path.join(WORK_DIR, "cam_back.npy"), cam_b)
    print(f"[Calib] FRONT cam={np.round(cam_f, 2)} | pad_RMSE={rmse_f:.2f}px")
    print(f"[Calib] BACK  cam={np.round(cam_b, 2)} | pad_RMSE={rmse_b:.2f}px")

    # Visual overlay audit: dibujar bordes del hexágono Z=25, borde interior camino (APOTHEM - 3.5) y anillos de pads
    vis_panels = []
    for name, img_f, cam_p, lm, pr in [("FRONT 0°", front, cam_f, lm_f, padr_f), ("BACK 180°", back, cam_b, lm_b, padr_b)]:
        pil = Image.fromarray((img_f * 255).astype(np.uint8))
        dr = ImageDraw.Draw(pil)
        ang = np.radians([30, 90, 150, 210, 270, 330])
        hx = R * np.cos(ang)
        hy = R * np.sin(ang)
        hz = np.full(6, Z_BASE)
        u_o, v_o, _ = m4.project_pts_vec(cam_p, hx, hy, hz, PAN_W, PAN_H)
        # Borde interior terreno ajustado (r_ter = pr mm)
        u_i, v_i, _ = m4.project_pts_vec(cam_p, pr * np.cos(ang), pr * np.sin(ang), hz, PAN_W, PAN_H)
        for k in range(6):
            kn = (k + 1) % 6
            dr.line([(u_o[k], v_o[k]), (u_o[kn], v_o[kn])], fill=(0, 255, 255), width=3)
            dr.line([(u_i[k], v_i[k]), (u_i[kn], v_i[kn])], fill=(80, 255, 120), width=2)
        # Centros de cajeados detectados (amarillo) y proyectados (rojo)
        for pt in [lm["fl_pad"], lm["fr_pad"], lm["bl_pad"], lm["br_pad"]]:
            dr.ellipse([pt[0] - 8, pt[1] - 8, pt[0] + 8, pt[1] + 8], outline=(255, 220, 0), width=3)
        pad_ang = np.radians([270, 330, 150, 90] if "FRONT" in name else [90, 150, 330, 270])
        pu, pv, _ = m4.project_pts_vec(cam_p, pr * np.cos(pad_ang), pr * np.sin(pad_ang), np.full(4, Z_BASE), PAN_W, PAN_H)
        for uu, vv in zip(pu, pv):
            dr.ellipse([uu - 6, vv - 6, uu + 6, vv + 6], outline=(255, 60, 60), width=3)
        # Eje vertical central (X=0, Y=0, Z=25..72)
        zu, zv, _ = m4.project_pts_vec(cam_p, np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([Z_BASE, Z_BASE + 46.0]), PAN_W, PAN_H)
        dr.line([(zu[0], zv[0]), (zu[1], zv[1])], fill=(255, 0, 255), width=3)
        vis_panels.append(np.asarray(pil.resize((684, 768), Image.LANCZOS)))

    out_ov = os.path.join(WORK_DIR, "calib_overlay.png")
    Image.fromarray(np.hstack(vis_panels)).save(out_ov)
    print(f"[Calib] Saved {out_ov}")


def to_screen_xy(X, Y):
    """Rota -30° al sistema local alineado a las vistas de foto (Sx=derecha en Front, Sy=profundidad Front->Back)."""
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    return X * c + Y * s, -X * s + Y * c


def ridge_seg(X, Y, x0, y0, h0, x1, y1, h1, w0, w1, sharp=0.95, seed=0.0):
    return m4.faceted_ridge(X, Y, x0, y0, h0, x1, y1, h1, w0, w1, sharp=sharp, seed_zig=seed, rib_amp=0.14)


def build_dual_base_dem(N=1024):
    """Construye el macizo centrado 360° en coordenadas (Sx, Sy) alineado a FRONT 0° y BACK 180°."""
    lin = np.linspace(-R, R, N)
    X, Y = np.meshgrid(lin, lin)
    Sx, Sy = to_screen_xy(X, Y)
    dist_h = m4.hex_dist(X, Y)
    dist_v = m4.vert_dist(X, Y)

    # Colchón piramidal más esbelto y agudo para dar siluetas cinceladas (sin masas hinchadas)
    r_c = np.hypot(Sx * 0.96, (Sy - 1.0) * 0.98)
    Z_base_dome = 12.5 * np.power(np.maximum(0.0, 1.0 - r_c / 48.0), 1.55)

    # Espinazo alpino afilado y facetado mapeado directamente desde FRONT (0°) y BACK (180°) en (Sx, Sy):
    ridges = [
        # 1. Cúspide central dominante ligeramente detrás del centro (costillas estrechas y verticales)
        (-3.8,   3.2, 42.5,   3.8,   2.2, 35.5,  10.8,  9.8, 0.78, 1.1),
        (-3.8,   3.2, 42.5,  -1.8,  11.8, 35.0,  10.5,  9.5, 0.78, 2.3),
        (-3.8,   3.2, 41.5, -13.5,   3.8, 29.5,  10.5,  8.8, 0.80, 3.7),
        ( 3.8,   2.2, 35.5,  14.8,   0.5, 30.5,  10.0,  8.8, 0.80, 4.9),

        # 2. Brazo delantero izquierdo en FRONT (flanquea el cañón terracota con pared escarpada)
        (-13.5,  3.8, 28.5, -10.8, -16.5, 22.5,   9.0,  7.8, 0.78, 6.2),
        (-10.8, -16.5, 22.5, -15.5, -29.5, 13.0,   7.8,  6.5, 0.82, 7.4),
        (-15.5, -29.5, 13.0, -24.5, -37.0,  4.8,   6.5,  5.0, 0.92, 8.1),
        (-13.5,  3.8, 28.0, -25.8,  -3.5, 22.0,   9.2,  7.8, 0.82, 9.5),
        (-25.8, -3.5, 22.0, -39.0,  -8.0,  8.0,   7.8,  5.8, 0.92, 10.8),

        # 3. Pirámide delantera derecha en FRONT (arista afilada iluminada a la derecha del cañón terracota)
        (  3.8,  2.2, 31.5,  15.5, -15.5, 25.5,   9.6,  8.8, 0.78, 11.4),
        ( 15.5, -15.5, 25.5,  10.2, -31.0, 12.0,   8.5,  6.5, 0.84, 12.7),
        ( 15.5, -15.5, 24.5,  26.5, -24.0, 13.0,   8.5,  6.5, 0.86, 13.6),
        ( 26.5, -24.0, 13.0,  36.5, -14.5,  6.2,   6.5,  4.8, 0.95, 14.2),
        ( 14.8,  0.5, 30.0,  29.0,   0.0, 20.5,   9.2,  7.6, 0.82, 15.3),
        ( 29.0,  0.0, 20.5,  40.5,   4.0,  7.0,   7.6,  5.5, 0.94, 16.5),

        # 4. Espigones traseros norte visibles en primer plano de BACK 180° (costillas esbeltas con valles intermedios)
        ( -1.8, 11.8, 34.5,   9.8,  14.5, 30.0,   9.2,  8.2, 0.78, 17.8),
        (  9.8, 14.5, 30.0,  14.8,  25.5, 19.0,   8.2,  6.8, 0.82, 19.1),
        ( 14.8, 25.5, 19.0,  22.5,  35.5,  6.8,   6.8,  5.0, 0.92, 20.4),
        ( 14.8,  0.5, 26.5,  27.8,  14.5, 17.5,   8.5,  6.8, 0.84, 21.7),
        ( 27.8, 14.5, 17.5,  38.5,  19.0,  6.2,   6.8,  4.8, 0.94, 22.9),

        ( -1.8, 11.8, 33.5, -14.5,  16.5, 25.0,   9.2,  7.8, 0.78, 24.2),
        (-14.5, 16.5, 25.0, -24.0,  26.5, 14.0,   7.8,  6.4, 0.84, 25.6),
        (-24.0, 26.5, 14.0, -33.5,  31.5,  5.2,   6.4,  4.6, 0.94, 26.8),
        ( -1.8, 11.8, 31.0,  -4.8,  25.0, 20.5,   8.4,  6.8, 0.80, 27.9),
        ( -4.8, 25.0, 20.5, -11.0,  36.5,  8.0,   6.8,  5.0, 0.92, 29.1),
        (-25.8, -3.5, 20.0, -32.0,  12.5, 13.0,   7.8,  6.2, 0.86, 30.3),
        (-32.0, 12.5, 13.0, -40.5,  10.5,  5.2,   6.2,  4.6, 0.95, 31.5),
    ]

    Z_r = Z_base_dome.copy()
    for rg in ridges:
        Z_r = smax(Z_r, ridge_seg(Sx, Sy, *rg), k=1.05)

    # 4 Cuencas/collados glaciares suaves (< 36°) anidados entre los espigones de granito (según FRONT 0° y BACK 180°)
    saddles = [
        (-13.5, -5.5, 17.5, 8.5, 7.2),   # Nevero amplio izquierdo (FRONT)
        ( 12.5,  1.5, 17.0, 7.8, 7.0),   # Nevero medio-derecho (FRONT)
        ( 14.5, 14.5, 16.5, 8.2, 7.5),   # Nevero delantero izquierdo (BACK)
        (-13.5, 15.0, 16.5, 8.2, 7.5),   # Nevero delantero derecho (BACK)
    ]
    Z_sad = np.zeros_like(Z_r)
    for (cx_s, cy_s, h_s, rx_s, ry_s) in saddles:
        r2 = ((Sx - cx_s) / rx_s) ** 2 + ((Sy - cy_s) / ry_s) ** 2
        Z_sad = np.maximum(Z_sad, h_s * np.exp(-0.5 * r2))
    Z_r = smax(Z_r, Z_sad, k=2.8)

    # Excavación en S del cañón terracota delantero en Sy < -6 (abierto hacia el borde sur)
    t_g = np.clip((-8.0 - Sy) / 39.0, 0.0, 1.0)
    cx_g = 1.6 * (1.0 - t_g) - 2.6 * np.sin(t_g * math.pi * 1.15) + 1.8 * t_g
    dg = np.abs(Sx - cx_g)
    wg = 2.0 + 1.15 * t_g
    gully_cut = smoothstep(-6.0, -13.0, Sy) * smoothstep(-49.0, -38.0, Sy) * np.exp(-0.5 * (dg / wg) ** 2)
    Z_r = Z_r * (1.0 - 0.82 * gully_cut) + 0.65 * (0.82 * gully_cut)

    # Canal secundario trasero suavizado en Sy > +12 para acentuar el valle entre las costillas traseras
    bg_cut = smoothstep(12.0, 22.0, Sy) * smoothstep(46.0, 34.0, Sy) * np.exp(-0.5 * ((Sx + 1.5) / 4.4) ** 2)
    Z_r = Z_r * (1.0 - 0.38 * bg_cut)

    # Estriado de cresta diedral fino para dar fractura granítica equilibrada (50-63° sin agujas de pelo)
    fl1 = m4.ridge_fbm(Sx, Sy, [0.36, 0.88, 1.75], [0.46, 0.34, 0.20], seed=511)
    fl2 = m4.ridge_fbm(Sx * 0.85 + Sy * 0.4, -Sx * 0.4 + Sy * 0.85, [0.58, 1.35], [0.58, 0.42], seed=612)
    Z_r = Z_r + smoothstep(2.8, 15.0, Z_r) * (1.45 * (fl1 - 0.46) + 1.05 * (fl2 - 0.46))

    # Delantal compacto respetando ROAD_MARGIN=3.5mm y PAD_RADIUS=10.0mm
    taper = smoothstep(3.85, 8.2, dist_h) * smoothstep(10.35, 14.0, dist_v)
    Z = Z_BASE + np.clip(Z_r, 0.0, None) * taper
    return X, Y, Sx, Sy, Z, dist_h, dist_v, taper


def extract_sky_silhouette(panel_rgb):
    """Extrae la máscara exacta del objeto montaña+plinto por inundación desde la franja de fondo superior."""
    H, W, _ = panel_rgb.shape
    bg_ref = np.median(panel_rgb[30:110, 80 : W - 80], axis=(0, 1))
    diff_bg = np.max(np.abs(panel_rgb - bg_ref[None, None, :]), axis=2)
    sat = panel_rgb.max(-1) - panel_rgb.min(-1)
    # El fondo del estudio es gris claro uniforme (lum > 0.72, sat < 0.055, cercano a bg_ref)
    lum = 0.2126 * panel_rgb[..., 0] + 0.7152 * panel_rgb[..., 1] + 0.0722 * panel_rgb[..., 2]
    is_sky_cand = (diff_bg < 0.095) & (sat < 0.055) & (lum > 0.70)
    # Inundación conectada únicamente desde el borde superior (y=0..25) para preservar neveros blancos internos
    seed = np.zeros((H, W), dtype=bool)
    seed[:25, :] = True
    sky = ndi.binary_propagation(seed, mask=is_sky_cand)
    return ~sky


def optimize_dual_skyline(X, Y, Z, taper, cam_f, cam_b, fg_f, fg_b, iters=16):
    """Ajusta Z simultáneamente contra los horizontes superiores exactos de FRONT 0° y BACK 180°."""
    Sx, Sy = to_screen_xy(X, Y)
    W_s, H_s = PAN_W // 2, PAN_H // 2
    fg_f_s = fg_f[::2, ::2]
    fg_b_s = fg_b[::2, ::2]
    y_tar_f = np.array([np.where(fg_f_s[:, c])[0].min() if fg_f_s[:, c].any() else H_s for c in range(W_s)], dtype=np.float32)
    y_tar_b = np.array([np.where(fg_b_s[:, c])[0].min() if fg_b_s[:, c].any() else H_s for c in range(W_s)], dtype=np.float32)

    # Calcular línea del borde trasero superior del plinto en Z=Z_BASE para cada cámara (solo optimizamos cielo encima del bisel)
    def bezel_top_line(cam_p, idx_pair):
        ang = np.radians([270, 330, 30, 90, 150, 210])
        tu, tv, _ = m4.project_pts_vec(cam_p, R * np.cos(ang), R * np.sin(ang), np.full(6, Z_BASE), W_s, H_s)
        a, b = idx_pair
        yb = np.full(W_s, float(H_s))
        u0, u1 = min(tu[a], tu[b]), max(tu[a], tu[b])
        va, vb = (tv[a], tv[b]) if tu[a] <= tu[b] else (tv[b], tv[a])
        for c in range(W_s):
            if u0 + 4 <= c <= u1 - 4:
                yb[c] = va + (c - u0) / max(u1 - u0, 1.0) * (vb - va)
        return yb, int(u0 + 6), int(u1 - 6)

    yb_f, c0_f, c1_f = bezel_top_line(cam_f, (4, 3))
    yb_b, c0_b, c1_b = bezel_top_line(cam_b, (0, 1))

    for it in range(iters):
        dZ_tot = np.zeros_like(Z, dtype=np.float64)
        w_tot = np.zeros_like(Z, dtype=np.float64)
        maes = []
        for name, cam_p, y_tar, yb, c0, c1 in [
            ("F", cam_f, y_tar_f, yb_f, c0_f, c1_f),
            ("B", cam_b, y_tar_b, yb_b, c0_b, c1_b),
        ]:
            _, sil_r, _, hx, hy, _, _, _, _, _ = m4.ray_march_heightfield(Z, cam_p, out_w=W_s, out_h=H_s)
            y_r = np.array([np.where(sil_r[:, c])[0].min() if sil_r[:, c].any() else H_s for c in range(W_s)], dtype=np.float32)
            cols = (np.arange(W_s) >= c0) & (np.arange(W_s) <= c1) & ((y_tar < yb - 3.0) | (y_r < yb - 3.0))
            dy_px = y_r[cols] - y_tar[cols]
            mae = float(np.mean(np.abs(dy_px)))
            maes.append(f"{name}:{mae:.1f}px")
            sig_xy = 5.6 if it < 7 else 4.0
            for c_idx, err_px in zip(np.arange(W_s)[cols], dy_px):
                if abs(err_px) < 0.5:
                    continue
                row_hit = int(np.clip(y_r[c_idx] + 2, 0, H_s - 1))
                if not sil_r[row_hit, c_idx]:
                    continue
                xw, yw = float(hx[row_hit, c_idx]), float(hy[row_hit, c_idx])
                dz_mm = np.clip(err_px * 0.20, -2.8, 2.8)
                rad = np.exp(-0.5 * (((X - xw) / sig_xy) ** 2 + ((Y - yw) / sig_xy) ** 2))
                dZ_tot += dz_mm * rad
                w_tot += rad
        print(f"  [Dual Skyline {it:02d}] {' | '.join(maes)} | Zmax={Z.max():.2f}")
        delta = np.where(w_tot > 1e-4, dZ_tot / np.maximum(w_tot, 1e-4), 0.0)
        delta = ndi.gaussian_filter(delta, 4.0) * smoothstep(2.5, 10.0, Z - Z_BASE) * taper
        Z = np.clip(Z + 0.58 * delta, Z_BASE, Z_BASE + 45.5)
    return Z


def project_single_view(Z_2k, cam_p, img_rgb, fg_mask, X, Y, dist_h, dist_v):
    """Proyecta una vista perspectiva 1368x1536 sobre el campo Z_2k preservando 100% la nieve real y bloqueando oclusiones/plinto."""
    _, _, _, hx, hy, hz, cam_xyz, nx_g, ny_g, nz_g = m4.ray_march_heightfield(Z_2k, cam_p, out_w=PAN_W, out_h=PAN_H)
    u_all, v_all, _ = m4.project_pts_vec(cam_p, X, Y, Z_2k, float(PAN_W), float(PAN_H))
    cam_dist_1k = np.sqrt((hx - cam_xyz[0]) ** 2 + (hy - cam_xyz[1]) ** 2 + (hz - cam_xyz[2]) ** 2)
    dist_grid = np.sqrt((X - cam_xyz[0]) ** 2 + (Y - cam_xyz[1]) ** 2 + (Z_2k - cam_xyz[2]) ** 2)
    ui = np.clip(np.round(u_all), 0, PAN_W - 1).astype(np.int32)
    vi = np.clip(np.round(v_all), 0, PAN_H - 1).astype(np.int32)

    vdir = cam_xyz[None, None, :] - np.stack([X, Y, Z_2k], axis=-1)
    vdir /= np.linalg.norm(vdir, axis=-1, keepdims=True)
    ndotv = np.clip(nx_g * vdir[..., 0] + ny_g * vdir[..., 1] + nz_g * vdir[..., 2], 0.0, 1.0)

    in_frame = (u_all >= 2.0) & (u_all <= PAN_W - 3.0) & (v_all >= 2.0) & (v_all <= PAN_H - 3.0)
    # Rechazar saltos de profundidad en siluetas de oclusión (evita estelas de primer plano estiradas sobre laderas traseras)
    cd_safe = np.where(np.isfinite(cam_dist_1k), cam_dist_1k, 999.0)
    gdy, gdx = np.gradient(cd_safe)
    occ_jump = np.hypot(gdx, gdy) > 1.8
    occ_dil = ndi.binary_dilation(occ_jump, iterations=4)

    # Usar fg_mask (inundación exacta del estudio exterior) para proteger 100% la nieve blanca interior sin captar cielo!
    fg_er = ndi.binary_erosion(fg_mask, iterations=2)
    vis_raw = (
        in_frame
        & fg_er[vi, ui]
        & (dist_grid <= cd_safe[vi, ui] + 0.35)
        & (~occ_dil[vi, ui])
        & (dist_h >= 6.8)
        & (dist_v >= 11.8)
        & (ndotv > 0.26)
    )

    # Evitar capturar el borde gris/negro del plinto en el perímetro exterior (sin tocar jamás la nieve blanca!)
    lum_im = 0.2126 * img_rgb[..., 0] + 0.7152 * img_rgb[..., 1] + 0.0722 * img_rgb[..., 2]
    sat_im = np.max(img_rgb, axis=-1) - np.min(img_rgb, axis=-1)
    is_dark_bezel = (lum_im < 0.45) & (sat_im < 0.12)
    rel_z_m = np.maximum(0.0, Z_2k - Z_BASE)
    near_edge_img = (u_all < 140.0) | (u_all > PAN_W - 140.0) | (v_all > 1055.0)
    vis_raw &= ~(((dist_h < 19.5) | (dist_v < 20.5) | (rel_z_m < 7.0) | near_edge_img) & is_dark_bezel[vi, ui])
    vis_er = ndi.binary_erosion(vis_raw, iterations=2)
    vis_w = ndi.gaussian_filter((vis_er * np.power(np.maximum(ndotv - 0.14, 0.0), 1.7)).astype(np.float32), sigma=2.4)

    # Muestreo bicúbico suave sobre img_rgb
    proj = np.stack(
        [ndi.map_coordinates(img_rgb[..., c], [v_all, u_all], order=3, mode="nearest") for c in range(3)],
        axis=-1,
    )
    proj = np.clip(proj, 0.0, 1.0).astype(np.float32)

    # De-lighting suave por iluminación de estudio (~+X, -Y/+Z suave) para homogeneizar Sur/Norte
    dx = 2.0 * R / (Z_2k.shape[0] - 1)
    gy_s, gx_s = np.gradient(ndi.gaussian_filter(Z_2k, 2.5 / dx), dx, dx)
    nlen = np.sqrt(1.0 + gx_s * gx_s + gy_s * gy_s)
    ld = np.array([0.48, -0.25, 0.84], dtype=np.float32)
    ld /= np.linalg.norm(ld)
    ndotl = np.clip((-gx_s * ld[0] - gy_s * ld[1] + ld[2]) / nlen, 0.0, 1.0)
    illum = np.clip(0.76 + 0.30 * ndotl, 0.68, 1.06)
    proj_delit = np.clip(proj / (illum[..., None] ** 0.42), 0.0, 1.0).astype(np.float32)
    return proj_delit, vis_w


def extract_centrada_swatches(front_rgb, back_rgb):
    """Extrae muestras puras 100% libres de nieve de granito, terracota, musgo y nieve desde los paneles 1368x1536."""
    def prep_patch(src, x0, x1, y0, y1, sz=256, sharp=True):
        crop = src[y0:y1, x0:x1].astype(np.float32)
        pil = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8)).resize((sz, sz), Image.LANCZOS)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        low = ndi.gaussian_filter(arr, (44.0, 44.0, 0.0))
        arr = np.clip(arr - 0.45 * (low - np.mean(arr, axis=(0, 1), keepdims=True)), 0.0, 1.0)
        if not sharp:
            return ndi.gaussian_filter(arr, (1.8, 1.8, 0.0))
        return np.clip(arr + 0.32 * (arr - ndi.gaussian_filter(arr, (1.2, 1.2, 0.0))), 0.0, 1.0)

    def pick_granite(img, min_dist=90):
        sz = 84
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        sat = img.max(-1) - img.min(-1)
        gy, gx = np.gradient(lum)
        gr = np.hypot(gx, gy)
        cands = []
        for y0 in range(320, 1020 - sz, 20):
            for x0 in range(160, 1200 - sz, 20):
                sub = img[y0 : y0 + sz, x0 : x0 + sz]
                ls = lum[y0 : y0 + sz, x0 : x0 + sz]
                ss = sat[y0 : y0 + sz, x0 : x0 + sz]
                # Cero nieve y cero musgo verde dentro del recorte de granito
                if np.mean((ls > 0.66) & (ss < 0.14)) > 0.008:
                    continue
                if np.mean((sub[..., 1] > sub[..., 0] * 0.89) & (sub[..., 1] - sub[..., 2] > 0.07)) > 0.025:
                    continue
                if np.mean(ls) < 0.34 or np.mean(ls) > 0.56:
                    continue
                if np.mean(sub[..., 0] - sub[..., 2]) > 0.20:
                    continue
                if np.mean(ss) < 0.06:
                    continue
                cands.append((float(np.mean(gr[y0 : y0 + sz, x0 : x0 + sz])), x0, y0))
        cands.sort(key=lambda t: t[0], reverse=True)
        chosen = []
        out = []
        for _, x0, y0 in cands:
            if any(math.hypot(x0 - cx, y0 - cy) < min_dist for cx, cy in chosen):
                continue
            chosen.append((x0, y0))
            out.append(prep_patch(img, x0, x0 + sz, y0, y0 + sz))
            if len(out) >= 3:
                break
        return out

    g_list = pick_granite(front_rgb) + pick_granite(back_rgb)
    if not g_list:
        g_list = [prep_patch(front_rgb, 855, 985, 760, 910), prep_patch(back_rgb, 675, 775, 475, 610)]

    return {
        "granite": g_list,
        "scree": [
            prep_patch(front_rgb, 655, 745, 920, 1030),  # Cascajo terracota con cantos (FRONT)
            prep_patch(front_rgb, 625, 705, 760, 860),   # Desfiladero superior terracota (FRONT)
        ],
        "meadow": [
            prep_patch(front_rgb, 1085, 1225, 865, 985), # Pradera musgo verde + piedrecitas (FRONT)
            prep_patch(front_rgb, 155, 295, 855, 975),   # Pradera izquierda (FRONT)
            prep_patch(back_rgb, 185, 325, 875, 995),    # Pradera esquina norte-este (BACK)
        ],
        "snow": [
            prep_patch(front_rgb, 365, 475, 655, 735, sharp=False),
            prep_patch(back_rgb, 765, 875, 660, 740, sharp=False),
        ],
    }


def synthesize_exemplar_quilt(X, Y, Z_2k, dist_h, dist_v, swatches):
    """Sintetiza textura ejemplar continua alineada a la pendiente para rellenar recovecos 120°/300° sin ningún borrón."""
    H, W = Z_2k.shape
    dx = 2.0 * R / (W - 1)
    gy_s, gx_s = np.gradient(ndi.gaussian_filter(Z_2k, 2.4 / dx), dx, dx)
    gmag = np.hypot(gx_s, gy_s) + 1e-6
    dx_dn = -gx_s / gmag
    dy_dn = -gy_s / gmag

    gy, gx = np.gradient(ndi.gaussian_filter(Z_2k, 0.35 / dx), dx, dx)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    rel_z = np.maximum(0.0, Z_2k - Z_BASE)
    cav = Z_2k - ndi.gaussian_filter(Z_2k, max(2.0, 0.75 / dx))

    def cell_rand(ix, iy, salt):
        h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ np.int64(salt * 83492791)
        return (((h * 2654435761) >> 8) & 0xFFFF).astype(np.float32) / 65535.0

    def quilt_bank(bank, cell_mm=8.5, scale_mm=9.5, align_slope=True, seed=10):
        gx_c = np.floor(X / cell_mm).astype(np.int32)
        gy_c = np.floor(Y / cell_mm).astype(np.int32)
        nb = len(bank)
        sz = bank[0].shape[0]
        d1 = np.full((H, W), 1e9, dtype=np.float32)
        d2 = np.full((H, W), 1e9, dtype=np.float32)
        c1 = np.zeros((H, W, 3), dtype=np.float32)
        c2 = np.zeros((H, W, 3), dtype=np.float32)
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                cx_i, cy_i = gx_c + ox, gy_c + oy
                jx = (cx_i + 0.2 + 0.6 * cell_rand(cx_i, cy_i, seed + 1)) * cell_mm
                jy = (cy_i + 0.2 + 0.6 * cell_rand(cx_i, cy_i, seed + 2)) * cell_mm
                rx, ry = X - jx, Y - jy
                dist = np.hypot(rx, ry).astype(np.float32)
                if align_slope:
                    si_x = np.clip(np.round((jx + R) / dx).astype(np.int32), 0, W - 1)
                    si_y = np.clip(np.round((jy + R) / dx).astype(np.int32), 0, H - 1)
                    sx_d, sy_d = dx_dn[si_y, si_x], dy_dn[si_y, si_x]
                    ja = (cell_rand(cx_i, cy_i, seed + 5) - 0.5) * 0.50
                    ca, sa = np.cos(ja), np.sin(ja)
                    jdn_x, jdn_y = sx_d * ca - sy_d * sa, sx_d * sa + sy_d * ca
                    jrt_x, jrt_y = -jdn_y, jdn_x
                    ul = 0.5 + (rx * jrt_x + ry * jrt_y) / (scale_mm * 1.40)
                    vl = 0.5 + (rx * jdn_x + ry * jdn_y) / (scale_mm * 1.40)
                else:
                    ang = cell_rand(cx_i, cy_i, seed + 3) * (2.0 * math.pi)
                    ca, sa = np.cos(ang), np.sin(ang)
                    ul = 0.5 + (rx * ca + ry * sa) / (scale_mm * 1.40)
                    vl = 0.5 + (-rx * sa + ry * ca) / (scale_mm * 1.40)
                su = np.clip((np.clip(ul, 0.03, 0.97) * (sz - 1)).astype(np.int32), 0, sz - 1)
                sv = np.clip((np.clip(vl, 0.03, 0.97) * (sz - 1)).astype(np.int32), 0, sz - 1)
                idx_s = (np.floor(cell_rand(cx_i, cy_i, seed + 4) * nb).astype(np.int32)) % nb
                cs = np.zeros((H, W, 3), dtype=np.float32)
                for bi in range(nb):
                    m = idx_s == bi
                    if np.any(m):
                        cs[m] = bank[bi][sv[m], su[m]]
                b1 = dist < d1
                b2 = (~b1) & (dist < d2)
                if np.any(b1):
                    d2[b1], c2[b1] = d1[b1], c1[b1]
                    d1[b1], c1[b1] = dist[b1], cs[b1]
                if np.any(b2):
                    d2[b2], c2[b2] = dist[b2], cs[b2]
        t_s = np.clip(0.5 - 0.5 * ((d2 - d1) / 0.95), 0.0, 0.5)[..., None]
        return np.clip(c1 * (1.0 - t_s) + c2 * t_s, 0.0, 1.0)

    q_granite = quilt_bank(swatches["granite"], cell_mm=8.5, scale_mm=9.5, align_slope=True, seed=101)
    q_granite = np.clip(q_granite * np.clip(1.0 + 0.22 * np.clip(cav, -1.2, 1.0), 0.74, 1.15)[..., None], 0.06, 0.92)
    q_meadow = quilt_bank(swatches["meadow"], cell_mm=8.5, scale_mm=9.5, align_slope=False, seed=202)
    q_snow = quilt_bank(swatches["snow"], cell_mm=9.5, scale_mm=10.5, align_slope=False, seed=303)

    f_m = m4.ridge_fbm(X, Y, [0.12, 0.28], [0.6, 0.4], seed=505).astype(np.float32)
    in_ter = smoothstep(ROAD_MARGIN + 0.2, ROAD_MARGIN + 1.2, dist_h) * smoothstep(PAD_RADIUS + 0.2, PAD_RADIUS + 1.2, dist_v)
    rim_turf = (1.0 - smoothstep(4.2, 7.6, dist_h)) * (1.0 - smoothstep(0.8, 3.2, rel_z))
    meadow_m = np.maximum((1.0 - smoothstep(2.6, 6.8, rel_z + 1.6 * (f_m - 0.5))) * (1.0 - smoothstep(24.0, 33.0, slope_deg)), rim_turf) * in_ter
    snow_m = smoothstep(8.2, 14.5, rel_z + 2.2 * (f_m - 0.5)) * (1.0 - smoothstep(35.0, 44.5, slope_deg)) * smoothstep(0.5, -0.4, cav) * (1.0 - meadow_m) * in_ter
    tex = q_granite * (1.0 - meadow_m[..., None]) + q_meadow * meadow_m[..., None]
    tex = tex * (1.0 - snow_m[..., None]) + q_snow * snow_m[..., None]
    return tex, q_granite, q_meadow, q_snow


def infill_unseen_regions(alb, w_valid, Z_2k, X, Y, dist_h, dist_v, front_rgb, back_rgb):
    """Funde proyección dual directa con textura ejemplar ultra-nítida en zonas laterales/ocultas (0 borrones, 0 blanco/gris de borde)."""
    swatches = extract_centrada_swatches(front_rgb, back_rgb)
    exemplar, q_granite, q_meadow, q_snow = synthesize_exemplar_quilt(X, Y, Z_2k, dist_h, dist_v, swatches)
    in_clean_proj = smoothstep(8.2, 10.6, dist_h) * smoothstep(11.8, 13.8, dist_v)
    w_eff = np.clip((w_valid - 0.02) / 0.15, 0.0, 1.0) * in_clean_proj
    blended = np.clip(alb * w_eff[..., None] + exemplar * (1.0 - w_eff[..., None]), 0.0, 1.0)
    # Limpiar cualquier medialuna gris/oscura residual del bisel de la foto dentro de la pradera baja perimetral
    lum_b = 0.2126 * blended[..., 0] + 0.7152 * blended[..., 1] + 0.0722 * blended[..., 2]
    sat_b = blended.max(-1) - blended.min(-1)
    rel_b = np.maximum(0.0, Z_2k - Z_BASE)
    # Limpiar restos oscuros/azulados del plinto dentro del perímetro y evitar hierba verde en laderas medias > 33°
    bad_rim = ((dist_h < 19.0) | (dist_v < 20.5)) & (rel_b < 7.5) & (sat_b < 0.13) & ((lum_b < 0.34) | (lum_b > 0.65))
    bad_navy = (dist_h < 23.0) & (lum_b < 0.27) & (sat_b < 0.16) & (blended[..., 2] >= blended[..., 0] * 0.94)
    if np.any(bad_rim):
        bw = np.clip(ndi.gaussian_filter(bad_rim.astype(np.float32), 1.4) * 1.8, 0.0, 1.0)[..., None]
        blended = blended * (1.0 - bw) + q_meadow * bw
    if np.any(bad_navy):
        nw = np.clip(ndi.gaussian_filter(bad_navy.astype(np.float32), 1.2) * 1.8, 0.0, 1.0)[..., None]
        blended = blended * (1.0 - nw) + np.where(rel_b[..., None] > 4.5, q_granite, q_meadow) * nw
    dx = 2.0 * R / (Z_2k.shape[1] - 1)
    gy_b, gx_b = np.gradient(ndi.gaussian_filter(Z_2k, 0.45 / dx), dx, dx)
    sl_b = np.degrees(np.arctan(np.hypot(gx_b, gy_b)))
    # Cero musgo verde por encima de rel_b > 5.5 mm o en pendientes > 32° y cero mancha terracota en agujas altas (> 21mm)
    is_green_cliff = ((sl_b > 32.0) & (rel_b > 3.2) | (rel_b > 5.8)) & (blended[..., 1] > blended[..., 0] * 0.90) & (blended[..., 1] - blended[..., 2] > 0.036)
    if np.any(is_green_cliff):
        gw = np.clip(ndi.gaussian_filter(is_green_cliff.astype(np.float32), 1.0) * 1.6, 0.0, 1.0)[..., None]
        blended = blended * (1.0 - gw) + q_granite * gw
    high_terra = (rel_b > 20.5) & ((blended[..., 0] - blended[..., 2]) > 0.19) & (blended[..., 0] > 0.44)
    if np.any(high_terra):
        htw = np.clip(ndi.gaussian_filter(high_terra.astype(np.float32), 1.4) * 1.6, 0.0, 1.0)[..., None]
        blended = blended * (1.0 - htw) + q_granite * htw
    return blended, q_granite, q_snow


def sculpt_2d_3d_synchronized(Z_2k, alb_2k, q_granite, q_snow, X, Y, dist_h, dist_v):
    """
    Sincroniza AL 100% la textura y la geometría 3D en 4 capas reales:
      1) Conservación de neveros reales (< 62° inicial -> suavizados a < 38.5° en cuencas glaciares sin tocar cumbres!)
      2) Relieve granítico meso+micro (costillas cinceladas + grietas profundas en Z y campo normal 3D [-1, 1])
      3) Cantos terracota + matas botánicas + piedrecitas grises hacia +Z
      4) Garantía estricta 0.00000% nieve en escarpes >= 44° usando granito ejemplar q_granite
    """
    H, W = Z_2k.shape
    dx = 2.0 * R / (W - 1)
    core = smoothstep(ROAD_MARGIN + 0.55, ROAD_MARGIN + 2.2, dist_h) * smoothstep(PAD_RADIUS + 0.55, PAD_RADIUS + 2.2, dist_v)
    Sx_k, Sy_k = to_screen_xy(X, Y)
    # Respetar altura exacta calibrada por silueta dual (sin inflar artificialmente la aguja a 62.5mm)
    Z_work = Z_BASE + np.maximum(0.0, Z_2k - Z_BASE) * 0.96

    rel_z = np.maximum(0.0, Z_work - Z_BASE)
    gy_p, gx_p = np.gradient(ndi.gaussian_filter(Z_work, 0.40 / dx), dx, dx)
    slope_pre = np.degrees(np.arctan(np.hypot(gx_p, gy_p)))
    curv_crest = Z_work - ndi.gaussian_filter(Z_work, 2.2 / dx)
    r0, g0, b0 = alb_2k[..., 0], alb_2k[..., 1], alb_2k[..., 2]
    lum0 = 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0
    sat0 = np.maximum(np.maximum(r0, g0), b0) - np.minimum(np.minimum(r0, g0), b0)

    # Identificar neveros auténticos en unidades físicas (mm) desde cota alpina >= 4.2mm
    raw_snow_mask = (lum0 > 0.57) & (sat0 < 0.15) & (rel_z > 4.2) & (curv_crest < 2.0) & (slope_pre < 77.0)
    # Solo crestas muy agudas en cima se excluyen de nieve; los circos pasan a repisas < 36°
    steep_white_pre = (lum0 > 0.65) & (sat0 < 0.14) & ((slope_pre >= 78.0) | (curv_crest > 2.65))
    alb_work = alb_2k.copy()
    if np.any(steep_white_pre):
        sw_w = np.clip(ndi.gaussian_filter(steep_white_pre.astype(np.float32), 0.7) * 1.8, 0.0, 1.0)[..., None]
        alb_work = alb_work * (1.0 - sw_w) + q_granite * sw_w

    # Colchón glacial quirúrgico (~1.15 mm) SOLO en los neveros reales (mantiene el granito exterior 100% cincelado!)
    peak_shield = np.maximum(
        np.exp(-0.5 * (np.hypot(Sx_k + 3.8, Sy_k - 3.2) / 3.4) ** 2),
        smoothstep(1.15, 2.25, curv_crest) * 0.94,
    )
    snow_prox = ndi.gaussian_filter(raw_snow_mask.astype(np.float32), 1.15 / dx)
    snow_basin_mask = smoothstep(0.08, 0.36, snow_prox) * np.clip(1.0 - peak_shield, 0.0, 1.0) * core

    n_p = 256
    dx_p = 2.0 * R / (n_p - 1)
    Z_p0 = ndi.zoom(Z_work, n_p / float(W), order=1).astype(np.float64)
    M_p = np.clip(ndi.zoom(snow_basin_mask, n_p / float(W), order=1), 0.0, 1.0).astype(np.float64)
    if M_p.max() > 0.05:
        Z_p = Z_p0.copy()
        max_dz = math.tan(math.radians(30.0)) * dx_p
        max_dz_d = max_dz * math.sqrt(2.0)
        w_h = np.minimum(M_p[:, :-1], M_p[:, 1:])
        w_v = np.minimum(M_p[:-1, :], M_p[1:, :])
        w_d1 = np.minimum(M_p[:-1, :-1], M_p[1:, 1:])
        w_d2 = np.minimum(M_p[:-1, 1:], M_p[1:, :-1])
        pin_in = M_p > 0.10
        for it_s in range(650):
            dZ_acc = np.zeros_like(Z_p)
            dif_h = Z_p[:, 1:] - Z_p[:, :-1]
            ex_h = np.sign(dif_h) * np.maximum(0.0, np.abs(dif_h) - max_dz) * w_h * 0.075
            dZ_acc[:, :-1] += ex_h
            dZ_acc[:, 1:] -= ex_h

            dif_v = Z_p[1:, :] - Z_p[:-1, :]
            ex_v = np.sign(dif_v) * np.maximum(0.0, np.abs(dif_v) - max_dz) * w_v * 0.075
            dZ_acc[:-1, :] += ex_v
            dZ_acc[1:, :] -= ex_v

            dif_d1 = Z_p[1:, 1:] - Z_p[:-1, :-1]
            ex_d1 = np.sign(dif_d1) * np.maximum(0.0, np.abs(dif_d1) - max_dz_d) * w_d1 * 0.045
            dZ_acc[:-1, :-1] += ex_d1
            dZ_acc[1:, 1:] -= ex_d1

            dif_d2 = Z_p[1:, :-1] - Z_p[:-1, 1:]
            ex_d2 = np.sign(dif_d2) * np.maximum(0.0, np.abs(dif_d2) - max_dz_d) * w_d2 * 0.045
            dZ_acc[:-1, 1:] += ex_d2
            dZ_acc[1:, :-1] -= ex_d2

            Z_p = Z_p + dZ_acc
            if it_s % 10 == 9:
                Z_avg = 0.25 * (Z_p[:-2, 1:-1] + Z_p[2:, 1:-1] + Z_p[1:-1, :-2] + Z_p[1:-1, 2:])
                Z_p[1:-1, 1:-1] = np.where(pin_in[1:-1, 1:-1], Z_p[1:-1, 1:-1] * 0.80 + Z_avg * 0.20, Z_p[1:-1, 1:-1])
        dZ_snow = ndi.zoom(np.clip(Z_p - Z_p0, -11.5, 11.5).astype(np.float32), W / float(n_p), order=3)
        Z_work = Z_work + dZ_snow * snow_basin_mask

    rel_z = np.maximum(0.0, Z_work - Z_BASE)
    gy_s2, gx_s2 = np.gradient(ndi.gaussian_filter(Z_work, 0.45 / dx), dx, dx)
    curv_crest = Z_work - ndi.gaussian_filter(Z_work, 2.2 / dx)

    r, g, b = alb_work[..., 0], alb_work[..., 1], alb_work[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    is_snow = smoothstep(0.55, 0.70, lum) * (1.0 - smoothstep(0.11, 0.18, sat)) * smoothstep(4.2, 7.2, rel_z) * (1.0 - smoothstep(1.4, 2.6, curv_crest))
    snow_zone = np.clip(ndi.gaussian_filter(is_snow, 0.40 / dx), 0.0, 1.0)

    is_terracotta = smoothstep(0.14, 0.28, (r - b)) * smoothstep(0.32, 0.50, r) * (1.0 - is_snow)
    is_meadow = (
        smoothstep(0.015, 0.075, g - np.minimum(r, b) * 0.92)
        * smoothstep(0.04, 0.12, g - b)
        * (1.0 - is_snow)
        * (1.0 - is_terracotta)
        * (1.0 - smoothstep(8.0, 20.0, rel_z))
    )

    # Contraste normalizado local multi-escala (micro 0.065mm + meso 0.24mm / 0.95mm para esculpir costillas reales en flancos graníticos!)
    l_loc = np.maximum(ndi.gaussian_filter(lum, 2.4 / dx), 0.14)
    lum_eq = np.clip((lum / l_loc) * 0.55, 0.08, 1.45)
    l_fine = ndi.gaussian_filter(lum_eq, max(1.0, 0.065 / dx))
    l_m1 = ndi.gaussian_filter(lum_eq, max(2.2, 0.24 / dx))
    l_m2 = ndi.gaussian_filter(lum_eq, max(5.5, 0.95 / dx))
    l_broad = ndi.gaussian_filter(lum_eq, max(10.0, 1.85 / dx))

    # Esculpir contrafuertes graníticos cincelados equilibrando aristas agudas (48-65° sin exceso vertical >70°)
    is_rock = np.clip((1.0 - 0.96 * np.maximum(snow_zone, snow_basin_mask * 0.85)) * (1.0 - 0.85 * is_meadow) * (1.0 - 0.60 * is_terracotta), 0.0, 1.0)
    rock_hp = np.clip((l_fine - l_m1) * 2.1 + (l_m1 - l_m2) * 2.2 + (l_m2 - l_broad) * 1.4, -1.0, 1.0)
    rock_cracks = np.clip((l_m2 - l_m1) * 1.15 + (l_m1 - l_fine) * 0.80 - 0.015, 0.0, 0.78) * 2.15
    f_chisel = m4.ridge_fbm(X, Y, [0.34, 0.78, 1.75], [0.44, 0.36, 0.20], seed=733).astype(np.float32)
    f_ribs = m4.ridge_fbm(X * 0.85 + Y * 0.45, -X * 0.45 + Y * 0.85, [0.45, 1.15, 2.4], [0.48, 0.34, 0.18], seed=744).astype(np.float32)
    az_flute = np.arctan2(Y - 2.0, X + 2.0) * 7.5 + 0.14 * np.hypot(X + 2.0, Y - 2.0)
    column_facets = (np.abs(np.sin(az_flute * 3.0)) - 0.48) * 0.75 + (f_chisel - 0.46) * 1.40 + (f_ribs - 0.46) * 1.05
    chisel_dz = is_rock * smoothstep(2.2, 8.5, rel_z) * column_facets
    rock_dz = is_rock * (1.32 * rock_hp - 1.32 * rock_cracks) + chisel_dz

    # --- CAPA B: Cantos terracota del barranco (bordes de camino encajados + piedras redondeadas hacia arriba) ---
    cob_hp = np.clip((l_fine - l_m2) * 2.8 + np.maximum(0.0, l_m2 - l_broad) * 0.95, -0.25, 0.95)
    cobble_domes = ndi.gaussian_filter(np.maximum(0.0, cob_hp), max(1.2, 0.11 / dx))
    cobble_gaps = np.clip((l_m1 - l_fine) * 1.6, 0.0, 0.45)
    terra_dz = is_terracotta * (0.90 * cobble_domes - 0.42 * cobble_gaps)

    # --- CAPA C: Matas de musgo oscuro + piedrecitas grises en las praderas bajas (SIEMPRE hacia +Z!) ---
    l_raw_fine = ndi.gaussian_filter(lum, max(1.0, 0.075 / dx))
    l_raw_meso = ndi.gaussian_filter(lum, max(3.5, 0.65 / dx))
    shrub_clusters = np.clip((l_raw_meso - l_raw_fine) * 5.6, 0.0, 0.95) * smoothstep(0.03, 0.12, g - b)
    pebble_dots = np.clip((l_raw_fine - l_raw_meso) * 6.2, 0.0, 0.98) * (1.0 - smoothstep(0.10, 0.22, sat))
    meadow_dz = is_meadow * (0.62 * shrub_clusters + 0.72 * pebble_dots)

    # --- CAPA D: Compuesto final + relajación suave exclusivamente en los bordes de nevero >= 38° ---
    Z_out = Z_work + core * (rock_dz + terra_dz + meadow_dz)

    # Micro-púas extremas >= 68° relajadas suavemente (preservando aristas agudas de 45°-66°)
    g_cap = math.tan(math.radians(68.0))
    for _ in range(7):
        gy_e, gx_e = np.gradient(Z_out, dx, dx)
        ge = np.hypot(gx_e, gy_e)
        ex_e = np.clip((ge - g_cap) / (g_cap * 0.32), 0.0, 1.0) * core
        if ex_e.max() < 0.02:
            break
        Z_out = Z_out + (ndi.gaussian_filter(Z_out, 0.38 / dx) - Z_out) * np.clip(ndi.gaussian_filter(ex_e, 0.26 / dx) * 1.35, 0.0, 1.0) * 0.40

    for _ in range(35):
        gy_c, gx_c = np.gradient(ndi.gaussian_filter(Z_out, 0.35 / dx), dx, dx)
        sl_c = np.degrees(np.arctan(np.hypot(gx_c, gy_c)))
        viol = (snow_zone > 0.25) & (sl_c >= 39.0)
        if not np.any(viol):
            break
        vw = np.clip(ndi.gaussian_filter(viol.astype(np.float32), 0.85 / dx) * 2.4, 0.0, 1.0) * snow_zone
        Z_out = Z_out + (ndi.gaussian_filter(Z_out, 1.50 / dx) - Z_out) * vw * 0.60

    # Campo normalizado 3D [-1, +1] para sculpt_cliff_walls_3d (esculpido perpendicular (nx,ny,nz) en paredes >= 32°)
    cliff_field = np.clip(
        is_rock * (1.45 * rock_hp - 1.55 * rock_cracks + 0.65 * column_facets)
        + is_terracotta * (1.10 * cobble_domes - 0.52 * cobble_gaps)
        + is_meadow * (0.85 * shrub_clusters + 0.98 * pebble_dots),
        -1.0,
        1.0,
    ) * (1.0 - 0.96 * snow_zone) * core

    # Limpieza post-esculpido: 0% musgo en escarpes, alzar levemente sombras negras aplastadas de granito, y 0.00000% nieve >= 44°
    gy_f, gx_f = np.gradient(ndi.gaussian_filter(Z_out, 0.35 / dx), dx, dx)
    slope_f = np.degrees(np.arctan(np.hypot(gx_f, gy_f)))
    rel_out = np.maximum(0.0, Z_out - Z_BASE)
    alb_clean = alb_work.copy()
    q_shaded = np.clip(q_granite * np.clip(0.95 + 0.34 * rock_hp + 0.14 * (f_chisel - 0.45), 0.62, 1.24)[..., None], 0.08, 0.62)
    # Levantar suavemente sombras extremadamente oscuras en granito para mantener el tono arena-granito cálido en PBR WebGL
    lum_pre = 0.2126 * alb_clean[..., 0] + 0.7152 * alb_clean[..., 1] + 0.0722 * alb_clean[..., 2]
    dark_lift = np.clip((0.18 - lum_pre) / 0.18, 0.0, 1.0) * is_rock * (rel_out > 4.0) * 0.42
    alb_clean = np.clip(alb_clean + dark_lift[..., None] * q_granite, 0.0, 1.0)

    moss_high = ((rel_out > 5.6) | ((slope_f > 30.0) & (rel_out > 2.8))) & (alb_clean[..., 1] >= alb_clean[..., 0] * 0.86) & (alb_clean[..., 1] - alb_clean[..., 2] > 0.028)
    if np.any(moss_high):
        mw = np.clip(ndi.gaussian_filter(moss_high.astype(np.float32), 0.9) * 1.7, 0.0, 1.0)[..., None]
        alb_clean = alb_clean * (1.0 - mw) + q_shaded * mw
    lum_c = 0.2126 * alb_clean[..., 0] + 0.7152 * alb_clean[..., 1] + 0.0722 * alb_clean[..., 2]
    sat_c = alb_clean.max(-1) - alb_clean.min(-1)
    bad_snow = (lum_c > 0.64) & (sat_c < 0.13) & (slope_f >= 43.2)
    if np.any(bad_snow):
        bw = np.clip(ndi.gaussian_filter(bad_snow.astype(np.float32), 0.70) * 1.9, 0.0, 1.0)[..., None]
        alb_clean = alb_clean * (1.0 - bw) + q_shaded * bw
    # Realzar sutilmente nieve alpina auténtica ya asentada < 38.5° y sellar paridad 2D↔3D en cuencas glaciares suaves
    lum_c2 = 0.2126 * alb_clean[..., 0] + 0.7152 * alb_clean[..., 1] + 0.0722 * alb_clean[..., 2]
    sat_c2 = alb_clean.max(-1) - alb_clean.min(-1)
    clean_snow_pocket = (lum0 > 0.56) & (sat0 < 0.15) & (lum_c2 > 0.55) & (sat_c2 < 0.14) & (rel_out > 4.5) & (slope_f < 38.5)
    if np.any(clean_snow_pocket):
        alb_clean[clean_snow_pocket] = np.clip(alb_clean[clean_snow_pocket] * 1.12 + 0.06, 0.0, 0.96)
    # Donde la geometría fue esculpida como cojín glaciar suave (< 40.5°) usar nieve ejemplar texturizada (q_snow) garantizando paridad 2D↔3D
    curv_fin = Z_out - ndi.gaussian_filter(Z_out, 2.2 / dx)
    smooth_basin_fill = (snow_basin_mask > 0.36) & (slope_f < 40.5) & (rel_out > 5.0) & (curv_fin < 0.90) & (is_terracotta < 0.25) & (is_meadow < 0.25)
    if np.any(smooth_basin_fill):
        sbf_w = np.clip(ndi.gaussian_filter(smooth_basin_fill.astype(np.float32), 0.70) * 1.6, 0.0, 1.0)[..., None]
        snow_tex = np.clip(q_snow * (1.02 - 0.08 * (slope_f / 40.5)[..., None]), 0.68, 0.96)
        alb_clean = alb_clean * (1.0 - sbf_w) + snow_tex * sbf_w
    lum_c3 = 0.2126 * alb_clean[..., 0] + 0.7152 * alb_clean[..., 1] + 0.0722 * alb_clean[..., 2]
    sat_c3 = alb_clean.max(-1) - alb_clean.min(-1)
    rem_snow = (lum_c3 > 0.65) & (sat_c3 < 0.13) & (slope_f >= 43.4)
    if np.any(rem_snow):
        alb_clean[rem_snow] = np.clip(q_shaded[rem_snow], 0.08, 0.58)

    return Z_out.astype(np.float32), alb_clean.astype(np.float32), cliff_field.astype(np.float32), slope_f


def run_prep():
    os.makedirs(WORK_DIR, exist_ok=True)
    front, back = load_panels()
    cam_f = np.load(os.path.join(WORK_DIR, "cam_front.npy"))
    cam_b = np.load(os.path.join(WORK_DIR, "cam_back.npy"))

    print("[Prep 1/5] Extrayendo siluetas y optimizando relieve 3D dual (0° + 180°)...")
    fg_f = extract_sky_silhouette(front)
    fg_b = extract_sky_silhouette(back)
    sky_cache = os.path.join(WORK_DIR, "Z_skyline_1k.npy")
    if os.path.exists(sky_cache) and not getattr(run_prep, "recompute", False):
        Z_1k = np.load(sky_cache).astype(np.float64)
        print(f"  [Cache] Cargado {sky_cache} (Zmax={Z_1k.max():.2f}mm)")
    else:
        X, Y, Sx, Sy, Z_1k, dist_h, dist_v, taper = build_dual_base_dem(N=1024)
        Z_1k = optimize_dual_skyline(X, Y, Z_1k, taper, cam_f, cam_b, fg_f, fg_b, iters=16)
        np.save(sky_cache, Z_1k.astype(np.float32))

    print("[Prep 2/5] Reproyectando textura dual 2048x2048 desde FRONT (0°) y BACK (180°)...")
    N2 = 2048
    Z_2k = ndi.zoom(Z_1k, N2 / 1024.0, order=3)
    lin2 = np.linspace(-R, R, N2)
    X2, Y2 = np.meshgrid(lin2, lin2)
    dist_h2 = m4.hex_dist(X2, Y2)
    dist_v2 = m4.vert_dist(X2, Y2)

    proj_f, wf = project_single_view(Z_2k, cam_f, front, fg_f, X2, Y2, dist_h2, dist_v2)
    proj_b, wb = project_single_view(Z_2k, cam_b, back, fg_b, X2, Y2, dist_h2, dist_v2)
    w_tot = wf + wb + 1e-6
    alb_raw = (proj_f * (wf[..., None] ** 1.3) + proj_b * (wb[..., None] ** 1.3)) / (
        (wf[..., None] ** 1.3) + (wb[..., None] ** 1.3) + 1e-6
    )
    alb_2k, q_granite, q_snow = infill_unseen_regions(alb_raw, wf + wb, Z_2k, X2, Y2, dist_h2, dist_v2, front, back)

    print("[Prep 3/5] Esculpiendo sincronía 2D↔3D en 4 capas (roca, cantos, matas, nieve)...")
    Z_sync, alb_clean, cliff_field, slope_f = sculpt_2d_3d_synchronized(Z_2k, alb_2k, q_granite, q_snow, X2, Y2, dist_h2, dist_v2)

    # Bisel grafito Catan con cajeados circulares sutiles y aros limpios (100% compatible con Catan R=64, road=3.5mm, pads Ø20mm)
    rel_z = np.maximum(0.0, Z_sync - Z_BASE)
    on_border = np.maximum(
        1.0 - smoothstep(ROAD_MARGIN + 0.45, ROAD_MARGIN + 1.05, dist_h2),
        1.0 - smoothstep(PAD_RADIUS + 0.35, PAD_RADIUS + 0.95, dist_v2),
    )
    collar = np.clip(on_border * (1.0 - smoothstep(0.14, 0.58, rel_z)), 0.0, 1.0)
    f_bez = m4.ridge_fbm(X2, Y2, [0.4, 1.5], [0.5, 0.5], seed=888)
    slate = np.array([0.145, 0.152, 0.165], dtype=np.float32)[None, None, :] * (0.95 + 0.10 * f_bez[..., None])
    rim_hi = np.exp(-0.5 * ((dist_h2 - 0.85) / 0.28) ** 2) * 0.042
    curb_sh = np.exp(-0.5 * ((dist_h2 - 3.55) / 0.40) ** 2) * 0.052
    pad_groove = np.exp(-0.5 * ((dist_v2 - 8.2) / 0.32) ** 2) * 0.042
    pad_hi = np.exp(-0.5 * ((dist_v2 - 9.0) / 0.28) ** 2) * 0.028
    slate = np.clip(slate + (rim_hi - curb_sh - pad_groove + pad_hi)[..., None], 0.06, 0.27)
    alb_final = alb_clean * (1.0 - collar[..., None]) + slate * collar[..., None]
    alb_final[:6, :6] = np.array([0.150, 0.157, 0.170], dtype=np.float32)

    print("[Prep 4/5] Guardando DEM, campo normal 3D y Albedo 4K ultra-nítido...")
    z_span = float(Z_sync.max() - Z_BASE)
    hm_norm = np.clip((Z_sync - Z_BASE) / max(z_span, 1e-5) * 255.0, 0.0, 255.0).astype(np.float32)
    np.save(os.path.join(WORK_DIR, "montana_centrada_hm_2k.npy"), np.flipud(hm_norm))
    np.save(os.path.join(WORK_DIR, "montana_centrada_cliff_sculpt.npy"), np.flipud(cliff_field))
    np.save(os.path.join(WORK_DIR, "montana_centrada_Z.npy"), Z_sync.astype(np.float32))

    a_u8 = np.clip(np.flipud(alb_final) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(a_u8).save(os.path.join(WORK_DIR, "montana_centrada_albedo_2k.png"))

    # Upscale 4096x4096 con realce direccional multi-escala sin halos (textura máxima resolución!)
    alb_4k_u8 = np.asarray(Image.fromarray(a_u8).resize((4096, 4096), Image.LANCZOS), dtype=np.float32) / 255.0
    b_4k = ndi.gaussian_filter(alb_4k_u8, (1.35, 1.35, 0.0))
    alb_4k_sharp = np.clip(alb_4k_u8 + 0.42 * (alb_4k_u8 - b_4k), 0.0, 1.0)
    Image.fromarray((alb_4k_sharp * 255.0).astype(np.uint8)).save(os.path.join(WORK_DIR, "montana_centrada_albedo_4k.png"))

    lum_f = 0.2126 * alb_final[..., 0] + 0.7152 * alb_final[..., 1] + 0.0722 * alb_final[..., 2]
    sat_f = alb_final.max(-1) - alb_final.min(-1)
    inner = (dist_h2 > 4.5) & (dist_v2 > 11.0)
    p = np.percentile(slope_f[inner], [50, 90, 99, 99.9])
    snow_tot = float(np.mean((lum_f > 0.66) & (sat_f < 0.13) & inner) * 100.0)
    snow_steep = float(np.mean((lum_f > 0.66) & (sat_f < 0.13) & (slope_f >= 44.0)) * 100.0)
    print(f"[Prep Metrics] Z_max={Z_sync.max():.2f}mm (relief={z_span:.2f}mm) | slope p50/p90/p99/p99.9={p.round(1)} deg | snow={snow_tot:.2f}% | snow>=44deg={snow_steep:.5f}%")

    print("[Prep 5/5] Renderizando auditoría previa FRONT/BACK PBR + Arcilla + 6 vistas 360°...")
    render_prep_audits(Z_sync, alb_final, front, back, cam_f, cam_b)


def render_prep_audits(Z, alb, front, back, cam_f, cam_b):
    try:
        fnt = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
    except Exception:
        fnt = ImageFont.load_default()

    def make_panel(img_arr, label):
        pil = Image.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8)).resize((560, 628), Image.LANCZOS)
        dr = ImageDraw.Draw(pil)
        dr.text((14, 12), label, fill=(25, 32, 46), font=fnt)
        return np.asarray(pil)

    pbr_f = m4.ray_march_heightfield(Z, cam_f, out_w=560, out_h=628, tex_rgb=alb, yaw_offset_deg=0.0, sun_rel=True)[0]
    clay_f = m4.ray_march_heightfield(Z, cam_f, out_w=560, out_h=628, tex_rgb=None, yaw_offset_deg=0.0, sun_rel=True)[0]
    pbr_b = m4.ray_march_heightfield(Z, cam_b, out_w=560, out_h=628, tex_rgb=alb, yaw_offset_deg=0.0, sun_rel=True)[0]
    clay_b = m4.ray_march_heightfield(Z, cam_b, out_w=560, out_h=628, tex_rgb=None, yaw_offset_deg=0.0, sun_rel=True)[0]

    row1 = np.hstack([make_panel(front, "REF FRONT (0°)"), make_panel(pbr_f, "3D FRONT PBR"), make_panel(clay_f, "3D FRONT ARCILLA")])
    row2 = np.hstack([make_panel(back, "REF BACK (180°)"), make_panel(pbr_b, "3D BACK PBR"), make_panel(clay_b, "3D BACK ARCILLA")])
    Image.fromarray(np.vstack([row1, row2])).save(os.path.join(WORK_DIR, "centrada_dual_vs_3d.png"))

    v6 = []
    for ang in [0, -60, -120, -180, -240, -300]:
        ci = m4.ray_march_heightfield(Z, cam_f, out_w=520, out_h=584, tex_rgb=alb, yaw_offset_deg=ang, sun_rel=True)[0]
        v6.append(make_panel(ci, f"{(-ang) % 360} deg"))
    Image.fromarray(np.vstack([np.hstack(v6[:3]), np.hstack(v6[3:])])).save(os.path.join(WORK_DIR, "centrada_6views_preview.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["calib", "prep"], default="prep")
    p.add_argument("--recompute-skyline", action="store_true")
    args = p.parse_args()
    run_prep.recompute = args.recompute_skyline
    if args.stage == "calib":
        run_calib()
    else:
        run_prep()
