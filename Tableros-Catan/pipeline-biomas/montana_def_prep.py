"""
Montaña DEF — Reconstrucción 3D alineada a las dos referencias exactas V4:
  * Concepto 360° (3/4 Sur + 3/4 Norte): catan_mountain_hex_concept_sheet_1794591150816.jpg
  * Albedo cenital de referencia:        v4_nanobanana_diffuse_albedo_v2_1794591373002.jpg
"""
import argparse
import math
import os
import shutil
import sys

import numpy as np
import scipy.ndimage as ndi
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

REF_SRC = r"C:/Users/migue/.gemini/antigravity/brain/ed072550-bd3b-430e-a82e-ebe9a6377769"
FILES = {
    "concept": "catan_mountain_hex_concept_sheet_1794591150816.jpg",
    "albedo": "v4_nanobanana_diffuse_albedo_v2_1794591373002.jpg",
    "hm": "catan_mountain_hex_heightmap_clean_1794591229875.jpg",
}
DEF_DIR = os.path.join(HERE, "montana_def")
REFS = os.path.join(DEF_DIR, "refs")
WORK = os.path.join(DEF_DIR, "work")

N = 1024
RELIEF_MM = 46.5
S_PX_MM = 6.962
CX = 511.5
CY = 515.0


def ensure_refs():
    os.makedirs(REFS, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    for k, f in FILES.items():
        dst = os.path.join(REFS, f)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(REF_SRC, f), dst)
    return {k: os.path.join(REFS, f) for k, f in FILES.items()}


def sample_ref(img, X, Y, order=3):
    u = CX - S_PX_MM * X
    v = CY + S_PX_MM * Y
    if img.ndim == 2:
        return ndi.map_coordinates(img, [v, u], order=order, mode="nearest")
    return np.stack([ndi.map_coordinates(img[..., c], [v, u], order=order, mode="nearest") for c in range(img.shape[-1])], axis=-1)


def ridge_seg(X, Y, x0, y0, h0, x1, y1, h1, w0, w1, sharp=1.15):
    vx, vy = x1 - x0, y1 - y0
    l2 = vx * vx + vy * vy + 1e-8
    t = np.clip(((X - x0) * vx + (Y - y0) * vy) / l2, 0.0, 1.0)
    ts = t * t * (3.0 - 2.0 * t)
    px = x0 + t * vx
    py = y0 + t * vy
    d = np.hypot(X - px, Y - py)
    w = w0 + ts * (w1 - w0)
    h = h0 + ts * (h1 - h0)
    end_taper = smoothstep(0.0, 0.08, t) * smoothstep(1.0, 0.92, t)
    u = np.clip(d / w, 0.0, 1.0)
    prof = np.power(1.0 - np.power(u, sharp), 1.45)
    return h * prof * (0.85 + 0.15 * end_taper)


def extract_dark_slate_quilt(alb, lum, ochre_mask, moss_mask, dist_h, X, Y):
    """Textura ejemplar 100% nítida de granito/pizarra oscura estriada seleccionada automáticamente sin nieve."""
    H, W = X.shape
    gy_l, gx_l = np.gradient(lum)
    grad_e = np.hypot(gx_l, gy_l)
    sz = 112
    candidates = []
    for iy in range(64, H - 64 - sz, 28):
        for ix in range(64, W - 64 - sz, 28):
            if dist_h[iy + sz // 2, ix + sz // 2] > 39.0:
                continue
            l_sub = lum[iy : iy + sz, ix : ix + sz]
            if np.percentile(l_sub, 96) > 0.34 or np.percentile(l_sub, 5) < 0.04:
                continue
            if np.mean(ochre_mask[iy : iy + sz, ix : ix + sz]) > 0.025:
                continue
            if np.mean(moss_mask[iy : iy + sz, ix : ix + sz]) > 0.025:
                continue
            score = float(np.mean(grad_e[iy : iy + sz, ix : ix + sz]))
            candidates.append((score, iy, ix))
    candidates.sort(key=lambda t: t[0], reverse=True)
    swatches = []
    used_centers = []
    for _, iy, ix in candidates:
        if any(math.hypot(iy - uy, ix - ux) < 70 for uy, ux in used_centers):
            continue
        crop = alb[iy : iy + sz, ix : ix + sz].copy()
        used_centers.append((iy, ix))
        swatches.append(crop)
        if len(swatches) >= 6:
            break
    if not swatches:
        swatches = [alb[H // 2 - sz // 2 : H // 2 + sz // 2, W // 2 - sz // 2 : W // 2 + sz // 2].copy()]

    cell_mm = 7.5
    scale_mm = 9.5
    gx_c = np.floor(X / cell_mm).astype(np.int32)
    gy_c = np.floor(Y / cell_mm).astype(np.int32)
    nb = len(swatches)
    d1 = np.full((H, W), 1e9, dtype=np.float32)
    d2 = np.full((H, W), 1e9, dtype=np.float32)
    col1 = np.zeros((H, W, 3), dtype=np.float32)
    col2 = np.zeros((H, W, 3), dtype=np.float32)

    def cell_rand(ix, iy, salt):
        h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ np.int64(salt * 83492791)
        return (((h * 2654435761) >> 8) & 0xFFFF).astype(np.float32) / 65535.0

    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            cx_i, cy_i = gx_c + ox, gy_c + oy
            jx = (cx_i + 0.20 + 0.60 * cell_rand(cx_i, cy_i, 11)) * cell_mm
            jy = (cy_i + 0.20 + 0.60 * cell_rand(cx_i, cy_i, 22)) * cell_mm
            rx, ry = X - jx, Y - jy
            dist = np.hypot(rx, ry).astype(np.float32)
            ang = (cell_rand(cx_i, cy_i, 33) - 0.5) * 0.35
            ca, sa = np.cos(ang), np.sin(ang)
            uu = np.clip(0.5 + (rx * ca + ry * sa) / (scale_mm * 1.35), 0.02, 0.98)
            vv = np.clip(0.5 + (-rx * sa + ry * ca) / (scale_mm * 1.35), 0.02, 0.98)
            su = np.clip((uu * (sz - 1)).astype(np.int32), 0, sz - 1)
            sv = np.clip((vv * (sz - 1)).astype(np.int32), 0, sz - 1)
            idx_s = (np.floor(cell_rand(cx_i, cy_i, 44) * nb).astype(np.int32)) % nb
            c_samp = np.zeros((H, W, 3), dtype=np.float32)
            for bi in range(nb):
                m = idx_s == bi
                if np.any(m):
                    c_samp[m] = swatches[bi][sv[m], su[m]]
            b1 = dist < d1
            b2 = (~b1) & (dist < d2)
            if np.any(b1):
                d2[b1], col2[b1] = d1[b1], col1[b1]
                d1[b1], col1[b1] = dist[b1], c_samp[b1]
            if np.any(b2):
                d2[b2], col2[b2] = dist[b2], c_samp[b2]

    t_seam = np.clip(0.5 - 0.55 * ((d2 - d1) / 0.65), 0.0, 0.5)[..., None]
    quilt = col1 * (1.0 - t_seam) + col2 * t_seam
    q_blur = ndi.gaussian_filter(quilt, (0.9, 0.9, 0.0))
    return np.clip(quilt + 0.45 * (quilt - q_blur), 0.03, 0.42)


def build_def():
    refs = ensure_refs()
    lin = np.linspace(-R, R, N)
    X, Y = np.meshgrid(lin, lin)
    dx = 2.0 * R / (N - 1)
    dist_h = m4.hex_dist(X, Y)
    dist_v = m4.vert_dist(X, Y)

    alb_src = np.asarray(Image.open(refs["albedo"]).convert("RGB"), dtype=np.float32) / 255.0
    hm_src = np.asarray(Image.open(refs["hm"]).convert("L"), dtype=np.float32) / 255.0

    alb = np.clip(sample_ref(alb_src, X, Y, order=3), 0.0, 1.0)
    hm_raw = np.clip(sample_ref(hm_src, X, Y, order=3), 0.0, 1.0)

    # ---------- 1. Clasificación de materiales desde el albedo de referencia ----------
    r_c, g_c, b_c = alb[..., 0], alb[..., 1], alb[..., 2]
    lum = 0.2126 * r_c + 0.7152 * g_c + 0.0722 * b_c
    sat = np.maximum(np.maximum(r_c, g_c), b_c) - np.minimum(np.minimum(r_c, g_c), b_c)

    ochre_score = (r_c - b_c) * 1.35 + (g_c - b_c) * 0.45
    ochre_mask = smoothstep(0.16, 0.30, ochre_score) * smoothstep(0.22, 0.36, r_c)
    ochre_smooth = ndi.gaussian_filter(ochre_mask, 1.8 / dx)

    moss_score = (g_c - r_c * 0.92) + (g_c - b_c) * 0.65
    moss_mask = smoothstep(0.06, 0.16, moss_score) * (1.0 - ochre_mask) * smoothstep(3.5, 16.0, 52.0 - dist_h)
    moss_smooth = ndi.gaussian_filter(moss_mask, 2.2 / dx)

    # Máscara de neveros en collados altos y circo glaciar norte (liberando los flancos exteriores de las Agujas Este y Oeste para que se eleven como granito oscuro afilado según el Concept Sheet 360°)
    snow_raw = (lum > 0.54) & (sat < 0.17) & (dist_h > 7.5)
    outer_flank_rock = (
        ((X > 16.5) & (Y < -9.0) & (X - 0.65 * Y > 25.0))
        | ((X < -21.5) & (Y < -2.0))
    )
    snow_keep = snow_raw & (~outer_flank_rock)
    snow_keep = ndi.binary_opening(snow_keep, structure=np.ones((3, 3)), iterations=1)

    # ---------- 2. Base piramidal alpina ancha + Esqueleto de Agujas alineado al Concepto 360° ----------
    # Un zócalo piramidal ancho evita pendientes laterales de 75° y da cohesión al macizo
    r_cen = np.hypot((X + 2.5) / 1.05, (Y - 1.5) / 1.05)
    Z_cone = 27.5 * np.power(np.maximum(0.0, 1.0 - r_cen / 49.0), 1.35)

    ridges = [
        # A. Aguja Central Dominante (-3.5, +3.5, 43.5 mm) — claramente más alta que las laterales
        (-3.5,  3.5, 43.5,  -2.5, 12.5, 36.5,  17.5, 14.5, 1.15),
        (-3.5,  3.5, 43.5,  -4.0, -8.0, 36.0,  17.5, 14.5, 1.15),
        (-3.5,  3.5, 43.5, -13.0,  2.5, 34.0,  16.5, 14.0, 1.15),
        (-3.5,  3.5, 43.5,   7.5,  1.5, 34.0,  16.5, 14.0, 1.15),

        # B. Macizo Oeste / Suroeste (~37.5 mm, prominente a la izquierda en Vista Sur)
        (-13.0,  2.5, 33.5, -24.0, -2.0, 37.5,  14.0, 15.5, 1.12),
        (-24.0, -2.0, 37.5, -27.0, -15.5, 32.5, 15.5, 14.0, 1.12),
        (-27.0, -15.5, 32.5, -34.0, -25.0, 21.0, 14.0, 11.0, 1.15),
        (-34.0, -25.0, 21.0, -40.0, -32.0,  9.5, 11.0,  8.0, 1.18),
        (-24.0, -2.0, 35.0, -18.5, -15.5, 27.5, 13.5, 11.5, 1.12),
        (-18.5, -15.5, 27.5, -22.5, -26.5, 17.5, 11.5,  9.5, 1.15),
        # Arista Noroeste (flanco derecho en Vista Norte)
        (-24.0, -2.0, 35.0, -25.5, 18.5, 29.5,  14.5, 13.0, 1.12),
        (-25.5, 18.5, 29.5, -27.5, 31.5, 20.5,  13.0, 10.5, 1.15),
        (-27.5, 31.5, 20.5, -32.5, 38.5, 10.0,  10.5,  8.0, 1.18),

        # C. Macizo Este / Sureste (~38.0 mm, aguja derecha afilada y alta en Vista Sur)
        (  7.5,  1.5, 33.5,  19.5,  3.5, 36.5,  14.0, 15.5, 1.12),
        ( 19.5,  3.5, 36.5,  21.0, -12.5, 38.0, 15.5, 14.5, 1.15),
        ( 21.0, -12.5, 38.0,  24.0, -27.5, 25.5, 14.5, 11.5, 1.15),
        ( 24.0, -27.5, 25.5,  24.0, -39.0, 13.0, 11.5,  8.5, 1.18),
        ( 21.0, -12.5, 34.0,  34.5,  -4.5, 24.5, 13.5, 11.0, 1.15),
        ( 34.5,  -4.5, 24.5,  41.5, -15.5, 12.0, 11.0,  8.0, 1.18),
        ( 34.5,  -4.5, 24.5,  41.0,   8.0, 13.5, 11.0,  8.5, 1.18),
        # Arista Noreste (flanco izquierdo en Vista Norte)
        ( 19.5,  3.5, 34.0,  16.5, 20.5, 27.0,  14.0, 11.5, 1.12),
        ( 16.5, 20.5, 27.0,  24.0, 32.0, 19.0,  11.5, 10.0, 1.15),
        ( 24.0, 32.0, 19.0,  29.5, 39.0,  9.5,  10.0,  7.5, 1.18),

        # D. Contrafuerte Central Sur (proa de pizarra prominente entre los dos cañones ocres del Sur)
        ( -4.0, -8.0, 33.5,  -3.0, -21.0, 27.5, 12.5, 10.5, 1.15),
        ( -3.0, -21.0, 27.5,  -3.5, -34.5, 20.5, 10.5,  9.5, 1.15),
        ( -3.5, -34.5, 20.5,  -2.5, -46.0,  9.5,  9.5,  7.5, 1.18),

        # E. Contrafuerte Frontal Norte (proa baja delante del circo glaciar norte)
        ( -3.5, 35.0, 14.0,  -2.0, 44.5,  9.5,  10.5,  8.0, 1.15),
        (-13.0, 34.5, 15.0, -11.0, 44.0,  9.5,   9.5,  7.5, 1.15),
        (  8.5, 34.0, 14.5,   7.5, 43.5,  9.0,   9.5,  7.5, 1.15),
    ]

    Z_skel = Z_cone.copy()
    for r in ridges:
        Z_skel = smax(Z_skel, ridge_seg(X, Y, *r), k=2.2)

    # ---------- 3. Detalle medio de facetas de roca y excavación en V de cañones ocres ----------
    # Usar escalas de 0.9 mm a 3.2 mm para facetas de roca sin oscilaciones de 75°
    hm_sm = ndi.gaussian_filter(hm_raw, 0.85 / dx)
    hm_base = ndi.gaussian_filter(hm_sm, 3.2 / dx)
    hm_ribs = np.clip(hm_sm - hm_base, -0.18, 0.22)

    lum_sm = ndi.gaussian_filter(lum, 0.70 / dx)
    lum_loc = ndi.gaussian_filter(lum_sm, 2.4 / dx)
    rock_striations = np.clip((lum_sm - lum_loc) / np.maximum(lum_loc, 0.14), -0.35, 0.35)

    # Máscara suave de todas las cuencas de nieve (incluyendo el fondo interior del circo norte mediante fill_holes)
    snow_filled = ndi.binary_fill_holes(snow_keep)
    snow_dil = ndi.binary_dilation(snow_filled, structure=np.ones((5, 5)), iterations=2)
    w_snow = np.clip(ndi.gaussian_filter(snow_dil.astype(np.float32), 1.45 / dx) * 1.38, 0.0, 1.0)
    rock_zone = 1.0 - w_snow

    # Realce de las 3 agujas principales exclusivamente sobre roca oscura (rock_zone), sin invadir neveros
    spire_rock_boost = rock_zone * (
        3.2 * np.exp(-0.5 * (np.hypot(X + 3.5, Y - 4.5) / 7.5) ** 2)
        + 4.2 * np.exp(-0.5 * (np.hypot(X + 25.0, Y - 1.5) / 8.0) ** 2)
        + 4.4 * np.exp(-0.5 * (np.hypot(X - 23.0, Y + 10.5) / 8.0) ** 2)
    )
    Z_rock = Z_skel + spire_rock_boost + rock_zone * (3.4 * hm_ribs + 1.2 * rock_striations)

    # Excavación en V de los cañones de cascajo ocre/cobre
    canyon_depth = (4.5 * ochre_smooth + 1.5 * ochre_mask) * smoothstep(3.5, 18.0, Z_rock) * rock_zone
    Z_rock = Z_rock - canyon_depth

    # Suavizado de colinas bajas en las faldas de musgo verde oliva
    f_hill = m4.smooth_fbm(X, Y, [0.12, 0.28, 0.65], [0.5, 0.3, 0.2], seed=404)
    Z_rock = np.maximum(1.0 + 1.6 * f_hill, Z_rock * (1.0 - 0.35 * moss_smooth) + (3.2 + 2.0 * f_hill) * (0.35 * moss_smooth))

    # ---------- 4. Terrazas y cuencas glaciares continuas con Z_rock (< 36° sin escalón de borde!) ----------
    # En el circo glaciar norte (-6, +25.5), moldear primero un anfiteatro cóncavo inclinado hacia el norte
    nc_w = np.exp(-0.5 * (((X + 6.0) / 13.0) ** 2 + ((Y - 25.5) / 10.0) ** 2))
    nc_bowl = 18.5 + 0.36 * np.clip(33.0 - Y, 0.0, 24.0) + 0.020 * ((X + 6.0) ** 2)
    Z_base_for_snow = Z_rock * (1.0 - 0.70 * nc_w) + nc_bowl * (0.70 * nc_w)

    # Suavizar con sigma = 5.8 mm sobre la propia Z_base_for_snow para que coincida en altura con la roca vecina (cero muros en el borde!)
    Z_snow_bench = ndi.gaussian_filter(Z_base_for_snow, 5.8 / dx) * 0.84 + 2.6
    Z_combined = Z_base_for_snow * (1.0 - w_snow) + Z_snow_bench * w_snow

    # Delantal hexagonal respetando estrictamente ROAD_MARGIN=3.50 mm y PAD_RADIUS=10.00 mm
    taper = smoothstep(3.85, 8.6, dist_h) * smoothstep(10.35, 15.0, dist_v)
    flat_zero = (dist_h <= ROAD_MARGIN + 0.25) | (dist_v <= PAD_RADIUS + 0.25)
    Zrel = np.where(flat_zero, 0.0, np.clip(Z_combined, 0.0, None) * taper)

    # Suavizado global ligero y escalado a RELIEF_MM ANTES de pulir pendientes máximas
    Zrel = np.where(flat_zero, 0.0, ndi.gaussian_filter(Zrel, 0.35 / dx) * smoothstep(3.80, 5.2, dist_h) * smoothstep(10.30, 11.8, dist_v))
    Zrel = Zrel * (RELIEF_MM / max(float(Zrel.max()), 1e-5))

    # Relajación suave de cualquier cresta puntual que supere 55° y garantía estricta < 36° en todos los neveros (w_snow)
    gmax_all = np.where(w_snow > 0.10, math.tan(math.radians(33.5)), math.tan(math.radians(55.0)))
    apex_w = np.exp(-0.5 * (((X + 3.5) / 5.5) ** 2 + ((Y - 4.5) / 5.5) ** 2)) * rock_zone
    for _ in range(60):
        gy, gx = np.gradient(Zrel, dx, dx)
        g = np.hypot(gx, gy)
        ex = np.clip((g - gmax_all) / (gmax_all * 0.25), 0.0, 1.0)
        if ex.max() < 0.01:
            break
        ex_s = np.clip(ndi.gaussian_filter(ex, 0.75 / dx) * 1.4, 0.0, 1.0)
        Zrel = Zrel + (ndi.gaussian_filter(Zrel, 1.25 / dx) - Zrel) * ex_s * 0.62
        if float(Zrel.max()) < RELIEF_MM:
            Zrel = Zrel + (RELIEF_MM - float(Zrel.max())) * apex_w

    Zrel = np.where(flat_zero, 0.0, Zrel * taper)
    Z = Z_BASE + Zrel

    # Garantía exacta en el campo suavizado Z que mide el auditor (sigma = 0.35 / dx) sobre snow_keep
    for _ in range(25):
        gy_f, gx_f = np.gradient(ndi.gaussian_filter(Z, 0.35 / dx), dx, dx)
        sl_f = np.degrees(np.arctan(np.hypot(gx_f, gy_f)))
        viol = snow_keep & (sl_f >= 40.5)
        if not np.any(viol):
            break
        v_w = np.clip(ndi.gaussian_filter(viol.astype(np.float32), 1.0 / dx) * 2.5, 0.0, 1.0)
        Z = Z + (ndi.gaussian_filter(Z, 1.5 / dx) - Z) * v_w * 0.65

    # ---------- 5. Refinamiento de Albedo con pizarra/granito estriado 100% nítido en flancos >= 42.5° ----------
    blur_orig = ndi.gaussian_filter(alb, (1.1, 1.1, 0.0))
    alb_sharp = np.clip(alb + 0.42 * (alb - blur_orig), 0.0, 1.0)
    slate_quilt = extract_dark_slate_quilt(alb_sharp, lum, ochre_mask, moss_mask, dist_h, X, Y)

    gy_f, gx_f = np.gradient(ndi.gaussian_filter(Z, 0.35 / dx), dx, dx)
    slope_deg = np.degrees(np.arctan(np.hypot(gx_f, gy_f)))

    on_steep_or_flank = np.maximum(
        smoothstep(39.5, 42.8, slope_deg),
        ndi.gaussian_filter(outer_flank_rock.astype(np.float32), 0.85 / dx),
    )
    replace_w = np.clip(
        smoothstep(0.42, 0.60, lum) * (1.0 - smoothstep(0.10, 0.18, sat)) * on_steep_or_flank,
        0.0,
        1.0,
    )
    replace_w = ndi.gaussian_filter(replace_w, 1.2)
    cavity_mod = np.clip(1.0 + 0.85 * hm_ribs + 0.35 * rock_striations, 0.74, 1.26)[..., None]
    rock_replacement = np.clip(slate_quilt * cavity_mod, 0.03, 0.40)
    alb_clean = alb_sharp * (1.0 - replace_w[..., None]) + rock_replacement * replace_w[..., None]

    lum2 = 0.2126 * alb_clean[..., 0] + 0.7152 * alb_clean[..., 1] + 0.0722 * alb_clean[..., 2]
    sat2 = alb_clean.max(-1) - alb_clean.min(-1)
    steep_viol = (slope_deg >= 42.5) & (lum2 > 0.58) & (sat2 < 0.15)
    if np.any(steep_viol):
        w_hard = np.clip(ndi.gaussian_filter(steep_viol.astype(np.float32), 1.1) * 1.8, 0.0, 1.0)[..., None]
        alb_clean = alb_clean * (1.0 - w_hard) + rock_replacement * w_hard

    # ---------- 6. Bisel grafito Catan + aros de bases ----------
    rel_z = np.maximum(0.0, Z - Z_BASE)
    on_border = np.maximum(
        1.0 - smoothstep(ROAD_MARGIN + 0.45, ROAD_MARGIN + 1.05, dist_h),
        1.0 - smoothstep(PAD_RADIUS + 0.35, PAD_RADIUS + 0.95, dist_v),
    )
    collar = np.clip(on_border * (1.0 - smoothstep(0.14, 0.62, rel_z)), 0.0, 1.0)
    f_bez = m4.ridge_fbm(X, Y, [0.4, 1.5], [0.5, 0.5], seed=888)
    slate = np.array([0.155, 0.162, 0.174], dtype=np.float32)[None, None, :] * (0.95 + 0.10 * f_bez[..., None])
    rim_hi = np.exp(-0.5 * ((dist_h - 0.90) / 0.30) ** 2) * 0.045
    curb_sh = np.exp(-0.5 * ((dist_h - 3.60) / 0.42) ** 2) * 0.055
    pad_groove = np.exp(-0.5 * ((dist_v - 8.2) / 0.32) ** 2) * 0.040
    pad_hi = np.exp(-0.5 * ((dist_v - 9.0) / 0.28) ** 2) * 0.028
    slate = np.clip(slate + (rim_hi - curb_sh - pad_groove + pad_hi)[..., None], 0.06, 0.28)
    alb_final = alb_clean * (1.0 - collar[..., None]) + slate * collar[..., None]
    alb_final[:4, :4] = np.array([0.160, 0.166, 0.178], dtype=np.float32)

    # ---------- 7. Exportación 2K/4K ----------
    z_span = float(Z.max() - Z_BASE)
    Z_2k = ndi.zoom(Z, 2, order=3)
    hm_norm = np.clip((Z_2k - Z_BASE) / max(z_span, 1e-5) * 255.0, 0.0, 255.0).astype(np.float32)
    np.save(os.path.join(WORK, "montana_def_hm_2k.npy"), np.flipud(hm_norm))
    np.save(os.path.join(WORK, "montana_def_Z.npy"), Z_2k.astype(np.float32))

    a_u8 = np.clip(np.flipud(alb_final) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(a_u8).save(os.path.join(WORK, "montana_def_albedo_2k.png"))
    alb_4k = Image.fromarray(a_u8).resize((4096, 4096), Image.LANCZOS)
    a4 = np.asarray(alb_4k, dtype=np.float32) / 255.0
    a4_sh = np.clip(a4 + 0.35 * (a4 - ndi.gaussian_filter(a4, (1.2, 1.2, 0.0))), 0.0, 1.0)
    Image.fromarray((a4_sh * 255.0).astype(np.uint8)).save(os.path.join(WORK, "montana_def_albedo_4k.png"))

    lum_f = 0.2126 * alb_final[..., 0] + 0.7152 * alb_final[..., 1] + 0.0722 * alb_final[..., 2]
    sat_f = alb_final.max(-1) - alb_final.min(-1)
    inner = (dist_h > 4.5) & (dist_v > 11.0)
    p = np.percentile(slope_deg[inner], [50, 90, 99, 99.9])
    snow_tot = float(np.mean((lum_f > 0.68) & (sat_f < 0.12) & inner) * 100.0)
    snow_steep = float(np.mean((lum_f > 0.68) & (sat_f < 0.12) & (slope_deg >= 44.0)) * 100.0)
    print(f"[def-prep] Z_max={Z.max():.2f}mm (relief={z_span:.2f}mm) | slope p50/p90/p99/p99.9={p.round(1)} deg | snow={snow_tot:.2f}% | snow>=44deg={snow_steep:.5f}%")
    render_previews(Z, alb_final, refs)


def render_previews(Z, alb, refs):
    cam = np.load(os.path.join(m4.REFS_DIR, "hero_cam_params.npy")).astype(np.float64)
    cam_lo = cam.copy()
    cam_lo[1] = 35.0
    cam_lo[2] = 520.0
    cam_lo[6] = 15.5
    cam_lo[5] = 32.0
    try:
        fnt = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
    except Exception:
        fnt = ImageFont.load_default()

    lum_a = 0.2126 * alb[..., 0] + 0.7152 * alb[..., 1] + 0.0722 * alb[..., 2]
    sat_a = alb.max(-1) - alb.min(-1)
    sn_m = smoothstep(0.62, 0.80, lum_a) * (1.0 - smoothstep(0.10, 0.18, sat_a))
    hp = lum_a - ndi.gaussian_filter(lum_a, 3.0)
    Z_vis = Z + (1.0 - 0.9 * sn_m) * np.clip(hp * 2.8, -0.85, 0.85) * smoothstep(4.2, 6.5, m4.hex_dist(*np.meshgrid(np.linspace(-R, R, Z.shape[0]), np.linspace(-R, R, Z.shape[0]))))

    def rend(yaw, tex, size=520):
        img = m4.ray_march_heightfield(Z_vis, cam_lo, out_w=size, out_h=size, tex_rgb=tex, yaw_offset_deg=yaw, sun_rel=True)[0]
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    concept = Image.open(refs["concept"]).convert("RGB")
    south = concept.crop((28, 76, 680, 744)).resize((520, 532), Image.LANCZOS)
    north = concept.crop((698, 76, 1348, 744)).resize((520, 532), Image.LANCZOS)
    s_r = Image.fromarray(rend(0, alb)).resize((520, 532))
    n_r = Image.fromarray(rend(-180, alb)).resize((520, 532))
    s_c = Image.fromarray(rend(0, None)).resize((520, 532))
    n_c = Image.fromarray(rend(-180, None)).resize((520, 532))
    top = np.hstack([np.asarray(south), np.asarray(s_r), np.asarray(s_c)])
    bot = np.hstack([np.asarray(north), np.asarray(n_r), np.asarray(n_c)])
    sheet = Image.fromarray(np.vstack([top, bot]))
    d = ImageDraw.Draw(sheet)
    for (x, y, t) in [
        (10, 8, "CONCEPTO SUR"), (530, 8, "3D SUR PBR"), (1050, 8, "3D SUR ARCILLA"),
        (10, 540, "CONCEPTO NORTE"), (530, 540, "3D NORTE PBR"), (1050, 540, "3D NORTE ARCILLA"),
    ]:
        d.text((x, y), t, fill=(30, 40, 55), font=fnt)
    sheet.save(os.path.join(WORK, "def_concept_vs_3d.png"))

    v6 = []
    for ang in [0, -60, -120, -180, -240, -300]:
        pil = Image.fromarray(rend(ang, alb, 420))
        ImageDraw.Draw(pil).text((12, 10), f"{(-ang) % 360} deg", fill=(30, 40, 55), font=fnt)
        v6.append(np.asarray(pil))
    Image.fromarray(np.vstack([np.hstack(v6[:3]), np.hstack(v6[3:])])).save(os.path.join(WORK, "def_textured_6views.png"))
    print("[def-prep] Previews saved in", WORK)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="prep")
    ap.parse_args()
    build_def()
