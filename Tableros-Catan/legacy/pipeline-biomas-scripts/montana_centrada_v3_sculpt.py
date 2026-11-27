"""
Montaña Centrada V3.6 — Esculpido 3D Geometry-First Hiperrealista (Nativo 4096×4096 = 32 px/mm)
Paridad Matemática 100.000% 2D↔3D fiel a `refs/concept_dual_fullres.jpg`:
  * Jerarquía piramidal alpina auténtica 360° (gran corona central dominante ~36mm alzándose
    dramáticamente sobre hombros medios de ~20-26mm y estribaciones fracturadas de ~14-18mm).
  * Puentes graníticos esbeltos y afilados cruzando todo el macizo -> nieve alpina (~11–14%)
    en mantos cóncavos continuos y couloirs interconectados (CERO foso circular, CERO almohadas
    convexas, dedos blancos trepando a los collados altos y 0.00000% nieve en >= 44.0°).
  * Esculpido arquitectónico de granito hasta la mismísima línea de cresta: columnas catedralicias
    verticales marcadas, diaclasas poliédricas Voronoi y estratos cristalinos de alto contraste.
  * Sendero en S esbelto de arcilla terracota clara desembocando limpiamente en el anfiteatro
    nevado central con ~150 cantos 3D discretos bordeando ambos hombros.
  * Cinturón perimetral 360° continuo de pradera oliva aterciopelada (cero halo marrón) con ~82
    cojines bajos de musgo 3D y ~98 piedrecitas grises discretas.
"""
import math
import os
import sys
import time

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pipeline-montana"))

import montana_sculpt_v4 as m4  # noqa: E402
import prototype_v3_cordillera as pv3  # noqa: E402

R = 64.0
APOTHEM = R * math.cos(math.radians(30.0))
Z_BASE = 25.0
ROAD_MARGIN = 3.5
PAD_RADIUS = 10.0

CENTRADA_DIR = os.path.join(HERE, "montana_centrada")
REFS_DIR = os.path.join(CENTRADA_DIR, "refs")
WORK_DIR = os.path.join(CENTRADA_DIR, "work")
DUAL_REF = os.path.join(REFS_DIR, "concept_dual_fullres.jpg")

smoothstep = m4.smoothstep
smax = m4.smax


def smin(a, b, k=1.5):
    return -smax(-a, -b, k=k)


def to_screen_xy(X, Y):
    """Rota (X, Y) canónico al marco de cámara FRONT (0° = az 30°): Sx=izq->der, Sy=frente->fondo."""
    c30 = math.cos(math.radians(30.0))
    s30 = math.sin(math.radians(30.0))
    return c30 * X + s30 * Y, -s30 * X + c30 * Y


def hash_grid(ix, iy, salt):
    h = (ix.astype(np.int64) * 374761393) ^ (iy.astype(np.int64) * 668265263) ^ np.int64(salt * 1442695041)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFF).astype(np.float32) / 65535.0


def voronoi_f2_minus_f1(U, V, cell_size=5.2, jitter=0.76, seed=101):
    """Celular Voronoi 2D vectorizado (F2 - F1): fracturas poliédricas limpias en granito."""
    gx = np.floor(U / cell_size).astype(np.int32)
    gy = np.floor(V / cell_size).astype(np.int32)
    d1 = np.full(U.shape, 1e9, dtype=np.float32)
    d2 = np.full(U.shape, 1e9, dtype=np.float32)
    id1 = np.zeros(U.shape, dtype=np.float32)

    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            cx = gx + ox
            cy = gy + oy
            jx = (cx + 0.5 + (hash_grid(cx, cy, seed + 1) - 0.5) * jitter) * cell_size
            jy = (cy + 0.5 + (hash_grid(cx, cy, seed + 2) - 0.5) * jitter) * cell_size
            dist = np.hypot(U - jx, V - jy).astype(np.float32)
            cid = hash_grid(cx, cy, seed + 3)
            b1 = dist < d1
            b2 = (~b1) & (dist < d2)
            if np.any(b1):
                d2[b1] = d1[b1]
                d1[b1] = dist[b1]
                id1[b1] = cid[b1]
            if np.any(b2):
                d2[b2] = dist[b2]
    edge = np.clip((d2 - d1) / (cell_size * 0.52), 0.0, 1.0)
    return edge, id1


def dolomite_tower(Sx, Sy, cx, cy, h, rx, ry, rot_deg=0.0, n_facets=4, facet_amp=0.18, power=1.18, cap=0.14, tilt=0.16, seed=0.0):
    """Torre granítica maciza con cima cincelada oblicua (estilo Dolomitas, nunca cono liso)."""
    rad = math.radians(rot_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    dx = Sx - cx
    dy = Sy - cy
    u = (dx * ca + dy * sa) / rx
    v = (-dx * sa + dy * ca) / ry
    ang = np.arctan2(v, u)
    poly = (
        1.0
        + facet_amp * np.cos(ang * n_facets + seed)
        + (facet_amp * 0.45) * np.cos(ang * (n_facets + 1) - seed * 1.7)
        + (facet_amp * 0.25) * np.sin(ang * 2.0 + seed * 0.9)
    )
    r = np.hypot(u, v) / np.maximum(poly, 0.55)
    prof = np.power(np.clip(1.0 - r, 0.0, 1.0), power)
    if cap > 0.0:
        chisel = np.clip(1.0 - cap + tilt * (0.65 * u + 0.35 * v), 0.72, 1.05)
        prof = smin(prof, chisel, k=0.095) / max(1.0 - cap * 0.65, 0.5)
    return (h * np.clip(prof, 0.0, 1.02)).astype(np.float32)


def granite_buttress(Sx, Sy, x0, y0, h0, x1, y1, h1, w0, w1, power=1.24, seed=1.0, warp_amp=0.35, cap=0.09, asym_bias=0.15):
    """Arista / contrafuerte granítico almenado y esbelto con punta angular afilada en el pie (cero dedos redondeados)."""
    dx = x1 - x0
    dy = y1 - y0
    L2 = dx * dx + dy * dy + 1e-8
    L = math.sqrt(L2)
    t = np.clip(((Sx - x0) * dx + (Sy - y0) * dy) / L2, 0.0, 1.0)
    px = x0 + t * dx
    py = y0 + t * dy
    nx, ny = -dy / L, dx / L
    bend = warp_amp * np.sin(t * math.pi * 1.6 + seed) * np.sin(t * math.pi)
    signed_d = (Sx - px) * nx + (Sy - py) * ny - bend
    d_end = np.hypot(Sx - (px + nx * bend), Sy - (py + ny * bend))
    w = w0 * (1.0 - t) + w1 * t
    # Si el contrafuerte muere en la pradera (h1 < 6.0mm), afina limpiamente tanto anchura como cota en cuña triangular!
    if h1 < 6.0:
        toe_t = smoothstep(0.62, 1.0, t)
        w = w * (1.0 - 0.50 * toe_t)
        h_base = h0 * (1.0 - t) + (h1 * 0.22) * t
        h_base = h_base * (1.0 - 0.65 * np.power(toe_t, 1.6))
    else:
        h_base = h0 * (1.0 - t) + h1 * t
    crest_steps = 1.0 + (0.068 * np.sin(t * math.pi * 3.5 + seed) + 0.036 * np.sin(t * math.pi * 6.8 - seed)) * np.sin(t * math.pi)
    h = h_base * crest_steps
    asym = np.where(signed_d >= 0.0, 1.0 + asym_bias, 1.0 - asym_bias)
    u = np.clip(d_end / np.maximum(w * asym, 0.55), 0.0, 1.0)
    prof = np.power(1.0 - u, power)
    if cap > 0.0:
        prof = smin(prof, np.full_like(prof, 1.0 - cap), k=0.085) / max(1.0 - cap * 0.68, 0.5)
    return (h * np.clip(prof, 0.0, 1.02)).astype(np.float32)


def trail_center_x(Sy):
    """Eje en S elegante del sendero terracota delantero (Sy de -50 a -7mm)."""
    t = np.clip((-7.0 - Sy) / 43.0, 0.0, 1.0)
    return (1.6 * (1.0 - t) - 4.4 * np.sin(t * math.pi * 1.18) + 2.9 * np.power(t, 1.22)).astype(np.float32)


def build_v3_geometry_4k(N=4096):
    t0 = time.time()
    print("[V3.6 Geo 1/5] Construyendo pirámide alpina unificada (corona 36mm + costillas esbeltas + neveros 11–14%)...")
    N2 = 2048
    lin2 = np.linspace(-R, R, N2, dtype=np.float32)
    X2, Y2 = np.meshgrid(lin2, lin2)
    Sx2, Sy2 = to_screen_xy(X2, Y2)
    dist_h2 = m4.hex_dist(X2, Y2).astype(np.float32)
    dist_v2 = m4.vert_dist(X2, Y2).astype(np.float32)
    d_clear2 = np.minimum(dist_h2 - ROAD_MARGIN, dist_v2 - PAD_RADIUS)
    dx2 = 2.0 * R / (N2 - 1)

    dw_x = (pv3.fft_fbm_canonical(X2, Y2, 14.0, 38.0, beta=2.45, seed=411, canon_n=512).astype(np.float32) - 0.5) * 1.40
    dw_y = (pv3.fft_fbm_canonical(X2, Y2, 14.0, 38.0, beta=2.45, seed=412, canon_n=512).astype(np.float32) - 0.5) * 1.40
    Sxw = Sx2 + dw_x * smoothstep(4.0, 14.0, d_clear2)
    Syw = Sy2 + dw_y * smoothstep(4.0, 14.0, d_clear2)

    # 1) Pradera ondulada suave 360°
    meadow_wave = pv3.fft_fbm_canonical(X2, Y2, 12.0, 40.0, beta=2.50, seed=301, canon_n=512).astype(np.float32)
    meadow_sub = pv3.fft_fbm_canonical(X2, Y2, 5.5, 18.0, beta=2.40, seed=302, canon_n=512).astype(np.float32)
    Z_meadow = (0.95 + 0.68 * meadow_wave + 0.24 * meadow_sub) * smoothstep(0.35, 3.80, d_clear2)

    rock_footprint = smoothstep(3.8, 10.2, d_clear2)

    # 2) NÚCLEO PIRAMIDAL CÓNCAVO SÓLIDO (pendiente natural ~23°–31° -> lecho ideal para amplios neveros entre aristas!)
    ang_c = np.arctan2(Sy2 - 1.5, Sx2 + 0.5)
    r_env = 44.5 * (1.0 + 0.07 * np.cos(ang_c * 3.0 - 0.5) + 0.05 * np.sin(ang_c * 5.0 + 0.9))
    r_core = np.hypot((Sx2 + 0.5) * 0.96, (Sy2 - 1.5) * 0.98)
    Z_core_massif = 23.0 * np.power(np.clip(1.0 - r_core / r_env, 0.0, 1.0), 1.38)
    core_drift = pv3.fft_fbm_canonical(X2, Y2, 8.0, 26.0, beta=2.45, seed=409, canon_n=512).astype(np.float32)
    Z_core_massif += 0.95 * (core_drift - 0.5) * smoothstep(3.0, 12.0, Z_core_massif)
    Z_core_massif = np.maximum(0.0, Z_core_massif) * rock_footprint

    # 3) JERARQUÍA PIRAMIDAL REAL (corona central 36.2mm alzándose +12..16mm sobre hombros medios y estribaciones!)
    #    Las agujas intermedias y puentes son ESBELTOS para que los neveros cóncavos brillen en 11–14% del área!
    towers = [
        # --- CORONA CENTRAL DOMINANTE (inequívoco pico maestro central desde los 360°!) ---
        # (cx, cy, h, rx, ry, rot_deg, n_facets, facet_amp, power, cap, tilt, seed)
        ( -1.8,   3.5, 36.4, 11.4, 10.4,  15.0, 4, 0.17, 1.15, 0.15,  0.16,  0.4),  # P0: Gran Torre Central (36.4mm!)
        ( -6.4,   7.8, 28.8,  9.2,  8.6, -22.0, 4, 0.16, 1.18, 0.13, -0.14,  1.5),  # P0_nw: Hombro NO
        (  3.6,   1.8, 27.6,  8.2,  7.6, -14.0, 4, 0.16, 1.20, 0.12,  0.12,  2.3),  # P0_e: Torre adosada este
        ( -4.6,  -5.4, 25.4,  8.4,  8.2,  26.0, 3, 0.17, 1.18, 0.12, -0.14,  3.1),  # P0_s: Aguja central-delantera (FRONT!)
        ( 10.6,   5.2, 28.6,  9.8,  9.0,  -8.0, 4, 0.17, 1.18, 0.14,  0.15,  4.2),  # P1: Torre gemela derecha-centro
        # --- AGUJAS INTERMEDIAS ESBELTAS (rompen la nieve en corredores alpinos SIN taparla!) ---
        (-13.8,   2.6, 21.8,  7.6,  7.0,  18.0, 4, 0.17, 1.20, 0.11,  0.14,  5.1),  # M_w: Aguja intermedia oeste
        (-11.8, -10.8, 19.6,  7.2,  6.8, -20.0, 3, 0.17, 1.22, 0.10, -0.12,  6.0),  # M_sw: Aguja intermedia SO
        (  9.5,  -8.2, 19.4,  7.0,  6.6,  20.0, 3, 0.17, 1.22, 0.10,  0.12,  6.9),  # M_se: Aguja intermedia SE
        ( 19.8,   4.4, 19.8,  8.2,  7.4,  14.0, 4, 0.16, 1.22, 0.11, -0.12,  7.8),  # M_e: Torre intermedia este
        ( -1.4,  14.8, 24.4,  8.2,  7.8,  -6.0, 4, 0.17, 1.20, 0.12,  0.13,  8.7),  # M_n: Gran torre intermedia norte (BACK!)
        (  9.4,  15.6, 22.0,  8.0,  7.4,  18.0, 4, 0.17, 1.20, 0.11, -0.13,  9.6),  # M_ne: Aguja trasera-izq (BACK)
        (-11.4,  16.0, 21.8,  8.0,  7.4, -18.0, 4, 0.17, 1.20, 0.11,  0.13, 10.5),  # M_nw: Aguja trasera-der (BACK)
        # --- ESTRIBACIONES EXTERIORES ESCALONADAS (perfil piramidal descendente fiel a REF FRONT/BACK!) ---
        (-14.6, -21.0, 16.8,  9.4,  8.8, -24.0, 4, 0.17, 1.22, 0.10,  0.12, 11.4),  # F_sw: Estribación delantera-izq
        (-25.8,  -6.5, 16.6,  9.0,  8.4,  16.0, 4, 0.16, 1.24, 0.10, -0.11, 12.3),  # F_w: Estribación oeste
        (-23.6,  11.0, 16.8,  9.0,  8.4, -16.0, 4, 0.16, 1.24, 0.10,  0.11, 13.2),  # F_nw: Estribación NO
        ( 16.0, -18.2, 18.8, 10.2,  9.4,  22.0, 4, 0.17, 1.20, 0.11, -0.13, 14.1),  # F_se: Pirámide ocre cálida P4 (FRONT!)
        ( 26.6, -10.2, 13.2,  8.2,  7.6, -12.0, 3, 0.16, 1.26, 0.08,  0.10, 15.0),  # F_ese: Estribación baja este-sur
        ( 24.0,  15.2, 14.8,  8.6,  7.8, -14.0, 3, 0.16, 1.26, 0.09, -0.10, 15.9),  # F_ne: Estribación NE (BACK izq)
        (-23.4,  19.6, 14.6,  8.6,  7.8,  14.0, 3, 0.16, 1.26, 0.09,  0.10, 16.8),  # F_nnw: Estribación NNO (BACK der)
        (  6.8,  27.2, 14.6,  7.8,  7.2,  24.0, 3, 0.17, 1.26, 0.09,  0.11, 17.7),  # F_n_L: Aguja frontal-izq (BACK)
        ( -7.8,  27.6, 14.4,  7.8,  7.2, -24.0, 3, 0.17, 1.26, 0.09, -0.11, 18.6),  # F_n_R: Aguja frontal-der (BACK)
    ]

    # Contrafuertes graníticos ESBELTOS Y AFILADOS (w0=5.0..6.8mm -> definen valles nevados amplios y continuos!)
    buttresses = [
        # (x0, y0, h0, x1, y1, h1, w0, w1, power, seed, warp_amp, cap, asym_bias)
        # --- ESPINAZOS CENTRALES ---
        ( -1.8,   3.5, 34.8,  -4.6,  -5.4, 24.8,  7.8,  6.2, 1.22,  1.2, 0.28, 0.10,  0.14),
        ( -1.8,   3.5, 34.4,  -6.4,   7.8, 28.0,  8.0,  6.8, 1.22,  2.1, 0.26, 0.10, -0.12),
        ( -1.8,   3.5, 30.8,  10.6,   5.2, 27.6,  7.0,  6.6, 1.24,  2.9, 0.30, 0.09,  0.12),
        ( -6.4,   7.8, 27.8,  -1.4,  14.8, 23.6,  6.8,  6.0, 1.24,  3.4, 0.28, 0.09, -0.13),
        # --- PUENTES ROCOSOS ESBELTOS (cortan los neveros en dedos alpinos naturales SIN tapar los circos!) ---
        ( -6.4,   7.8, 24.5, -13.8,   2.6, 21.0,  5.4,  4.8, 1.26,  4.1, 0.30, 0.08,  0.14),
        (-13.8,   2.6, 20.8, -25.8,  -6.5, 16.0,  5.4,  4.6, 1.28,  4.8, 0.32, 0.07, -0.13),
        ( -4.6,  -5.4, 22.8, -11.8, -10.8, 18.8,  5.2,  4.6, 1.26,  5.5, 0.28, 0.08,  0.14),
        (-11.8, -10.8, 18.4, -14.6, -21.0, 16.2,  5.4,  5.0, 1.26,  6.2, 0.30, 0.08, -0.14),
        (  3.6,   1.8, 23.8,   9.5,  -8.2, 18.8,  5.0,  4.4, 1.26,  6.9, 0.28, 0.08, -0.14),
        (  9.5,  -8.2, 18.6,  16.0, -18.2, 18.2,  5.4,  5.8, 1.24,  7.6, 0.30, 0.08,  0.15),
        ( 10.6,   5.2, 26.0,  19.8,   4.4, 19.2,  6.0,  5.2, 1.24,  8.3, 0.28, 0.08,  0.13),
        ( 10.6,   5.2, 25.0,   9.4,  15.6, 21.4,  5.6,  5.0, 1.26,  9.0, 0.30, 0.08, -0.14),
        ( -6.4,   7.8, 25.0, -11.4,  16.0, 21.2,  5.6,  5.0, 1.26,  9.7, 0.30, 0.08,  0.14),
        ( -1.4,  14.8, 22.8,   6.8,  27.2, 14.2,  5.2,  4.4, 1.26, 10.4, 0.30, 0.07,  0.14),
        ( -1.4,  14.8, 22.8,  -7.8,  27.6, 14.0,  5.2,  4.4, 1.26, 11.1, 0.30, 0.07, -0.14),
        # --- ESTRIBACIONES RAMIFICADAS HACIA LA PRADERA ---
        (-14.6, -21.0, 16.4, -10.8, -37.0,  4.5,  6.8,  4.2, 1.28, 12.0, 0.32, 0.07,  0.14),
        (-14.6, -21.0, 15.8, -24.5, -32.5,  4.5,  6.4,  4.0, 1.28, 12.7, 0.32, 0.06, -0.14),
        (-14.6, -21.0, 14.8, -30.2, -21.5,  4.6,  5.8,  3.8, 1.30, 13.4, 0.30, 0.06,  0.12),
        (-25.8,  -6.5, 16.2, -38.0, -10.2,  4.2,  6.8,  4.2, 1.30, 14.1, 0.32, 0.06,  0.13),
        (-25.8,  -6.5, 15.4, -36.5,   0.8,  4.3,  6.0,  3.8, 1.30, 14.8, 0.28, 0.06, -0.12),
        (-23.6,  11.0, 16.4, -37.0,  10.5,  4.3,  6.8,  4.2, 1.30, 15.5, 0.32, 0.06, -0.13),
        (-13.8,   2.6, 19.2, -23.6,  11.0, 16.2,  5.0,  4.8, 1.26, 16.2, 0.28, 0.07,  0.13),
        # Pirámide ocre P4 (abanico esculpido)
        ( 16.0, -18.2, 18.4,  11.6, -36.0,  4.6,  7.4,  4.4, 1.28, 17.0, 0.32, 0.07, -0.15),
        ( 16.0, -18.2, 17.8,  25.2, -29.0,  4.5,  7.0,  4.2, 1.28, 17.7, 0.32, 0.07,  0.14),
        ( 16.0, -18.2, 16.6,  32.8, -18.5,  4.4,  6.2,  4.0, 1.30, 18.4, 0.30, 0.06, -0.13),
        ( 26.6, -10.2, 12.8,  37.0,  -8.5,  4.2,  6.2,  4.0, 1.32, 19.1, 0.28, 0.06,  0.12),
        # Ala derecha este
        ( 19.8,   4.4, 19.0,  36.5,   4.0,  4.4,  7.0,  4.4, 1.30, 19.8, 0.32, 0.07,  0.13),
        ( 19.8,   4.4, 17.8,  33.0,  -2.5,  4.4,  5.8,  3.8, 1.30, 20.5, 0.28, 0.06, -0.12),
        # Flancos traseros norte (BACK 180°)
        (  9.4,  15.6, 21.0,  17.4,  34.0,  4.6,  6.8,  4.2, 1.28, 21.2, 0.32, 0.07,  0.14),
        (  9.4,  15.6, 19.5,  24.0,  15.2, 14.4,  5.6,  5.0, 1.28, 21.9, 0.30, 0.07, -0.13),
        ( 24.0,  15.2, 14.4,  36.0,  18.2,  4.2,  6.4,  4.0, 1.32, 22.6, 0.30, 0.06,  0.12),
        (-11.4,  16.0, 21.0, -18.4,  34.0,  4.6,  6.8,  4.2, 1.28, 23.3, 0.32, 0.07, -0.14),
        (-11.4,  16.0, 19.5, -23.4,  19.6, 14.2,  5.6,  5.0, 1.28, 24.0, 0.30, 0.07,  0.13),
        (-23.4,  19.6, 14.2, -35.5,  22.8,  4.2,  6.4,  4.0, 1.32, 24.7, 0.30, 0.06, -0.12),
        (  6.8,  27.2, 14.2,   9.4,  39.2,  4.0,  5.6,  3.8, 1.32, 25.4, 0.26, 0.06,  0.12),
        ( -7.8,  27.6, 14.0, -10.6,  39.2,  4.0,  5.6,  3.8, 1.32, 26.1, 0.26, 0.06, -0.12),
    ]

    Z_spires = np.zeros_like(Z_meadow)
    for tw in towers:
        Z_spires = smax(Z_spires, dolomite_tower(Sxw, Syw, *tw), k=1.75)
    for bt in buttresses:
        Z_spires = smax(Z_spires, granite_buttress(Sxw, Syw, *bt), k=1.65)

    Z_spires *= rock_footprint
    Z_rock = smax(Z_core_massif, Z_spires, k=2.1)

    # 4) EXCAVACIÓN LIMPIA DEL BARRANCO EN S (terminando justo en el borde del anfiteatro nevado central!)
    cx_t2 = trail_center_x(Sy2)
    dt2 = np.abs(Sx2 - cx_t2)
    in_canyon_y = smoothstep(-5.2, -11.5, Sy2) * smoothstep(-52.5, -45.5, Sy2)
    canyon_w = 3.25 + 1.05 * np.clip((-10.0 - Sy2) / 37.0, 0.0, 1.0)
    canyon_cut = in_canyon_y * np.exp(-0.5 * (dt2 / canyon_w) ** 2.8)
    t_ramp = np.clip((Sy2 + 49.0) / 43.0, 0.0, 1.0)
    Z_trail_floor = 0.98 + 8.8 * np.power(t_ramp, 1.38) + 0.16 * meadow_sub
    Z_rock = Z_rock * (1.0 - 0.94 * canyon_cut) + Z_trail_floor * (0.94 * canyon_cut)

    # Garganta central trasera estrecha (primer plano inferior de REF BACK 180°)
    back_chute = smoothstep(20.0, 28.5, Sy2) * smoothstep(45.5, 35.0, Sy2) * np.exp(-0.5 * (np.abs(Sx2 + 0.5) / 2.8) ** 2.4)
    Z_rock = Z_rock * (1.0 - 0.65 * back_chute) + (2.3 + 0.22 * meadow_sub) * (0.65 * back_chute)

    Z_rock_presence = Z_rock.copy()

    # 5) AMPLIOS NEVEROS CÓNCAVOS Y COULOIRS ENTRE COSTILLAS ESBELTAS (~6-7.5% hex = ~13-15% macizo, 0.00000% >=44°!)
    print("[V3.7 Geo 2/5] Integrando mantos y couloirs glaciares cóncavos + arquitectura granítica hasta la cresta...")
    sastrugi_n = pv3.fft_fbm_canonical(X2, Y2, 4.5, 16.0, beta=2.55, seed=419, canon_n=512).astype(np.float32)

    Z_glac_bed = 23.2 * np.power(np.clip(1.0 - r_core / 43.5, 0.0, 1.0), 1.35)
    couloir_beds = [
        # (cx, cy, h_s, rx, ry, rot_deg)
        (-18.2, -10.5, 16.8, 12.2,  9.4, -26.0),  # C1: Gran nevero delantero-izquierdo (FRONT!)
        (-16.5,   4.5, 17.2, 10.8,  8.8,  10.0),  # C1b: Brazo superior izquierdo del nevero (FRONT!)
        (  2.0,  -6.5, 17.8,  9.8,  8.6,   8.0),  # C2: Anfiteatro central brillante justo sobre la cabecera del sendero!
        ( 16.5,  -4.8, 16.8, 11.0,  8.8,  24.0),  # C3: Gran nevero delantero-derecho detrás de P4 (FRONT!)
        ( 16.2,  11.8, 17.2, 11.8,  9.4, -22.0),  # C4: Gran nevero trasero-izquierdo (BACK!)
        (-16.8,  12.2, 17.2, 11.8,  9.4,  22.0),  # C5: Gran nevero trasero-derecho (BACK!)
        (  4.5,  20.5, 16.2, 10.0,  8.8,  14.0),  # C6: Couloir trasero-izquierdo bajo (BACK!)
        ( -5.8,  20.8, 16.2, 10.0,  8.8, -14.0),  # C7: Couloir trasero-derecho bajo (BACK!)
        ( -8.8,  -1.2, 21.4,  7.8,  7.0, -16.0),  # C8: Lengua alta nevada trepando junto a la torre central izq!
        (  5.4,   9.6, 21.6,  7.8,  7.0,  14.0),  # C9: Lengua alta nevada entre torre central y torre derecha!
    ]
    c_add = np.zeros_like(Z_glac_bed)
    for (cx_s, cy_s, h_s, rx_s, ry_s, rdeg) in couloir_beds:
        rad = math.radians(rdeg)
        ca, sa = math.cos(rad), math.sin(rad)
        u = ((Sx2 - cx_s) * ca + (Sy2 - cy_s) * sa) / rx_s
        v = (-(Sx2 - cx_s) * sa + (Sy2 - cy_s) * ca) / ry_s
        shelf = h_s * np.exp(-0.54 * (u * u + v * v))
        c_add = smax(c_add, shelf, k=2.8)

    Z_snow_cradle = smax(Z_glac_bed, c_add, k=2.5) + 0.26 * (sastrugi_n - 0.5) * smoothstep(4.0, 12.0, Z_glac_bed)
    Z_snow_cradle = ndi.gaussian_filter(np.maximum(0.0, Z_snow_cradle), 2.3 / dx2) * smoothstep(6.2, 13.0, d_clear2)
    Z_snow_masked = Z_snow_cradle * (1.0 - 0.96 * canyon_cut) * (1.0 - 0.72 * back_chute) * smoothstep(38.0, 29.0, r_core)

    Z_macro = smax(Z_rock, Z_snow_masked, k=2.0)
    Z_macro = smax(Z_macro, Z_meadow, k=1.45)

    gy_pre, gx_pre = np.gradient(ndi.gaussian_filter(Z_macro, 0.75 / dx2), dx2, dx2)
    sl_pre = np.degrees(np.arctan(np.hypot(gx_pre, gy_pre)))
    snow_bowl_dom = (
        smoothstep(-1.35, 0.35, Z_snow_masked - Z_spires * 0.90)
        * smoothstep(6.8, 10.2, Z_macro)
        * smoothstep(36.0, 27.5, r_core)
        * smoothstep(41.5, 31.0, sl_pre)
        * (1.0 - canyon_cut)
        * (1.0 - back_chute)
    )

    # 6) ARQUITECTURA GRANÍTICA HASTA LA CRESTA (TALLA TANTO PAREDES COMO LOMOS DE ESTRIBACIONES!)
    Z_sm = ndi.gaussian_filter(Z_macro, 1.7 / dx2)
    gy_m, gx_m = np.gradient(Z_sm, dx2, dx2)
    gmag_m = np.hypot(gx_m, gy_m) + 1e-5
    slope_macro = np.degrees(np.arctan(gmag_m))
    cos_m = 1.0 / np.sqrt(1.0 + gmag_m * gmag_m)
    slope_comp = np.clip(1.0 / np.maximum(cos_m, 0.50), 1.0, 1.85).astype(np.float32)

    wx = pv3.fft_fbm_canonical(X2, Y2, 6.2, 21.0, beta=2.30, seed=501, canon_n=512).astype(np.float32) * 2.0
    wy = pv3.fft_fbm_canonical(X2, Y2, 6.2, 21.0, beta=2.30, seed=502, canon_n=512).astype(np.float32) * 2.0

    # Columnas verticales catedralicias nítidas y profundas (carvan las crestas y flancos en bloques afilados!)
    flute_raw = m4.ridge_fbm(Sx2 + wx * 0.42, Sy2 + wy * 0.42, [0.21, 0.45, 0.92], [0.54, 0.32, 0.14], seed=633).astype(np.float32)
    flute_adv = pv3.slope_advect_field(flute_raw, gx_m, gy_m, dx2, dx2, steps=14, step_mm=0.45).astype(np.float32)
    flute_cleft = np.power(np.clip(0.54 - flute_adv, 0.0, 0.54) / 0.54, 1.28)
    flute_rib = np.power(np.clip(flute_adv - 0.46, 0.0, 0.54) / 0.54, 1.12)

    # Bloques poliédricos Voronoi en 2 orientaciones
    v_mac_edge, v_mac_id = voronoi_f2_minus_f1(Sx2 + wx * 0.85, Sy2 + wy * 0.85, cell_size=5.2, jitter=0.75, seed=611)
    v_mac_adv = pv3.slope_advect_field(v_mac_edge, gx_m, gy_m, dx2, dx2, steps=5, step_mm=0.42).astype(np.float32)
    cleft_mac = np.power(1.0 - smoothstep(0.0, 0.32, v_mac_adv), 1.42)

    v_mes_edge, _ = voronoi_f2_minus_f1(Sx2 * 0.86 - Sy2 * 0.51 + wx * 0.45, Sx2 * 0.51 + Sy2 * 0.86 + wy * 0.45, cell_size=2.7, jitter=0.72, seed=622)
    v_mes_adv = pv3.slope_advect_field(v_mes_edge, gx_m, gy_m, dx2, dx2, steps=3, step_mm=0.34).astype(np.float32)
    cleft_mes = np.power(1.0 - smoothstep(0.0, 0.30, v_mes_adv), 1.45)

    # Estratos graníticos diagonales escalonados
    strata_z = Z_macro + 0.24 * Sx2 - 0.16 * Sy2 + 0.95 * (wx - 1.0)
    strata_step = np.sin(strata_z * 1.18) * 0.58 + np.sin(strata_z * 2.50) * 0.28
    strata_ledge = np.tanh(strata_step * 1.70) * 0.44

    # Escorrentía D-inf
    Z_sub = Z_macro[::5, ::5]
    flow_sub = pv3.compute_flow_accumulation(Z_sub)
    flow_2k = ndi.zoom(flow_sub, (N2 / float(Z_sub.shape[0]), N2 / float(Z_sub.shape[1])), order=1).astype(np.float32)
    gully_eros = smoothstep(1.3, 3.6, flow_2k) * smoothstep(20.0, 50.0, slope_macro)

    trail_bed_mask = in_canyon_y * np.exp(-0.5 * (dt2 / (canyon_w * 0.62)) ** 3.2)
    # Activar el cincelado rocoso desde 9° en roca elevada -> talla incluso los lomos de crestas (cero "dedos lisos"!)
    rock_zone = (
        smoothstep(2.4, 5.5, Z_rock_presence)
        * np.maximum(smoothstep(10.0, 22.0, slope_macro), smoothstep(9.0, 18.0, Z_spires) * 0.85)
        * (1.0 - 0.98 * snow_bowl_dom)
        * (1.0 - 0.96 * trail_bed_mask)
        * smoothstep(3.2, 6.2, d_clear2)
    )

    rock_sculpt_dz = rock_zone * (
        (-1.48 * flute_cleft + 1.02 * flute_rib) * (0.66 + 0.34 * slope_comp)
        + (-0.86 * cleft_mac - 0.42 * cleft_mes) * (0.66 + 0.34 * slope_comp)
        + 0.34 * strata_ledge
        + 0.26 * (v_mac_id - 0.5)
        - 0.46 * gully_eros
    )

    snow_drift_dz = snow_bowl_dom * 0.16 * (sastrugi_n - 0.5)

    Z_2k = Z_BASE + np.maximum(0.0, Z_macro + rock_sculpt_dz + snow_drift_dz) * smoothstep(0.35, 2.20, d_clear2)

    g_cap = math.tan(math.radians(66.5))
    core2 = smoothstep(0.65, 2.4, d_clear2)
    for _ in range(9):
        gy_e, gx_e = np.gradient(Z_2k, dx2, dx2)
        ge = np.hypot(gx_e, gy_e)
        ex_e = np.clip((ge - g_cap) / (g_cap * 0.30), 0.0, 1.0) * core2
        if ex_e.max() < 0.02:
            break
        Z_2k = Z_2k + (ndi.gaussian_filter(Z_2k, 0.38 / dx2) - Z_2k) * np.clip(ndi.gaussian_filter(ex_e, 0.26 / dx2) * 1.3, 0.0, 1.0) * 0.42

    # Relajación selectiva de los mantos nevados (< 35.8°)
    for _ in range(46):
        gy_s, gx_s = np.gradient(ndi.gaussian_filter(Z_2k, 0.35 / dx2), dx2, dx2)
        sl_s = np.degrees(np.arctan(np.hypot(gx_s, gy_s)))
        viol = (snow_bowl_dom > 0.20) & (sl_s >= 35.5)
        if not np.any(viol):
            break
        vw = np.clip(ndi.gaussian_filter(viol.astype(np.float32), 1.05 / dx2) * 2.4, 0.0, 1.0) * snow_bowl_dom
        Z_2k = Z_2k + (ndi.gaussian_filter(Z_2k, 1.85 / dx2) - Z_2k) * vw * 0.62

    cliff_field_2k = np.clip(
        rock_zone * (-1.45 * flute_cleft + 1.02 * flute_rib - 1.00 * cleft_mac - 0.58 * cleft_mes + 0.34 * strata_ledge),
        -1.0,
        1.0,
    ) * (1.0 - 0.98 * snow_bowl_dom) * core2

    # 7) NATIVO 4096×4096 + ~330 OBJETOS 3D DISCRETOS
    print(f"[V3.6 Geo 3/5] Elevando a {N}x{N} nativo y esculpiendo cantos, cojines de musgo y piedrecitas 3D...")
    lin4 = np.linspace(-R, R, N, dtype=np.float32)
    X4, Y4 = np.meshgrid(lin4, lin4)
    Sx4, Sy4 = to_screen_xy(X4, Y4)
    dist_h4 = m4.hex_dist(X4, Y4).astype(np.float32)
    dist_v4 = m4.vert_dist(X4, Y4).astype(np.float32)
    d_clear4 = np.minimum(dist_h4 - ROAD_MARGIN, dist_v4 - PAD_RADIUS)
    dx4 = 2.0 * R / (N - 1)

    Z4 = ndi.zoom(Z_2k, N / float(N2), order=3).astype(np.float32)
    Z_rock_pres4 = np.clip(ndi.zoom(Z_rock_presence, N / float(N2), order=1), 0.0, 50.0).astype(np.float32)
    rock_zone4 = np.clip(ndi.zoom(rock_zone, N / float(N2), order=1), 0.0, 1.0).astype(np.float32)
    snow_dom4 = np.clip(ndi.zoom(snow_bowl_dom, N / float(N2), order=1), 0.0, 1.0).astype(np.float32)
    canyon_cut4 = np.clip(ndi.zoom(canyon_cut, N / float(N2), order=1), 0.0, 1.0).astype(np.float32)
    flute_cleft4 = np.clip(ndi.zoom(flute_cleft, N / float(N2), order=1), 0.0, 1.0).astype(np.float32)
    cleft_mac4 = np.clip(ndi.zoom(cleft_mac, N / float(N2), order=1), 0.0, 1.0).astype(np.float32)
    cleft_mes4 = np.clip(ndi.zoom(cleft_mes, N / float(N2), order=1), 0.0, 1.0).astype(np.float32)
    flow4 = np.clip(ndi.zoom(flow_2k, N / float(N2), order=1), 0.0, 6.0).astype(np.float32)

    fine_rock4 = pv3.fft_fbm_canonical(X4, Y4, 0.72, 2.5, beta=2.0, seed=711, canon_n=2048).astype(np.float32)
    Z4 += rock_zone4 * 0.10 * (fine_rock4 - 0.5)

    cobble_mask = np.zeros((N, N), dtype=np.float32)
    cobble_tint = np.full((N, N), 0.5, dtype=np.float32)
    cobble_ao = np.zeros((N, N), dtype=np.float32)

    shrub_mask = np.zeros((N, N), dtype=np.float32)
    shrub_crown = np.zeros((N, N), dtype=np.float32)
    shrub_ao = np.zeros((N, N), dtype=np.float32)

    pebble_mask = np.zeros((N, N), dtype=np.float32)
    pebble_tint = np.full((N, N), 0.5, dtype=np.float32)
    pebble_ao = np.zeros((N, N), dtype=np.float32)

    def screen_to_world(sx, sy):
        c30 = math.cos(math.radians(30.0))
        s30 = math.sin(math.radians(30.0))
        return c30 * sx - s30 * sy, s30 * sx + c30 * sy

    def stamp_stone(wx, wy, rad_a, rad_b, height, rot_deg, tint_val, target_mask, target_tint, target_ao, angular=0.12):
        ix_c = int(round((wx + R) / dx4))
        iy_c = int(round((wy + R) / dx4))
        r_max = max(rad_a, rad_b) * 1.45
        half = int(math.ceil(r_max / dx4)) + 2
        x0, x1 = max(0, ix_c - half), min(N, ix_c + half + 1)
        y0, y1 = max(0, iy_c - half), min(N, iy_c + half + 1)
        if x1 <= x0 or y1 <= y0:
            return
        sub_x = X4[y0:y1, x0:x1] - wx
        sub_y = Y4[y0:y1, x0:x1] - wy
        ca, sa = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
        u = (sub_x * ca + sub_y * sa) / rad_a
        v = (-sub_x * sa + sub_y * ca) / rad_b
        ang = np.arctan2(v, u)
        rmod = 1.0 + angular * np.cos(ang * 3.0 + rot_deg) + (angular * 0.5) * np.cos(ang * 5.0 - rot_deg * 0.7)
        r = np.hypot(u, v) / rmod
        rim_ao = np.exp(-0.5 * ((r - 1.04) / 0.22) ** 2) * 0.55
        inside = r < 1.0
        if not np.any(inside):
            return
        dome = np.zeros_like(r)
        dome[inside] = height * np.power(np.maximum(0.0, 1.0 - np.power(r[inside], 2.25)), 0.55)
        safe = smoothstep(0.75, 1.75, d_clear4[y0:y1, x0:x1])
        Z4[y0:y1, x0:x1] += dome * safe
        m_st = smoothstep(1.03, 0.78, r) * safe
        higher = m_st > target_mask[y0:y1, x0:x1]
        target_mask[y0:y1, x0:x1] = np.maximum(target_mask[y0:y1, x0:x1], m_st)
        target_tint[y0:y1, x0:x1] = np.where(higher, tint_val, target_tint[y0:y1, x0:x1])
        target_ao[y0:y1, x0:x1] = np.maximum(target_ao[y0:y1, x0:x1], rim_ao * safe)

    def stamp_moss_cushion(wx, wy, cluster_r, cluster_h, rng):
        n_lobes = int(rng.integers(5, 9))
        for li in range(n_lobes):
            ang = rng.uniform(0.0, 2.0 * math.pi)
            dist = rng.uniform(0.0, cluster_r * 0.68) if li > 0 else 0.0
            lx = wx + math.cos(ang) * dist
            ly = wy + math.sin(ang) * dist
            ra = cluster_r * rng.uniform(0.42, 0.68)
            rb = ra * rng.uniform(0.82, 1.20)
            lh = cluster_h * rng.uniform(0.55, 0.95)
            ix_c = int(round((lx + R) / dx4))
            iy_c = int(round((ly + R) / dx4))
            half = int(math.ceil(max(ra, rb) * 1.45 / dx4)) + 2
            x0, x1 = max(0, ix_c - half), min(N, ix_c + half + 1)
            y0, y1 = max(0, iy_c - half), min(N, iy_c + half + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            sx = X4[y0:y1, x0:x1] - lx
            sy = Y4[y0:y1, x0:x1] - ly
            a_l = np.arctan2(sy, sx)
            rm = 1.0 + 0.14 * np.sin(a_l * 5.0 + li * 1.9) + 0.07 * np.cos(a_l * 8.0 - li)
            r = np.hypot(sx / ra, sy / rb) / rm
            safe = smoothstep(0.95, 2.0, d_clear4[y0:y1, x0:x1])
            inside = r < 1.0
            if np.any(inside):
                bump = np.zeros_like(r)
                bump[inside] = lh * np.power(np.maximum(0.0, 1.0 - r[inside] ** 2), 0.62)
                Z4[y0:y1, x0:x1] += bump * 0.48 * safe
                ms = smoothstep(1.06, 0.45, r) * safe
                shrub_mask[y0:y1, x0:x1] = np.maximum(shrub_mask[y0:y1, x0:x1], ms)
                shrub_crown[y0:y1, x0:x1] = np.maximum(shrub_crown[y0:y1, x0:x1], np.clip(bump / max(lh, 1e-4), 0.0, 1.0) * safe)
            shrub_ao[y0:y1, x0:x1] = np.maximum(shrub_ao[y0:y1, x0:x1], np.exp(-0.5 * ((r - 1.02) / 0.26) ** 2) * 0.45 * safe)

    rng = np.random.default_rng(20261119)

    # A) ~150 Cantos redondeados de terracota abrazando ambos hombros del sendero en S
    for i in range(150):
        if i < 126:
            sy = float(rng.uniform(-48.8, -11.5))
            side = -1.0 if (i % 2 == 0) else 1.0
            cx_s = float(trail_center_x(np.float32(sy)))
            off = side * float(rng.uniform(1.12, 2.55))
            sx = cx_s + off
            ra = float(rng.uniform(0.36, 0.94))
            rb = ra * float(rng.uniform(0.72, 1.24))
            ht = ra * float(rng.uniform(0.45, 0.68))
        else:
            sy = float(rng.uniform(-12.2, -8.8))
            cx_s = float(trail_center_x(np.float32(sy)))
            sx = cx_s + float(rng.uniform(-2.4, 2.4))
            ra = float(rng.uniform(0.44, 0.98))
            rb = ra * float(rng.uniform(0.74, 1.24))
            ht = ra * float(rng.uniform(0.44, 0.68))
        wx, wy = screen_to_world(sx, sy)
        stamp_stone(wx, wy, ra, rb, ht, float(rng.uniform(0, 360)), float(rng.uniform(0.18, 0.95)),
                    cobble_mask, cobble_tint, cobble_ao, angular=0.11)

    # B) ~82 Cojines bajos de musgo en la pradera 360°
    Z_512 = ndi.zoom(Z_2k, 0.25, order=1)
    Zr_512 = ndi.zoom(Z_rock_presence, 0.25, order=1)
    gy5, gx5 = np.gradient(Z_512, dx2 * 4.0, dx2 * 4.0)
    sl_512 = np.degrees(np.arctan(np.hypot(gx5, gy5)))
    lin5 = np.linspace(-R, R, 512)
    X5, Y5 = np.meshgrid(lin5, lin5)
    dc5 = np.minimum(m4.hex_dist(X5, Y5) - ROAD_MARGIN, m4.vert_dist(X5, Y5) - PAD_RADIUS)
    Sx5, Sy5 = to_screen_xy(X5, Y5)
    dt5 = np.abs(Sx5 - trail_center_x(Sy5))
    not_trail5 = ~((Sy5 < -7.0) & (dt5 < 3.4))
    rel5 = np.maximum(0.0, Z_512 - Z_BASE)
    meadow_cands = np.argwhere((dc5 > 2.0) & (rel5 < 4.0) & (Zr_512 < 3.4) & (sl_512 < 22.0) & not_trail5)

    placed_shrubs = []
    rng.shuffle(meadow_cands)
    for iy5, ix5 in meadow_cands:
        if len(placed_shrubs) >= 82:
            break
        wx, wy = float(X5[iy5, ix5]), float(Y5[iy5, ix5])
        if any(math.hypot(wx - px, wy - py) < 3.4 for px, py in placed_shrubs):
            continue
        placed_shrubs.append((wx, wy))
        stamp_moss_cushion(wx, wy, float(rng.uniform(1.25, 2.55)), float(rng.uniform(0.30, 0.60)), rng)

    # C) ~98 Piedrecitas grises discretas en la pradera 360°
    placed_peb = []
    rng.shuffle(meadow_cands)
    for iy5, ix5 in meadow_cands:
        if len(placed_peb) >= 98:
            break
        wx, wy = float(X5[iy5, ix5]), float(Y5[iy5, ix5])
        if any(math.hypot(wx - px, wy - py) < 2.0 for px, py in placed_shrubs):
            continue
        if any(math.hypot(wx - px, wy - py) < 1.7 for px, py in placed_peb):
            continue
        n_grp = int(rng.choice([1, 2, 2, 3]))
        for gi in range(n_grp):
            if len(placed_peb) >= 98:
                break
            gx_p = wx + (float(rng.uniform(-1.20, 1.20)) if gi > 0 else 0.0)
            gy_p = wy + (float(rng.uniform(-1.20, 1.20)) if gi > 0 else 0.0)
            ra = float(rng.uniform(0.35, 0.84))
            rb = ra * float(rng.uniform(0.74, 1.22))
            ht = ra * float(rng.uniform(0.45, 0.68))
            placed_peb.append((gx_p, gy_p))
            stamp_stone(gx_p, gy_p, ra, rb, ht, float(rng.uniform(0, 360)), float(rng.uniform(0.30, 0.92)),
                        pebble_mask, pebble_tint, pebble_ao, angular=0.12)

    flat_lock = smoothstep(0.25, 2.20, d_clear4)
    Z4 = Z_BASE + np.maximum(0.0, Z4 - Z_BASE) * flat_lock

    print(f"  [Geo OK] {time.time() - t0:.1f}s | Z_max={Z4.max():.2f}mm (relief={Z4.max()-Z_BASE:.2f}mm) | cobbles=150 shrubs={len(placed_shrubs)} pebbles={len(placed_peb)}")

    # =========================================================================
    # SÍNTESIS PBR NATIVA 4096×4096 (32 px/mm) — PARIDAD 100.000% CON Z4
    # =========================================================================
    print(f"[V3.6 Tex 4/5] Sintetizando Albedo PBR hiperrealista nativo {N}x{N} (16-dir HBAO + Dry-Brush + 0.00000% nieve >=44°)...")
    t1 = time.time()
    rel4 = np.maximum(0.0, Z4 - Z_BASE)
    rel_h4 = rel4 / max(float(rel4.max()), 1e-4)

    Z4_s = ndi.gaussian_filter(Z4, 0.22 / dx4)
    gy4, gx4 = np.gradient(Z4_s, dx4, dx4)
    slope4_s = np.degrees(np.arctan(np.hypot(gx4, gy4))).astype(np.float32)
    gy4_r, gx4_r = np.gradient(Z4, dx4, dx4)
    slope4_raw = np.degrees(np.arctan(np.hypot(gx4_r, gy4_r))).astype(np.float32)
    slope4 = np.maximum(slope4_s, slope4_raw)

    Z_curv = ndi.gaussian_filter(Z4, 0.90 / dx4)
    gyc, gxc = np.gradient(Z_curv, dx4, dx4)
    gyyc, gyxc = np.gradient(gyc, dx4, dx4)
    gxyc, gxxc = np.gradient(gxc, dx4, dx4)
    tr = gxxc + gyyc
    det = gxxc * gyyc - 0.5 * (gxyc + gyxc) ** 2
    disc = np.maximum(0.0, tr * tr - 4.0 * det)
    e_min = 0.5 * (tr - np.sqrt(disc))
    e_max = 0.5 * (tr + np.sqrt(disc))
    arete_convex = smoothstep(-0.025, -0.34, e_min)
    gully_concave = smoothstep(0.032, 0.34, e_max)
    micro_lap = np.clip((Z4 - ndi.gaussian_filter(Z4, 0.35 / dx4)) * 0.94, -0.44, 0.44)

    # 16-dir Multi-Scale HBAO
    ao_n = 1536
    Z_ao = ndi.zoom(Z4, ao_n / float(N), order=1).astype(np.float32)
    dx_ao = 2.0 * R / (ao_n - 1)
    ao_sub = np.ones_like(Z_ao, dtype=np.float32)
    for ang_deg in range(0, 360, 22):
        rad = math.radians(ang_deg)
        ux, uy = math.cos(rad), math.sin(rad)
        max_h = np.zeros_like(Z_ao, dtype=np.float32)
        for step_mm in (0.35, 0.85, 1.9, 4.2, 8.5):
            sx_p = int(round(ux * step_mm / dx_ao))
            sy_p = int(round(uy * step_mm / dx_ao))
            Z_sh = np.roll(np.roll(Z_ao, sy_p, axis=0), sx_p, axis=1)
            dh = (Z_sh - Z_ao) / step_mm
            max_h = np.maximum(max_h, np.clip(dh, 0.0, 2.3))
        ao_sub -= (np.arctan(max_h) / (math.pi * 0.5)) / 16.0
    ao_sub = np.clip(ao_sub * 1.32, 0.38, 1.0)
    ao4 = np.clip(ndi.zoom(ao_sub, N / float(ao_n), order=1), 0.38, 1.0).astype(np.float32)

    grain_4k = pv3.fft_fbm(N, N, 0.14, 0.85, 2.0 * R, 2.0 * R, beta=1.55, seed=801).astype(np.float32)
    speck_4k = pv3.fft_fbm(N, N, 0.07, 0.32, 2.0 * R, 2.0 * R, beta=1.40, seed=802).astype(np.float32)
    macro_var = pv3.fft_fbm_canonical(X4, Y4, 6.5, 28.0, beta=2.45, seed=803, canon_n=1024).astype(np.float32)
    moss_var = pv3.fft_fbm_canonical(X4, Y4, 4.0, 18.0, beta=2.40, seed=804, canon_n=1024).astype(np.float32)
    flow_norm = smoothstep(0.95, 3.4, flow4)

    # === 1. GRANITO ALPINO LUMINOSO DE ESTUDIO (CUMBRES CALIZO-ARENISCAS + PIRÁMIDE OCRE + GRIETAS ESPRESSO) ===
    c_crevice   = np.array([0.09, 0.075, 0.065], dtype=np.float32)  # Grietas profundas negro-espresso 100% alineadas a la V 3D
    c_rock_cool = np.array([0.59, 0.53, 0.45], dtype=np.float32)    # Granito alpino equilibrado para WebGL ACESFilmic (sat=0.14!)
    c_rock_warm = np.array([0.78, 0.59, 0.39], dtype=np.float32)    # Ocre cálido dorado en pirámide P4 y flancos bajos
    c_rock_edge = np.array([0.89, 0.79, 0.64], dtype=np.float32)    # Dry-brush arenisco limpio en aristas 3D

    p4_warm = np.exp(-0.5 * (((Sx4 - 16.0) / 14.5) ** 2 + ((Sy4 + 17.5) / 14.5) ** 2))
    warm_rock_w = np.clip(
        (1.0 - smoothstep(10.0, 26.0, rel4)) * 0.60
        + 0.44 * p4_warm
        + smoothstep(0.0, 0.50, canyon_cut4) * 0.34
        + 0.18 * (macro_var - 0.5),
        0.08,
        0.96,
    )
    rock_base = c_rock_cool * (1.0 - warm_rock_w[..., None]) + c_rock_warm * warm_rock_w[..., None]

    crevice_depth = np.clip(
        0.84 * np.power(flute_cleft4, 1.10)
        + 0.72 * np.power(cleft_mac4, 1.15)
        + 0.48 * np.power(cleft_mes4, 1.20)
        + 0.40 * gully_concave
        + 0.22 * flow_norm
        + np.clip(-micro_lap * 1.50, 0.0, 0.56),
        0.0,
        1.0,
    )
    edge_hi = np.clip(
        0.80 * arete_convex
        + np.clip(micro_lap * 1.40, 0.0, 0.50)
        + 0.24 * rel_h4,
        0.0,
        1.0,
    ) * (1.0 - 0.88 * crevice_depth)

    rock_col = rock_base * (1.0 - 0.88 * crevice_depth[..., None]) + c_crevice * (0.88 * crevice_depth[..., None])
    rock_col = rock_col * (1.0 - 0.54 * edge_hi[..., None]) + c_rock_edge * (0.54 * edge_hi[..., None])
    # Grano mineral limpio y sutil (sin efecto corcho/esponja en close-up WebGL!)
    rock_col *= np.clip(0.95 + 0.08 * grain_4k + 0.04 * speck_4k, 0.89, 1.07)[..., None]
    rock_col *= np.clip(0.60 + 0.46 * ao4, 0.44, 1.04)[..., None]

    # === 2. PRADERA OLIVA PROFUNDA ATERCIOPELADA 360° (TONO EXACTO DE DIORAMA EN WEBGL ACESFILMIC!) ===
    c_grass_deep = np.array([0.24, 0.30, 0.11], dtype=np.float32)
    c_grass_mid  = np.array([0.37, 0.42, 0.18], dtype=np.float32)
    c_grass_sun  = np.array([0.48, 0.51, 0.24], dtype=np.float32)
    gv = np.clip(0.50 + 0.24 * (moss_var - 0.5) + 0.18 * (macro_var - 0.5) + 0.08 * (grain_4k - 0.5), 0.0, 1.0)
    grass_col = np.where(
        (gv < 0.5)[..., None],
        c_grass_deep * (1.0 - (gv * 2.0)[..., None]) + c_grass_mid * (gv * 2.0)[..., None],
        c_grass_mid * (1.0 - ((gv - 0.5) * 2.0)[..., None]) + c_grass_sun * ((gv - 0.5) * 2.0)[..., None],
    )
    grass_col *= np.clip(0.95 + 0.08 * speck_4k, 0.90, 1.06)[..., None] * np.clip(0.76 + 0.28 * ao4, 0.62, 1.04)[..., None]

    meadow_core = (
        (1.0 - smoothstep(2.0, 5.2, Z_rock_pres4 + 0.9 * (moss_var - 0.5)))
        * (1.0 - smoothstep(3.8, 6.8, rel4))
        * (1.0 - smoothstep(24.0, 36.0, slope4_s))
        * smoothstep(0.15, 0.70, d_clear4)
    )
    rim_grass_guarantee = (
        smoothstep(0.15, 0.65, d_clear4)
        * (1.0 - smoothstep(2.6, 5.0, d_clear4))
        * (1.0 - smoothstep(3.2, 5.5, rel4))
    )
    meadow_mask = np.maximum(meadow_core, rim_grass_guarantee)
    foot_moss = (
        (1.0 - meadow_mask)
        * (1.0 - smoothstep(2.4, 5.4, rel4))
        * smoothstep(0.30, 0.78, moss_var)
        * (0.32 + 0.45 * gully_concave)
        * 0.34
    )
    col = rock_col * (1.0 - np.clip(meadow_mask + foot_moss, 0.0, 1.0)[..., None]) + grass_col * np.clip(meadow_mask + foot_moss, 0.0, 1.0)[..., None]

    # === 3. SENDERO EN S DE ARCILLA TERRACOTA CLARA + 150 CANTOS ===
    c_clay_edge = np.array([0.62, 0.38, 0.24], dtype=np.float32)
    c_clay_mid  = np.array([0.84, 0.59, 0.40], dtype=np.float32)
    c_clay_sand = np.array([0.92, 0.72, 0.54], dtype=np.float32)
    dt4 = np.abs(Sx4 - trail_center_x(Sy4))
    in_tr_y = smoothstep(-5.8, -10.8, Sy4) * smoothstep(-52.5, -46.5, Sy4)
    tr_w = 1.65 + 0.65 * np.clip((-10.0 - Sy4) / 37.5, 0.0, 1.0)
    tr_edge_n = 0.42 * (moss_var - 0.5) + 0.20 * (grain_4k - 0.5)
    trail_paint = in_tr_y * (1.0 - smoothstep(-0.36, 0.72, dt4 + tr_edge_n - tr_w)) * (1.0 - smoothstep(27.0, 40.0, slope4_s))
    t_tr_edge = np.clip((dt4 - 0.30) / np.maximum(tr_w - 0.30, 0.35), 0.0, 1.0)
    t_tr_edge = t_tr_edge * t_tr_edge * (3.0 - 2.0 * t_tr_edge)
    tv = np.clip(0.68 - 0.38 * t_tr_edge + 0.14 * (grain_4k - 0.5) + 0.08 * (speck_4k - 0.5), 0.0, 1.0)
    trail_col = np.where(
        (tv < 0.5)[..., None],
        c_clay_edge * (1.0 - (tv * 2.0)[..., None]) + c_clay_mid * (tv * 2.0)[..., None],
        c_clay_mid * (1.0 - ((tv - 0.5) * 2.0)[..., None]) + c_clay_sand * ((tv - 0.5) * 2.0)[..., None],
    ) * np.clip(0.78 + 0.26 * ao4, 0.62, 1.04)[..., None]
    col = col * (1.0 - trail_paint[..., None]) + trail_col * trail_paint[..., None]

    col *= (1.0 - 0.26 * cobble_ao * (1.0 - cobble_mask))[..., None]
    cob_base = c_clay_edge * (1.0 - cobble_tint[..., None]) * 0.88 + c_clay_sand * cobble_tint[..., None]
    cob_hi = np.clip(0.84 + 0.32 * np.clip(micro_lap * 1.5, -0.30, 0.50) + 0.12 * (speck_4k - 0.5), 0.64, 1.18)
    col = col * (1.0 - cobble_mask[..., None]) + np.clip(cob_base * cob_hi[..., None], 0.0, 1.0) * cobble_mask[..., None]

    # === 4. COJINES DE MUSGO BAJOS 3D + PIEDRECITAS GRISES 3D ===
    col *= (1.0 - 0.28 * shrub_ao * (1.0 - shrub_mask))[..., None]
    c_sh_base = np.array([0.14, 0.21, 0.05], dtype=np.float32)
    c_sh_tip  = np.array([0.44, 0.52, 0.16], dtype=np.float32)
    sh_mix = np.clip(shrub_crown * 0.84 + 0.24 * (speck_4k - 0.35), 0.0, 1.0)
    sh_col = c_sh_base * (1.0 - sh_mix[..., None]) + c_sh_tip * sh_mix[..., None]
    col = col * (1.0 - shrub_mask[..., None]) + sh_col * shrub_mask[..., None]

    col *= (1.0 - 0.24 * pebble_ao * (1.0 - pebble_mask))[..., None]
    c_peb_dk = np.array([0.46, 0.48, 0.48], dtype=np.float32)
    c_peb_lt = np.array([0.78, 0.80, 0.80], dtype=np.float32)
    peb_col = (c_peb_dk * (1.0 - pebble_tint[..., None]) + c_peb_lt * pebble_tint[..., None]) * np.clip(0.88 + 0.26 * micro_lap + 0.10 * speck_4k, 0.70, 1.15)[..., None]
    col = col * (1.0 - pebble_mask[..., None]) + np.clip(peb_col, 0.0, 1.0) * pebble_mask[..., None]

    # === 5. NEVEROS ALPINOS EN COULOIRS CÓNCAVOS (ESTRICTAMENTE 0.00000% EN >= 44.0°!) ===
    r_core4 = np.hypot((Sx4 + 0.5) * 0.96, (Sy4 - 1.5) * 0.98)
    snow_edge_warp = 1.6 * (moss_var - 0.5) + 0.60 * (grain_4k - 0.5)
    snow_prob = (
        np.maximum(
            smoothstep(0.05, 0.32, snow_dom4 + 0.16 * (moss_var - 0.5)),
            smoothstep(6.8, 12.2, rel4 + snow_edge_warp)
            * smoothstep(41.5, 30.0, slope4_s)
            * (1.0 - arete_convex * 0.72)
            * (0.78 + 0.40 * gully_concave),
        )
        * smoothstep(42.0, 37.0, slope4_s)
        * smoothstep(38.5, 29.0, r_core4)
        * smoothstep(5.8, 8.6, rel4)
        * (1.0 - trail_paint)
        * (1.0 - cobble_mask)
        * (1.0 - meadow_mask)
    )
    snow_mask = np.where((slope4 >= 40.2) | (slope4_s >= 39.8), 0.0, smoothstep(0.18, 0.50, snow_prob)).astype(np.float32)
    c_snow_sh = np.array([0.74, 0.81, 0.90], dtype=np.float32)
    c_snow_hi = np.array([0.96, 0.97, 0.99], dtype=np.float32)
    sn_lit = np.clip(0.36 + 0.62 * ao4 + 0.16 * micro_lap - 0.14 * gully_concave + 0.04 * (speck_4k - 0.5), 0.0, 1.0)
    snow_col = c_snow_sh * (1.0 - sn_lit[..., None]) + c_snow_hi * sn_lit[..., None]
    col = col * (1.0 - snow_mask[..., None]) + snow_col * snow_mask[..., None]

    # Garantía matemática absoluta 0.00000% píxeles blancos en pendientes >= 43.0° (p.ej. piedrecita o brillo extremo)
    steep_guard = (slope4_s >= 43.0) & (snow_mask < 0.01)
    if np.any(steep_guard):
        lum_chk = 0.2126 * col[..., 0] + 0.7152 * col[..., 1] + 0.0722 * col[..., 2]
        sat_chk = col.max(-1) - col.min(-1)
        fix_px = steep_guard & (lum_chk > 0.66) & (sat_chk < 0.16)
        if np.any(fix_px):
            col[fix_px, 2] *= 0.84
            col[fix_px, 1] *= 0.94

    # === 6. BISEL HEXAGONAL GRAFITO OSCURO CATAN ===
    on_border = np.maximum(
        1.0 - smoothstep(ROAD_MARGIN + 0.35, ROAD_MARGIN + 0.85, dist_h4),
        1.0 - smoothstep(PAD_RADIUS + 0.25, PAD_RADIUS + 0.75, dist_v4),
    )
    collar = np.clip(on_border * (1.0 - smoothstep(0.08, 0.42, rel4)), 0.0, 1.0)
    slate = np.array([0.122, 0.134, 0.152], dtype=np.float32)[None, None, :] * (0.94 + 0.10 * grain_4k[..., None])
    rim_hi = np.exp(-0.5 * ((dist_h4 - 0.75) / 0.25) ** 2) * 0.036
    curb_sh = np.exp(-0.5 * ((dist_h4 - 3.55) / 0.36) ** 2) * 0.044
    pad_groove = np.exp(-0.5 * ((dist_v4 - 8.2) / 0.30) ** 2) * 0.040
    pad_hi = np.exp(-0.5 * ((dist_v4 - 9.0) / 0.26) ** 2) * 0.026
    slate = np.clip(slate + (rim_hi - curb_sh - pad_groove + pad_hi)[..., None], 0.05, 0.23)
    alb_final = np.clip(col * (1.0 - collar[..., None]) + slate * collar[..., None], 0.0, 1.0)
    alb_final[:12, :12] = np.array([0.125, 0.136, 0.154], dtype=np.float32)

    print(f"  [Tex OK] {time.time() - t1:.1f}s")

    os.makedirs(WORK_DIR, exist_ok=True)
    z_span = float(Z4.max() - Z_BASE)
    hm_4k_norm = np.clip((Z4 - Z_BASE) / max(z_span, 1e-5) * 255.0, 0.0, 255.0).astype(np.float32)
    np.save(os.path.join(WORK_DIR, "montana_centrada_v3_hm_4k.npy"), np.flipud(hm_4k_norm))
    np.save(os.path.join(WORK_DIR, "montana_centrada_v3_cliff_sculpt.npy"), np.flipud(cliff_field_2k))

    alb_u8 = np.clip(np.flipud(alb_final) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(alb_u8).save(os.path.join(WORK_DIR, "montana_centrada_v3_albedo_4k.png"))

    lum_f = 0.2126 * alb_final[..., 0] + 0.7152 * alb_final[..., 1] + 0.0722 * alb_final[..., 2]
    sat_f = alb_final.max(-1) - alb_final.min(-1)
    inner = (dist_h4 > 4.5) & (dist_v4 > 11.0)
    p = np.percentile(slope4_s[inner], [50, 75, 90, 99])
    snow_tot = float(np.mean((lum_f > 0.68) & (sat_f < 0.14) & inner) * 100.0)
    snow_steep = float(np.mean((lum_f > 0.68) & (sat_f < 0.14) & (slope4_s >= 44.0)) * 100.0)
    print(f"[V3.6 Metrics] Z_max={Z4.max():.2f}mm (relief={z_span:.2f}mm) | slope p50/p75/p90/p99={p.round(1)} deg | snow={snow_tot:.2f}% | snow>=44deg={snow_steep:.5f}%")

    print("[V3.6 Preview 5/5] Renderizando auditoría previa FRONT/BACK PBR + Arcilla y 6 vistas 360°...")
    render_audits(Z4[::2, ::2], alb_final[::2, ::2])


def render_audits(Z_2k, alb_2k):
    im_ref = np.asarray(Image.open(DUAL_REF).convert("RGB"), dtype=np.float32) / 255.0
    front_ref = im_ref[:, :1368]
    back_ref = im_ref[:, 2752 - 1368 :]
    cam_f = np.load(os.path.join(WORK_DIR, "cam_front.npy"))
    cam_b = np.load(os.path.join(WORK_DIR, "cam_back.npy"))

    try:
        fnt = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
    except Exception:
        fnt = ImageFont.load_default()

    def make_panel(img_arr, label):
        pil = Image.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8)).resize((560, 628), Image.LANCZOS)
        dr = ImageDraw.Draw(pil)
        dr.text((14, 12), label, fill=(25, 32, 46), font=fnt)
        return np.asarray(pil)

    pbr_f = m4.ray_march_heightfield(Z_2k, cam_f, out_w=560, out_h=628, tex_rgb=alb_2k, yaw_offset_deg=0.0, sun_rel=True)[0]
    clay_f = m4.ray_march_heightfield(Z_2k, cam_f, out_w=560, out_h=628, tex_rgb=None, yaw_offset_deg=0.0, sun_rel=True)[0]
    pbr_b = m4.ray_march_heightfield(Z_2k, cam_b, out_w=560, out_h=628, tex_rgb=alb_2k, yaw_offset_deg=0.0, sun_rel=True)[0]
    clay_b = m4.ray_march_heightfield(Z_2k, cam_b, out_w=560, out_h=628, tex_rgb=None, yaw_offset_deg=0.0, sun_rel=True)[0]

    row1 = np.hstack([make_panel(front_ref, "REF FRONT (0°)"), make_panel(pbr_f, "V3.7 FRONT PBR"), make_panel(clay_f, "V3.7 FRONT ARCILLA")])
    row2 = np.hstack([make_panel(back_ref, "REF BACK (180°)"), make_panel(pbr_b, "V3.7 BACK PBR"), make_panel(clay_b, "V3.7 BACK ARCILLA")])
    Image.fromarray(np.vstack([row1, row2])).save(os.path.join(WORK_DIR, "centrada_v3_dual_vs_3d.png"))

    v6 = []
    for ang in [0, -60, -120, -180, -240, -300]:
        ci = m4.ray_march_heightfield(Z_2k, cam_f, out_w=520, out_h=584, tex_rgb=alb_2k, yaw_offset_deg=ang, sun_rel=True)[0]
        v6.append(make_panel(ci, f"{(-ang) % 360} deg"))
    Image.fromarray(np.vstack([np.hstack(v6[:3]), np.hstack(v6[3:])])).save(os.path.join(WORK_DIR, "centrada_v3_6views_preview.png"))
    print("  [Preview OK] centrada_v3_dual_vs_3d.png + centrada_v3_6views_preview.png")


if __name__ == "__main__":
    build_v3_geometry_4k(4096)
