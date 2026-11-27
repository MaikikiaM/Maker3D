"""
Horneado PBR 8K semántico y renders de auditoría para Bioma Campo v05 (Colab L4 Zero-Idle):
  - `--stage prep-local`: calcula prominencia local y color pictórico por esquina de triángulo (CORNER)
    sobre los 15 IDs semánticos (`face_slot 0..14`), aisla UVs `Plinth_Gunmetal` vs `Organic_Campo_8K`
    (`Smart UV Project` en PC local a 0 créditos) y guarda `campo_gpu_bake.blend`.
  - `--stage gpu-bake-only`: ejecuta Cycles OPTIX GPU en Colab L4 horneando `baked_albedo.png` (8K con
    Cycles AO), `baked_normal.png` (8K tangente micro-relieve) y `baked_orm.png` (4K AO/Rough/Metal),
    más 6 vistas PBR + ortográfico + close-ups macro de árboles y props.
  - `--stage finalize-local`: empaqueta las texturas horneadas en `tile_pbr_v05.glb` y compone el
    mosaico `audit_6views_pbr.png` en local (0 créditos Colab).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = REPO_ROOT / "Tableros-Catan" / "losetas" / "campo" / "work"

R = 64.0
Z_BASE = 25.0

# IDs semánticos producidos por build_campo_clay.py
SEM_PLINTH = 0
SEM_TERRAIN = 1
SEM_OAK_LEAF = 2
SEM_BARK = 3
SEM_CYPRESS_LEAF = 4
SEM_BUSH = 5
SEM_STONE = 6
SEM_FENCE_WOOD = 7
SEM_FENCE_BERM = 8
SEM_STRAW = 9
SEM_ROPE = 10
SEM_WHEAT_CROWN = 11
SEM_CART_WOOD = 12
SEM_IRON = 13
SEM_WHEEL_BACK = 14


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    c = np.clip(rgb, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bilinear_arr(arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    fx = np.clip((x + R) / (2.0 * R), 0.0, 1.0) * (w - 1)
    fy = np.clip((y + R) / (2.0 * R), 0.0, 1.0) * (h - 1)
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    tx, ty = fx - x0, fy - y0
    return (
        arr[y0, x0] * (1.0 - tx) * (1.0 - ty)
        + arr[y0, x1] * tx * (1.0 - ty)
        + arr[y1, x0] * (1.0 - tx) * ty
        + arr[y1, x1] * tx * ty
    )


def _nearest_arr(arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    ix = np.clip(np.rint(np.clip((x + R) / (2.0 * R), 0.0, 1.0) * (w - 1)).astype(np.int32), 0, w - 1)
    iy = np.clip(np.rint(np.clip((y + R) / (2.0 * R), 0.0, 1.0) * (h - 1)).astype(np.int32), 0, h - 1)
    return arr[iy, ix]


def _compute_vertex_prominence(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Calcula prominencia local (puntas Voronoi/crestas positivas vs grietas negativas) en <1s."""
    n_v = len(V)
    e0 = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    e1 = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    deg = np.bincount(e0, minlength=n_v).astype(np.float32)
    deg = np.maximum(deg, 1.0)

    # Media vecina en 2 pasadas para capturar crestas y hendiduras finas (0.06-0.25 mm)
    pos_avg = V.astype(np.float32).copy()
    for _ in range(3):
        acc = np.zeros_like(pos_avg)
        np.add.at(acc, e0, pos_avg[e1])
        pos_avg = acc / deg[:, None]

    # Normal aproximada de vértice desde vecinos/caras
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0).astype(np.float32)
    vn = np.zeros((n_v, 3), dtype=np.float32)
    np.add.at(vn, F[:, 0], fn)
    np.add.at(vn, F[:, 1], fn)
    np.add.at(vn, F[:, 2], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-8)

    delta = V.astype(np.float32) - pos_avg
    prom = np.sum(delta * vn, axis=1)
    return np.clip(prom / 0.055, -1.0, 1.0)


def _hash_noise_3d(V: np.ndarray) -> np.ndarray:
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    n1 = np.sin(x * 1.37 + np.cos(y * 1.13)) * np.cos(z * 1.71 + np.sin(x * 0.89))
    n2 = np.sin(x * 4.31 - y * 3.77 + z * 2.93)
    return np.clip(0.5 + 0.30 * n1 + 0.20 * n2, 0.0, 1.0)


def compute_semantic_corner_attributes(
    V: np.ndarray,
    F: np.ndarray,
    face_slot: np.ndarray,
    fields: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (loop_col_linear_rgba [3*F,4], loop_matspec [3*F,4]) sin sangrado entre materiales."""
    prom_v = _compute_vertex_prominence(V, F)
    noise_v = _hash_noise_3d(V)

    n_f = len(F)
    f_v0, f_v1, f_v2 = F[:, 0], F[:, 1], F[:, 2]
    # Evaluar ruido/prominencia por esquina de triángulo (reutiliza cálculo de vértice con slot por cara exacto)
    # Nota: construimos arrays por cara para terreno vs props para velocidad y 0 sangrado en fronteras
    V_fc = ((V[f_v0] + V[f_v1] + V[f_v2]) / 3.0).astype(np.float64)
    prom_f = ((prom_v[f_v0] + prom_v[f_v1] + prom_v[f_v2]) / 3.0).astype(np.float32)
    noise_f = ((noise_v[f_v0] + noise_v[f_v1] + noise_v[f_v2]) / 3.0).astype(np.float32)

    # Colores base en sRGB [0..1] inspirados directamente en paint_000.png
    f_srgb = np.zeros((n_f, 3), dtype=np.float32)
    f_rough = np.full(n_f, 0.72, dtype=np.float32)
    f_metal = np.zeros(n_f, dtype=np.float32)
    f_bump = np.full(n_f, 0.65, dtype=np.float32)

    # ------------------------------------------------------------------
    # 1. Terreno (`SEM_TERRAIN`): parcelas de trigo, tierra entre surcos, caminos y lindes
    # ------------------------------------------------------------------
    m_ter = face_slot == SEM_TERRAIN
    if np.any(m_ter):
        tx = V_fc[m_ter, 0]
        ty = V_fc[m_ter, 1]
        tz = V_fc[m_ter, 2]

        wm = _bilinear_arr(fields["wheat_mask"], tx, ty)
        ws = _bilinear_arr(fields["wheat_sculpt"], tx, ty)
        rm = _bilinear_arr(fields["road_mask"], tx, ty)
        ruts = _bilinear_arr(fields["ruts"], tx, ty)
        stm = _bilinear_arr(fields["staging_mask"], tx, ty)
        fcl = _bilinear_arr(fields["fence_clear"], tx, ty)
        fbm = _bilinear_arr(fields["fbm_fine"], tx, ty)
        pid = _nearest_arr(fields["parcel_id"], tx, ty)
        de = _bilinear_arr(fields["de"], tx, ty)
        dv = _bilinear_arr(fields["dv"], tx, ty)
        d_road = _bilinear_arr(fields["d_road"], tx, ty)

        # Paletas sutilmente diferenciadas entre las 4 parcelas (de dorado miel a trigo tostado de paint_000.png)
        tip_pal = np.array([
            [0.82, 0.60, 0.25],  # fallback
            [0.86, 0.63, 0.27],  # parcela 1 NW
            [0.78, 0.54, 0.21],  # parcela 2 SW
            [0.88, 0.66, 0.30],  # parcela 3 NE
            [0.83, 0.59, 0.24],  # parcela 4 SE
        ], dtype=np.float32)[np.clip(pid, 0, 4)]

        mid_pal = np.array([
            [0.64, 0.43, 0.16],
            [0.68, 0.46, 0.18],
            [0.58, 0.37, 0.13],
            [0.70, 0.49, 0.20],
            [0.65, 0.43, 0.16],
        ], dtype=np.float32)[np.clip(pid, 0, 4)]

        furrow_earth = np.array([0.39, 0.25, 0.12], dtype=np.float32)

        # Factor de altura de espiga normalizado al rango real del cultivo (compatible v05 / fine / ploughed)
        ws_field = fields["wheat_sculpt"]
        wm_field = fields["wheat_mask"]
        pure_ws = ws_field[wm_field > 0.85]
        if len(pure_ws) > 100:
            ws_lo = float(np.percentile(pure_ws, 4.0))
            ws_hi = float(np.percentile(pure_ws, 97.0))
        else:
            ws_lo, ws_hi = 0.35, 1.35
        stalk_t = np.clip((ws - ws_lo) / max(ws_hi - ws_lo, 0.15) * 0.94 + 0.22 * prom_f[m_ter], 0.0, 1.0)
        wind_mod = 0.92 + 0.14 * np.clip(fbm, -1.0, 1.0) + 0.08 * (noise_f[m_ter] - 0.5)

        wheat_rgb = (
            furrow_earth[None, :] * (1.0 - _smoothstep(0.0, 0.32, stalk_t))[:, None]
            + mid_pal * (_smoothstep(0.0, 0.32, stalk_t) * (1.0 - _smoothstep(0.45, 0.95, stalk_t)))[:, None]
            + tip_pal * _smoothstep(0.45, 0.95, stalk_t)[:, None]
        ) * wind_mod[:, None]

        # Camino rural compacto + roderas más oscuras + hierba verde oliva en márgenes y vallas
        loam_light = np.array([0.56, 0.45, 0.31], dtype=np.float32)
        loam_rut = np.array([0.35, 0.26, 0.17], dtype=np.float32)
        grass_olive = np.array([0.32, 0.41, 0.17], dtype=np.float32)
        grass_sun = np.array([0.44, 0.52, 0.22], dtype=np.float32)
        slate_rim = np.array([0.28, 0.27, 0.24], dtype=np.float32)

        rut_w = np.clip(ruts * 5.2 + 0.35 * np.exp(-((d_road - 0.42) / 0.26) ** 2) * rm, 0.0, 1.0)
        road_rgb = (loam_light[None, :] * (1.0 - rut_w[:, None]) + loam_rut[None, :] * rut_w[:, None]) * (0.94 + 0.12 * noise_f[m_ter][:, None])

        # Claro agrícola: tierra apisonada con briznas doradas sueltas
        clearing_rgb = loam_light * 0.90 + np.array([0.12, 0.09, 0.03], dtype=np.float32) * (noise_f[m_ter] > 0.62)[:, None]

        # Hierba en bordes de caminos, junto a vallas y faldón perimetral
        verge_mask = np.clip((1.0 - wm) * (1.0 - np.clip(rm * 1.4, 0.0, 1.0)) * (1.0 - np.clip(stm * 1.5, 0.0, 1.0)), 0.0, 1.0)
        grass_t = np.clip(0.5 + 0.35 * fbm + 0.25 * (noise_f[m_ter] - 0.5), 0.0, 1.0)
        verge_rgb = grass_olive[None, :] * (1.0 - grass_t[:, None]) + grass_sun[None, :] * grass_t[:, None]

        # Transición limpia al borde de carril/pad hacia pizarra terrosa en Z ~ 25.0 mm
        outer_mask = 1.0 - _smoothstep(3.8, 5.2, de) * _smoothstep(10.3, 11.7, dv)
        # Franja herbácea bajo vallas
        fence_verge = (1.0 - fcl) * (1.0 - np.clip(rm * 1.5, 0.0, 1.0))

        ter_rgb = verge_rgb.copy()
        ter_rgb = ter_rgb * (1.0 - np.clip(rm * 1.3, 0.0, 1.0))[:, None] + road_rgb * np.clip(rm * 1.3, 0.0, 1.0)[:, None]
        ter_rgb = ter_rgb * (1.0 - stm)[:, None] + clearing_rgb * stm[:, None]
        ter_rgb = ter_rgb * (1.0 - wm)[:, None] + wheat_rgb * wm[:, None]
        ter_rgb = ter_rgb * (1.0 - 0.55 * fence_verge)[:, None] + verge_rgb * (0.55 * fence_verge)[:, None]
        ter_rgb = ter_rgb * (1.0 - 0.72 * outer_mask)[:, None] + slate_rim[None, :] * (0.72 * outer_mask)[:, None]

        # Micro-claros de tierra bajo árboles
        ter_rgb = np.where((tz[:, None] < Z_BASE + 0.15) & (outer_mask[:, None] > 0.6), slate_rim[None, :], ter_rgb)
        f_srgb[m_ter] = np.clip(ter_rgb, 0.02, 0.96)
        f_rough[m_ter] = 0.86 - 0.14 * wm
        f_bump[m_ter] = 0.55 + 0.35 * wm

    # ------------------------------------------------------------------
    # 2. Roble / Encina (`SEM_OAK_LEAF`): racimos Voronoi iluminados vs grietas musgo
    # ------------------------------------------------------------------
    m = face_slot == SEM_OAK_LEAF
    if np.any(m):
        t = np.clip(0.52 + 0.42 * prom_f[m] + 0.18 * (noise_f[m] - 0.5), 0.0, 1.0)
        c_dark = np.array([0.13, 0.23, 0.06], dtype=np.float32)
        c_mid = np.array([0.27, 0.42, 0.12], dtype=np.float32)
        c_sun = np.array([0.45, 0.59, 0.19], dtype=np.float32)
        f_srgb[m] = np.where((t < 0.5)[:, None], c_dark + (c_mid - c_dark) * (2.0 * t)[:, None], c_mid + (c_sun - c_mid) * (2.0 * t - 1.0)[:, None])
        f_rough[m] = 0.80
        f_bump[m] = 0.85

    # 3. Troncos y ramas (`SEM_BARK`)
    m = face_slot == SEM_BARK
    if np.any(m):
        t = np.clip(0.50 + 0.45 * prom_f[m] + 0.16 * (noise_f[m] - 0.5), 0.0, 1.0)
        f_srgb[m] = np.array([0.17, 0.11, 0.07]) * (1.0 - t)[:, None] + np.array([0.40, 0.28, 0.18]) * t[:, None]
        f_rough[m] = 0.88
        f_bump[m] = 0.95

    # 4. Cipreses mediterráneos (`SEM_CYPRESS_LEAF`)
    m = face_slot == SEM_CYPRESS_LEAF
    if np.any(m):
        t = np.clip(0.50 + 0.42 * prom_f[m] + 0.18 * (noise_f[m] - 0.5), 0.0, 1.0)
        f_srgb[m] = np.array([0.08, 0.16, 0.06]) * (1.0 - t)[:, None] + np.array([0.25, 0.40, 0.16]) * t[:, None]
        f_rough[m] = 0.82
        f_bump[m] = 0.90

    # 5. Arbustos de linde (`SEM_BUSH`)
    m = face_slot == SEM_BUSH
    if np.any(m):
        t = np.clip(0.52 + 0.42 * prom_f[m] + 0.16 * (noise_f[m] - 0.5), 0.0, 1.0)
        f_srgb[m] = np.array([0.16, 0.27, 0.08]) * (1.0 - t)[:, None] + np.array([0.42, 0.56, 0.19]) * t[:, None]
        f_rough[m] = 0.82
        f_bump[m] = 0.80

    # 6. Piedras calizas (`SEM_STONE`)
    m = face_slot == SEM_STONE
    if np.any(m):
        t = np.clip(0.50 + 0.35 * prom_f[m] + 0.24 * (noise_f[m] - 0.5), 0.0, 1.0)
        f_srgb[m] = np.array([0.36, 0.34, 0.30]) * (1.0 - t)[:, None] + np.array([0.66, 0.62, 0.55]) * t[:, None]
        f_rough[m] = 0.86
        f_bump[m] = 0.85

    # 7. Madera partida de vallas (`SEM_FENCE_WOOD`)
    m = face_slot == SEM_FENCE_WOOD
    if np.any(m):
        t = np.clip(0.50 + 0.40 * prom_f[m] + 0.18 * (noise_f[m] - 0.5), 0.0, 1.0)
        f_srgb[m] = np.array([0.24, 0.16, 0.10]) * (1.0 - t)[:, None] + np.array([0.50, 0.37, 0.24]) * t[:, None]
        f_rough[m] = 0.84
        f_bump[m] = 0.90

    # 8. Talud/sombra bajo vallas (`SEM_FENCE_BERM`)
    m = face_slot == SEM_FENCE_BERM
    if np.any(m):
        z_rel = V_fc[m, 2] - Z_BASE
        low_earth = np.clip(1.0 - (z_rel - 2.2) / 1.5, 0.0, 1.0)
        c_shadow = np.array([0.11, 0.09, 0.07], dtype=np.float32)
        c_berm = np.array([0.29, 0.34, 0.15], dtype=np.float32)
        f_srgb[m] = c_shadow[None, :] * (1.0 - low_earth[:, None]) + c_berm[None, :] * low_earth[:, None]
        f_rough[m] = 0.92
        f_bump[m] = 0.40

    # 9. Paja enrollada de balas y cuerpos de gavillas (`SEM_STRAW`)
    m = face_slot == SEM_STRAW
    if np.any(m):
        t = np.clip(0.52 + 0.42 * prom_f[m] + 0.16 * (noise_f[m] - 0.5), 0.0, 1.0)
        f_srgb[m] = np.array([0.60, 0.43, 0.15]) * (1.0 - t)[:, None] + np.array([0.93, 0.77, 0.37]) * t[:, None]
        f_rough[m] = 0.80
        f_bump[m] = 0.90

    # 10. Cuerda trenzada de cáñamo (`SEM_ROPE`)
    m = face_slot == SEM_ROPE
    if np.any(m):
        t = np.clip(0.50 + 0.42 * prom_f[m], 0.0, 1.0)
        f_srgb[m] = np.array([0.26, 0.17, 0.08]) * (1.0 - t)[:, None] + np.array([0.52, 0.38, 0.20]) * t[:, None]
        f_rough[m] = 0.88
        f_bump[m] = 0.85

    # 11. Espigas superiores de gavillas (`SEM_WHEAT_CROWN`)
    m = face_slot == SEM_WHEAT_CROWN
    if np.any(m):
        t = np.clip(0.52 + 0.42 * prom_f[m] + 0.14 * (noise_f[m] - 0.5), 0.0, 1.0)
        f_srgb[m] = np.array([0.58, 0.39, 0.11]) * (1.0 - t)[:, None] + np.array([0.94, 0.73, 0.28]) * t[:, None]
        f_rough[m] = 0.78
        f_bump[m] = 0.85

    # 12. Madera de carro y arado (`SEM_CART_WOOD`)
    m = face_slot == SEM_CART_WOOD
    if np.any(m):
        t = np.clip(0.50 + 0.38 * prom_f[m] + 0.16 * (noise_f[m] - 0.5), 0.0, 1.0)
        f_srgb[m] = np.array([0.24, 0.14, 0.07]) * (1.0 - t)[:, None] + np.array([0.48, 0.32, 0.18]) * t[:, None]
        f_rough[m] = 0.74
        f_bump[m] = 0.75

    # 13. Hierro forjado (bandaje de rueda, 16 remaches, eje y reja de arado `SEM_IRON`)
    m = face_slot == SEM_IRON
    if np.any(m):
        t = np.clip(0.50 + 0.40 * prom_f[m] + 0.16 * (noise_f[m] - 0.5), 0.0, 1.0)
        f_srgb[m] = np.array([0.12, 0.13, 0.15]) * (1.0 - t)[:, None] + np.array([0.30, 0.29, 0.28]) * t[:, None]
        f_rough[m] = 0.44
        f_metal[m] = 0.72
        f_bump[m] = 0.65

    # 14. Fondo rehundido oscuro de rueda (`SEM_WHEEL_BACK`)
    m = face_slot == SEM_WHEEL_BACK
    if np.any(m):
        f_srgb[m] = np.array([0.16, 0.13, 0.10], dtype=np.float32)
        f_rough[m] = 0.90
        f_bump[m] = 0.35

    # Convertir sRGB -> Linear RGBA y expandir por esquina de triángulo (3 loops por cara, cero sangrado)
    f_lin = _srgb_to_linear(f_srgb)
    loop_col = np.ones((n_f * 3, 4), dtype=np.float32)
    loop_col[:, :3] = np.repeat(f_lin, 3, axis=0)

    loop_spec = np.ones((n_f * 3, 4), dtype=np.float32)
    loop_spec[:, 0] = np.repeat(f_rough, 3)
    loop_spec[:, 1] = np.repeat(f_metal, 3)
    loop_spec[:, 2] = np.repeat(f_bump, 3)
    return loop_col, loop_spec


def _build_lighting_and_cameras(sc, res_px: int = 1024):
    import bpy
    import mathutils

    world = bpy.data.worlds.get("StudioPBRWorld") or bpy.data.worlds.new("StudioPBRWorld")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.76, 0.78, 0.80, 1.0)
        bg.inputs["Strength"].default_value = 0.85

    for lo in [o for o in sc.collection.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(lo, do_unlink=True)

    sun_specs = [
        ("KeySun", (75.0, -65.0, 92.0), 2.55, (1.0, 0.97, 0.91), math.radians(12.0)),
        ("FillSun", (-85.0, -35.0, 78.0), 0.92, (0.86, 0.92, 1.0), math.radians(36.0)),
        ("RimSun", (0.0, 95.0, 70.0), 1.05, (0.98, 0.99, 1.0), math.radians(24.0)),
    ]
    for name, loc, en, col, ang in sun_specs:
        ld = bpy.data.lights.new(name, type="SUN")
        ld.energy = en
        ld.color = col
        ld.angle = ang
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = (mathutils.Vector((0.0, 0.0, 29.0)) - mathutils.Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(lo)


def _setup_final_materials(obj, out_dir: Path, pack_images: bool = False):
    import bpy

    plinth_p = out_dir / "plinth_albedo_1k.png"
    if not plinth_p.exists():
        sys.path.insert(0, str(REPO_ROOT / "tools" / "blender"))
        import catan_hero360_bake as cb
        plinth_p = cb.build_plinth_texture(out_dir, res=1024)

    alb_p = out_dir / "baked_albedo.png"
    nrm_p = out_dir / "baked_normal.png"
    orm_p = out_dir / "baked_orm.png"

    poly_mat = np.empty(len(obj.data.polygons), dtype=np.int32)
    obj.data.polygons.foreach_get("material_index", poly_mat)
    obj.data.materials.clear()

    # Material 0: Plinth_Gunmetal (UV plana sobre plinth_albedo_1k.png)
    m_pl = bpy.data.materials.new("Plinth_Gunmetal")
    m_pl.use_nodes = True
    nt_p = m_pl.node_tree
    bsdf_p = next(n for n in nt_p.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf_p.inputs["Roughness"].default_value = 0.44
    if "Metallic" in bsdf_p.inputs:
        bsdf_p.inputs["Metallic"].default_value = 0.22
    t_pl = nt_p.nodes.new("ShaderNodeTexImage")
    im_pl = bpy.data.images.load(str(plinth_p), check_existing=False)
    im_pl.colorspace_settings.name = "sRGB"
    if pack_images:
        im_pl.pack()
    t_pl.image = im_pl
    nt_p.links.new(t_pl.outputs["Color"], bsdf_p.inputs["Base Color"])
    obj.data.materials.append(m_pl)

    # Material 1: Campo_PBR_8K (Albedo 8K + Normal 8K + ORM 4K)
    m_org = bpy.data.materials.new("Campo_PBR_8K")
    m_org.use_nodes = True
    nt = m_org.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")

    t_alb = nt.nodes.new("ShaderNodeTexImage")
    im_alb = bpy.data.images.load(str(alb_p), check_existing=False)
    im_alb.colorspace_settings.name = "sRGB"
    if pack_images:
        im_alb.pack()
    t_alb.image = im_alb
    nt.links.new(t_alb.outputs["Color"], bsdf.inputs["Base Color"])

    t_orm = nt.nodes.new("ShaderNodeTexImage")
    im_orm = bpy.data.images.load(str(orm_p), check_existing=False)
    im_orm.colorspace_settings.name = "Non-Color"
    if pack_images:
        im_orm.pack()
    t_orm.image = im_orm
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(t_orm.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    if "Metallic" in bsdf.inputs:
        nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])

    t_nrm = nt.nodes.new("ShaderNodeTexImage")
    im_nrm = bpy.data.images.load(str(nrm_p), check_existing=False)
    im_nrm.colorspace_settings.name = "Non-Color"
    if pack_images:
        im_nrm.pack()
    t_nrm.image = im_nrm
    nrm_node = nt.nodes.new("ShaderNodeNormalMap")
    nrm_node.inputs["Strength"].default_value = 0.85
    nt.links.new(t_nrm.outputs["Color"], nrm_node.inputs["Color"])
    nt.links.new(nrm_node.outputs["Normal"], bsdf.inputs["Normal"])

    obj.data.materials.append(m_org)
    obj.data.polygons.foreach_set("material_index", poly_mat.tolist())
    obj.data.update()


def _assemble_pbr_mosaic(out_dir: Path) -> None:
    from PIL import Image, ImageDraw
    views_dir = out_dir / "pbr_views"
    tiles = []
    for deg in (0, 60, 120, 180, 240, 300):
        im = Image.open(views_dir / f"pbr_{deg:03d}.png").convert("RGB").resize((640, 640), Image.Resampling.LANCZOS)
        dr = ImageDraw.Draw(im)
        dr.rectangle([10, 10, 130, 38], fill=(24, 28, 34))
        dr.text((18, 16), f"PBR {deg} deg", fill=(240, 244, 250))
        tiles.append(np.asarray(im))
    top_row = np.concatenate(tiles[:3], axis=1)
    bot_row = np.concatenate(tiles[3:], axis=1)
    Image.fromarray(np.concatenate([top_row, bot_row], axis=0)).save(out_dir / "audit_6views_pbr.png")


def run_prep_local(work_dir: Path) -> None:
    import bpy

    sys.path.insert(0, str(REPO_ROOT / "tools" / "blender"))
    import catan_hero360_bake as cb

    t0 = time.time()
    print("[PBR 1/3 PrepLocal] Cargando malla semántica y campos...", flush=True)
    mdata = np.load(str(work_dir / "clay_master_mesh.npz"))
    V = mdata["vertices"].astype(np.float64)
    F = mdata["faces"].astype(np.int32)
    face_slot = mdata["face_slot"].astype(np.int32)

    f_npz = np.load(str(work_dir / "campo_semantic_fields.npz"))
    fields = {k: f_npz[k] for k in f_npz.files}

    loop_col, loop_spec = compute_semantic_corner_attributes(V, F, face_slot, fields)
    print(f"[PBR PrepLocal] Atributos CORNER calculados en {time.time() - t0:.1f}s (tris={len(F):,}).", flush=True)

    plinth_p = cb.build_plinth_texture(work_dir, res=1024)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"

    me = bpy.data.meshes.new("Catan_Campo_PBR_Mesh")
    me.from_pydata(V.tolist(), [], F.tolist())

    is_org = (face_slot >= 1).astype(np.int32)
    me.polygons.foreach_set("material_index", is_org.tolist())
    me.polygons.foreach_set("use_smooth", (face_slot >= 1).tolist())

    # Escribir atributos CORNER `Col` y `MatSpec`
    attr_col = me.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="CORNER")
    attr_col.data.foreach_set("color", loop_col.ravel())
    attr_spec = me.color_attributes.new(name="MatSpec", type="FLOAT_COLOR", domain="CORNER")
    attr_spec.data.foreach_set("color", loop_spec.ravel())

    uv_layer = me.uv_layers.new(name="UVMap")
    uv_layer.active = True
    uv_layer.active_render = True
    me.update()

    obj = bpy.data.objects.new("Catan_Campo", me)
    sc.collection.objects.link(obj)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Material 0: Plinth_Gunmetal (UV XY plana)
    m_pl = bpy.data.materials.new("Plinth_Gunmetal")
    m_pl.use_nodes = True
    nt_p = m_pl.node_tree
    bsdf_p = next(n for n in nt_p.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf_p.inputs["Roughness"].default_value = 0.44
    if "Metallic" in bsdf_p.inputs:
        bsdf_p.inputs["Metallic"].default_value = 0.22
    t_pl = nt_p.nodes.new("ShaderNodeTexImage")
    im_pl = bpy.data.images.load(str(plinth_p), check_existing=False)
    im_pl.colorspace_settings.name = "sRGB"
    im_pl.pack()
    t_pl.image = im_pl
    nt_p.links.new(t_pl.outputs["Color"], bsdf_p.inputs["Base Color"])
    me.materials.append(m_pl)

    # Material 1: Organic_Campo_Bake (combina Col + Cycles AO + micro-ruido 3D para hornear en GPU)
    m_org = bpy.data.materials.new("Organic_Campo_Bake")
    m_org.use_nodes = True
    nt = m_org.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")

    a_col = nt.nodes.new("ShaderNodeAttribute")
    a_col.attribute_name = "Col"
    a_spec = nt.nodes.new("ShaderNodeAttribute")
    a_spec.attribute_name = "MatSpec"
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(a_spec.outputs["Color"], sep.inputs["Color"])

    # AO raytraced real en Cycles + micro-variación pictórica 3D
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = 12
    ao.only_local = False
    ao.inputs["Distance"].default_value = 4.2

    ao_ramp = nt.nodes.new("ShaderNodeMapRange")
    ao_ramp.inputs["From Min"].default_value = 0.15
    ao_ramp.inputs["From Max"].default_value = 1.0
    ao_ramp.inputs["To Min"].default_value = 0.44
    ao_ramp.inputs["To Max"].default_value = 1.05
    nt.links.new(ao.outputs["AO"], ao_ramp.inputs["Value"])

    t_coord = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 4.2
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.62
    nt.links.new(t_coord.outputs["Object"], noise.inputs["Vector"])

    n_map = nt.nodes.new("ShaderNodeMapRange")
    n_map.inputs["From Min"].default_value = 0.2
    n_map.inputs["From Max"].default_value = 0.8
    n_map.inputs["To Min"].default_value = 0.88
    n_map.inputs["To Max"].default_value = 1.12
    nt.links.new(noise.outputs["Fac"], n_map.inputs["Value"])

    mul_ao = nt.nodes.new("ShaderNodeMix")
    mul_ao.data_type = "RGBA"
    mul_ao.blend_type = "MULTIPLY"
    mul_ao.inputs["Factor"].default_value = 1.0
    nt.links.new(a_col.outputs["Color"], mul_ao.inputs[6])
    nt.links.new(ao_ramp.outputs["Result"], mul_ao.inputs[7])

    mul_n = nt.nodes.new("ShaderNodeMix")
    mul_n.data_type = "RGBA"
    mul_n.blend_type = "MULTIPLY"
    mul_n.inputs["Factor"].default_value = 1.0
    nt.links.new(mul_ao.outputs[2], mul_n.inputs[6])
    nt.links.new(n_map.outputs["Result"], mul_n.inputs[7])

    # Nodo Emission auxiliar etiquetado para conmutación directa durante el bake GPU
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.name = "BakeEmit"
    emit.inputs["Strength"].default_value = 1.0

    nt.links.new(mul_n.outputs[2], bsdf.inputs["Base Color"])
    nt.links.new(mul_n.outputs[2], emit.inputs["Color"])
    nt.links.new(sep.outputs["Red"], bsdf.inputs["Roughness"])
    if "Metallic" in bsdf.inputs:
        nt.links.new(sep.outputs["Green"], bsdf.inputs["Metallic"])

    # Micro-bump fino procedural multiplicado por BumpWeight (MatSpec.Blue)
    noise_b = nt.nodes.new("ShaderNodeTexNoise")
    noise_b.inputs["Scale"].default_value = 11.5
    noise_b.inputs["Detail"].default_value = 10.0
    noise_b.inputs["Roughness"].default_value = 0.65
    nt.links.new(t_coord.outputs["Object"], noise_b.inputs["Vector"])

    bump_w = nt.nodes.new("ShaderNodeMath")
    bump_w.operation = "MULTIPLY"
    nt.links.new(sep.outputs["Blue"], bump_w.inputs[0])
    bump_w.inputs[1].default_value = 0.24

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = 0.14
    nt.links.new(bump_w.outputs["Value"], bump.inputs["Strength"])
    nt.links.new(noise_b.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Nodo ORM (R=AO, G=Roughness, B=Metallic) preparado para horneado limpio
    comb_orm = nt.nodes.new("ShaderNodeCombineColor")
    comb_orm.name = "CombineORM"
    nt.links.new(ao.outputs["AO"], comb_orm.inputs["Red"])
    nt.links.new(sep.outputs["Red"], comb_orm.inputs["Green"])
    nt.links.new(sep.outputs["Green"], comb_orm.inputs["Blue"])

    me.materials.append(m_org)

    # Desplegar UV SOLO sobre caras orgánicas (`slot >= 1`) en PC local (0 créditos GPU)
    t_uv = time.time()
    print("[PBR PrepLocal] Desplegando Smart UV Project sobre 100% de caras orgánicas (zócalo aislado)...", flush=True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    me.polygons.foreach_set("select", (face_slot >= 1).tolist())
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.0012, area_weight=1.0, correct_aspect=True)
    bpy.ops.object.mode_set(mode="OBJECT")

    # UV planar XY para zócalo (`slot == 0`)
    n_l = len(me.loops)
    loop_vi = np.empty(n_l, dtype=np.int32)
    me.loops.foreach_get("vertex_index", loop_vi)
    uv_arr = np.empty(n_l * 2, dtype=np.float32)
    me.uv_layers["UVMap"].data.foreach_get("uv", uv_arr)
    uv2 = uv_arr.reshape(n_l, 2)
    l_plinth = np.repeat(face_slot == 0, 3)
    v_pl = V[loop_vi[l_plinth]]
    uv2[l_plinth, 0] = np.clip((v_pl[:, 0] + R) / (2.0 * R), 0.0, 1.0)
    uv2[l_plinth, 1] = np.clip((v_pl[:, 1] + R) / (2.0 * R), 0.0, 1.0)
    me.uv_layers["UVMap"].data.foreach_set("uv", uv2.ravel())
    me.update()
    print(f"[PBR PrepLocal] UVs aisladas listas en {time.time() - t_uv:.1f}s.", flush=True)

    blend_out = work_dir / "campo_gpu_bake.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_out), compress=True)
    print(f"[PBR PrepLocal Done] {blend_out} ({blend_out.stat().st_size / (1024*1024):.1f} MB) en {time.time() - t0:.1f}s.", flush=True)


def run_gpu_bake_only(work_dir: Path, device_pref: str = "gpu", alb_res: int = 8192, orm_res: int = 4096) -> None:
    import bpy
    import mathutils

    sys.path.insert(0, str(REPO_ROOT / "tools" / "blender"))
    import macro_bake_helpers as mbh

    t0 = time.time()
    blend_in = work_dir / "campo_gpu_bake.blend"
    bpy.ops.wm.open_mainfile(filepath=str(blend_in))
    mbh.relink_blend_images([work_dir])

    sc = bpy.context.scene
    dev_used = mbh.configure_cycles_device(sc, device_pref=device_pref)
    if device_pref == "gpu" and not dev_used.startswith("GPU"):
        raise RuntimeError(f"Se exigió GPU OPTIX/CUDA pero se detectó {dev_used}")

    orig_obj = bpy.data.objects["Catan_Campo"]
    # Clonar objeto temporal SOLO con caras orgánicas (material_index == 1) para evitar hornear sobre el zócalo empaquetado
    bake_mesh = orig_obj.data.copy()
    bake_obj = bpy.data.objects.new("Catan_Campo_BakeOnly", bake_mesh)
    sc.collection.objects.link(bake_obj)
    bpy.ops.object.select_all(action="DESELECT")
    bake_obj.select_set(True)
    bpy.context.view_layer.objects.active = bake_obj

    poly_mat = np.empty(len(bake_mesh.polygons), dtype=np.int32)
    bake_mesh.polygons.foreach_get("material_index", poly_mat)
    if np.any(poly_mat == 0):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        bake_mesh.polygons.foreach_set("select", (poly_mat == 0).tolist())
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.delete(type="FACE")
        bpy.ops.object.mode_set(mode="OBJECT")
        bake_mesh.materials.pop(index=0)

    m_org = bake_obj.data.materials[0]
    nt = m_org.node_tree
    out_node = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    emit = nt.nodes["BakeEmit"]
    comb_orm = nt.nodes["CombineORM"]
    mul_n = emit.inputs["Color"].links[0].from_socket

    # Desactivar render sobre caras de zócalo para que smart UV solo hornee orgánicas
    tb = nt.nodes.new("ShaderNodeTexImage")
    nt.nodes.active = tb
    sc.render.bake.margin = 6
    sc.render.bake.use_clear = True

    def _bake_img(name: str, res: int, colorspace: str, bake_type: str, samples: int, out_path: Path):
        t_b = time.time()
        im = bpy.data.images.new(name, width=res, height=res, alpha=False)
        im.colorspace_settings.name = colorspace
        tb.image = im
        sc.cycles.samples = samples
        sc.cycles.use_denoising = False
        if bake_type == "NORMAL":
            bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT")
        else:
            bpy.ops.object.bake(type=bake_type)
        im.filepath_raw = str(out_path)
        im.file_format = "PNG"
        im.save()
        bpy.data.images.remove(im)
        print(f"  [GpuBake] {out_path.name} ({res}x{res}, {bake_type}) en {time.time() - t_b:.1f}s", flush=True)

    # 1. Albedo 8K (EMIT de Col * CyclesAO * MicroNoise -> nítido sin ruido ni sombras cocidas)
    nt.links.new(mul_n, emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])
    _bake_img("BakedAlbedo", alb_res, "sRGB", "EMIT", 6, work_dir / "baked_albedo.png")

    # 2. ORM 4K (R=AO, G=Roughness, B=Metallic)
    nt.links.new(comb_orm.outputs["Color"], emit.inputs["Color"])
    _bake_img("BakedORM", orm_res, "Non-Color", "EMIT", 6, work_dir / "baked_orm.png")

    # 3. Normal tangente 8K (micro-relieve procedural sin doble inclinación macro)
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
    _bake_img("BakedNormal", alb_res, "Non-Color", "NORMAL", 2, work_dir / "baked_normal.png")

    # Limpiar objeto temporal de horneado y renderizar con objeto completo (zócalo + orgánico 8K)
    bpy.data.objects.remove(bake_obj, do_unlink=True)
    bpy.data.meshes.remove(bake_mesh)
    obj = orig_obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Asignar texturas horneadas finales y renderizar 6 vistas PBR + ortho + macro árboles/props en GPU OPTIX
    _setup_final_materials(obj, work_dir, pack_images=False)
    _build_lighting_and_cameras(sc, res_px=1024)

    sc.cycles.samples = 32
    sc.cycles.use_denoising = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.resolution_x = 1024
    sc.render.resolution_y = 1024

    cam_data = bpy.data.cameras.new("PBRCam")
    cam_data.type = "PERSP"
    fov_deg = 15.68
    cam_data.sensor_width = 36.0
    cam_data.lens = 18.0 / math.tan(math.radians(fov_deg * 0.5))
    cam_obj = bpy.data.objects.new("PBRCam", cam_data)
    sc.collection.objects.link(cam_obj)
    sc.camera = cam_obj

    pbr_dir = work_dir / "pbr_views"
    pbr_dir.mkdir(parents=True, exist_ok=True)
    r_horiz = 399.87
    cam_z = 365.78
    target = mathutils.Vector((0.0, 0.0, 27.5))
    for deg in (0, 60, 120, 180, 240, 300):
        az = math.radians(deg)
        cam_obj.location = (-r_horiz * math.sin(az), -r_horiz * math.cos(az), cam_z)
        cam_obj.rotation_euler = (target - cam_obj.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        sc.render.filepath = str(pbr_dir / f"pbr_{deg:03d}.png")
        bpy.ops.render.render(write_still=True)

    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 142.0
    cam_obj.location = (0.0, 0.0, 220.0)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    sc.render.filepath = str(work_dir / "audit_ortho_pbr.png")
    bpy.ops.render.render(write_still=True)

    cam_data.type = "PERSP"
    cam_data.lens = 65.0
    cam_obj.location = (22.0, -20.0, 56.0)
    cam_obj.rotation_euler = (mathutils.Vector((3.5, 16.5, 34.0)) - cam_obj.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = str(work_dir / "pbr_trees_closeup.png")
    bpy.ops.render.render(write_still=True)

    cam_data.lens = 48.0
    cam_obj.location = (0.0, -37.0, 44.0)
    cam_obj.rotation_euler = (mathutils.Vector((5.5, -15.5, 27.2)) - cam_obj.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = str(work_dir / "pbr_props_closeup.png")
    bpy.ops.render.render(write_still=True)

    print(f"[GpuBakeOnly Done] 8K Maps + 9 PBR Renders en {time.time() - t0:.1f}s.", flush=True)


def run_finalize_local(work_dir: Path, glb_name: str = "tile_pbr_v05.glb") -> None:
    import bpy

    t0 = time.time()
    try:
        _assemble_pbr_mosaic(work_dir)
    except ImportError:
        pass  # Blender Python no incluye PIL; se compone con Python del sistema

    bpy.ops.wm.open_mainfile(filepath=str(work_dir / "campo_gpu_bake.blend"))
    obj = bpy.data.objects["Catan_Campo"]
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    for attr in list(obj.data.color_attributes):
        obj.data.color_attributes.remove(attr)

    _setup_final_materials(obj, work_dir, pack_images=True)

    glb_out = work_dir / glb_name
    bpy.ops.export_scene.gltf(
        filepath=str(glb_out),
        export_format="GLB",
        use_selection=True,
        export_materials="EXPORT",
        export_yup=True,
    )
    print(f"[PBR FinalizeLocal Done] {glb_out} ({glb_out.stat().st_size / (1024*1024):.2f} MB) en {time.time() - t0:.1f}s.", flush=True)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description="Horneado PBR 8K Campo (Zero-Idle GPU)")
    p.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    p.add_argument("--stage", choices=["prep-local", "gpu-bake-only", "finalize-local"], required=True)
    p.add_argument("--device", default="gpu")
    p.add_argument("--alb-res", type=int, default=8192)
    p.add_argument("--orm-res", type=int, default=4096)
    p.add_argument("--glb-name", default="tile_pbr_v05.glb")
    args = p.parse_args(argv)

    work_dir = Path(args.work_dir).resolve()
    if args.stage == "prep-local":
        run_prep_local(work_dir)
    elif args.stage == "gpu-bake-only":
        run_gpu_bake_only(work_dir, device_pref=args.device, alb_res=args.alb_res, orm_res=args.orm_res)
    else:
        run_finalize_local(work_dir, glb_name=args.glb_name)


if __name__ == "__main__":
    main()
