"""
Constructor 3D Watertight v2.2 para Bioma Campo (Full Trigo · Campos por Arar · Sin Estructuras).
- Campiña suave y baja (Z_suelo ~26-31 mm sobre Z_BASE=25 mm) con 4 parcelas alineadas exactamente
  a 3 brazos de camino rural con roderas, sin hueco de número ni edificios.
- Surcos e hileras finas de trigo de alta densidad (n_seg=760, arista ~0.083 mm, DEM 2560x2560).
- Árboles de linde proporcionados al cultivo (~9-12x la altura del trigo: frondosos 11-13 mm,
  cipreses 13-15 mm), setos bajos, piedras de linde, vallas, balas, gavillas, rueda y arado.
- Garantía estricta: is_watertight=True, euler_number=2 (genus=0), 0 estructuras/edificios.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import bpy  # type: ignore
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

if not IN_BLENDER:
    import manifold3d as mf
    import scipy.ndimage as ndi
    import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_STL = REPO_ROOT / "Tableros-Catan" / "losetas" / "montana" / "work_brep" / "catan_base_brep.stl"
DEFAULT_OUT_DIR = REPO_ROOT / "Tableros-Catan" / "losetas" / "campo" / "work"

R = 64.0000
APOTHEM = R * math.cos(math.radians(30.0))  # 55.425626 mm
Z_BASE = 25.0000
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hex_distances(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    de = np.full_like(x, 999.0, dtype=np.float64)
    for k in range(6):
        ang = k * math.pi / 3.0
        de = np.minimum(de, APOTHEM - (x * math.cos(ang) + y * math.sin(ang)))
    dv = np.full_like(x, 999.0, dtype=np.float64)
    for k in range(6):
        ang = math.radians(30.0 + 60.0 * k)
        dv = np.minimum(dv, np.hypot(x - R * math.cos(ang), y - R * math.sin(ang)))
    return de, dv


def _fbm(shape: Tuple[int, int], seed: int, sigmas=(42.0, 18.0, 7.0), weights=(0.62, 0.26, 0.12)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros(shape, dtype=np.float64)
    for s, w in zip(sigmas, weights):
        n = rng.normal(0.0, 1.0, size=shape)
        f = ndi.gaussian_filter(n, sigma=s)
        f = (f - f.mean()) / max(float(f.std()), 1e-6)
        acc += w * f
    return acc


def _min_dist_to_polyline(x: np.ndarray, y: np.ndarray, pts: List[Tuple[float, float]]) -> np.ndarray:
    dmin = np.full_like(x, 999.0, dtype=np.float64)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        vx, vy = x1 - x0, y1 - y0
        l2 = vx * vx + vy * vy
        if l2 < 1e-9:
            continue
        t = np.clip(((x - x0) * vx + (y - y0) * vy) / l2, 0.0, 1.0)
        dmin = np.minimum(dmin, np.hypot(x - (x0 + t * vx), y - (y0 + t * vy)))
    return dmin


def build_campo_terrain_fields(res: int = 2560, seed: int = 209) -> Dict[str, np.ndarray]:
    xs = np.linspace(-R, R, res)
    ys = np.linspace(-R, R, res)
    X, Y = np.meshgrid(xs, ys)
    de, dv = _hex_distances(X, Y)

    rim_taper = _smoothstep(ROAD_MARGIN + 0.9, ROAD_MARGIN + 5.0, de) * _smoothstep(
        PAD_RADIUS + 0.9, PAD_RADIUS + 4.2, dv
    )

    fbm_broad = _fbm((res, res), seed=seed, sigmas=(95.0, 38.0, 12.0), weights=(0.64, 0.25, 0.11))
    fbm_fine = _fbm((res, res), seed=seed + 53, sigmas=(18.0, 7.5, 3.2), weights=(0.55, 0.30, 0.15))

    # Campiña suave y baja (max ~5-6 mm de ondulación de suelo sobre Z_BASE=25 mm -> Z_suelo ~26-31 mm)
    hill_nw = 4.2 * np.exp(-(((X + 12.0) / 30.0) ** 2 + ((Y - 14.0) / 28.0) ** 2))
    hill_e = 3.1 * np.exp(-(((X - 20.0) / 26.0) ** 2 + ((Y - 3.0) / 28.0) ** 2))
    hill_sw = 2.0 * np.exp(-(((X + 16.0) / 24.0) ** 2 + ((Y + 20.0) / 23.0) ** 2))
    base_hills = (hill_nw + hill_e + hill_sw + 1.05 + 0.45 * fbm_broad) * rim_taper

    # Polilíneas del camino rural en 3 brazos curvos
    # Polilínea continua S -> NE para definir exactamente qué está al Oeste vs al Este
    main_lane = [
        (-14.0, -45.0), (-9.5, -31.0), (-2.0, -17.0), (2.0, -4.0),
        (5.5, 9.0), (14.5, 22.0), (28.0, 38.0),
    ]
    east_spur = [(2.0, -4.0), (15.0, -6.5), (27.0, -11.0), (42.0, -16.5)]
    west_spur = [(-2.0, -17.0), (-16.0, -12.0), (-30.0, -6.0), (-44.0, 0.0)]

    d_road = np.minimum.reduce([
        _min_dist_to_polyline(X, Y, main_lane),
        _min_dist_to_polyline(X, Y, east_spur),
        _min_dist_to_polyline(X, Y, west_spur),
    ])

    # Claro agrícola junto al cruce (+11.2, -13.8) y claros de raíz bajo todos los árboles de linde
    d_staging = np.hypot(X - 11.2, Y + 13.8)
    staging_mask = 1.0 - _smoothstep(3.8, 5.6, d_staging)
    oak_clear = np.ones_like(X)
    for ox, oy in [(-0.5, 9.2), (24.0, 22.0), (-28.0, -4.5), (28.0, -10.5), (7.5, 25.5), (11.5, 27.5), (-16.5, -28.0)]:
        oak_clear = np.minimum(oak_clear, _smoothstep(2.2, 3.8, np.hypot(X - ox, Y - oy)))

    # Fronteras EXACTAS alineadas con las polilíneas (interpolación por Y en main_lane y por X en ramales)
    lane_pts_y = np.array([p[1] for p in main_lane])
    lane_pts_x = np.array([p[0] for p in main_lane])
    x_main_at_y = np.interp(Y, lane_pts_y, lane_pts_x)

    west_pts_x = np.array([p[0] for p in reversed(west_spur)])
    west_pts_y = np.array([p[1] for p in reversed(west_spur)])
    y_west_at_x = np.interp(X, west_pts_x, west_pts_y)

    east_pts_x = np.array([p[0] for p in east_spur])
    east_pts_y = np.array([p[1] for p in east_spur])
    y_east_at_x = np.interp(X, east_pts_x, east_pts_y)

    is_east = X > x_main_at_y
    parcel_id = np.zeros((res, res), dtype=np.int32)
    parcel_id[~is_east & (Y >= y_west_at_x)] = 1   # Loma NW grande
    parcel_id[~is_east & (Y < y_west_at_x)] = 2    # Parcela SW
    parcel_id[is_east & (Y >= y_east_at_x)] = 3    # Loma NE/E
    parcel_id[is_east & (Y < y_east_at_x)] = 4     # Parcela SE

    # Escalonado suave entre parcelas
    parcel_offsets = {1: 0.65, 2: 0.15, 3: 0.45, 4: 0.0}
    terrace_bump = np.zeros_like(base_hills)
    for pid, off in parcel_offsets.items():
        terrace_bump += off * (parcel_id == pid).astype(np.float64)
    terrace_bump = ndi.gaussian_filter(terrace_bump, sigma=12.0) * rim_taper

    road_half_w = 1.15
    road_trench = (1.0 - _smoothstep(road_half_w * 0.65, road_half_w * 1.55, d_road)) * rim_taper
    ruts = np.exp(-((d_road - 0.42) / 0.20) ** 2) * 0.15 * road_trench

    border_buf = _smoothstep(road_half_w * 1.15, road_half_w * 2.15, d_road)
    wheat_mask = (
        _smoothstep(ROAD_MARGIN + 1.9, ROAD_MARGIN + 4.3, de)
        * _smoothstep(PAD_RADIUS + 1.9, PAD_RADIUS + 4.1, dv)
        * border_buf
        * (1.0 - staging_mask)
        * oak_clear
    )

    parcel_angles_deg = {1: 26.0, 2: -32.0, 3: -58.0, 4: 64.0}
    row_pitch_mm = 0.94
    grain_pitch_mm = 0.62
    wheat_sculpt = np.zeros_like(base_hills)

    for pid, ang_deg in parcel_angles_deg.items():
        m = (parcel_id == pid).astype(np.float64)
        if not np.any(m):
            continue
        th = math.radians(ang_deg)
        ct, st = math.cos(th), math.sin(th)
        u = X * ct + Y * st + 2.1 * np.sin((X * (-st) + Y * ct) / 16.0)
        v = -X * st + Y * ct

        phase_u = 2.0 * math.pi * u / row_pitch_mm
        row_ridge = 0.5 + 0.5 * np.cos(phase_u - 0.35 * np.sin(phase_u))
        row_ridge = np.power(np.clip(row_ridge, 0.0, 1.0), 0.85)

        phase_v = 2.0 * math.pi * v / grain_pitch_mm + 1.8 * np.sin(u * 2.9)
        grain_beads = 0.5 + 0.5 * np.cos(phase_v)

        wind_swell = 0.84 + 0.16 * np.sin(u / 6.5 + v / 9.0)
        wheat_sculpt += m * wind_swell * (0.46 + 0.58 * row_ridge + 0.36 * row_ridge * grain_beads)

    wheat_sculpt = ndi.gaussian_filter(wheat_sculpt, sigma=0.45)
    soil_z = base_hills + terrace_bump + 0.22 * fbm_fine * rim_taper - 0.45 * road_trench - ruts - 0.25 * staging_mask * rim_taper
    dem = np.maximum(0.0, soil_z + wheat_mask * wheat_sculpt) * rim_taper

    return {
        "X": X,
        "Y": Y,
        "dem": dem,
        "soil_z": np.maximum(0.0, soil_z) * rim_taper,
        "wheat_mask": wheat_mask,
        "road_mask": road_trench,
        "staging_mask": staging_mask,
        "parcel_id": parcel_id,
        "d_road": d_road,
        "de": de,
        "dv": dv,
    }


def _bilinear(arr: np.ndarray, x: float, y: float) -> float:
    h, w = arr.shape[:2]
    fx = max(0.0, min(1.0, (x + R) / (2.0 * R))) * (w - 1)
    fy = max(0.0, min(1.0, (y + R) / (2.0 * R))) * (h - 1)
    x0 = int(math.floor(fx))
    y0 = int(math.floor(fy))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    tx, ty = fx - x0, fy - y0
    return (
        float(arr[y0, x0]) * (1.0 - tx) * (1.0 - ty)
        + float(arr[y0, x1]) * tx * (1.0 - ty)
        + float(arr[y1, x0]) * (1.0 - tx) * ty
        + float(arr[y1, x1]) * tx * ty
    )


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


def build_terrain_cap_manifold(dem: np.ndarray, n_seg: int = 440, z_bot: float = 20.0, z_flat_hidden: float = 24.45):
    r_cap = R - 0.65
    verts_hex = [
        (r_cap * math.cos(math.radians(30.0 + 60.0 * k)), r_cap * math.sin(math.radians(30.0 + 60.0 * k)))
        for k in range(6)
    ]

    verts_xy: List[Tuple[float, float]] = []
    vert_map: Dict[Tuple[int, int], int] = {}
    faces: List[Tuple[int, int, int]] = []
    sector_idx: List[Dict[Tuple[int, int], int]] = []

    def get_v(x: float, y: float) -> int:
        key = (int(round(x * 1000.0)), int(round(y * 1000.0)))
        idx = vert_map.get(key)
        if idx is None:
            idx = len(verts_xy)
            verts_xy.append((x, y))
            vert_map[key] = idx
        return idx

    for k in range(6):
        v1, v2 = verts_hex[k], verts_hex[(k + 1) % 6]
        smap: Dict[Tuple[int, int], int] = {}
        for i in range(n_seg + 1):
            u = i / n_seg
            for j in range(n_seg + 1 - i):
                w = j / n_seg
                smap[(i, j)] = get_v(u * v1[0] + w * v2[0], u * v1[1] + w * v2[1])
        sector_idx.append(smap)
        for i in range(n_seg):
            for j in range(n_seg - i):
                a, b, c = smap[(i, j)], smap[(i + 1, j)], smap[(i, j + 1)]
                faces.append((a, b, c))
                if i + j + 1 < n_seg:
                    d = smap[(i + 1, j + 1)]
                    faces.append((b, d, c))

    xy_arr = np.array(verts_xy, dtype=np.float64)
    z_arr = z_flat_hidden + np.maximum(0.0, _bilinear_arr(dem, xy_arr[:, 0], xy_arr[:, 1]))
    verts_3d = np.column_stack([xy_arr[:, 0], xy_arr[:, 1], z_arr]).tolist()

    rim_top: List[int] = []
    for k in range(6):
        smap = sector_idx[k]
        for j in range(n_seg):
            rim_top.append(smap[(n_seg - j, j)])

    rim_bot: List[int] = []
    for vi in rim_top:
        x, y, _ = verts_3d[vi]
        rim_bot.append(len(verts_3d))
        verts_3d.append([x, y, z_bot])

    bot_center = len(verts_3d)
    verts_3d.append([0.0, 0.0, z_bot])

    m_rim = len(rim_top)
    for idx in range(m_rim):
        nxt = (idx + 1) % m_rim
        faces.append((rim_bot[idx], rim_bot[nxt], rim_top[nxt]))
        faces.append((rim_bot[idx], rim_top[nxt], rim_top[idx]))
        faces.append((bot_center, rim_bot[nxt], rim_bot[idx]))

    mesh = mf.Mesh(
        vert_properties=np.array(verts_3d, dtype=np.float32),
        tri_verts=np.array(faces, dtype=np.uint32),
    )
    return mf.Manifold(mesh)


def _batch_compose_manifolds(parts: List, chunk: int = 24):
    if not parts:
        return mf.Manifold()
    cur = parts
    while len(cur) > 1:
        nxt: List = []
        for i in range(0, len(cur), chunk):
            grp = cur[i : i + chunk]
            nxt.append(grp[0] if len(grp) == 1 else mf.Manifold.batch_boolean(grp, mf.OpType.Add))
        cur = nxt
    return cur[0]


_ICO_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _get_icosphere(subdiv: int) -> Tuple[np.ndarray, np.ndarray]:
    """Genera y cachea una icosaedro subdividido unitario isotrópico (CCW hacia afuera)."""
    if subdiv in _ICO_CACHE:
        return _ICO_CACHE[subdiv]

    t = (1.0 + math.sqrt(5.0)) * 0.5
    verts: List[List[float]] = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    verts_np = np.asarray(verts, dtype=np.float64)
    verts_np /= np.linalg.norm(verts_np, axis=1, keepdims=True)
    verts_list = verts_np.tolist()

    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    for _ in range(subdiv):
        mid_cache: Dict[Tuple[int, int], int] = {}

        def get_mid(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            idx = mid_cache.get(key)
            if idx is not None:
                return idx
            va, vb = verts_list[a], verts_list[b]
            vm = [0.5 * (va[0] + vb[0]), 0.5 * (va[1] + vb[1]), 0.5 * (va[2] + vb[2])]
            nrm = math.sqrt(vm[0] * vm[0] + vm[1] * vm[1] + vm[2] * vm[2]) + 1e-12
            idx = len(verts_list)
            verts_list.append([vm[0] / nrm, vm[1] / nrm, vm[2] / nrm])
            mid_cache[key] = idx
            return idx

        new_faces: List[Tuple[int, int, int]] = []
        for a, b, c in faces:
            ab = get_mid(a, b)
            bc = get_mid(b, c)
            ca = get_mid(c, a)
            new_faces.append((a, ab, ca))
            new_faces.append((b, bc, ab))
            new_faces.append((c, ca, bc))
            new_faces.append((ab, bc, ca))
        faces = new_faces

    v_arr = np.asarray(verts_list, dtype=np.float64)
    f_arr = np.asarray(faces, dtype=np.uint32)
    _ICO_CACHE[subdiv] = (v_arr, f_arr)
    return v_arr, f_arr


def _fibonacci_sphere_jittered(n_pts: int, rng: np.random.Generator, jitter: float = 0.25) -> np.ndarray:
    idx = np.arange(n_pts, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n_pts)
    theta = math.pi * (1.0 + math.sqrt(5.0)) * idx
    pts = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])
    pts += rng.normal(0.0, jitter / math.sqrt(max(n_pts, 1)), size=pts.shape)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
    return pts


def _botanical_foliage_lobe(
    radius: float,
    scale_xyz: Tuple[float, float, float],
    rng: np.random.Generator,
    subdiv: int = 4,
    n_tufts: int = 120,
    tuft_amp: float = 0.25,
    micro_amp: float = 0.11,
    fluting_z: float = 0.0,
):
    """Lóbulo de follaje esculpido de alta resolución con racimos Voronoi en grieta V nítida."""
    u, f_arr = _get_icosphere(subdiv)

    # Celdas Voronoi reales (domos de racimo con hendidura marcada en cada frontera de celda)
    c1 = _fibonacci_sphere_jittered(n_tufts, rng, jitter=1.15)
    d1 = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - (u @ c1.T)))).min(axis=1)
    r_cell1 = 1.15 * math.sqrt(4.0 / max(n_tufts, 1))
    cushion1 = np.power(np.clip(1.0 - (d1 / r_cell1) ** 1.35, 0.0, 1.0), 0.85)

    # Segundo orden de micro-hojas con grietas nítidas
    n_micro = max(60, int(round(n_tufts * 3.2)))
    c2 = _fibonacci_sphere_jittered(n_micro, rng, jitter=1.25)
    d2 = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - (u @ c2.T)))).min(axis=1)
    r_cell2 = 1.15 * math.sqrt(4.0 / max(n_micro, 1))
    cushion2 = np.power(np.clip(1.0 - (d2 / r_cell2) ** 1.35, 0.0, 1.0), 0.90)

    c1_n = cushion1 - float(np.mean(cushion1))
    c2_n = cushion2 - float(np.mean(cushion2))
    bot_atten = 0.35 + 0.65 * np.clip((u[:, 2] + 0.55) / 0.88, 0.0, 1.0)

    radial_disp = bot_atten * (tuft_amp * c1_n + micro_amp * c2_n)
    if fluting_z > 0.0:
        ang = np.arctan2(u[:, 1], u[:, 0])
        p_fl = float(rng.uniform(0, 6.28))
        plumes = np.power(0.5 + 0.5 * np.cos(14.0 * ang + 2.6 * u[:, 2] + p_fl), 0.70) - 0.5
        radial_disp += bot_atten * fluting_z * plumes * (1.0 - 0.45 * u[:, 2] ** 2)

    r_scale = np.clip(1.0 + radial_disp, 0.70, 1.36)
    vp = u * (radius * r_scale[:, None])
    vp[:, 0] *= scale_xyz[0]
    vp[:, 1] *= scale_xyz[1]
    vp[:, 2] *= scale_xyz[2]

    mesh = mf.Mesh(
        vert_properties=np.ascontiguousarray(vp, dtype=np.float32),
        tri_verts=np.ascontiguousarray(f_arr, dtype=np.uint32),
    )
    return mf.Manifold(mesh)


def _make_fluted_trunk(
    height: float,
    r_bot: float,
    r_mid: float,
    r_top: float,
    z_start: float,
    rng: np.random.Generator,
    rad_segs: int = 44,
    z_segs: int = 24,
    root_flares: int = 5,
    twist: float = 0.75,
):
    """Tronco orgánico estanco de alta resolución con raíces acanaladas, curvatura y corteza vertical."""
    thetas = np.linspace(0.0, 2.0 * math.pi, rad_segs, endpoint=False)
    ts = np.linspace(0.0, 1.0, z_segs + 1)

    p1 = float(rng.uniform(0.0, 6.28))
    p2 = float(rng.uniform(0.0, 6.28))
    lean_ang = float(rng.uniform(0.0, 6.28))
    lean_amp = r_bot * float(rng.uniform(0.18, 0.42))

    verts: List[List[float]] = []
    for t in ts:
        # Perfil suave r_bot -> r_mid -> r_top con ensanche radicular en t < 0.35
        r_base = ((1.0 - t) ** 2) * r_bot + 2.0 * (1.0 - t) * t * r_mid + (t ** 2) * r_top
        flare = 1.0 + 0.48 * np.exp(-((t / 0.18) ** 2))
        cx = lean_amp * math.sin(math.pi * t) * math.cos(lean_ang)
        cy = lean_amp * math.sin(math.pi * t) * math.sin(lean_ang)
        z = z_start + t * height

        for th in thetas:
            root_mod = 1.0 + 0.24 * math.exp(-((t / 0.22) ** 2)) * (0.5 + 0.5 * math.cos(root_flares * th + p1))
            bark = (
                0.055 * math.cos(14.0 * th + twist * 6.0 * t + p2)
                + 0.032 * math.sin(23.0 * th - twist * 9.0 * t + p1)
            )
            rr = r_base * flare * root_mod * (1.0 + bark)
            verts.append([cx + rr * math.cos(th), cy + rr * math.sin(th), z])

    bot_c = len(verts)
    verts.append([0.0, 0.0, z_start])
    top_cx = lean_amp * math.sin(math.pi) * math.cos(lean_ang)
    top_cy = lean_amp * math.sin(math.pi) * math.sin(lean_ang)
    top_c = len(verts)
    verts.append([top_cx, top_cy, z_start + height])

    faces: List[Tuple[int, int, int]] = []
    for iz in range(z_segs):
        row0 = iz * rad_segs
        row1 = (iz + 1) * rad_segs
        for j in range(rad_segs):
            jn = (j + 1) % rad_segs
            a, b = row0 + j, row0 + jn
            c, d = row1 + j, row1 + jn
            faces.append((a, b, d))
            faces.append((a, d, c))

    top_row = z_segs * rad_segs
    for j in range(rad_segs):
        jn = (j + 1) % rad_segs
        faces.append((bot_c, jn, j))
        faces.append((top_c, top_row + j, top_row + jn))

    mesh = mf.Mesh(
        vert_properties=np.asarray(verts, dtype=np.float32),
        tri_verts=np.asarray(faces, dtype=np.uint32),
    )
    return mf.Manifold(mesh)


def _make_bush_cluster(x: float, y: float, z: float, radius: float, rng: np.random.Generator):
    """Arbusto en cojín alta resolución (icosphere subdiv=3 con racimos foliares Voronoi, chi=2)."""
    core = _botanical_foliage_lobe(
        radius, (1.12, 1.06, 0.90), rng, subdiv=3, n_tufts=44, tuft_amp=0.20, micro_amp=0.07
    ).translate([x, y, z + radius * 0.16])
    anchor = mf.Manifold.cylinder(radius * 2.0, radius * 1.10, radius * 0.78, circular_segments=16).translate([x, y, z - radius * 1.20])
    return core + anchor


def _make_oak_hedge_tree(x: float, y: float, z: float, height: float, rad: float, rng: np.random.Generator):
    """Roble/encina de linde de alta resolución (~110k Δ): tronco acanalado, 5 ramas y 13 sub-copas Voronoi."""
    trunk = _make_fluted_trunk(
        height=height * 0.58 + 2.6,
        r_bot=rad * 0.36,
        r_mid=rad * 0.22,
        r_top=rad * 0.15,
        z_start=z - 2.6,
        rng=rng,
        rad_segs=44,
        z_segs=26,
        root_flares=5,
    ).translate([x, y, 0.0])

    # 5 ramas cortas visibles entrando limpiamente en el faldón inferior sin formar bucles (genus=0)
    parts = [trunk]
    base_ang = float(rng.uniform(0.0, 6.28))
    limb_tips: List[Tuple[float, float, float]] = []
    for k in range(5):
        az = base_ang + k * (2.0 * math.pi / 5.0) + float(rng.uniform(-0.16, 0.16))
        tilt_deg = float(rng.uniform(22.0, 29.0))
        b_len = height * 0.24
        limb = mf.Manifold.cylinder(b_len, rad * 0.16, rad * 0.11, circular_segments=18)
        limb = limb.rotate([0.0, tilt_deg, 0.0]).rotate([0.0, 0.0, math.degrees(az)])
        parts.append(limb.translate([x, y, z + height * 0.28]))
        horiz = (height * 0.22) * math.sin(math.radians(tilt_deg))
        limb_tips.append((
            x + (horiz + 0.18 * rad) * math.cos(az),
            y + (horiz + 0.18 * rad) * math.sin(az),
            z + height * 0.52,
        ))

    # Collar inferior >= 45 deg bajo el centro de la copa para impresión limpia sin soportes
    collar = mf.Manifold.cylinder(height * 0.20, rad * 0.21, rad * 0.64, circular_segments=28).translate([x, y, z + height * 0.30])
    parts.append(collar)

    # Copa sólida conectada en estrella desde núcleo central (solape >= 35%, 0 huecos ni bucles)
    boughs_specs: List[Tuple[float, float, float, float, Tuple[float, float, float], int, int]] = [
        (x, y, z + height * 0.56, rad * 0.72, (1.03, 1.00, 0.90), 5, 110),
        (x, y, z + height * 0.72, rad * 0.60, (0.98, 0.98, 0.96), 5, 100),
    ]
    for k in range(5):
        tx, ty, tz = limb_tips[k]
        rr = rad * float(rng.uniform(0.46, 0.54))
        boughs_specs.append((tx, ty, tz, rr, (1.04, 0.96, 0.90), 5, 115))
        az_s = base_ang + k * (2.0 * math.pi / 5.0) + 0.34
        boughs_specs.append((
            x + rad * 0.34 * math.cos(az_s),
            y + rad * 0.34 * math.sin(az_s),
            z + height * float(rng.uniform(0.68, 0.76)),
            rad * float(rng.uniform(0.42, 0.48)),
            (0.96, 0.98, 0.95),
            4,
            90,
        ))
    boughs_specs.append((x + 0.10 * rad, y - 0.08 * rad, z + height * 0.83, rad * 0.46, (0.94, 0.94, 1.02), 5, 100))
    boughs_specs.append((x - 0.06 * rad, y + 0.08 * rad, z + height * 0.90, rad * 0.35, (0.92, 0.92, 1.06), 4, 80))

    for bx, by, bz, rr, sc_xyz, sdv, n_tf in boughs_specs:
        lobe = _botanical_foliage_lobe(
            rr, sc_xyz, rng, subdiv=sdv, n_tufts=n_tf, tuft_amp=0.26, micro_amp=0.115
        ).translate([bx, by, bz])
        parts.append(lobe)
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def _make_cypress_tree(x: float, y: float, z: float, height: float, rad: float, rng: np.random.Generator):
    """Ciprés mediterráneo de alta resolución (~90k Δ): tronco estriado y 10 penachos foliares en flama."""
    trunk = _make_fluted_trunk(
        height=height * 0.55 + 2.5,
        r_bot=rad * 0.42,
        r_mid=rad * 0.28,
        r_top=rad * 0.18,
        z_start=z - 2.5,
        rng=rng,
        rad_segs=36,
        z_segs=20,
        root_flares=4,
    ).translate([x, y, 0.0])
    collar = mf.Manifold.cylinder(height * 0.16, rad * 0.28, rad * 0.78, circular_segments=20).translate([x, y, z + height * 0.15])
    parts = [trunk, collar]
    tiers = [
        (0.06, -0.04, height * 0.26, rad * 0.96, 1.38, 5, 110),
        (-0.07, 0.05, height * 0.34, rad * 0.92, 1.44, 5, 105),
        (0.05, 0.06, height * 0.43, rad * 0.86, 1.50, 5, 100),
        (-0.05, -0.04, height * 0.52, rad * 0.78, 1.56, 5, 95),
        (0.04, -0.05, height * 0.61, rad * 0.68, 1.60, 4, 85),
        (-0.04, 0.04, height * 0.70, rad * 0.58, 1.66, 4, 78),
        (0.02, 0.03, height * 0.78, rad * 0.48, 1.72, 4, 70),
        (-0.02, -0.02, height * 0.85, rad * 0.36, 1.80, 4, 60),
        (0.0, 0.0, height * 0.91, rad * 0.25, 1.88, 4, 48),
        (0.0, 0.0, height * 0.96, rad * 0.15, 1.95, 4, 34),
    ]
    for dx, dy, dz, rr, sz, sdv, n_tf in tiers:
        lobe = _botanical_foliage_lobe(
            rr, (1.0, 1.0, sz), rng, subdiv=sdv, n_tufts=n_tf, tuft_amp=0.25, micro_amp=0.105, fluting_z=0.14
        ).translate([x + dx * rad, y + dy * rad, z + dz])
        parts.append(lobe)
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def _make_stone_boulder(x: float, y: float, z: float, r: float, rng: np.random.Generator):
    u, f_arr = _get_icosphere(2)
    c_planes = _fibonacci_sphere_jittered(12, rng, jitter=1.4)
    # Aplanado poligonal tipo piedra partida natural
    proj = np.max(u @ c_planes.T, axis=1)
    r_fac = np.clip(1.0 / np.maximum(proj, 0.65), 0.72, 1.15)
    vp = u * (r * 0.88 * r_fac[:, None])
    vp[:, 0] *= float(rng.uniform(0.88, 1.32))
    vp[:, 1] *= float(rng.uniform(0.78, 1.22))
    vp[:, 2] *= float(rng.uniform(0.55, 0.82))
    yaw = float(rng.uniform(0.0, 2.0 * math.pi))
    cy, sy = math.cos(yaw), math.sin(yaw)
    xr = vp[:, 0] * cy - vp[:, 1] * sy + x
    yr = vp[:, 0] * sy + vp[:, 1] * cy + y
    zr = vp[:, 2] + z
    mesh2 = mf.Mesh(
        vert_properties=np.ascontiguousarray(np.column_stack([xr, yr, zr]), dtype=np.float32),
        tri_verts=np.ascontiguousarray(f_arr, dtype=np.uint32),
    )
    return mf.Manifold(mesh2)


def _make_solid_fence_segment(p0: Tuple[float, float], p1: Tuple[float, float], dem: np.ndarray):
    """Valla rústica maciza con zócalo de tierra/madera hasta -2.0 mm bajo el suelo (chi=2 estricto)."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    ang_deg = math.degrees(math.atan2(dy, dx))
    parts: List = []

    n_posts = max(2, int(round(L / 4.2)) + 1)
    zs = []
    for k in range(n_posts):
        t = k / (n_posts - 1)
        px, py = x0 + t * dx, y0 + t * dy
        pz = Z_BASE - 0.55 + _bilinear(dem, px, py)
        zs.append(pz)
        post = mf.Manifold.cylinder(height=5.0, radius_low=0.65, radius_high=0.55, circular_segments=10)
        parts.append(post.translate([px, py, pz - 2.0]))

    for k in range(n_posts - 1):
        t0, t1 = k / (n_posts - 1), (k + 1) / (n_posts - 1)
        mx = x0 + 0.5 * (t0 + t1) * dx
        my = y0 + 0.5 * (t0 + t1) * dy
        mz = 0.5 * (zs[k] + zs[k + 1])
        seg_len = L / (n_posts - 1) + 0.35
        dz = zs[k + 1] - zs[k]
        pitch_deg = -math.degrees(math.atan2(dz, L / (n_posts - 1)))

        # Rotación explícita pitch primero (Y) y yaw después (Z) + muro enterrado -2.5 mm (chi=2 estricto)
        wall = mf.Manifold.cube([seg_len, 0.65, 5.8], center=True).rotate([0.0, pitch_deg, 0.0]).rotate([0.0, 0.0, ang_deg]).translate([mx, my, mz - 0.15])
        parts.append(wall)
        for h_rail in (1.25, 2.25):
            rail = mf.Manifold.cube([seg_len, 1.05, 0.70], center=True).rotate([0.0, pitch_deg, 0.0]).rotate([0.0, 0.0, ang_deg]).translate([mx, my, mz + h_rail])
            parts.append(rail)

    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def _make_straw_bale(x: float, y: float, z: float, yaw_deg: float, radius: float = 1.60, length: float = 3.5):
    body = mf.Manifold.cylinder(length, radius, radius, circular_segments=22, center=True)
    ring1 = mf.Manifold.cylinder(0.28, radius * 1.03, radius * 1.03, circular_segments=22, center=True).translate([0, 0, -length * 0.25])
    ring2 = mf.Manifold.cylinder(0.28, radius * 1.03, radius * 1.03, circular_segments=22, center=True).translate([0, 0, length * 0.25])
    return (body + ring1 + ring2).rotate([0.0, 90.0, 0.0]).rotate([0.0, 0.0, yaw_deg]).translate([x, y, z + radius * 0.58])


def _make_wheat_sheaf(x: float, y: float, z: float, yaw_deg: float, scale: float = 1.0):
    bot = mf.Manifold.cylinder(2.2 * scale, 1.10 * scale, 0.54 * scale, circular_segments=14).translate([0, 0, -0.85 * scale])
    tie = mf.Manifold.cylinder(0.42 * scale, 0.64 * scale, 0.64 * scale, circular_segments=14).translate([0, 0, 1.25 * scale])
    top = mf.Manifold.cylinder(2.0 * scale, 0.54 * scale, 1.15 * scale, circular_segments=14).translate([0, 0, 1.45 * scale])
    cap = mf.Manifold.sphere(1.08 * scale, 14).scale([1.0, 1.0, 0.42]).translate([0, 0, 3.35 * scale])
    return mf.Manifold.batch_boolean([bot, tie, top, cap], mf.OpType.Add).rotate([3.0, -3.0, 0.0]).rotate([0.0, 0.0, yaw_deg]).translate([x, y, z])


def _make_solid_cartwheel(x: float, y: float, z: float, yaw_deg: float, radius: float = 3.3):
    plate = mf.Manifold.cylinder(0.75, radius - 0.15, radius - 0.15, circular_segments=24, center=True)
    outer = mf.Manifold.cylinder(1.05, radius, radius, circular_segments=24, center=True)
    inner = mf.Manifold.cylinder(1.25, radius - 0.85, radius - 0.85, circular_segments=24, center=True)
    rim = outer - inner
    hub = mf.Manifold.cylinder(1.35, 0.85, 0.85, circular_segments=14, center=True)
    parts = [plate, rim, hub]
    for k in range(8):
        spoke = mf.Manifold.cube([(radius - 0.35) * 2.0, 0.52, 0.88], center=True).rotate([0, 0, k * 45.0])
        parts.append(spoke)
    wheel = mf.Manifold.batch_boolean(parts, mf.OpType.Add).rotate([78.0, 0.0, 0.0]).rotate([0.0, 0.0, yaw_deg]).translate([x, y, z + radius * 0.55])
    plinth_wedge = mf.Manifold.cylinder(2.6, radius * 0.72, radius * 0.55, circular_segments=16).translate([x, y, z - 1.5])
    return wheel + plinth_wedge


def _make_hand_plow(x: float, y: float, z: float, yaw_deg: float):
    """Arado manual con cuña continua sólida y rotaciones ordenadas (chi=2 estricto)."""
    parts = [
        mf.Manifold.cube([6.4, 0.90, 0.95], center=True).rotate([0, -10, 0]).translate([0.0, 0.0, 1.35]),
        mf.Manifold.cube([3.2, 1.05, 3.2], center=True).rotate([0, 20, 0]).translate([-1.4, 0.0, 0.20]),
        mf.Manifold.cube([3.4, 1.55, 2.5], center=True).translate([1.4, 0.0, 0.75]),
        mf.Manifold.cube([4.2, 0.85, 0.85], center=True).rotate([0, 24, 0]).rotate([0, 0, 10]).translate([2.6, 0.62, 2.30]),
        mf.Manifold.cube([4.2, 0.85, 0.85], center=True).rotate([0, 24, 0]).rotate([0, 0, -10]).translate([2.6, -0.62, 2.30]),
    ]
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add).rotate([0.0, 0.0, yaw_deg]).translate([x, y, z - 0.45])


def build_agricultural_props_manifold(fields: Dict[str, np.ndarray], seed: int = 707):
    rng = np.random.default_rng(seed)
    dem = fields["dem"]
    d_road = fields["d_road"]
    de = fields["de"]
    dv = fields["dv"]

    props: List = []

    # 1. Árboles de linde ÚNICAMENTE junto a bordes de caminos o esquinas de parcela (~9-12x altura del trigo)
    oak_spots = [
        (-0.5, 9.2, 12.6, 3.15),     # Árbol protagonista junto al camino norte
        (24.0, 22.0, 11.4, 2.95),    # Salida NE junto al camino
        (-28.0, -4.5, 11.2, 2.90),   # Borde W junto al ramal
        (28.0, -10.5, 10.6, 2.80),   # Límite E junto al ramal
    ]
    for tx, ty, th, tr in oak_spots:
        tz = Z_BASE - 0.55 + _bilinear(dem, tx, ty)
        props.append(_make_oak_hedge_tree(tx, ty, tz, th, tr, rng))

    cypress_spots = [
        (7.5, 25.5, 14.8, 1.95),     # Grupo de 2 cipreses esbeltos en linde norte del camino
        (11.5, 27.5, 12.8, 1.75),
        (-16.5, -28.0, 13.6, 1.85),  # Ciprés junto a la entrada sur del camino
    ]
    for tx, ty, th, tr in cypress_spots:
        tz = Z_BASE - 0.55 + _bilinear(dem, tx, ty)
        props.append(_make_cypress_tree(tx, ty, tz, th, tr, rng))

    # 2. Setos orgánicos bajos agrupados en los márgenes del camino y en 3 rincones de perímetro
    xs = np.linspace(-APOTHEM + 5.5, APOTHEM - 5.5, 56)
    ys = np.linspace(-R + 6.5, R - 6.5, 60)
    bush_centers: List[Tuple[float, float]] = []
    # Focos naturales de seto en bordes exteriores (en vez de bolitas por todo el perímetro)
    rim_bush_foci = [(-36.0, 18.0), (0.0, 44.0), (37.0, -4.0), (-26.0, -34.0)]
    for gy in ys:
        for gx in xs:
            x = float(gx + rng.uniform(-0.35, 0.35))
            y = float(gy + rng.uniform(-0.35, 0.35))
            edge_d = _bilinear(de, x, y)
            vert_d = _bilinear(dv, x, y)
            if edge_d < ROAD_MARGIN + 2.5 or vert_d < PAD_RADIUS + 2.5:
                continue
            rd = _bilinear(d_road, x, y)
            near_lane = 1.65 <= rd <= 2.95
            near_rim_pocket = (
                ROAD_MARGIN + 2.5 <= edge_d <= ROAD_MARGIN + 4.8
                and any(math.hypot(x - fx, y - fy) < 11.5 for fx, fy in rim_bush_foci)
            )
            if not (near_lane or near_rim_pocket):
                continue
            if math.hypot(x - 11.2, y + 13.8) < 5.2:
                continue
            if any(math.hypot(x - tx, y - ty) < 4.2 for tx, ty, _, _ in oak_spots + cypress_spots):
                continue
            if any(math.hypot(x - bx, y - by) < 3.4 for bx, by in bush_centers):
                continue
            if rng.random() > (0.50 if near_lane else 0.62):
                continue
            z = Z_BASE - 0.55 + _bilinear(dem, x, y)
            rad = float(rng.uniform(0.75, 1.22))
            props.append(_make_bush_cluster(x, y, z - 0.15, rad, rng))
            bush_centers.append((x, y))
            if len(bush_centers) >= 48:
                break
        if len(bush_centers) >= 48:
            break

    # 3. Piedras bajas agrupadas en los taludes del camino (sin collar perimetral artificial)
    stone_centers: List[Tuple[float, float]] = []
    for _ in range(1400):
        x = float(rng.uniform(-APOTHEM + 6.0, APOTHEM - 6.0))
        y = float(rng.uniform(-R + 7.0, R - 7.0))
        edge_d = _bilinear(de, x, y)
        vert_d = _bilinear(dv, x, y)
        if edge_d < ROAD_MARGIN + 2.5 or vert_d < PAD_RADIUS + 2.5:
            continue
        rd = _bilinear(d_road, x, y)
        if not (1.45 <= rd <= 2.55):
            continue
        if any(math.hypot(x - sx, y - sy) < 2.6 for sx, sy in stone_centers):
            continue
        if any(math.hypot(x - bx, y - by) < 2.1 for bx, by in bush_centers):
            continue
        z = Z_BASE - 0.55 + _bilinear(dem, x, y)
        sr = float(rng.uniform(0.50, 0.92))
        props.append(_make_stone_boulder(x, y, z - 0.30 * sr, sr, rng))
        stone_centers.append((x, y))
        if len(stone_centers) >= 34:
            break

    # 4. Vallas rústicas macizas (chi=2)
    fence_segments = [
        ((-10.8, -28.0), (-4.0, -18.5)),
        ((7.2, 8.5), (14.5, 19.5)),
        ((-15.0, -10.5), (-25.0, -6.5)),
    ]
    for p0, p1 in fence_segments:
        props.append(_make_solid_fence_segment(p0, p1, dem))

    # 5. Balas, gavillas, rueda y arado manual (a escala agrícola real)
    bale_specs = [
        (10.5, -14.8, 22.0, 1.15, 2.50),
        (14.5, -13.6, -18.0, 1.10, 2.35),
        (8.8, -10.8, 65.0, 1.08, 2.30),
        (12.8, -10.8, 10.0, 1.05, 2.30),
    ]
    for bx, by, byaw, br, bl in bale_specs:
        bz = Z_BASE - 0.55 + _bilinear(dem, bx, by)
        props.append(_make_straw_bale(bx, by, bz - 0.20, byaw, br, bl))

    sheaf_specs = [
        (-5.5, -24.0, 15.0, 0.74),
        (-7.8, -22.5, -30.0, 0.68),
        (16.2, -16.8, 12.0, 0.72),
        (18.2, -15.2, -40.0, 0.68),
        (-13.5, 17.5, 20.0, 0.72),
        (-15.8, 16.0, -25.0, 0.68),
    ]
    for sx, sy, syaw, ss in sheaf_specs:
        sz = Z_BASE - 0.55 + _bilinear(dem, sx, sy)
        props.append(_make_wheat_sheaf(sx, sy, sz, syaw, ss))

    wx, wy = 2.2, -14.8
    props.append(_make_solid_cartwheel(wx, wy, Z_BASE - 0.55 + _bilinear(dem, wx, wy), yaw_deg=-25.0, radius=2.35))

    px, py = 6.4, -15.6
    props.append(_make_hand_plow(px, py, Z_BASE - 0.55 + _bilinear(dem, px, py), yaw_deg=35.0))

    print(
        f"[CampoClay v2.1] Props: {len(oak_spots)} frondosos + {len(cypress_spots)} cipreses en lindes, "
        f"{len(bush_centers)} arbustos, {len(stone_centers)} piedras, {len(fence_segments)} vallas, "
        f"{len(bale_specs)} balas, {len(sheaf_specs)} gavillas, rueda y arado.",
        flush=True,
    )
    return props


def _stl_to_manifold(stl_path: Path):
    tm = trimesh.load(str(stl_path), force="mesh")
    mesh = mf.Mesh(
        vert_properties=np.asarray(tm.vertices, dtype=np.float32),
        tri_verts=np.asarray(tm.faces, dtype=np.uint32),
    )
    return mf.Manifold(mesh)


def run_build_geometry(base_stl: Path, out_dir: Path, n_seg: int = 760, res_dem: int = 2560) -> Dict[str, object]:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CampoClay v2.2 1/4] Generando campiña suave + 4 parcelas alineadas (DEM={res_dem})...", flush=True)
    fields = build_campo_terrain_fields(res=res_dem, seed=209)

    print(f"[CampoClay v2.1 2/4] Construyendo TerrainCap estanco de alta densidad (n_seg={n_seg})...", flush=True)
    terrain_mf = build_terrain_cap_manifold(fields["dem"], n_seg=n_seg)

    print("[CampoClay v2.1 3/4] Construyendo props agrícolas puros y verificando genus==0...", flush=True)
    props_list = build_agricultural_props_manifold(fields, seed=707)

    base_mf = _stl_to_manifold(base_stl).as_original()
    id_plinth = base_mf.original_id()

    # Unión rápida por bloques de 16 con verificación estricta de 1 único sólido sin túneles (genus==0, comps==1)
    organic_mf = terrain_mf
    dropped_genus = 0
    for i in range(0, len(props_list), 16):
        grp = props_list[i : i + 16]
        grp_mf = _batch_compose_manifolds(grp, chunk=16)
        cand = organic_mf + grp_mf
        if cand.genus() == 0 and len(cand.decompose()) == 1:
            organic_mf = cand
        else:
            for pr in grp:
                c2 = organic_mf + pr
                if c2.genus() == 0 and len(c2.decompose()) == 1:
                    organic_mf = c2
                else:
                    dropped_genus += 1
    if dropped_genus:
        print(f"[CampoClay v2.1] Filtrados {dropped_genus} props con micro-túnel para garantizar genus=0.", flush=True)
    organic_mf = organic_mf.as_original()
    master_mf = base_mf + organic_mf
    print(f"[CampoClay v2.1] Topología Manifold: terrain_genus={terrain_mf.genus()}, "
          f"organic_genus={organic_mf.genus()}, master_genus={master_mf.genus()}", flush=True)

    m_out = master_mf.to_mesh()
    verts = np.asarray(m_out.vert_properties, dtype=np.float32)[:, :3].astype(np.float64)
    faces = np.asarray(m_out.tri_verts, dtype=np.uint32)

    # Relajar vértices de micro-astillas booleanas (<1e-9 mm² en float32) garantizando degen=0 en STL
    for _it in range(8):
        p0, p1, p2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        bad = areas < 1e-8
        if not np.any(bad):
            break
        bad_v = np.unique(faces[bad])
        mask_f = np.isin(faces[:, 0], bad_v) | np.isin(faces[:, 1], bad_v) | np.isin(faces[:, 2], bad_v)
        sub_f = faces[mask_f]
        for v in bad_v:
            inc = sub_f[np.any(sub_f == v, axis=1)]
            nbrs = np.setdiff1d(np.unique(inc), [v])
            if len(nbrs):
                verts[v] = 0.60 * verts[v] + 0.40 * verts[nbrs].mean(axis=0)
        verts = verts.astype(np.float32).astype(np.float64)

    n_f = len(faces)
    face_slot = np.ones(n_f, dtype=np.uint8)
    ri = np.asarray(m_out.run_index, dtype=np.int64)
    roids = np.asarray(m_out.run_original_id, dtype=np.int64)
    for k in range(len(roids)):
        if roids[k] == id_plinth:
            face_slot[int(ri[k] // 3) : int(ri[k + 1] // 3)] = 0

    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    master_stl = out_dir / "tile_clay_master.stl"
    tm.export(str(master_stl))

    mesh_npz = out_dir / "clay_master_mesh.npz"
    np.savez_compressed(mesh_npz, vertices=verts.astype(np.float32), faces=faces, face_slot=face_slot)
    np.save(out_dir / "dem_campo.npy", fields["dem"].astype(np.float32))

    metrics = {
        "n_verts": int(len(verts)),
        "n_faces": int(len(faces)),
        "watertight": bool(tm.is_watertight),
        "euler_number": int(tm.euler_number),
        "genus": int(master_mf.genus()),
        "z_min_mm": round(float(verts[:, 2].min()), 3),
        "z_max_mm": round(float(verts[:, 2].max()), 3),
        "build_time_s": round(time.time() - t0, 1),
        "master_stl": str(master_stl),
    }
    (out_dir / "clay_build_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[CampoClay v2.1 4/4] Listo en {metrics['build_time_s']}s: {metrics}", flush=True)
    return metrics


def run_blender_render_views(out_dir: Path, res_px: int = 1024, samples: int = 24) -> None:
    import bpy
    import mathutils

    t0 = time.time()
    data = np.load(str(out_dir / "clay_master_mesh.npz"))
    V = data["vertices"].astype(np.float64)
    F = data["faces"].astype(np.int32)
    slots = data["face_slot"].astype(np.int32)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"

    world = bpy.data.worlds.new("StudioClayWorld")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.80, 0.80, 0.80, 1.0)
        bg.inputs["Strength"].default_value = 0.90

    me = bpy.data.meshes.new("Catan_Campo_Mesh")
    me.from_pydata(V.tolist(), [], F.tolist())
    me.polygons.foreach_set("material_index", slots.tolist())
    me.polygons.foreach_set("use_smooth", (slots >= 1).tolist())
    me.update()

    def _mk_mat(name, color, rough, metal=0.0):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = rough
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metal
        return m

    mat_plinth = _mk_mat("Plinth_Gunmetal", (0.042, 0.046, 0.052, 1.0), rough=0.45, metal=0.22)
    mat_clay = _mk_mat("Clay_Sculpt", (0.58, 0.56, 0.53, 1.0), rough=0.72, metal=0.0)
    me.materials.append(mat_plinth)
    me.materials.append(mat_clay)

    obj = bpy.data.objects.new("Catan_Campo", me)
    sc.collection.objects.link(obj)

    sun_specs = [
        ("KeySun", (75.0, -65.0, 85.0), 2.3, (1.0, 0.99, 0.96), math.radians(16.0)),
        ("FillSun", (-80.0, -40.0, 75.0), 1.05, (0.90, 0.94, 1.0), math.radians(38.0)),
        ("RimSun", (0.0, 95.0, 70.0), 1.10, (0.98, 0.99, 1.0), math.radians(28.0)),
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

    cam_data = bpy.data.cameras.new("ClayCam")
    cam_data.type = "PERSP"
    fov_deg = 15.68
    cam_data.sensor_width = 36.0
    cam_data.lens = 18.0 / math.tan(math.radians(fov_deg * 0.5))
    cam_obj = bpy.data.objects.new("ClayCam", cam_data)
    sc.collection.objects.link(cam_obj)
    sc.camera = cam_obj

    views_dir = out_dir / "clay_views"
    views_dir.mkdir(parents=True, exist_ok=True)
    sc.render.resolution_x = res_px
    sc.render.resolution_y = res_px

    r_horiz = 399.87
    cam_z = 365.78
    target = mathutils.Vector((0.0, 0.0, 27.5))

    cams_json: Dict[str, Dict] = {}
    for deg in (0, 60, 120, 180, 240, 300):
        az = math.radians(deg)
        cx = -r_horiz * math.sin(az)
        cy = -r_horiz * math.cos(az)
        cam_obj.location = (cx, cy, cam_z)
        cam_obj.rotation_euler = (target - cam_obj.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()

        vp = views_dir / f"clay_{deg:03d}.png"
        sc.render.filepath = str(vp)
        bpy.ops.render.render(write_still=True)

        mat = cam_obj.matrix_world
        fwd = -mat.col[2].xyz.normalized()
        right_r = mat.col[0].xyz.normalized()
        up_r = mat.col[1].xyz.normalized()
        cams_json[str(deg)] = {
            "cam_xyz": [float(cx), float(cy), float(cam_z)],
            "fwd": [float(fwd.x), float(fwd.y), float(fwd.z)],
            "right_r": [float(right_r.x), float(right_r.y), float(right_r.z)],
            "up_r": [float(up_r.x), float(up_r.y), float(up_r.z)],
            "fov_deg": fov_deg,
            "W": res_px,
            "H": res_px,
            "angle_deg": deg,
            "image": str(vp),
        }

    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 142.0
    cam_obj.location = (0.0, 0.0, 220.0)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    sc.render.filepath = str(out_dir / "audit_ortho.png")
    bpy.ops.render.render(write_still=True)

    # Render macro dedicado a los árboles de linde para inspección visual directa
    cam_data.type = "PERSP"
    cam_data.lens = 65.0
    cam_obj.location = (22.0, -20.0, 56.0)
    cam_obj.rotation_euler = (mathutils.Vector((3.5, 16.5, 34.0)) - cam_obj.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()
    sc.render.filepath = str(out_dir / "clay_trees_closeup.png")
    bpy.ops.render.render(write_still=True)

    (out_dir / "cameras.json").write_text(json.dumps(cams_json, indent=2), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "campo_clay_scene.blend"))
    print(f"[CampoClay v2.1 Render] 6 vistas + ortho + cameras.json listos en {time.time() - t0:.1f}s", flush=True)


def _assemble_clay_mosaic(out_dir: Path) -> None:
    from PIL import Image, ImageDraw
    views_dir = out_dir / "clay_views"
    tiles = []
    for deg in (0, 60, 120, 180, 240, 300):
        im = Image.open(views_dir / f"clay_{deg:03d}.png").convert("RGB").resize((640, 640), Image.Resampling.LANCZOS)
        dr = ImageDraw.Draw(im)
        dr.rectangle([10, 10, 130, 38], fill=(24, 28, 34))
        dr.text((18, 16), f"Clay {deg} deg", fill=(240, 244, 250))
        tiles.append(np.asarray(im))
    top_row = np.concatenate(tiles[:3], axis=1)
    bot_row = np.concatenate(tiles[3:], axis=1)
    Image.fromarray(np.concatenate([top_row, bot_row], axis=0)).save(out_dir / "audit_6views_clay.png")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description="Constructor 3D Clay Bioma Campo (Full Trigo)")
    p.add_argument("--mode", choices=["all", "geom", "render"], default="all")
    p.add_argument("--base-stl", default=str(DEFAULT_BASE_STL))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--n-seg", type=int, default=760)
    p.add_argument("--dem-res", type=int, default=2560)
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--samples", type=int, default=24)
    args = p.parse_args(argv)

    base_stl = Path(args.base_stl).resolve()
    out_dir = Path(args.out_dir).resolve()

    if args.mode == "geom":
        run_build_geometry(base_stl, out_dir, n_seg=args.n_seg, res_dem=args.dem_res)
    elif args.mode == "render":
        run_blender_render_views(out_dir, res_px=args.res, samples=args.samples)
    else:
        run_build_geometry(base_stl, out_dir, n_seg=args.n_seg, res_dem=args.dem_res)
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import tool_paths as tp
        blender_exe = tp.find_blender()
        cmd = [
            blender_exe, "-b", "--python-exit-code", "1",
            "--python", str(Path(__file__).resolve()),
            "--", "--mode", "render", "--out-dir", str(out_dir),
            "--res", str(args.res), "--samples", str(args.samples),
        ]
        res = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0:
            raise RuntimeError(f"Blender render falló (rc={res.returncode})")
        _assemble_clay_mosaic(out_dir)


if __name__ == "__main__":
    main()
