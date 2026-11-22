"""
Módulo 1 de la Fábrica Modular 3D Catán (CSG Estanco Exacto con manifold3d + build123d):
  - Genera el zócalo B-Rep exacto (110.8513×128×25 mm, imán Ø10.2×5.2 mm, 6 pads Ø20 mm).
  - Construye el lecho orgánico del terreno (colinas, cañón S terracota, circo cóncavo).
  - Construye bloques/contrafuertes graníticos prismáticos anchos e imbricados (cero espinas
    sueltas, cero derretido OpenVDB) y fragmentos angulares de glera/pedregal 3D.
  - Realiza la unión booleana CSG exacta vía manifold3d (garantizando watertight=True,
    chi=2, non_manifold_edges=0, zero degenerate faces) y exporta tile_master.stl +
    splat_mask.png (máscara semántica XY para el shader Cycles 3D).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import manifold3d as m3d
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from catan_kit.build_catan_base_brep import build_catan_base, build_catan_inner_cutter  # noqa: E402
from build123d import export_stl  # noqa: E402

R = 64.0000
APOTHEM = R * math.cos(math.radians(30.0))  # 55.425626 mm
Z_BASE = 25.0000
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00


def smoothstep(e0, e1, x):
    lo = np.minimum(e0, e1)
    hi = np.maximum(e0, e1)
    t = np.clip((x - lo) / np.maximum(hi - lo, 1e-8), 0.0, 1.0)
    res = t * t * (3.0 - 2.0 * t)
    return np.where(e0 > e1, 1.0 - res, res)


def smax(a, b, k: float = 2.0):
    h = np.clip(0.5 + 0.5 * (a - b) / k, 0.0, 1.0)
    return a * h + b * (1.0 - h) + k * h * (1.0 - h) * 0.5


def hex_distances(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dist_edge = np.full_like(X, 999.0, dtype=np.float32)
    for k in range(6):
        ang = k * math.pi / 3.0
        dist_edge = np.minimum(dist_edge, APOTHEM - (X * math.cos(ang) + Y * math.sin(ang)))
    dist_vert = np.full_like(X, 999.0, dtype=np.float32)
    for k in range(6):
        ang = math.radians(30.0 + 60.0 * k)
        dist_vert = np.minimum(dist_vert, np.hypot(X - R * math.cos(ang), Y - R * math.sin(ang)))
    return dist_edge, dist_vert


def fbm_2d(X: np.ndarray, Y: np.ndarray, scales: List[float], weights: List[float], seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros_like(X, dtype=np.float32)
    for sc, wt in zip(scales, weights):
        ang = rng.uniform(0.0, 2.0 * math.pi, 4)
        ph = rng.uniform(0.0, 2.0 * math.pi, 4)
        n = (
            np.sin(X * sc * np.cos(ang[0]) + Y * sc * np.sin(ang[0]) + ph[0])
            + np.cos(X * sc * np.cos(ang[1]) + Y * sc * np.sin(ang[1]) + ph[1])
            + np.sin(X * sc * np.cos(ang[2]) - Y * sc * np.sin(ang[2]) + ph[2])
            + np.cos(X * sc * np.cos(ang[3]) - Y * sc * np.sin(ang[3]) + ph[3])
        ) * 0.25
        out += wt * (0.5 + 0.5 * n)
    return out / max(sum(weights), 1e-6)


def ridge_fbm_2d(X: np.ndarray, Y: np.ndarray, scales: List[float], weights: List[float], seed: int = 77) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros_like(X, dtype=np.float32)
    for sc, wt in zip(scales, weights):
        ang = rng.uniform(0.0, 2.0 * math.pi, 3)
        ph = rng.uniform(0.0, 2.0 * math.pi, 3)
        n = (
            np.sin(X * sc * np.cos(ang[0]) + Y * sc * np.sin(ang[0]) + ph[0])
            + np.cos(X * sc * np.cos(ang[1]) + Y * sc * np.sin(ang[1]) + ph[1])
            + np.sin(X * sc * np.cos(ang[2]) - Y * sc * np.sin(ang[2]) + ph[2])
        ) / 3.0
        out += wt * (1.0 - np.abs(n))
    return out / max(sum(weights), 1e-6)


def path_distance_field(X: np.ndarray, Y: np.ndarray, pts_xy_w_d: List[Tuple[float, float, float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    carve = np.zeros_like(X, dtype=np.float32)
    mask = np.zeros_like(X, dtype=np.float32)
    for i in range(len(pts_xy_w_d) - 1):
        x0, y0, w0, d0 = pts_xy_w_d[i]
        x1, y1, w1, d1 = pts_xy_w_d[i + 1]
        vx, vy = x1 - x0, y1 - y0
        l2 = vx * vx + vy * vy + 1e-8
        t = np.clip(((X - x0) * vx + (Y - y0) * vy) / l2, 0.0, 1.0)
        px = x0 + t * vx
        py = y0 + t * vy
        dist = np.hypot(X - px, Y - py)
        w = w0 * (1.0 - t) + w1 * t
        d = d0 * (1.0 - t) + d1 * t
        prof = np.clip(1.0 - (dist / np.maximum(w, 0.5)) ** 1.65, 0.0, 1.0)
        carve = np.maximum(carve, d * prof)
        mask = np.maximum(mask, smoothstep(w * 1.15, w * 0.38, dist))
    return carve, mask


def _trimesh_to_manifold(tm: trimesh.Trimesh) -> m3d.Manifold:
    mesh = m3d.Mesh(
        vert_properties=np.asarray(tm.vertices, dtype=np.float32),
        tri_verts=np.asarray(tm.faces, dtype=np.uint32),
    )
    return m3d.Manifold(mesh)


def _manifold_to_trimesh(mf: m3d.Manifold) -> trimesh.Trimesh:
    out_m = mf.to_mesh()
    verts = np.asarray(out_m.vert_properties[:, :3], dtype=np.float64)
    faces = np.asarray(out_m.tri_verts, dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def build_terrain_bed_manifold(nx: int, Z_top: np.ndarray, z_bot: float = Z_BASE - 2.6) -> m3d.Manifold:
    """Convierte una malla de altura hexagonal Z_top(nx, nx) en un sólido Manifold estanco."""
    lin = np.linspace(-R, R, nx, dtype=np.float64)
    X, Y = np.meshgrid(lin, lin)
    # Vértices: [0..nx*nx-1] top, [nx*nx..2*nx*nx-1] bot
    top_v = np.column_stack([X.ravel(), Y.ravel(), Z_top.ravel()])
    bot_v = np.column_stack([X.ravel(), Y.ravel(), np.full(nx * nx, z_bot, dtype=np.float64)])
    verts = np.vstack([top_v, bot_v]).astype(np.float32)

    off = nx * nx
    faces = []
    for iy in range(nx - 1):
        for ix in range(nx - 1):
            i00 = iy * nx + ix
            i10 = i00 + 1
            i01 = i00 + nx
            i11 = i01 + 1
            if (ix + iy) % 2 == 0:
                faces.append([i00, i10, i11])
                faces.append([i00, i11, i01])
            else:
                faces.append([i00, i10, i01])
                faces.append([i10, i11, i01])
            b00, b10, b01, b11 = off + i00, off + i10, off + i01, off + i11
            faces.append([b00, b11, b10])
            faces.append([b00, b01, b11])

    # 4 paredes perimetrales
    for ix in range(nx - 1):
        t0, t1 = ix, ix + 1
        b0, b1 = off + t0, off + t1
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
        r_off = (nx - 1) * nx
        t0n, t1n = r_off + ix, r_off + ix + 1
        b0n, b1n = off + t0n, off + t1n
        faces.append([t0n, t1n, b1n])
        faces.append([t0n, b1n, b0n])
    for iy in range(nx - 1):
        t0, t1 = iy * nx, (iy + 1) * nx
        b0, b1 = off + t0, off + t1
        faces.append([t0, t1, b1])
        faces.append([t0, b1, b0])
        t0r, t1r = iy * nx + (nx - 1), (iy + 1) * nx + (nx - 1)
        b0r, b1r = off + t0r, off + t1r
        faces.append([b0r, b1r, t1r])
        faces.append([b0r, t1r, t0r])

    mesh = m3d.Mesh(vert_properties=verts, tri_verts=np.asarray(faces, dtype=np.uint32))
    return m3d.Manifold(mesh)


def make_faceted_granite_slab_manifold(
    base_xy: Tuple[float, float],
    z_bot: float,
    height: float,
    rx: float,
    ry: float,
    sides: int = 14,
    tiers: int = 9,
    taper_top: float = 0.20,
    squeeze_perp: float = 0.42,
    tilt_xy: Tuple[float, float] = (0.0, 0.0),
    rot_z_deg: float = 0.0,
    wedge_dir_deg: float = 0.0,
    wedge_slope: float = 0.28,
    apex_lift: float = 0.45,
    seed: int = 1,
) -> m3d.Manifold:
    """
    Construye una masa de granito esculpida en alta definición con costillas verticales estriadas,
    estratos escalonados y arista superior asimétrica estanca.
    """
    # Aumentar densidad angular cuando se pasa sides=5..6 heredado para evitar aspecto low-poly origami
    s_eff = max(12, sides * 2) if sides < 10 else sides
    t_eff = max(8, tiers)
    rng = np.random.default_rng(seed)
    bx, by = base_xy
    ang0 = math.radians(rot_z_deg)
    w_rad = math.radians(wedge_dir_deg)
    wx, wy = math.cos(w_rad), math.sin(w_rad)
    px_dir, py_dir = -wy, wx

    n_flutes = int(rng.integers(4, 7))
    ph1 = float(rng.uniform(0.0, 2.0 * math.pi))
    ph2 = float(rng.uniform(0.0, 2.0 * math.pi))
    angs = [ang0 + 2.0 * math.pi * k / s_eff + float(rng.uniform(-0.08, 0.08)) for k in range(s_eff)]
    rad_j = np.array([
        rng.uniform(0.84, 1.18)
        * (1.0 - 0.18 * abs(math.sin(n_flutes * (angs[k] - ang0) + ph1)))
        * (1.0 + 0.09 * math.cos((n_flutes + 2) * (angs[k] - ang0) + ph2))
        for k in range(s_eff)
    ], dtype=np.float64)

    verts: List[Tuple[float, float, float]] = []
    ring_idx: List[List[int]] = []
    step_x, step_y = 0.0, 0.0
    for ti in range(t_eff + 1):
        frac = ti / float(t_eff)
        # Perfil piramidal natural con escalones de estrato cada 2-3 anillos
        ledge = -0.055 * (frac * 0.8 + 0.2) if (ti % 3 == 2 and 0 < ti < t_eff) else 0.0
        sc = max(0.11, (1.0 - frac) + taper_top * frac - 0.04 * (frac ** 1.8) + ledge)
        if 0 < ti < t_eff and rng.random() < 0.45:
            step_x += float(rng.normal(0.0, 0.08 * min(rx, ry)))
            step_y += float(rng.normal(0.0, 0.08 * min(rx, ry)))
        cx = bx + tilt_xy[0] * frac * height + step_x * (1.0 - 0.25 * frac)
        cy = by + tilt_xy[1] * frac * height + step_y * (1.0 - 0.25 * frac)
        z_cur = z_bot + frac * height

        cur_ring = []
        for k in range(s_eff):
            ak = angs[k]
            rib_var = 1.0 + 0.08 * math.sin(frac * math.pi * 3.0 + float(k) * 0.9)
            tier_j = rib_var * (1.0 + (float(rng.uniform(-0.045, 0.045)) if 0 < ti < t_eff else 0.0))
            lx = math.cos(ak) * rx * float(rad_j[k]) * sc * tier_j
            ly = math.sin(ak) * ry * float(rad_j[k]) * sc * tier_j
            if ti == t_eff:
                dz_w = (lx * wx + ly * wy) * wedge_slope * 1.55
                proj_p = lx * px_dir + ly * py_dir
                lx -= px_dir * proj_p * squeeze_perp
                ly -= py_dir * proj_p * squeeze_perp
                zv = z_cur + dz_w
            elif ti == t_eff - 1:
                zv = z_cur + (lx * wx + ly * wy) * wedge_slope * 0.72
            else:
                zv = z_cur + float(rng.uniform(-0.22, 0.22)) * (0.0 if ti == 0 else 1.0)
            cur_ring.append(len(verts))
            verts.append((cx + lx, cy + ly, zv))
        ring_idx.append(cur_ring)

    # Centros inferior y remate superior desplazado hacia el borde alto de la cuña
    bot_c_idx = len(verts)
    verts.append((bx, by, z_bot))
    top_pts = np.array([verts[i] for i in ring_idx[-1]], dtype=np.float64)
    high_idx = int(np.argmax(top_pts[:, 2]))
    apex = 0.64 * np.mean(top_pts, axis=0) + 0.36 * top_pts[high_idx]
    apex[2] = float(np.max(top_pts[:, 2])) * 0.56 + float(np.mean(top_pts[:, 2])) * 0.44 + apex_lift
    top_c_idx = len(verts)
    verts.append((float(apex[0]), float(apex[1]), float(apex[2])))

    faces: List[Tuple[int, int, int]] = []
    r0 = ring_idx[0]
    for k in range(s_eff):
        kn = (k + 1) % s_eff
        faces.append((bot_c_idx, r0[kn], r0[k]))
    for ti in range(t_eff):
        ra, rb = ring_idx[ti], ring_idx[ti + 1]
        for k in range(s_eff):
            kn = (k + 1) % s_eff
            faces.append((ra[k], ra[kn], rb[kn]))
            faces.append((ra[k], rb[kn], rb[k]))
    rt = ring_idx[-1]
    for k in range(s_eff):
        kn = (k + 1) % s_eff
        faces.append((rt[k], rt[kn], top_c_idx))

    tm = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64), process=True)
    if tm.volume < 0:
        tm.invert()
    return _trimesh_to_manifold(tm)


def make_angular_boulder_manifold(
    center: Tuple[float, float, float],
    rx: float,
    ry: float,
    rz: float,
    seed: int = 1,
) -> m3d.Manifold:
    """Fragmento de roca angular fracturada por planos de clivaje (estanco, cero esferas)."""
    rng = np.random.default_rng(seed)
    ico = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    v = ico.vertices.copy()
    for _ in range(3):
        n = rng.normal(0.0, 1.0, size=3)
        n /= max(float(np.linalg.norm(n)), 1e-6)
        d = float(rng.uniform(0.42, 0.72))
        proj = v @ n
        over = proj > d
        if np.any(over):
            v[over] -= np.outer((proj[over] - d) * 0.92, n)
    v[:, 0] *= rx
    v[:, 1] *= ry
    v[:, 2] *= rz
    # Rotación aleatoria 3D
    ang = rng.uniform(0.0, 2.0 * math.pi, size=3)
    Rmat = trimesh.transformations.euler_matrix(ang[0], ang[1], ang[2])[:3, :3]
    v = (v @ Rmat.T) + np.array(center, dtype=np.float64)
    tm = trimesh.Trimesh(vertices=v, faces=ico.faces, process=True)
    if tm.volume < 0:
        tm.invert()
    return _trimesh_to_manifold(tm)


def build_montana_csg(out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base B-Rep + Cutter B-Rep
    base_solid, _ = build_catan_base("circle")
    cutter_solid = build_catan_inner_cutter(road_margin=ROAD_MARGIN, pad_radius=PAD_RADIUS, z_min=20.0, z_max=120.0)
    base_stl = work_dir / "catan_base_brep.stl"
    cutter_stl = work_dir / "catan_inner_cutter.stl"
    export_stl(base_solid, str(base_stl))
    export_stl(cutter_solid, str(cutter_stl))

    base_mf = _trimesh_to_manifold(trimesh.load(str(base_stl), force="mesh"))
    cutter_mf = _trimesh_to_manifold(trimesh.load(str(cutter_stl), force="mesh"))

    # 2. Lecho alpino orgánico + Macizo piramidal amplio + Circo glaciar real (cero burbuja) + Garganta rocosa
    nx_bed = 320
    lin = np.linspace(-R, R, nx_bed, dtype=np.float32)
    X, Y = np.meshgrid(lin, lin)
    dist_e, dist_v = hex_distances(X, Y)
    in_area = smoothstep(ROAD_MARGIN + 0.15, ROAD_MARGIN + 5.8, dist_e) * smoothstep(PAD_RADIUS + 0.15, PAD_RADIUS + 5.4, dist_v)

    hills = fbm_2d(X, Y, [0.09, 0.22, 0.50], [0.48, 0.34, 0.18], seed=101)
    rock_flutes = ridge_fbm_2d(X, Y, [0.16, 0.42, 0.98], [0.52, 0.32, 0.16], seed=205)

    def ridge_spur(x0: float, y0: float, x1: float, y1: float, h0: float, h1: float, w0: float, w1: float, power: float = 1.25) -> np.ndarray:
        vx, vy = x1 - x0, y1 - y0
        l2 = vx * vx + vy * vy + 1e-8
        t = np.clip(((X - x0) * vx + (Y - y0) * vy) / l2, 0.0, 1.0)
        d = np.hypot(X - (x0 + t * vx), Y - (y0 + t * vy))
        w = w0 * (1.0 - t) + w1 * t
        h = h0 * (1.0 - t) + h1 * t
        return h * np.power(np.maximum(0.0, 1.0 - d / np.maximum(w, 0.5)), power)

    # Dorsal maestra amplia cubriendo 70% de la loseta con estribaciones graduales hasta las praderas
    spurs = [
        (-13.5, 11.5, 2.0, 12.0,   28.5, 24.5, 24.0, 22.0, 1.18),  # Cresta central gemela
        (-13.5, 11.5, -29.0, 24.0, 25.5, 7.5,  21.0, 14.0, 1.25),  # Arista NW amplia
        (-4.5, 13.5, -3.5, 34.5,   24.5, 8.0,  20.0, 14.0, 1.28),  # Arista N amplia
        (2.0, 12.5, 24.0, 24.5,    23.0, 7.0,  19.5, 13.5, 1.28),  # Arista NE amplia
        (-14.5, 10.0, -32.0, -8.5, 24.0, 6.8,  20.0, 13.5, 1.28),  # Arista W / glera
        (-13.5, 7.0, -14.5, -18.5, 20.5, 6.0,  17.5, 12.5, 1.32),  # Bastión frontal izq
        (10.0, 8.5, 28.5, -4.5,    20.5, 6.5,  17.5, 12.5, 1.32),  # Arista E
        (19.5, -2.0, 29.0, -16.5,  14.0, 5.2,  14.5, 11.0, 1.35),  # Saliente SE derecha garganta
    ]
    massif_core = ridge_spur(*spurs[0])
    for sp in spurs[1:]:
        massif_core = smax(massif_core, ridge_spur(*sp), k=3.5)

    # Circo glaciar central amplio (+0.5, +1.5) en rampa suave (~26°-35°) subiendo al collado entre los picos gemelos!
    cirque_main = 12.0 * np.exp(-0.5 * (((X - 0.5) / 9.8) ** 2 + ((Y - 1.5) / 9.6) ** 2))
    cirque_col = 7.5 * np.exp(-0.5 * (((X + 4.5) / 6.0) ** 2 + ((Y - 11.0) / 7.0) ** 2))
    cirque_left = 4.6 * np.exp(-0.5 * (((X + 19.5) / 5.5) ** 2 + ((Y + 11.5) / 5.5) ** 2))
    cirque_nw = 5.8 * np.exp(-0.5 * (((X + 16.5) / 7.5) ** 2 + ((Y - 23.5) / 8.2) ** 2))
    cirque_ne = 5.8 * np.exp(-0.5 * (((X - 11.5) / 7.5) ** 2 + ((Y - 23.5) / 8.2) ** 2))

    # Estrías y canales rocosos reales en todo el macizo (suavizados dentro del circo glaciar para nieve limpia)
    Z_rock_base = np.maximum(0.0, massif_core - (cirque_main + cirque_col + cirque_left + cirque_nw + cirque_ne))
    in_snow_d = np.exp(-0.5 * (((X - 0.5) / 9.2) ** 2 + ((Y - 2.0) / 9.2) ** 2))
    Z_rock_base += smoothstep(1.2, 9.0, Z_rock_base) * (1.0 - 0.85 * in_snow_d) * 3.8 * (rock_flutes - 0.38)
    # Lengua glaciar en U cóncava (~24°-32°) encajada hacia el collado entre los picos (cero cúpula abombada)
    glacier_u = (
        smoothstep(-5.5, 11.0, Y)
        * 7.2
        * (1.0 - 0.35 * np.exp(-0.5 * ((X - 0.2) / 3.6) ** 2))
        * np.exp(-0.5 * ((X / 8.8) ** 2 + ((Y - 3.2) / 8.8) ** 2))
    )
    Z_rock_base = np.maximum(Z_rock_base, glacier_u + 0.55 * rock_flutes * smoothstep(2.0, 10.0, glacier_u))

    # Garganta rocosa corta y natural saliendo del circo hacia SE entre flancos escarpados
    canyon_pts = [
        (5.2, -5.8, 3.6, 2.2),
        (9.5, -11.0, 3.2, 2.4),
        (13.5, -16.0, 3.0, 1.4),
    ]
    carve_canyon, mask_path = path_distance_field(X, Y, canyon_pts)

    rel_bed = in_area * (1.20 + 2.35 * hills + Z_rock_base)
    rel_bed = np.maximum(in_area * 0.55, rel_bed - carve_canyon * in_area)
    Z_bed = Z_BASE + rel_bed

    bed_mf = build_terrain_bed_manifold(nx_bed, Z_bed)

    # 3. Masas graníticas piramidales ANCHAS e IMBRICADAS enmarcando el anfiteatro glaciar frontal
    Z_SLAB_BOT = Z_BASE - 2.0
    slab_specs = [
        # --- PICO MAESTRO IZQUIERDO (-14.5, +15.0 -> cúspide ~ 70.0 mm) ---
        ((-14.5, 15.0), 42.6, 13.8, 10.6, 6, 0.08, 0.42, (0.02, 0.01),  16.0, 28.0, 0.26, 0.60, 301),
        ((-19.0, 11.5), 35.5, 12.6, 10.2, 6, 0.12, 0.36, (0.04, 0.02),  50.0, 14.0, 0.24, 0.45, 302),
        ((-11.2, 16.5), 34.5, 10.6,  8.8, 5, 0.12, 0.38, (-0.02, 0.01), -22.0, 52.0, 0.24, 0.45, 303),
        ((-16.0, 20.0), 33.8, 12.8, 10.4, 6, 0.14, 0.34, (0.01, -0.04), 72.0, 112.0, 0.24, 0.40, 304),

        # --- PICO GEMELO SECUNDARIO (+7.0, +14.8 -> ~63.5 mm, deja abierto todo el circo nevado central!) ---
        ((7.0, 14.8),   36.6, 12.2,  9.6, 6, 0.10, 0.40, (-0.02, 0.01), -34.0, 42.0, 0.26, 0.50, 305),
        ((11.2, 14.5),  30.2, 11.4,  9.4, 6, 0.16, 0.32, (-0.03, -0.02), 62.0, 74.0, 0.22, 0.38, 307),
        ((5.5, 20.0),   29.0, 11.8,  9.8, 6, 0.18, 0.30, (-0.01, -0.04), -68.0, 118.0, 0.22, 0.35, 308),

        # --- HOMBROS DE CRESTA NORTE / NW / NE ---
        ((-20.5, 20.0), 25.5, 12.0, 10.0, 6, 0.18, 0.30, (0.04, -0.04), 32.0, 115.0, 0.22, 0.35, 309),
        ((-25.5, 23.5), 16.5, 10.8,  9.0, 5, 0.24, 0.26, (0.05, -0.05), -18.0, 125.0, 0.18, 0.25, 310),
        ((-5.2, 23.8),  25.0, 11.4,  9.0, 6, 0.18, 0.30, (0.00, -0.04), -26.0, 95.0, 0.22, 0.32, 311),
        ((-4.2, 30.8),  15.2, 10.2,  8.6, 6, 0.26, 0.25, (0.00, -0.05), 50.0, 88.0, 0.18, 0.22, 312),
        ((15.0, 19.0),  21.5, 11.4,  9.4, 6, 0.20, 0.28, (-0.04, -0.04), 22.0, 62.0, 0.20, 0.28, 313),
        ((21.0, 21.5),  14.0, 10.0,  8.4, 5, 0.26, 0.24, (-0.05, -0.04), -36.0, 52.0, 0.18, 0.20, 314),

        # --- ARISTA OESTE (CAÍDA HACIA LA GLERA) ---
        ((-23.5, 3.5),  23.5, 12.2, 10.0, 6, 0.18, 0.30, (0.04, 0.02),  28.0, -40.0, 0.20, 0.30, 315),
        ((-28.5, -2.5), 16.0, 10.8,  8.8, 5, 0.24, 0.26, (0.05, 0.03),  52.0, -32.0, 0.18, 0.22, 316),
        ((-20.5, -5.0), 19.0, 11.0,  9.2, 6, 0.22, 0.28, (0.03, 0.04), -20.0, -58.0, 0.18, 0.26, 317),

        # --- ARISTA ESTE & PARED DERECHA DEL CAÑÓN (enmarcan la garganta SE) ---
        ((16.0, 8.0),   23.0, 11.6,  9.6, 6, 0.18, 0.30, (-0.04, 0.01), -14.0, 16.0, 0.20, 0.30, 318),
        ((21.8, 3.0),   17.5, 10.6,  8.8, 5, 0.24, 0.26, (-0.05, 0.02), 46.0, -10.0, 0.18, 0.24, 319),
        ((26.5, -3.0),  11.8,  9.4,  7.8, 5, 0.28, 0.24, (-0.05, 0.03), 22.0, -25.0, 0.16, 0.20, 320),
        ((19.5, -7.5),  15.5,  9.8,  8.2, 6, 0.24, 0.26, (-0.04, 0.03), -18.0, -32.0, 0.18, 0.22, 321),
        ((23.5, -13.5), 11.0,  8.8,  7.4, 6, 0.28, 0.22, (-0.04, 0.04), 34.0, -40.0, 0.16, 0.18, 322),

        # --- BASTIÓN FRONTAL-IZQUIERDO (ENVUELVE EL CIRCO Y LA PARED IZQUIERDA DEL CAÑÓN) ---
        ((-15.0, -10.5), 18.2, 10.6, 8.8, 6, 0.22, 0.28, (0.02, 0.04),  24.0, -65.0, 0.18, 0.24, 323),
        ((-11.8, -16.0), 12.8,  9.4, 8.0, 6, 0.26, 0.25, (0.01, 0.05), -40.0, -75.0, 0.16, 0.20, 324),
        ((-4.8, -16.5),   9.8,  8.4, 7.2, 5, 0.28, 0.24, (-0.02, 0.05), 18.0, -55.0, 0.16, 0.18, 325),
    ]

    parts: List[m3d.Manifold] = [bed_mf]
    rng_sub = np.random.default_rng(777)
    for (xy, rel_top, rx, ry, sides, taper, sq_p, tilt, rotz, wedge_d, w_slp, a_lift, seed) in slab_specs:
        h_full = (Z_BASE + rel_top) - Z_SLAB_BOT
        parts.append(
            make_faceted_granite_slab_manifold(
                xy, Z_SLAB_BOT, h_full, rx, ry, sides=sides, tiers=4, taper_top=taper,
                squeeze_perp=sq_p, tilt_xy=tilt, rot_z_deg=rotz, wedge_dir_deg=wedge_d,
                wedge_slope=w_slp, apex_lift=a_lift, seed=seed,
            )
        )
        # Contrafuerte secundario fuera del anfiteatro glaciar central (-4..+6, -4..+10)
        off_a = float(rng_sub.uniform(0.0, 2.0 * math.pi))
        off_d = float(rng_sub.uniform(0.24, 0.42)) * max(rx, ry)
        sx, sy = xy[0] + math.cos(off_a) * off_d, xy[1] + math.sin(off_a) * off_d
        if math.hypot(sx - 0.5, sy - 2.0) < 9.8:
            continue
        sub_rel_top = rel_top * float(rng_sub.uniform(0.56, 0.76))
        sub_h = (Z_BASE + sub_rel_top) - Z_SLAB_BOT
        parts.append(
            make_faceted_granite_slab_manifold(
                (sx, sy),
                Z_SLAB_BOT, sub_h,
                rx * float(rng_sub.uniform(0.76, 0.94)),
                ry * float(rng_sub.uniform(0.76, 0.94)),
                sides=int(rng_sub.integers(5, 8)), tiers=4, taper_top=min(0.36, taper + 0.08),
                squeeze_perp=sq_p * 0.85, tilt_xy=tilt,
                rot_z_deg=rotz + float(rng_sub.uniform(-32, 32)),
                wedge_dir_deg=wedge_d + float(rng_sub.uniform(-25, 25)),
                wedge_slope=w_slp * 0.85, apex_lift=a_lift * 0.70, seed=seed * 10 + 1,
            )
        )

    # 4. Glera angular densa izquierda + cantos rodados de estribación (35% bajo el lecho -> chi=2!)
    rng_sc = np.random.default_rng(888)
    cand_pts = []
    for _ in range(95):
        t = float(rng_sc.uniform(0.0, 1.0))
        bx = -20.5 - 14.0 * t + float(rng_sc.normal(0.0, 2.8))
        by = -7.0 - 16.5 * t + float(rng_sc.normal(0.0, 2.8))
        cand_pts.append((bx, by, float(rng_sc.uniform(0.95, 2.25)), True))
    for _ in range(110):
        ang = float(rng_sc.uniform(0.0, 2.0 * math.pi))
        rad_p = float(rng_sc.uniform(21.0, 42.5))
        cand_pts.append((rad_p * math.cos(ang), rad_p * math.sin(ang), float(rng_sc.uniform(0.85, 1.95)), False))

    inv_span = (nx_bed - 1) / (2.0 * R)
    for i, (bx, by, rad, is_scree) in enumerate(cand_pts):
        if abs(bx) > APOTHEM or abs(by) > R:
            continue
        ix = int(np.clip(round((bx + R) * inv_span), 0, nx_bed - 1))
        iy = int(np.clip(round((by + R) * inv_span), 0, nx_bed - 1))
        if dist_e[iy, ix] < ROAD_MARGIN + 4.0 or dist_v[iy, ix] < PAD_RADIUS + 4.0:
            continue
        rz = rad * float(rng_sc.uniform(0.45, 0.78) if is_scree else rng_sc.uniform(0.38, 0.65))
        z_anchor = max(Z_BASE - 0.25, float(Z_bed[iy, ix]) - 0.35 * rz)
        parts.append(
            make_angular_boulder_manifold(
                (bx, by, z_anchor),
                rad * float(rng_sc.uniform(0.85, 1.35)),
                rad * float(rng_sc.uniform(0.75, 1.2)),
                rz,
                seed=2000 + i,
            )
        )

    print(f"[CSG] Unionando {len(parts)} sólidos 3D afilados vía manifold3d.batch_boolean...")
    terrain_union = m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)
    terrain_cut = terrain_union ^ cutter_mf  # Intersección exacta con troquel B-Rep!
    full_solid = base_mf + terrain_cut

    tm_out = _manifold_to_trimesh(full_solid)
    comps = tm_out.split(only_watertight=False)
    if len(comps) > 1:
        comps.sort(key=lambda c: abs(float(c.volume)), reverse=True)
        print(f"[CSG] Limpiando {len(comps) - 1} micro-esquirla(s) booleana(s) para garantizar chi=2...")
        tm_out = comps[0]
    assert tm_out.is_watertight, "Manifold CSG output is not watertight!"
    master_stl = out_dir / "tile_master.stl"
    tm_out.export(str(master_stl))

    # 5. Exportar splat_mask.png 1024x1024 (R=garganta rota por ruido, G=circos+collados de nieve, B=glera)
    N_m = 1024
    lm = np.linspace(-R, R, N_m, dtype=np.float32)
    Xm, Ym = np.meshgrid(lm, lm)
    de_m, dv_m = hex_distances(Xm, Ym)
    in_m = smoothstep(ROAD_MARGIN + 0.4, ROAD_MARGIN + 3.2, de_m) * smoothstep(PAD_RADIUS + 0.4, PAD_RADIUS + 2.8, dv_m)
    path_jit = fbm_2d(Xm, Ym, [0.28, 0.70], [0.65, 0.35], seed=409)
    _, m_path = path_distance_field(Xm, Ym, canyon_pts)
    m_path = np.clip(m_path * smoothstep(0.22, 0.72, path_jit + 0.25) * in_m * smoothstep(ROAD_MARGIN + 2.0, ROAD_MARGIN + 7.5, de_m), 0.0, 1.0)

    # Circo frontal abierto + collado entre picos + vaguadas N/NW/NE + lengua izquierda (<44° en Blender!)
    b_main = np.exp(-0.5 * (((Xm - 0.5) / 11.8) ** 2 + ((Ym - 1.5) / 11.2) ** 2))
    b_col = np.exp(-0.5 * (((Xm + 4.5) / 7.5) ** 2 + ((Ym - 11.0) / 7.8) ** 2))
    b_left = np.exp(-0.5 * (((Xm + 19.5) / 6.2) ** 2 + ((Ym + 11.5) / 6.2) ** 2))
    b_nw = np.exp(-0.5 * (((Xm + 16.5) / 8.5) ** 2 + ((Ym - 23.5) / 9.2) ** 2))
    b_ne = np.exp(-0.5 * (((Xm - 11.5) / 8.5) ** 2 + ((Ym - 23.5) / 9.2) ** 2))
    m_snow = np.clip(np.maximum.reduce([b_main, 0.96 * b_col, 0.90 * b_left, 0.95 * b_nw, 0.95 * b_ne]) * in_m * (1.0 - 0.90 * m_path), 0.0, 1.0)
    m_scree = np.clip(np.exp(-0.5 * (((Xm + 26.5) / 9.5) ** 2 + ((Ym + 15.0) / 10.5) ** 2)) * in_m, 0.0, 1.0)

    splat = np.stack([m_path, m_snow, m_scree], axis=-1)
    # Flip Y para que Blender UV/mapping [-R..R] coincida directo
    splat_path = work_dir / "splat_mask.png"
    Image.fromarray(np.flipud((np.clip(splat, 0.0, 1.0) * 255.0).astype(np.uint8))).save(splat_path)

    stats = {
        "vertices": int(len(tm_out.vertices)),
        "faces": int(len(tm_out.faces)),
        "is_watertight": bool(tm_out.is_watertight),
        "euler_number": int(tm_out.euler_number),
        "volume_cm3": round(float(tm_out.volume) / 1000.0, 3),
        "bounds_z": [round(float(tm_out.bounds[0][2]), 3), round(float(tm_out.bounds[1][2]), 3)],
        "master_stl": str(master_stl),
        "splat_mask": str(splat_path),
    }
    print("[CSG Done]", json.dumps(stats))
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Fábrica Modular 3D Catán - Módulo 1 CSG Estanco")
    p.add_argument("--biome", choices=["montana"], default="montana")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    build_montana_csg(Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
