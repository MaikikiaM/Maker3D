"""
Escultor volumétrico 3D 100% local en Blender 5.3 para Montaña Centrada (Maker3D 2.0):
  1. Compone 28 monolitos catedralicios 3D facetados + circos cóncavos + placa pradera.
  2. Fusión OpenVDB Voxel Remesh (densidad 3D isotrópica en paredes de 85°).
  3. Suavizado Taubin vectorizado + esculpido anisotrópico 3D en normales (fisuras
     catedralicias, estratos y micro-extraplomos) + tallado del sendero S terracota.
  4. Generación e integración booleana MANIFOLD de ~105 cantos 3D discretos, ~55 matas
     de musgo 3D orgánicas y ~35 guijarros calizos.
  5. Recorte quirúrgico B-Rep (carril 3.50mm + 6 pads Ø20mm) y unión estanca con
     catan_base_brep.stl (imán Ø10.20×5.20mm).
  6. Pintado PBR 3D por vértice (COLOR_0) con cavidad AO, dry-brush y 0.0000% nieve >= 44°.
  7. Renders de auditoría (6 vistas PBR, 6 vistas Arcilla, Orto y 2x2 FRONT/BACK).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import addon_utils
import bmesh
import bpy
import mathutils
import numpy as np

R = 64.0000
APOTHEM = R * math.cos(math.radians(30.0))  # 55.425626 mm
Z_BASE = 25.0000
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00

C30 = math.cos(math.radians(30.0))
S30 = math.sin(math.radians(30.0))


def to_uv_screen(x: np.ndarray | float, y: np.ndarray | float):
    """Marco canónico FRONT (0° = mirando desde az -60°): +u derecha, +v fondo (BACK 180°)."""
    return C30 * x + S30 * y, -S30 * x + C30 * y


def uv_to_world(u: float, v: float) -> Tuple[float, float]:
    """Inversa exacta de to_uv_screen."""
    return C30 * u - S30 * v, S30 * u + C30 * v


def trail_u_of_v(v: np.ndarray | float) -> np.ndarray | float:
    """Eje central del sendero en S desde el borde frontal (v=-50) al circo central (v=-3)."""
    return 3.4 * np.sin((v + 48.0) * 0.092) + 1.2 * np.cos((v + 20.0) * 0.055)


def smoothstep(e0, e1, x):
    lo = np.minimum(e0, e1)
    hi = np.maximum(e0, e1)
    t = np.clip((x - lo) / np.maximum(hi - lo, 1e-8), 0.0, 1.0)
    res = t * t * (3.0 - 2.0 * t)
    return np.where(e0 > e1, 1.0 - res, res)


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


def _noise_3d_trigonometric(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int = 42, octaves: int = 4) -> np.ndarray:
    """Ruido FBM/Ridged 3D totalmente vectorizado en numpy sin dependencias externas."""
    rng = np.random.default_rng(seed)
    acc = np.zeros_like(x, dtype=np.float32)
    amp = 1.0
    freq = 0.24
    norm = 0.0
    for _ in range(octaves):
        # Suma de 3 ondas planas pseudo-aleatorias por octava para isotropía limpia
        w_acc = np.zeros_like(x, dtype=np.float32)
        for _k in range(3):
            dir_v = rng.normal(0.0, 1.0, size=3)
            dir_v /= max(float(np.linalg.norm(dir_v)), 1e-6)
            ph = float(rng.uniform(0.0, 2.0 * math.pi))
            w_acc += np.sin((x * dir_v[0] + y * dir_v[1] + z * dir_v[2]) * freq + ph)
        ridge = 1.0 - np.abs(w_acc / 3.0)
        acc += amp * ridge
        norm += amp
        amp *= 0.52
        freq *= 1.98
    return acc / max(norm, 1e-6)


def _seg_dist(U: np.ndarray, V: np.ndarray, a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Distancia euclidiana y parámetro t∈[0,1] de cada punto (U,V) al segmento 2D a->b."""
    ax, ay = a
    bx, by = b
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby + 1e-8
    t = np.clip(((U - ax) * abx + (V - ay) * aby) / ab2, 0.0, 1.0)
    px = ax + t * abx
    py = ay + t * aby
    return np.hypot(U - px, V - py), t


def _smax(a: np.ndarray, b: np.ndarray, k: float = 1.8) -> np.ndarray:
    """Máximo suave sin sesgo cuando ambos términos están cerca de 0."""
    raw = np.maximum(a, b) + np.log1p(np.exp(-k * np.abs(a - b))) / k
    # Evitar elevar el plano base cuando ambos son ~0
    zero_floor = np.log(2.0) / k
    return np.maximum(0.0, raw - zero_floor * np.exp(-2.0 * np.minimum(a, b)))


def _evaluate_alpine_massif_field(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Campo continuo piramidal unificado de Montaña Centrada (sin conos sueltos):
      - Red de 12 aristas tectónicas conectadas desde una gran corona central dominante
        hacia estribaciones delanteras, laterales y traseras (360° heroico).
      - Circos cóncavos suaves (16°-34°) atrapados entre aristas para nieve alpina ~11-14%.
      - Corredor abierto frontal del sendero en S desde v=-50 hasta v=-3.
    """
    U, V = to_uv_screen(X, Y)
    dist_edge, dist_vert = hex_distances(X, Y)
    collar = smoothstep(ROAD_MARGIN + 0.25, ROAD_MARGIN + 6.5, dist_edge) * smoothstep(PAD_RADIUS + 0.25, PAD_RADIUS + 5.5, dist_vert)

    # 1. Cúpula de pradera oliva suave sobre toda la placa interior (0.2..1.6 mm sobre Z_BASE)
    z_meadow = 0.45 + 0.95 * smoothstep(ROAD_MARGIN + 1.5, ROAD_MARGIN + 14.0, dist_edge) + 0.45 * _noise_3d_trigonometric(X * 0.25, Y * 0.25, np.zeros_like(X), seed=19, octaves=3)

    # 2. Manto alpino base cóncavo entre aristas + relieve fracturado para circos de nieve reales
    r_eff = np.hypot(U * 0.88, (V - 3.5) * 0.92)
    cirque_dome = 9.8 * np.exp(-((r_eff / 29.0) ** 2.1))
    cirque_flanks = (
        3.6 * np.exp(-(((U + 15.0) / 15.0) ** 2 + ((V - 1.5) / 17.0) ** 2))
        + 3.5 * np.exp(-(((U - 15.5) / 15.0) ** 2 + ((V - 2.5) / 17.0) ** 2))
        + 3.4 * np.exp(-(((U + 1.0) / 16.0) ** 2 + ((V - 18.5) / 15.0) ** 2))
    )
    mid_crags = (
        5.2
        * np.maximum(0.0, _noise_3d_trigonometric(X * 0.30, Y * 0.30, np.zeros_like(X) + 3.0, seed=61, octaves=3) - 0.35)
        * np.exp(-((r_eff / 32.0) ** 2.0))
    )
    saddle_field = np.maximum(0.0, cirque_dome + cirque_flanks + mid_crags - 2.0) * smoothstep(10.0, 26.0, dist_edge + 10.0)

    # 3. Red de 19 aristas orgánicas piramidales afiladas + bastiones frontales flanqueando el sendero
    spines = [
        ((-2.5, 6.0), (2.0, 2.5), 40.5, 34.5, 8.8, 7.6),
        ((2.0, 2.5), (12.5, 4.5), 34.0, 31.5, 7.6, 6.8),
        ((-2.5, 6.0), (-14.0, 4.5), 38.5, 29.0, 8.2, 6.8),
        ((-1.5, 6.0), (2.5, 18.0), 37.5, 29.5, 8.2, 7.0),
        ((2.5, 18.0), (-12.5, 25.5), 26.2, 18.8, 6.4, 5.6),
        ((2.5, 18.0), (15.5, 23.5), 26.2, 18.2, 6.4, 5.6),
        ((-12.5, 25.5), (-18.5, 38.0), 18.2, 5.5, 5.6, 4.8),
        ((15.5, 23.5), (15.5, 38.0), 18.0, 5.5, 5.6, 4.8),
        ((1.5, 17.5), (-0.5, 39.5), 25.0, 8.2, 6.0, 5.0),
        ((-14.0, 4.5), (-39.0, -2.0), 27.0, 4.5, 6.6, 5.0),
        ((12.5, 4.5), (39.5, -1.0), 27.5, 4.5, 6.6, 5.0),
        ((-14.0, 4.5), (-30.5, 19.0), 26.0, 5.8, 6.2, 5.0),
        ((12.5, 4.5), (31.5, 17.5), 26.5, 5.8, 6.2, 5.0),
        ((-14.0, 4.5), (-18.5, -16.5), 28.0, 22.0, 7.0, 6.4),
        ((-18.5, -16.5), (-14.5, -39.5), 22.0, 4.5, 6.4, 4.8),
        ((12.5, 4.5), (17.5, -15.0), 28.5, 23.0, 7.2, 6.5),
        ((17.5, -15.0), (15.0, -39.0), 23.0, 4.8, 6.5, 5.0),
        ((-18.5, -16.5), (-37.0, -19.5), 18.5, 3.8, 5.8, 4.6),
        ((17.5, -15.0), (37.0, -19.0), 19.5, 4.0, 5.8, 4.6),
    ]

    ridge_field = np.zeros_like(X, dtype=np.float32)
    # Deformación de dominio centrada en su media analítica (~0.68)
    warp_u = 4.8 * (_noise_3d_trigonometric(X * 0.13, Y * 0.13, np.zeros_like(X), seed=31, octaves=3) - 0.68)
    warp_v = 4.8 * (_noise_3d_trigonometric(X * 0.13, Y * 0.13, np.zeros_like(X) + 5.0, seed=47, octaves=3) - 0.68)
    Uw, Vw = U + warp_u, V + warp_v

    for idx_s, (a, b, ha, hb, wa, wb) in enumerate(spines):
        d, t = _seg_dist(Uw, Vw, a, b)
        knobby = 0.84 + 0.24 * np.sin(t * 7.8 + idx_s * 1.7)
        h_seg = (ha * (1.0 - t) + hb * t) * knobby
        w_seg = wa * (1.0 - t) + wb * t
        u_d = d / np.maximum(w_seg, 1.0)
        prof = h_seg * np.exp(-1.12 * np.power(u_d, 0.92))
        ridge_field = np.maximum(ridge_field, prof)

    rock_macro = np.maximum(saddle_field, ridge_field)

    # 4. Despejar suavemente el cañón del sendero en S (desvanecido orgánico al subir hacia el circo, cero escalón recto)
    u_tr = trail_u_of_v(V)
    d_tr = np.abs(U - u_tr)
    trail_fade = smoothstep(-2.0, -17.0, V) * smoothstep(-53.0, -47.0, V)
    tr_half_w = 3.2 + 0.9 * smoothstep(-48.0, -10.0, V)
    tr_mask = smoothstep(tr_half_w + 4.5, tr_half_w * 0.45, d_tr) * trail_fade
    tr_floor = 0.50 + 10.5 * smoothstep(-50.0, -5.0, V) + 0.7 * _noise_3d_trigonometric(X * 0.35, Y * 0.35, np.zeros_like(X), seed=88, octaves=2)
    rock_macro = rock_macro * (1.0 - tr_mask * 0.88) + np.minimum(rock_macro, tr_floor) * (tr_mask * 0.88)

    rel_h = np.maximum(z_meadow, rock_macro) * collar
    return Z_BASE - 0.25 + rel_h


def _add_summit_overhang_crag(
    bm: bmesh.types.BMesh,
    u0: float,
    v0: float,
    base_z: float,
    height: float,
    rad: float,
    tilt_u: float,
    tilt_v: float,
    yaw_deg: float,
    seed: int,
) -> None:
    """Aguja/espolón angular asimétrico 3D (sin forma de estrella) para romper límite 2.5D en cumbres."""
    rng = np.random.default_rng(seed)
    n_sides = 10
    n_rings = 5
    t_vals = [0.0, 0.28, 0.58, 0.84, 1.0]
    scale_t = [1.32, 0.92, 0.56, 0.25, 0.06]
    yaw = math.radians(yaw_deg)
    aniso = np.array([rng.uniform(1.05, 1.32) if k % 2 == 0 else rng.uniform(0.58, 0.78) for k in range(n_sides)])

    rings: List[List[bmesh.types.BMVert]] = []
    for t, sc in zip(t_vals, scale_t):
        cu = u0 + tilt_u * t
        cv = v0 + tilt_v * t
        cz = base_z + height * t
        ring = []
        for k in range(n_sides):
            ang = 2.0 * math.pi * k / n_sides + yaw
            rr = rad * sc * float(aniso[k])
            wx, wy = uv_to_world(cu + rr * math.cos(ang), cv + rr * math.sin(ang))
            ring.append(bm.verts.new((wx, wy, cz)))
        rings.append(ring)

    bx, by = uv_to_world(u0, v0)
    tx, ty = uv_to_world(u0 + tilt_u, v0 + tilt_v)
    vb = bm.verts.new((bx, by, base_z))
    vt = bm.verts.new((tx, ty, base_z + height))
    for k in range(n_sides):
        kn = (k + 1) % n_sides
        bm.faces.new((vb, rings[0][kn], rings[0][k]))
        for r in range(n_rings - 1):
            bm.faces.new((rings[r][k], rings[r][kn], rings[r + 1][kn], rings[r + 1][k]))
        bm.faces.new((rings[-1][k], rings[-1][kn], vt))


def _create_macro_mountain_core() -> bpy.types.Object:
    """Construye el macizo piramidal continuo + 7 agujas 3D con extraplomo sutil en cumbres."""
    bm = bmesh.new()

    # Malla hexagonal uniforme de alta resolución (n_seg=64 -> ~25k vértices top) cerrando prisma sólido en Z=20.0
    n_seg = 64
    r_cap = (APOTHEM - ROAD_MARGIN * 0.45) / math.cos(math.radians(30.0))
    verts_hex = [(r_cap * math.cos(math.radians(30.0 + 60.0 * k)), r_cap * math.sin(math.radians(30.0 + 60.0 * k))) for k in range(6)]
    top_map: Dict[Tuple[int, int], bmesh.types.BMVert] = {}
    sectors: List[Dict[Tuple[int, int], bmesh.types.BMVert]] = []

    pts_list: List[Tuple[int, int, float, float]] = []
    for k in range(6):
        v1, v2 = verts_hex[k], verts_hex[(k + 1) % 6]
        for i in range(n_seg + 1):
            for j in range(n_seg + 1 - i):
                x = (i / n_seg) * v1[0] + (j / n_seg) * v2[0]
                y = (i / n_seg) * v1[1] + (j / n_seg) * v2[1]
                key = (int(round(x * 1000.0)), int(round(y * 1000.0)))
                if key not in top_map:
                    top_map[key] = None  # placeholder
                    pts_list.append((key[0], key[1], x, y))

    # Evaluar altura del macizo vectorizada de una vez
    xs_a = np.array([p[2] for p in pts_list], dtype=np.float32)
    ys_a = np.array([p[3] for p in pts_list], dtype=np.float32)
    zs_a = _evaluate_alpine_massif_field(xs_a, ys_a)
    for (kx, ky, x, y), z in zip(pts_list, zs_a):
        top_map[(kx, ky)] = bm.verts.new((float(x), float(y), float(z)))

    for k in range(6):
        v1, v2 = verts_hex[k], verts_hex[(k + 1) % 6]
        smap = {}
        for i in range(n_seg + 1):
            for j in range(n_seg + 1 - i):
                x = (i / n_seg) * v1[0] + (j / n_seg) * v2[0]
                y = (i / n_seg) * v1[1] + (j / n_seg) * v2[1]
                smap[(i, j)] = top_map[(int(round(x * 1000.0)), int(round(y * 1000.0)))]
        sectors.append(smap)
        for i in range(n_seg):
            for j in range(n_seg - i):
                a, b, c = smap[(i, j)], smap[(i + 1, j)], smap[(i, j + 1)]
                bm.faces.new((a, b, c))
                if i + j + 1 < n_seg:
                    bm.faces.new((b, smap[(i + 1, j + 1)], c))

    # Cerrar prisma hacia abajo en Z=20.0
    rim_top = [sectors[k][(n_seg - j, j)] for k in range(6) for j in range(n_seg)]
    rim_bot = [bm.verts.new((v.co.x, v.co.y, 20.0)) for v in rim_top]
    bot_c = bm.verts.new((0.0, 0.0, 20.0))
    m_rim = len(rim_top)
    for idx in range(m_rim):
        nxt = (idx + 1) % m_rim
        bm.faces.new((rim_bot[idx], rim_bot[nxt], rim_top[nxt], rim_top[idx]))
        bm.faces.new((bot_c, rim_bot[nxt], rim_bot[idx]))

    # Añadir 6 grandes picos piramidales estriados y enraizados en el macizo
    crown_spires = [
        (-2.2, 6.2, 33.5, 31.5, 10.2, -0.8, 1.0, 18.0, 201),
        (10.5, 4.8, 31.5, 24.5, 8.4, 1.1, 0.4, -35.0, 203),
        (-12.8, 4.8, 31.0, 22.0, 8.2, -1.1, 0.5, 44.0, 204),
        (2.2, 16.5, 31.0, 23.5, 8.6, 0.5, 1.2, -12.0, 205),
        (-17.5, -15.5, 28.5, 17.5, 7.6, -0.7, -0.8, 72.0, 206),
        (16.8, -14.2, 28.5, 18.5, 7.8, 0.8, -0.7, -25.0, 208),
    ]
    for sp in crown_spires:
        _add_summit_overhang_crag(bm, *sp)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new("MountainCoreRaw")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("MountainCore", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _voxel_remesh_object(obj: bpy.types.Object, voxel_size: float = 0.24) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new(name="OpenVDBRemesh", type="REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel_size
    mod.adaptivity = 0.004
    mod.use_smooth_shade = False
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _vectorized_taubin_and_3d_sculpt(obj: bpy.types.Object) -> Dict[str, float]:
    """
    Aplica suavizado Taubin sin contracción + esculpido 3D real en normales (fisuras
    verticales catedralicias + estratos en acantilados >= 45° + tallado del sendero S),
    seguido de cálculo exacto de curvatura 3D y pintado PBR COLOR_0.
    """
    mesh = obj.data
    n_verts = len(mesh.vertices)
    coords = np.empty(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    V = coords.reshape((n_verts, 3)).copy()

    # Leer polígonos (Voxel Remesh genera quads puros) y construir aristas de adyacencia
    n_polys = len(mesh.polygons)
    tris_list = []
    mesh.calc_loop_triangles()
    n_tris = len(mesh.loop_triangles)
    tri_idx = np.empty(n_tris * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri_idx)
    F = tri_idx.reshape((n_tris, 3))

    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e_min = np.minimum(edges[:, 0], edges[:, 1])
    e_max = np.maximum(edges[:, 0], edges[:, 1])
    edges_u = np.unique(np.column_stack([e_min, e_max]), axis=0)
    u_e, v_e = edges_u[:, 0], edges_u[:, 1]

    deg = np.zeros(n_verts, dtype=np.float32)
    np.add.at(deg, u_e, 1.0)
    np.add.at(deg, v_e, 1.0)
    deg = np.maximum(deg, 1.0)

    def neighbor_avg(arr: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(arr, dtype=np.float32)
        np.add.at(acc, u_e, arr[v_e])
        np.add.at(acc, v_e, arr[u_e])
        return acc / deg[:, None]

    def compute_normals(pts: np.ndarray) -> np.ndarray:
        e1 = pts[F[:, 1]] - pts[F[:, 0]]
        e2 = pts[F[:, 2]] - pts[F[:, 0]]
        fn = np.cross(e1, e2)
        vn = np.zeros_like(pts, dtype=np.float32)
        np.add.at(vn, F[:, 0], fn)
        np.add.at(vn, F[:, 1], fn)
        np.add.at(vn, F[:, 2], fn)
        ln = np.linalg.norm(vn, axis=1, keepdims=True)
        return vn / np.maximum(ln, 1e-8)

    # 1. Proteger la cara inferior (z < 22.0) durante todo el esculpido/suavizado
    not_bottom = V[:, 2] > 22.2

    # Suavizado Taubin inicial (3 ciclos) para eliminar cualquier escalón de vóxel sin encoger agujas
    for _ in range(3):
        avg = neighbor_avg(V)
        V[not_bottom] += 0.36 * (avg[not_bottom] - V[not_bottom])
        avg = neighbor_avg(V)
        V[not_bottom] += -0.37 * (avg[not_bottom] - V[not_bottom])

    # Coordenadas canónicas en pantalla FRONT y distancias al hexágono/pads
    X, Y, Z = V[:, 0], V[:, 1], V[:, 2]
    U, Vscr = to_uv_screen(X, Y)
    dist_edge, dist_vert = hex_distances(X, Y)
    collar_in = smoothstep(ROAD_MARGIN + 0.8, ROAD_MARGIN + 3.2, dist_edge) * smoothstep(PAD_RADIUS + 0.6, PAD_RADIUS + 3.0, dist_vert)
    N = compute_normals(V)

    # 2. Tallado suave del sendero en S (solo en faja frontal v in [-51, -3])
    u_tr = trail_u_of_v(Vscr)
    d_tr = np.abs(U - u_tr)
    on_trail_zone = (Vscr >= -52.0) & (Vscr <= -3.0) & (dist_edge > ROAD_MARGIN + 0.2) & (dist_vert > PAD_RADIUS + 0.3)
    trail_fade_3d = smoothstep(-2.0, -16.0, Vscr) * on_trail_zone.astype(np.float32)
    trail_width = 3.0 + 0.9 * smoothstep(-48.0, -8.0, Vscr)
    trail_w = smoothstep(trail_width + 2.5, trail_width * 0.45, d_tr) * trail_fade_3d
    trail_floor_z = 25.35 + 10.2 * smoothstep(-50.0, -5.5, Vscr)
    # Presionar suavemente el suelo de la vaguada sin escalones duros
    drop = np.maximum(0.0, Z - trail_floor_z) * trail_w * 0.72
    V[:, 2] -= drop * not_bottom.astype(np.float32)

    # 3. Esculpido 3D anisotrópico en normales (fisuras catedralicias verticales + estratos)
    N = compute_normals(V)
    X, Y, Z = V[:, 0], V[:, 1], V[:, 2]
    U, Vscr = to_uv_screen(X, Y)
    slope_deg = np.degrees(np.arccos(np.clip(N[:, 2], -1.0, 1.0)))

    # Máscara rocosa para fluting catedralicio vertical y fracturas
    height_rel = np.maximum(0.0, Z - Z_BASE)
    flute_noise = _noise_3d_trigonometric(X * 2.15, Y * 2.15, Z * 0.28, seed=77, octaves=4)
    strata_noise = _noise_3d_trigonometric(X * 0.95, Y * 0.95, Z * 1.95, seed=91, octaves=3)
    rock_activity = (
        smoothstep(1.8, 5.0, height_rel)
        * smoothstep(18.0, 44.0, slope_deg)
        * collar_in
        * (1.0 - trail_w * 0.88)
    )
    # Desplazamiento 3D a lo largo de la normal (fisuras nítidas y estrías sin destruir cuencas)
    disp_mm = rock_activity * ((flute_noise - 0.68) * 3.1 + (strata_noise - 0.68) * 1.15)
    V += N * disp_mm[:, None] * not_bottom[:, None]

    # Ligera relajación selectiva en pradera verde y pista del sendero para contraste frente a roca viva
    soft_zone = not_bottom & ((height_rel < 2.2) | (trail_w > 0.55))
    for _ in range(2):
        avg = neighbor_avg(V)
        V[soft_zone] += 0.25 * (avg[soft_zone] - V[soft_zone])

    # Escribir vértices esculpidos en Blender
    mesh.vertices.foreach_set("co", V.ravel())
    mesh.update()

    # 4. Calcular métricas 3D finales para pintado PBR por vértice (normales + cavidad 3D + pendiente real)
    N_fin = compute_normals(V)
    avg_fin = neighbor_avg(V)
    lap = avg_fin - V
    curvature = np.sum(-lap * N_fin, axis=1)  # positivo = arista viva, negativo = grieta/cuenca cóncava
    curv_norm = np.clip(curvature / 0.16, -1.0, 1.0)

    X, Y, Z = V[:, 0], V[:, 1], V[:, 2]
    U, Vscr = to_uv_screen(X, Y)
    dist_edge, dist_vert = hex_distances(X, Y)
    height_rel = np.maximum(0.0, Z - Z_BASE)
    slope_fin = np.degrees(np.arccos(np.clip(N_fin[:, 2], -1.0, 1.0)))

    # Sendero terracota en S (desvanecido orgánico sin corte recto arriba)
    trail_fade_c = smoothstep(-5.5, -22.0, Vscr)
    on_trail_c = (Vscr >= -52.5) & (Vscr <= -5.5) & (dist_edge > ROAD_MARGIN + 0.1) & (dist_vert > PAD_RADIUS + 0.2)
    trail_p = smoothstep(3.3, 1.4, d_tr) * on_trail_c.astype(np.float32) * trail_fade_c * (1.0 - smoothstep(40.0, 54.0, slope_fin))

    # Ruido mineral de micro-color
    c_noise = _noise_3d_trigonometric(X * 0.9, Y * 0.9, Z * 0.9, seed=313, octaves=3)

    # Paleta fiel a concept_dual_fullres.jpg (espacio lineal sRGB->linear aproximado con **2.2 para Blender/glTF!)
    def srgb_to_lin(rgb_tup):
        arr = np.array(rgb_tup, dtype=np.float32) / 255.0
        return np.power(arr, 2.2)

    col_meadow_dark = srgb_to_lin((72, 90, 38))
    col_meadow_light = srgb_to_lin((110, 128, 54))
    col_ochre_rock = srgb_to_lin((168, 132, 92))
    col_granite = srgb_to_lin((114, 102, 92))
    col_crevice = srgb_to_lin((36, 30, 26))
    col_highlight = srgb_to_lin((210, 194, 172))
    col_trail_mid = srgb_to_lin((198, 128, 78))
    col_trail_deep = srgb_to_lin((146, 86, 48))
    col_snow = srgb_to_lin((244, 247, 253))

    # Base roca: mezcla cálida ocre en estribaciones delanteras + granito frío en cumbres/fondo
    ochre_w = np.clip(smoothstep(18.0, -32.0, Vscr) * (1.0 - smoothstep(14.0, 28.0, height_rel)) * 0.78 + 0.18 * c_noise, 0.0, 1.0)
    rock_col = col_granite[None, :] * (1.0 - ochre_w[:, None]) + col_ochre_rock[None, :] * ochre_w[:, None]

    # Cavidad AO profundo en fisuras (curv_norm < 0) y dry-brush nítido en aristas vivas (curv_norm > 0)
    cav_dark = np.clip(-curv_norm, 0.0, 1.0)
    edge_bright = np.clip(curv_norm, 0.0, 1.0) * smoothstep(2.0, 7.5, height_rel)
    rock_col = rock_col * (1.0 - 0.68 * cav_dark[:, None]) + col_crevice[None, :] * (0.68 * cav_dark[:, None])
    rock_col = rock_col * (1.0 - 0.46 * edge_bright[:, None]) + col_highlight[None, :] * (0.46 * edge_bright[:, None])

    # Pradera oliva en cota baja y baja pendiente (cubre suavemente todo el cinturón exterior hasta los pies del macizo)
    meadow_col = col_meadow_dark[None, :] * (1.0 - c_noise[:, None]) + col_meadow_light[None, :] * c_noise[:, None]
    meadow_w = (1.0 - smoothstep(2.8, 6.2, height_rel)) * (1.0 - smoothstep(24.0, 44.0, slope_fin))
    rgb_v = rock_col * (1.0 - meadow_w[:, None]) + meadow_col * meadow_w[:, None]

    # Sendero terracota en S
    trail_col = col_trail_deep[None, :] * (1.0 - c_noise[:, None]) + col_trail_mid[None, :] * c_noise[:, None]
    rgb_v = rgb_v * (1.0 - trail_p[:, None]) + trail_col * trail_p[:, None]

    # Nieve alpina en collados y cuencas glaciares (~10-14% cobertura) con REGLA DURA: 0.0000% en >= 44.0°
    snow_jitter = 2.8 * (_noise_3d_trigonometric(X * 0.42, Y * 0.42, Z * 0.42, seed=555, octaves=2) - 0.5)
    snow_alt = smoothstep(4.2, 9.0, height_rel + snow_jitter)
    snow_slope = 1.0 - smoothstep(29.0, 43.4, slope_fin)
    snow_pref = snow_alt * snow_slope * (1.0 - smoothstep(0.18, 0.58, curv_norm))
    snow_w = np.where(
        (slope_fin >= 44.0) | (trail_p > 0.25) | (height_rel >= 32.5),
        0.0,
        np.clip((snow_pref - 0.12) * 2.2, 0.0, 1.0),
    )
    rgb_v = rgb_v * (1.0 - snow_w[:, None]) + col_snow[None, :] * snow_w[:, None]

    # Verificar regla dura 0.0000% nieve en >= 44°
    steep_snow_pct = float(np.mean(snow_w[slope_fin >= 44.0] > 0.05) * 100.0) if np.any(slope_fin >= 44.0) else 0.0
    assert steep_snow_pct == 0.0, f"Nieve detectada en >=44°: {steep_snow_pct}%"

    # Escribir Color Attribute por esquina (loop)
    ca = mesh.color_attributes.get("Color") or mesh.color_attributes.new("Color", "BYTE_COLOR", "CORNER")
    mesh.color_attributes.active_color = ca
    loop_verts = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_verts)
    rgba = np.ones((n_verts, 4), dtype=np.float32)
    rgba[:, :3] = np.clip(rgb_v, 0.0, 1.0)
    ca.data.foreach_set("color", rgba[loop_verts].ravel())

    return {
        "sculpted_vertices": int(n_verts),
        "max_z_mm": round(float(Z.max()), 3),
        "snow_coverage_pct": round(float(np.mean(snow_w > 0.45) * 100.0), 2),
        "snow_on_steep_ge44_pct": round(steep_snow_pct, 6),
        "steep_ge44_area_pct": round(float(np.mean(slope_fin >= 44.0) * 100.0), 2),
        "V": V,
        "slope_fin": slope_fin,
    }


def _create_discrete_3d_props(terrain_V: np.ndarray, slope_fin: np.ndarray) -> Tuple[bpy.types.Object, Dict[str, int]]:
    """
    Genera props 3D sólidos discretos (cantos en el sendero S, matas de musgo 3D y
    guijarros calizos) espaciados sin auto-intersecciones y pintados en COLOR_0.
    """
    bm = bmesh.new()
    rng = np.random.default_rng(2026)

    # Submuestra rápida KD-simple del suelo para anclar Z exacto en props
    Xs, Ys, Zs = terrain_V[:, 0], terrain_V[:, 1], terrain_V[:, 2]
    Us, Vs = to_uv_screen(Xs, Ys)

    def terrain_z_at(u_q: float, v_q: float, max_slope: float = 52.0) -> Optional[float]:
        d2 = (Us - u_q) ** 2 + (Vs - v_q) ** 2
        idx = int(np.argmin(d2))
        if d2[idx] > 4.0:
            return None
        if slope_fin[idx] > max_slope:
            return None
        return float(Zs[idx])

    placed_xyr: List[Tuple[float, float, float]] = []
    face_colors: List[Tuple[float, float, float, float]] = []

    def can_place(wx: float, wy: float, r_check: float) -> bool:
        de, dv = hex_distances(np.array([wx], dtype=np.float32), np.array([wy], dtype=np.float32))
        if float(de[0]) < ROAD_MARGIN + 2.2 or float(dv[0]) < PAD_RADIUS + 2.2:
            return False
        return all(math.hypot(wx - ox, wy - oy) >= (r_check + orad + 0.18) for ox, oy, orad in placed_xyr)

    def add_faceted_stone(wx: float, wy: float, wz: float, rx: float, ry: float, rz: float, col_lin: Tuple[float, float, float, float]) -> None:
        res = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
        yaw = float(rng.uniform(0.0, 2.0 * math.pi))
        cy, sy = math.cos(yaw), math.sin(yaw)
        vs = res["verts"]
        for v in vs:
            jitter = float(rng.uniform(0.78, 1.18))
            lx, ly, lz = v.co.x * rx * jitter, v.co.y * ry * jitter, v.co.z * rz * jitter
            rxw = lx * cy - ly * sy
            ryw = lx * sy + ly * cy
            v.co = (wx + rxw, wy + ryw, wz + lz)
        new_faces = {f for v in vs for f in v.link_faces}
        for f in new_faces:
            f.material_index = len(face_colors)
            face_colors.append(col_lin)

    def srgb_lin(r, g, b, mult=1.0):
        return (
            float(((r / 255.0) * mult) ** 2.2),
            float(((g / 255.0) * mult) ** 2.2),
            float(((b / 255.0) * mult) ** 2.2),
            1.0,
        )

    # 1. Cantos rodados 3D a lo largo de los hombros del sendero terracota (~95 cantos profundamente anclados)
    n_cobbles = 0
    for _ in range(950):
        if n_cobbles >= 95:
            break
        vq = float(rng.uniform(-49.5, -6.5))
        side = -1.0 if rng.random() < 0.5 else 1.0
        uq = float(trail_u_of_v(vq)) + side * float(rng.uniform(1.45, 3.85))
        wx, wy = uv_to_world(uq, vq)
        rad = float(rng.uniform(0.45, 1.08))
        if not can_place(wx, wy, rad):
            continue
        zt = terrain_z_at(uq, vq, max_slope=38.0)
        if zt is None or zt > 36.5:
            continue
        placed_xyr.append((wx, wy, rad))
        col = srgb_lin(182, 136, 98, float(rng.uniform(0.84, 1.14)))
        add_faceted_stone(wx, wy, zt - rad * 0.14, rad, rad * float(rng.uniform(0.75, 1.1)), rad * 0.78, col)
        n_cobbles += 1

    # 2. Matas orgánicas de musgo 3D sobre la pradera perimetral (~78 matas sin solapamiento intra-operando)
    n_moss = 0
    for _ in range(950):
        if n_moss >= 78:
            break
        uq = float(rng.uniform(-49.0, 49.0))
        vq = float(rng.uniform(-49.0, 49.0))
        if abs(uq - float(trail_u_of_v(vq))) < 5.0 and vq < -2.0:
            continue
        wx, wy = uv_to_world(uq, vq)
        rad = float(rng.uniform(0.70, 1.25))
        if not can_place(wx, wy, rad):
            continue
        zt = terrain_z_at(uq, vq, max_slope=32.0)
        if zt is None or zt < 25.05 or zt > 30.8:
            continue
        placed_xyr.append((wx, wy, rad))
        col = srgb_lin(52, 82, 22, float(rng.uniform(0.80, 1.18)))
        add_faceted_stone(wx, wy, zt - rad * 0.14, rad, rad * float(rng.uniform(0.80, 1.18)), rad * 0.68, col)
        n_moss += 1

    # 3. Guijarros calizos dispersos en la pradera (~32 piedras profundamente ancladas)
    n_pebbles = 0
    for _ in range(420):
        if n_pebbles >= 32:
            break
        uq = float(rng.uniform(-49.0, 49.0))
        vq = float(rng.uniform(-49.0, 49.0))
        wx, wy = uv_to_world(uq, vq)
        rad = float(rng.uniform(0.45, 0.90))
        if not can_place(wx, wy, rad):
            continue
        zt = terrain_z_at(uq, vq, max_slope=34.0)
        if zt is None or zt < 25.05 or zt > 31.2:
            continue
        placed_xyr.append((wx, wy, rad))
        col = srgb_lin(162, 166, 170, float(rng.uniform(0.85, 1.10)))
        add_faceted_stone(wx, wy, zt - rad * 0.14, rad, rad * 0.85, rad * 0.75, col)
        n_pebbles += 1

    bm.faces.ensure_lookup_table()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new("Props3D")
    bm.to_mesh(mesh)
    bm.free()

    ca = mesh.color_attributes.new("Color", "BYTE_COLOR", "CORNER")
    mesh.color_attributes.active_color = ca
    for poly in mesh.polygons:
        cidx = poly.material_index
        col = face_colors[cidx] if 0 <= cidx < len(face_colors) else (0.2, 0.2, 0.2, 1.0)
        poly.material_index = 0
        for li in poly.loop_indices:
            ca.data[li].color = col

    obj = bpy.data.objects.new("Props3D", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj, {"cobbles": n_cobbles, "moss_cushions": n_moss, "pebbles": n_pebbles}


def _apply_vertex_color_material(obj: bpy.types.Object, mat_name: str = "Mountain_3D_PBR") -> bpy.types.Material:
    mat = bpy.data.materials.new(name=mat_name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        nt.nodes.clear()
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "Color"
    nt.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.82
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.22
    obj.data.materials.append(mat)
    return mat


def _make_solid_material(name: str, rgba: Tuple[float, float, float, float], roughness: float = 0.56) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        nt.nodes.clear()
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.28
    return mat


def _boolean_op(target: bpy.types.Object, tool: bpy.types.Object, operation: str) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    mod = target.modifiers.new(name=f"Bool_{operation}", type="BOOLEAN")
    mod.operation = operation
    mod.object = tool
    solvers = [e.identifier for e in mod.bl_rna.properties["solver"].enum_items]
    mod.solver = "MANIFOLD" if "MANIFOLD" in solvers else "EXACT"
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _setup_lighting(scene: bpy.types.Scene) -> None:
    world = bpy.data.worlds.new("HeroWorld")
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.72, 0.75, 0.78, 1.0)
        bg.inputs["Strength"].default_value = 0.65

    suns = [
        ("KeySun", (60.0, -55.0, 90.0), 3.1, (1.0, 0.97, 0.92), math.radians(8.0)),
        ("FillSun", (-75.0, -25.0, 62.0), 1.1, (0.85, 0.92, 1.0), math.radians(22.0)),
        ("RimSun", (-10.0, 80.0, 52.0), 1.4, (0.98, 0.99, 1.0), math.radians(14.0)),
        ("BotSun", (12.0, -20.0, -90.0), 1.6, (0.94, 0.96, 1.0), math.radians(18.0)),
    ]
    for name, loc, energy, color, angle in suns:
        ld = bpy.data.lights.new(name=name, type="SUN")
        ld.energy = energy
        ld.color = color
        ld.angle = angle
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        d = mathutils.Vector((0.0, 0.0, 28.0)) - mathutils.Vector(loc)
        lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(lo)


def _render_grid(
    scene: bpy.types.Scene,
    cam_obj: bpy.types.Object,
    views: List[Tuple[float, float, float, Tuple[float, float, float]]],
    out_png: Path,
    cols: int,
    rows: int,
    res_px: int,
    overrides: Optional[List[Optional[bpy.types.Material]]] = None,
) -> None:
    scene.render.resolution_x = res_px
    scene.render.resolution_y = res_px
    tiles: List[np.ndarray] = []

    with tempfile.TemporaryDirectory() as td:
        tp_dir = Path(td)
        for idx, (cx, cy, cz, target) in enumerate(views):
            bpy.context.view_layer.material_override = overrides[idx] if overrides else None
            cam_obj.data.type = "PERSP"
            cam_obj.data.lens = 50.0
            cam_obj.location = (cx, cy, cz)
            d = mathutils.Vector(target) - mathutils.Vector((cx, cy, cz))
            cam_obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
            vp = tp_dir / f"v_{idx}.png"
            scene.render.filepath = str(vp)
            bpy.ops.render.render(write_still=True)
            im = bpy.data.images.load(str(vp), check_existing=False)
            tiles.append(np.array(im.pixels[:], dtype=np.float32).reshape((res_px, res_px, 4)))
            bpy.data.images.remove(im)

    bpy.context.view_layer.material_override = None
    row_strips = [np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1) for r in range(rows - 1, -1, -1)]
    canvas = np.concatenate(row_strips, axis=0)
    out_im = bpy.data.images.new(out_png.stem, width=cols * res_px, height=rows * res_px, alpha=True)
    out_im.pixels.foreach_set(canvas.ravel())
    out_im.filepath_raw = str(out_png)
    out_im.file_format = "PNG"
    out_im.save()
    bpy.data.images.remove(out_im)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Escultor volumétrico 100% local Blender 5.3 para Montaña Centrada")
    parser.add_argument("--base-stl", required=True)
    parser.add_argument("--cutter-stl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.38)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args(argv)

    base_stl = Path(args.base_stl).resolve()
    cutter_stl = Path(args.cutter_stl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    try:
        addon_utils.enable("cycles")
    except Exception:
        pass
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    if hasattr(scene.cycles, "use_denoising"):
        try:
            scene.cycles.use_denoising = True
        except Exception:
            pass
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    # 1. Crear núcleo macro + OpenVDB Voxel Remesh + Esculpido 3D en normales + Pintado PBR COLOR_0
    mtn_obj = _create_macro_mountain_core()
    _voxel_remesh_object(mtn_obj, voxel_size=args.voxel_size)
    sculpt_stats = _vectorized_taubin_and_3d_sculpt(mtn_obj)
    mat_mtn = _apply_vertex_color_material(mtn_obj, "Mountain_3D_PBR")

    # 2. Props 3D discretos (cantos en sendero + musgo 3D + guijarros)
    props_obj, props_stats = _create_discrete_3d_props(sculpt_stats["V"], sculpt_stats["slope_fin"])
    props_obj.data.materials.append(mat_mtn)

    # 3. Recorte quirúrgico B-Rep del núcleo esculpido antes de fusionar props/base
    bpy.ops.wm.stl_import(filepath=str(cutter_stl))
    cutter_obj = bpy.context.selected_objects[0]
    _boolean_op(mtn_obj, cutter_obj, operation="INTERSECT")
    bpy.data.objects.remove(cutter_obj, do_unlink=True)

    # 4. Importar base B-Rep exacta con Color Attribute gris carbón mate (#23272D lineal)
    bpy.ops.wm.stl_import(filepath=str(base_stl))
    base_obj = bpy.context.selected_objects[0]
    base_obj.name = "CatanTileHeroMaster"
    plinth_rgba = (float((36 / 255.0) ** 2.2), float((40 / 255.0) ** 2.2), float((46 / 255.0) ** 2.2), 1.0)
    mat_plinth = _make_solid_material("Plinth_Slate", plinth_rgba, roughness=0.52)
    base_ca = base_obj.data.color_attributes.new("Color", "BYTE_COLOR", "CORNER")
    for d in base_ca.data:
        d.color = plinth_rgba
    base_obj.data.color_attributes.active_color = base_ca
    base_obj.data.materials.clear()
    base_obj.data.materials.append(mat_plinth)
    base_obj.data.materials.append(mat_mtn)

    # 5. Unir Base B-Rep + Montaña Esculpida + Props 3D vía MANIFOLD Boolean
    _boolean_op(base_obj, mtn_obj, operation="UNION")
    bpy.data.objects.remove(mtn_obj, do_unlink=True)
    _boolean_op(base_obj, props_obj, operation="UNION")
    bpy.data.objects.remove(props_obj, do_unlink=True)

    # Triangulación, limpieza de caras degeneradas y filtrado BFS a componente conexa principal única
    bm_chk = bmesh.new()
    bm_chk.from_mesh(base_obj.data)
    bm_chk.faces.ensure_lookup_table()
    bmesh.ops.triangulate(bm_chk, faces=list(bm_chk.faces))
    bmesh.ops.dissolve_degenerate(bm_chk, dist=1e-6, edges=list(bm_chk.edges))
    orphan_verts = [v for v in bm_chk.verts if not v.link_faces]
    if orphan_verts:
        bmesh.ops.delete(bm_chk, geom=orphan_verts, context="VERTS")

    visited = set()
    components = []
    for v_start in bm_chk.verts:
        if v_start in visited:
            continue
        comp = []
        stack = [v_start]
        visited.add(v_start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for edge in cur.link_edges:
                other = edge.other_vert(cur)
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        components.append(comp)

    if len(components) > 1:
        largest = max(components, key=len)
        keep_set = set(largest)
        drop_verts = [v for v in bm_chk.verts if v not in keep_set]
        bmesh.ops.delete(bm_chk, geom=drop_verts, context="VERTS")
        print(f"BFS cleaned {len(components) - 1} disconnected micro-components ({len(drop_verts)} verts).")

    bmesh.ops.recalc_face_normals(bm_chk, faces=bm_chk.faces)
    non_man = sum(1 for e in bm_chk.edges if not e.is_manifold)
    boundary = sum(1 for e in bm_chk.edges if e.is_boundary)
    if non_man > 0 or boundary > 0:
        bmesh.ops.remove_doubles(bm_chk, verts=bm_chk.verts, dist=1e-5)
        bmesh.ops.recalc_face_normals(bm_chk, faces=bm_chk.faces)
        non_man = sum(1 for e in bm_chk.edges if not e.is_manifold)
        boundary = sum(1 for e in bm_chk.edges if e.is_boundary)
    connected_components = 1 if len(bm_chk.verts) > 0 else 0
    n_verts_fin = len(bm_chk.verts)
    n_faces_fin = len(bm_chk.faces)
    bm_chk.to_mesh(base_obj.data)
    bm_chk.free()
    base_obj.data.update()

    # 6. Exportar Master STL y Viewer GLB (con Color Attribute COLOR_0 embebido)
    bpy.ops.object.select_all(action="DESELECT")
    base_obj.select_set(True)
    bpy.context.view_layer.objects.active = base_obj
    master_stl = out_dir / "tile_master.stl"
    viewer_glb = out_dir / "tile_viewer.glb"
    bpy.ops.wm.stl_export(filepath=str(master_stl), export_selected_objects=True, ascii_format=False)
    bpy.ops.export_scene.gltf(
        filepath=str(viewer_glb),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_attributes=True,
    )

    # 7. Renders de auditoría (estudio claro neutral estilo concept_dual_fullres.jpg)
    _setup_lighting(scene)
    cam_data = bpy.data.cameras.new("AuditCam")
    cam_obj = bpy.data.objects.new("AuditCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    mat_clay = _make_solid_material("Clay_CAD", (0.58, 0.56, 0.54, 1.0), roughness=0.55)

    # Ángulo FRONT 0° = az -60°, BACK 180° = az 120° (encuadre cercano tipo concept_dual_fullres.jpg)
    fx, fy = 152.0 * math.cos(math.radians(-60.0)), 152.0 * math.sin(math.radians(-60.0))
    bx, by = 152.0 * math.cos(math.radians(120.0)), 152.0 * math.sin(math.radians(120.0))
    views_fb = [
        (fx, fy, 88.0, (0.0, 0.0, 29.0)),
        (bx, by, 88.0, (0.0, 0.0, 29.0)),
        (fx, fy, 88.0, (0.0, 0.0, 29.0)),
        (bx, by, 88.0, (0.0, 0.0, 29.0)),
    ]
    _render_grid(
        scene, cam_obj, views_fb,
        out_dir / "audit_front_back_pbr_clay.png",
        cols=2, rows=2, res_px=640,
        overrides=[None, None, mat_clay, mat_clay],
    )

    views_6 = [
        (185.0 * math.cos(math.radians(-60)), 185.0 * math.sin(math.radians(-60)), 118.0, (0.0, 0.0, 28.0)),
        (185.0 * math.cos(math.radians(60)), 185.0 * math.sin(math.radians(60)), 118.0, (0.0, 0.0, 28.0)),
        (185.0 * math.cos(math.radians(120)), 185.0 * math.sin(math.radians(120)), 118.0, (0.0, 0.0, 28.0)),
        (185.0 * math.cos(math.radians(240)), 185.0 * math.sin(math.radians(240)), 118.0, (0.0, 0.0, 28.0)),
        (0.0, -90.0, 195.0, (0.0, 0.0, 28.0)),
        (15.0, -35.0, -180.0, (0.0, 0.0, 8.0)),
    ]
    _render_grid(scene, cam_obj, views_6, out_dir / "audit_6views_pbr.png", cols=3, rows=2, res_px=480, overrides=None)
    _render_grid(scene, cam_obj, views_6, out_dir / "audit_6views_clay.png", cols=3, rows=2, res_px=480, overrides=[mat_clay] * 6)

    # Ortográfico cenital
    cam_obj.data.type = "ORTHO"
    cam_obj.data.ortho_scale = 138.0
    cam_obj.location = (0.0, 0.0, 180.0)
    cam_obj.rotation_euler = (0.0, 0.0, math.radians(-30.0))
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.filepath = str(out_dir / "audit_ortho.png")
    bpy.ops.render.render(write_still=True)

    xs_f = [v.co.x for v in base_obj.data.vertices]
    ys_f = [v.co.y for v in base_obj.data.vertices]
    zs_f = [v.co.z for v in base_obj.data.vertices]
    out_metrics = {
        "vertices": n_verts_fin,
        "faces": n_faces_fin,
        "non_manifold_edges": non_man,
        "boundary_edges": boundary,
        "connected_components": connected_components,
        "voxel_size_mm": args.voxel_size,
        "max_z_mm": sculpt_stats["max_z_mm"],
        "snow_coverage_pct": sculpt_stats["snow_coverage_pct"],
        "snow_on_steep_ge44_pct": sculpt_stats["snow_on_steep_ge44_pct"],
        "steep_ge44_area_pct": sculpt_stats["steep_ge44_area_pct"],
        "props": props_stats,
        "bounds_mm": {
            "dx": round(max(xs_f) - min(xs_f), 4),
            "dy": round(max(ys_f) - min(ys_f), 4),
            "dz": round(max(zs_f) - min(zs_f), 4),
        },
        "master_stl": str(master_stl),
        "viewer_glb": str(viewer_glb),
    }
    (out_dir / "blender_metrics.json").write_text(json.dumps(out_metrics, indent=2), encoding="utf-8")
    print("HERO_LOCAL_METRICS:", json.dumps(out_metrics))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
