"""
Micro-relieve geológico procedural en NumPy puro para mallas Hero 360° en Blender:
- Fracturación de granito alpino (crestas asimétricas rígidas y diaclasas verticales/diagonales
  no periódicas, sin anillos horizontales tipo escalera ni hexágonos Voronoi).
- Cornisas/estratos rotos locales muy irregulares y discontinuos, ancho >= 0.26 mm (boquilla 0.2 mm).
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

R = 64.0
Z_BASE = 25.0
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    assert e1 > e0, f"_smoothstep requiere e0 < e1 (recibido {e0}, {e1})"
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hex_distances(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ax = np.abs(x)
    ay = np.abs(y)
    apothem = R * math.sqrt(3.0) * 0.5
    hex_r = np.maximum(ax, 0.5 * ax + 0.5 * math.sqrt(3.0) * ay)
    de = apothem - hex_r
    dv = np.full_like(x, 1e9, dtype=np.float64)
    for k in range(6):
        a = math.radians(30.0 + 60.0 * k)
        dv = np.minimum(dv, np.hypot(x - R * math.cos(a), y - R * math.sin(a)))
    return de, dv


def _value_noise_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int = 11) -> np.ndarray:
    """Ruido trilineal interpolado 3D real por hash de rejilla (cero patrones sinusoidales ni rejillas cruzadas)."""
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    iz = np.floor(z).astype(np.int64)
    fx = x - ix
    fy = y - iy
    fz = z - iz
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    uz = fz * fz * (3.0 - 2.0 * fz)

    def _h(ox: int, oy: int, oz: int) -> np.ndarray:
        h = ((ix + ox) * 73856093) ^ ((iy + oy) * 19349663) ^ ((iz + oz) * 83492791) ^ (seed * 2654435761)
        h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
        return (h.astype(np.float64) / 2147483647.5) - 1.0

    c000, c100 = _h(0, 0, 0), _h(1, 0, 0)
    c010, c110 = _h(0, 1, 0), _h(1, 1, 0)
    c001, c101 = _h(0, 0, 1), _h(1, 0, 1)
    c011, c111 = _h(0, 1, 1), _h(1, 1, 1)
    x00 = c000 + ux * (c100 - c000)
    x10 = c010 + ux * (c110 - c010)
    x01 = c001 + ux * (c101 - c001)
    x11 = c011 + ux * (c111 - c011)
    y0 = x00 + uy * (x10 - x00)
    y1 = x01 + uy * (x11 - x01)
    return y0 + uz * (y1 - y0)


def _ridged_granite_mf(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Multifractal 3D anisotrópico (alargado en Z) basado en hash trilineal real: cero crosshatch."""
    # Domain warp suave para deformar las fracturas verticalmente
    wx = _value_noise_3d(x * 0.25, y * 0.25, z * 0.14, seed=101)
    wy = _value_noise_3d(x * 0.25 + 13.7, y * 0.25 - 9.2, z * 0.14, seed=202)
    px = x + 1.45 * wx
    py = y + 1.45 * wy
    pz = z * 0.34  # alargamiento ~3x vertical típico de espolones y diedros de granito

    # Ridged multifractal de 3 octavas a lacunaridad 2.13 (crestas afiladas sin ondas cruzadas)
    r1 = (1.0 - np.abs(_value_noise_3d(px * 0.42, py * 0.42, pz * 0.42, seed=11))) ** 2.1
    r2 = (1.0 - np.abs(_value_noise_3d(px * 0.91 + 4.1, py * 0.91 - 2.3, pz * 0.91, seed=29))) ** 2.0
    r3 = (1.0 - np.abs(_value_noise_3d(px * 1.93 - 7.4, py * 1.93 + 5.8, pz * 1.93, seed=53))) ** 1.8
    ridge_val = 0.54 * r1 + 0.31 * r2 + 0.15 * r3
    mod_val = _smoothstep(-0.25, 0.45, _value_noise_3d(x * 0.19, y * 0.19, z * 0.12, seed=77))
    return ridge_val, mod_val


def compute_geologic_microstrata_displacement(
    V: np.ndarray,
    N: np.ndarray,
    snow_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Desplazamiento físico (mm) a lo largo de la normal exclusivamente en acantilados (>= 40°):
    - Granito alpino fracturado multifractal 3D trilineal anisotrópico (aristas duras y fisuras).
    - Cero anillos horizontales repetitivos ni rejilla cruzada.
    """
    n_verts = len(V)
    de, dv = _hex_distances(V[:, 0], V[:, 1])
    safe_border = (
        _smoothstep(ROAD_MARGIN + 2.5, ROAD_MARGIN + 4.5, de)
        * _smoothstep(PAD_RADIUS + 2.5, PAD_RADIUS + 4.5, dv)
        * _smoothstep(Z_BASE + 0.45, Z_BASE + 2.2, V[:, 2])
    )
    slope_mask = (1.0 - _smoothstep(0.68, 0.79, N[:, 2])) * _smoothstep(-0.25, 0.02, N[:, 2])

    if snow_mask is None:
        glacier_guess = _smoothstep(43.0, 54.0, V[:, 2]) * _smoothstep(0.55, 0.74, N[:, 2])
        rock_mask = 1.0 - 0.88 * glacier_guess
    else:
        rock_mask = 1.0 - 0.92 * np.clip(snow_mask, 0.0, 1.0)

    mask = safe_border * slope_mask * rock_mask
    active = mask > 1e-4
    disp = np.zeros(n_verts, dtype=np.float64)
    if not np.any(active):
        return disp

    Va = V[active]
    x, y, z = Va[:, 0], Va[:, 1], Va[:, 2]

    ridge_val, band_mod = _ridged_granite_mf(x, y, z)
    # Fractura secundaria asimétrica escalonada para cornisas oblicuas muy discontinuas
    ledge_n = _value_noise_3d(x * 0.28 + 3.0, y * 0.28 - 5.0, z * 0.55, seed=91)
    ledge_step = (np.tanh(2.2 * ledge_n) * 0.048) * band_mod

    granite_disp = (ridge_val - 0.46) * (0.28 + 0.14 * band_mod) + ledge_step
    disp[active] = np.clip(mask[active] * granite_disp, -0.24, 0.16)
    return disp
