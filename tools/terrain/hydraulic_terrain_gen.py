"""
Motor geológico e hidrológico físico basado en Landlab para losetas Catán y
dioramas 3D. Garantiza drenaje fluvial monótono descendente, máscara estricta
de carreteras/pads Catán y 0.0000% de nieve en pendientes >= 44 grados.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from landlab import RasterModelGrid
from landlab.components import (
    DepressionFinderAndRouter,
    FastscapeEroder,
    FlowAccumulator,
    LinearDiffuser,
)

R = 64.0000
APOTHEM = R * math.cos(math.radians(30.0))  # 55.425626 mm
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00
Z_BASE = 25.0000


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hex_distances(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normals_hex = [[math.cos(k * math.pi / 3.0), math.sin(k * math.pi / 3.0)] for k in range(6)]
    dist_edge = np.full_like(X, 999.0, dtype=np.float32)
    for nxh, nyh in normals_hex:
        dist_edge = np.minimum(dist_edge, APOTHEM - (X * nxh + Y * nyh))

    v_angles = [(30.0 + 60.0 * k) * math.pi / 180.0 for k in range(6)]
    dist_vert = np.full_like(X, 999.0, dtype=np.float32)
    for a in v_angles:
        vx, vy = R * math.cos(a), R * math.sin(a)
        dist_vert = np.minimum(dist_vert, np.hypot(X - vx, Y - vy))
    return dist_edge, dist_vert


def _fbm_field(shape: Tuple[int, int], seed: int, sigmas=(18.0, 7.0, 2.5), weights=(0.60, 0.28, 0.12)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros(shape, dtype=np.float32)
    for s, w in zip(sigmas, weights):
        noise = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
        f = ndi.gaussian_filter(noise, sigma=s)
        f = (f - f.mean()) / max(float(f.std()), 1e-6)
        acc += w * f
    return acc


def initial_relief(
    preset: str,
    X: np.ndarray,
    Y: np.ndarray,
    dist_edge: np.ndarray,
    dist_vert: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, float, float]:
    """Genera topografía tectónica inicial y parámetros físicos (k_sp, diffusivity)."""
    shape = X.shape
    fbm = _fbm_field(shape, seed=seed, sigmas=(16.0, 6.0, 2.2), weights=(0.55, 0.30, 0.15))
    fbm2 = _fbm_field(shape, seed=seed + 37, sigmas=(9.0, 3.5, 1.4), weights=(0.55, 0.30, 0.15))
    ridge_noise = 1.0 - np.abs(_fbm_field(shape, seed=seed + 17, sigmas=(10.0, 4.0, 1.6), weights=(0.55, 0.30, 0.15)))
    r_norm = np.clip(np.hypot(X, Y) / APOTHEM, 0.0, 1.2)

    if preset == "alpine_canyon":
        # Valle fluvial sinuoso en S flanqueado por dos macizos escarpados con contrafuertes y praderas graduales
        canyon_x = 7.5 * np.sin(Y / 17.0 + 0.35)
        dist_river = np.abs(X - canyon_x)
        # Envolvente que muere suavemente hacia la carretera y el valle (evita muros artificiales en el borde)
        rim_taper = smoothstep(ROAD_MARGIN + 2.5, ROAD_MARGIN + 19.0, dist_edge) * smoothstep(PAD_RADIUS + 2.0, PAD_RADIUS + 14.0, dist_vert)
        valley_profile = smoothstep(3.0, 26.0, dist_river)
        east_peak = np.exp(-(((X - 21.0) / 14.0) ** 2 + ((Y - 6.0) / 18.0) ** 2))
        west_peak = np.exp(-(((X + 22.0) / 14.0) ** 2 + ((Y + 2.0) / 19.0) ** 2))
        north_crest = np.exp(-(((X - 2.0) / 18.0) ** 2 + ((Y - 22.0) / 12.0) ** 2)) * 0.65
        spurs = 0.45 + 0.65 * np.maximum(ridge_noise, 0.0)
        ns_slope = 0.35 + 0.65 * np.clip((Y + APOTHEM) / (2.0 * APOTHEM), 0.0, 1.0)
        massifs = (27.0 * np.maximum(east_peak, west_peak * 0.90) + 17.0 * north_crest) * spurs
        z = (3.8 * valley_profile * ns_slope * rim_taper + massifs * smoothstep(2.0, 13.0, dist_river) + 2.2 * np.maximum(fbm2 + 0.6, 0.0) * rim_taper)
        z += 2.0 * np.clip((Y + APOTHEM) / (2.0 * APOTHEM), 0.0, 1.0) * rim_taper
        return z.astype(np.float64), 2.2e-4, 0.14
    elif preset == "cordillera":
        axis = (0.72 * X + 0.69 * Y)
        spine = np.exp(-(axis / 18.0) ** 2) * (0.55 + 0.65 * ridge_noise)
        z = 26.0 * spine + 3.5 * np.maximum(fbm + 1.0, 0.0)
        return z.astype(np.float64), 2.5e-4, 0.09
    elif preset == "terrace_hills":
        domes = np.maximum(0.0, 1.0 - r_norm ** 2) * (0.6 + 0.4 * fbm)
        terraced = domes * 12.0 + 1.6 * np.sin(domes * 6.0)
        z = np.maximum(0.2, terraced + 2.0 * (fbm + 1.0))
        return z.astype(np.float64), 1.6e-4, 0.14
    else:  # river_delta
        meander = 9.0 * np.sin(Y / 14.0)
        banks = smoothstep(4.5, 19.0, np.abs(X - meander))
        z = 8.5 * banks * (0.55 + 0.45 * np.clip((Y + APOTHEM) / (2.0 * APOTHEM), 0.0, 1.0))
        z += 2.2 * np.clip((Y + APOTHEM) / (2.0 * APOTHEM), 0.0, 1.0) + 1.4 * np.maximum(fbm, -0.5)
        return z.astype(np.float64), 4.0e-4, 0.10


def simulate_terrain(
    preset: str = "alpine_canyon",
    nx: int = 384,
    max_height_mm: float = 32.0,
    steps: int = 35,
    seed: int = 42,
) -> Dict[str, np.ndarray | Dict]:
    # Simular hidrología Landlab en malla óptima (<= 160x160) para O(N) veloz y
    # reescalar bicúbicamente (order=3) a --nx antes de calcular pendientes y máscaras.
    sim_nx = min(nx, 144)
    xs_s = np.linspace(-R, R, sim_nx, dtype=np.float32)
    dx_s = float(xs_s[1] - xs_s[0])
    Xs, Ys = np.meshgrid(xs_s, xs_s)
    dist_edge_s, dist_vert_s = hex_distances(Xs, Ys)

    grid = RasterModelGrid((sim_nx, sim_nx), xy_spacing=(dx_s, dx_s))
    z_init, k_sp, diff_d = initial_relief(preset, Xs, Ys, dist_edge_s, dist_vert_s, seed)
    collar_s = smoothstep(ROAD_MARGIN, ROAD_MARGIN + 4.5, dist_edge_s) * smoothstep(PAD_RADIUS, PAD_RADIUS + 4.0, dist_vert_s)
    interior_s = dist_edge_s > (ROAD_MARGIN + 0.5)
    z_init = np.maximum(z_init, 0.0) * collar_s
    grid.add_field("topographic__elevation", z_init.ravel(), at="node")

    grid.set_closed_boundaries_at_grid_edges(right_is_closed=False, top_is_closed=False, left_is_closed=False, bottom_is_closed=False)

    fa = FlowAccumulator(grid, flow_director="D8")
    dfr = DepressionFinderAndRouter(grid)
    sp = FastscapeEroder(grid, K_sp=k_sp, m_sp=0.5, n_sp=1.0)
    ld = LinearDiffuser(grid, linear_diffusivity=diff_d)

    dt = 25.0
    # Levantamiento tectónico proporcional al relieve (nunca meseta plana uniforme)
    uplift_shape = np.clip(z_init / max(float(np.max(z_init)), 1.0), 0.0, 1.0) * collar_s
    uplift_rate = 0.014 if preset in ("alpine_canyon", "cordillera") else 0.007

    for step_idx in range(steps):
        z_vec = grid.at_node["topographic__elevation"]
        z_vec[:] += (uplift_shape.ravel() * uplift_rate * dt)
        fa.run_one_step()
        if step_idx % 5 == 0 or step_idx == steps - 1:
            dfr.map_depressions()
        sp.run_one_step(dt=dt)
        ld.run_one_step(dt=dt)

    fa.run_one_step()
    dfr.map_depressions()

    z_sim_s = np.maximum(grid.at_node["topographic__elevation"].reshape((sim_nx, sim_nx)).astype(np.float32), 0.0)
    drainage_s = grid.at_node["drainage_area"].reshape((sim_nx, sim_nx)).astype(np.float32)

    # Reescalado bicúbico (order=3) a la malla objetivo --nx
    if sim_nx != nx:
        zoom_f = nx / float(sim_nx)
        z_sim = np.maximum(ndi.zoom(z_sim_s, zoom_f, order=3, mode="nearest"), 0.0).astype(np.float32)
        drainage = np.maximum(ndi.zoom(drainage_s, zoom_f, order=1, mode="nearest"), 0.0).astype(np.float32)
    else:
        z_sim, drainage = z_sim_s, drainage_s

    xs = np.linspace(-R, R, nx, dtype=np.float32)
    ys = np.linspace(-R, R, nx, dtype=np.float32)
    dx = float(xs[1] - xs[0])
    X, Y = np.meshgrid(xs, ys)
    dist_edge, dist_vert = hex_distances(X, Y)
    interior_hex = dist_edge > (ROAD_MARGIN + 0.5)

    # Collar suave Catán: exactamente 0.000 mm en carril de carretera (<=3.5mm) y pads (<=10.0mm)
    collar = smoothstep(ROAD_MARGIN, ROAD_MARGIN + 5.5, dist_edge) * smoothstep(PAD_RADIUS, PAD_RADIUS + 4.5, dist_vert)
    z_clean = np.maximum(z_sim, 0.0) * collar

    # Escalar altura útil al máximo solicitado respetando proporciones reales
    peak = float(np.max(z_clean))
    if peak > 1e-5:
        z_clean = z_clean * (max_height_mm / peak)

    # Identificar talweg principal exclusivamente dentro del cañón central (evita bordes del hexágono)
    canyon_center_x = 7.5 * np.sin(Y / 17.0 + 0.35) if preset in ("alpine_canyon", "river_delta") else np.zeros_like(X)
    in_canyon_corridor = (np.abs(X - canyon_center_x) < 12.0) & (collar > 0.48)
    flow_thresh = float(np.percentile(drainage[interior_hex], 90.0))
    river_core = (drainage >= flow_thresh) & in_canyon_corridor & (z_clean < 0.50 * max_height_mm)
    thalweg_cols = []
    for row in range(nx - 1, -1, -1):
        valid = in_canyon_corridor[row] & (river_core[row] if np.any(river_core[row]) else (np.abs(X[row] - canyon_center_x[row]) < 4.0))
        if np.any(valid):
            col = int(np.argmin(z_clean[row] + np.where(valid, 0.0, 1e6)))
            thalweg_cols.append((row, col))
    if len(thalweg_cols) >= 4:
        # Enforzar monotonicidad suave N -> S en el talweg con transición transversal amplia
        prev_z = float(z_clean[thalweg_cols[0]])
        half_w = max(4, int(round(3.2 / dx)))
        for row, col in thalweg_cols[1:]:
            target_z = min(float(z_clean[row, col]), prev_z)
            c0, c1 = max(0, col - half_w), min(nx, col + half_w + 1)
            cols_idx = np.arange(c0, c1)
            w = np.exp(-0.5 * ((cols_idx - col) / max(half_w * 0.50, 1.0)) ** 2)
            # Solo rebajar suavemente si el punto actual excede prev_z
            if z_clean[row, col] > prev_z:
                z_clean[row, c0:c1] = np.minimum(
                    z_clean[row, c0:c1],
                    z_clean[row, c0:c1] * (1.0 - w) + target_z * w,
                )
            prev_z = float(z_clean[row, col])

    # Recalcular gradiente y pendiente real en grados
    gy, gx = np.gradient(z_clean, dx, dx)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    # Máscaras geológicas físicas (nieve irregular en crestas altas < 44°)
    core_vals = z_clean[collar > 0.40]
    snow_alt_thresh = max(0.56 * max_height_mm, float(np.percentile(core_vals, 78.0)) if len(core_vals) else 0.65 * max_height_mm)
    snow_jitter = 4.2 * _fbm_field((nx, nx), seed + 55, sigmas=(9.0, 3.2), weights=(0.65, 0.35))
    snow_mask = (
        ((z_clean + snow_jitter) >= snow_alt_thresh)
        & (z_clean >= 0.46 * max_height_mm)
        & (slope_deg < 44.0)
        & (collar > 0.45)
    )
    snow_f = ndi.gaussian_filter(snow_mask.astype(np.float32), sigma=1.1)
    # Regla dura: 0.0000% de nieve en pendientes >= 44.0 grados
    snow_f = np.where(slope_deg >= 44.0, 0.0, snow_f)
    snow_bin = (snow_f >= 0.45) & (slope_deg < 44.0)

    # Río acotado estrictamente al canal central del valle (cero halos en bordes del hexágono)
    thalweg_mask = np.zeros((nx, nx), dtype=bool)
    for row, col in thalweg_cols:
        thalweg_mask[row, col] = True
    near_thalweg = ndi.binary_dilation(thalweg_mask, iterations=max(4, int(round(2.4 / dx))))
    flow_norm = np.log1p(np.maximum(drainage, 0.0))
    if float(np.max(flow_norm)) > 1e-5:
        flow_norm /= float(np.max(flow_norm))
    flow_bin = near_thalweg & in_canyon_corridor & (z_clean < 0.24 * max_height_mm) & (collar > 0.42)
    flow_channel = ndi.binary_dilation(flow_bin, iterations=1) & in_canyon_corridor & (z_clean < 0.25 * max_height_mm) & (collar > 0.38)

    rock_bin = ((slope_deg >= 29.0) | (z_clean >= 0.42 * max_height_mm)) & (~snow_bin) & (~flow_channel) & (collar > 0.05)
    meadow_bin = (collar > 0.05) & (~snow_bin) & (~rock_bin) & (~flow_channel)

    # Composición de albedo PBR coherente (sin sombras duras arrojadas)
    rgb = np.zeros((nx, nx, 3), dtype=np.float32)
    collar_slate = np.array([0.21, 0.23, 0.26], dtype=np.float32)
    meadow_c = np.array([0.22, 0.43, 0.17], dtype=np.float32)
    dry_grass_c = np.array([0.36, 0.46, 0.21], dtype=np.float32)
    rock_c = np.array([0.46, 0.42, 0.38], dtype=np.float32)
    cliff_c = np.array([0.29, 0.27, 0.26], dtype=np.float32)
    snow_c = np.array([0.93, 0.95, 0.98], dtype=np.float32)
    river_c = np.array([0.13, 0.36, 0.54], dtype=np.float32)

    alt_norm = np.clip(z_clean / max(max_height_mm, 1.0), 0.0, 1.0)
    micro_tex = 0.90 + 0.16 * np.clip(_fbm_field((nx, nx), seed + 99, sigmas=(4.0, 1.5), weights=(0.6, 0.4)), -1.0, 1.0)
    grass_blend = meadow_c * (1.0 - alt_norm[..., None]) + dry_grass_c * alt_norm[..., None]
    rgb[:] = grass_blend * micro_tex[..., None]

    rock_w = np.maximum(smoothstep(24.0, 39.0, slope_deg), smoothstep(0.36, 0.58, alt_norm) * 0.92)
    cliff_w = smoothstep(44.0, 66.0, slope_deg)
    base_rock = rock_c * (1.0 - cliff_w[..., None]) + cliff_c * cliff_w[..., None]
    rgb = rgb * (1.0 - rock_w[..., None]) + (base_rock * micro_tex[..., None]) * rock_w[..., None]

    river_w = np.clip(ndi.gaussian_filter(flow_channel.astype(np.float32), sigma=1.4) * 1.2, 0.0, 1.0)
    rgb = rgb * (1.0 - river_w[..., None]) + river_c * river_w[..., None]

    snow_w = np.where(slope_deg >= 44.0, 0.0, np.clip(snow_f, 0.0, 1.0))
    rgb = rgb * (1.0 - snow_w[..., None]) + snow_c * snow_w[..., None]

    # Zócalo/caminos en gris pizarra limpio
    border_w = 1.0 - smoothstep(0.02, 0.18, collar)
    rgb = rgb * (1.0 - border_w[..., None]) + collar_slate * border_w[..., None]
    albedo_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)

    # Chequeos métricos
    snow_on_steep_pct = float(np.mean(snow_bin[slope_deg >= 44.0]) * 100.0) if np.any(slope_deg >= 44.0) else 0.0
    road_zone = (dist_edge <= ROAD_MARGIN) | (dist_vert <= PAD_RADIUS)
    max_road_error_mm = float(np.max(np.abs(z_clean[road_zone])))

    thalweg_monotonic = True
    if len(thalweg_cols) >= 4:
        tz = [float(z_clean[r, c]) for r, c in thalweg_cols]
        thalweg_monotonic = all(tz[i + 1] <= tz[i] + 1e-4 for i in range(len(tz) - 1))

    metrics = {
        "preset": preset,
        "nx": nx,
        "dx_mm": round(dx, 4),
        "max_height_mm": round(float(np.max(z_clean)), 4),
        "max_slope_deg": round(float(np.max(slope_deg)), 2),
        "snow_on_steep_ge44_pct": round(snow_on_steep_pct, 6),
        "max_road_pad_deviation_mm": round(max_road_error_mm, 6),
        "thalweg_monotonic_downhill": thalweg_monotonic,
        "thalweg_points": len(thalweg_cols),
    }

    assert snow_on_steep_pct == 0.0, f"Nieve detectada en >=44°: {snow_on_steep_pct}%"
    assert max_road_error_mm < 1e-3, f"Carril/pad deformado: {max_road_error_mm} mm"
    assert thalweg_monotonic, "Talweg no monótono descendente"

    return {
        "dem": z_clean,
        "slope_deg": slope_deg,
        "flow_bin": flow_bin,
        "snow_bin": snow_bin,
        "rock_bin": rock_bin,
        "meadow_bin": meadow_bin,
        "albedo_u8": albedo_u8,
        "metrics": metrics,
    }


def _save_uv_png(arr: np.ndarray, path: Path, mode: str = "RGB") -> None:
    """Voltea verticalmente para que fila 0 de la imagen corresponda a Y=+R (v=1)."""
    Image.fromarray(np.flipud(arr), mode=mode).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generador geomorfológico Landlab para Maker3D 2.0")
    parser.add_argument("--preset", choices=["alpine_canyon", "cordillera", "terrace_hills", "river_delta"], default="alpine_canyon")
    parser.add_argument("--nx", type=int, default=384)
    parser.add_argument("--max-height", type=float, default=32.0)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = simulate_terrain(
        preset=args.preset,
        nx=args.nx,
        max_height_mm=args.max_height,
        steps=args.steps,
        seed=args.seed,
    )
    dem = res["dem"]
    np.save(out_dir / "dem_float32.npy", dem.astype(np.float32))
    np.save(out_dir / "slope_deg.npy", res["slope_deg"].astype(np.float32))

    # PNGs (flipped Y-up para corresponder directamente a UVs (x+R)/(2R), (y+R)/(2R))
    peak = max(float(np.max(dem)), 1e-5)
    dem16 = np.clip((dem / peak) * 65535.0, 0, 65535).astype(np.uint16)
    Image.fromarray(np.flipud(dem16)).save(out_dir / "dem_16bit.png")
    _save_uv_png(res["albedo_u8"], out_dir / "albedo_base.png", mode="RGB")
    _save_uv_png((res["flow_bin"].astype(np.uint8) * 255), out_dir / "flow_mask.png", mode="L")
    _save_uv_png((res["snow_bin"].astype(np.uint8) * 255), out_dir / "snow_mask.png", mode="L")
    _save_uv_png((res["rock_bin"].astype(np.uint8) * 255), out_dir / "rock_mask.png", mode="L")
    _save_uv_png((res["meadow_bin"].astype(np.uint8) * 255), out_dir / "meadow_mask.png", mode="L")

    metrics_path = out_dir / "terrain_metrics.json"
    metrics_path.write_text(json.dumps(res["metrics"], indent=2), encoding="utf-8")
    print(json.dumps(res["metrics"], indent=2))


if __name__ == "__main__":
    main()
