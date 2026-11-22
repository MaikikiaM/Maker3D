"""
Colocación geométrica de derrubio / pedregal real (rocas CC0 de Poly Haven) en el valle de Catán
con muestreo Poisson-disk ponderado al pie de acantilados (pendiente <= 28°, nunca en paredes >= 35°),
diversidad vecinal, entierro 35–50% y unión booleana por lotes en Manifold3D estanco (chi == 2).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import manifold3d as m3d
import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from blender.catan_3d_factory_csg import _manifold_to_trimesh, _trimesh_to_manifold

R = 64.0
Z_BASE = 25.0
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00


def _hex_distances_arr(x: np.ndarray, y: np.ndarray):
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


def _random_rot_matrix_aligned(normal: np.ndarray, max_tilt_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Matriz de rotación 3x3 con giro aleatorio completo y eje Z alineado a normal ± max_tilt_deg."""
    z_axis = np.asarray(normal, dtype=np.float64)
    z_axis /= max(np.linalg.norm(z_axis), 1e-8)
    # Desviar levemente de la normal para evitar uniformidad
    tilt = math.radians(rng.uniform(0.0, max_tilt_deg))
    phi = rng.uniform(0.0, 2.0 * math.pi)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(z_axis, ref); t1 /= np.linalg.norm(t1)
    t2 = np.cross(z_axis, t1)
    up_t = math.cos(tilt) * z_axis + math.sin(tilt) * (math.cos(phi) * t1 + math.sin(phi) * t2)
    up_t /= np.linalg.norm(up_t)

    # Yaw completo aleatorio alrededor de up_t
    yaw = rng.uniform(0.0, 2.0 * math.pi)
    # Mezclar ocasionalmente con un giro completo extra en pitch/roll suave (±25°) para no parecer "plantadas derechas"
    x_dir = math.cos(yaw) * t1 + math.sin(yaw) * t2
    x_dir -= np.dot(x_dir, up_t) * up_t
    x_dir /= np.linalg.norm(x_dir)
    y_dir = np.cross(up_t, x_dir)
    rot = np.column_stack([x_dir, y_dir, up_t])
    # Elegir eje local aleatorio de la roca para evitar misma cara arriba siempre
    perm_idx = int(rng.integers(0, 6))
    perms = [
        np.eye(3),
        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
    ]
    return rot @ perms[perm_idx]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-stl", required=True)
    p.add_argument("--rocks-catalog", default="Tableros-Catan/assets/polyhaven/rocks/rocks_catalog.json")
    p.add_argument("--out-stl", required=True)
    p.add_argument("--out-manifest", required=True)
    p.add_argument("--target-count", type=int, default=190)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    cat = json.loads(Path(args.rocks_catalog).read_text(encoding="utf-8"))
    # Filtrar lajas demasiado finas (< 0.35 extensión relativa, ej. moon_rock_04)
    rock_ids = sorted(k for k, v in cat.items() if min(v.get("extents_unit", [0.0])) >= 0.35)
    if not rock_ids:
        raise RuntimeError(f"Catálogo sin rocas volumétricas válidas en {args.rocks_catalog}")

    # Cargar rocas sólidas unitarias
    rock_meshes: Dict[str, trimesh.Trimesh] = {}
    for rid in rock_ids:
        tm_r = trimesh.load(cat[rid]["stl"], force="mesh")
        rock_meshes[rid] = tm_r

    print(f"[Talus] Cargando base {args.base_stl}...")
    base_tm = trimesh.load(args.base_stl, force="mesh")
    c = base_tm.triangles_center
    n = base_tm.face_normals
    a = base_tm.area_faces
    de, dv = _hex_distances_arr(c[:, 0], c[:, 1])

    # Zonas candidatas de valle bajo: pendiente <= 24° (nz >= cos(24°)=0.9135), fuera de lenguas de nieve (z <= 34.5 mm) y bordes
    valley_mask = (
        (n[:, 2] >= math.cos(math.radians(24.0)))
        & (c[:, 2] >= Z_BASE + 0.08)
        & (c[:, 2] <= Z_BASE + 9.5)
        & (de >= ROAD_MARGIN + 3.0)
        & (dv >= PAD_RADIUS + 3.0)
    )
    # Puntos de rotura de pendiente (pie de acantilados: 30°–50°)
    toe_mask = (
        (n[:, 2] <= math.cos(math.radians(30.0)))
        & (n[:, 2] >= math.cos(math.radians(52.0)))
        & (c[:, 2] >= Z_BASE + 1.0)
        & (c[:, 2] <= Z_BASE + 14.0)
        & (de >= ROAD_MARGIN + 2.5)
        & (dv >= PAD_RADIUS + 2.5)
    )

    val_idx = np.where(valley_mask)[0]
    toe_pts = c[toe_mask]
    if len(val_idx) == 0:
        raise RuntimeError("No se encontraron caras candidatas de valle para derrubio.")

    # Submuestrear pool de 15,000 candidatos ponderados por área
    probs_area = a[val_idx] / np.sum(a[val_idx])
    pool_size = min(16000, len(val_idx))
    pool_idx = rng.choice(val_idx, size=pool_size, replace=False, p=probs_area)
    cand_pts = c[pool_idx]
    cand_nrm = n[pool_idx]

    # Distancia al pie de acantilado más cercano (chunks vectorizados)
    from scipy.spatial import cKDTree
    toe_tree = cKDTree(toe_pts[:, :2])
    d_cliff, _ = toe_tree.query(cand_pts[:, :2], k=1)
    # Peso máximo en rotura de pendiente [0..6 mm], decae hacia el valle abierto
    w_cand = np.exp(-d_cliff / 4.8) + 0.16
    order = rng.choice(pool_size, size=pool_size, replace=False, p=w_cand / np.sum(w_cand))

    placed: List[Dict[str, Any]] = []
    placed_xy = []
    placed_r = []
    placed_ids = []

    for ci in order:
        if len(placed) >= args.target_count:
            break
        pt = cand_pts[ci]
        nm = cand_nrm[ci]
        dc = float(d_cliff[ci])

        # Tamaño natural: bloques medianos junto a escarpes (2.2..4.2 mm), piedras bajas en pradera (1.4..2.8 mm)
        if dc < 5.5:
            major_mm = float(rng.uniform(2.2, 4.2))
        else:
            major_mm = float(rng.uniform(1.4, 2.8))
        eff_r = 0.52 * major_mm

        # Comprobar solapamiento Poisson-disk
        if placed_xy:
            pxy = np.asarray(placed_xy)
            pr = np.asarray(placed_r)
            dists = np.hypot(pxy[:, 0] - pt[0], pxy[:, 1] - pt[1])
            if np.any(dists < 0.85 * (pr + eff_r)):
                continue
            # Diversidad: evitar misma roca base que los vecinos cercanos (< 11 mm)
            near_mask = dists < 11.0
            near_ids = {placed_ids[j] for j in np.where(near_mask)[0]}
            avail = [rid for rid in rock_ids if rid not in near_ids] or rock_ids
        else:
            avail = rock_ids

        rid = avail[int(rng.integers(0, len(avail)))]
        sc_xyz = major_mm * rng.uniform(0.85, 1.15, size=3)
        # Garantizar espesor mínimo físico >= 0.60 mm para boquilla 0.2 mm
        min_ext = np.min(np.asarray(cat[rid]["extents_unit"]) * sc_xyz)
        if min_ext < 0.62:
            sc_xyz *= (0.62 / max(min_ext, 1e-4))
            eff_r = 0.52 * float(np.max(np.asarray(cat[rid]["extents_unit"]) * sc_xyz))

        rot_m = _random_rot_matrix_aligned(nm, max_tilt_deg=15.0, rng=rng)
        # Enterrar 45–55% exactos desde la cota mínima proyectada sobre la normal
        bury_frac = float(rng.uniform(0.45, 0.55))
        tm_unit = rock_meshes[rid]
        V_loc = tm_unit.vertices * sc_xyz[None, :]
        V_rot = V_loc @ rot_m.T
        proj_n = V_rot @ nm
        min_n = float(proj_n.min())
        proj_h = float(proj_n.max() - min_n)
        center = pt - nm * (min_n + bury_frac * proj_h)
        if center[2] < Z_BASE + 0.15:
            continue

        placed.append({
            "index": len(placed),
            "rock_id": rid,
            "diff_texture": cat[rid]["diff_texture"],
            "center_xyz": [round(float(v), 4) for v in center],
            "scale_xyz": [round(float(v), 4) for v in sc_xyz],
            "rot_3x3": [[round(float(x), 6) for x in row] for row in rot_m],
            "cliff_dist_mm": round(dc, 3),
        })
        placed_xy.append(pt[:2].tolist())
        placed_r.append(eff_r)
        placed_ids.append(rid)

    print(f"[Talus] Seleccionadas {len(placed)} instancias estancas diversas; uniendo vía Manifold3D...")

    def _to_mf(tm: trimesh.Trimesh) -> m3d.Manifold:
        return m3d.Manifold(
            m3d.Mesh(
                vert_properties=np.asarray(tm.vertices, dtype=np.float32),
                tri_verts=np.asarray(tm.faces, dtype=np.uint32),
            )
        )

    def _build_rock_mf(inst: Dict[str, Any]) -> m3d.Manifold:
        tm_u = rock_meshes[inst["rock_id"]]
        sc = np.asarray(inst["scale_xyz"], dtype=np.float64)
        rot = np.asarray(inst["rot_3x3"], dtype=np.float64)
        ctr = np.asarray(inst["center_xyz"], dtype=np.float64)
        V_w = (tm_u.vertices * sc[None, :]) @ rot.T + ctr[None, :]
        return m3d.Manifold(
            m3d.Mesh(
                vert_properties=np.asarray(V_w, dtype=np.float32),
                tri_verts=np.asarray(tm_u.faces, dtype=np.uint32),
            )
        )

    base_mf = _to_mf(base_tm).as_original()
    base_orig_id = base_mf.original_id()
    accepted: List[Dict[str, Any]] = []

    def _add_preserving_genus(cur_mf: m3d.Manifold, items: List[Dict[str, Any]], mfs: List[m3d.Manifold]) -> m3d.Manifold:
        if not items:
            return cur_mf
        try:
            u = mfs[0] if len(mfs) == 1 else m3d.Manifold.batch_boolean(mfs, m3d.OpType.Add)
            cand = cur_mf + u
            if cand.status() == m3d.Error.NoError and not cand.is_empty() and cand.genus() == 0:
                accepted.extend(items)
                return cand
        except Exception:
            pass
        if len(items) == 1:
            print(f"  [Skip Rock] {items[0]['rock_id']} descartada (aumentaba genus o desconectada)")
            return cur_mf
        mid = len(items) // 2
        cur_mf = _add_preserving_genus(cur_mf, items[:mid], mfs[:mid])
        cur_mf = _add_preserving_genus(cur_mf, items[mid:], mfs[mid:])
        return cur_mf

    chunk_sz = 48
    for start in range(0, len(placed), chunk_sz):
        chunk = placed[start : start + chunk_sz]
        mfs = [_build_rock_mf(inst).as_original() for inst in chunk]
        base_mf = _add_preserving_genus(base_mf, chunk, mfs)
        print(f"  [Talus Batch {start // chunk_sz + 1}] acumuladas {len(accepted)} rocas (genus={base_mf.genus()})")

    # Extraer máscara exacta de caras de roca mediante run_original_id / run_index antes de exportar STL
    out_mesh = base_mf.to_mesh()
    n_tris = len(out_mesh.tri_verts)
    tri_orig = np.full(n_tris, base_orig_id, dtype=np.int32)
    run_idx = np.asarray(out_mesh.run_index, dtype=np.int64)
    run_ids = np.asarray(out_mesh.run_original_id, dtype=np.int32)
    for r_i, oid in enumerate(run_ids):
        s_tri = int(run_idx[r_i] // 3)
        e_tri = int(run_idx[r_i + 1] // 3) if (r_i + 1) < len(run_idx) else n_tris
        tri_orig[s_tri:e_tri] = oid
    is_rock_raw = tri_orig != base_orig_id

    out_tm = trimesh.Trimesh(
        vertices=np.asarray(out_mesh.vert_properties[:, :3], dtype=np.float64),
        faces=np.asarray(out_mesh.tri_verts, dtype=np.int64),
        process=False,
    )
    # Si hubiera fragmentos minúsculos sueltos tras boolean, conservar componente principal y ajustar máscara
    if len(out_tm.faces) > 0:
        comps = out_tm.split(only_watertight=False)
        if len(comps) > 1:
            # Rarísimo en batch Add sobre un sólido unido; re-asignar por centroide si se dividió
            comps.sort(key=lambda cp: abs(float(cp.volume)), reverse=True)
            out_tm = comps[0]
            rock_tree_pts = np.asarray([inst["center_xyz"] for inst in accepted], dtype=np.float64)
            from scipy.spatial import cKDTree
            d_rk, idx_rk = cKDTree(rock_tree_pts).query(out_tm.triangles_center, k=1)
            r_rad = np.asarray([max(inst["scale_xyz"]) * 0.62 for inst in accepted], dtype=np.float64)
            is_rock_face = (d_rk < r_rad[idx_rk]) & (out_tm.triangles_center[:, 2] >= Z_BASE + 0.04)
        else:
            is_rock_face = is_rock_raw
    else:
        is_rock_face = is_rock_raw

    assert out_tm.is_watertight and out_tm.euler_number == 2, (
        f"Malla tras talus no estanca: wt={out_tm.is_watertight}, chi={out_tm.euler_number}"
    )

    out_stl = Path(args.out_stl)
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    out_tm.export(str(out_stl))
    rock_mask_npy = out_stl.with_suffix(".rock_faces.npy")
    np.save(str(rock_mask_npy), is_rock_face.astype(bool))
    rock_ctrs_npy = out_stl.with_suffix(".rock_centers.npy")
    np.save(str(rock_ctrs_npy), out_tm.triangles_center[is_rock_face].astype(np.float32))

    manifest = {
        "base_stl": str(args.base_stl),
        "out_stl": str(out_stl),
        "rock_faces_npy": str(rock_mask_npy),
        "rock_faces_count": int(is_rock_face.sum()),
        "accepted_count": len(accepted),
        "requested_count": args.target_count,
        "faces": int(len(out_tm.faces)),
        "vertices": int(len(out_tm.vertices)),
        "euler_number": int(out_tm.euler_number),
        "watertight": bool(out_tm.is_watertight),
        "unique_rock_textures": sorted({inst["diff_texture"] for inst in accepted if inst.get("diff_texture")}),
        "instances": accepted,
    }
    Path(args.out_manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Talus Done] {len(accepted)} rocas ({int(is_rock_face.sum())} caras) fusionadas -> {out_stl} ({len(out_tm.faces)} F, chi={out_tm.euler_number})")


if __name__ == "__main__":
    main()
