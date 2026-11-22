"""
Descarga un pack de rocas CC0 escaneadas reales desde la API pública de Poly Haven,
documenta sus licencias en LICENSES.md y prepara cada roca como sólido watertight decimated (4k–10k Δ)
centrado en el origen y normalizado para colocación de derrubio en valles (place_talus.py).
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import manifold3d as m3d
import numpy as np
import pymeshlab
import trimesh

ROCK_IDS = [
    "namaqualand_boulder_03",
    "namaqualand_boulder_04",
    "namaqualand_boulder_05",
    "namaqualand_boulder_06",
    "rock_07",
    "rock_09",
    "stone_01",
    "moon_rock_02",
    "moon_rock_04",
    "moon_rock_05",
    "moon_rock_06",
    "moon_rock_07",
]

HEADERS = {"User-Agent": "Maker3D/2.0 (Antigravity CC0 PolyHaven Fetcher)"}


def _download_file(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=45) as resp:
        dst.write_bytes(resp.read())


def _repair_and_decimate_rock(gltf_path: Path, out_stl: Path, target_faces: int = 6000) -> dict:
    scene_or_mesh = trimesh.load(str(gltf_path), force="scene", process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [g for g in scene_or_mesh.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 50]
        meshes.sort(key=lambda m: len(m.faces), reverse=True)
        tm = meshes[0]
    else:
        tm = scene_or_mesh

    v_min = tm.vertices.min(axis=0)
    v_max = tm.vertices.max(axis=0)
    ext = np.maximum(v_max - v_min, 1e-6)
    max_d = float(ext.max())
    ctr = 0.5 * (v_min + v_max)
    tm.vertices = (tm.vertices - ctr) / max_d

    tmp_in = out_stl.with_suffix(".raw.ply")
    tm.export(str(tmp_in))
    del tm, scene_or_mesh

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(tmp_in))
    if tmp_in.exists():
        tmp_in.unlink()

    if ms.current_mesh().face_number() > target_faces * 2:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces * 2,
            preserveboundary=False,
            preservenormal=True,
            preservetopology=False,
            optimalplacement=True,
        )
    ms.generate_splitting_by_connected_components(delete_source_mesh=True)
    best_m = max(list(ms), key=lambda m: m.face_number())
    ms.set_current_mesh(best_m.id())
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=10000)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preserveboundary=False,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=True,
    )
    ms.save_current_mesh(str(out_stl))

    tm_final = trimesh.load(str(out_stl), force="mesh")
    if (not tm_final.is_watertight) or (tm_final.euler_number != 2):
        # Solidificar con Screened Poisson seleccionando la nueva malla generada y su mayor componente
        ms2 = pymeshlab.MeshSet()
        ms2.load_new_mesh(str(out_stl))
        ms2.generate_surface_reconstruction_screened_poisson(depth=7, preclean=True)
        # Borrar malla original y quedarse solo con la reconstrucción Poisson cerrada
        ms2.set_current_mesh(list(ms2)[-1].id())
        ms2.generate_splitting_by_connected_components(delete_source_mesh=True)
        best_p = max(list(ms2), key=lambda m: m.face_number())
        ms2.set_current_mesh(best_p.id())
        ms2.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preserveboundary=False,
            preservenormal=True,
            preservetopology=True,
        )
        ms2.save_current_mesh(str(out_stl))
        tm_final = trimesh.load(str(out_stl), force="mesh")

    tm_final.fix_normals()
    tm_final.apply_translation(-tm_final.centroid)
    tm_final.apply_scale(1.0 / max(float(tm_final.extents.max()), 1e-6))
    tm_final.export(str(out_stl))

    mf = m3d.Manifold(
        m3d.Mesh(
            vert_properties=np.asarray(tm_final.vertices, dtype=np.float32),
            tri_verts=np.asarray(tm_final.faces, dtype=np.uint32),
        )
    )
    mf_ok = (mf.status() == m3d.Error.NoError) and (not mf.is_empty())
    return {
        "faces": int(len(tm_final.faces)),
        "watertight": bool(tm_final.is_watertight),
        "euler": int(tm_final.euler_number),
        "extents_unit": [round(float(x), 4) for x in tm_final.extents],
        "manifold_ok": bool(mf_ok),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="Tableros-Catan/assets/polyhaven/rocks")
    p.add_argument("--licenses-md", default="Tableros-Catan/assets/polyhaven/LICENSES.md")
    p.add_argument("--res", default="1k")
    p.add_argument("--target-faces", type=int, default=6500)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_req = urllib.request.Request("https://api.polyhaven.com/assets?t=models", headers=HEADERS)
    catalog = json.loads(urllib.request.urlopen(meta_req, timeout=20).read())

    lic_lines = [
        "# Licencias Poly Haven (CC0 1.0 Universal)",
        "",
        "Modelos fotogramétricos de rocas escaneadas descargados desde la API pública de Poly Haven para derrubio de valle (`montana_hero360` v05).",
        "",
        "| Asset ID | Nombre | Licencia | Autores | Caras preparadas | Estanco | Manifold3D | URL |",
        "|---|---|---|---|---|---|---|---|",
    ]
    manifest = {}

    for rid in ROCK_IDS:
        rdir = out_dir / rid
        rdir.mkdir(parents=True, exist_ok=True)
        files_req = urllib.request.Request(f"https://api.polyhaven.com/files/{rid}", headers=HEADERS)
        fjson = json.loads(urllib.request.urlopen(files_req, timeout=20).read())
        gltf_spec = fjson.get("gltf", {}).get(args.res, {}).get("gltf")
        if not gltf_spec:
            print(f"[SKIP] {rid}: sin formato gltf {args.res}")
            continue

        gltf_p = rdir / f"{rid}_{args.res}.gltf"
        _download_file(gltf_spec["url"], gltf_p)
        diff_tex = None
        for rel_name, info in gltf_spec.get("include", {}).items():
            dst = rdir / rel_name
            _download_file(info["url"], dst)
            if "_diff_" in rel_name.lower():
                diff_tex = str(dst)

        stl_p = rdir / f"{rid}_solid.stl"
        stats = _repair_and_decimate_rock(gltf_p, stl_p, target_faces=args.target_faces)
        info_cat = catalog.get(rid, {})
        authors = ", ".join(info_cat.get("authors", {}).keys()) or "Poly Haven"
        name = info_cat.get("name", rid)
        url = f"https://polyhaven.com/a/{rid}"
        print(f"  [Rock {rid}] {stats} diff={Path(diff_tex).name if diff_tex else None}", flush=True)
        if stats["watertight"] and stats["euler"] == 2 and stats["manifold_ok"]:
            manifest[rid] = {
                "id": rid,
                "stl": str(stl_p),
                "diff_texture": diff_tex,
                **stats,
            }
            lic_lines.append(
                f"| `{rid}` | {name} | CC0 1.0 | {authors} | {stats['faces']} | {stats['watertight']} | {stats['manifold_ok']} | {url} |"
            )
            if len(manifest) >= 8:
                break

    Path(args.licenses_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.licenses_md).write_text("\n".join(lic_lines) + "\n", encoding="utf-8")
    (out_dir / "rocks_catalog.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[PolyHaven Done] {len(manifest)}/{len(ROCK_IDS)} rocas estancas listas -> {args.licenses_md}", flush=True)


if __name__ == "__main__":
    main()
