"""
Auditoría matemática, topológica, de imprimibilidad FDM, empaquetado 3MF (lib3mf)
y laminado CLI headless real en Bambu Studio para Maker3D 2.0.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import lib3mf
import manifold3d
import numpy as np
import pymeshlab
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import tool_paths as tp

CATAN_W = 2.0 * 64.0 * math.cos(math.radians(30.0))  # 110.851252 mm
CATAN_D = 128.0000
CATAN_R = 64.0000


def load_as_trimesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"Escena vacía en {path}")
        return trimesh.util.concatenate(meshes)
    return loaded


def check_manifold_and_pymeshlab(stl_path: Path) -> Dict[str, Any]:
    """Evalúa topología con Manifold3D y PyMeshLab."""
    out: Dict[str, Any] = {}
    mesh = load_as_trimesh(stl_path)
    # Test Manifold3D
    try:
        m = manifold3d.Manifold(
            manifold3d.Mesh(
                vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
                tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
            )
        )
        out["manifold_status"] = str(m.status())
        out["manifold_valid"] = "NoError" in out["manifold_status"]
        out["manifold_num_tri"] = int(m.num_tri())
    except Exception as exc:
        out["manifold_status"] = f"error: {exc}"
        out["manifold_valid"] = False

    # Test PyMeshLab (ignorando avisos inocuos de DLL de ejemplo)
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(stl_path))
        topo = ms.get_topological_measures()
        out["pymeshlab_non_manifold_edges"] = int(topo.get("non_two_manifold_edges", -1))
        out["pymeshlab_boundary_edges"] = int(topo.get("boundary_edges", -1))
        out["pymeshlab_connected_components"] = int(topo.get("connected_components_number", -1))
    except Exception as exc:
        out["pymeshlab_error"] = str(exc)
    return out


def repair_mesh_stl(in_stl: Path, out_stl: Path) -> Path:
    """Repara duplicados, caras degeneradas y estanqueidad vía PyMeshLab / trimesh."""
    mesh = load_as_trimesh(in_stl)
    if mesh.is_watertight and int(mesh.euler_number) == 2 and int((mesh.area_faces < 1e-10).sum()) > 0:
        V = mesh.vertices.copy()
        F = mesh.faces.copy()
        for _ in range(6):
            p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
            bad = areas < 1e-10
            if not np.any(bad):
                break
            bad_v = np.unique(F[bad])
            mask_f = np.isin(F[:, 0], bad_v) | np.isin(F[:, 1], bad_v) | np.isin(F[:, 2], bad_v)
            sub_f = F[mask_f]
            for v in bad_v:
                inc = sub_f[np.any(sub_f == v, axis=1)]
                nbrs = np.setdiff1d(np.unique(inc), [v])
                if len(nbrs):
                    V[v] = 0.80 * V[v] + 0.20 * V[nbrs].mean(axis=0)
        mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

    if (not mesh.is_watertight) or (int(mesh.euler_number) != 2):
        mesh.merge_vertices()
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-9))
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)

    if not mesh.is_watertight:
        # Segundo paso con PyMeshLab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(in_stl))
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_null_faces()
        ms.meshing_repair_non_manifold_edges(method="Remove Faces")
        ms.meshing_close_holes(maxholesize=120)
        ms.save_current_mesh(str(out_stl))
    else:
        mesh.export(str(out_stl))
    return out_stl


def export_verified_3mf(mesh: trimesh.Trimesh, out_3mf: Path) -> Dict[str, Any]:
    out_3mf.parent.mkdir(parents=True, exist_ok=True)
    wrapper = lib3mf.Wrapper()
    model = wrapper.CreateModel()
    model.SetUnit(lib3mf.ModelUnit.MilliMeter)

    mesh_obj = model.AddMeshObject()
    mesh_obj.SetName("Maker3D_Part")

    verts = [lib3mf.Position((float(v[0]), float(v[1]), float(v[2]))) for v in mesh.vertices]
    tris = [lib3mf.Triangle((int(f[0]), int(f[1]), int(f[2]))) for f in mesh.faces]
    mesh_obj.SetGeometry(verts, tris)
    model.AddBuildItem(mesh_obj, wrapper.GetIdentityTransform())

    writer = model.QueryWriter("3mf")
    writer.WriteToFile(str(out_3mf))

    # Relectura de verificación
    verify_model = wrapper.CreateModel()
    reader = verify_model.QueryReader("3mf")
    reader.ReadFromFile(str(out_3mf))
    it = verify_model.GetMeshObjects()
    read_count = 0
    read_tris = 0
    while it.MoveNext():
        mo = it.GetCurrentMeshObject()
        read_count += 1
        read_tris += mo.GetTriangleCount()

    return {
        "path": str(out_3mf),
        "size_bytes": out_3mf.stat().st_size,
        "verified_objects": read_count,
        "verified_triangles": read_tris,
    }


def run_bambu_slice(input_model: Path, out_dir: Path, printer: str = "a1") -> Dict[str, Any]:
    bambu_info = tp.find_bambu_studio()
    exe = bambu_info.get("bambu_studio_exe")
    if not exe or not Path(exe).exists():
        raise FileNotFoundError("bambu-studio.exe no encontrado")
    prof = bambu_info["profiles"].get(printer)
    if not prof or not prof.get("machine") or not prof.get("process") or not prof.get("filament"):
        raise RuntimeError(f"Perfiles BBL incompletos para impresora '{printer}': {prof}")

    sliced_3mf = out_dir / f"{input_model.stem}_{printer}_sliced.gcode.3mf"
    if sliced_3mf.exists():
        sliced_3mf.unlink()

    cmd = [
        exe,
        "--slice", "0",
        "--debug", "3",
        "--load-settings", f"{prof['machine']};{prof['process']}",
        "--load-filaments", prof["filament"],
        "--export-3mf", str(sliced_3mf.resolve()),
        str(input_model.resolve()),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=240, cwd=str(out_dir.resolve()))
    ok = (res.returncode == 0) and sliced_3mf.exists() and (sliced_3mf.stat().st_size > 1000)

    gcode_meta: Dict[str, Any] = {}
    has_gcode = False
    if sliced_3mf.exists():
        with zipfile.ZipFile(sliced_3mf, "r") as z:
            names = z.namelist()
            has_gcode = "Metadata/plate_1.gcode" in names
            if has_gcode:
                header = z.read("Metadata/plate_1.gcode").decode("utf-8", errors="replace")[:3500]
                for pat, key in [
                    (r"total estimated time:\s*([^\r\n;]+)", "estimated_time"),
                    (r"total layer number:\s*(\d+)", "layer_count"),
                    (r"total filament length \[mm\]\s*:\s*([0-9.]+)", "filament_length_mm"),
                    (r"total filament weight \[g\]\s*:\s*([0-9.]+)", "filament_weight_g"),
                ]:
                    m = re.search(pat, header)
                    if m:
                        gcode_meta[key] = m.group(1).strip()

    return {
        "ok": bool(ok and has_gcode),
        "returncode": res.returncode,
        "printer": printer,
        "sliced_3mf_path": str(sliced_3mf),
        "sliced_3mf_bytes": sliced_3mf.stat().st_size if sliced_3mf.exists() else 0,
        "has_plate_1_gcode": has_gcode,
        "gcode_summary": gcode_meta,
    }


def check_catan_geometry(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    bounds = mesh.bounds
    dx = float(bounds[1][0] - bounds[0][0])
    dy = float(bounds[1][1] - bounds[0][1])
    z_min = float(bounds[0][2])
    z_max = float(bounds[1][2])

    # Comprobar hueco central en Z_min (los triángulos en centro r<4.0 deben tener Z >= z_min + 4.0)
    v = mesh.vertices
    c_tri = v[mesh.faces].mean(axis=1)
    r_center = np.hypot(c_tri[:, 0], c_tri[:, 1])
    center_low = c_tri[r_center < 3.5]
    magnet_roof_z = float(np.min(center_low[:, 2])) if len(center_low) else 0.0
    magnet_cavity_ok = magnet_roof_z >= (z_min + 4.0)

    # Comprobar los 6 pads planos en Z = 25.00 mm mediante raycast vertical exacto en NumPy (sin dependencia rtree)
    tris = v[mesh.faces]  # (F, 3, 3)
    tri_xmin = tris[:, :, 0].min(axis=1)
    tri_xmax = tris[:, :, 0].max(axis=1)
    tri_ymin = tris[:, :, 1].min(axis=1)
    tri_ymax = tris[:, :, 1].max(axis=1)

    pad_ok = True
    pad_zs = []
    for k in range(6):
        ang = math.radians(30.0 + 60.0 * k)
        ca, sa = math.cos(ang), math.sin(ang)
        top_per_ray = []
        for d_in in (3.5, 6.0, 8.2):
            qx, qy = (CATAN_R - d_in) * ca, (CATAN_R - d_in) * sa
            cand = tris[(tri_xmin <= qx + 1e-6) & (tri_xmax >= qx - 1e-6) & (tri_ymin <= qy + 1e-6) & (tri_ymax >= qy - 1e-6)]
            if len(cand) == 0:
                continue
            x0, y0, z0 = cand[:, 0, 0], cand[:, 0, 1], cand[:, 0, 2]
            v0x, v0y = cand[:, 1, 0] - x0, cand[:, 1, 1] - y0
            v1x, v1y = cand[:, 2, 0] - x0, cand[:, 2, 1] - y0
            det = v0x * v1y - v1x * v0y
            valid = np.abs(det) > 1e-12
            if not np.any(valid):
                continue
            px, py = qx - x0[valid], qy - y0[valid]
            inv_d = 1.0 / det[valid]
            u_b = (px * v1y[valid] - v1x[valid] * py) * inv_d
            v_b = (v0x[valid] * py - px * v0y[valid]) * inv_d
            w_b = 1.0 - u_b - v_b
            inside = (u_b >= -1e-5) & (v_b >= -1e-5) & (w_b >= -1e-5)
            if np.any(inside):
                z_hits = w_b[inside] * z0[valid][inside] + u_b[inside] * cand[valid, 1, 2][inside] + v_b[inside] * cand[valid, 2, 2][inside]
                top_per_ray.append(float(np.max(z_hits)))
        if len(top_per_ray) != 3 or any(abs(z_h - (z_min + 25.0)) > 0.25 for z_h in top_per_ray):
            pad_ok = False
        if top_per_ray:
            pad_zs.append(round(float(np.mean(top_per_ray)), 3))

    dims_ok = (abs(dx - CATAN_W) <= 0.25) and (abs(dy - CATAN_D) <= 0.25) and (z_max - z_min >= 24.8)
    return {
        "catan_dims_ok": dims_ok,
        "dx_mm": round(dx, 4),
        "dy_mm": round(dy, 4),
        "height_mm": round(z_max - z_min, 4),
        "magnet_cavity_ok": magnet_cavity_ok,
        "magnet_roof_z_mm": round(magnet_roof_z, 3),
        "pads_flat_25mm_ok": pad_ok,
        "pad_mean_z_mm": pad_zs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoría estanca, FDM y Bambu Studio CLI (Maker3D 2.0)")
    parser.add_argument("--input", required=True, help="Malla de entrada (.stl / .glb / .obj)")
    parser.add_argument("--out-dir", required=True, help="Directorio de salida para informes y archivos")
    parser.add_argument("--nozzle-mm", type=float, default=0.4)
    parser.add_argument("--overhang-deg", type=float, default=45.0)
    parser.add_argument("--repair", action="store_true", help="Reparar defectos si existen y guardar <nombre>_repaired.stl")
    parser.add_argument("--catan-check", action="store_true", help="Validar cotas e interfaz canónica de Catán")
    parser.add_argument("--export-3mf", help="Ruta del .3mf válido generado vía lib3mf")
    parser.add_argument(
        "--allow-high-genus",
        action="store_true",
        help="Permitir sólidos cerrados 2-manifold estancos de género g>=1 (χ = 2 - 2g <= 2, 1 componente conexa) en mallas generativas 3D",
    )
    parser.add_argument("--bambu-slice", action="store_true", help="Ejecutar laminado CLI real en bambu-studio.exe")
    parser.add_argument("--printer", choices=["a1", "p1s", "x1c", "a1_n02", "p1s_n02", "x1c_n02"], default="a1")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    work_stl = out_dir / f"{in_path.stem}_work.stl"
    mesh = load_as_trimesh(in_path)
    mesh.export(str(work_stl))

    needs_repair = (not mesh.is_watertight) or (int((mesh.area_faces < 1e-10).sum()) > 0) or (int(mesh.euler_number) != 2)
    repaired_used = False
    if args.repair and needs_repair:
        rep_stl = out_dir / f"{in_path.stem}_repaired.stl"
        repair_mesh_stl(work_stl, rep_stl)
        work_stl = rep_stl
        mesh = load_as_trimesh(work_stl)
        repaired_used = True

    degen = int((mesh.area_faces < 1e-10).sum())
    bounds = mesh.bounds
    z_min = float(bounds[0][2])

    # Cálculo de voladizos (> overhang_deg desde vertical -> nz < -sin(overhang_deg))
    normals = mesh.face_normals
    areas = mesh.area_faces
    centroids_z = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    # Excluir base plana en cama y techo del imán (puente interno corto Ø10mm)
    exclude_bed = (centroids_z <= z_min + 0.35) & (normals[:, 2] < -0.95)
    exclude_mag = (centroids_z <= z_min + 5.6) & (np.hypot(
        mesh.vertices[mesh.faces].mean(axis=1)[:, 0],
        mesh.vertices[mesh.faces].mean(axis=1)[:, 1],
    ) < 6.0) & (normals[:, 2] < -0.9)
    crit_cos = -math.cos(math.radians(90.0 - args.overhang_deg))  # nz < -sin(overhang_deg)
    overhang_thresh = -math.sin(math.radians(args.overhang_deg))
    overhang_mask = (normals[:, 2] < overhang_thresh) & (~exclude_bed) & (~exclude_mag)
    total_area = float(np.sum(areas)) if len(areas) else 1.0
    overhang_pct = float(np.sum(areas[overhang_mask]) / total_area * 100.0)

    # Estimación rápida de paredes finas (< 2 * nozzle_mm) por muestreo de rayos opuestos
    min_wall_mm = 2.0 * args.nozzle_mm
    thin_wall_pct = 0.0
    try:
        n_samp = min(400, len(mesh.faces))
        idx = np.linspace(0, len(mesh.faces) - 1, n_samp, dtype=int)
        origins = mesh.triangles_center[idx] - mesh.face_normals[idx] * 0.05
        dirs = -mesh.face_normals[idx]
        locs, ray_ids, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        if len(locs):
            dists = np.linalg.norm(locs - origins[ray_ids], axis=1) + 0.05
            thin_wall_pct = round(float(np.mean(dists < min_wall_mm) * 100.0), 2)
    except Exception:
        thin_wall_pct = 0.0

    topo_extra = check_manifold_and_pymeshlab(work_stl)

    report: Dict[str, Any] = {
        "input": str(in_path),
        "audited_stl": str(work_stl),
        "repaired": repaired_used,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "degenerate_faces": degen,
        "volume_cm3": round(float(mesh.volume) / 1000.0, 3) if mesh.is_watertight else None,
        "bounds_mm": [[round(float(x), 4) for x in bounds[0]], [round(float(x), 4) for x in bounds[1]]],
        "overhang_gt_deg_pct": round(overhang_pct, 3),
        "thin_wall_lt_2nozzle_pct": thin_wall_pct,
        "topology_engines": topo_extra,
    }

    if args.catan_check:
        report["catan_check"] = check_catan_geometry(mesh)

    if args.export_3mf:
        report["lib3mf_export"] = export_verified_3mf(mesh, Path(args.export_3mf))

    if args.bambu_slice:
        slice_input = Path(args.export_3mf) if args.export_3mf else work_stl
        report["bambu_slice"] = run_bambu_slice(slice_input, out_dir, printer=args.printer)

    report_path = out_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    euler_ok = (
        report["euler_number"] == 2
        or (
            args.allow_high_genus
            and report["euler_number"] <= 2
            and (report["euler_number"] % 2 == 0)
            and bool(topo_extra.get("manifold_valid"))
            and topo_extra.get("pymeshlab_non_manifold_edges") == 0
            and topo_extra.get("pymeshlab_boundary_edges") == 0
            and topo_extra.get("pymeshlab_connected_components") == 1
        )
    )

    ok = (
        report["is_watertight"]
        and report["degenerate_faces"] == 0
        and euler_ok
        and (not args.catan_check or (
            report["catan_check"]["catan_dims_ok"]
            and report["catan_check"]["magnet_cavity_ok"]
            and report["catan_check"]["pads_flat_25mm_ok"]
        ))
        and (not args.bambu_slice or report["bambu_slice"]["ok"])
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
