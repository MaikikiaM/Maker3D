"""
Exportador Puente Full Color (0 créditos Colab) para iDither / OrcaSlicer / Prusa ColorMix Lab:
  1) `catan_v07_fullcolor.glb`: malla completa (6.15M tris) + solo Base Color 4K WebP (Meshopt, sin Normal/Rough).
  2) `catan_v07_fullcolor.obj` + `.mtl` + `albedo_4k.png` + `plinth_1k.png`: Wavefront OBJ estándar.
  3) `catan_v07_fullcolor_web_2M.glb`: variante ligera (< 35 MB) sin compresión de malla ni dependencias
     Meshopt/Draco + Base Color 4K WebP para apps en navegador como Color Mix Lab.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = REPO_ROOT / "Tableros-Catan" / "pipeline-biomas" / "montana_hero360" / "work_v05"
DEFAULT_SCRATCH_VNEW = Path(
    r"C:\Users\migue\.gemini\antigravity\brain\34342c68-2385-41c9-a165-20883e62734d\scratch\v07_gbuf\V_new.npy"
)


def _find_gltf_transform() -> str:
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import tool_paths as tp
    gt = tp.find_webgl_tools().get("gltf_transform_cmd")
    if not gt or not Path(gt).exists():
        raise FileNotFoundError("No se encontró gltf-transform.cmd")
    return gt


def _run_gltf_transform(args_list: list[str]) -> None:
    gt = _find_gltf_transform()
    res = subprocess.run([gt] + args_list, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(f"gltf-transform falló ({res.returncode}):\n{res.stdout}\n{res.stderr}")


def _prepare_color_only_textures(work_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    alb_4k_p = out_dir / "albedo_4k.png"
    plinth_1k_p = out_dir / "plinth_1k.png"

    src_alb = work_dir / "baked_albedo.png"
    im = cv2.imread(str(src_alb), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"No se pudo leer {src_alb}")
    if im.shape[0] != 4096 or im.shape[1] != 4096:
        im = cv2.resize(im, (4096, 4096), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(alb_4k_p), im, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])

    src_pl = work_dir / "plinth_albedo_1k.png"
    if src_pl.exists():
        shutil.copy2(src_pl, plinth_1k_p)
    else:
        raise FileNotFoundError(f"No se encontró {src_pl}")

    return alb_4k_p, plinth_1k_p


def _run_blender_export(work_dir: Path, out_dir: Path, vnew_path: Path, decimate_ratio: float) -> None:
    import bpy
    import numpy as np

    t0 = time.time()
    blend_p = work_dir / "gpu_job_scene.blend"
    bpy.ops.wm.open_mainfile(filepath=str(blend_p))
    obj = bpy.data.objects.get("Catan_Hero360") or bpy.context.selected_objects[0]
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if vnew_path.exists():
        V_new = np.load(str(vnew_path)).astype(np.float64)
        if len(V_new) == len(obj.data.vertices):
            obj.data.vertices.foreach_set("co", V_new.ravel())
            print(f"[FullColorExport] Aplicado V_new.npy de v07 ({len(V_new):,} verts)", flush=True)

    me = obj.data
    poly_mat = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("material_index", poly_mat)
    is_ter_face = poly_mat >= 1
    me.polygons.foreach_set("material_index", is_ter_face.astype(np.int32).tolist())
    me.polygons.foreach_set("use_smooth", is_ter_face.tolist())
    me.update()

    for ul in list(me.uv_layers):
        if ul.name != "UVMap":
            me.uv_layers.remove(ul)
        else:
            ul.active = True
            ul.active_render = True
    for attr in list(me.color_attributes):
        me.color_attributes.remove(attr)

    alb_4k_p = out_dir / "albedo_4k.png"
    plinth_1k_p = out_dir / "plinth_1k.png"

    def _color_mat(name: str, img_path: Path, rough: float) -> bpy.types.Material:
        m = bpy.data.materials.new(name=name)
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["Metallic"].default_value = 0.0
        tc = nt.nodes.new("ShaderNodeTexCoord")
        tx = nt.nodes.new("ShaderNodeTexImage")
        tx.image = bpy.data.images.load(str(img_path), check_existing=False)
        nt.links.new(tc.outputs["UV"], tx.inputs["Vector"])
        nt.links.new(tx.outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        return m

    m_pl = _color_mat("Catan_Plinth_Color", plinth_1k_p, 0.55)
    m_ter = _color_mat("Catan_Terrain_Color", alb_4k_p, 0.85)
    me.materials.clear()
    me.materials.append(m_pl)
    me.materials.append(m_ter)
    me.polygons.foreach_set("material_index", is_ter_face.astype(np.int32).tolist())
    me.update()

    # 1) Exportar Wavefront OBJ + MTL enlazando albedo_4k.png y plinth_1k.png en out_dir
    obj_p = out_dir / "catan_v07_fullcolor.obj"
    bpy.ops.wm.obj_export(
        filepath=str(obj_p),
        export_selected_objects=True,
        export_materials=True,
        path_mode="RELATIVE",
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
        global_scale=1.0,
        export_uv=True,
        export_normals=True,
    )
    print(f"[FullColorExport] OBJ listo ({time.time() - t0:.1f}s) -> {obj_p}", flush=True)

    # 2) Exportar GLB completo (6.15M) solo Base Color sin Normal/Roughness (para comprimir a Meshopt+WebP)
    raw_full_glb = out_dir / "_tmp_full_color.glb"
    bpy.ops.export_scene.gltf(filepath=str(raw_full_glb), export_format="GLB", use_selection=True, export_apply=True)
    print(f"[FullColorExport] GLB base 6.15M listo ({time.time() - t0:.1f}s)", flush=True)

    # 3) Decimar protegiendo estrictamente el zócalo/pads B-Rep (Z <= 25.05) para versión web (< 35 MB sin Meshopt)
    n_v = len(me.vertices)
    co = np.empty(n_v * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    V = co.reshape(n_v, 3)
    rel_idx = np.where(V[:, 2] > 25.08)[0].tolist()
    vg = obj.vertex_groups.new(name="TerrainDecimate")
    vg.add(rel_idx, 1.0, "REPLACE")

    dec = obj.modifiers.new(name="DecimateWeb", type="DECIMATE")
    dec.decimate_type = "COLLAPSE"
    dec.ratio = decimate_ratio
    dec.vertex_group = vg.name
    dec.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=dec.name)
    print(f"[FullColorExport] Malla web decimada a {len(obj.data.polygons):,} caras ({time.time() - t0:.1f}s)", flush=True)

    raw_web_glb = out_dir / "_tmp_web_color.glb"
    bpy.ops.export_scene.gltf(filepath=str(raw_web_glb), export_format="GLB", use_selection=True, export_apply=True)
    print(f"[FullColorExport] GLB web base listo ({time.time() - t0:.1f}s)", flush=True)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description="Exportador Puente Full Color (GLB/OBJ/Web) para Maker3D v07")
    p.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--vnew-npy", default=str(DEFAULT_SCRATCH_VNEW))
    p.add_argument("--decimate-ratio", type=float, default=0.17)
    p.add_argument("--blender-worker", action="store_true")
    p.add_argument("--skip-blender", action="store_true")
    args = p.parse_args(argv)

    work_dir = Path(args.work_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (work_dir / "fullcolor_export")
    vnew_path = Path(args.vnew_npy).resolve()

    if args.blender_worker:
        _run_blender_export(work_dir, out_dir, vnew_path, args.decimate_ratio)
        return

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import tool_paths as tp

    if not args.skip_blender:
        _prepare_color_only_textures(work_dir, out_dir)
        blender_exe = tp.find_blender()
        cmd = [
            blender_exe, "-b", "--python-exit-code", "1",
            "--python", str(Path(__file__).resolve()),
            "--", "--work-dir", str(work_dir), "--out-dir", str(out_dir),
            "--vnew-npy", str(vnew_path), "--decimate-ratio", str(args.decimate_ratio),
            "--blender-worker",
        ]
        res = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0:
            raise RuntimeError(f"Blender worker falló ({res.returncode})")

    raw_full = out_dir / "_tmp_full_color.glb"
    raw_web = out_dir / "_tmp_web_color.glb"
    final_full = out_dir / "catan_v07_fullcolor.glb"
    final_web = out_dir / "catan_v07_fullcolor_web_2M.glb"

    print("[FullColorExport] Comprimiendo catan_v07_fullcolor.glb (Meshopt + WebP 4K)...", flush=True)
    _run_gltf_transform([
        "optimize", str(raw_full), str(final_full),
        "--compress", "meshopt", "--meshopt-level", "high",
        "--texture-compress", "webp", "--texture-size", "4096", "--simplify", "false",
    ])

    print("[FullColorExport] Empaquetando catan_v07_fullcolor_web_2M.glb (sin Meshopt, WebP 4K)...", flush=True)
    _run_gltf_transform([
        "webp", str(raw_web), str(final_web), "--quality", "92",
    ])

    raw_full.unlink(missing_ok=True)
    raw_web.unlink(missing_ok=True)
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name:36s} {f.stat().st_size / (1024*1024):6.2f} MB", flush=True)


if __name__ == "__main__":
    main()
