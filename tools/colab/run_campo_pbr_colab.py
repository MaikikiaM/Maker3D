"""
Orquestador Zero-Idle GPU (4 Fases) para horneado PBR 8K de Bioma Campo v05 en Google Colab Pro NVIDIA L4:
  - Fase 1 (PC Local, 0 créditos): `campo_pbr_bake.py --stage prep-local` + paquete tar.
  - Fase 2 (Drive Staging, GPU APAGADA = 0 créditos): subida resumable a `Maker3D_Colab_Staging`.
  - Fase 3 (Colab NVIDIA L4 OPTIX): descarga interna Drive + `--stage gpu-bake-only --device gpu`
    + subida Drive de mapas PNG + `delete_runtime()` inmediato ANTES de descargar al PC local.
  - Fase 4 (PC Local, 0 créditos): descarga Drive + limpieza Drive + `--stage finalize-local`.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "colab"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
import colab_api_runner as cr  # noqa: E402
import drive_staging as ds  # noqa: E402
import tool_paths as tp  # noqa: E402


def _build_remote_script(
    oauth_token: str,
    in_items: list,
    staging_folder_id: str,
    alb_res: int,
    orm_res: int,
) -> str:
    helpers = ds.build_remote_drive_helpers_code()
    items_json = json.dumps(in_items)
    return f"""
import json, os, shutil, subprocess, tarfile, threading, time
from pathlib import Path

stop_telem = threading.Event()
def _telem_loop():
    while not stop_telem.is_set():
        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                text=True,
            ).strip().splitlines()[0].split(",")
            print(f"[TELEM] GPU={{smi[0].strip()}} util={{smi[1].strip()}}% vram={{smi[2].strip()}}/{{smi[3].strip()}}MiB temp={{smi[4].strip()}}C", flush=True)
        except Exception as e:
            print("[TELEM_ERR]", e, flush=True)
        stop_telem.wait(4.0)

threading.Thread(target=_telem_loop, daemon=True).start()
{helpers}
try:
    t0 = time.time()
    items = json.loads({items_json!r})
    root_dir = Path("/content/campo_batch")
    shutil.rmtree(root_dir, ignore_errors=True)
    root_dir.mkdir(parents=True, exist_ok=True)

    bdir = Path("/content/blender-5.3.0-linux-x64")
    bbin = bdir / "blender"
    if not bbin.exists():
        print("[STAGE] Instalando Blender 5.3.0 Linux x64...", flush=True)
        tgz = Path("/content/blender53.tar.xz")
        subprocess.run(["wget", "-q", "-O", str(tgz), "https://download.blender.org/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz"], check=True)
        subprocess.run(["tar", "-xf", str(tgz), "-C", "/content"], check=True)
        tgz.unlink()
    subprocess.run("apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y libxrender1 libxi6 libxxf86vm1 libxfixes3 libgl1 libxkbcommon0 > /dev/null", shell=True, check=True)

    out_results = []
    files_to_pack = [
        "baked_albedo.png", "baked_normal.png", "baked_orm.png",
        "audit_ortho_pbr.png", "pbr_trees_closeup.png", "pbr_props_closeup.png",
        "pbr_views/pbr_000.png", "pbr_views/pbr_060.png", "pbr_views/pbr_120.png",
        "pbr_views/pbr_180.png", "pbr_views/pbr_240.png", "pbr_views/pbr_300.png",
    ]

    for idx, it in enumerate(items):
        tag = it["tag"]
        run_dir = root_dir / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[STAGE] ({{idx+1}}/{{len(items)}}) Descargando {{tag}} desde Drive...", flush=True)
        in_tar = root_dir / f"{{tag}}_in.tar"
        _drive_download("{oauth_token}", it["file_id"], str(in_tar), it["sha256"])
        with tarfile.open(in_tar, "r:*") as tf:
            tf.extractall(run_dir)
        in_tar.unlink()

        print(f"[STAGE] ({{idx+1}}/{{len(items)}}) Ejecutando gpu-bake-only (Cycles OPTIX GPU) para {{tag}}...", flush=True)
        cmd = [
            str(bbin), "-b", "--python-exit-code", "1",
            "--python", str(run_dir / "tools/blender/campo_pbr_bake.py"),
            "--",
            "--work-dir", str(run_dir / "work"),
            "--stage", "gpu-bake-only",
            "--device", "gpu",
            "--alb-res", "{alb_res}",
            "--orm-res", "{orm_res}",
        ]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, errors="replace")
        for ln in p.stdout:
            print(f"[BLENDER {{tag}}] " + ln.rstrip(), flush=True)
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Blender gpu-bake-only falló en {{tag}} (rc={{rc}})")

        out_tar = root_dir / f"{{tag}}_out.tar.gz"
        w = run_dir / "work"
        existing = [f for f in files_to_pack if (w / f).exists()]
        subprocess.run(["tar", "--use-compress-program=gzip -1", "-cf", str(out_tar), "-C", str(w)] + existing, check=True)
        meta = _drive_upload("{oauth_token}", str(out_tar), "{staging_folder_id}", f"campo_{{tag}}_out_{{int(time.time())}}.tar.gz")
        out_tar.unlink()
        shutil.rmtree(run_dir, ignore_errors=True)
        meta["tag"] = tag
        out_results.append(meta)

    manifest = {{"results": out_results, "gpu_elapsed_s": round(time.time() - t0, 1)}}
    print("OUT_MANIFEST:" + json.dumps(manifest), flush=True)
finally:
    stop_telem.set()
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Colab L4 Zero-Idle Bake para Campo PBR (individual o lote)")
    p.add_argument("--work-dir", default=str(REPO_ROOT / "Tableros-Catan" / "losetas" / "campo" / "work"))
    p.add_argument("--work-dirs", nargs="+", help="Lista de directorios de trabajo para hornear en una sola sesión GPU")
    p.add_argument("--alb-res", type=int, default=8192)
    p.add_argument("--orm-res", type=int, default=4096)
    p.add_argument("--glb-name", default="tile_pbr_v05.glb")
    p.add_argument("--glb-names", nargs="+", help="Nombres GLB de salida paralelos a --work-dirs")
    p.add_argument("--prep-only", action="store_true")
    p.add_argument("--skip-prep", action="store_true")
    args = p.parse_args()

    work_dirs = [Path(w).resolve() for w in (args.work_dirs or [args.work_dir])]
    if args.glb_names:
        if len(args.glb_names) != len(work_dirs):
            raise ValueError("--glb-names debe tener la misma longitud que --work-dirs")
        glb_names = args.glb_names
    else:
        glb_names = [args.glb_name if len(work_dirs) == 1 else f"tile_pbr_{w.name}.glb" for w in work_dirs]

    blender_exe = tp.find_blender()
    subprocess_mod = __import__("subprocess")

    staged_tars = []
    for wd in work_dirs:
        scratch = wd / "_colab_tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        if not args.skip_prep or not (wd / "campo_gpu_bake.blend").exists():
            print(f"[Fase 1] Ejecutando prep-local en Blender CPU local para {wd.name} (0 créditos Colab)...", flush=True)
            cmd_prep = [
                blender_exe, "-b", "--python-exit-code", "1",
                "--python", str(REPO_ROOT / "tools/blender/campo_pbr_bake.py"),
                "--", "--work-dir", str(wd), "--stage", "prep-local",
            ]
            res = subprocess_mod.run(cmd_prep, cwd=str(REPO_ROOT), text=True, encoding="utf-8", errors="replace")
            if res.returncode != 0:
                raise RuntimeError(f"prep-local falló en {wd} (rc={res.returncode})")

        in_tar = scratch / f"campo_{wd.name}_in.tar"
        with tarfile.open(in_tar, "w") as tf:
            for rel in ("campo_gpu_bake.blend", "plinth_albedo_1k.png"):
                tf.add(wd / rel, arcname=f"work/{rel}")
            for rel in ("tools/blender/campo_pbr_bake.py", "tools/blender/macro_bake_helpers.py"):
                tf.add(REPO_ROOT / rel, arcname=rel)
        print(f"[Fase 1] Paquete {in_tar.name} = {in_tar.stat().st_size / (1024*1024):.1f} MB", flush=True)
        staged_tars.append((wd, in_tar))

    if args.prep_only:
        return

    creds = cr.get_credentials(interactive=False, require_drive=True)
    proj = cr._read_project_id()

    print("[Fase 2] Subiendo paquetes a Drive con GPU APAGADA (0 créditos)...", flush=True)
    folder_id = ds.ensure_staging_folder(creds)
    in_items = []
    for wd, in_tar in staged_tars:
        in_meta = ds.upload_staging_file(creds, in_tar, folder_id=folder_id)
        in_tar.unlink(missing_ok=True)
        print(f"[Fase 2] Subida Drive OK ({wd.name}): file_id={in_meta['file_id']}", flush=True)
        in_items.append({"tag": wd.name, "file_id": in_meta["file_id"], "sha256": in_meta["sha256"]})

    out_manifest: Dict[str, Any] = {}
    rt_name = ""
    try:
        specs = cr.list_runtime_specs(creds, proj)
        candidates = cr.rank_gpu_specs(specs, accel_req="auto", prefer_highmem=True)
        rt = None
        for spec_key in candidates:
            try:
                print(f"[Fase 3] Provisionando GPU Colab: {spec_key}...", flush=True)
                rt = cr.create_runtime(creds, proj, spec_key)
                break
            except Exception as exc:
                print(f"  Sin disponibilidad {spec_key.get('accelerator')}: {exc}", flush=True)
        if rt is None:
            raise RuntimeError("No se pudo provisionar ninguna GPU Colab.")
        rt_name = rt["name"]
        conn = cr.get_runtime(creds, proj, rt_name)["connectionInfo"]
        fresh_tok = ds._auth_token(creds)
        remote_code = _build_remote_script(fresh_tok, in_items, folder_id, args.alb_res, args.orm_res)

        def _cb(stream: str, line: str) -> None:
            if line.startswith("OUT_MANIFEST:"):
                out_manifest.update(json.loads(line[len("OUT_MANIFEST:"):]))
            print(line, flush=True)

        rc = cr.execute_jupyter_code(conn, remote_code, timeout_s=3000, line_cb=_cb)
        if rc != 0 or not out_manifest.get("results"):
            raise RuntimeError(f"Colab GPU bake falló (rc={rc}, out_manifest={out_manifest})")
    finally:
        if rt_name:
            print(f"[Fase 3] Ejecutando DeleteRuntime inmediato ANTES de descargar ({rt_name})...", flush=True)
            cr.delete_runtime(creds, proj, rt_name)
            rem = cr.list_runtimes(creds, proj)
            print(f"[Fase 3] Runtimes activos tras DeleteRuntime: {[r.get('name') for r in rem]}", flush=True)
        for it in in_items:
            ds.delete_staging_file(creds, it["file_id"])

    by_tag = {r["tag"]: r for r in out_manifest["results"]}
    sys.path.insert(0, str(REPO_ROOT / "tools" / "blender"))
    import campo_pbr_bake as cpb

    for wd, glb_name in zip(work_dirs, glb_names):
        res_meta = by_tag[wd.name]
        print(f"[Fase 4] Descargando PNGs 8K de {wd.name} desde Drive (GPU apagada = 0 créditos)...", flush=True)
        out_tar = wd / "_colab_tmp" / f"campo_{wd.name}_out.tar.gz"
        try:
            ds.download_staging_file(creds, res_meta["file_id"], out_tar, expected_sha256=res_meta.get("sha256"))
        finally:
            ds.delete_staging_file(creds, res_meta["file_id"])
        with tarfile.open(out_tar, "r:gz") as tf:
            tf.extractall(wd)
        out_tar.unlink(missing_ok=True)

        print(f"[Fase 4] Ejecutando finalize-local para {wd.name} -> {glb_name} (0 créditos)...", flush=True)
        cmd_fin = [
            blender_exe, "-b", "--python-exit-code", "1",
            "--python", str(REPO_ROOT / "tools/blender/campo_pbr_bake.py"),
            "--", "--work-dir", str(wd), "--stage", "finalize-local", "--glb-name", glb_name,
        ]
        res2 = subprocess_mod.run(cmd_fin, cwd=str(REPO_ROOT), text=True, encoding="utf-8", errors="replace")
        if res2.returncode != 0:
            raise RuntimeError(f"finalize-local falló en {wd.name} (rc={res2.returncode})")
        cpb._assemble_pbr_mosaic(wd)

    print("=== Campo PBR 8K lote completado con Zero-Idle GPU ===", flush=True)


if __name__ == "__main__":
    main()
