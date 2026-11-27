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
    in_file_id: str,
    in_sha256: str,
    staging_folder_id: str,
    alb_res: int,
    orm_res: int,
) -> str:
    helpers = ds.build_remote_drive_helpers_code()
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
    run_dir = Path("/content/campo_v05")
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[STAGE] Descargando campo_gpu_in.tar por red interna Drive...", flush=True)
    in_tar = Path("/content/campo_gpu_in.tar")
    _drive_download("{oauth_token}", "{in_file_id}", str(in_tar), "{in_sha256}")
    print(f"[STAGE] Drive pull OK en {{time.time()-t0:.1f}}s", flush=True)
    with tarfile.open(in_tar, "r:*") as tf:
        tf.extractall(run_dir)
    in_tar.unlink()

    bdir = Path("/content/blender-5.3.0-linux-x64")
    bbin = bdir / "blender"
    if not bbin.exists():
        print("[STAGE] Instalando Blender 5.3.0 Linux x64...", flush=True)
        tgz = Path("/content/blender53.tar.xz")
        subprocess.run(["wget", "-q", "-O", str(tgz), "https://download.blender.org/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz"], check=True)
        subprocess.run(["tar", "-xf", str(tgz), "-C", "/content"], check=True)
        tgz.unlink()
    subprocess.run("apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y libxrender1 libxi6 libxxf86vm1 libxfixes3 libgl1 libxkbcommon0 > /dev/null", shell=True, check=True)

    print("[STAGE] Ejecutando campo_pbr_bake.py --stage gpu-bake-only (Cycles OPTIX GPU)...", flush=True)
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
        print("[BLENDER] " + ln.rstrip(), flush=True)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Blender gpu-bake-only falló (rc={{rc}})")

    print("[STAGE] Empaquetando PNGs PBR 8K y vistas de auditoría hacia Drive...", flush=True)
    out_tar = Path("/content/campo_gpu_out.tar.gz")
    w = run_dir / "work"
    files = [
        "baked_albedo.png", "baked_normal.png", "baked_orm.png",
        "audit_ortho_pbr.png", "pbr_trees_closeup.png", "pbr_props_closeup.png",
        "pbr_views/pbr_000.png", "pbr_views/pbr_060.png", "pbr_views/pbr_120.png",
        "pbr_views/pbr_180.png", "pbr_views/pbr_240.png", "pbr_views/pbr_300.png",
    ]
    existing = [f for f in files if (w / f).exists()]
    subprocess.run(["tar", "--use-compress-program=gzip -1", "-cf", str(out_tar), "-C", str(w)] + existing, check=True)
    meta = _drive_upload("{oauth_token}", str(out_tar), "{staging_folder_id}", f"campo_gpu_out_{{int(time.time())}}.tar.gz")
    out_tar.unlink()
    meta["gpu_elapsed_s"] = round(time.time() - t0, 1)
    print("OUT_MANIFEST:" + json.dumps(meta), flush=True)
finally:
    stop_telem.set()
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Colab L4 Zero-Idle Bake para Campo v05 PBR")
    p.add_argument("--work-dir", default=str(REPO_ROOT / "Tableros-Catan" / "losetas" / "campo" / "work"))
    p.add_argument("--alb-res", type=int, default=8192)
    p.add_argument("--orm-res", type=int, default=4096)
    p.add_argument("--prep-only", action="store_true")
    p.add_argument("--skip-prep", action="store_true")
    args = p.parse_args()

    work_dir = Path(args.work_dir).resolve()
    scratch = work_dir / "_colab_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    blender_exe = tp.find_blender()

    if not args.skip_prep or not (work_dir / "campo_gpu_bake.blend").exists():
        print("[Fase 1] Ejecutando prep-local en Blender CPU local (0 créditos Colab)...", flush=True)
        cmd_prep = [
            blender_exe, "-b", "--python-exit-code", "1",
            "--python", str(REPO_ROOT / "tools/blender/campo_pbr_bake.py"),
            "--", "--work-dir", str(work_dir), "--stage", "prep-local",
        ]
        res = __import__("subprocess").run(cmd_prep, cwd=str(REPO_ROOT), text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0:
            raise RuntimeError(f"prep-local falló (rc={res.returncode})")

    in_tar = scratch / "campo_gpu_in.tar"
    with tarfile.open(in_tar, "w") as tf:
        for rel in ("campo_gpu_bake.blend", "plinth_albedo_1k.png"):
            tf.add(work_dir / rel, arcname=f"work/{rel}")
        for rel in ("tools/blender/campo_pbr_bake.py", "tools/blender/macro_bake_helpers.py"):
            tf.add(REPO_ROOT / rel, arcname=rel)
    print(f"[Fase 1] Paquete {in_tar.name} = {in_tar.stat().st_size / (1024*1024):.1f} MB", flush=True)
    if args.prep_only:
        return

    creds = cr.get_credentials(interactive=False, require_drive=True)
    proj = cr._read_project_id()

    print("[Fase 2] Subiendo paquete a Drive con GPU APAGADA (0 créditos)...", flush=True)
    folder_id = ds.ensure_staging_folder(creds)
    in_meta = ds.upload_staging_file(creds, in_tar, folder_id=folder_id)
    in_tar.unlink(missing_ok=True)
    print(f"[Fase 2] Subida Drive OK: file_id={in_meta['file_id']}", flush=True)

    out_meta: Dict[str, Any] = {}
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
        remote_code = _build_remote_script(fresh_tok, in_meta["file_id"], in_meta["sha256"], folder_id, args.alb_res, args.orm_res)

        def _cb(stream: str, line: str) -> None:
            if line.startswith("OUT_MANIFEST:"):
                out_meta.update(json.loads(line[len("OUT_MANIFEST:"):]))
            print(line, flush=True)

        rc = cr.execute_jupyter_code(conn, remote_code, timeout_s=1800, line_cb=_cb)
        if rc != 0 or not out_meta.get("file_id"):
            raise RuntimeError(f"Colab GPU bake falló (rc={rc}, out_meta={out_meta})")
    finally:
        if rt_name:
            print(f"[Fase 3] Ejecutando DeleteRuntime inmediato ANTES de descargar ({rt_name})...", flush=True)
            cr.delete_runtime(creds, proj, rt_name)
            rem = cr.list_runtimes(creds, proj)
            print(f"[Fase 3] Runtimes activos tras DeleteRuntime: {[r.get('name') for r in rem]}", flush=True)
        ds.delete_staging_file(creds, in_meta["file_id"])

    print("[Fase 4] Descargando PNGs 8K desde Drive (GPU apagada = 0 créditos)...", flush=True)
    out_tar = scratch / "campo_gpu_out.tar.gz"
    try:
        ds.download_staging_file(creds, out_meta["file_id"], out_tar, expected_sha256=out_meta.get("sha256"))
    finally:
        ds.delete_staging_file(creds, out_meta["file_id"])
    with tarfile.open(out_tar, "r:gz") as tf:
        tf.extractall(work_dir)
    out_tar.unlink(missing_ok=True)

    print("[Fase 4] Ejecutando finalize-local en PC local (0 créditos)...", flush=True)
    cmd_fin = [
        blender_exe, "-b", "--python-exit-code", "1",
        "--python", str(REPO_ROOT / "tools/blender/campo_pbr_bake.py"),
        "--", "--work-dir", str(work_dir), "--stage", "finalize-local",
    ]
    res2 = __import__("subprocess").run(cmd_fin, cwd=str(REPO_ROOT), text=True, encoding="utf-8", errors="replace")
    if res2.returncode != 0:
        raise RuntimeError(f"finalize-local falló (rc={res2.returncode})")
    sys.path.insert(0, str(REPO_ROOT / "tools" / "blender"))
    import campo_pbr_bake as cpb
    cpb._assemble_pbr_mosaic(work_dir)
    print("=== Campo v05 PBR 8K completado con Zero-Idle GPU ===", flush=True)


if __name__ == "__main__":
    main()
