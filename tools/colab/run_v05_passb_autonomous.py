"""
Orquestador autónomo Zero-Idle GPU (4 Fases) para horneado 8K en Google Colab Pro NVIDIA L4:
  Fase 1 (CPU Local, 0 créditos): Alineación macro + `--stage prep-local` + paquete limpio (~235 MB).
  Fase 2 (Drive Pre-Staging, GPU APAGADA = 0 créditos): Subida previa resumable a `Maker3D_Colab_Staging`.
  Fase 3 (Ráfaga Efímera L4, ~2 min): Descarga interna Drive (<2s) + `--stage gpu-bake-only` (solo OptiX PNGs)
         + subida Drive (<2s) + `DeleteRuntime` inmediato ANTES de descargar a PC local.
  Fase 4 (PC Local, GPU APAGADA = 0 créditos): Descarga Drive + borrado temporal Drive +
         `--stage finalize-local` (`sculpt_3d` anti-cráter, STL, GLB) + auditoría FDM `0.20 mm`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "colab"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
import colab_api_runner as cr  # noqa: E402
import drive_staging as ds  # noqa: E402
import tool_paths as tp  # noqa: E402

WORK_DIR = REPO_ROOT / "Tableros-Catan" / "pipeline-biomas" / "montana_hero360" / "work_v05"
MACRO_DIR = WORK_DIR / "macro"
TELEM_PATH = WORK_DIR / "live_telemetry.json"
CONSOLE_PATH = WORK_DIR / "live_console.log"
SCRATCH_DIR = Path(
    r"C:\Users\migue\.gemini\antigravity\brain\34342c68-2385-41c9-a165-20883e62734d\scratch"
)

T0 = time.time()
TELEM_LOCK = threading.Lock()

TELEM: Dict[str, Any] = {
    "phase": "Iniciando",
    "detail": "Preparando pipeline Zero-Idle GPU",
    "progress_pct": 0,
    "elapsed_s": 0.0,
    "runtime_active": False,
    "runtime_status": "GPU APAGADA (0 CRÉDITOS)",
    "completed": False,
    "ram_used_gib": 0.0,
    "ram_total_gib": 0.0,
    "vram_used_gib": 0.0,
    "vram_total_gib": 0.0,
    "gpu_util_pct": 0.0,
    "gpu_temp_c": None,
    "gpu_power_w": None,
    "gpu_name": "--",
    "disk_used_gib": 0.0,
    "disk_total_gib": 0.0,
    "disk_free_gib": 0.0,
    "xfer_done_mb": 0.0,
    "xfer_total_mb": 0.0,
    "blender_stage": "--",
}


def update_telem(**kw: Any) -> None:
    with TELEM_LOCK:
        TELEM.update(kw)
        TELEM["elapsed_s"] = round(time.time() - T0, 1)
        payload = json.dumps(TELEM, indent=2)
        tmp = TELEM_PATH.with_suffix(".tmp")
        for attempt in range(10):
            try:
                tmp.write_text(payload, encoding="utf-8")
                tmp.replace(TELEM_PATH)
                return
            except PermissionError:
                time.sleep(0.03 * (attempt + 1))
        try:
            TELEM_PATH.write_text(payload, encoding="utf-8")
        except Exception:
            pass


def log_line(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with TELEM_LOCK:
        print(line, flush=True)
        with open(CONSOLE_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _run_stream(cmd: List[str], prefix: str = "") -> None:
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for ln in proc.stdout:
        log_line(f"{prefix}{ln.rstrip()}")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Comando falló ({rc}): {' '.join(cmd[:4])}")


def build_macro_grid_16() -> Path:
    rep = json.loads((MACRO_DIR / "macro_report.json").read_text(encoding="utf-8"))
    keys = sorted(rep["patches"].keys())
    tile = 320
    cols = 4
    rows = (len(keys) + cols - 1) // cols
    canvas = np.full((rows * tile, cols * tile, 3), 24, dtype=np.uint8)
    for idx, k in enumerate(keys):
        r, c = divmod(idx, cols)
        p = MACRO_DIR / "nb" / f"patch_{k}_nb.png"
        if not p.exists():
            p = MACRO_DIR / "nb" / f"patch_{k}_nb.jpg"
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, (tile, tile), interpolation=cv2.INTER_AREA)
        info = rep["patches"][k]
        ok = info.get("accepted", False)
        col = (110, 240, 120) if ok else (90, 90, 250)
        cv2.rectangle(im, (0, 0), (tile, 44), (15, 18, 25), -1)
        t1 = f"{k} {'OK' if ok else 'FAIL'} rec={info.get('ridge_recall_5px', 0):.2f}"
        t2 = f"ssim={info.get('ssim_low', 0):.2f} flow={info.get('mean_flow_px', 0):.1f} dE={info.get('chroma_deltaE', 0):.1f}"
        cv2.putText(im, t1, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
        cv2.putText(im, t2, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (215, 225, 240), 1, cv2.LINE_AA)
        canvas[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = im
    out_p = MACRO_DIR / "macro_grid_16.jpg"
    cv2.imwrite(str(out_p), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return out_p


def phase1_local_prep(skip_f53: bool, projection_mode: str) -> Path:
    update_telem(
        phase="Fase 1 (CPU Local · 0 créditos)",
        detail="Alineando parches + limpiando gpu_job_scene.blend...",
        progress_pct=5,
        runtime_status="GPU APAGADA (0 CRÉDITOS)",
    )
    if not skip_f53:
        log_line("=== Fase 1A: Alineación óptica macro (aligned_rgb + aligned_hp) ===")
        _run_stream([
            sys.executable, "-u", str(REPO_ROOT / "tools" / "generative" / "macro_patch_align.py"),
            "--macro-dir", str(MACRO_DIR), "--sigma-hp", "6.0",
        ])

    rep = json.loads((MACRO_DIR / "macro_report.json").read_text(encoding="utf-8"))
    grid_p = build_macro_grid_16()
    log_line(f"F5.3 verificada: accepted={rep['accepted']}/{rep['accepted'] + rep['rejected']} | {grid_p.name}")
    if int(rep.get("accepted", 0)) < 14:
        raise RuntimeError(f"Demasiados parches rechazados ({rep['accepted']}/16 < 14). Abortando con 0 créditos gastados.")

    blender_exe = tp.find_blender()
    log_line(f"=== Fase 1B: Blender local --stage prep-local (zstd + caché BVH exacta, 0 créditos) ===")
    _run_stream([
        blender_exe, "-b", "--python-exit-code", "1",
        "--python", str(REPO_ROOT / "tools" / "blender" / "catan_hero360_bake.py"),
        "--", "--cameras", str(WORK_DIR / "cameras.json"),
        "--out-dir", str(WORK_DIR), "--macro-dir", str(MACRO_DIR),
        "--stage", "prep-local",
    ], prefix="[PREP] ")

    in_tar = SCRATCH_DIR / "gpu_job_in.tar"
    in_tar.parent.mkdir(parents=True, exist_ok=True)
    files_to_pack: List[tuple[Path, str]] = [
        (WORK_DIR / "cameras.json", "work_v05/cameras.json"),
        (WORK_DIR / "gpu_job_scene.blend", "work_v05/gpu_job_scene.blend"),
        (WORK_DIR / "passA_albedo_8k.png", "work_v05/passA_albedo_8k.png"),
        (WORK_DIR / "plinth_albedo_1k.png", "work_v05/plinth_albedo_1k.png"),
        (MACRO_DIR / "macro_cameras.json", "work_v05/macro/macro_cameras.json"),
        (MACRO_DIR / "macro_report.json", "work_v05/macro/macro_report.json"),
        (MACRO_DIR / "macro_vis_cache.npz", "work_v05/macro/macro_vis_cache.npz"),
        (REPO_ROOT / "tools" / "blender" / "catan_hero360_bake.py", "tools/blender/catan_hero360_bake.py"),
        (REPO_ROOT / "tools" / "blender" / "macro_bake_helpers.py", "tools/blender/macro_bake_helpers.py"),
        (REPO_ROOT / "tools" / "blender" / "geology_micro_detail.py", "tools/blender/geology_micro_detail.py"),
    ]
    for p in sorted((MACRO_DIR / "renders").glob("patch_*_mask.png")):
        files_to_pack.append((p, f"work_v05/macro/renders/{p.name}"))
    if projection_mode == "full":
        for p in sorted((MACRO_DIR / "aligned_rgb").glob("patch_*_aligned.png")):
            files_to_pack.append((p, f"work_v05/macro/aligned_rgb/{p.name}"))
    else:
        for p in sorted((MACRO_DIR / "aligned_hp").glob("patch_*_hp16.png")):
            files_to_pack.append((p, f"work_v05/macro/aligned_hp/{p.name}"))

    with tarfile.open(in_tar, "w") as tf:
        for src, arc in files_to_pack:
            if not src.exists():
                raise FileNotFoundError(f"Falta insumo crítico para GPU bake: {src}")
            tf.add(src, arcname=arc)
    sz_mb = in_tar.stat().st_size / (1024 * 1024)
    log_line(f"Paquete gpu_job_in.tar listo: {sz_mb:.1f} MB (modo={projection_mode})")
    update_telem(progress_pct=18, detail=f"Paquete listo ({sz_mb:.1f} MB)")
    return in_tar


def build_remote_gpu_script(
    oauth_token: str,
    in_file_id: str,
    in_sha256: str,
    staging_folder_id: str,
    projection_mode: str,
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
            mem = {{}}
            for ln in Path("/proc/meminfo").read_text().splitlines():
                k, v = ln.split(":", 1)
                mem[k.strip()] = int(v.strip().split()[0])
            rtot = mem["MemTotal"] / (1024 * 1024)
            rused = (mem["MemTotal"] - mem.get("MemAvailable", mem["MemFree"])) / (1024 * 1024)
            du = shutil.disk_usage("/content")
            print("[TELEM] " + json.dumps({{
                "gpu_name": smi[0].strip(),
                "gpu_util_pct": round(float(smi[1].strip()), 1),
                "vram_used_gib": round(float(smi[2].strip()) / 1024.0, 2),
                "vram_total_gib": round(float(smi[3].strip()) / 1024.0, 2),
                "gpu_temp_c": round(float(smi[4].strip()), 1),
                "gpu_power_w": round(float(smi[5].strip()), 1),
                "ram_used_gib": round(rused, 2),
                "ram_total_gib": round(rtot, 2),
                "disk_used_gib": round(du.used / (1024**3), 2),
                "disk_total_gib": round(du.total / (1024**3), 2),
                "disk_free_gib": round(du.free / (1024**3), 2),
            }}), flush=True)
        except Exception as e:
            print("[TELEM_ERR]", e, flush=True)
        stop_telem.wait(2.0)

# Telemetría arranca en t=0s ANTES de cualquier descarga o instalación
threading.Thread(target=_telem_loop, daemon=True).start()
{helpers}
try:
    t_gpu0 = time.time()
    run_dir = Path("/content/run_v06")
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[STAGE] Descargando gpu_job_in.tar por red interna Drive...", flush=True)
    in_tar = Path("/content/gpu_job_in.tar")
    _drive_download("{oauth_token}", "{in_file_id}", str(in_tar), "{in_sha256}")
    print(f"[STAGE] Drive pull completado en {{time.time()-t_gpu0:.1f}}s", flush=True)
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

    print("[STAGE] Ejecutando --stage gpu-bake-only (100% OptiX GPU, sin CPU STL/GLB)...", flush=True)
    cmd = [
        str(bbin), "-b", "--python-exit-code", "1",
        "--python", str(run_dir / "tools/blender/catan_hero360_bake.py"),
        "--",
        "--cameras", str(run_dir / "work_v05/cameras.json"),
        "--out-dir", str(run_dir / "work_v05"),
        "--macro-dir", str(run_dir / "work_v05/macro"),
        "--projection-mode", "{projection_mode}",
        "--bake-res", "8192",
        "--normal-res", "8192",
        "--rough-res", "4096",
        "--bake-samples", "16",
        "--device", "gpu",
        "--stage", "gpu-bake-only",
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, errors="replace")
    for ln in p.stdout:
        print("[BLENDER] " + ln.rstrip(), flush=True)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Blender gpu-bake-only falló con código {{rc}}")

    print("[STAGE] Empaquetando solo PNGs horneados hacia Drive...", flush=True)
    out_tar = Path("/content/gpu_job_out.tar.gz")
    w = run_dir / "work_v05"
    arcs = [a for a in ("baked_albedo.png", "baked_normal.png", "baked_roughness.png", "macro_hp_height_8k.png", "macro/ao_cavity_8k.png") if (w / a).exists()]
    subprocess.run(["tar", "--use-compress-program=gzip -1", "-cf", str(out_tar), "-C", str(w)] + arcs, check=True)
    meta = _drive_upload("{oauth_token}", str(out_tar), "{staging_folder_id}", f"gpu_job_out_{{int(time.time())}}.tar.gz")
    out_tar.unlink()
    meta["gpu_elapsed_s"] = round(time.time() - t_gpu0, 1)
    print("OUT_MANIFEST:" + json.dumps(meta), flush=True)
finally:
    stop_telem.set()
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Orquestador Zero-Idle GPU (4 fases) para Maker3D Colab Pro")
    p.add_argument("--skip-f53", action="store_true")
    p.add_argument("--projection-mode", choices=["full", "highpass"], default="full")
    p.add_argument("--no-audit-render", action="store_true")
    p.add_argument("--dry-prep-only", action="store_true", help="Ejecuta solo Fase 1 local (0 créditos Colab)")
    args = p.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CONSOLE_PATH.write_text("", encoding="utf-8")

    # Fase 1: Preparación CPU local (0 créditos)
    in_tar = phase1_local_prep(skip_f53=args.skip_f53, projection_mode=args.projection_mode)
    if args.dry_prep_only:
        log_line("Modo --dry-prep-only completado (0 créditos Colab usados).")
        update_telem(completed=True, progress_pct=100, phase="Prep Local Listo (0 créditos)")
        return

    # Verificar Drive scope OBLIGATORIO antes de encender ninguna GPU
    creds = cr.get_credentials(interactive=False, require_drive=True)
    proj = cr._read_project_id()

    # Fase 2: Pre-staging en Google Drive con GPU APAGADA (0 créditos)
    total_mb = in_tar.stat().st_size / (1024 * 1024)
    update_telem(
        phase="Fase 2 (Drive Staging · GPU APAGADA)",
        detail=f"Subiendo {total_mb:.1f} MB a Drive antes de encender GPU...",
        progress_pct=24,
        runtime_status="SUBIENDO A DRIVE (GPU APAGADA — 0 CRÉDITOS)",
        xfer_done_mb=0.0,
        xfer_total_mb=round(total_mb, 1),
    )
    folder_id = ds.ensure_staging_folder(creds)
    in_meta = ds.upload_staging_file(
        creds, in_tar, folder_id=folder_id,
        progress_cb=lambda s, t: update_telem(
            progress_pct=24 + int(16 * s / max(t, 1)),
            xfer_done_mb=round(s / (1024 * 1024), 1),
            detail=f"Drive staging {s / (1024*1024):.1f}/{total_mb:.1f} MB (0 créditos)",
        ),
    )
    in_tar.unlink(missing_ok=True)
    log_line(f"Fase 2 OK: file_id={in_meta['file_id']} subido a Drive a 0 créditos GPU.")

    # Fase 3: Ráfaga efímera L4 (GPU ON solo durante Cycles OptiX + red interna Drive)
    out_meta: Dict[str, Any] = {}
    rt_name = ""
    try:
        specs = cr.list_runtime_specs(creds, proj)
        candidates = cr.rank_gpu_specs(specs, accel_req="auto", prefer_highmem=True)
        update_telem(phase="Fase 3 (Ráfaga GPU L4)", detail="Provisionando NVIDIA L4...", progress_pct=42)
        rt = None
        for spec_key in candidates:
            try:
                rt = cr.create_runtime(creds, proj, spec_key)
                break
            except Exception as exc:
                log_line(f"  Sin disponibilidad {spec_key.get('accelerator')}: {exc}")
                if "TOO_MANY_ACTIVE_RUNTIMES" in str(exc):
                    raise
        if rt is None:
            raise RuntimeError("No se pudo provisionar ninguna GPU Colab.")

        rt_name = rt["name"]
        acc_name = rt.get("runtimeSpec", {}).get("accelerator", "GPU")
        update_telem(runtime_active=True, runtime_status=f"CONECTADO: NVIDIA {acc_name}", gpu_name=f"NVIDIA {acc_name}")
        log_line(f"Runtime L4 activo: {rt_name} ({acc_name})")

        conn = cr.get_runtime(creds, proj, rt_name)["connectionInfo"]
        fresh_tok = ds._auth_token(creds)
        remote_code = build_remote_gpu_script(
            fresh_tok, in_meta["file_id"], in_meta["sha256"], folder_id, args.projection_mode
        )

        def _on_line(stream: str, line: str) -> None:
            if line.startswith("[TELEM] "):
                try:
                    update_telem(**json.loads(line[8:]))
                except Exception:
                    pass
                return
            if line.startswith("OUT_MANIFEST:"):
                out_meta.update(json.loads(line[len("OUT_MANIFEST:"):]))
                log_line(line)
                return
            if line.startswith("[STAGE] "):
                update_telem(detail=line[8:], blender_stage=line[8:50])
            log_line(line)

        rc = cr.execute_jupyter_code(conn, remote_code, timeout_s=1800, line_cb=_on_line)
        if rc != 0 or not out_meta.get("file_id"):
            raise RuntimeError(f"Ráfaga GPU falló (rc={rc}, out_meta={out_meta})")
    finally:
        # APAGADO INMEDIATO DE GPU ANTES DE CUALQUIER DESCARGA LOCAL
        if rt_name:
            log_line(f"Apagando GPU inmediatamente antes de descargar ({rt_name})...")
            try:
                cr.delete_runtime(creds, proj, rt_name)
                rem_rts = cr.list_runtimes(creds, proj)
                log_line(f"Verificación post-DeleteRuntime: runtimes activos = {[r.get('name') for r in rem_rts]}")
            except Exception as exc:
                log_line(f"Aviso DeleteRuntime: {exc}")
            update_telem(
                runtime_active=False,
                runtime_status="DESCARGANDO DE DRIVE (GPU YA APAGADA — 0 CRÉDITOS)",
            )
        ds.delete_staging_file(creds, in_meta["file_id"])

    # Fase 4: Descarga local de Drive + finalización CPU local (GPU YA APAGADA = 0 créditos)
    out_tar = SCRATCH_DIR / "gpu_job_out.tar.gz"
    try:
        ds.download_staging_file(creds, out_meta["file_id"], out_tar, expected_sha256=out_meta.get("sha256"))
    finally:
        ds.delete_staging_file(creds, out_meta["file_id"])

    with tarfile.open(out_tar, "r:gz") as tf:
        tf.extractall(WORK_DIR)
    out_tar.unlink(missing_ok=True)

    update_telem(
        phase="Fase 4 (CPU Local · 0 créditos)",
        detail="Ejecutando --stage finalize-local (sculpt anti-cráter + STL/GLB)...",
        progress_pct=88,
        runtime_status="APAGADO (0 unidades consumiéndose)",
    )
    blender_exe = tp.find_blender()
    render_smp = "0" if args.no_audit_render else "32"
    _run_stream([
        blender_exe, "-b", "--python-exit-code", "1",
        "--python", str(REPO_ROOT / "tools" / "blender" / "catan_hero360_bake.py"),
        "--", "--cameras", str(WORK_DIR / "cameras.json"),
        "--out-dir", str(WORK_DIR), "--stage", "finalize-local",
        "--device", "cpu", "--render-samples", render_smp,
    ], prefix="[FINALIZE] ")
    update_telem(completed=True, progress_pct=100, phase="COMPLETADO (Zero-Idle GPU)")
    log_line("=== Pipeline 4 Fases Zero-Idle completado ===")


if __name__ == "__main__":
    main()
