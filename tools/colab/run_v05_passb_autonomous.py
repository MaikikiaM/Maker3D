"""
Orquestador autónomo F5.3 + F6 (--stage passB 8K en Colab Pro NVIDIA L4) con streaming
en tiempo real a work_v05/live_telemetry.json y work_v05/live_console.log para localhost:8799.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "colab"))
import colab_api_runner as cr  # noqa: E402

WORK_DIR = REPO_ROOT / "Tableros-Catan" / "pipeline-biomas" / "montana_hero360" / "work_v05"
MACRO_DIR = WORK_DIR / "macro"
TELEM_PATH = WORK_DIR / "live_telemetry.json"
CONSOLE_PATH = WORK_DIR / "live_console.log"
SCRATCH_DIR = Path(
    r"C:\Users\migue\.gemini\antigravity\brain\34342c68-2385-41c9-a165-20883e62734d\scratch"
)

CHUNK_BYTES = 8 * 1024 * 1024  # 8 MB crudos (~10.7 MB Base64, óptimo para transferencia paralela)
MAX_WORKERS = 8
T0 = time.time()
TELEM_LOCK = threading.Lock()
_THREAD_LOCAL = threading.local()

TELEM: Dict[str, Any] = {
    "phase": "Iniciando",
    "detail": "Preparando pipeline",
    "progress_pct": 0,
    "elapsed_s": 0.0,
    "runtime_active": False,
    "runtime_status": "ESPERANDO",
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


def _get_session() -> requests.Session:
    s = getattr(_THREAD_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        _THREAD_LOCAL.session = s
    return s


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
        t1 = f"{k} {'OK' if ok else 'FAIL'} hp={info.get('hp_std', 0):.3f}"
        t2 = f"ssim={info.get('ssim_low', 0):.2f} flow={info.get('mean_flow_px', 0):.1f} dE={info.get('chroma_deltaE', 0):.1f}"
        cv2.putText(im, t1, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)
        cv2.putText(im, t2, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (215, 225, 240), 1, cv2.LINE_AA)
        canvas[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = im
    out_p = MACRO_DIR / "macro_grid_16.jpg"
    cv2.imwrite(str(out_p), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return out_p


def run_f53_align() -> None:
    update_telem(phase="F5.3 Alineando 16 parches", detail="Ejecutando macro_patch_align.py...", progress_pct=5)
    log_line("=== F5.3: Alineación óptica ECC + Farneback + High-Pass 16-bit ===")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(REPO_ROOT / "tools" / "generative" / "macro_patch_align.py"),
         "--macro-dir", str(MACRO_DIR), "--sigma-hp", "6.0"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for ln in proc.stdout:
        log_line(ln.rstrip())
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"macro_patch_align.py falló con código {rc}")
    rep = json.loads((MACRO_DIR / "macro_report.json").read_text(encoding="utf-8"))
    grid_p = build_macro_grid_16()
    log_line(f"F5.3 completado: accepted={rep['accepted']}/{rep['accepted'] + rep['rejected']} | Grid={grid_p.name}")
    if int(rep.get("accepted", 0)) < 14:
        raise RuntimeError(f"Demasiados parches rechazados ({rep['accepted']}/16 < 14). Abortando antes de usar Colab L4.")
    update_telem(detail=f"Alineados {rep['accepted']}/16 parches", progress_pct=15)


def package_passb_inputs(tar_path: Path, blend_src: Path | None = None) -> None:
    update_telem(phase="Empaquetando escena Pass B", detail="Creando passB_in.tar (sin re-comprimir zstd/PNG)...", progress_pct=18)
    log_line("=== Empaquetando passB_in.tar (blend zstd + 8K + máscaras + HP16 + scripts) ===")
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    blend_path = blend_src if blend_src else (WORK_DIR / "passA_scene.blend")
    files_to_pack: List[tuple[Path, str]] = [
        (WORK_DIR / "cameras.json", "work_v05/cameras.json"),
        (blend_path, "work_v05/passA_scene.blend"),
        (WORK_DIR / "passA_albedo_8k.png", "work_v05/passA_albedo_8k.png"),
        (WORK_DIR / "plinth_albedo_1k.png", "work_v05/plinth_albedo_1k.png"),
        (MACRO_DIR / "macro_cameras.json", "work_v05/macro/macro_cameras.json"),
        (MACRO_DIR / "macro_report.json", "work_v05/macro/macro_report.json"),
        (REPO_ROOT / "tools" / "blender" / "catan_hero360_bake.py", "tools/blender/catan_hero360_bake.py"),
        (REPO_ROOT / "tools" / "blender" / "macro_bake_helpers.py", "tools/blender/macro_bake_helpers.py"),
        (REPO_ROOT / "tools" / "blender" / "geology_micro_detail.py", "tools/blender/geology_micro_detail.py"),
    ]
    for p in sorted((MACRO_DIR / "renders").glob("patch_*_mask.png")):
        files_to_pack.append((p, f"work_v05/macro/renders/{p.name}"))
    for p in sorted((MACRO_DIR / "aligned_hp").glob("patch_*_hp16.png")):
        files_to_pack.append((p, f"work_v05/macro/aligned_hp/{p.name}"))

    with tarfile.open(tar_path, "w") as tf:
        for src, arc in files_to_pack:
            if not src.exists():
                raise FileNotFoundError(f"Falta entrada crítica para PassB: {src}")
            tf.add(src, arcname=arc)
    sz_mb = tar_path.stat().st_size / (1024 * 1024)
    log_line(f"Empaquetado listo: {tar_path.name} = {sz_mb:.1f} MB")


def _proxy_headers(conn_info: Dict[str, Any]) -> Dict[str, str]:
    return {
        "X-Colab-Runtime-Proxy-Token": conn_info["token"],
        "X-Colab-Client-Agent": "vscode",
        "Content-Type": "application/json",
    }


def upload_chunks_to_colab(creds: Any, proj: str | None, rt_name: str, local_file: Path) -> tuple[str, int, str]:
    total = local_file.stat().st_size
    h_sha = hashlib.sha256()
    with open(local_file, "rb") as f_h:
        while True:
            b = f_h.read(4 * 1024 * 1024)
            if not b:
                break
            h_sha.update(b)
    sha = h_sha.hexdigest()
    n_chunks = (total + CHUNK_BYTES - 1) // CHUNK_BYTES
    total_mb = total / (1024 * 1024)

    rt = cr.get_runtime(creds, proj, rt_name)
    conn = rt["connectionInfo"]
    base_url = conn["url"].rstrip("/")
    hdrs = _proxy_headers(conn)

    # Detectar raíz de /api/contents ("content" o raíz)
    root_prefix = ""
    chk = requests.get(f"{base_url}/api/contents", headers=hdrs, timeout=20)
    if chk.ok:
        names = [e.get("name") for e in chk.json().get("content", []) if isinstance(e, dict)]
        if "sample_data" not in names and "content" in names:
            root_prefix = "content/"
    log_line(f"Subiendo {total_mb:.1f} MB en {n_chunks} chunks ({MAX_WORKERS} hilos) hacia /api/contents/{root_prefix} ...")
    update_telem(phase="Subiendo escena a Colab L4", detail=f"0 / {n_chunks} bloques", progress_pct=22, xfer_done_mb=0.0, xfer_total_mb=round(total_mb, 1))

    state_lock = threading.Lock()
    state = {"done_bytes": 0, "done_chunks": 0}

    def _upload_one(idx: int) -> None:
        offset = idx * CHUNK_BYTES
        with open(local_file, "rb") as f:
            f.seek(offset)
            raw = f.read(CHUNK_BYTES)
        b64 = base64.b64encode(raw).decode("ascii")
        rpath = f"{root_prefix}_in_chunk_{idx:04d}.bin"
        for attempt in range(4):
            sess = _get_session()
            try:
                r = sess.put(
                    f"{base_url}/api/contents/{rpath}",
                    headers=hdrs,
                    json={"type": "file", "format": "base64", "content": b64},
                    timeout=180,
                )
                if r.ok:
                    break
                if attempt == 3:
                    raise RuntimeError(f"Error subiendo chunk {idx}: {r.status_code} {r.text[:200]}")
            except Exception as exc:
                try:
                    sess.close()
                except Exception:
                    pass
                _THREAD_LOCAL.session = requests.Session()
                if attempt == 3:
                    raise RuntimeError(f"Excepción subiendo chunk {idx}: {exc}") from exc
            time.sleep(1.5 * (attempt + 1))
        with state_lock:
            state["done_bytes"] += len(raw)
            state["done_chunks"] += 1
            db = state["done_bytes"]
            dc = state["done_chunks"]
        pct = 22 + int(16 * dc / n_chunks)
        update_telem(
            detail=f"Chunk {dc}/{n_chunks} ({db / (1024*1024):.1f}/{total_mb:.1f} MB)",
            progress_pct=pct,
            xfer_done_mb=round(db / (1024 * 1024), 1),
        )
        log_line(f"  Upload chunk {dc}/{n_chunks} (#{idx}) OK ({db / (1024*1024):.1f} MB)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_upload_one, i) for i in range(n_chunks)]
        for fut in concurrent.futures.as_completed(futs):
            fut.result()
    return root_prefix, n_chunks, sha


def build_remote_kernel_script(n_chunks: int, expected_sha: str) -> str:
    return f"""
import glob, hashlib, json, os, shutil, subprocess, tarfile, threading, time
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
            gname = smi[0].strip()
            gutil = float(smi[1].strip())
            vused = float(smi[2].strip()) / 1024.0
            vtot = float(smi[3].strip()) / 1024.0
            gtemp = float(smi[4].strip())
            gpow = float(smi[5].strip())
            mem = {{}}
            for ln in Path("/proc/meminfo").read_text().splitlines():
                k, v = ln.split(":", 1)
                mem[k.strip()] = int(v.strip().split()[0])
            rtot = mem["MemTotal"] / (1024 * 1024)
            rused = (mem["MemTotal"] - mem.get("MemAvailable", mem["MemFree"])) / (1024 * 1024)
            du = shutil.disk_usage("/content")
            payload = {{
                "gpu_name": gname,
                "gpu_util_pct": round(gutil, 1),
                "vram_used_gib": round(vused, 2),
                "vram_total_gib": round(vtot, 2),
                "gpu_temp_c": round(gtemp, 1),
                "gpu_power_w": round(gpow, 1),
                "ram_used_gib": round(rused, 2),
                "ram_total_gib": round(rtot, 2),
                "disk_used_gib": round(du.used / (1024**3), 2),
                "disk_total_gib": round(du.total / (1024**3), 2),
                "disk_free_gib": round(du.free / (1024**3), 2),
            }}
            print("[TELEM] " + json.dumps(payload), flush=True)
        except Exception as e:
            print("[TELEM_ERR]", e, flush=True)
        stop_telem.wait(2.0)

t_th = threading.Thread(target=_telem_loop, daemon=True)
t_th.start()

try:
    base_dir = Path("/content")
    if not (base_dir / "_in_chunk_0000.bin").exists() and Path("_in_chunk_0000.bin").exists():
        base_dir = Path.cwd()
    run_dir = Path("/content/run_v05")
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    in_tar = Path("/content/passB_in.tar")
    h = hashlib.sha256()
    with open(in_tar, "wb") as out_f:
        for i in range({n_chunks}):
            cp = base_dir / f"_in_chunk_{{i:04d}}.bin"
            data = cp.read_bytes()
            h.update(data)
            out_f.write(data)
            cp.unlink()
    got_sha = h.hexdigest()
    assert got_sha == "{expected_sha}", f"SHA mismatch {{got_sha}} != {expected_sha}"
    print("[REMOTE] passB_in.tar SHA256 OK", flush=True)

    with tarfile.open(in_tar, "r:*") as tf:
        tf.extractall(run_dir)
    in_tar.unlink()

    bdir = Path("/content/blender-5.3.0-linux-x64")
    bbin = bdir / "blender"
    if not bbin.exists():
        print("[STAGE] Descargando Blender 5.3.0 Linux x64...", flush=True)
        tgz = Path("/content/blender53.tar.xz")
        subprocess.run(["wget", "-q", "-O", str(tgz), "https://download.blender.org/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz"], check=True)
        subprocess.run(["tar", "-xf", str(tgz), "-C", "/content"], check=True)
        tgz.unlink()
    subprocess.run("apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y libxrender1 libxi6 libxxf86vm1 libxfixes3 libgl1 libxkbcommon0 > /dev/null", shell=True, check=True)
    print("[STAGE] Blender 5.3.0 listo, arrancando Pass B 8K en OptiX/CUDA...", flush=True)

    cmd = [
        str(bbin), "-b", "--python-exit-code", "1",
        "--python", str(run_dir / "tools/blender/catan_hero360_bake.py"),
        "--",
        "--cameras", str(run_dir / "work_v05/cameras.json"),
        "--out-dir", str(run_dir / "work_v05"),
        "--macro-dir", str(run_dir / "work_v05/macro"),
        "--bake-res", "8192",
        "--normal-res", "8192",
        "--rough-res", "4096",
        "--bake-samples", "24",
        "--render-samples", "48",
        "--device", "gpu",
        "--stage", "passB",
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, errors="replace")
    for ln in p.stdout:
        print("[BLENDER] " + ln.rstrip(), flush=True)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Blender passB falló con código {{rc}}")

    print("[STAGE] Empaquetando artefactos de salida Pass B...", flush=True)
    out_tar = Path("/content/passB_out.tar.gz")
    w = run_dir / "work_v05"
    cand_arcs = [
        "tile_master.stl",
        "tile_viewer.glb",
        "baked_albedo.png",
        "baked_normal.png",
        "baked_roughness.png",
        "macro_hp_height_8k.png",
        "macro/ao_cavity_8k.png",
        "audit_6views_pbr.png",
        "audit_6views_clay.png",
        "audit_ortho.png",
    ]
    existing_arcs = []
    for arc in cand_arcs:
        src = w / arc
        if src.exists():
            existing_arcs.append(arc)
            print(f"[OUT_FILE] {{arc}} = {{src.stat().st_size / (1024*1024):.2f}} MB", flush=True)
    subprocess.run(["tar", "--use-compress-program=gzip -1", "-cf", str(out_tar), "-C", str(w)] + existing_arcs, check=True)

    c_bytes = 8 * 1024 * 1024
    h_out = hashlib.sha256()
    idx = 0
    with open(out_tar, "rb") as f:
        while True:
            buf = f.read(c_bytes)
            if not buf:
                break
            h_out.update(buf)
            (base_dir / f"_out_chunk_{{idx:04d}}.bin").write_bytes(buf)
            idx += 1
    out_sz = out_tar.stat().st_size
    out_tar.unlink()
    print("OUT_MANIFEST:" + json.dumps({{"chunks": idx, "bytes": out_sz, "sha256": h_out.hexdigest()}}), flush=True)
finally:
    stop_telem.set()
"""


def download_chunks_from_colab(creds: Any, proj: str | None, rt_name: str, root_prefix: str, manifest: Dict[str, Any], out_tar: Path) -> None:
    n_chunks = int(manifest["chunks"])
    total_mb = manifest["bytes"] / (1024 * 1024)
    update_telem(phase="Descargando resultados Pass B", detail=f"0 / {n_chunks} bloques", progress_pct=82, xfer_done_mb=0.0, xfer_total_mb=round(total_mb, 1))
    log_line(f"Descargando {total_mb:.1f} MB en {n_chunks} chunks ({MAX_WORKERS} hilos) desde Colab L4...")

    conn = cr.get_runtime(creds, proj, rt_name)["connectionInfo"]
    base_url = conn["url"].rstrip("/")
    hdrs = _proxy_headers(conn)
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_tar.parent
    state_lock = threading.Lock()
    state = {"done_bytes": 0, "done_chunks": 0}

    def _download_one(idx: int) -> Path:
        rpath = f"{root_prefix}_out_chunk_{idx:04d}.bin"
        r = None
        for attempt in range(4):
            sess = _get_session()
            try:
                r = sess.get(f"{base_url}/api/contents/{rpath}?content=1", headers=hdrs, timeout=180)
                if r.ok:
                    break
                if attempt == 3:
                    raise RuntimeError(f"Error descargando chunk {idx}: {r.status_code} {r.text[:200]}")
            except Exception as exc:
                try:
                    sess.close()
                except Exception:
                    pass
                _THREAD_LOCAL.session = requests.Session()
                if attempt == 3:
                    raise RuntimeError(f"Excepción descargando chunk {idx}: {exc}") from exc
            time.sleep(1.5 * (attempt + 1))
        assert r is not None
        raw = base64.b64decode(r.json()["content"])
        part_p = tmp_dir / f"_dl_{idx:04d}.bin"
        part_p.write_bytes(raw)
        with state_lock:
            state["done_bytes"] += len(raw)
            state["done_chunks"] += 1
            db = state["done_bytes"]
            dc = state["done_chunks"]
        pct = 82 + int(12 * dc / n_chunks)
        update_telem(
            detail=f"Chunk {dc}/{n_chunks} ({db / (1024*1024):.1f}/{total_mb:.1f} MB)",
            progress_pct=pct,
            xfer_done_mb=round(db / (1024 * 1024), 1),
        )
        log_line(f"  Download chunk {dc}/{n_chunks} (#{idx}) OK ({db / (1024*1024):.1f} MB)")
        return part_p

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_download_one, i) for i in range(n_chunks)]
        for fut in concurrent.futures.as_completed(futs):
            fut.result()

    h = hashlib.sha256()
    with open(out_tar, "wb") as out_f:
        for i in range(n_chunks):
            part_p = tmp_dir / f"_dl_{i:04d}.bin"
            data = part_p.read_bytes()
            h.update(data)
            out_f.write(data)
            part_p.unlink(missing_ok=True)

    if h.hexdigest() != manifest["sha256"]:
        raise RuntimeError(f"SHA256 descarga inválido: {h.hexdigest()} != {manifest['sha256']}")
    log_line("SHA256 verificado OK. Extrayendo en work_v05/ ...")
    with tarfile.open(out_tar, "r:gz") as tf:
        tf.extractall(WORK_DIR)
    out_tar.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-f53", action="store_true")
    p.add_argument("--zstd-blend", default=None)
    args = p.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CONSOLE_PATH.write_text("", encoding="utf-8")
    update_telem(phase="F5.3 Iniciando", detail="Alineación macro High-Pass", progress_pct=2)

    if args.skip_f53:
        rep = json.loads((MACRO_DIR / "macro_report.json").read_text(encoding="utf-8"))
        grid_p = build_macro_grid_16()
        log_line(f"F5.3 (--skip-f53) verificado: accepted={rep['accepted']}/{rep['accepted'] + rep['rejected']} | Grid={grid_p.name}")
        if int(rep.get("accepted", 0)) < 14:
            raise RuntimeError(f"Demasiados parches rechazados ({rep['accepted']}/16 < 14). Abortando antes de usar Colab L4.")
        update_telem(detail=f"Alineados {rep['accepted']}/16 parches", progress_pct=15)
    else:
        run_f53_align()

    in_tar = SCRATCH_DIR / "passB_in.tar"
    zstd_path = Path(args.zstd_blend) if args.zstd_blend else None
    package_passb_inputs(in_tar, blend_src=zstd_path)

    creds = cr.get_credentials(interactive=False)
    proj = cr._read_project_id()
    specs = cr.list_runtime_specs(creds, proj)
    candidates = cr.rank_gpu_specs(specs, accel_req="auto", prefer_highmem=True)

    update_telem(phase="Conectando Colab Pro GPU", detail="Provisionando NVIDIA L4...", progress_pct=20)
    rt = None
    for spec_key in candidates:
        log_line(f"Solicitando runtime Colab: {spec_key} ...")
        try:
            rt = cr.create_runtime(creds, proj, spec_key)
            break
        except Exception as exc:
            log_line(f"  Fallo {spec_key.get('accelerator')}: {exc}")
            if "TOO_MANY_ACTIVE_RUNTIMES" in str(exc):
                raise
    if rt is None:
        raise RuntimeError("No se pudo provisionar ninguna GPU elegible en Colab Pro.")

    rt_name = rt["name"]
    acc_name = rt.get("runtimeSpec", {}).get("accelerator", "GPU")
    update_telem(runtime_active=True, runtime_status=f"CONECTADO: NVIDIA {acc_name}", gpu_name=f"NVIDIA {acc_name}")
    log_line(f"Runtime conectado: {rt_name} ({acc_name})")

    try:
        root_prefix, n_chunks, sha_in = upload_chunks_to_colab(creds, proj, rt_name, in_tar)
        in_tar.unlink(missing_ok=True)

        update_telem(phase="Blender 5.3 Cycles GPU (L4)", detail="Descargando Blender 5.3 + preparando OptiX...", progress_pct=40, blender_stage="setup")
        conn = cr.get_runtime(creds, proj, rt_name)["connectionInfo"]
        remote_code = build_remote_kernel_script(n_chunks, sha_in)
        manifest_holder: Dict[str, Any] = {}

        def _on_line(stream: str, line: str) -> None:
            if line.startswith("[TELEM] "):
                try:
                    t = json.loads(line[8:])
                    update_telem(**t)
                except Exception:
                    pass
                return
            if line.startswith("OUT_MANIFEST:"):
                manifest_holder.update(json.loads(line[len("OUT_MANIFEST:"):]))
                log_line(line)
                return
            if line.startswith("[STAGE] "):
                msg = line[8:]
                update_telem(detail=msg, blender_stage=msg[:42])
            elif "[MacroHP]" in line:
                update_telem(detail=line, progress_pct=52, blender_stage="Macro High-Pass 8K")
            elif "[Sculpt3D]" in line or "[MicroHP Sculpt]" in line:
                update_telem(detail=line, progress_pct=62, blender_stage="Esculpido 3D 6.15M")
            elif "Cycles NORMAL" in line:
                update_telem(detail="Horneando NORMAL 8K (OptiX)", progress_pct=68, blender_stage="Bake Normal 8K")
            elif "Cycles ROUGHNESS" in line:
                update_telem(detail="Horneando ROUGHNESS 4K (OptiX)", progress_pct=74, blender_stage="Bake Roughness 4K")
            elif "[AuditRender]" in line or "Saved:" in line:
                update_telem(detail="Renderizando 6 vistas de auditoría", progress_pct=78, blender_stage="Audit Renders")
            log_line(line)

        rc = cr.execute_jupyter_code(conn, remote_code, timeout_s=2400, line_cb=_on_line)
        if rc != 0 or not manifest_holder:
            raise RuntimeError(f"Ejecución remota Pass B falló (rc={rc}, manifest={manifest_holder})")

        out_tar = SCRATCH_DIR / "passB_out.tar.gz"
        download_chunks_from_colab(creds, proj, rt_name, root_prefix, manifest_holder, out_tar)
        update_telem(phase="F6 Completado en L4", detail="Apagando runtime y validando localmente...", progress_pct=96, blender_stage="Completado")
    finally:
        log_line(f"Apagando runtime {rt_name} (DeleteRuntime)...")
        try:
            cr.delete_runtime(creds, proj, rt_name)
        except Exception as e:
            log_line(f"Aviso DeleteRuntime: {e}")
        rem = cr.list_runtimes(creds, proj)
        log_line(f"Runtimes activos restantes: {len(rem)}")
        update_telem(
            runtime_active=False,
            runtime_status="APAGADO (DeleteRuntime OK — 0 unidades consumiéndose)",
        )


if __name__ == "__main__":
    main()
