"""
Cliente CLI oficial de Google Colab API (colaboratory.googleapis.com v1beta + v1)
para aprovisionar GPUs/TPUs de Colab Pro de forma headless y ejecutar cargas Blender/Python
por WebSocket Jupyter sin navegador tras una única autorización OAuth2 local.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
import websocket
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/colaboratory"]
COLAB_V1 = "https://colaboratory.googleapis.com/v1"
COLAB_V1BETA = "https://colaboratory.googleapis.com/v1beta"

SECRETS_DIR = Path.home() / ".gemini" / "secrets"
DEFAULT_CLIENT_CANDIDATES = [
    SECRETS_DIR / "colab-oauth.json",
    SECRETS_DIR / "colab-auth.json.json",
    SECRETS_DIR / "colab-auth.json",
]
DEFAULT_TOKEN_PATH = SECRETS_DIR / "colab-token.json"

# Prioridad uso-calidad por defecto para Blender Cycles OptiX/CUDA en Colab Pro
GPU_PREFERENCE_ORDER = [
    "L4",
    "A100",
    "G4",
    "RTX_PRO_6000",
    "BLACKWELL_PRO_6000",
    "H100",
    "V100",
    "T4",
]


def _find_client_secret(explicit: Optional[str] = None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"No existe el archivo OAuth indicado: {p}")
    for c in DEFAULT_CLIENT_CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No se encontró JSON de credenciales OAuth en {SECRETS_DIR} "
        f"(buscado: {[c.name for c in DEFAULT_CLIENT_CANDIDATES]})."
    )


def get_credentials(
    client_secret_path: Optional[str] = None,
    token_path: Path = DEFAULT_TOKEN_PATH,
    interactive: bool = False,
) -> Credentials:
    """Carga token cacheado y lo refresca; solo abre navegador si interactive=True."""
    creds: Optional[Credentials] = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json(), encoding="utf-8")
            return creds
        except Exception as exc:
            if not interactive:
                raise RuntimeError(
                    f"El token OAuth caducó y no se pudo refrescar automáticamente ({exc}). "
                    "Ejecuta `python tools/colab/colab_api_runner.py auth`."
                ) from exc
    if not interactive:
        raise FileNotFoundError(
            f"No hay token válido en {token_path}. "
            "Ejecuta `python tools/colab/colab_api_runner.py auth` una vez para autorizar en navegador."
        )
    secret_p = _find_client_secret(client_secret_path)
    flow = InstalledAppFlow.from_client_secrets_file(str(secret_p), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def _api_headers(creds: Credentials, project_id: Optional[str] = None) -> Dict[str, str]:
    h = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
    }
    if project_id:
        h["x-goog-user-project"] = project_id
    return h


def _read_project_id(client_secret_path: Optional[str] = None) -> Optional[str]:
    try:
        p = _find_client_secret(client_secret_path)
        data = json.loads(p.read_text(encoding="utf-8"))
        for k in ("installed", "web"):
            if k in data and data[k].get("project_id"):
                return str(data[k]["project_id"])
    except Exception:
        pass
    return None


def get_subscription(creds: Credentials, project_id: Optional[str]) -> Dict[str, Any]:
    r = requests.get(f"{COLAB_V1BETA}/subscription", headers=_api_headers(creds, project_id), timeout=20)
    r.raise_for_status()
    return r.json()


def list_runtime_specs(creds: Credentials, project_id: Optional[str]) -> List[Dict[str, Any]]:
    r = requests.get(f"{COLAB_V1BETA}/runtimespecs", headers=_api_headers(creds, project_id), timeout=20)
    r.raise_for_status()
    return r.json().get("runtimeSpecs", [])


def list_runtimes(creds: Credentials, project_id: Optional[str]) -> List[Dict[str, Any]]:
    r = requests.get(f"{COLAB_V1BETA}/runtimes", headers=_api_headers(creds, project_id), timeout=20)
    r.raise_for_status()
    return r.json().get("runtimes", [])


def pick_best_gpu_spec(
    specs: List[Dict[str, Any]],
    accel_req: str = "auto",
    prefer_highmem: bool = True,
) -> Dict[str, Any]:
    """Selecciona RuntimeSpec GPU elegible según preferencia uso-calidad."""
    gpu_specs = [
        s for s in specs
        if s.get("eligible") and s.get("key", {}).get("variant") == "VARIANT_GPU"
    ]
    if not gpu_specs:
        raise RuntimeError(f"No hay runtimeSpecs GPU elegibles en esta cuenta. Recibidos: {specs}")

    if accel_req.lower() != "auto":
        target = accel_req.strip().upper()
        matched = [
            s for s in gpu_specs
            if target in s.get("key", {}).get("accelerator", "").upper()
        ]
        if not matched:
            avail = [s.get("key") for s in gpu_specs]
            raise RuntimeError(f"Acelerador '{accel_req}' no disponible. Elegibles: {avail}")
        pool = matched
    else:
        pool = gpu_specs

    def _score(s: Dict[str, Any]) -> tuple:
        k = s.get("key", {})
        acc = k.get("accelerator", "").upper()
        shape = k.get("shape", "")
        if accel_req.lower() == "auto":
            rank = len(GPU_PREFERENCE_ORDER) + 10
            for idx, pref in enumerate(GPU_PREFERENCE_ORDER):
                if pref in acc:
                    rank = idx
                    break
        else:
            rank = 0
        shape_pen = 0 if (prefer_highmem and shape == "SHAPE_HIGHMEM") or (not prefer_highmem and shape == "SHAPE_STANDARD") else 1
        return (rank, shape_pen, acc)

    pool.sort(key=_score)
    return pool[0]["key"]


def create_runtime(
    creds: Credentials,
    project_id: Optional[str],
    spec_key: Dict[str, Any],
    timeout_s: int = 300,
) -> Dict[str, Any]:
    payload = {"runtimeSpec": spec_key}
    r = requests.post(
        f"{COLAB_V1BETA}/runtimes",
        headers=_api_headers(creds, project_id),
        json=payload,
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"CreateRuntime HTTP {r.status_code}: {r.text}")
    op = r.json()
    if op.get("done"):
        if "error" in op:
            raise RuntimeError(f"CreateRuntime error: {op['error']}")
        return op.get("response", op)

    op_name = op["name"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        wr = requests.get(
            f"{COLAB_V1}/{op_name}",
            headers=_api_headers(creds, project_id),
            timeout=25,
        )
        wr.raise_for_status()
        cur = wr.json()
        if cur.get("done"):
            if "error" in cur:
                raise RuntimeError(f"CreateRuntime operation error: {cur['error']}")
            return cur.get("response", {})
        time.sleep(2.5)
    raise TimeoutError(f"CreateRuntime superó {timeout_s}s ({op_name})")


def delete_runtime(creds: Credentials, project_id: Optional[str], runtime_name: str) -> None:
    r = requests.delete(
        f"{COLAB_V1BETA}/{runtime_name}",
        headers=_api_headers(creds, project_id),
        timeout=20,
    )
    if r.status_code not in (200, 202, 204, 404):
        r.raise_for_status()


def execute_jupyter_code(
    conn_info: Dict[str, Any],
    code: str,
    timeout_s: int = 1800,
) -> int:
    """Crea kernel Jupyter en el runtime Colab y ejecuta código transmitiendo stdout/stderr."""
    base_url = conn_info["url"].rstrip("/")
    token = conn_info["token"]
    j_headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    kr = requests.post(f"{base_url}/api/kernels", headers=j_headers, json={"name": "python3"}, timeout=30)
    if kr.status_code in (401, 403):
        j_headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        kr = requests.post(f"{base_url}/api/kernels", headers=j_headers, json={"name": "python3"}, timeout=30)
    kr.raise_for_status()
    kid = kr.json()["id"]

    parsed = urlparse(base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed.netloc}{parsed.path}/api/kernels/{kid}/channels?token={token}"
    ws = websocket.create_connection(ws_url, header=[f"Authorization: {j_headers['Authorization']}"], timeout=timeout_s)

    msg_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex
    req_msg = {
        "header": {
            "msg_id": msg_id,
            "username": "maker3d",
            "session": session_id,
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "channel": "shell",
    }
    exit_code = 0
    try:
        ws.send(json.dumps(req_msg))
        while True:
            raw = ws.recv()
            msg = json.loads(raw)
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mtype = msg.get("msg_type")
            content = msg.get("content", {})
            if mtype == "stream":
                text = content.get("text", "")
                if content.get("name") == "stderr":
                    sys.stderr.write(text)
                    sys.stderr.flush()
                else:
                    sys.stdout.write(text)
                    sys.stdout.flush()
            elif mtype == "error":
                exit_code = 1
                tb = "\n".join(content.get("traceback", []))
                sys.stderr.write(f"\n[Colab Kernel Error] {content.get('ename')}: {content.get('evalue')}\n{tb}\n")
                sys.stderr.flush()
            elif mtype == "execute_reply":
                if content.get("status") != "ok":
                    exit_code = 1
                break
    finally:
        ws.close()
        try:
            requests.delete(f"{base_url}/api/kernels/{kid}", headers=j_headers, timeout=10)
        except Exception:
            pass
    return exit_code


def main() -> None:
    p = argparse.ArgumentParser(description="Runner autónomo Google Colab API (Blender/GPU)")
    p.add_argument("--client-secret", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("auth", help="Autorizar en navegador 1 vez y listar suscripción + GPUs")
    sub.add_parser("specs", help="Listar suscripción, GPUs elegibles y runtimes activos")

    p_run = sub.add_parser("run", help="Provisionar GPU Colab, ejecutar código/script y cerrar runtime")
    p_run.add_argument("--gpu", default="auto", help="Acelerador (auto=mejor uso-calidad, L4, A100, T4, etc.)")
    p_run.add_argument("--no-highmem", action="store_true")
    p_run.add_argument("--code", default=None, help="Código Python inline")
    p_run.add_argument("--script", default=None, help="Archivo .py local a ejecutar en Colab")
    p_run.add_argument("--keep-runtime", action="store_true", help="No borrar runtime al terminar")

    p_del = sub.add_parser("delete-all", help="Cerrar todos los runtimes Colab API activos")

    args = p.parse_args()
    proj = _read_project_id(args.client_secret)

    if args.cmd == "auth":
        creds = get_credentials(args.client_secret, interactive=True)
        print("TOKEN GUARDADO EN:", DEFAULT_TOKEN_PATH)
        print("SUBSCRIPTION:", json.dumps(get_subscription(creds, proj), indent=2))
        specs = list_runtime_specs(creds, proj)
        print("ELIGIBLE SPECS:", json.dumps([s for s in specs if s.get("eligible")], indent=2))
        return

    creds = get_credentials(args.client_secret, interactive=False)
    if args.cmd == "specs":
        print("SUBSCRIPTION:", json.dumps(get_subscription(creds, proj), indent=2))
        specs = list_runtime_specs(creds, proj)
        print("ELIGIBLE GPU SPECS:", json.dumps([s for s in specs if s.get("eligible") and s.get("key", {}).get("variant") == "VARIANT_GPU"], indent=2))
        print("ACTIVE RUNTIMES:", json.dumps(list_runtimes(creds, proj), indent=2))
        return

    if args.cmd == "delete-all":
        for rt in list_runtimes(creds, proj):
            print("Deleting:", rt["name"])
            delete_runtime(creds, proj, rt["name"])
        return

    if args.cmd == "run":
        if not args.code and not args.script:
            raise SystemExit("Debes pasar --code o --script")
        code = Path(args.script).read_text(encoding="utf-8") if args.script else args.code
        specs = list_runtime_specs(creds, proj)
        spec_key = pick_best_gpu_spec(specs, accel_req=args.gpu, prefer_highmem=(not args.no_highmem))
        print(f"[Colab] Provisionando runtime: {spec_key} ...")
        rt = create_runtime(creds, proj, spec_key)
        rt_name = rt.get("name", "")
        print(f"[Colab] Runtime activo: {rt_name}")
        rc = 1
        try:
            rc = execute_jupyter_code(rt["connectionInfo"], code)
        finally:
            if rt_name and not args.keep_runtime:
                print(f"[Colab] Cerrando runtime para ahorrar unidades: {rt_name}")
                delete_runtime(creds, proj, rt_name)
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
