"""
Staging previo en Google Drive (scope mínimo drive.file) para transferir insumos pesados
al runtime de Google Colab Pro con la GPU apagada (0 créditos de cómputo consumidos).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"
DRIVE_API = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD_API = "https://www.googleapis.com/upload/drive/v3"
STAGING_FOLDER_NAME = "Maker3D_Colab_Staging"
UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024  # 16 MB (múltiplo estricto de 256 KiB requerido por Drive API)
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024


def _auth_token(creds: Credentials) -> str:
    if (not creds.valid or creds.expired) and creds.refresh_token:
        creds.refresh(Request())
    return str(creds.token)


def has_drive_scope(creds: Credentials, token_path: Optional[Path] = None) -> bool:
    """Comprueba si el token OAuth dispone de drive.file o drive y responde /drive/v3/files."""
    scopes = set(getattr(creds, "scopes", None) or [])
    if token_path and token_path.exists():
        try:
            raw = json.loads(token_path.read_text(encoding="utf-8"))
            for s in raw.get("scopes") or []:
                scopes.add(s)
        except Exception:
            pass
    if not any(s.endswith("/auth/drive.file") or s.endswith("/auth/drive") for s in scopes):
        return False
    try:
        tok = _auth_token(creds)
        r = requests.get(
            f"{DRIVE_API}/files",
            headers={"Authorization": f"Bearer {tok}"},
            params={"pageSize": 1, "fields": "files(id)"},
            timeout=(10, 20),
        )
        return r.status_code == 200
    except Exception:
        return False


def ensure_staging_folder(creds: Credentials, folder_name: str = STAGING_FOLDER_NAME) -> str:
    """Busca o crea la carpeta dedicada de staging en Google Drive y devuelve su folder_id."""
    tok = _auth_token(creds)
    hdrs = {"Authorization": f"Bearer {tok}"}
    q = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    r = requests.get(
        f"{DRIVE_API}/files",
        headers=hdrs,
        params={"q": q, "pageSize": 5, "fields": "files(id,name)"},
        timeout=(10, 25),
    )
    r.raise_for_status()
    files = r.json().get("files", [])
    if files:
        return str(files[0]["id"])

    cr = requests.post(
        f"{DRIVE_API}/files",
        headers={**hdrs, "Content-Type": "application/json"},
        json={"name": folder_name, "mimeType": "application/vnd.google-apps.folder"},
        timeout=(10, 25),
    )
    cr.raise_for_status()
    return str(cr.json()["id"])


def upload_staging_file(
    creds: Credentials,
    local_path: Path,
    folder_id: Optional[str] = None,
    remote_name: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Sube un archivo a Drive mediante resumable upload (16 MB chunks) calculando SHA256 al vuelo."""
    fid = folder_id or ensure_staging_folder(creds)
    total = local_path.stat().st_size
    name = remote_name or local_path.name
    tok = _auth_token(creds)

    init_r = requests.post(
        f"{DRIVE_UPLOAD_API}/files?uploadType=resumable&fields=id,name,size",
        headers={
            "Authorization": f"Bearer {tok}",
            "Content-Type": "application/json; charset=utf-8",
            "X-Upload-Content-Type": "application/octet-stream",
            "X-Upload-Content-Length": str(total),
        },
        json={"name": name, "parents": [fid]},
        timeout=(10, 30),
    )
    init_r.raise_for_status()
    upload_url = init_r.headers["Location"]

    h = hashlib.sha256()
    sent = 0
    file_meta: Dict[str, Any] = {}
    with open(local_path, "rb") as f:
        while sent < total:
            chunk = f.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
            end = sent + len(chunk) - 1
            hdrs = {
                "Content-Length": str(len(chunk)),
                "Content-Range": f"bytes {sent}-{end}/{total}",
            }
            resp = None
            for attempt in range(4):
                try:
                    resp = requests.put(upload_url, headers=hdrs, data=chunk, timeout=(15, 120))
                    if resp.status_code in (200, 201, 308):
                        break
                except requests.RequestException:
                    pass
                if attempt == 3:
                    code = getattr(resp, "status_code", "timeout")
                    raise RuntimeError(f"Error Drive resumable upload offset={sent} ({code})")
            assert resp is not None
            if resp.status_code in (200, 201):
                file_meta = resp.json()
            sent += len(chunk)
            if progress_cb:
                progress_cb(sent, total)

    return {
        "file_id": str(file_meta["id"]),
        "folder_id": fid,
        "name": str(file_meta.get("name", name)),
        "bytes": total,
        "sha256": h.hexdigest(),
    }


def download_staging_file(
    creds: Credentials,
    file_id: str,
    out_path: Path,
    expected_sha256: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Descarga en streaming desde Google Drive verificando opcionalmente SHA256."""
    tok = _auth_token(creds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    done = 0
    total = 0

    for attempt in range(4):
        try:
            with requests.get(
                f"{DRIVE_API}/files/{file_id}",
                headers={"Authorization": f"Bearer {tok}"},
                params={"alt": "media"},
                stream=True,
                timeout=(15, 120),
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", "0") or "0")
                h = hashlib.sha256()
                done = 0
                with open(out_path, "wb") as f:
                    for buf in r.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                        if not buf:
                            continue
                        f.write(buf)
                        h.update(buf)
                        done += len(buf)
                        if progress_cb and total > 0:
                            progress_cb(done, total)
            break
        except Exception:
            if attempt == 3:
                raise
            tok = _auth_token(creds)

    if expected_sha256 and h.hexdigest() != expected_sha256:
        raise RuntimeError(f"SHA256 Drive download mismatch: {h.hexdigest()} != {expected_sha256}")
    return out_path


def delete_staging_file(creds: Credentials, file_id: str) -> None:
    """Elimina un artefacto temporal de staging en Google Drive (ignora 404)."""
    if not file_id:
        return
    try:
        tok = _auth_token(creds)
        r = requests.delete(
            f"{DRIVE_API}/files/{file_id}",
            headers={"Authorization": f"Bearer {tok}"},
            timeout=(10, 20),
        )
        if r.status_code not in (200, 204, 404):
            r.raise_for_status()
    except Exception:
        pass


def build_remote_drive_helpers_code() -> str:
    """Devuelve helpers autónomos para descargar/subir dentro de Colab por red interna Drive."""
    return '''
def _drive_download(token, file_id, dst_path, expected_sha=None):
    import hashlib, requests
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    hdrs = {"Authorization": f"Bearer {token}"}
    h = hashlib.sha256()
    with requests.get(url, headers=hdrs, stream=True, timeout=(15, 120)) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    h.update(chunk)
    got = h.hexdigest()
    if expected_sha and got != expected_sha:
        raise RuntimeError(f"Drive SHA mismatch {got} != {expected_sha}")
    return got

def _drive_upload(token, src_path, folder_id, name):
    import hashlib, os, requests
    total = os.path.getsize(src_path)
    init = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id,name,size",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
            "X-Upload-Content-Type": "application/octet-stream",
            "X-Upload-Content-Length": str(total),
        },
        json={"name": name, "parents": [folder_id]},
        timeout=(15, 30),
    )
    init.raise_for_status()
    u_url = init.headers["Location"]
    h = hashlib.sha256()
    sent = 0
    meta = {}
    with open(src_path, "rb") as f:
        while sent < total:
            buf = f.read(32 * 1024 * 1024)
            if not buf:
                break
            h.update(buf)
            end = sent + len(buf) - 1
            r = requests.put(
                u_url,
                headers={"Content-Length": str(len(buf)), "Content-Range": f"bytes {sent}-{end}/{total}"},
                data=buf,
                timeout=(15, 120),
            )
            if r.status_code not in (200, 201, 308):
                raise RuntimeError(f"Drive upload failed: {r.status_code} {r.text[:200]}")
            if r.status_code in (200, 201):
                meta = r.json()
            sent += len(buf)
    return {"file_id": str(meta["id"]), "bytes": total, "sha256": h.hexdigest()}
'''
