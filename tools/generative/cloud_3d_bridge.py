"""
Puente Image-to-3D en la nube para Maker3D 2.0.
Conecta arte conceptual de Nano Banana (image-generator) con:
  - Hugging Face ZeroGPU gratuito vía gradio_client (TRELLIS.2, Hunyuan3D-2.1, Hi3DGen)
  - fal.ai (opcional con FAL_KEY en .env)
  - Replicate (opcional con REPLICATE_API_TOKEN en .env)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env", override=False)

HF_SPACES: Dict[str, List[str]] = {
    "trellis2": ["microsoft/TRELLIS.2", "microsoft/TRELLIS"],
    "hunyuan3d": ["tencent/Hunyuan3D-2.1", "tencent/Hunyuan3D-2"],
    "hi3dgen": ["Stable-X/Hi3DGen"],
}


def _find_glb_in_result(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        if obj.lower().endswith((".glb", ".gltf", ".obj", ".ply", ".stl")):
            return obj
        return None
    if isinstance(obj, dict):
        for k in ("value", "path", "url", "name"):
            if k in obj:
                found = _find_glb_in_result(obj[k])
                if found:
                    return found
        for v in obj.values():
            found = _find_glb_in_result(v)
            if found:
                return found
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _find_glb_in_result(item)
            if found:
                return found
    return None


def _save_output_asset(src_or_url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if src_or_url.startswith(("http://", "https://")):
        urllib.request.urlretrieve(src_or_url, str(out_path))
    elif Path(src_or_url).exists():
        shutil.copy2(src_or_url, out_path)
    else:
        raise FileNotFoundError(f"Artefacto 3D devuelto no accesible: {src_or_url}")


def doctor() -> Dict[str, Any]:
    import gradio_client
    import fal_client
    import replicate

    hf_tok = bool(os.environ.get("HF_TOKEN", "").strip())
    fal_key = bool(os.environ.get("FAL_KEY", "").strip())
    rep_tok = bool(os.environ.get("REPLICATE_API_TOKEN", "").strip())

    # Chequeo ligero de disponibilidad HTTP de la API pública de HF Spaces
    hf_spaces_status: Dict[str, str] = {}
    for model_key, spaces in HF_SPACES.items():
        space_id = spaces[0]
        try:
            req = urllib.request.Request(
                f"https://huggingface.co/api/spaces/{space_id}",
                headers={"User-Agent": "Maker3D-2.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                info = json.loads(r.read().decode("utf-8"))
                stage = info.get("runtime", {}).get("stage", "unknown")
                hf_spaces_status[f"{model_key} ({space_id})"] = stage
        except Exception as exc:
            hf_spaces_status[f"{model_key} ({space_id})"] = f"unreachable ({exc})"

    return {
        "libraries_ok": True,
        "gradio_client_version": getattr(gradio_client, "__version__", "installed"),
        "fal_client_installed": bool(fal_client),
        "replicate_installed": bool(replicate),
        "providers": {
            "hf-zero": {
                "ready": True,
                "auth": "HF_TOKEN set (higher ZeroGPU quota)" if hf_tok else "anonymous free ZeroGPU tier",
                "spaces": hf_spaces_status,
            },
            "fal": {
                "ready": fal_key,
                "auth": "FAL_KEY configured" if fal_key else "unset (optional, add FAL_KEY to .env)",
            },
            "replicate": {
                "ready": rep_tok,
                "auth": "REPLICATE_API_TOKEN configured" if rep_tok else "unset (optional, add REPLICATE_API_TOKEN to .env)",
            },
        },
    }


def run_hf_zero(model: str, image_path: Path, out_path: Path, seed: int = 42) -> Path:
    from gradio_client import Client, handle_file

    candidates = HF_SPACES.get(model)
    if not candidates:
        raise ValueError(f"Modelo hf-zero desconocido: {model}. Usa: {list(HF_SPACES.keys())}")
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    last_err: Optional[Exception] = None
    for space_id in candidates:
        try:
            print(f"[hf-zero] Conectando con Space {space_id} ...")
            client = Client(space_id, token=hf_token)
            api_info = client.view_api(return_format="dict")
            named = api_info.get("named_endpoints", {})

            # Descubrir endpoint principal de generación 3D
            ep_candidates = [
                ep for ep in named.keys()
                if any(k in ep.lower() for k in ("image_to_3d", "gen", "shape", "run", "extract_glb", "predict"))
            ] or list(named.keys())

            for ep in ep_candidates:
                params = named[ep].get("parameters", [])
                args: List[Any] = []
                used_img = False
                for p in params:
                    pname = (p.get("parameter_name") or p.get("label") or "").lower()
                    ptype = str(p.get("type") or p.get("python_type", {}).get("type", "")).lower()
                    if not used_img and ("image" in pname or "filepath" in ptype or "imagedata" in ptype):
                        args.append(handle_file(str(image_path)))
                        used_img = True
                    elif "seed" in pname:
                        args.append(seed)
                    elif p.get("parameter_has_default"):
                        args.append(p.get("parameter_default"))
                    else:
                        args.append(None)
                if not used_img and params:
                    args[0] = handle_file(str(image_path))
                res = client.predict(*args, api_name=ep)
                glb = _find_glb_in_result(res)
                if glb:
                    _save_output_asset(glb, out_path)
                    print(f"[hf-zero] Guardado GLB desde {space_id}{ep} -> {out_path}")
                    return out_path
        except Exception as exc:
            last_err = exc
            print(f"[hf-zero] Aviso en {space_id}: {exc}")
    raise RuntimeError(f"No se pudo completar hf-zero ({model}): {last_err}")


def run_fal(model: str, image_path: Path, out_path: Path) -> Path:
    import fal_client

    if not os.environ.get("FAL_KEY", "").strip():
        raise RuntimeError("FAL_KEY no está configurada en .env")

    endpoint_map = {
        "trellis": "fal-ai/trellis",
        "trellis2": "fal-ai/trellis",
        "hunyuan3d": "fal-ai/hunyuan3d/v2",
        "meshy": "fal-ai/meshy/v4/image-to-3d",
    }
    ep = endpoint_map.get(model)
    if not ep:
        raise ValueError(f"Modelo fal desconocido: {model}. Usa: {list(endpoint_map.keys())}")

    img_url = fal_client.upload_file(str(image_path))
    print(f"[fal] Ejecutando {ep} ...")
    res = fal_client.subscribe(ep, arguments={"image_url": img_url})
    glb = _find_glb_in_result(res)
    if not glb:
        raise RuntimeError(f"Respuesta fal sin URL de malla: {res}")
    _save_output_asset(glb, out_path)
    print(f"[fal] Guardado -> {out_path}")
    return out_path


def run_replicate(image_path: Path, out_path: Path) -> Path:
    import replicate

    if not os.environ.get("REPLICATE_API_TOKEN", "").strip():
        raise RuntimeError("REPLICATE_API_TOKEN no está configurada en .env")
    with open(image_path, "rb") as f:
        res = replicate.run(
            "firtoz/trellis:4876f2a8da1c544772dffa32e8889da4a1bab3a1f5c1937bfcfccb99ae347251",
            input={"images": [f], "generate_model": True},
        )
    glb = _find_glb_in_result(res)
    if not glb:
        raise RuntimeError(f"Respuesta Replicate sin GLB: {res}")
    _save_output_asset(glb, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Puente Image-to-3D en la nube para Maker3D 2.0")
    parser.add_argument("--doctor", action="store_true", help="Verificar dependencias y estado de proveedores")
    parser.add_argument("--provider", choices=["hf-zero", "fal", "replicate"], default="hf-zero")
    parser.add_argument("--model", default="trellis2")
    parser.add_argument("--image", help="Imagen de referencia limpia (Nano Banana)")
    parser.add_argument("--out", help="Ruta de salida .glb")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.doctor:
        info = doctor()
        print(json.dumps(info, indent=2, ensure_ascii=False))
        sys.exit(0 if info.get("libraries_ok") else 1)

    if not args.image or not args.out:
        parser.error("Se requieren --image y --out cuando no se usa --doctor")

    img_p = Path(args.image)
    out_p = Path(args.out)
    if not img_p.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {img_p}")

    if args.provider == "hf-zero":
        run_hf_zero(args.model, img_p, out_p, seed=args.seed)
    elif args.provider == "fal":
        run_fal(args.model, img_p, out_p)
    else:
        run_replicate(img_p, out_p)


if __name__ == "__main__":
    main()
