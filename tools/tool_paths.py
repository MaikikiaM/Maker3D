"""
Autodescubrimiento centralizado de ejecutables, bibliotecas, perfiles de laminado
y configuraciones MCP para Maker3D 2.0 en Windows.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PROGRAMS = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs"
USER_HOME = Path.home()
APPDATA_DIR = Path(os.environ.get("APPDATA", ""))


def _first_existing(candidates: list[Optional[Path]]) -> Optional[str]:
    for c in candidates:
        if c and c.exists():
            return str(c.resolve())
    return None


def find_blender() -> Optional[str]:
    which = shutil.which("blender")
    candidates: list[Optional[Path]] = [
        LOCAL_PROGRAMS / "Blender5.3" / "blender-5.3.0-windows-x64" / "blender.exe",
    ]
    if LOCAL_PROGRAMS.exists():
        for match in sorted(LOCAL_PROGRAMS.glob("Blender*/**/blender.exe"), reverse=True):
            candidates.append(match)
    if which:
        candidates.append(Path(which))
    return _first_existing(candidates)


def find_openscad() -> Dict[str, Optional[str]]:
    candidates_com: list[Optional[Path]] = [
        LOCAL_PROGRAMS / "OpenSCAD" / "OpenSCAD-2026.11.21-x86-64" / "openscad.com",
    ]
    candidates_exe: list[Optional[Path]] = [
        LOCAL_PROGRAMS / "OpenSCAD" / "OpenSCAD-2026.11.21-x86-64" / "openscad.exe",
        Path(r"C:\Program Files\OpenSCAD\openscad.exe"),
    ]
    if LOCAL_PROGRAMS.exists():
        for match in sorted(LOCAL_PROGRAMS.glob("OpenSCAD/**/openscad.com"), reverse=True):
            candidates_com.append(match)
        for match in sorted(LOCAL_PROGRAMS.glob("OpenSCAD/**/openscad.exe"), reverse=True):
            candidates_exe.append(match)
    which_com = shutil.which("openscad.com")
    which_exe = shutil.which("openscad")
    if which_com:
        candidates_com.append(Path(which_com))
    if which_exe:
        candidates_exe.append(Path(which_exe))
    com_path = _first_existing(candidates_com)
    exe_path = _first_existing(candidates_exe)
    return {
        "openscad_com": com_path,
        "openscad_exe": exe_path,
        "openscad_cli": com_path or exe_path,
    }


def find_bosl2() -> Optional[str]:
    candidates: list[Optional[Path]] = [
        USER_HOME / "Documents" / "OpenSCAD" / "libraries" / "BOSL2",
    ]
    env_path = os.environ.get("OPENSCADPATH", "")
    for entry in env_path.split(";"):
        if entry.strip():
            candidates.append(Path(entry.strip()) / "BOSL2")
    for c in candidates:
        if c and (c / "std.scad").exists():
            return str(c.resolve())
    return None


def find_bambu_studio() -> Dict[str, Any]:
    exe = _first_existing([
        Path(r"C:\Program Files\Bambu Studio\bambu-studio.exe"),
        Path(shutil.which("bambu-studio")) if shutil.which("bambu-studio") else None,
    ])
    bbl_dir = Path(r"C:\Program Files\Bambu Studio\resources\profiles\BBL")
    profiles: Dict[str, Dict[str, Optional[str]]] = {}
    if bbl_dir.exists():
        spec_map = {
            "a1": {
                "machine": [bbl_dir / "machine" / "Bambu Lab A1 0.4 nozzle.json"],
                "process": [bbl_dir / "process" / "0.20mm Standard @BBL A1.json"],
                "filament": [bbl_dir / "filament" / "Bambu PLA Basic @BBL A1.json"],
            },
            "p1s": {
                "machine": [bbl_dir / "machine" / "Bambu Lab P1S 0.4 nozzle.json"],
                "process": [
                    bbl_dir / "process" / "0.20mm Standard @BBL P1S.json",
                    bbl_dir / "process" / "0.20mm Standard @BBL X1C.json",
                ],
                "filament": [
                    bbl_dir / "filament" / "Bambu PLA Basic @BBL P1S 0.4 nozzle.json",
                    bbl_dir / "filament" / "Bambu PLA Basic @BBL X1C.json",
                ],
            },
            "x1c": {
                "machine": [bbl_dir / "machine" / "Bambu Lab X1 Carbon 0.4 nozzle.json"],
                "process": [bbl_dir / "process" / "0.20mm Standard @BBL X1C.json"],
                "filament": [bbl_dir / "filament" / "Bambu PLA Basic @BBL X1C.json"],
            },
        }
        for key, group in spec_map.items():
            profiles[key] = {
                role: _first_existing(paths) for role, paths in group.items()
            }
    return {
        "bambu_studio_exe": exe,
        "bbl_profiles_dir": str(bbl_dir) if bbl_dir.exists() else None,
        "profiles": profiles,
    }


def find_uv() -> Dict[str, Optional[str]]:
    local_app = Path(os.environ.get("LOCALAPPDATA", ""))
    winget_glob = glob.glob(str(local_app / "Microsoft" / "WinGet" / "Packages" / "astral-sh.uv*" / "uv.exe"))
    uv_candidates: list[Optional[Path]] = [
        Path(shutil.which("uv")) if shutil.which("uv") else None,
        *[Path(p) for p in winget_glob],
        USER_HOME / ".local" / "bin" / "uv.exe",
    ]
    uv_exe = _first_existing(uv_candidates)
    uvx_exe = _first_existing([
        Path(uv_exe).with_name("uvx.exe") if uv_exe else None,
        Path(shutil.which("uvx")) if shutil.which("uvx") else None,
    ])
    return {"uv": uv_exe, "uvx": uvx_exe}


def find_webgl_tools() -> Dict[str, Optional[str]]:
    bin_dir = REPO_ROOT / "tools" / "webgl" / "node_modules" / ".bin"
    gltf_transform = _first_existing([
        bin_dir / "gltf-transform.cmd",
        Path(shutil.which("gltf-transform.cmd")) if shutil.which("gltf-transform.cmd") else None,
    ])
    npm_cmd = _first_existing([
        Path(shutil.which("npm.cmd")) if shutil.which("npm.cmd") else None,
        Path(r"C:\Program Files\nodejs\npm.cmd"),
    ])
    return {
        "gltf_transform_cmd": gltf_transform,
        "npm_cmd": npm_cmd,
    }


def find_mcp_paths() -> Dict[str, Optional[str]]:
    mcp_cfg = USER_HOME / ".gemini" / "config" / "mcp_config.json"
    addons_dir = APPDATA_DIR / "Blender Foundation" / "Blender" / "5.3" / "scripts" / "addons"
    local_bin = USER_HOME / ".local" / "bin"
    return {
        "mcp_config": str(mcp_cfg) if mcp_cfg.exists() else None,
        "blender_addons_dir": str(addons_dir) if addons_dir.exists() else None,
        "blender_mcp_addon": str(addons_dir / "blender_mcp.py") if (addons_dir / "blender_mcp.py").exists() else None,
        "build123d_mcp": _first_existing([local_bin / "build123d-mcp.exe"]),
        "openscad_mcp": _first_existing([local_bin / "openscad-mcp.exe"]),
        "mcp_for_blender": _first_existing([local_bin / "mcp-for-blender.exe"]),
    }


def discover_all() -> Dict[str, Any]:
    openscad = find_openscad()
    bambu = find_bambu_studio()
    uv_info = find_uv()
    webgl = find_webgl_tools()
    mcp = find_mcp_paths()
    return {
        "repo_root": str(REPO_ROOT),
        "blender_exe": find_blender(),
        **openscad,
        "bosl2_dir": find_bosl2(),
        **bambu,
        **uv_info,
        **webgl,
        **mcp,
        "dotenv_file": str(REPO_ROOT / ".env") if (REPO_ROOT / ".env").exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Autodescubrimiento de rutas Maker3D 2.0")
    parser.add_argument("--json", action="store_true", help="Imprimir salida en JSON")
    args = parser.parse_args()
    info = discover_all()
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        for k, v in info.items():
            if k == "profiles":
                print(f"bambu_profiles: {list(v.keys())}")
            else:
                print(f"{k}: {v}")


if __name__ == "__main__":
    main()
