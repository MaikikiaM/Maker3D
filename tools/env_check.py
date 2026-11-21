"""
Doctor integral de entorno Maker3D 2.0:
Verifica binarios CAD/3D, paquetes Python 3.12, toolchain WebGL, skills YAML,
addon Blender, configuracion MCP y handshake JSON-RPC 2.0 real.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tool_paths as tp

REQUIRED_PY_PACKAGES = [
    ("build123d", "build123d"),
    ("pymeshlab", "pymeshlab"),
    ("landlab", "landlab"),
    ("lib3mf", "lib3mf"),
    ("trimesh", "trimesh"),
    ("manifold3d", "manifold3d"),
    ("gradio_client", "gradio-client"),
    ("fal_client", "fal-client"),
    ("replicate", "replicate"),
    ("dotenv", "python-dotenv"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("PIL", "pillow"),
    ("matplotlib", "matplotlib"),
]

REQUIRED_SKILLS = [
    "maker3d-router",
    "parametric-brep-cad",
    "generative-3d-mesh",
    "terrain-geology-engine",
    "blender-sculpt-bake",
    "print-audit-and-webgl",
    "nanobanana-3d-sculpt",
]


def _run_version(cmd: List[str], timeout: int = 30) -> Tuple[bool, str]:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (res.stdout or res.stderr or "").strip().splitlines()
        first = out[0] if out else f"exit={res.returncode}"
        return res.returncode == 0, first
    except Exception as exc:
        return False, str(exc)


def _check_frontmatter(skill_md: Path) -> Tuple[bool, str]:
    if not skill_md.exists():
        return False, "missing SKILL.md"
    txt = skill_md.read_text(encoding="utf-8", errors="replace")
    m = re.match(r"^---\s*\n(.*?)\n---", txt, re.DOTALL)
    if not m:
        return False, "invalid YAML frontmatter"
    body = m.group(1)
    has_name = bool(re.search(r"^name:\s*\S+", body, re.MULTILINE))
    has_desc = bool(re.search(r"^description:\s*", body, re.MULTILINE))
    return (has_name and has_desc), ("ok" if (has_name and has_desc) else "missing name/description")


def probe_mcp_server(name: str, spec: Dict[str, Any]) -> Tuple[bool, str]:
    cmd_path = spec.get("command")
    if not cmd_path or not Path(cmd_path).exists():
        return False, f"binary missing: {cmd_path}"
    env = os.environ.copy()
    for k, v in spec.get("env", {}).items():
        env[k] = str(v)
    proc = subprocess.Popen(
        [cmd_path] + spec.get("args", []),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "maker3d-env-check", "version": "2.0"},
            },
        }
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(init_req) + "\n")
        proc.stdin.flush()

        init_resp = None
        t0 = time.time()
        while time.time() - t0 < 20.0:
            line = proc.stdout.readline()
            if not line:
                break
            s = line.strip()
            if s.startswith("{"):
                msg = json.loads(s)
                if msg.get("id") == 1:
                    init_resp = msg
                    break
        if not init_resp or "error" in init_resp:
            return False, f"initialize failed: {init_resp}"

        notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        proc.stdin.write(json.dumps(notif) + "\n" + json.dumps(list_req) + "\n")
        proc.stdin.flush()

        list_resp = None
        t0 = time.time()
        while time.time() - t0 < 15.0:
            line = proc.stdout.readline()
            if not line:
                break
            s = line.strip()
            if s.startswith("{"):
                msg = json.loads(s)
                if msg.get("id") == 2:
                    list_resp = msg
                    break
        tools = list_resp.get("result", {}).get("tools", []) if list_resp else []
        srv = init_resp.get("result", {}).get("serverInfo", {})
        if not tools:
            return False, f"no tools returned (srv={srv})"
        return True, f"{srv.get('name')} v={srv.get('version')} ({len(tools)} tools)"
    except Exception as exc:
        return False, str(exc)
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


def run_checks(skip_mcp: bool = False) -> Dict[str, Any]:
    paths = tp.discover_all()
    repo = Path(paths["repo_root"])
    results: Dict[str, Any] = {"binaries": {}, "python": {}, "skills": {}, "mcp": {}, "env_optional": {}}
    all_ok = True

    # 1. Binarios y librerias externas
    b_checks: List[Tuple[str, bool, str]] = []
    if paths["blender_exe"]:
        ok, msg = _run_version([paths["blender_exe"], "-b", "--version"])
        b_checks.append(("blender", ok, f"{msg} ({paths['blender_exe']})"))
    else:
        b_checks.append(("blender", False, "not found"))

    if paths["openscad_cli"]:
        ok, msg = _run_version([paths["openscad_cli"], "--version"])
        b_checks.append(("openscad", ok, f"{msg} ({paths['openscad_cli']})"))
    else:
        b_checks.append(("openscad", False, "not found"))

    bosl2_ok = bool(paths["bosl2_dir"]) and (Path(paths["bosl2_dir"]) / "std.scad").exists()
    b_checks.append(("bosl2", bosl2_ok, paths["bosl2_dir"] or "missing"))

    bambu_ok = bool(paths["bambu_studio_exe"]) and bool(paths["profiles"].get("a1", {}).get("machine"))
    b_checks.append(("bambu_studio", bambu_ok, paths["bambu_studio_exe"] or "missing"))

    if paths["uv"]:
        ok, msg = _run_version([paths["uv"], "--version"])
        b_checks.append(("uv", ok, msg))
    else:
        b_checks.append(("uv", False, "not found"))

    if paths["gltf_transform_cmd"]:
        ok, msg = _run_version([paths["gltf_transform_cmd"], "--version"])
        b_checks.append(("gltf_transform", ok, msg))
    else:
        b_checks.append(("gltf_transform", False, "not found"))

    for key, ok, detail in b_checks:
        results["binaries"][key] = {"ok": ok, "detail": detail}
        if not ok:
            all_ok = False

    # 2. Paquetes Python
    for mod_name, dist_name in REQUIRED_PY_PACKAGES:
        try:
            importlib.import_module(mod_name)
            ver = importlib.metadata.version(dist_name)
            results["python"][dist_name] = {"ok": True, "version": ver}
        except Exception as exc:
            results["python"][dist_name] = {"ok": False, "error": str(exc)}
            all_ok = False

    # 3. Skills y Reglas
    rule_path = repo / ".agents" / "rules" / "maker3d-architecture.md"
    results["skills"]["_rule_maker3d_architecture"] = {
        "ok": rule_path.exists() and rule_path.stat().st_size > 200,
        "path": str(rule_path),
    }
    if not results["skills"]["_rule_maker3d_architecture"]["ok"]:
        all_ok = False

    for sk in REQUIRED_SKILLS:
        sk_md = repo / ".agents" / "skills" / sk / "SKILL.md"
        ok, msg = _check_frontmatter(sk_md)
        results["skills"][sk] = {"ok": ok, "detail": msg}
        if not ok:
            all_ok = False

    # 4. MCP Config, Addon y Handshake JSON-RPC
    addon_ok = bool(paths["blender_mcp_addon"])
    results["mcp"]["blender_addon"] = {"ok": addon_ok, "path": paths["blender_mcp_addon"]}
    if not addon_ok:
        all_ok = False

    cfg_path = paths["mcp_config"]
    if not cfg_path or not Path(cfg_path).exists():
        results["mcp"]["mcp_config"] = {"ok": False, "detail": "missing mcp_config.json"}
        all_ok = False
    else:
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        servers = cfg.get("mcpServers", {})
        results["mcp"]["mcp_config"] = {"ok": len(servers) >= 3, "servers": list(servers.keys())}
        if len(servers) < 3:
            all_ok = False
        if not skip_mcp:
            for srv_name, spec in servers.items():
                ok, detail = probe_mcp_server(srv_name, spec)
                results["mcp"][f"rpc_{srv_name}"] = {"ok": ok, "detail": detail}
                if not ok:
                    all_ok = False

    # 5. Claves opcionales (.env)
    from dotenv import dotenv_values
    env_vals = dotenv_values(repo / ".env") if (repo / ".env").exists() else {}
    for var in ["HF_TOKEN", "FAL_KEY", "REPLICATE_API_TOKEN", "MESHY_API_KEY", "TRIPO_API_KEY"]:
        val = os.environ.get(var) or env_vals.get(var) or ""
        results["env_optional"][var] = "configured" if str(val).strip() else "unset (optional)"

    results["all_critical_ok"] = all_ok
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Maker3D 2.0 Environment & MCP Doctor")
    parser.add_argument("--json", action="store_true", help="Emitir informe JSON")
    parser.add_argument("--skip-mcp", action="store_true", help="Omitir handshake JSON-RPC de servidores MCP")
    args = parser.parse_args()

    res = run_checks(skip_mcp=args.skip_mcp)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        print("=== MAKER3D 2.0 ENV CHECK ===")
        for cat in ("binaries", "python", "skills", "mcp"):
            print(f"\n[{cat.upper()}]")
            for k, v in res[cat].items():
                mark = "PASS" if v.get("ok") else "FAIL"
                info = v.get("version") or v.get("detail") or v.get("path") or v.get("servers") or v.get("error") or ""
                print(f"  {mark:4s} {k:28s} {info}")
        print("\n[OPTIONAL CLOUD KEYS]")
        for k, v in res["env_optional"].items():
            print(f"  INFO {k:28s} {v}")
        print(f"\nOVERALL: {'PASS (100% READY)' if res['all_critical_ok'] else 'FAIL'}")

    sys.exit(0 if res["all_critical_ok"] else 1)


if __name__ == "__main__":
    main()
