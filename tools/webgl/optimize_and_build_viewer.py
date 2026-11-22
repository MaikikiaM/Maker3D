"""
Compresión glTF real con Meshopt + WebP vía gltf-transform.cmd y empaquetado de
visor HTML WebGL Three.js autónomo single-file (< 20 MB objetivo) para Maker3D 2.0.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import tool_paths as tp

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #0b111b;
      --panel: rgba(15, 23, 42, 0.84);
      --border: rgba(148, 163, 184, 0.22);
      --accent: #38bdf8;
      --text: #f1f5f9;
      --muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { width: 100%; height: 100%; overflow: hidden; background: var(--bg); color: var(--text); font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
    #canvas-wrap { position: fixed; inset: 0; }
    .top-pill {
      position: fixed; top: 14px; left: 14px; z-index: 20;
      background: var(--panel); backdrop-filter: blur(12px);
      border: 1px solid var(--border); border-radius: 12px;
      padding: 10px 14px; max-width: calc(100vw - 90px);
    }
    .top-pill h1 { font-size: 14px; font-weight: 700; color: #f8fafc; }
    .top-pill p { font-size: 11px; color: var(--muted); margin-top: 2px; }
    #drawer-toggle {
      position: fixed; top: 14px; right: 14px; z-index: 30;
      background: var(--panel); backdrop-filter: blur(12px);
      border: 1px solid var(--border); color: var(--text);
      border-radius: 10px; padding: 9px 12px; font-size: 12px; font-weight: 600; cursor: pointer;
    }
    #drawer {
      position: fixed; top: 58px; right: 14px; z-index: 25;
      width: 288px; max-height: calc(100vh - 78px); overflow-y: auto;
      background: var(--panel); backdrop-filter: blur(14px);
      border: 1px solid var(--border); border-radius: 14px; padding: 14px;
      transition: transform 0.2s ease, opacity 0.2s ease;
    }
    #drawer.collapsed { transform: translateX(320px); opacity: 0; pointer-events: none; }
    .sec-title { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 700; margin: 12px 0 6px; }
    .sec-title:first-child { margin-top: 0; }
    .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
    .btn-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; }
    button.ctrl-btn {
      background: rgba(30, 41, 59, 0.78); border: 1px solid rgba(148, 163, 184, 0.24);
      color: #e2e8f0; border-radius: 8px; padding: 8px 6px; font-size: 11.5px; font-weight: 600; cursor: pointer;
    }
    button.ctrl-btn.active { background: rgba(56, 189, 248, 0.24); border-color: var(--accent); color: #fff; }
    .slider-box { margin-top: 4px; }
    .slider-head { display: flex; justify-content: space-between; font-size: 11.5px; color: #cbd5e1; margin-bottom: 4px; }
    input[type="range"] { width: 100%; accent-color: var(--accent); cursor: pointer; }
    .specs-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px;
      background: rgba(15, 23, 42, 0.7); border-radius: 9px; padding: 9px;
      border: 1px solid rgba(148, 163, 184, 0.14); font-size: 11px;
    }
    .spec-item span { display: block; color: var(--muted); font-size: 9.5px; }
    .spec-item strong { color: #f8fafc; font-size: 11.5px; }
    #loading-overlay {
      position: fixed; inset: 0; z-index: 50; background: #070b12;
      display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px;
      font-size: 13px; color: #bae6fd;
    }
    .spinner { width: 32px; height: 32px; border: 3px solid rgba(56,189,248,0.2); border-top-color: #38bdf8; border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    #err-banner {
      display: none; position: fixed; bottom: 16px; left: 16px; right: 16px; z-index: 60;
      background: rgba(127, 29, 29, 0.94); border: 1px solid #f87171; color: #fee2e2;
      padding: 10px 14px; border-radius: 10px; font-size: 12px;
    }
  </style>
  <script type="importmap">
  {
    "imports": {
      "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
    }
  }
  </script>
</head>
<body>
  <div id="loading-overlay">
    <div class="spinner"></div>
    <div id="load-msg">Descomprimiendo GLB (__COMPRESS_LABEL__)...</div>
  </div>
  <div id="canvas-wrap"></div>
  <div id="err-banner"></div>

  <div class="top-pill">
    <h1>__TITLE__</h1>
    <p id="sub-info">Compresión __COMPRESS_LABEL__ · __RAW_KB__ KB → __OPT_KB__ KB (__RATIO__% ahorro)</p>
  </div>

  <button id="drawer-toggle">Controles</button>
  <aside id="drawer">
    <div class="sec-title">Modo de Render</div>
    <div class="btn-grid-3">
      <button class="ctrl-btn active" data-mode="pbr">PBR</button>
      <button class="ctrl-btn" data-mode="clay">Arcilla CAD</button>
      <button class="ctrl-btn" data-mode="wire">Wireframe</button>
    </div>

    <div class="sec-title">Atmósfera</div>
    <div class="btn-grid-3">
      <button class="ctrl-btn active" data-atmo="studio">Estudio</button>
      <button class="ctrl-btn" data-atmo="alpine">Sol Alpino</button>
      <button class="ctrl-btn" data-atmo="forge">Forja</button>
    </div>

    <div class="sec-title">Corte Z (Sección Sólida)</div>
    <div class="slider-box">
      <div class="slider-head"><span>Altura corte</span><strong id="clip-val">MAX</strong></div>
      <input type="range" id="clip-slider" min="0" max="100" step="0.5" value="100" />
    </div>

    <div class="sec-title">Cámaras Rápidas</div>
    <div class="btn-grid">
      <button class="ctrl-btn" data-cam="iso">Isométrica</button>
      <button class="ctrl-btn" data-cam="top">Planta</button>
      <button class="ctrl-btn" data-cam="side">Perfil</button>
      <button class="ctrl-btn" data-cam="bottom">Inferior / Imán</button>
    </div>

    <div class="sec-title">Telemetría GLB</div>
    <div class="specs-grid">
      <div class="spec-item"><span>Triángulos</span><strong id="stat-tris">—</strong></div>
      <div class="spec-item"><span>Dimensiones (mm)</span><strong id="stat-dims">—</strong></div>
      <div class="spec-item"><span>GLB Optimizado</span><strong>__OPT_KB__ KB</strong></div>
      <div class="spec-item"><span>HTML Total</span><strong>__HTML_MB__ MB</strong></div>
    </div>
  </aside>

  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
    import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
    import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
    import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

    const GLB_B64 = "__GLB_BASE64__";
    const drawer = document.getElementById("drawer");
    if (window.innerWidth <= 860) drawer.classList.add("collapsed");
    document.getElementById("drawer-toggle").addEventListener("click", () => drawer.classList.toggle("collapsed"));

    window.__MAKER3D_VIEWER__ = { loaded: false, mode: "pbr", atmo: "studio", triangles: 0, errors: [] };
    function reportErr(msg) {
      window.__MAKER3D_VIEWER__.errors.push(String(msg));
      const eb = document.getElementById("err-banner");
      eb.style.display = "block";
      eb.textContent = "Error WebGL: " + msg;
    }
    window.addEventListener("error", (e) => reportErr(e.message || e.error));
    window.addEventListener("unhandledrejection", (e) => reportErr(e.reason));

    const wrap = document.getElementById("canvas-wrap");
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.localClippingEnabled = true;
    wrap.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05080f);
    const pmremGen = new THREE.PMREMGenerator(renderer);
    scene.environment = pmremGen.fromScene(new RoomEnvironment(), 0.04).texture;
    if ("environmentIntensity" in scene) scene.environmentIntensity = 0.14;
    pmremGen.dispose();
    const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.2, 5000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;

    const hemi = new THREE.HemisphereLight(0xdcebff, 0x161d29, 0.44);
    const keyLight = new THREE.DirectionalLight(0xfff7ea, 2.25);
    keyLight.position.set(140, 150, 120);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(2048, 2048);
    keyLight.shadow.bias = -0.00025;
    keyLight.shadow.normalBias = 0.08;
    const fillLight = new THREE.DirectionalLight(0x93c5fd, 0.20);
    fillLight.position.set(-140, 90, -110);
    const rimLight = new THREE.DirectionalLight(0x7dd3fc, 0.42);
    rimLight.position.set(-110, 85, -120);
    scene.add(hemi, keyLight, fillLight, rimLight);

    const clipPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 9999);
    const clayMat = new THREE.MeshStandardMaterial({
      color: 0xc8c5be,
      roughness: 0.56,
      metalness: 0.02,
      vertexColors: false,
      side: THREE.DoubleSide,
      clippingPlanes: [clipPlane],
    });
    const wireMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      roughness: 0.45,
      metalness: 0.1,
      vertexColors: false,
      wireframe: true,
      clippingPlanes: [clipPlane],
    });

    const originalMats = new Map();
    let rootGroup = null;
    let modelBox = new THREE.Box3();
    let modelCenter = new THREE.Vector3();
    let maxDim = 100;

    function base64ToArrayBuffer(b64) {
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return bytes.buffer;
    }

    function applyMode(mode) {
      if (!rootGroup) return;
      window.__MAKER3D_VIEWER__.mode = mode;
      rootGroup.traverse((obj) => {
        if (!obj.isMesh) return;
        if (mode === "clay") obj.material = clayMat;
        else if (mode === "wire") obj.material = wireMat;
        else obj.material = originalMats.get(obj) || clayMat;
      });
    }

    function applyAtmo(atmo) {
      window.__MAKER3D_VIEWER__.atmo = atmo;
      if (atmo === "alpine") {
        scene.background = new THREE.Color(0x07111c);
        if ("environmentIntensity" in scene) scene.environmentIntensity = 0.16;
        hemi.color.setHex(0xdbeafe); hemi.groundColor.setHex(0x1e293b); hemi.intensity = 0.52;
        keyLight.color.setHex(0xffffff); keyLight.intensity = 2.40;
        fillLight.color.setHex(0x7dd3fc); fillLight.intensity = 0.26;
        rimLight.color.setHex(0xbae6fd); rimLight.intensity = 0.48;
      } else if (atmo === "forge") {
        scene.background = new THREE.Color(0x0d0806);
        if ("environmentIntensity" in scene) scene.environmentIntensity = 0.12;
        hemi.color.setHex(0xfed7aa); hemi.groundColor.setHex(0x1c1917); hemi.intensity = 0.38;
        keyLight.color.setHex(0xfb923c); keyLight.intensity = 2.50;
        fillLight.color.setHex(0x38bdf8); fillLight.intensity = 0.22;
        rimLight.color.setHex(0xfdba74); rimLight.intensity = 0.45;
      } else {
        scene.background = new THREE.Color(0x05080f);
        if ("environmentIntensity" in scene) scene.environmentIntensity = 0.14;
        hemi.color.setHex(0xdcebff); hemi.groundColor.setHex(0x161d29); hemi.intensity = 0.44;
        keyLight.color.setHex(0xfff7ea); keyLight.intensity = 2.25;
        fillLight.color.setHex(0x93c5fd); fillLight.intensity = 0.20;
        rimLight.color.setHex(0x7dd3fc); rimLight.intensity = 0.42;
      }
    }

    function setCameraPreset(preset) {
      controls.target.copy(modelCenter);
      if (preset === "top") {
        camera.position.set(modelCenter.x, modelCenter.y + maxDim * 1.9, modelCenter.z + 0.01);
      } else if (preset === "side") {
        camera.position.set(modelCenter.x + maxDim * 1.75, modelCenter.y + 0.25 * maxDim, modelCenter.z);
      } else if (preset === "bottom") {
        camera.position.set(modelCenter.x + 0.35 * maxDim, modelCenter.y - maxDim * 1.65, modelCenter.z + 0.45 * maxDim);
      } else {
        camera.position.set(modelCenter.x + 1.25 * maxDim, modelCenter.y + 1.15 * maxDim, modelCenter.z + 1.35 * maxDim);
      }
      controls.update();
    }

    async function init() {
      try {
        await MeshoptDecoder.ready;
        const loader = new GLTFLoader();
        loader.setMeshoptDecoder(MeshoptDecoder);
        const draco = new DRACOLoader();
        draco.setDecoderPath("https://www.gstatic.com/draco/versioned/decoders/1.5.7/");
        loader.setDRACOLoader(draco);

        const buf = base64ToArrayBuffer(GLB_B64);
        loader.parse(buf, "", (gltf) => {
          rootGroup = gltf.scene;
          scene.add(rootGroup);

          let tris = 0;
          const maxAniso = Math.min(16, renderer.capabilities.getMaxAnisotropy() || 4);
          rootGroup.traverse((obj) => {
            if (obj.isMesh) {
              obj.castShadow = true;
              obj.receiveShadow = true;
              const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
              mats.forEach((m) => {
                if (m) {
                  m.side = THREE.DoubleSide;
                  m.clippingPlanes = [clipPlane];
                  [m.map, m.normalMap, m.roughnessMap, m.metalnessMap, m.aoMap].forEach((tex) => {
                    if (tex) {
                      tex.anisotropy = maxAniso;
                      tex.needsUpdate = true;
                    }
                  });
                  if (m.normalMap && m.normalScale) m.normalScale.set(1.30, 1.30);
                }
              });
              originalMats.set(obj, obj.material);
              const g = obj.geometry;
              if (g.index) tris += g.index.count / 3;
              else if (g.attributes.position) tris += g.attributes.position.count / 3;
            }
          });

          modelBox.setFromObject(rootGroup);
          const size = new THREE.Vector3();
          modelBox.getSize(size);
          modelBox.getCenter(modelCenter);
          maxDim = Math.max(size.x, size.y, size.z, 10);

          keyLight.target.position.copy(modelCenter);
          scene.add(keyLight.target);
          keyLight.position.set(modelCenter.x + maxDim * 1.15, modelCenter.y + maxDim * 1.18, modelCenter.z + maxDim * 0.95);
          fillLight.position.set(modelCenter.x - maxDim * 1.10, modelCenter.y + maxDim * 0.70, modelCenter.z - maxDim * 0.90);
          rimLight.position.set(modelCenter.x - maxDim * 0.95, modelCenter.y + maxDim * 0.75, modelCenter.z - maxDim * 1.10);
          const shHalf = maxDim * 0.65;
          keyLight.shadow.camera.left = -shHalf;
          keyLight.shadow.camera.right = shHalf;
          keyLight.shadow.camera.top = shHalf;
          keyLight.shadow.camera.bottom = -shHalf;
          keyLight.shadow.camera.near = maxDim * 0.25;
          keyLight.shadow.camera.far = maxDim * 4.0;
          keyLight.shadow.camera.updateProjectionMatrix();

          const slider = document.getElementById("clip-slider");
          const clipVal = document.getElementById("clip-val");
          slider.min = modelBox.min.y.toFixed(1);
          slider.max = (modelBox.max.y + 1.5).toFixed(1);
          slider.value = slider.max;
          clipPlane.constant = parseFloat(slider.value);
          clipVal.textContent = (size.y).toFixed(1) + " mm";
          slider.addEventListener("input", () => {
            const v = parseFloat(slider.value);
            clipPlane.constant = v;
            clipVal.textContent = Math.max(0, v - modelBox.min.y).toFixed(1) + " mm";
          });

          // Escala en Three.js Y-up -> glTF suele llegar en metros si Blender exportó m (0.001) o mm según escena
          const scaleFactor = maxDim < 2.0 ? 1000.0 : 1.0;
          document.getElementById("stat-tris").textContent = Math.round(tris).toLocaleString("es-ES");
          document.getElementById("stat-dims").textContent =
            `${(size.x * scaleFactor).toFixed(1)}×${(size.z * scaleFactor).toFixed(1)}×${(size.y * scaleFactor).toFixed(1)}`;
          camera.near = maxDim * 0.002;
          camera.far = maxDim * 40.0;
          camera.updateProjectionMatrix();
          setCameraPreset("iso");

          window.__MAKER3D_VIEWER__.loaded = true;
          window.__MAKER3D_VIEWER__.triangles = Math.round(tris);
          document.getElementById("loading-overlay").style.display = "none";
        }, (err) => {
          reportErr(err);
        });
      } catch (err) {
        reportErr(err);
      }
    }

    document.querySelectorAll("[data-mode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-mode]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        applyMode(btn.dataset.mode);
      });
    });
    document.querySelectorAll("[data-atmo]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-atmo]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        applyAtmo(btn.dataset.atmo);
      });
    });
    document.querySelectorAll("[data-cam]").forEach((btn) => {
      btn.addEventListener("click", () => setCameraPreset(btn.dataset.cam));
    });

    window.addEventListener("resize", () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    init();
    animate();
  </script>
</body>
</html>
"""


def optimize_glb(
    in_glb: Path,
    out_glb: Path,
    compress: str = "meshopt",
    texture_format: str = "webp",
    max_texture_size: int = 4096,
    simplify_ratio: float | None = None,
    no_simplify: bool = False,
) -> Dict[str, float | str]:
    webgl_tools = tp.find_webgl_tools()
    cli = webgl_tools.get("gltf_transform_cmd")
    if not cli or not Path(cli).exists():
        raise FileNotFoundError("gltf-transform.cmd no encontrado en tools/webgl/node_modules/.bin")

    raw_bytes = in_glb.stat().st_size
    cmd = [
        cli,
        "optimize",
        str(in_glb),
        str(out_glb),
        "--compress", compress,
        "--texture-compress", texture_format,
        "--texture-size", str(max_texture_size),
    ]
    if compress == "meshopt":
        cmd.extend(["--meshopt-level", "high"])
    if no_simplify:
        cmd.extend(["--simplify", "false"])
    elif simplify_ratio is not None and 0.05 <= simplify_ratio < 1.0:
        cmd.extend(["--simplify", "true", "--simplify-error", f"{max(0.0001, (1.0 - simplify_ratio) * 0.01):.5f}"])

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0 or not out_glb.exists():
        raise RuntimeError(f"gltf-transform optimize falló ({res.returncode}):\n{res.stdout}\n{res.stderr}")

    opt_bytes = out_glb.stat().st_size
    savings_pct = (1.0 - (opt_bytes / max(raw_bytes, 1))) * 100.0
    return {
        "raw_bytes": raw_bytes,
        "opt_bytes": opt_bytes,
        "raw_kb": round(raw_bytes / 1024.0, 2),
        "opt_kb": round(opt_bytes / 1024.0, 2),
        "savings_pct": round(savings_pct, 2),
        "compress": f"{compress}+{texture_format}",
    }


def build_viewer_html(
    in_glb: Path,
    out_html: Path,
    title: str = "Maker3D 2.0 Viewer",
    compress: str = "meshopt",
    texture_format: str = "webp",
    max_texture_size: int = 4096,
    simplify_ratio: float | None = None,
    no_simplify: bool = False,
    save_opt_glb: Path | None = None,
) -> Dict[str, float | str]:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_d:
        opt_glb = save_opt_glb or (Path(tmp_d) / f"{in_glb.stem}_opt.glb")
        stats = optimize_glb(
            in_glb=in_glb,
            out_glb=opt_glb,
            compress=compress,
            texture_format=texture_format,
            max_texture_size=max_texture_size,
            simplify_ratio=simplify_ratio,
            no_simplify=no_simplify,
        )
        glb_b64 = base64.b64encode(opt_glb.read_bytes()).decode("ascii")

    # Primera pasada estimada para conocer tamaño HTML final y rellenar marcador exacto
    safe_title = html.escape(title)
    rendered = (
        HTML_TEMPLATE
        .replace("__TITLE__", safe_title)
        .replace("__COMPRESS_LABEL__", html.escape(str(stats["compress"]).upper()))
        .replace("__RAW_KB__", f"{stats['raw_kb']:.1f}")
        .replace("__OPT_KB__", f"{stats['opt_kb']:.1f}")
        .replace("__RATIO__", f"{stats['savings_pct']:.1f}")
        .replace("__GLB_BASE64__", glb_b64)
    )
    html_mb = round(len(rendered.encode("utf-8")) / (1024.0 * 1024.0), 3)
    rendered = rendered.replace("__HTML_MB__", f"{html_mb:.2f}")
    out_html.write_text(rendered, encoding="utf-8")

    stats["html_path"] = str(out_html)
    stats["html_mb"] = round(out_html.stat().st_size / (1024.0 * 1024.0), 3)
    if stats["html_mb"] > 95.0:
        print(f"[WARN] Visor HTML pesa {stats['html_mb']} MB (>95 MB): excluir de GitHub (.gitignore) y guardar en local/Drive.", file=sys.stderr)
    elif stats["html_mb"] > 50.0:
        print(f"[INFO] Visor HTML pesa {stats['html_mb']} MB (>50 MB): recordar excluirlo en .gitignore antes de git-sync.", file=sys.stderr)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimizador Meshopt+WebP y generador de Visor HTML Maker3D 2.0")
    parser.add_argument("--input", required=True, help="Archivo .glb de entrada")
    parser.add_argument("--out-html", required=True, help="Ruta HTML de salida")
    parser.add_argument("--title", default="Maker3D 2.0 — Visor WebGL PBR")
    parser.add_argument("--compress", choices=["meshopt", "draco"], default="meshopt")
    parser.add_argument("--texture-format", choices=["webp", "ktx2", "avif"], default="webp")
    parser.add_argument("--max-texture-size", type=int, default=4096)
    parser.add_argument("--simplify-ratio", type=float, default=None)
    parser.add_argument("--no-simplify", action="store_true", help="Desactivar simplificación de malla en optimize")
    parser.add_argument("--save-opt-glb", default=None, help="Opcional: guardar GLB optimizado en disco")
    args = parser.parse_args()

    stats = build_viewer_html(
        in_glb=Path(args.input),
        out_html=Path(args.out_html),
        title=args.title,
        compress=args.compress,
        texture_format=args.texture_format,
        max_texture_size=args.max_texture_size,
        simplify_ratio=args.simplify_ratio,
        no_simplify=args.no_simplify,
        save_opt_glb=Path(args.save_opt_glb) if args.save_opt_glb else None,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
