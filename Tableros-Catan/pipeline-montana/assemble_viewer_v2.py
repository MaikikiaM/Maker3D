"""
Responsive Android- & Desktop-Friendly Three.js Full-Color CAD Viewer Builder (V2)
Builds a self-contained single-file HTML viewer from:
- scratch_v2/viewer_mesh_v2.npz (~210k tris smooth interactive mesh)
- Loseta_Montana_3D/catan_mountain_diffuse_2k.png
- scratch_v2/catan_mountain_normal_2k.png
- scratch_v2/master_metrics_v2.json

Deploys identically to all 5 repository viewer paths:
1. Tableros-Catan/Loseta_Montana_3D/catan_mountain_viewer.html
2. Tableros-Catan/catan_mountain_viewer.html
3. Tableros-Catan/Loseta-Montana-3D/catan_mountain_viewer.html
4. Visores-HTML/catan-loseta-montana-detailed.html
5. Visores-HTML/catan-loseta-montana.html
"""
import os
import io
import json
import gzip
import math
import base64
import time
import numpy as np
from PIL import Image

OUT_DIR = "Tableros-Catan/Loseta_Montana_3D"
SCRATCH_DIR = "Tableros-Catan/pipeline-montana/scratch_v2"
TRIPLANAR_MODE = "none"  # "none" | "forest" | "clay"
USE_SMOOTH_TANGENTS = False
SHADOW_BIAS = -0.0006
SHADOW_NORMAL_BIAS = 0.35

TARGET_PATHS = [
    "Tableros-Catan/Loseta_Montana_3D/catan_mountain_viewer.html",
    "Tableros-Catan/catan_mountain_viewer.html",
    "Tableros-Catan/Loseta-Montana-3D/catan_mountain_viewer.html",
    "Visores-HTML/catan-loseta-montana-detailed.html",
    "Visores-HTML/catan-loseta-montana.html",
]


def encode_img_b64(png_path, fmt="JPEG", quality=91):
    im = Image.open(png_path).convert("RGB")
    buf = io.BytesIO()
    if fmt.upper() == "WEBP":
        im.save(buf, format="WEBP", quality=quality, method=5)
        mime = "image/webp"
    else:
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii"), len(buf.getvalue())


def build_viewer():
    t0 = time.time()
    print("[Viewer V2] Loading compressed mesh, textures, and master telemetry...")
    mesh_data = np.load(os.path.join(SCRATCH_DIR, "viewer_mesh_v2.npz"))
    verts = mesh_data["verts"].astype(np.float32)
    faces = mesh_data["faces"].astype(np.uint32)

    with open(os.path.join(SCRATCH_DIR, "master_metrics_v2.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Precompute UVs cleanly: u=(x+R)/(2R), v=(y+R)/(2R) matching flipped PNG top=+R
    R = 64.0
    APOTHEM = R * math.cos(math.radians(30.0))
    Z_BASE = 25.0
    uvs = np.column_stack([
        np.clip((verts[:, 0] + R) / (2.0 * R), 0.0, 1.0),
        np.clip((verts[:, 1] + R) / (2.0 * R), 0.0, 1.0),
    ]).astype(np.float32)

    # Split vertical plinth & bottom faces onto sharp architectural vertices with clean slate UV (zero smear!)
    # Since split plinth triangles don't share vertices across corners or with terrain, Three.js computeVertexNormals()
    # automatically produces exact planar flat normals on walls/bottom & smooth normals on terrain with 0KB extra payload!
    is_plinth_face = np.any(verts[faces, 2] < (Z_BASE - 0.05), axis=1)
    pf = faces[is_plinth_face]
    n_pf = len(pf)
    verts_p = verts[pf.ravel()]
    uvs_p = np.full((n_pf * 3, 2), 0.015, dtype=np.float32)

    has_vcols = "vcols" in mesh_data
    if has_vcols:
        vcols = mesh_data["vcols"].astype(np.float32)
        vcols_p = np.zeros((n_pf * 3, 4), dtype=np.float32)
        vcols = np.vstack([vcols, vcols_p]).astype(np.float32)
        vcols_gz = base64.b64encode(gzip.compress(vcols.tobytes(), compresslevel=9)).decode("ascii")
    else:
        vcols_gz = ""

    base_idx = len(verts)
    verts = np.vstack([verts, verts_p]).astype(np.float32)
    uvs = np.vstack([uvs, uvs_p]).astype(np.float32)
    faces = faces.copy()
    faces[is_plinth_face] = (np.arange(n_pf * 3, dtype=np.uint32).reshape(n_pf, 3) + base_idx)

    # Gzip-compress binary buffers for compact mobile-friendly single-file payload
    verts_gz = base64.b64encode(gzip.compress(verts.tobytes(), compresslevel=9)).decode("ascii")
    uvs_gz = base64.b64encode(gzip.compress(uvs.tobytes(), compresslevel=9)).decode("ascii")
    faces_gz = base64.b64encode(gzip.compress(faces.tobytes(), compresslevel=9)).decode("ascii")

    diff_uri, diff_sz = encode_img_b64(os.path.join(OUT_DIR, "catan_mountain_diffuse_2k.png"), fmt="WEBP", quality=96)
    norm_uri, norm_sz = encode_img_b64(os.path.join(SCRATCH_DIR, "catan_mountain_normal_2k.png"), fmt="WEBP", quality=94)
    rough_uri, rough_sz = encode_img_b64(os.path.join(SCRATCH_DIR, "catan_mountain_roughness_2k.png"), fmt="JPEG", quality=88)

    m = metrics["master"]
    cov = metrics["coverage"]
    vm = metrics["viewer"]

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <title>Cordillera Montaña Catan 3D (V3) — {m['F']:,} Triángulos Master</title>
  <style>
    :root {{
      --bg: #080c12;
      --panel: rgba(13, 19, 30, 0.84);
      --border: rgba(56, 189, 248, 0.24);
      --accent: #38bdf8;
      --accent2: #22c55e;
      --text: #f1f5f9;
      --muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
    html, body {{
      margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden;
      background: radial-gradient(circle at 46% 38%, #1e293b 0%, #0f172a 55%, #070b11 100%);
      color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    #canvas-wrap {{
      position: fixed; inset: 0; width: 100%; height: 100%; touch-action: none;
    }}
    .top-pill {{
      position: fixed; top: 12px; left: 12px; z-index: 20;
      background: var(--panel); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--border); border-radius: 12px; padding: 8px 13px;
      pointer-events: none; max-width: calc(100vw - 140px);
    }}
    .top-pill h1 {{
      margin: 0; font-size: 14px; font-weight: 700; letter-spacing: 0.2px; color: #f8fafc;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .top-pill p {{
      margin: 2px 0 0; font-size: 11px; color: var(--accent); font-weight: 600;
    }}
    #drawer-toggle {{
      position: fixed; top: 12px; right: 12px; z-index: 30;
      background: var(--panel); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
      color: #f8fafc; border: 1px solid var(--border); border-radius: 10px;
      padding: 9px 13px; font-size: 13px; font-weight: 600; cursor: pointer;
      display: flex; align-items: center; gap: 6px; box-shadow: 0 6px 20px rgba(0,0,0,0.45);
    }}
    #drawer-toggle:active {{ transform: scale(0.97); }}
    #drawer {{
      position: fixed; top: 56px; right: 12px; z-index: 25;
      width: 268px; max-width: calc(100vw - 24px); max-height: calc(100vh - 88px);
      overflow-y: auto; background: var(--panel); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
      border: 1px solid var(--border); border-radius: 14px; padding: 12px;
      box-shadow: 0 12px 32px rgba(0,0,0,0.6); transition: transform 0.22s ease, opacity 0.22s ease;
    }}
    #drawer.collapsed {{
      transform: translateX(calc(100% + 24px)); opacity: 0; pointer-events: none;
    }}
    .sec-title {{
      font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.9px;
      color: var(--muted); font-weight: 700; margin: 11px 0 6px;
    }}
    .sec-title:first-child {{ margin-top: 0; }}
    .btn-grid {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
    }}
    button.ctrl-btn {{
      background: rgba(30, 41, 59, 0.75); border: 1px solid rgba(148, 163, 184, 0.22);
      color: #e2e8f0; border-radius: 8px; padding: 8px 6px; font-size: 11.5px; font-weight: 600;
      cursor: pointer; transition: 0.15s;
    }}
    button.ctrl-btn.active {{
      background: rgba(56, 189, 248, 0.22); border-color: var(--accent); color: #ffffff;
    }}
    .row-toggle {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 6px 0; font-size: 12.5px; color: #e2e8f0; cursor: pointer;
    }}
    .row-toggle input {{ accent-color: var(--accent); width: 16px; height: 16px; cursor: pointer; }}
    .slider-box {{ margin: 6px 0 4px; }}
    .slider-head {{ display: flex; justify-content: space-between; font-size: 12px; color: #cbd5e1; margin-bottom: 4px; }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent); cursor: pointer; }}
    .specs-grid {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px;
      background: rgba(15, 23, 42, 0.72); border-radius: 9px; padding: 9px;
      border: 1px solid rgba(148, 163, 184, 0.14); font-size: 11px;
    }}
    .spec-item span {{ display: block; color: var(--muted); font-size: 9.8px; }}
    .spec-item strong {{ color: #f8fafc; font-size: 11.5px; }}
    #loading-overlay {{
      position: fixed; inset: 0; z-index: 50; background: #080c12;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      font-size: 14px; color: #bae6fd; gap: 10px;
    }}
    .spinner {{
      width: 34px; height: 34px; border: 3px solid rgba(56,189,248,0.2);
      border-top-color: #38bdf8; border-radius: 50%; animation: spin 0.8s linear infinite;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    .bottom-hint {{
      position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); z-index: 15;
      font-size: 11px; color: rgba(203, 213, 225, 0.75); background: rgba(15, 23, 42, 0.55);
      padding: 4px 10px; border-radius: 999px; pointer-events: none; white-space: nowrap;
    }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="loading-overlay">
    <div class="spinner"></div>
    <div id="load-msg">Descomprimiendo geometría alpina 3D y texturas 2K...</div>
  </div>

  <div id="canvas-wrap"></div>

  <div class="top-pill">
    <h1>🏔️ Cordillera Montaña Catan 3D · V3 PBR</h1>
    <p>{m['F']:,} tris (χ={m['chi']}) · R=64mm · Z=83mm · HDR + Sombras</p>
  </div>

  <button id="drawer-toggle" aria-label="Mostrar u ocultar controles">⚙️ Controles</button>

  <aside id="drawer">
    <div class="sec-title">Modo de Renderizado</div>
    <div class="btn-grid">
      <button class="ctrl-btn active" data-mode="pbr">Hiperreal PBR</button>
      <button class="ctrl-btn" data-mode="clay">Arcilla CAD</button>
      <button class="ctrl-btn" data-mode="slope">Mapa Pendientes ≥44°</button>
      <button class="ctrl-btn" data-mode="wire">Malla + Relieve</button>
    </div>

    <div class="sec-title">Atmósfera de Iluminación</div>
    <div class="btn-grid" style="grid-template-columns: 1fr 1fr 1fr;">
      <button class="ctrl-btn active" data-atmo="alpine">Sol Alpino</button>
      <button class="ctrl-btn" data-atmo="sunset">Atardecer</button>
      <button class="ctrl-btn" data-atmo="overcast">Nublado</button>
    </div>

    <div class="sec-title">Cotas CAD &amp; Tolerancias</div>
    <label class="row-toggle">
      <span>Cotas Hexágono (110.85×128×83)</span>
      <input type="checkbox" id="chk-dims" />
    </label>
    <label class="row-toggle">
      <span>6 Triángulos Equiláteros (64mm)</span>
      <input type="checkbox" id="chk-tris" />
    </label>
    <label class="row-toggle">
      <span>Carril 3.5mm, Bases Ø20 &amp; Imán Ø10.2</span>
      <input type="checkbox" id="chk-tol" />
    </label>
    <label class="row-toggle">
      <span>Rotación Turntable Suave</span>
      <input type="checkbox" id="chk-spin" />
    </label>

    <div class="sec-title">Sección Transversal Sólida (Z)</div>
    <div class="slider-box">
      <div class="slider-head">
        <span>Plano de corte Z (imán abajo 0..5.2mm)</span>
        <strong id="clip-val">83.0 mm</strong>
      </div>
      <input type="range" id="clip-slider" min="1.0" max="83.5" step="0.5" value="83.5" />
    </div>

    <div class="sec-title">Iluminación Solar Azimutal</div>
    <div class="slider-box">
      <div class="slider-head">
        <span>Ángulo del Sol</span>
        <strong id="sun-val">-52°</strong>
      </div>
      <input type="range" id="sun-slider" min="-180" max="180" step="2" value="-52" />
    </div>

    <div class="sec-title">Cámaras Rápidas</div>
    <div class="btn-grid">
      <button class="ctrl-btn" data-cam="iso">Isométrica SE</button>
      <button class="ctrl-btn" data-cam="top">Cenital (UV)</button>
      <button class="ctrl-btn" data-cam="north">Cara Norte ≥44°</button>
      <button class="ctrl-btn" data-cam="eye">Mesa / Imán</button>
    </div>

    <div class="sec-title">Auditoría Técnica Master V3</div>
    <div class="specs-grid">
      <div class="spec-item"><span>Malla Master</span><strong>{m['F']:,} Δ</strong></div>
      <div class="spec-item"><span>Visor WebGL</span><strong>{vm['F']:,} Δ</strong></div>
      <div class="spec-item"><span>Manifold Euler</span><strong>χ = {m['chi']} (Sólido)</strong></div>
      <div class="spec-item"><span>Volumen Total</span><strong>{m['volume_cm3']:.2f} cm³</strong></div>
      <div class="spec-item"><span>Materiales PBR</span><strong>Diff + Norm + Rough</strong></div>
      <div class="spec-item"><span>Nieve en ≥44°</span><strong>{cov['max_snow_steep']:.6f} (0%)</strong></div>
    </div>
  </aside>

  <div class="bottom-hint">1 dedo: Rotar · 2 dedos / Rueda: Zoom · Shift/2 dedos: Mover</div>

  <script>
  (function() {{
    const VERTS_GZ = "{verts_gz}";
    const UVS_GZ   = "{uvs_gz}";
    const FACES_GZ = "{faces_gz}";
    const VCOLS_GZ = "{vcols_gz}";
    const TRIPLANAR_MODE = "{TRIPLANAR_MODE}";
    const USE_SMOOTH_TANGENTS = {"true" if USE_SMOOTH_TANGENTS else "false"};
    const DIFF_URI = "{diff_uri}";
    const NORM_URI = "{norm_uri}";
    const ROUGH_URI = "{rough_uri}";

    const R = 64.0;
    const APOTHEM = {APOTHEM:.8f};
    const Z_BASE = 25.0;
    const Z_PEAK = 83.0;
    const ROAD_MARGIN = 3.5;
    const PAD_R = 10.0;
    const MAG_R = 5.1;

    // Collapse drawer automatically on mobile / narrow screens (< 900px) so the 3D mountain shines unobstructed
    const drawer = document.getElementById("drawer");
    if (window.innerWidth <= 900) drawer.classList.add("collapsed");
    document.getElementById("drawer-toggle").addEventListener("click", () => {{
      drawer.classList.toggle("collapsed");
    }});

    async function gunzipBase64(b64) {{
      const binStr = atob(b64);
      const bytes = new Uint8Array(binStr.length);
      for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
      const ds = new DecompressionStream("gzip");
      const stream = new Blob([bytes]).stream().pipeThrough(ds);
      return await new Response(stream).arrayBuffer();
    }}

    function loadTex(uri, isSRGB) {{
      return new Promise((res) => {{
        const img = new Image();
        img.onload = () => {{
          const t = new THREE.Texture(img);
          t.flipY = true;
          t.anisotropy = 16;
          t.encoding = isSRGB ? THREE.sRGBEncoding : THREE.LinearEncoding;
          t.needsUpdate = true;
          res(t);
        }};
        img.src = uri;
      }});
    }}

    function makeTextSprite(text, colorStr, scaleMm) {{
      const c = document.createElement("canvas");
      const ctx = c.getContext("2d");
      ctx.font = "bold 36px sans-serif";
      const tw = Math.ceil(ctx.measureText(text).width) + 32;
      c.width = tw; c.height = 58;
      ctx.fillStyle = "rgba(8, 12, 18, 0.86)";
      ctx.strokeStyle = colorStr;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(2, 2, tw - 4, 54, 12) : ctx.rect(2, 2, tw - 4, 54);
      ctx.fill(); ctx.stroke();
      ctx.font = "bold 34px sans-serif";
      ctx.fillStyle = colorStr;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(text, tw / 2, 29);
      const tex = new THREE.CanvasTexture(c);
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({{ map: tex, depthTest: false }}));
      sp.scale.set(scaleMm * (tw / 58), scaleMm, 1);
      sp.renderOrder = 20;
      return sp;
    }}

    async function init() {{
      try {{
        const [vBuf, uvBuf, fBuf, vcBuf, diffTex, normTex, roughTex] = await Promise.all([
          gunzipBase64(VERTS_GZ),
          gunzipBase64(UVS_GZ),
          gunzipBase64(FACES_GZ),
          VCOLS_GZ ? gunzipBase64(VCOLS_GZ) : Promise.resolve(null),
          loadTex(DIFF_URI, true),
          loadTex(NORM_URI, false),
          loadTex(ROUGH_URI, false),
        ]);

        const posArr = new Float32Array(vBuf);
        const uvArr = new Float32Array(uvBuf);
        const idxArr = new Uint32Array(fBuf);

        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
        geom.setAttribute("uv", new THREE.BufferAttribute(uvArr, 2));
        if (vcBuf) {{
          geom.setAttribute("treeVCol", new THREE.BufferAttribute(new Float32Array(vcBuf), 4));
        }}
        geom.setIndex(new THREE.BufferAttribute(idxArr, 1));
        geom.computeVertexNormals();

        // Precompute vertex slope color attribute for Slope Audit mode & smooth tangents
        const nrmAttr = geom.getAttribute("normal");
        if (USE_SMOOTH_TANGENTS) {{
          const tanArr = new Float32Array(nrmAttr.count * 4);
          for (let i = 0; i < nrmAttr.count; i++) {{
            const nx = nrmAttr.getX(i), ny = nrmAttr.getY(i), nz = nrmAttr.getZ(i);
            const useY = Math.abs(nx) > 0.94;
            const ax = useY ? 0.0 : 1.0, ay = useY ? 1.0 : 0.0;
            const d = ax * nx + ay * ny;
            let tx = ax - d * nx, ty = ay - d * ny, tz = -d * nz;
            const tl = Math.hypot(tx, ty, tz) || 1.0;
            tanArr[i * 4] = tx / tl;
            tanArr[i * 4 + 1] = ty / tl;
            tanArr[i * 4 + 2] = tz / tl;
            tanArr[i * 4 + 3] = 1.0;
          }}
          geom.setAttribute("tangent", new THREE.BufferAttribute(tanArr, 4));
        }}
        const slopeCols = new Float32Array(posArr.length);
        for (let i = 0; i < nrmAttr.count; i++) {{
          const nz = Math.abs(nrmAttr.getZ(i));
          const deg = Math.acos(Math.min(1, nz)) * 180.0 / Math.PI;
          const z = posArr[i * 3 + 2];
          if (z < Z_BASE - 0.05) {{
            slopeCols[i*3] = 0.20; slopeCols[i*3+1] = 0.24; slopeCols[i*3+2] = 0.30;
          }} else if (deg >= 44.0) {{
            // Steep cliff >=44 deg (zero snow guarantee)
            slopeCols[i*3] = 0.96; slopeCols[i*3+1] = 0.26; slopeCols[i*3+2] = 0.30;
          }} else if (deg < 2.0 && z <= Z_BASE + 0.08) {{
            // Flat 3.5mm road lane + Ø20mm pads
            slopeCols[i*3] = 0.22; slopeCols[i*3+1] = 0.84; slopeCols[i*3+2] = 0.95;
          }} else {{
            const t = deg / 44.0;
            slopeCols[i*3] = 0.18 + 0.72 * t; slopeCols[i*3+1] = 0.68 - 0.15 * t; slopeCols[i*3+2] = 0.34 - 0.20 * t;
          }}
        }}
        geom.setAttribute("color", new THREE.BufferAttribute(slopeCols, 3));

        const container = document.getElementById("canvas-wrap");
        const renderer = new THREE.WebGLRenderer({{ antialias: true, stencil: true, powerPreference: "high-performance" }});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.outputEncoding = THREE.sRGBEncoding;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 0.96;
        renderer.localClippingEnabled = true;
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(renderer.domElement);

        const scene = new THREE.Scene();

        // Procedural Studio Alpine HDRI Environment Map (PMREM IBL — zero external URL needed)
        (function buildStudioIBL() {{
          const ew = 128, eh = 64;
          const data = new Uint8Array(ew * eh * 4);
          for (let y = 0; y < eh; y++) {{
            const v = y / (eh - 1);
            const theta = (1.0 - v) * Math.PI;
            const sy = Math.cos(theta);
            for (let x = 0; x < ew; x++) {{
              const u = x / (ew - 1);
              const phi = u * 2.0 * Math.PI;
              const sx = Math.sin(theta) * Math.cos(phi);
              const sz = Math.sin(theta) * Math.sin(phi);
              const skyMix = Math.max(0, sy);
              const groundMix = Math.max(0, -sy);
              let r = 135 * skyMix + 28 * groundMix + 175 * (1.0 - Math.abs(sy));
              let g = 175 * skyMix + 34 * groundMix + 200 * (1.0 - Math.abs(sy));
              let b = 235 * skyMix + 44 * groundMix + 225 * (1.0 - Math.abs(sy));
              const sunDot = Math.max(0, sx * 0.52 - sz * 0.44 + sy * 0.72);
              const sunGlow = Math.pow(sunDot, 22) * 195 + Math.pow(sunDot, 3) * 32;
              r += sunGlow * 1.04; g += sunGlow * 0.96; b += sunGlow * 0.84;
              const idx = (y * ew + x) * 4;
              data[idx] = Math.min(255, r); data[idx + 1] = Math.min(255, g);
              data[idx + 2] = Math.min(255, b); data[idx + 3] = 255;
            }}
          }}
          const equirect = new THREE.DataTexture(data, ew, eh, THREE.RGBAFormat);
          equirect.mapping = THREE.EquirectangularReflectionMapping;
          equirect.magFilter = THREE.LinearFilter;
          equirect.minFilter = THREE.LinearFilter;
          equirect.needsUpdate = true;
          const pmrem = new THREE.PMREMGenerator(renderer);
          pmrem.compileEquirectangularShader();
          scene.environment = pmrem.fromEquirectangular(equirect).texture;
          equirect.dispose();
          pmrem.dispose();
        }})();

        const camera = new THREE.PerspectiveCamera(36, container.clientWidth / container.clientHeight, 1.0, 1200.0);
        camera.up.set(0, 0, 1);
        camera.position.set(168, -172, 130);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(6, 0, 34);
        controls.enableDamping = true;
        controls.dampingFactor = 0.07;
        controls.maxDistance = 420;
        controls.minDistance = 45;
        controls.update();

        // AAA 3-Point Alpine Lighting + 2048x2048 Soft Self-Shadows
        const hemiLight = new THREE.HemisphereLight(0xdcebff, 0x1e293b, 0.48);
        scene.add(hemiLight);
        const sunLight = new THREE.DirectionalLight(0xfff7ea, 1.38);
        sunLight.position.set(112, -96, 138);
        sunLight.castShadow = true;
        sunLight.shadow.mapSize.set(2048, 2048);
        sunLight.shadow.camera.left = -74;
        sunLight.shadow.camera.right = 74;
        sunLight.shadow.camera.top = 74;
        sunLight.shadow.camera.bottom = -74;
        sunLight.shadow.camera.near = 25;
        sunLight.shadow.camera.far = 360;
        sunLight.shadow.bias = {SHADOW_BIAS};
        sunLight.shadow.normalBias = {SHADOW_NORMAL_BIAS};
        scene.add(sunLight);
        const rimLight = new THREE.DirectionalLight(0x7dd3fc, 0.36);
        rimLight.position.set(-120, 120, 85);
        scene.add(rimLight);

        const clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 83.5);

        // Materials
        const hasTreeVCol = !!vcBuf;
        const useTriShader = (TRIPLANAR_MODE !== "none") || hasTreeVCol;
        const tanOpt = USE_SMOOTH_TANGENTS ? {{ vertexTangents: true, defines: {{ USE_TANGENT: "" }} }} : {{}};
        const matPBR = new THREE.MeshStandardMaterial({{
          map: diffTex,
          normalMap: normTex,
          normalScale: new THREE.Vector2(1.05, 1.05),
          roughnessMap: roughTex,
          roughness: 1.0,
          metalness: 0.02,
          envMapIntensity: 0.52,
          clippingPlanes: [clipPlane],
          ...tanOpt,
        }});
        if (useTriShader) {{
          matPBR.onBeforeCompile = (shader) => {{
            shader.uniforms.uTriMode = {{ value: TRIPLANAR_MODE === "clay" ? 1.0 : (TRIPLANAR_MODE === "forest" ? 2.0 : (TRIPLANAR_MODE === "rock" ? 3.0 : 0.0)) }};
            shader.uniforms.uHasTree = {{ value: hasTreeVCol ? 1.0 : 0.0 }};
            shader.vertexShader = shader.vertexShader
              .replace(
                "#include <common>",
                `#include <common>
                varying vec3 vObjPos;
                varying vec3 vObjNrm;
                ${{hasTreeVCol ? "attribute vec4 treeVCol; varying vec4 vTreeVCol;" : ""}}`
              )
              .replace(
                "#include <begin_vertex>",
                `#include <begin_vertex>
                vObjPos = position;
                vObjNrm = normal;
                ${{hasTreeVCol ? "vTreeVCol = treeVCol;" : ""}}`
              );
            shader.fragmentShader = shader.fragmentShader
              .replace(
                "#include <common>",
                `#include <common>
                uniform float uTriMode;
                uniform float uHasTree;
                varying vec3 vObjPos;
                varying vec3 vObjNrm;
                ${{hasTreeVCol ? "varying vec4 vTreeVCol;" : ""}}
                float hash31(vec3 p) {{
                  p = fract(p * 0.1031);
                  p += dot(p, p.yzx + 33.33);
                  return fract((p.x + p.y) * p.z);
                }}
                float noise3D(vec3 x) {{
                  vec3 i = floor(x);
                  vec3 f = fract(x);
                  f = f * f * (3.0 - 2.0 * f);
                  return mix(
                    mix(mix(hash31(i + vec3(0,0,0)), hash31(i + vec3(1,0,0)), f.x),
                        mix(hash31(i + vec3(0,1,0)), hash31(i + vec3(1,1,0)), f.x), f.y),
                    mix(mix(hash31(i + vec3(0,0,1)), hash31(i + vec3(1,0,1)), f.x),
                        mix(hash31(i + vec3(0,1,1)), hash31(i + vec3(1,1,1)), f.x), f.y), f.z);
                }}
                float fbm3D(vec3 p) {{
                  float v = 0.0; float a = 0.5;
                  for (int i = 0; i < 3; i++) {{
                    v += a * noise3D(p);
                    p = p * 2.07 + vec3(13.1, 7.7, 5.3);
                    a *= 0.5;
                  }}
                  return v;
                }}`
              )
              .replace(
                "#include <map_fragment>",
                `#include <map_fragment>
                float treeMix = ${{hasTreeVCol ? "clamp(vTreeVCol.a, 0.0, 1.0)" : "0.0"}};
                float steepWall = (uTriMode > 0.5 && vObjPos.z > 25.08) ? smoothstep(0.68, 0.24, abs(normalize(vObjNrm).z)) : 0.0;
                if (steepWall > 0.01 && treeMix < 0.5) {{
                  float du = 0.0075;
                  vec3 s1 = mapTexelToLinear(texture2D(map, vUv + vec2(du, 0.0))).rgb;
                  vec3 s2 = mapTexelToLinear(texture2D(map, vUv - vec2(du, 0.0))).rgb;
                  vec3 s3 = mapTexelToLinear(texture2D(map, vUv + vec2(0.0, du))).rgb;
                  vec3 s4 = mapTexelToLinear(texture2D(map, vUv - vec2(0.0, du))).rgb;
                  vec3 s5 = mapTexelToLinear(texture2D(map, vUv + vec2(du * 0.7, du * 0.7))).rgb;
                  vec3 s6 = mapTexelToLinear(texture2D(map, vUv - vec2(du * 0.7, du * 0.7))).rgb;
                  vec3 avgCol = (diffuseColor.rgb + s1 + s2 + s3 + s4 + s5 + s6) / 7.0;
                  if (uTriMode < 1.5) {{
                    // Terracotta & ochre 3D horizontal sedimentary clay strata (zero vertical UV streak)
                    float warp = fbm3D(vObjPos * 0.16) * 2.8;
                    float strata1 = sin(vObjPos.z * 3.1 + warp);
                    float strata2 = sin(vObjPos.z * 8.4 + warp * 1.4);
                    float fine = fbm3D(vObjPos * 1.25) - 0.5;
                    vec3 bandTint = mix(vec3(0.76, 0.65, 0.59), vec3(1.22, 1.12, 0.96), smoothstep(-0.65, 0.72, strata1));
                    bandTint *= (0.94 + 0.09 * strata2);
                    vec3 cliffCol = avgCol * bandTint * (0.94 + 0.24 * fine);
                    diffuseColor.rgb = mix(diffuseColor.rgb, cliffCol, steepWall * 0.95);
                  }} else if (uTriMode < 2.5) {{
                    // Forest boulder / earth 3D granite stone & moss grain (zero vertical streak)
                    float duf = 0.0095;
                    vec3 sf1 = mapTexelToLinear(texture2D(map, vUv + vec2(duf, duf))).rgb;
                    vec3 sf2 = mapTexelToLinear(texture2D(map, vUv - vec2(duf, duf))).rgb;
                    vec3 fBlur = (avgCol + sf1 + sf2) * 0.3333;
                    float gran = fbm3D(vObjPos * 1.15) - 0.5;
                    float frac = abs(fbm3D(vObjPos * 0.42) * 2.0 - 1.0);
                    float isStone = smoothstep(0.06, 0.015, fBlur.g - max(fBlur.r, fBlur.b));
                    vec3 slateMix = mix(fBlur, vec3(0.18, 0.19, 0.18), isStone * 0.32);
                    vec3 rockCol = slateMix * (0.88 + 0.34 * gran) * (0.84 + 0.24 * frac);
                    diffuseColor.rgb = mix(diffuseColor.rgb, rockCol, steepWall * 0.90);
                  }} else {{
                    // Basalt / Gold Canyon 3D horizontal rock ledges & crisp crystalline quartz detail
                    // Preserve sharp overhead UV texels everywhere except near-vertical UV-stretched scarps
                    float sheer = smoothstep(0.42, 0.14, abs(normalize(vObjNrm).z));
                    vec3 crispBase = mix(diffuseColor.rgb, avgCol, sheer * 0.65);
                    float warp = fbm3D(vObjPos * 0.22) * 2.1;
                    float bed = sin(vObjPos.z * 4.4 + warp);
                    float fineGrain = fbm3D(vObjPos * 4.8) - 0.5;
                    float gran = fbm3D(vObjPos * 1.9) - 0.5;
                    float crack = smoothstep(0.06, 0.42, abs(fbm3D(vObjPos * 0.72) * 2.0 - 1.0));
                    vec3 rockCol = crispBase * (0.90 + 0.14 * bed) * (0.94 + 0.22 * gran + 0.14 * fineGrain) * (0.82 + 0.22 * crack);
                    diffuseColor.rgb = mix(diffuseColor.rgb, rockCol, smoothstep(0.56, 0.18, abs(normalize(vObjNrm).z)) * 0.88);
                  }}
                }}
                ${{hasTreeVCol ? `
                if (treeMix > 0.01) {{
                  vec3 baseV = pow(max(vTreeVCol.rgb, vec3(0.0)), vec3(2.2));
                  float isBark = smoothstep(0.05, 0.01, baseV.g - baseV.r);
                  float needleN = fbm3D(vObjPos * vec3(2.8, 2.8, 3.8)) - 0.5;
                  float macroN = fbm3D(vObjPos * 0.65) - 0.5;
                  float barkN = fbm3D(vec3(vObjPos.x * 2.8, vObjPos.y * 2.8, vObjPos.z * 0.42)) - 0.5;
                  float nrmZ = normalize(vObjNrm).z;
                  float underCav = mix(0.68, 1.06, smoothstep(-0.55, 0.65, nrmZ));
                  vec3 pineShade = baseV * (1.0 + mix(0.34 * needleN + 0.16 * macroN, 0.38 * barkN, isBark)) * mix(underCav, 1.0, isBark);
                  diffuseColor.rgb = mix(diffuseColor.rgb, pineShade, treeMix);
                }}` : ""}}`
              )
              .replace(
                "#include <roughnessmap_fragment>",
                `#include <roughnessmap_fragment>
                float rTreeMix = ${{hasTreeVCol ? "clamp(vTreeVCol.a, 0.0, 1.0)" : "0.0"}};
                float rSteep = (uTriMode > 0.5 && vObjPos.z > 25.08) ? smoothstep(0.55, 0.20, abs(normalize(vObjNrm).z)) : 0.0;
                roughnessFactor = mix(roughnessFactor, 0.88, max(rTreeMix, rSteep * 0.55));
                if (uTriMode > 2.5 && vObjPos.z > 25.05) {{
                  float gM = smoothstep(0.46, 0.72, (diffuseColor.r + diffuseColor.g) * 0.5 - diffuseColor.b) * smoothstep(0.30, 0.60, diffuseColor.r);
                  float wM = smoothstep(0.03, 0.14, min(diffuseColor.g, diffuseColor.b) - diffuseColor.r) * (1.0 - smoothstep(0.32, 0.58, diffuseColor.r));
                  roughnessFactor = mix(roughnessFactor, 0.22, gM * 0.88);
                  roughnessFactor = mix(roughnessFactor, 0.14, wM * 0.85);
                }}`
              )
              .replace(
                "#include <metalnessmap_fragment>",
                `#include <metalnessmap_fragment>
                if (uTriMode > 2.5 && vObjPos.z > 25.05) {{
                  float gMet = smoothstep(0.46, 0.72, (diffuseColor.r + diffuseColor.g) * 0.5 - diffuseColor.b) * smoothstep(0.30, 0.60, diffuseColor.r);
                  metalnessFactor = mix(metalnessFactor, 0.82, gMet);
                }}`
              )
              .replace(
                "#include <normal_fragment_maps>",
                `#include <normal_fragment_maps>
                float nTreeMix = ${{hasTreeVCol ? "clamp(vTreeVCol.a, 0.0, 1.0)" : "0.0"}};
                float nSteep = (uTriMode > 0.5 && vObjPos.z > 25.08) ? smoothstep(0.46, 0.15, abs(normalize(vObjNrm).z)) * 0.55 : 0.0;
                normal = normalize(mix(normal, normalize(vNormal), clamp(max(nTreeMix, nSteep), 0.0, 1.0)));`
              );
          }};
          matPBR.customProgramCacheKey = () => "catan-tri-" + TRIPLANAR_MODE + "-" + (hasTreeVCol ? "vc" : "uv");
        }}
        const matClay = new THREE.MeshStandardMaterial({{
          color: 0xcbd5e1,
          normalMap: normTex,
          normalScale: new THREE.Vector2(0.75, 0.75),
          roughness: 0.54,
          metalness: 0.04,
          envMapIntensity: 0.75,
          clippingPlanes: [clipPlane],
          ...tanOpt,
        }});
        if (useTriShader) {{
          matClay.onBeforeCompile = (shader) => {{
            shader.vertexShader = shader.vertexShader
              .replace("#include <common>", `#include <common>\nvarying vec3 vObjPos;\nvarying vec3 vObjNrm;\n${{hasTreeVCol ? "attribute vec4 treeVCol; varying vec4 vTreeVCol;" : ""}}`)
              .replace("#include <begin_vertex>", `#include <begin_vertex>\nvObjPos = position; vObjNrm = normal;\n${{hasTreeVCol ? "vTreeVCol = treeVCol;" : ""}}`);
            shader.fragmentShader = shader.fragmentShader
              .replace("#include <common>", `#include <common>\nvarying vec3 vObjPos;\nvarying vec3 vObjNrm;\n${{hasTreeVCol ? "varying vec4 vTreeVCol;" : ""}}`)
              .replace("#include <normal_fragment_maps>", `#include <normal_fragment_maps>
                float nTree = ${{hasTreeVCol ? "clamp(vTreeVCol.a, 0.0, 1.0)" : "0.0"}};
                float nSteep = (vObjPos.z > 25.08) ? smoothstep(0.65, 0.25, abs(normalize(vObjNrm).z)) * 0.85 : 0.0;
                normal = normalize(mix(normal, normalize(vNormal), clamp(max(nTree, nSteep), 0.0, 1.0)));`);
          }};
          matClay.customProgramCacheKey = () => "catan-clay-" + TRIPLANAR_MODE + "-" + (hasTreeVCol ? "vc" : "uv");
        }}
        const matSlope = new THREE.MeshStandardMaterial({{
          vertexColors: true,
          roughness: 0.62,
          metalness: 0.04,
          clippingPlanes: [clipPlane],
        }});
        const matWire = new THREE.MeshStandardMaterial({{
          map: diffTex,
          wireframe: true,
          roughness: 0.8,
          clippingPlanes: [clipPlane],
        }});

        const mainMesh = new THREE.Mesh(geom, matPBR);
        mainMesh.castShadow = true;
        mainMesh.receiveShadow = true;
        scene.add(mainMesh);

        // Stencil Solid Capping Group for Z Cross-Section Slider
        const stencilGroup = new THREE.Group();
        const backStencilMat = new THREE.MeshBasicMaterial({{
          depthWrite: false, depthTest: false, colorWrite: false,
          stencilWrite: true, stencilFunc: THREE.AlwaysStencilFunc,
          side: THREE.BackSide,
          stencilFail: THREE.IncrementWrapStencilOp,
          stencilZFail: THREE.IncrementWrapStencilOp,
          stencilZPass: THREE.IncrementWrapStencilOp,
          clippingPlanes: [clipPlane],
        }});
        const frontStencilMat = new THREE.MeshBasicMaterial({{
          depthWrite: false, depthTest: false, colorWrite: false,
          stencilWrite: true, stencilFunc: THREE.AlwaysStencilFunc,
          side: THREE.FrontSide,
          stencilFail: THREE.DecrementWrapStencilOp,
          stencilZFail: THREE.DecrementWrapStencilOp,
          stencilZPass: THREE.DecrementWrapStencilOp,
          clippingPlanes: [clipPlane],
        }});
        const backStencil = new THREE.Mesh(geom, backStencilMat);
        const frontStencil = new THREE.Mesh(geom, frontStencilMat);
        backStencil.renderOrder = 1;
        frontStencil.renderOrder = 2;

        const capPlaneGeom = new THREE.PlaneGeometry(170, 170);
        const capPlaneMat = new THREE.MeshStandardMaterial({{
          color: 0x38bdf8, roughness: 0.55, metalness: 0.1, side: THREE.DoubleSide,
          stencilWrite: true, stencilRef: 0,
          stencilFunc: THREE.NotEqualStencilFunc,
          stencilFail: THREE.ReplaceStencilOp,
          stencilZFail: THREE.ReplaceStencilOp,
          stencilZPass: THREE.ReplaceStencilOp,
        }});
        const capMesh = new THREE.Mesh(capPlaneGeom, capPlaneMat);
        capMesh.renderOrder = 3;
        capMesh.receiveShadow = true;
        capMesh.position.z = 83.5;
        stencilGroup.add(backStencil, frontStencil, capMesh);
        stencilGroup.visible = false;
        scene.add(stencilGroup);

        // Unobstructed Perimeter CAD Dimension Overlays (hidden initially for pure photoreal view)
        const dimGroup = new THREE.Group();
        const triGroup = new THREE.Group();
        const tolGroup = new THREE.Group();
        dimGroup.visible = false;
        triGroup.visible = false;
        tolGroup.visible = false;
        scene.add(dimGroup, triGroup, tolGroup);

        const hexPts = [];
        for (let k = 0; k <= 6; k++) {{
          const a = (30 + 60 * (k % 6)) * Math.PI / 180;
          hexPts.push(new THREE.Vector3(R * Math.cos(a), R * Math.sin(a), 0.25));
        }}
        // Bottom perimeter outline + top base deck Z=25mm outline
        const hexLineBot = new THREE.Line(new THREE.BufferGeometry().setFromPoints(hexPts), new THREE.LineBasicMaterial({{ color: 0x38bdf8 }}));
        const hexLineTop = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(hexPts.map(p => new THREE.Vector3(p.x, p.y, Z_BASE + 0.15))),
          new THREE.LineBasicMaterial({{ color: 0x38bdf8, transparent: true, opacity: 0.65 }})
        );
        dimGroup.add(hexLineBot, hexLineTop);

        // External caliper lines: Flat-to-Flat W=110.85mm (outside bottom -Y), Vertex-to-Vertex D=128.00mm (outside +X), Height 83mm (outside -X)
        function addLine(grp, pts, col) {{
          grp.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({{ color: col }})));
        }}
        addLine(dimGroup, [new THREE.Vector3(-APOTHEM, -72, 0.3), new THREE.Vector3(APOTHEM, -72, 0.3)], 0x38bdf8);
        addLine(dimGroup, [new THREE.Vector3(-APOTHEM, -64, 0.3), new THREE.Vector3(-APOTHEM, -75, 0.3)], 0x38bdf8);
        addLine(dimGroup, [new THREE.Vector3(APOTHEM, -64, 0.3), new THREE.Vector3(APOTHEM, -75, 0.3)], 0x38bdf8);
        const spW = makeTextSprite("W = 110.85 mm", "#38bdf8", 7.2);
        spW.position.set(0, -79, 2);
        dimGroup.add(spW);

        addLine(dimGroup, [new THREE.Vector3(64, -R, 0.3), new THREE.Vector3(64, R, 0.3)], 0xfacc15);
        const spD = makeTextSprite("D = 128.00 mm (L=64)", "#facc15", 7.0);
        spD.position.set(78, 0, 2);
        dimGroup.add(spD);

        addLine(dimGroup, [new THREE.Vector3(-68, 0, 0), new THREE.Vector3(-68, 0, Z_PEAK)], 0xf472b6);
        addLine(dimGroup, [new THREE.Vector3(-68, 0, Z_PEAK), new THREE.Vector3(-4, 0, Z_PEAK)], 0xf472b6);
        const spH = makeTextSprite("Z = 83.00 mm", "#f472b6", 7.2);
        spH.position.set(-76, 0, 46);
        dimGroup.add(spH);

        // 6 Equilateral Triangles (64x64x64mm, h=55.43mm) clearly suspended BELOW at Z=-0.2mm & ABOVE at Z=86mm
        for (let k = 0; k < 6; k++) {{
          const a = (30 + 60 * k) * Math.PI / 180;
          const x = R * Math.cos(a), y = R * Math.sin(a);
          addLine(triGroup, [new THREE.Vector3(0, 0, -0.2), new THREE.Vector3(x, y, -0.2)], 0xfacc15);
          addLine(triGroup, [new THREE.Vector3(0, 0, 86.0), new THREE.Vector3(x, y, 86.0)], 0xfacc15);
        }}
        addLine(triGroup, hexPts.map(p => new THREE.Vector3(p.x, p.y, 86.0)), 0xfacc15);
        // Apothem line h=55.43mm at Z=86mm
        addLine(triGroup, [new THREE.Vector3(0, 0, 86.0), new THREE.Vector3(APOTHEM, 0, 86.0)], 0xf472b6);
        const spTri = makeTextSprite("6× Δ Equil. 64×64×64mm (h=55.43mm)", "#facc15", 6.2);
        spTri.position.set(0, 32, 91.0);
        triGroup.add(spTri);

        // Tolerance overlays: 3.50mm inner road lane + 6x Ø20.00mm vertex pads + bottom Ø10.20mm magnet pocket
        const rIn = (APOTHEM - ROAD_MARGIN) / Math.cos(Math.PI / 6);
        const roadPts = [];
        for (let k = 0; k <= 6; k++) {{
          const a = (30 + 60 * (k % 6)) * Math.PI / 180;
          roadPts.push(new THREE.Vector3(rIn * Math.cos(a), rIn * Math.sin(a), Z_BASE + 0.25));
        }}
        addLine(tolGroup, roadPts, 0x4ade80);
        for (let k = 0; k < 6; k++) {{
          const a = (30 + 60 * k) * Math.PI / 180;
          const vx = R * Math.cos(a), vy = R * Math.sin(a);
          const ring = [];
          for (let j = 0; j <= 28; j++) {{
            const th = (j / 28) * 2 * Math.PI;
            ring.push(new THREE.Vector3(vx + PAD_R * Math.cos(th), vy + PAD_R * Math.sin(th), Z_BASE + 0.25));
          }}
          addLine(tolGroup, ring, 0xfb923c);
        }}
        const magRing = [];
        for (let j = 0; j <= 32; j++) {{
          const th = (j / 32) * 2 * Math.PI;
          magRing.push(new THREE.Vector3(MAG_R * Math.cos(th), MAG_R * Math.sin(th), 0.1));
        }}
        addLine(tolGroup, magRing, 0xf43f5e);

        // UI Events
        const modeBtns = document.querySelectorAll("[data-mode]");
        modeBtns.forEach(btn => btn.addEventListener("click", () => {{
          modeBtns.forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          const mName = btn.dataset.mode;
          if (mName === "pbr") mainMesh.material = matPBR;
          else if (mName === "clay") mainMesh.material = matClay;
          else if (mName === "slope") mainMesh.material = matSlope;
          else if (mName === "wire") mainMesh.material = matWire;
        }}));

        let currentSunZ = 138;
        const atmoBtns = document.querySelectorAll("[data-atmo]");
        atmoBtns.forEach(btn => btn.addEventListener("click", () => {{
          atmoBtns.forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          const aName = btn.dataset.atmo;
          if (aName === "alpine") {{
            sunLight.color.setHex(0xfff7ea); sunLight.intensity = 1.38;
            hemiLight.color.setHex(0xdcebff); hemiLight.groundColor.setHex(0x1e293b); hemiLight.intensity = 0.48;
            rimLight.color.setHex(0x7dd3fc); rimLight.intensity = 0.36;
            renderer.toneMappingExposure = 0.96; currentSunZ = 138;
          }} else if (aName === "sunset") {{
            sunLight.color.setHex(0xffb469); sunLight.intensity = 1.68;
            hemiLight.color.setHex(0x9ec5ff); hemiLight.groundColor.setHex(0x241928); hemiLight.intensity = 0.44;
            rimLight.color.setHex(0x60a5fa); rimLight.intensity = 0.46;
            renderer.toneMappingExposure = 1.02; currentSunZ = 74;
          }} else {{
            sunLight.color.setHex(0xeef4ff); sunLight.intensity = 0.96;
            hemiLight.color.setHex(0xe2ecff); hemiLight.groundColor.setHex(0x334155); hemiLight.intensity = 0.85;
            rimLight.color.setHex(0xbae6fd); rimLight.intensity = 0.36;
            renderer.toneMappingExposure = 0.94; currentSunZ = 165;
          }}
          const az = parseFloat(document.getElementById("sun-slider").value) * Math.PI / 180;
          sunLight.position.set(145 * Math.cos(az), 145 * Math.sin(az), currentSunZ);
        }}));

        document.getElementById("chk-dims").addEventListener("change", e => dimGroup.visible = e.target.checked);
        document.getElementById("chk-tris").addEventListener("change", e => triGroup.visible = e.target.checked);
        document.getElementById("chk-tol").addEventListener("change", e => tolGroup.visible = e.target.checked);

        let spinEnabled = false;
        document.getElementById("chk-spin").addEventListener("change", e => spinEnabled = e.target.checked);

        const clipSlider = document.getElementById("clip-slider");
        const clipVal = document.getElementById("clip-val");
        clipSlider.addEventListener("input", () => {{
          const z = parseFloat(clipSlider.value);
          clipPlane.constant = z;
          capMesh.position.z = z;
          const slicing = z < 83.1;
          stencilGroup.visible = slicing;
          clipVal.textContent = slicing ? z.toFixed(1) + " mm" : "83.0 mm";
        }});

        const sunSlider = document.getElementById("sun-slider");
        const sunVal = document.getElementById("sun-val");
        sunSlider.addEventListener("input", () => {{
          const az = parseFloat(sunSlider.value) * Math.PI / 180;
          sunLight.position.set(145 * Math.cos(az), 145 * Math.sin(az), currentSunZ);
          sunVal.textContent = sunSlider.value + "°";
        }});

        document.querySelectorAll("[data-cam]").forEach(btn => btn.addEventListener("click", () => {{
          const c = btn.dataset.cam;
          if (c === "iso") {{ camera.position.set(168, -172, 130); controls.target.set(6, 0, 34); }}
          else if (c === "top") {{ camera.position.set(0, -1.5, 225); controls.target.set(0, 0, 28); }}
          else if (c === "north") {{ camera.position.set(-22, 165, 92); controls.target.set(0, 0, 40); }}
          else if (c === "eye") {{ camera.position.set(125, -110, 12); controls.target.set(0, 0, 22); }}
          controls.update();
        }}));

        window.addEventListener("resize", () => {{
          camera.aspect = container.clientWidth / container.clientHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(container.clientWidth, container.clientHeight);
        }});

        document.getElementById("loading-overlay").style.display = "none";

        function animate() {{
          requestAnimationFrame(animate);
          if (spinEnabled) {{
            mainMesh.rotation.z += 0.004;
            backStencil.rotation.z += 0.004;
            frontStencil.rotation.z += 0.004;
            tolGroup.rotation.z += 0.004;
          }}
          controls.update();
          renderer.clear();
          renderer.render(scene, camera);
        }}
        animate();
      }} catch (err) {{
        document.getElementById("load-msg").textContent = "Error al inicializar WebGL/GZip: " + err.message;
      }}
    }}
    init();
  }})();
  </script>
</body>
</html>
"""

    for p in TARGET_PATHS:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as wf:
            wf.write(html)
        sz_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"  Deployed viewer: {p} ({sz_mb:.2f} MB)")

    print(f"[Viewer V3] Complete in {time.time()-t0:.2f}s (diff={diff_sz/1024:.0f}KB, norm={norm_sz/1024:.0f}KB, rough={rough_sz/1024:.0f}KB)")


if __name__ == "__main__":
    build_viewer()
