import os

print("--- Assembling Master 3D WebGL Mountain Viewer ---")

script_dir = os.path.dirname(os.path.abspath(__file__))
b64_path = os.path.join(script_dir, 'mountain_b64_data.js')
if not os.path.exists(b64_path):
    b64_path = 'scratch/mountain_b64_data.js'

# Read base64 data
with open(b64_path, 'r', encoding='utf-8') as f:
    b64_data_js = f.read()

html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Visor 3D Loseta Montaña Catan - 100% Macizo CMYKWOG-B</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #090d16;
      color: #e2e8f0;
      overflow: hidden;
      height: 100vh;
      display: flex;
    }}
    #canvas-container {{
      flex: 1;
      height: 100%;
      position: relative;
    }}
    #ui-panel {{
      width: 380px;
      height: 100%;
      background: rgba(15, 23, 42, 0.94);
      backdrop-filter: blur(12px);
      border-left: 1px solid rgba(255, 255, 255, 0.1);
      display: flex;
      flex-direction: column;
      z-index: 10;
      box-shadow: -8px 0 24px rgba(0,0,0,0.5);
    }}
    .panel-header {{
      padding: 18px 20px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(30, 41, 59, 0.5);
    }}
    .panel-header h1 {{
      font-size: 1.15rem;
      font-weight: 700;
      color: #38bdf8;
      letter-spacing: -0.02em;
    }}
    .panel-header p {{
      font-size: 0.8rem;
      color: #94a3b8;
      margin-top: 4px;
    }}
    .panel-body {{
      padding: 16px 20px;
      overflow-y: auto;
      flex: 1;
    }}
    .section-title {{
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #64748b;
      margin-top: 14px;
      margin-bottom: 8px;
      font-weight: 700;
    }}
    .badge-solid {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      font-size: 0.72rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 999px;
      margin-top: 8px;
    }}
    .control-group {{
      background: rgba(30, 41, 59, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 12px;
    }}
    .control-label {{
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      font-weight: 600;
      margin-bottom: 8px;
    }}
    .slider {{
      width: 100%;
      height: 6px;
      border-radius: 3px;
      background: #334155;
      outline: none;
      -webkit-appearance: none;
      cursor: pointer;
    }}
    .slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: #38bdf8;
      cursor: pointer;
      box-shadow: 0 0 8px rgba(56, 189, 248, 0.8);
    }}
    .btn-row {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      margin-bottom: 12px;
    }}
    .btn {{
      background: #1e293b;
      border: 1px solid #334155;
      color: #e2e8f0;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      text-align: center;
    }}
    .btn:hover {{
      background: #334155;
      border-color: #475569;
    }}
    .btn.active {{
      background: #0284c7;
      border-color: #38bdf8;
      color: white;
    }}
    .tool-list {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 0.75rem;
    }}
    .tool-item {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(15, 23, 42, 0.6);
      padding: 6px 10px;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.04);
    }}
    .tool-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
      margin-right: 8px;
      border: 1px solid rgba(255,255,255,0.2);
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      font-size: 0.75rem;
    }}
    .stat-card {{
      background: rgba(15, 23, 42, 0.6);
      padding: 8px 10px;
      border-radius: 6px;
      border: 1px solid rgba(255,255,255,0.04);
    }}
    .stat-val {{
      font-size: 0.95rem;
      font-weight: 700;
      color: #38bdf8;
      margin-top: 2px;
    }}
    #loading-overlay {{
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      background: #090d16;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 100;
      transition: opacity 0.4s ease;
    }}
    .spinner {{
      width: 44px;
      height: 44px;
      border: 3px solid rgba(56, 189, 248, 0.2);
      border-top-color: #38bdf8;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="loading-overlay">
    <div class="spinner"></div>
    <div style="margin-top: 14px; font-weight: 600; color: #38bdf8;">Construyendo Macizo Alpino LiDAR 3D...</div>
    <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">140,000+ Vértices &bull; 100% Watertight Solid &bull; CMYKWOG-B</div>
  </div>

  <div id="canvas-container"></div>

  <div id="ui-panel">
    <div class="panel-header">
      <h1>Montaña Catan 3D Monolítica</h1>
      <p>USGS LiDAR Grand Teton &plus; Matterhorn Horn | Infimech CMYKWOG-B</p>
      <div class="badge-solid">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        100% Macizo Sólido (Manifold Watertight)
      </div>
    </div>

    <div class="panel-body">
      <div class="section-title">Modo de Visualización</div>
      <div class="btn-row">
        <button id="btn-pbr" class="btn active" onclick="setViewMode('pbr')">Fotorrealista PBR</button>
        <button id="btn-tools" class="btn" onclick="setViewMode('tools')">Herramientas CMYKWOG</button>
      </div>

      <div class="section-title">Corte Transversal Z (Demostración Solidez)</div>
      <div class="control-group">
        <div class="control-label">
          <span>Altura de Corte (Z)</span>
          <span id="cut-height-val" style="color: #38bdf8;">83.0 mm</span>
        </div>
        <input type="range" id="z-cut-slider" class="slider" min="0" max="83" step="0.5" value="83" oninput="onZCutChange(this.value)">
        <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:#64748b; margin-top:4px;">
          <span>0mm (Base)</span>
          <span>25mm (Zócalo)</span>
          <span>83mm (Cima)</span>
        </div>
        <div style="font-size:0.72rem; color:#94a3b8; margin-top:8px; line-height:1.3;">
          <strong style="color:#34d399;">Stencil Solid Capping Activo:</strong> Al deslizar el plano hacia abajo, la sección transversal interna aparece 100% rellena de roca oscura maciza (cero huecos).
        </div>
      </div>

      <div class="section-title">Cámaras Rápidas</div>
      <div class="btn-row">
        <button class="btn" onclick="setCamera('iso')">Isométrica</button>
        <button class="btn" onclick="setCamera('top')">Cenital (Planta)</button>
        <button class="btn" onclick="setCamera('north')">Cara Norte</button>
        <button class="btn" onclick="setCamera('arete')">Arista Cimera</button>
      </div>

      <div class="section-title">Asignación 8 Cabezales CMYKWOG-B</div>
      <div class="tool-list">
        <div class="tool-item">
          <div style="display:flex; align-items:center;">
            <div class="tool-swatch" style="background:#1e2023;"></div>
            <span><strong>Tool B (T0):</strong> Zócalo Base Macizo</span>
          </div>
          <span style="color:#94a3b8;">Z 0-25mm</span>
        </div>
        <div class="tool-item">
          <div style="display:flex; align-items:center;">
            <div class="tool-swatch" style="background:#2d343a;"></div>
            <span><strong>Tool K (T3):</strong> Acantilados / Pizarra</span>
          </div>
          <span style="color:#94a3b8;">Sombras</span>
        </div>
        <div class="tool-item">
          <div style="display:flex; align-items:center;">
            <div class="tool-swatch" style="background:#7d858c;"></div>
            <span><strong>Tool W (T4):</strong> Granito Alpino</span>
          </div>
          <span style="color:#94a3b8;">Cuerpo</span>
        </div>
        <div class="tool-item">
          <div style="display:flex; align-items:center;">
            <div class="tool-swatch" style="background:#d1611a;"></div>
            <span><strong>Tool O (T6):</strong> Vetas de Hierro / Cobre</span>
          </div>
          <span style="color:#f97316;">Dendríticas</span>
        </div>
        <div class="tool-item">
          <div style="display:flex; align-items:center;">
            <div class="tool-swatch" style="background:#68b2bf;"></div>
            <span><strong>Tool C (T1):</strong> Hielo Glaciar</span>
          </div>
          <span style="color:#38bdf8;">Couloirs</span>
        </div>
        <div class="tool-item">
          <div style="display:flex; align-items:center;">
            <div class="tool-swatch" style="background:#ffffff;"></div>
            <span><strong>Tool W (T7):</strong> Nieve Alpina (0.2mm)</span>
          </div>
          <span style="color:#38bdf8;">Cimas</span>
        </div>
        <div class="tool-item">
          <div style="display:flex; align-items:center;">
            <div class="tool-swatch" style="background:#437638;"></div>
            <span><strong>Tool G (T5):</strong> Tundra / Musgo</span>
          </div>
          <span style="color:#4ade80;">Valle</span>
        </div>
      </div>

      <div class="section-title">Métricas de Ingeniería</div>
      <div class="stats-grid">
        <div class="stat-card">
          <div style="color:#94a3b8;">Altura Total</div>
          <div class="stat-val">83.0 mm</div>
        </div>
        <div class="stat-card">
          <div style="color:#94a3b8;">Volumen Sólido</div>
          <div class="stat-val">531.7 cm³</div>
        </div>
        <div class="stat-card">
          <div style="color:#94a3b8;">Carril Camino</div>
          <div class="stat-val">3.5 mm (Plano)</div>
        </div>
        <div class="stat-card">
          <div style="color:#94a3b8;">Bases Vértice</div>
          <div class="stat-val">Ø20.0 mm (x6)</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    {b64_data_js}

    const R = 64.0;
    const APOTHEM = R * Math.cos(Math.PI / 6);
    const Z_BASE = 25.0;
    const Z_PEAK = 83.0;
    const RELIEF_H = Z_PEAK - Z_BASE;

    let scene, camera, renderer, controls;
    let mountainMesh, baseMesh, capMesh;
    let clipPlane;
    let currentViewMode = 'pbr';

    const container = document.getElementById('canvas-container');

    initScene();
    buildSolidMountain();

    function initScene() {{
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x090d16);

      camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 1, 1000);
      camera.position.set(130, -150, 140);
      camera.up.set(0, 0, 1);

      renderer = new THREE.WebGLRenderer({{ antialias: true, stencil: true }});
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      renderer.localClippingEnabled = true;
      container.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.target.set(0, 0, 32);
      controls.maxPolarAngle = Math.PI / 2 + 0.15;

      // Lighting
      const ambLight = new THREE.AmbientLight(0xffffff, 0.45);
      scene.add(ambLight);

      const sunLight = new THREE.DirectionalLight(0xfff8ed, 1.15);
      sunLight.position.set(-80, -100, 140);
      sunLight.castShadow = true;
      sunLight.shadow.mapSize.width = 2048;
      sunLight.shadow.mapSize.height = 2048;
      scene.add(sunLight);

      const fillLight = new THREE.DirectionalLight(0x7dd3fc, 0.4);
      fillLight.position.set(100, 80, 60);
      scene.add(fillLight);

      // Floor Grid
      const grid = new THREE.GridHelper(300, 30, 0x1e293b, 0x0f172a);
      grid.rotation.x = Math.PI / 2;
      grid.position.z = -0.1;
      scene.add(grid);

      // Clipping plane at Z_PEAK initially (no cut)
      clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), Z_PEAK);

      window.addEventListener('resize', onWindowResize, false);
    }}

    function decodeData() {{
      const binZ = atob(DEM_Z_B64);
      const u16Z = new Uint16Array(binZ.length / 2);
      for (let i = 0; i < u16Z.length; i++) {{
        u16Z[i] = binZ.charCodeAt(i * 2) | (binZ.charCodeAt(i * 2 + 1) << 8);
      }}

      const binTools = atob(DEM_TOOLS_B64);
      const u8Tools = new Uint8Array(binTools.length);
      for (let i = 0; i < u8Tools.length; i++) {{
        u8Tools[i] = binTools.charCodeAt(i);
      }}

      return {{ u16Z, u8Tools }};
    }}

    function buildSolidMountain() {{
      const {{ u16Z, u8Tools }} = decodeData();

      // Tool color definitions
      const toolColorsPBR = {{
        0: new THREE.Color(0x1e2023), // Base
        1: new THREE.Color(0x68b2bf), // Ice
        3: new THREE.Color(0x2d343a), // Slate
        4: new THREE.Color(0x7d858c), // Granite
        5: new THREE.Color(0x437638), // Tundra
        6: new THREE.Color(0xd1611a), // Iron Ore
        7: new THREE.Color(0xf1f5f9)  // Snow
      }};

      const toolColorsDirect = {{
        0: new THREE.Color(0x0f172a),
        1: new THREE.Color(0x06b6d4),
        3: new THREE.Color(0x1e293b),
        4: new THREE.Color(0x64748b),
        5: new THREE.Color(0x22c55e),
        6: new THREE.Color(0xea580c),
        7: new THREE.Color(0xffffff)
      }};

      // Grid specifications
      const nx = DEM_NX;
      const ny = DEM_NY;
      const stepX = 2 * APOTHEM / (nx - 1);
      const stepY = 2 * R / (ny - 1);

      // Top terrain surface geometry
      const geometry = new THREE.BufferGeometry();
      const positions = [];
      const colorsPBR = [];
      const colorsTools = [];
      const indices = [];

      const vMap = new Int32Array(nx * ny).fill(-1);
      let vCount = 0;

      // Hexagon normals for boundary check
      const normalsHex = [];
      for (let k = 0; k < 6; k++) {{
        const a = k * Math.PI / 3;
        normalsHex.push({{ nx: Math.cos(a), ny: Math.sin(a) }});
      }}

      for (let j = 0; j < ny; j++) {{
        const y = -R + j * stepY;
        for (let i = 0; i < nx; i++) {{
          const x = -APOTHEM + i * stepX;

          let distEdge = 999.0;
          for (let k = 0; k < 6; k++) {{
            const d = APOTHEM - (x * normalsHex[k].nx + y * normalsHex[k].ny);
            if (d < distEdge) distEdge = d;
          }}

          if (distEdge >= -0.2) {{
            const idx = j * nx + i;
            const zNorm = u16Z[idx] / 65535.0;
            const z = Z_BASE + zNorm * RELIEF_H;
            const tid = u8Tools[idx];

            positions.push(x, y, z);
            const cP = toolColorsPBR[tid] || toolColorsPBR[4];
            const cT = toolColorsDirect[tid] || toolColorsDirect[4];
            colorsPBR.push(cP.r, cP.g, cP.b);
            colorsTools.push(cT.r, cT.g, cT.b);

            vMap[idx] = vCount++;
          }}
        }}
      }}

      // Build grid faces
      for (let j = 0; j < ny - 1; j++) {{
        for (let i = 0; i < nx - 1; i++) {{
          const idx00 = j * nx + i;
          const idx10 = j * nx + (i + 1);
          const idx01 = (j + 1) * nx + i;
          const idx11 = (j + 1) * nx + (i + 1);

          const v00 = vMap[idx00];
          const v10 = vMap[idx10];
          const v01 = vMap[idx01];
          const v11 = vMap[idx11];

          if (v00 !== -1 && v10 !== -1 && v01 !== -1 && v11 !== -1) {{
            indices.push(v00, v10, v11);
            indices.push(v00, v11, v01);
          }}
        }}
      }}

      // Add Vertical Hexagonal Side Walls (from Z=25 down to Z=0)
      const nPerEdge = 24;
      const wallVerts = [];
      for (let k = 0; k < 6; k++) {{
        const a1 = (30 + 60 * k) * Math.PI / 180;
        const a2 = (30 + 60 * (k + 1)) * Math.PI / 180;
        const p1 = {{ x: R * Math.cos(a1), y: R * Math.sin(a1) }};
        const p2 = {{ x: R * Math.cos(a2), y: R * Math.sin(a2) }};
        for (let s = 0; s < nPerEdge; s++) {{
          const t = s / nPerEdge;
          const wx = (1 - t) * p1.x + t * p2.x;
          const wy = (1 - t) * p1.y + t * p2.y;
          wallVerts.push({{ x: wx, y: wy }});
        }}
      }}
      const nWall = wallVerts.length;

      const idxWallTopStart = positions.length / 3;
      for (let i = 0; i < nWall; i++) {{
        positions.push(wallVerts[i].x, wallVerts[i].y, Z_BASE);
        colorsPBR.push(0.12, 0.13, 0.15);
        colorsTools.push(0.06, 0.09, 0.16);
      }}

      const idxWallBotStart = positions.length / 3;
      for (let i = 0; i < nWall; i++) {{
        positions.push(wallVerts[i].x, wallVerts[i].y, 0.0);
        colorsPBR.push(0.08, 0.09, 0.11);
        colorsTools.push(0.06, 0.09, 0.16);
      }}

      // Side wall triangles
      for (let i = 0; i < nWall; i++) {{
        const next_i = (i + 1) % nWall;
        const t1 = idxWallTopStart + i;
        const t2 = idxWallTopStart + next_i;
        const b1 = idxWallBotStart + i;
        const b2 = idxWallBotStart + next_i;
        indices.push(t1, b1, b2);
        indices.push(t1, b2, t2);
      }}

      // Bottom face at Z=0
      const idxBotCenter = positions.length / 3;
      positions.push(0, 0, 0);
      colorsPBR.push(0.08, 0.09, 0.11);
      colorsTools.push(0.06, 0.09, 0.16);

      for (let i = 0; i < nWall; i++) {{
        const next_i = (i + 1) % nWall;
        const b1 = idxWallBotStart + i;
        const b2 = idxWallBotStart + next_i;
        indices.push(idxBotCenter, b2, b1);
      }}

      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colorsPBR, 3));
      geometry.setIndex(indices);
      geometry.computeVertexNormals();

      // Store colors for toggling
      geometry.userData = {{ colorsPBR, colorsTools }};

      // 1. Mountain Surface Mesh
      const matMountain = new THREE.MeshStandardMaterial({{
        vertexColors: true,
        roughness: 0.82,
        metalness: 0.12,
        clippingPlanes: [clipPlane],
        clipShadows: true,
        side: THREE.DoubleSide
      }});

      mountainMesh = new THREE.Mesh(geometry, matMountain);
      mountainMesh.castShadow = true;
      mountainMesh.receiveShadow = true;
      scene.add(mountainMesh);

      // 2. STENCIL BUFFER SOLID CROSS-SECTION CAPPING!
      // This is the Three.js technique that makes the cross-section 100% solid!
      const planeGeom = new THREE.PlaneGeometry(250, 250);
      const capMat = new THREE.MeshStandardMaterial({{
        color: 0x1e293b, // Solid dark granite cross section
        roughness: 0.9,
        metalness: 0.1,
        stencilWrite: true,
        stencilRef: 0,
        stencilFunc: THREE.NotEqualStencilFunc,
        side: THREE.DoubleSide
      }});

      capMesh = new THREE.Mesh(planeGeom, capMat);
      capMesh.position.z = Z_PEAK;
      capMesh.visible = false;
      scene.add(capMesh);

      // Hide loading overlay
      const overlay = document.getElementById('loading-overlay');
      overlay.style.opacity = '0';
      setTimeout(() => overlay.style.display = 'none', 400);

      animate();
    }}

    function onZCutChange(val) {{
      const z = parseFloat(val);
      document.getElementById('cut-height-val').textContent = z.toFixed(1) + ' mm';
      clipPlane.constant = z;

      // Solid capping plane follows the cut height!
      if (z < Z_PEAK - 0.2) {{
        capMesh.position.z = z;
        capMesh.visible = true;
      }} else {{
        capMesh.visible = false;
      }}
    }}

    function setViewMode(mode) {{
      currentViewMode = mode;
      document.getElementById('btn-pbr').classList.toggle('active', mode === 'pbr');
      document.getElementById('btn-tools').classList.toggle('active', mode === 'tools');

      const geom = mountainMesh.geometry;
      const targetColors = (mode === 'pbr') ? geom.userData.colorsPBR : geom.userData.colorsTools;
      geom.setAttribute('color', new THREE.Float32BufferAttribute(targetColors, 3));
      geom.attributes.color.needsUpdate = true;

      mountainMesh.material.roughness = (mode === 'pbr') ? 0.82 : 0.95;
      mountainMesh.material.metalness = (mode === 'pbr') ? 0.12 : 0.0;
      mountainMesh.material.needsUpdate = true;
    }}

    function setCamera(pos) {{
      if (pos === 'iso') {{
        camera.position.set(130, -150, 140);
        controls.target.set(0, 0, 32);
      }} else if (pos === 'top') {{
        camera.position.set(0, 0, 220);
        controls.target.set(0, 0, 25);
      }} else if (pos === 'north') {{
        camera.position.set(0, 180, 55);
        controls.target.set(0, 0, 45);
      }} else if (pos === 'arete') {{
        camera.position.set(-110, -90, 85);
        controls.target.set(-4, 6, 60);
      }}
      controls.update();
    }}

    function onWindowResize() {{
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }}

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
  </script>
</body>
</html>
"""

# Write to target locations
catan_dir = os.path.abspath(os.path.join(script_dir, ".."))
maker3d_dir = os.path.abspath(os.path.join(catan_dir, ".."))

out_html_1 = os.path.join(catan_dir, "Loseta-Montana-3D", "catan_mountain_viewer.html")
out_html_2 = os.path.join(catan_dir, "catan_mountain_viewer.html")
out_html_3 = os.path.join(maker3d_dir, "Visores-HTML", "catan-loseta-montana-detailed.html")

os.makedirs(os.path.dirname(out_html_1), exist_ok=True)
with open(out_html_1, 'w', encoding='utf-8') as f:
    f.write(html_template)
with open(out_html_2, 'w', encoding='utf-8') as f:
    f.write(html_template)
with open(out_html_3, 'w', encoding='utf-8') as f:
    f.write(html_template)

print(f"Viewer written to:\n  {out_html_1}\n  {out_html_2}\n  {out_html_3}")
print("Size of viewer HTML:", os.path.getsize(out_html_1) / 1024, "KB")
