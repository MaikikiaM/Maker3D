import os
import shutil

print("--- Assembling Master Full-Color 3D WebGL Viewer with Detailed Base Dimensions ---")

proj_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

with open('scratch/fullcolor_viewer_data.js', 'r', encoding='utf-8') as f:
    b64_data_js = f.read()

html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Visor 3D Full-Color Loseta Montaña Catan - 5.11M Triángulos</title>
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
      width: 420px;
      height: 100%;
      background: rgba(15, 23, 42, 0.95);
      backdrop-filter: blur(14px);
      border-left: 1px solid rgba(255, 255, 255, 0.1);
      display: flex;
      flex-direction: column;
      z-index: 10;
      box-shadow: -8px 0 28px rgba(0,0,0,0.6);
    }}
    .panel-header {{
      padding: 16px 20px;
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
      font-size: 0.76rem;
      color: #94a3b8;
      margin-top: 4px;
      line-height: 1.3;
    }}
    .panel-body {{
      padding: 14px 20px;
      overflow-y: auto;
      flex: 1;
    }}
    .section-title {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #64748b;
      margin-top: 14px;
      margin-bottom: 8px;
      font-weight: 700;
    }}
    .badge-fullcolor {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(168, 85, 247, 0.2));
      border: 1px solid rgba(56, 189, 248, 0.4);
      color: #38bdf8;
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
      margin-bottom: 10px;
    }}
    .control-label {{
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      font-weight: 600;
      color: #cbd5e1;
      margin-bottom: 8px;
    }}
    .slider {{
      width: 100%;
      height: 6px;
      background: #334155;
      border-radius: 3px;
      outline: none;
      -webkit-appearance: none;
      cursor: pointer;
    }}
    .slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #38bdf8;
      cursor: pointer;
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.6);
    }}
    .btn-row {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      margin-bottom: 8px;
    }}
    .btn {{
      background: #1e293b;
      border: 1px solid #334155;
      color: #cbd5e1;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }}
    .btn:hover {{
      background: #334155;
      color: #ffffff;
      border-color: #475569;
    }}
    .btn.active {{
      background: #0284c7;
      color: #ffffff;
      border-color: #38bdf8;
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
    }}
    .spec-card {{
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 12px;
      font-size: 0.75rem;
      line-height: 1.5;
      color: #94a3b8;
    }}
    .spec-card strong {{
      color: #e2e8f0;
    }}
    #loading-overlay {{
      position: absolute;
      inset: 0;
      background: #090d16;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 50;
      transition: opacity 0.4s ease;
    }}
    .spinner {{
      width: 44px;
      height: 44px;
      border: 3px solid rgba(56, 189, 248, 0.2);
      border-top-color: #38bdf8;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="loading-overlay">
    <div class="spinner"></div>
    <div style="margin-top: 16px; font-weight: 600; color: #38bdf8; font-size: 0.88rem;">Cargando Topografía 3D y Textura 2K PBR...</div>
    <div style="margin-top: 6px; font-size: 0.72rem; color: #64748b;">Malla de alta densidad (5.11M Triángulos)</div>
  </div>

  <div id="canvas-container"></div>

  <div id="ui-panel">
    <div class="panel-header">
      <h1>Montaña Catan 3D Full-Color</h1>
      <p>Espectro Infinito Fotorrealista | 5.11M Triángulos | Base Hexagonal Catan</p>
      <div class="badge-fullcolor">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 4.24 4.24"/><path d="m14.83 9.17 4.24-4.24"/><path d="m14.83 14.83 4.24 4.24"/><path d="m9.17 14.83-4.24 4.24"/></svg>
        Full-Color Continuo (Cero Píxeles / Cero Pasos)
      </div>
    </div>

    <div class="panel-body">
      <div class="section-title">Modo de Shading / Superficie</div>
      <div class="btn-row">
        <button id="btn-tex" class="btn active" onclick="setShadingMode('textured')">Texturizado 2K PBR</button>
        <button id="btn-wire" class="btn" onclick="toggleWireframe()">Malla Alámbrica</button>
      </div>

      <div class="section-title">Cotas y Visualización de Medidas 3D</div>
      <div class="btn-row">
        <button id="btn-cotas" class="btn active" onclick="toggleDimensions()">📐 Cotas 3D Base</button>
        <button id="btn-tri" class="btn" onclick="toggleTriangles()">🔺 6 Triángulos Base</button>
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
          <strong style="color:#34d399;">Stencil Solid Capping:</strong> Sección transversal 100% maciza en granito oscuro (cero cavidades internas).
        </div>
      </div>

      <div class="section-title">Cámaras Rápidas e Inspección</div>
      <div class="btn-row">
        <button class="btn" onclick="setCamera('iso')">Isométrica General</button>
        <button class="btn" onclick="setCamera('top')">Cenital (Planta)</button>
        <button class="btn" onclick="setCamera('base')">Vista Base Inferior</button>
        <button class="btn" onclick="setCamera('zoom_summit')">🔍 Super-Zoom Cima</button>
      </div>

      <!-- SECCIÓN SOLICITADA POR EL USUARIO: COTAS EXACTAS DE LA BASE -->
      <div class="section-title">Cotas de Ingeniería - Base Hexagonal</div>
      <div class="spec-card" style="border-left: 3px solid #38bdf8; background: rgba(14, 116, 144, 0.12);">
        <div style="font-size:0.8rem; font-weight:700; color:#38bdf8; margin-bottom:6px;">Dimensiones Planas del Hexágono Regular:</div>
        <div>&bull; <strong>Distancia de Lado a Lado (Ancho transversal plano):</strong> <span style="color:#f8fafc; font-weight:700;">110.85 mm</span> (2 &times; Apotema 55.43 mm)</div>
        <div>&bull; <strong>Distancia Vértice a Vértice (Diámetro mayor):</strong> <span style="color:#f8fafc; font-weight:700;">128.00 mm</span> (2 &times; Radio 64.00 mm)</div>
        <div>&bull; <strong>Longitud de cada Lado del Hexágono:</strong> <span style="color:#f8fafc; font-weight:700;">64.00 mm</span></div>
        <div style="margin-top:8px; border-top: 1px solid rgba(255,255,255,0.1); padding-top:6px;">
          <strong style="color:#f59e0b;">Sectores Triangulares (6 Triángulos Equiláteros Base):</strong>
        </div>
        <div>&bull; <strong>Lados de cada triángulo:</strong> 64.00 mm &times; 64.00 mm &times; 64.00 mm</div>
        <div>&bull; <strong>Altura de cada triángulo (Apotema):</strong> 55.43 mm</div>
        <div>&bull; <strong>Área de cada triángulo:</strong> 1,773.62 mm&sup2;</div>
        <div>&bull; <strong>Área total de la base hexagonal:</strong> 10,641.72 mm&sup2; (106.42 cm&sup2;)</div>
        <div style="margin-top:8px; border-top: 1px solid rgba(255,255,255,0.1); padding-top:6px;">
          <strong style="color:#34d399;">Cotas Verticales Z:</strong>
        </div>
        <div>&bull; <strong>Cota Base / Zócalo:</strong> Z = 25.00 mm (Carril carreteras de 3.5 mm plano)</div>
        <div>&bull; <strong>Cota Cumbre Principal:</strong> Z = 83.00 mm (Relieve alpino puro = 58.00 mm)</div>
        <div>&bull; <strong>Alojamiento Imán Inferior:</strong> &Oslash;10.2 &times; 5.2 mm (Abierto por Z = 0.0 mm)</div>
      </div>

      <!-- ESPECIFICACIONES DE INGENIERÍA AUDITADAS Y ACTUALIZADAS -->
      <div class="section-title">Especificaciones de Ingeniería y Formatos</div>
      <div class="spec-card">
        <div><strong>Resolución Textura:</strong> 4096&times;4096 (31.2 &mu;m) PBR / 2048&times;2048 (62.5 &mu;m)</div>
        <div><strong>Complejidad Malla:</strong> <span style="color:#38bdf8; font-weight:700;">5,111,056 caras | 2,555,530 vértices</span></div>
        <div><strong>Volumen Sólido:</strong> 416.20 cm&sup3; (100% Watertight Manifold, &chi; = 2)</div>
        <div><strong>Peso Estimado PLA:</strong> 516.1 g (Densidad 1.24 g/cm&sup3;)</div>
        <div><strong>Alojamiento Imán:</strong> &Oslash;10.2 &times; 5.2 mm (Neodimio N52 cilíndrico)</div>
        <div><strong>Formatos Generados:</strong> STL Monolítico Watertight, 3MF Full-Color (4K incrustada), PLY 24-bit TrueColor, OBJ+MTL</div>
      </div>

      <div class="section-title">Paleta Calibrada: 45 Grises + 15 Verdes + Física de Nieve</div>
      <div class="spec-card" style="font-size: 0.72rem; line-height: 1.45;">
        <div>&bull; <strong>45 Tonos de Gris Geológico:</strong> Desde Blanco Ártico (#ffffff) en cumbre hasta Basalto Profundo (#111317) en base.</div>
        <div>&bull; <strong>15 Tonos de Verde Botánico:</strong> Líquenes sulfurados, fescue alpino y musgo de turbera en zócalo bajo.</div>
        <div>&bull; <strong>Física de Nieve Alpina:</strong> <span style="color:#38bdf8; font-weight:700;">0.00% Nieve en pendientes &ge; 44&deg;</span> (roca madre expuesta por gravedad y viento).</div>
        <div>&bull; <strong>Separación Ecológica:</strong> Cero vegetación en cota de nieve alpina (Z &gt; 33 mm).</div>
      </div>
    </div>
  </div>

  <script>
    // Master Full-Color Continuous Topography & 2K Diffuse Map
{b64_data_js}

    const R = 64.0;
    const APOTHEM = R * Math.cos(Math.PI / 6); // 55.4256 mm
    const Z_BASE = 25.0;
    const Z_PEAK = 83.0;
    const RELIEF_H = Z_PEAK - Z_BASE;

    let scene, camera, renderer, controls;
    let mountainMesh, capMesh, stencilGroup, diffuseTexture;
    let dimensionsGroup, trianglesGroup;
    let isWireframe = false;
    let showDimensions = true;
    let showTriangles = false;
    let clipPlane;

    const container = document.getElementById('canvas-container');

    init();

    function init() {{
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x090d16);

      camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.5, 1000);
      camera.up.set(0, 0, 1);
      camera.position.set(130, -150, 120);

      renderer = new THREE.WebGLRenderer({{ antialias: true, stencil: true }});
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      renderer.localClippingEnabled = true;
      container.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.screenSpacePanning = true;
      controls.rotateSpeed = 0.8;
      controls.zoomSpeed = 1.0;
      controls.minPolarAngle = 0.001;
      controls.maxPolarAngle = Math.PI - 0.001;
      controls.target.set(0, 0, 32);
      controls.update();

      clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), Z_PEAK);

      // Iluminación Natural Alpina
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.42);
      scene.add(ambientLight);

      const hemiLight = new THREE.HemisphereLight(0xe0f2fe, 0x1e293b, 0.45);
      scene.add(hemiLight);

      const sunLight = new THREE.DirectionalLight(0xfffbeb, 1.15);
      sunLight.position.set(120, -140, 180);
      sunLight.castShadow = true;
      sunLight.shadow.mapSize.width = 2048;
      sunLight.shadow.mapSize.height = 2048;
      sunLight.shadow.bias = -0.0001;
      scene.add(sunLight);

      const fillLight = new THREE.DirectionalLight(0x93c5fd, 0.45);
      fillLight.position.set(-100, 120, 90);
      scene.add(fillLight);

      // Grid de referencia base
      const grid = new THREE.GridHelper(260, 26, 0x334155, 0x1e293b);
      grid.rotation.x = Math.PI / 2;
      grid.position.z = -0.05;
      scene.add(grid);

      // Crear Cotas 3D y Sectores Triangulares
      buildDimensions();

      // Cargar Textura 2K PBR
      const texLoader = new THREE.TextureLoader();
      texLoader.load(DEM_TEX_B64, (tex) => {{
        diffuseTexture = tex;
        diffuseTexture.anisotropy = renderer.capabilities.getMaxAnisotropy();
        diffuseTexture.minFilter = THREE.LinearMipmapLinearFilter;
        diffuseTexture.magFilter = THREE.LinearFilter;
        diffuseTexture.wrapS = THREE.ClampToEdgeWrapping;
        diffuseTexture.wrapT = THREE.ClampToEdgeWrapping;
        buildSolidMountain();
      }});

      window.addEventListener('resize', onWindowResize, false);
    }}

    function buildDimensions() {{
      dimensionsGroup = new THREE.Group();
      trianglesGroup = new THREE.Group();

      const hexCorners = [];
      for (let k = 0; k < 6; k++) {{
        const a = (30 + 60 * k) * Math.PI / 180;
        hexCorners.push(new THREE.Vector3(R * Math.cos(a), R * Math.sin(a), 0));
      }}

      // 1. Perímetro Hexagonal a Z=0
      const hexLineGeom = new THREE.BufferGeometry().setFromPoints([...hexCorners, hexCorners[0]]);
      const hexLineMat = new THREE.LineBasicMaterial({{ color: 0x38bdf8, linewidth: 2 }});
      const hexLine = new THREE.Line(hexLineGeom, hexLineMat);
      dimensionsGroup.add(hexLine);

      // 2. Línea de cota Vértice a Vértice (Diámetro 128.00 mm)
      const dVGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(hexCorners[0].x, hexCorners[0].y, 0.5),
        new THREE.Vector3(hexCorners[3].x, hexCorners[3].y, 0.5)
      ]);
      const dVMat = new THREE.LineDashedMaterial({{ color: 0x38bdf8, dashSize: 3, gapSize: 2 }});
      const dVLine = new THREE.Line(dVGeom, dVMat);
      dVLine.computeLineDistances();
      dimensionsGroup.add(dVLine);

      // 3. Línea de cota Lado a Lado (Ancho 110.85 mm)
      const sideMid0 = new THREE.Vector3((hexCorners[0].x + hexCorners[1].x)/2, (hexCorners[0].y + hexCorners[1].y)/2, 0.5);
      const sideMid3 = new THREE.Vector3((hexCorners[3].x + hexCorners[4].x)/2, (hexCorners[3].y + hexCorners[4].y)/2, 0.5);
      const dSGeom = new THREE.BufferGeometry().setFromPoints([sideMid0, sideMid3]);
      const dSMat = new THREE.LineDashedMaterial({{ color: 0x34d399, dashSize: 3, gapSize: 2 }});
      const dSLine = new THREE.Line(dSGeom, dSMat);
      dSLine.computeLineDistances();
      dimensionsGroup.add(dSLine);

      // 4. Alojamiento de imán base (Ø10.2mm)
      const magCircleGeom = new THREE.RingGeometry(5.0, 5.2, 32);
      const magMat = new THREE.MeshBasicMaterial({{ color: 0xf43f5e, side: THREE.DoubleSide }});
      const magRing = new THREE.Mesh(magCircleGeom, magMat);
      magRing.position.z = 0.2;
      dimensionsGroup.add(magRing);

      // 5. 6 Triángulos Equiláteros (Radios desde el centro)
      const centerPt = new THREE.Vector3(0, 0, 0.5);
      for (let k = 0; k < 6; k++) {{
        const spokeGeom = new THREE.BufferGeometry().setFromPoints([centerPt, new THREE.Vector3(hexCorners[k].x, hexCorners[k].y, 0.5)]);
        const spokeMat = new THREE.LineBasicMaterial({{ color: 0xf59e0b, linewidth: 2 }});
        trianglesGroup.add(new THREE.Line(spokeGeom, spokeMat));
      }}

      // Rótulos 3D en lona / sprite
      function makeTextSprite(message, color) {{
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 128;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.roundRect(10, 10, 492, 108, 16);
        ctx.fill();
        ctx.stroke();

        ctx.font = 'bold 38px sans-serif';
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(message, 256, 64);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMat = new THREE.SpriteMaterial({{ map: texture }});
        const sprite = new THREE.Sprite(spriteMat);
        sprite.scale.set(24, 6, 1);
        return sprite;
      }}

      // Etiquetas en el espacio 3D
      const labelSide = makeTextSprite("Lado = 64.00 mm", "#38bdf8");
      labelSide.position.set((hexCorners[4].x + hexCorners[5].x)/2 - 14, (hexCorners[4].y + hexCorners[5].y)/2 - 10, 2);
      dimensionsGroup.add(labelSide);

      const labelDistSides = makeTextSprite("Lado a Lado = 110.85 mm", "#34d399");
      labelDistSides.position.set(0, -18, 3);
      dimensionsGroup.add(labelDistSides);

      const labelDistVerts = makeTextSprite("Vértice a Vértice = 128.00 mm", "#38bdf8");
      labelDistVerts.position.set(20, 22, 3);
      dimensionsGroup.add(labelDistVerts);

      const labelMag = makeTextSprite("Imán Ø10.2 mm", "#f43f5e");
      labelMag.position.set(0, 10, 3);
      labelMag.scale.set(18, 4.5, 1);
      dimensionsGroup.add(labelMag);

      const labelTri = makeTextSprite("6 Triángulos Equiláteros L=64mm", "#f59e0b");
      labelTri.position.set(-25, 0, 4);
      trianglesGroup.add(labelTri);

      dimensionsGroup.visible = showDimensions;
      trianglesGroup.visible = showTriangles;
      scene.add(dimensionsGroup);
      scene.add(trianglesGroup);
    }}

    function toggleDimensions() {{
      showDimensions = !showDimensions;
      dimensionsGroup.visible = showDimensions;
      document.getElementById('btn-cotas').classList.toggle('active', showDimensions);
    }}

    function toggleTriangles() {{
      showTriangles = !showTriangles;
      trianglesGroup.visible = showTriangles;
      document.getElementById('btn-tri').classList.toggle('active', showTriangles);
    }}

    function decodeZ() {{
      const binZ = atob(DEM_Z_B64);
      const u16Z = new Uint16Array(binZ.length / 2);
      for (let i = 0; i < u16Z.length; i++) {{
        u16Z[i] = binZ.charCodeAt(i * 2) | (binZ.charCodeAt(i * 2 + 1) << 8);
      }}
      return u16Z;
    }}

    function buildSolidMountain() {{
      const u16Z = decodeZ();
      const nx = DEM_NX;
      const ny = DEM_NY;
      const stepX = 2 * APOTHEM / (nx - 1);
      const stepY = 2 * R / (ny - 1);

      const geometry = new THREE.BufferGeometry();
      const positions = [];
      const uvs = [];
      const indices = [];

      const vMap = new Int32Array(nx * ny).fill(-1);
      let vCount = 0;

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

            positions.push(x, y, z);
            const u = Math.min(1.0, Math.max(0.0, (x + APOTHEM) / (2 * APOTHEM)));
            const v = Math.min(1.0, Math.max(0.0, (y + R) / (2 * R)));
            uvs.push(u, v);

            vMap[idx] = vCount++;
          }}
        }}
      }}

      // Top Triangles
      for (let j = 0; j < ny - 1; j++) {{
        for (let i = 0; i < nx - 1; i++) {{
          const idx0 = j * nx + i;
          const idx1 = j * nx + (i + 1);
          const idx2 = (j + 1) * nx + (i + 1);
          const idx3 = (j + 1) * nx + i;

          const v0 = vMap[idx0];
          const v1 = vMap[idx1];
          const v2 = vMap[idx2];
          const v3 = vMap[idx3];

          if (v0 !== -1 && v1 !== -1 && v2 !== -1) {{
            indices.push(v0, v1, v2);
          }}
          if (v0 !== -1 && v2 !== -1 && v3 !== -1) {{
            indices.push(v0, v2, v3);
          }}
        }}
      }}

      // Hexagon Vertical Skirt Wall
      const wallPts2D = [];
      const nPerEdge = 24;
      for (let k = 0; k < 6; k++) {{
        const a1 = (30 + 60 * k) * Math.PI / 180;
        const a2 = (30 + 60 * (k + 1)) * Math.PI / 180;
        const p1 = {{ x: R * Math.cos(a1), y: R * Math.sin(a1) }};
        const p2 = {{ x: R * Math.cos(a2), y: R * Math.sin(a2) }};
        for (let s = 0; s < nPerEdge; s++) {{
          const t = s / nPerEdge;
          wallPts2D.push({{ x: (1 - t) * p1.x + t * p2.x, y: (1 - t) * p1.y + t * p2.y }});
        }}
      }}

      const nWall = wallPts2D.length;
      const idxWallTopStart = positions.length / 3;

      for (let i = 0; i < nWall; i++) {{
        positions.push(wallPts2D[i].x, wallPts2D[i].y, Z_BASE);
        uvs.push(0.05, 0.05);
      }}

      const idxWallBotStart = positions.length / 3;
      for (let i = 0; i < nWall; i++) {{
        positions.push(wallPts2D[i].x, wallPts2D[i].y, 0.0);
        uvs.push(0.01, 0.01);
      }}

      for (let i = 0; i < nWall; i++) {{
        const next_i = (i + 1) % nWall;
        indices.push(idxWallTopStart + i, idxWallBotStart + i, idxWallBotStart + next_i);
        indices.push(idxWallTopStart + i, idxWallBotStart + next_i, idxWallTopStart + next_i);
      }}

      // Bottom face
      const idxBotCenter = positions.length / 3;
      positions.push(0, 0, 0);
      uvs.push(0.01, 0.01);

      for (let i = 0; i < nWall; i++) {{
        const next_i = (i + 1) % nWall;
        indices.push(idxBotCenter, idxWallBotStart + next_i, idxWallBotStart + i);
      }}

      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
      geometry.setIndex(indices);
      geometry.computeVertexNormals();

      // Standard PBR Material with 2K Diffuse Texture
      const matMountain = new THREE.MeshStandardMaterial({{
        map: diffuseTexture,
        roughness: 0.85,
        metalness: 0.08,
        clippingPlanes: [clipPlane],
        clipShadows: true,
        shadowSide: THREE.DoubleSide
      }});

      mountainMesh = new THREE.Mesh(geometry, matMountain);
      mountainMesh.castShadow = true;
      mountainMesh.receiveShadow = true;
      mountainMesh.renderOrder = 6;
      scene.add(mountainMesh);

      // Stencil Buffer Solid Cross-Section Capping
      stencilGroup = new THREE.Group();
      const mat0 = new THREE.MeshBasicMaterial({{
        depthWrite: false, depthTest: false, colorWrite: false,
        stencilWrite: true, stencilFunc: THREE.AlwaysStencilFunc,
        side: THREE.BackSide, clippingPlanes: [clipPlane],
        stencilFail: THREE.KeepStencilOp, stencilZFail: THREE.KeepStencilOp,
        stencilZPass: THREE.IncrementWrapStencilOp
      }});
      const mesh0 = new THREE.Mesh(geometry, mat0);
      mesh0.renderOrder = 1;
      stencilGroup.add(mesh0);

      const mat1 = new THREE.MeshBasicMaterial({{
        depthWrite: false, depthTest: false, colorWrite: false,
        stencilWrite: true, stencilFunc: THREE.AlwaysStencilFunc,
        side: THREE.FrontSide, clippingPlanes: [clipPlane],
        stencilFail: THREE.KeepStencilOp, stencilZFail: THREE.KeepStencilOp,
        stencilZPass: THREE.DecrementWrapStencilOp
      }});
      const mesh1 = new THREE.Mesh(geometry, mat1);
      mesh1.renderOrder = 2;
      stencilGroup.add(mesh1);
      scene.add(stencilGroup);

      // Solid Capping Plane
      const planeGeom = new THREE.PlaneGeometry(300, 300);
      const capMat = new THREE.MeshStandardMaterial({{
        color: 0x1e293b,
        roughness: 0.85,
        metalness: 0.1,
        stencilWrite: true,
        stencilRef: 0,
        stencilFunc: THREE.NotEqualStencilFunc,
        stencilFail: THREE.ReplaceStencilOp,
        stencilZFail: THREE.ReplaceStencilOp,
        stencilZPass: THREE.ReplaceStencilOp,
        side: THREE.DoubleSide
      }});

      capMesh = new THREE.Mesh(planeGeom, capMat);
      capMesh.position.z = Z_PEAK;
      capMesh.renderOrder = 3;
      capMesh.onAfterRender = function(r) {{ r.clearStencil(); }};
      capMesh.visible = false;
      scene.add(capMesh);

      const overlay = document.getElementById('loading-overlay');
      overlay.style.opacity = '0';
      setTimeout(() => overlay.style.display = 'none', 400);

      animate();
    }}

    function onZCutChange(val) {{
      const z = parseFloat(val);
      document.getElementById('cut-height-val').textContent = z.toFixed(1) + ' mm';
      clipPlane.constant = z;

      if (z < Z_PEAK - 0.2) {{
        capMesh.position.z = z;
        capMesh.visible = true;
      }} else {{
        capMesh.visible = false;
      }}
    }}

    function toggleWireframe() {{
      isWireframe = !isWireframe;
      mountainMesh.material.wireframe = isWireframe;
      document.getElementById('btn-wire').classList.toggle('active', isWireframe);
    }}

    function setShadingMode(mode) {{
      if (mode === 'textured') {{
        mountainMesh.material.map = diffuseTexture;
        mountainMesh.material.needsUpdate = true;
        document.getElementById('btn-tex').classList.add('active');
      }}
    }}

    function setCamera(pos) {{
      camera.up.set(0, 0, 1);
      if (pos === 'iso') {{
        camera.position.set(130, -150, 120);
        controls.target.set(0, 0, 32);
      }} else if (pos === 'top') {{
        camera.position.set(0.001, -0.001, 230);
        controls.target.set(0, 0, 25);
      }} else if (pos === 'base') {{
        camera.position.set(0.001, -140, -120);
        controls.target.set(0, 0, 10);
      }} else if (pos === 'zoom_summit') {{
        camera.position.set(-15, -15, 110);
        controls.target.set(-3.5, 4.5, 83);
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

# Write to project directory
out_html = os.path.join(proj_dir, "catan_mountain_viewer.html")
with open(out_html, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"Saved Master HTML: {out_html} ({os.path.getsize(out_html)/1024/1024:.2f} MB)")

# Copy to brain artifact directory
brain_html = os.path.join(brain_dir, "catan_mountain_viewer.html")
shutil.copy(out_html, brain_html)
print(f"Copied to brain: {brain_html}")
