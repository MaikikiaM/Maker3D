---
name: print-audit-and-webgl
description: >-
  Auditoría estanca y de imprimibilidad FDM (trimesh, PyMeshLab, Manifold3D,
  lib3mf 3MF + laminado CLI real en Bambu Studio) y optimización/empaquetado de
  visores WebGL HTML autónomos con gltf-transform Meshopt + WebP (4K sin recorte),
  verificados en navegador headless sin errores.
---

# Auditoría Estanca (`lib3mf` + Topología) y Visor WebGL (`Meshopt + WebP`)

Ningún diseño 3D en Maker3D 2.0 se da por terminado sin superar la **doble puerta de calidad**: malla estanca verificada geométricamente (`trimesh`/`PyMeshLab`/`Manifold3D`) y visor WebGL ligero verificado sin errores. **Por defecto NO se lamina con Bambu Studio** (el usuario puede usar otra impresora o método de fabricación/pintado; usar `--bambu-slice` solo cuando lo pida expresamente).

---

## 1. Puerta 1: Auditoría de Malla, Reparación y `3MF` (`--bambu-slice` opcional bajo petición)

Ejecuta `tools/audit/mesh_and_print_audit.py`:

```powershell
python tools/audit/mesh_and_print_audit.py `
  --input <ruta_modelo.stl> `
  --out-dir <dir_salida>/audit `
  --nozzle-mm 0.2 `
  --overhang-deg 45 `
  --catan-check `
  --export-3mf <dir_salida>/audit/modelo.3mf
```

*(Estándar único del proyecto: **toda** auditoría usa `--nozzle-mm 0.2` pensando en **PLA Full Color sobre InfiMech MX Pro** de `8` boquillas de `0.20 mm`. Añadir `--bambu-slice --printer a1_n02` únicamente si el usuario solicita explícitamente estimar laminado.)*

### Qué verifica exactamente:
1. **Topología matemática (`trimesh` / `PyMeshLab` / `manifold3d`):**
   - `is_watertight == True`, `chi == 2` (una única envolvente cerrada), `degenerate_faces == 0`, `non_manifold_edges == 0`.
   - Volumen en cm³, bounding box exacto y área superficial.
2. **Reparación conservadora (`--repair`):** si una malla ya estanca (`χ=2`) presenta pocas caras nanométricas `< 1e-10 mm²` por cuantización `float32` de STL, aplica primero relajación local 1-ring (`0.80*V + 0.20*mean(nbrs)`, hasta 6 pasadas) antes que borrado destructivo de caras para alcanzar `degenerate_faces == 0` sin alterar `χ=2`.
3. **Imprimibilidad FDM:**
   - Porcentaje de caras con voladizo severo (`> overhang_deg`, excluyendo la cara de apoyo en `Z_min` y el techo del imán si `--catan-check`).
   - Estimación de secciones delgadas inferiores a `2 * nozzle_mm` (`0.40 mm` bajo el estándar exclusivo de boquilla `0.20 mm` y `PLA` Full Color).
4. **Verificación física Catán (`--catan-check`):** confirma ancho `110.8513 ± 0.15 mm`, largo `128.0000 ± 0.15 mm`, presencia de cavidad centrada en base inferior y planitud de los 6 pads `Ø20 mm`.
5. **Contenedor `3MF` (`lib3mf`) y Laminado Real en `bambu-studio.exe` (`--bambu-slice`):**
   - Empaqueta `.3mf` válido en milímetros con `lib3mf` y verifica su lectura.
   - Invoca `C:\Program Files\Bambu Studio\bambu-studio.exe` en modo CLI headless con los perfiles oficiales BBL descubiertos por `tools/tool_paths.py` (`--slice 0 --debug 3 --load-settings ... --load-filaments ... --export-3mf ...`).
   - Valida que el `.gcode.3mf` generado contiene `Metadata/plate_1.gcode` real y extrae estimación de filamento/tiempo cuando está disponible.

---

## 2. Puerta 2: Compresión `Meshopt + WebP` y Visor HTML Autónomo

> **Regla Windows:** en PowerShell ejecutar siempre `gltf-transform.cmd` / `npm.cmd`, nunca los scripts `.ps1` bloqueados por ExecutionPolicy.

Ejecuta `tools/webgl/optimize_and_build_viewer.py`:

```powershell
python tools/webgl/optimize_and_build_viewer.py `
  --input <ruta_modelo.glb> `
  --out-html <ruta_visor.html> `
  --title "Título del Modelo" `
  --compress meshopt `
  --texture-format webp `
  --max-texture-size 4096 `
  --no-simplify
```

### Características del Visor v2 Generado:
- **Compresión real `gltf-transform.cmd optimize`:** reduce el tamaño GLB manteniendo fidelidad `4K` y geometría íntegra (`--no-simplify`) mediante Meshopt (`meshoptimizer`) y texturas `WebP`. Sin tope artificial de `< 20 MB` ni bloqueo de `< 50 MB` (si un visor supera `50 MB`, excluirlo de Git en `.gitignore` y conservarlo en local/Drive).
- **Visor single-file autónomo:** Three.js moderno + `GLTFLoader` + `MeshoptDecoder` + `DRACOLoader` con modos **`PBR`**, **`Arcilla CAD`** (sin texturas para auditar relieve 3D real), **`Wireframe`**, **`Corte Z`** en tiempo real, **3 Atmósferas** de iluminación y presets de cámara (incluyendo vista inferior para revisar alojamientos de imán).
- **Iluminación inicial (`Sol Alpino`) y sombras reales (`PCFSoftShadowMap`):** arrancar por defecto en atmósfera **`Sol Alpino`** (`environmentIntensity = 0.40`, `HemisphereLight = 0.74`, `keyLight = 2.60` con `castShadow = true` `2048×2048`, `fillLight = 0.50`, `rimLight = 0.55`) para igualar el brillo y legibilidad diurna de `Montaña V4` sin lavar grietas ni el zócalo oscuro.
- **Verificación en navegador:** abre el HTML mediante servidor HTTP temporal e inspecciona en el subagente `browser` (`Model="flash"`) comprobando render activo y **`0` errores WebGL/consola**.
- **Visores `8K` / RAW `> 50 MB` por bloques:** `optimize_and_build_viewer.py` trocea automáticamente el GLB en múltiples `<script class="glb-chunk">` (`48 MB` raw / `64 MB` Base64) reensamblados vía `loadChunkedGlbBuffer()`. Tanto los `*-raw.html` como cualquier visor `8K` comprimido que supere `50 MB` deben listarse explícitamente en `.gitignore` y conservarse en local/Drive.
