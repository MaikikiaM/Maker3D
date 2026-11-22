---
description: Arquitectura modular Maker3D 2.0, matriz de enrutamiento de motores 3D, tolerancias físicas FDM/Catan y auditoría visual obligatoria.
---

# Reglas de Arquitectura — Maker3D 2.0

## 1. Matriz de Enrutamiento (Elegir la Herramienta Adecuada)

Antes de escribir código o generar geometría, consultar `.agents/skills/maker3d-router/SKILL.md` y seleccionar el motor óptimo:

| Tipo de Solicitud | Motor / Pipeline Principal | Skills Activas | Scripts `tools/` / MCP |
|---|---|---|---|
| **Pieza mecánica, soporte, caja, rosca, adaptador, repuesto** | CAD B-Rep (`build123d` → STEP+STL) u `OpenSCAD Nightly` (`--backend=Manifold` + `BOSL2`) | `maker3d-router` → `parametric-brep-cad` → `print-audit-and-webgl` | `build123d` MCP, `openscad` MCP, `tools/audit/mesh_and_print_audit.py` |
| **Loseta Catán / Terreno geológico con ríos/escarpes** | B-Rep base (`build123d`) + Hidrología física (`Landlab`) + Esculpido normales/Voxel/PBR (`Blender 5.3`) | `maker3d-router` → `parametric-brep-cad` + `terrain-geology-engine` + `blender-sculpt-bake` → `print-audit-and-webgl` | `tools/catan_kit/build_catan_base_brep.py`, `tools/terrain/hydraulic_terrain_gen.py`, `tools/blender/sculpt_fuse_bake_render.py` |
| **Miniatura orgánica, figura, prop heroico, busto** | Concepto `image-generator` → Nube 3D (`TRELLIS.2` / `Hunyuan3D` / `Hi3DGen`) → Remesh/Base CAD (`Blender 5.3`) | `maker3d-router` → `generative-3d-mesh` → `blender-sculpt-bake` → `print-audit-and-webgl` | `tools/generative/cloud_3d_bridge.py`, `tools/blender/sculpt_fuse_bake_render.py` |
| **Escena híbrida (CAD exacto + árboles/edificios/ruinas 3D)** | Zócalo B-Rep exacto + Props 3D sólidos + Fusión booleana estanca (`Blender 5.3` / `manifold3d`) | `maker3d-router` → `parametric-brep-cad` + `generative-3d-mesh` + `blender-sculpt-bake` → `print-audit-and-webgl` | `tools/blender/sculpt_fuse_bake_render.py`, `tools/audit/mesh_and_print_audit.py` |

> **Regla de preservación:** Las 8 losetas existentes de `Tableros-Catan/` y `.agents/skills/nanobanana-3d-sculpt/` permanecen intactas como referencia histórica y pipeline específico legacy.

> **Guardrails de calidad visual 3D (lección V4/V5/TRELLIS.2/CSG):**
> - **Terrenos orgánicos heroicos 360° (Catán rocoso / alpino):** NO generar montañas apilando prismas/conos procedurales en Python/CSG (efecto low-poly artificial), ni usar `TRELLIS.2` desde 1 foto cuando requiera aristas y fisuras cortantes (suaviza como plastilina). Partir siempre de una malla de relieve densa validada cubriendo todo el hexágono interior + proyección multi-cámara **Cycles UV 3D (`Smart UV 3D`)** desde vistas calibradas de Nano Banana (nunca mapa cenital `2.5D` único sobre escarpes `>= 55°`).

---

## 2. Invariantes Físicos y Tolerancias de Ingeniería

### Invariantes de Impresión FDM (Boquilla estándar 0.40 mm)
- **Estanqueidad obligatoria:** `is_watertight == True`, característica de Euler `chi == 2` (1 componente sólido, sin bordes abiertos ni autointersecciones), `degenerate_faces == 0`.
- **Pared mínima estructural:** `>= 2 * nozzle_mm` (`>= 0.80 mm`, recomendado `>= 1.20 mm` en clips y paredes portantes).
- **Holgura de encajes:**
  - Ajuste deslizante / ensamblaje removible: `+0.20 mm` radial / `0.25 mm` por cara (`+0.30 mm` en puentes impresos).
  - Ajuste a presión (imán / pin): `+0.10 mm` radial (p. ej. imán Ø10×5 mm → alojamiento Ø10.20 × 5.20 mm).
- **Voladizos sin soporte:** Ángulo `<= 45°` respecto a la vertical siempre que sea posible; las caras funcionales que excedan `45°` deben incluir chaflán de transición (`45°`) o documentarse para soportes en el slicer del usuario.
- **Compensación de pata de elefante:** Chaflán inferior de `0.6–0.8 mm` a `45°` en piezas que requieran ajuste en zócalo/placa.

### Invariantes Canon de Losetas Catán
- Hexágono regular **pointy-top**: `R = 64.00 mm`, apotema `W/2 = 55.4256 mm`, ancho plano a plano `W = 110.8513 mm`, largo vértice a vértice `D = 128.0000 mm`.
- **Zócalo sólido:** `Z_BASE = 25.00 mm`.
- **Carril perimetral plano de caminos:** `ROAD_MARGIN = 3.50 mm` desde los 6 lados a `Z = 25.00 mm`.
- **6 Pads circulares de poblado:** `Ø20.00 mm` (`PAD_RADIUS = 10.00 mm`) centrados en los 6 vértices, estrictamente planos a `Z = 25.00 mm`.
- **Hueco inferior para imán:** centrado en `(0, 0, 0)`, `Ø10.20 × 5.20 mm` (soporte de cilindro con micro-chaflán o hexágono anti-rebose).
- **Hidrología y nieve física:** canales de río estrictamente monótonos descendentes (`slope >= 0` hacia la salida) y **`0.0000%` de nieve en pendientes `>= 44.0°`**.
- **Techo local STL Master:** `<= 10,000,000` triángulos.

---

## 3. Reglas de Exportación, WebGL y Auto-Auditoría Visual

1. **Tres entregables estándar por diseño terminado:**
   - Paramétrico editable (`.step` o `.scad`), si aplica geometría CAD.
   - Malla de fabricación validada (`.stl` estanco y/o `.3mf` válido vía `lib3mf`; sin laminar con Bambu Studio por defecto salvo petición explícita del usuario).
   - Visor WebGL autónomo (`.html` con GLB comprimido mediante `Meshopt + WebP` vía `gltf-transform.cmd`, objetivo `< 20 MB`, límite estricto `< 50 MB` para `Visores-HTML/`).
2. **Auto-auditoría visual obligatoria (nunca entregar a ciegas):**
   - Renderizar e inspeccionar visualmente mediante `read_file` los paneles `audit_6views_pbr.png`, `audit_6views_clay.png` (arcilla sin textura para auditar relieve real vs pintura 2D) y `audit_ortho.png` antes de declarar completa cualquier pieza o loseta.
   - **Criterio visual estricto sin indulgencia técnica:** jamás valorar ni dar por bueno un activo por métricas técnicas (`chi=2`, UVs sin estirar, HTML ligero) si las capturas PBR/Arcilla muestran aspecto low-poly, manchas planas o pérdida de detalle frente a la referencia hero `0°` (comparar siempre cara a cara todas las vistas `0°–300°` contra la referencia objetivo).
   - **Sincronía de archivos multimedia:** nunca ejecutar `read_file` sobre una imagen mientras siga en ejecución el comando o subagente que la escribe (esperar siempre a que termine el proceso antes de leer).
   - Verificar en navegador headless (subagente `browser`) todo visor HTML generado (`0` errores de consola WebGL/JS).

---

## 4. Convenciones Windows y Rutas de Entorno

- Resolver todas las rutas de herramientas mediante `tools/tool_paths.py` (autodescubrimiento de Blender 5.3, OpenSCAD Nightly + `BOSL2`, Bambu Studio + perfiles BBL, `uv`/`uvx`, `gltf-transform.cmd`).
- En PowerShell ejecutar **`npm.cmd`**, **`npx.cmd`** y **`gltf-transform.cmd`** (nunca `.ps1` bloqueados por ExecutionPolicy) y usar `--backend=Manifold` al invocar OpenSCAD CLI.
- No registrar secretos ni binarios pesados (`*.stl`, `*.3mf`, `*.glb`, `*.step`, `*.blend`, `node_modules/`) en Git.
