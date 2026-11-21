---
name: maker3d-router
description: >-
  Orquestador principal de Maker3D 2.0. Usar SIEMPRE como primer punto de entrada
  para cualquier petición nueva de diseño 3D, modelado CAD paramétrico, losetas de
  Catán, miniaturas orgánicas, props o piezas imprimibles FDM. Diagnostica el tipo
  de geometría y enruta hacia la combinación exacta de skills especializadas,
  servidores MCP (build123d, openscad, blender) y scripts de tools/.
---

# Maker3D 2.0 — Enrutador Maestro de Diseño 3D

Nunca resuelvas un problema 3D con un único martillo. Clasifica la geometría solicitada y activa solo los módulos necesarios en cadena.

## 1. Diagnóstico Inicial (Antes de Modelar)

1. Verifica el estado del entorno si hay dudas sobre binarios o MCPs:
   ```powershell
   python tools/env_check.py
   ```
2. Consulta `.agents/rules/maker3d-architecture.md` para confirmar las tolerancias FDM / Catán vigentes.
3. Clasifica el encargo en una de estas **5 clases geométricas**:

| Clase de Diseño | Ejemplos Reales | Pipeline / Skills a Cargar | Herramientas Principales |
|---|---|---|---|
| **A. Ingeniería Paramétrica B-Rep** | Soportes, cajas de electrónica, tapas a presión, engranajes, soportes Gridfinity, piezas funcionales con filetes/roscas exactas | `parametric-brep-cad` → `print-audit-and-webgl` | `build123d` MCP o `openscad` MCP (`BOSL2` + `--backend=Manifold`), `mesh_and_print_audit.py` |
| **B. Escultura y Relieve Geológico (Catán / Diorama)** | Nuevas losetas hexagonales con ríos, cañones, terrazas, acantilados o terrenos hidrológicos | `parametric-brep-cad` + `terrain-geology-engine` + `blender-sculpt-bake` → `print-audit-and-webgl` | `build_catan_base_brep.py` → `hydraulic_terrain_gen.py` → `sculpt_fuse_bake_render.py` → `mesh_and_print_audit.py` → `optimize_and_build_viewer.py` |
| **C. Orgánico / Figurativo Heroico** | Miniaturas, personajes, monstruos, criaturas, cabezas, estatuas, props narrativos sin cotas mecánicas | `generative-3d-mesh` → `blender-sculpt-bake` → `print-audit-and-webgl` | `image-generator` (turnaround) → `cloud_3d_bridge.py` (`hf-zero` / `fal`) → Blender Voxel Remesh + zócalo → Bambu CLI |
| **D. Ensamblaje Híbrido (CAD + Orgánico 3D)** | Loseta Catán con árboles/casas/puentes 3D reales, trofeo con base hexagonal roscada, relicario biomecánico | `parametric-brep-cad` + `terrain-geology-engine` + `generative-3d-mesh` + `blender-sculpt-bake` → `print-audit-and-webgl` | Zócalo exacto (`build123d`) + props 3D (`cloud_3d_bridge.py` / procedurales) + fusión estanca en `sculpt_fuse_bake_render.py` |
| **E. Iteración sobre Loseta Catán Existente (Legacy)** | Ajustes incrementales sobre las 8 losetas ya creadas en `Tableros-Catan/` | `.agents/skills/nanobanana-3d-sculpt/SKILL.md` | Pipeline NumPy/SciPy existente (`build_catan_biome.py`, `nanobanana_to_catan_hex.py`) |

---

## 2. Protocolo Estándar de Ejecución en 4 Fases

1. **Bloque de Tolerancias:** Define explícitamente las cotas críticas antes de escribir código (holguras `0.20–0.30 mm`, diámetro de imanes `10.20×5.20 mm`, grosor mínimo `>= 0.80 mm`, dimensiones de cama Bambu A1/P1S/X1C).
2. **Geometría Exacta + Detalle Orgánico Desacoplado:**
   - Jamás deformes caras funcionales (pads, carreteras, huecos de imán, roscas, chasis) con desplazamiento procedural o voxel remesh destructivo.
   - Protege la base B-Rep y une el detalle esculpido de forma estanca (`Manifold` / `Boolean` / corte limpio).
3. **Auto-Auditoría Visual y Topológica Obligatoria:**
   - Ejecuta `tools/audit/mesh_and_print_audit.py` con `--bambu-slice` para verificar `watertight=True`, `chi=2`, voladizos, `.3mf` válido vía `lib3mf` y laminado real en Bambu Studio.
   - Si pasas por Blender, inspecciona con `read_file` los renders `audit_6views_pbr.png`, `audit_6views_clay.png` (arcilla sin textura para comprobar 100% paridad 2D-a-3D) y `audit_ortho.png`.
4. **Entrega Triple al Usuario:**
   - Archivo paramétrico fuente (`.step` / `.scad` / `.py`), cuando aplique.
   - Archivo listo para imprimir (`.stl` / `.3mf` validado y verificado en Bambu Studio CLI).
   - Visor WebGL autónomo (`.html` optimizado con `Meshopt + WebP` vía `optimize_and_build_viewer.py`, `< 20 MB`, verificado sin errores en navegador headless).
